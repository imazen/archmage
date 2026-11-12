//! Static codegen equivalence-class & `llvm-mca` analyzer (`cargo xtask mca`).
//!
//! Compiles standalone `#[unsafe(no_mangle)] #[inline(never)]` `stream64` and
//! `chain64` probes across x86-64, AArch64, and WASM tiers, resolves LLVM ICF
//! symbol aliases (`.set` / `=`), groups policy flags into exact normalized
//! hot-body assembly equivalence classes, verifies zero unexpected hot-path
//! `call`/`bl` instructions in SIMD tiers, and runs `llvm-mca` once per unique
//! equivalence class.

use crate::registry::Registry;
use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Clone)]
struct TierSpec {
    target: &'static str,
    rustflags: &'static str,
    token_ty: &'static str,
    registry_token: Option<&'static str>,
    mca_triple: Option<&'static str>,
    mca_cpus: &'static [&'static str],
}

fn tier_spec(tier: &str) -> Result<TierSpec> {
    Ok(match tier {
        "v3" => TierSpec {
            target: "x86_64-unknown-linux-gnu",
            rustflags: "",
            token_ty: "archmage::X64V3Token",
            registry_token: Some("X64V3Token"),
            mca_triple: Some("x86_64-unknown-linux-gnu"),
            mca_cpus: &["znver5", "arrowlake"],
        },
        "v3-vnni" | "v3_vnni" => TierSpec {
            target: "x86_64-unknown-linux-gnu",
            rustflags: "",
            token_ty: "archmage::X64V3VnniToken",
            registry_token: Some("X64V3VnniToken"),
            mca_triple: Some("x86_64-unknown-linux-gnu"),
            mca_cpus: &["znver5", "arrowlake"],
        },
        "v4" => TierSpec {
            target: "x86_64-unknown-linux-gnu",
            rustflags: "",
            token_ty: "archmage::X64V4Token",
            registry_token: Some("X64V4Token"),
            mca_triple: Some("x86_64-unknown-linux-gnu"),
            mca_cpus: &["znver5", "skylake-avx512"],
        },
        "v4x" => TierSpec {
            target: "x86_64-unknown-linux-gnu",
            rustflags: "",
            token_ty: "archmage::X64V4xToken",
            registry_token: Some("X64V4xToken"),
            mca_triple: Some("x86_64-unknown-linux-gnu"),
            mca_cpus: &["znver5", "icelake-server"],
        },
        "scalar" => TierSpec {
            target: "x86_64-unknown-linux-gnu",
            rustflags: "",
            token_ty: "archmage::ScalarToken",
            registry_token: None,
            mca_triple: Some("x86_64-unknown-linux-gnu"),
            mca_cpus: &["znver5"],
        },
        "neon" => TierSpec {
            target: "aarch64-unknown-linux-gnu",
            rustflags: "",
            token_ty: "archmage::NeonToken",
            registry_token: Some("NeonToken"),
            mca_triple: Some("aarch64-unknown-linux-gnu"),
            mca_cpus: &["apple-m4", "neoverse-n1"],
        },
        "wasm128" => TierSpec {
            target: "wasm32-wasip1",
            rustflags: "-Ctarget-feature=+simd128",
            token_ty: "archmage::Wasm128Token",
            registry_token: None,
            mca_triple: None,
            mca_cpus: &[],
        },
        "wasm128-relaxed" | "wasm128_relaxed" => TierSpec {
            target: "wasm32-wasip1",
            rustflags: "-Ctarget-feature=+simd128,+relaxed-simd",
            token_ty: "archmage::Wasm128Token",
            registry_token: None,
            mca_triple: None,
            mca_cpus: &[],
        },
        other => bail!("unknown --tiers entry '{other}'"),
    })
}

#[derive(Clone, Copy)]
struct WidthSpec {
    name: &'static str,
    elem: &'static str,
    lanes: usize,
    has_recip: bool,
}

const WIDTH_SPECS: &[WidthSpec] = &[
    WidthSpec {
        name: "f32x4",
        elem: "f32",
        lanes: 4,
        has_recip: true,
    },
    WidthSpec {
        name: "f32x8",
        elem: "f32",
        lanes: 8,
        has_recip: true,
    },
    WidthSpec {
        name: "f32x16",
        elem: "f32",
        lanes: 16,
        has_recip: true,
    },
    WidthSpec {
        name: "f64x2",
        elem: "f64",
        lanes: 2,
        has_recip: false,
    },
    WidthSpec {
        name: "f64x4",
        elem: "f64",
        lanes: 4,
        has_recip: false,
    },
    WidthSpec {
        name: "f64x8",
        elem: "f64",
        lanes: 8,
        has_recip: false,
    },
];

fn find_width(name: &str) -> Result<WidthSpec> {
    WIDTH_SPECS
        .iter()
        .copied()
        .find(|w| w.name == name)
        .with_context(|| format!("unknown width '{name}'"))
}

struct McaConfig {
    tiers: Vec<String>,
    ops: Vec<String>,
    widths: Vec<WidthSpec>,
    remote: Option<String>,
    remote_repo: Option<String>,
    out_dir: PathBuf,
}

fn home_tmp() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME is not set")?;
    let p = PathBuf::from(home).join("tmp");
    fs::create_dir_all(&p)?;
    Ok(p)
}

fn parse_args(args: &[String]) -> Result<McaConfig> {
    let mut tiers_raw = String::from("v3,neon,wasm128,wasm128-relaxed");
    let mut ops_raw = String::from("min,max,recip,rsqrt");
    let mut widths_raw = String::from("f32x8,f64x4");
    let mut remote: Option<String> = None;
    let mut remote_repo: Option<String> = None;
    let mut out_dir: Option<PathBuf> = None;

    let mut it = args.iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--tiers" => tiers_raw = it.next().context("--tiers needs a value")?.clone(),
            "--ops" => ops_raw = it.next().context("--ops needs a value")?.clone(),
            "--widths" => widths_raw = it.next().context("--widths needs a value")?.clone(),
            "--remote" => remote = Some(it.next().context("--remote needs a host")?.clone()),
            "--remote-repo" => {
                remote_repo = Some(it.next().context("--remote-repo needs a path")?.clone())
            }
            "--out" => out_dir = Some(PathBuf::from(it.next().context("--out needs a path")?)),
            "--help" | "-h" => {
                println!(
                    "Usage: cargo xtask mca [--tiers v3,neon,wasm128,wasm128-relaxed] \
                     [--ops min,max,recip,rsqrt] [--widths f32x8,f64x4|all] \
                     [--remote <host>] [--remote-repo <path>] [--out <dir>]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown mca argument: {other}"),
        }
    }

    let tiers: Vec<String> = tiers_raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    let ops: Vec<String> = ops_raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    let widths: Vec<WidthSpec> = if widths_raw.trim() == "all" {
        WIDTH_SPECS.to_vec()
    } else {
        widths_raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(find_width)
            .collect::<Result<_>>()?
    };
    let out_dir = match out_dir {
        Some(p) => p,
        None => home_tmp()?.join("archmage-mca-out"),
    };
    Ok(McaConfig {
        tiers,
        ops,
        widths,
        remote,
        remote_repo,
        out_dir,
    })
}

fn load_policy_flags(root: &Path) -> Result<BTreeMap<String, Vec<(u32, String)>>> {
    let path = root.join("benchmarks/primitive_variants_2026-11-10/src/policy_registry.rs");
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let macro_re = Regex::new(r"^macro_rules!\s+(\w+)\s*\{").unwrap();
    let apply_re = Regex::new(r#"\$apply!\((\d+),\s*"([^"]+)"\);"#).unwrap();

    let mut blocks: BTreeMap<String, Vec<(u32, String)>> = BTreeMap::new();
    let mut current: Option<String> = None;
    for line in text.lines() {
        if let Some(cap) = macro_re.captures(line) {
            let name = cap[1].to_string();
            blocks.entry(name.clone()).or_default();
            current = Some(name);
            continue;
        }
        if current.is_some() && line.starts_with('}') {
            current = None;
            continue;
        }
        if let (Some(cur), Some(cap)) = (&current, apply_re.captures(line)) {
            let bits: u32 = cap[1].parse()?;
            blocks
                .get_mut(cur)
                .unwrap()
                .push((bits, cap[2].to_string()));
        }
    }
    Ok(blocks)
}

#[derive(Clone)]
struct ProbeManifestEntry {
    symbol: String,
    tier: String,
    op: String,
    width: String,
    shape: &'static str,
    arm: String,
}

fn generate_probe_crate(
    probe_dir: &Path,
    repo_path: &str,
    reg: &Registry,
    tiers: &[String],
    ops: &[String],
    widths: &[WidthSpec],
    flags_map: &BTreeMap<String, Vec<(u32, String)>>,
) -> Result<Vec<ProbeManifestEntry>> {
    fs::create_dir_all(probe_dir.join("src"))?;
    let cargo_toml = format!(
        "[package]\n\
         name = \"archmage-mca-probe\"\n\
         version = \"0.0.0\"\n\
         edition = \"2024\"\n\
         publish = false\n\n\
         [workspace]\n\n\
         [lib]\n\
         crate-type = [\"rlib\"]\n\n\
         [dependencies]\n\
         archmage = {{ path = \"{repo_path}\", features = [\"std\", \"avx512\"] }}\n\
         magetypes = {{ path = \"{repo_path}/magetypes\", features = [\"std\", \"w512\", \"avx512\"] }}\n\
         magemath = {{ path = \"{repo_path}/magemath\", features = [\"std\", \"w512\", \"avx512\"] }}\n"
    );
    fs::write(probe_dir.join("Cargo.toml"), cargo_toml)?;

    let mut src = String::from(
        "#![allow(unused_imports, dead_code, non_snake_case)]\nuse archmage::SimdToken;\n",
    );
    let mut manifest = Vec::new();

    for tier in tiers {
        let spec = tier_spec(tier)?;
        let tf_attr = if let Some(tok_name) = spec.registry_token {
            let tok = reg
                .find_token(tok_name)
                .with_context(|| format!("missing token {tok_name} in registry"))?;
            format!("#[target_feature(enable = \"{}\")]", tok.features.join(","))
        } else {
            String::new()
        };
        let tier_ident = tier.replace('-', "_");

        for w in widths {
            for op in ops {
                let is_recip = matches!(op.as_str(), "recip" | "rsqrt");
                if is_recip && !w.has_recip {
                    continue;
                }
                let case_key = if is_recip {
                    "recip_cases"
                } else {
                    "minmax_cases"
                };
                let mut arms: Vec<(Option<u32>, String)> =
                    vec![(None, format!("existing Magetypes {op}"))];
                if let Some(cases) = flags_map.get(case_key) {
                    for (bits, label) in cases {
                        arms.push((Some(*bits), label.clone()));
                    }
                }

                for (bits_opt, arm_label) in arms {
                    let arm_ident = if bits_opt.is_none() {
                        "legacy"
                    } else {
                        arm_label.as_str()
                    };
                    let expr = match (op.as_str(), bits_opt) {
                        ("min", None) => "va.min(vb)".to_string(),
                        ("min", Some(b)) => format!(
                            "magemath::minmax::try_min_{}_with::<{b}, _>(t, va, vb).unwrap()",
                            w.name
                        ),
                        ("max", None) => "va.max(vb)".to_string(),
                        ("max", Some(b)) => format!(
                            "magemath::minmax::try_max_{}_with::<{b}, _>(t, va, vb).unwrap()",
                            w.name
                        ),
                        ("recip", None) => "va.recip()".to_string(),
                        ("recip", Some(b)) => {
                            format!(
                                "magemath::recip::try_recip_{}_with::<{b}, _>(t, va).unwrap()",
                                w.name
                            )
                        }
                        ("rsqrt", None) => "va.rsqrt()".to_string(),
                        ("rsqrt", Some(b)) => {
                            format!(
                                "magemath::recip::try_rsqrt_{}_with::<{b}, _>(t, va).unwrap()",
                                w.name
                            )
                        }
                        (other, _) => bail!("unsupported --ops entry '{other}'"),
                    };

                    let sym_stream =
                        format!("mca_{tier_ident}_{op}_{}_stream64_{arm_ident}", w.name);
                    if is_recip {
                        let _ = writeln!(
                            src,
                            "\n#[unsafe(no_mangle)]\n#[inline(never)]\n{tf_attr}\n\
                             pub fn {sym_stream}(t: {}, a: &[{}; 64], out: &mut [{}; 64]) {{\n\
                                 use magetypes::simd::generic::{} as V;\n\
                                 for (ca, cout) in a.as_chunks::<{}>().0.iter().zip(out.as_chunks_mut::<{}>().0) {{\n\
                                     let va = V::from_array_t(t, *ca);\n\
                                     *cout = ({expr}).to_array();\n\
                                 }}\n\
                             }}",
                            spec.token_ty, w.elem, w.elem, w.name, w.lanes, w.lanes
                        );
                    } else {
                        let _ = writeln!(
                            src,
                            "\n#[unsafe(no_mangle)]\n#[inline(never)]\n{tf_attr}\n\
                             pub fn {sym_stream}(t: {}, a: &[{}; 64], b: &[{}; 64], out: &mut [{}; 64]) {{\n\
                                 use magetypes::simd::generic::{} as V;\n\
                                 for ((ca, cb), cout) in a.as_chunks::<{}>().0.iter().zip(b.as_chunks::<{}>().0).zip(out.as_chunks_mut::<{}>().0) {{\n\
                                     let va = V::from_array_t(t, *ca);\n\
                                     let vb = V::from_array_t(t, *cb);\n\
                                     *cout = ({expr}).to_array();\n\
                                 }}\n\
                             }}",
                            spec.token_ty,
                            w.elem,
                            w.elem,
                            w.elem,
                            w.name,
                            w.lanes,
                            w.lanes,
                            w.lanes
                        );
                    }
                    manifest.push(ProbeManifestEntry {
                        symbol: sym_stream,
                        tier: tier.clone(),
                        op: op.clone(),
                        width: w.name.to_string(),
                        shape: "stream64",
                        arm: arm_label.clone(),
                    });

                    let sym_chain = format!("mca_{tier_ident}_{op}_{}_chain64_{arm_ident}", w.name);
                    let chain_step = if is_recip {
                        expr.replace("va", "x")
                    } else {
                        expr.replace("va", "x")
                            .replace("vb", &format!("V::splat_t(t, 2.0 as {})", w.elem))
                    };
                    let add_const = if is_recip { "0.25" } else { "0.03125" };
                    let _ = writeln!(
                        src,
                        "\n#[unsafe(no_mangle)]\n#[inline(never)]\n{tf_attr}\n\
                         pub fn {sym_chain}(t: {}, initial: {}) -> [{}; {}] {{\n\
                             use magetypes::simd::generic::{} as V;\n\
                             let mut x = V::splat_t(t, initial);\n\
                             for _ in 0..64 {{\n\
                                 x = ({chain_step}) + V::splat_t(t, {add_const} as {});\n\
                             }}\n\
                             x.to_array()\n\
                         }}",
                        spec.token_ty, w.elem, w.elem, w.lanes, w.name, w.elem
                    );
                    manifest.push(ProbeManifestEntry {
                        symbol: sym_chain,
                        tier: tier.clone(),
                        op: op.clone(),
                        width: w.name.to_string(),
                        shape: "chain64",
                        arm: arm_label,
                    });
                }
            }
        }
    }

    fs::write(probe_dir.join("src/lib.rs"), src)?;
    Ok(manifest)
}

const SKIP_DIRECTIVES: &[&str] = &[
    ".cfi_",
    ".loc",
    ".file",
    ".p2align",
    ".align",
    ".type",
    ".size",
    ".section",
    ".text",
    ".globl",
    ".hidden",
    ".local",
    ".functype",
    "#",
    "//",
];

fn is_directive_or_blank(line: &str) -> bool {
    let s = line.trim();
    s.is_empty() || SKIP_DIRECTIVES.iter().any(|d| s.starts_with(d))
}

fn parse_asm_functions(asm_text: &str) -> BTreeMap<String, Vec<String>> {
    let alias_re =
        Regex::new(r"^(?:\.set\s+)?_?(mca_[A-Za-z0-9_]+)\s*(?:=|,)\s*_?(mca_[A-Za-z0-9_]+)\s*$")
            .unwrap();
    let label_re = Regex::new(r"^_?(mca_[A-Za-z0-9_]+):(?:\s*(?:#|//).*)?$").unwrap();

    let mut funcs: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut aliases: BTreeMap<String, String> = BTreeMap::new();
    let mut cur_names: Vec<String> = Vec::new();
    let mut cur_lines: Vec<String> = Vec::new();

    for raw in asm_text.lines() {
        let s = raw.trim();
        if let Some(cap) = alias_re.captures(s) {
            aliases.insert(cap[1].to_string(), cap[2].to_string());
            continue;
        }
        if let Some(cap) = label_re.captures(s) {
            let sym = cap[1].to_string();
            if cur_names.is_empty() {
                cur_names.push(sym);
                cur_lines.clear();
            } else if cur_lines.iter().all(|ln| is_directive_or_blank(ln)) {
                cur_names.push(sym);
            } else {
                cur_lines.push(raw.to_string());
            }
            continue;
        }
        if !cur_names.is_empty() {
            if matches!(s, ".cfi_endproc" | "end_function") || s.starts_with(".Lfunc_end") {
                for name in cur_names.drain(..) {
                    funcs.insert(name, cur_lines.clone());
                }
                cur_lines.clear();
            } else {
                cur_lines.push(raw.to_string());
            }
        }
    }

    for src in aliases.keys().cloned().collect::<Vec<_>>() {
        let mut cursor = src.as_str();
        let mut depth = 0usize;
        while depth < 16 {
            if let Some(body) = funcs.get(cursor).cloned() {
                funcs.insert(src.clone(), body);
                break;
            }
            match aliases.get(cursor) {
                Some(next) => {
                    cursor = next;
                    depth += 1;
                }
                None => break,
            }
        }
    }

    funcs
}

#[derive(Clone, Serialize)]
struct AsmMetrics {
    total_insts: usize,
    hot_insts: usize,
    mca_insts: usize,
    mca_is_loop: bool,
    hot_calls: Vec<String>,
    cold_calls: Vec<String>,
    stack_ops: usize,
}

fn normalize_body(lines: &[String]) -> (Vec<String>, Vec<String>, AsmMetrics) {
    let lbl_re = Regex::new(r"\.LBB\d+_\d+|\.Ltmp\d+").unwrap();
    let cpi_re = Regex::new(r"\.LCPI\d+_\d+").unwrap();
    let anon_re = Regex::new(r"\.Lanon\.[0-9a-f]+\.\d+").unwrap();
    let ws_re = Regex::new(r"\s+").unwrap();
    let x86_sp_re = Regex::new(r"^subq?\s+\$\d+,\s*%rsp").unwrap();
    let arm_sp_re = Regex::new(r"^sub\s+sp,\s*sp,").unwrap();

    let mut label_map: BTreeMap<String, String> = BTreeMap::new();
    let mut const_map: BTreeMap<String, String> = BTreeMap::new();

    let map_tokens =
        |re: &Regex, text: &str, table: &mut BTreeMap<String, String>, prefix: &str| -> String {
            let mut out = String::with_capacity(text.len());
            let mut last = 0usize;
            for m in re.find_iter(text) {
                out.push_str(&text[last..m.start()]);
                let next_id = table.len();
                let mapped = table
                    .entry(m.as_str().to_string())
                    .or_insert_with(|| format!("{prefix}{next_id}"));
                out.push_str(mapped);
                last = m.end();
            }
            out.push_str(&text[last..]);
            out
        };

    let mut norm = Vec::new();
    let mut hot_norm = Vec::new();
    let mut hot_lines = Vec::new();
    let mut past_ret = false;
    let mut hot_calls = Vec::new();
    let mut cold_calls = Vec::new();
    let mut stack_ops = 0usize;
    let mut total_insts = 0usize;

    for raw in lines {
        let mut s = raw.trim();
        if s.is_empty() {
            continue;
        }
        if let Some((before, _)) = s.split_once("//") {
            s = before.trim();
        }
        if !s.starts_with('#')
            && let Some((before, _)) = s.split_once(" #")
        {
            s = before.trim();
        }
        if is_directive_or_blank(s) {
            continue;
        }

        if s.ends_with(':') {
            let n_lbl = map_tokens(&lbl_re, s, &mut label_map, ".L");
            norm.push(n_lbl.clone());
            if !past_ret {
                hot_norm.push(n_lbl);
                hot_lines.push(s.to_string());
            }
            continue;
        }

        total_insts += 1;
        let step1 = map_tokens(&lbl_re, s, &mut label_map, ".L");
        let step2 = map_tokens(&cpi_re, &step1, &mut const_map, ".LCPI");
        let step3 = anon_re.replace_all(&step2, ".Lanon");
        let n_inst = ws_re.replace_all(&step3, " ").into_owned();
        norm.push(n_inst.clone());

        let op = s.split_whitespace().next().unwrap_or("");
        if x86_sp_re.is_match(s)
            || arm_sp_re.is_match(s)
            || raw.contains("global.get\t$__stack_pointer")
            || raw.contains("global.set\t$__stack_pointer")
        {
            stack_ops += 1;
        }

        if matches!(op, "call" | "callq" | "bl" | "blr") {
            let is_cold_panic =
                s.contains("unwrap_failed") || s.contains("panic") || s.contains("handle_error");
            if past_ret || is_cold_panic {
                cold_calls.push(s.to_string());
            } else {
                hot_calls.push(s.to_string());
            }
        }

        if !past_ret {
            hot_norm.push(n_inst);
            if matches!(op, "ret" | "retq" | "end_function") {
                past_ret = true;
            } else {
                hot_lines.push(raw.clone());
            }
        }
    }

    let (mca_is_loop, mca_lines) = extract_mca_region(&hot_lines);
    let hot_insts = hot_lines
        .iter()
        .filter(|ln| !ln.trim().ends_with(':'))
        .count();
    let metrics = AsmMetrics {
        total_insts,
        hot_insts,
        mca_insts: mca_lines.len(),
        mca_is_loop,
        hot_calls,
        cold_calls,
        stack_ops,
    };
    let canon = if hot_norm.is_empty() { norm } else { hot_norm };
    (canon, mca_lines, metrics)
}

fn extract_mca_region(hot_lines: &[String]) -> (bool, Vec<String>) {
    let mut label_positions: BTreeMap<String, usize> = BTreeMap::new();
    for (idx, raw) in hot_lines.iter().enumerate() {
        let s = raw.trim();
        if let Some(lbl) = s.strip_suffix(':') {
            label_positions.insert(lbl.to_string(), idx);
        }
    }

    let mut best_loop: Option<Vec<String>> = None;
    for (idx, raw) in hot_lines.iter().enumerate() {
        let s = raw.trim();
        if s.ends_with(':') {
            continue;
        }
        for (lbl, &pos) in &label_positions {
            if pos < idx && s.contains(lbl.as_str()) {
                let region: Vec<String> = hot_lines[pos + 1..=idx]
                    .iter()
                    .filter(|ln| !ln.trim().ends_with(':'))
                    .cloned()
                    .collect();
                if !region.is_empty() && best_loop.as_ref().is_none_or(|b| region.len() > b.len()) {
                    best_loop = Some(region);
                }
            }
        }
    }
    if let Some(region) = best_loop {
        return (true, region);
    }

    let straight = hot_lines
        .iter()
        .filter(|raw| {
            let s = raw.trim();
            if s.ends_with(':') {
                return false;
            }
            let op = s.split_whitespace().next().unwrap_or("");
            if (op.starts_with('j') || op.starts_with("b.")) && s.contains(".LBB") {
                return false;
            }
            if matches!(op, "call" | "callq" | "bl")
                && (s.contains("unwrap_failed") || s.contains("panic"))
            {
                return false;
            }
            true
        })
        .cloned()
        .collect();
    (false, straight)
}

/// Deterministic 64-bit FNV-1a hex digest (12 hex chars) — zero external crate deps.
fn fnv1a_hex12(lines: &[String]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            hash ^= b'\n' as u64;
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        for &b in line.as_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    format!("{hash:016x}")[..12].to_string()
}

fn find_llvm_mca() -> Option<String> {
    for cand in ["llvm-mca-21", "llvm-mca-20", "llvm-mca-19", "llvm-mca"] {
        if Command::new(cand)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
        {
            return Some(cand.to_string());
        }
    }
    None
}

#[derive(Clone, Serialize)]
#[serde(untagged)]
enum McaScore {
    Ok {
        cycles_per_iter: Option<f64>,
        uops_per_iter: Option<f64>,
        block_rthroughput: Option<f64>,
        ipc: Option<f64>,
    },
    Err {
        error: String,
    },
}

fn run_llvm_mca(mca_bin: &str, triple: &str, cpu: &str, inst_lines: &[String]) -> McaScore {
    if inst_lines.is_empty() {
        return McaScore::Err {
            error: "empty snippet".to_string(),
        };
    }
    let rip_re = Regex::new(r"\.?LCPI[0-9_]+\(%rip\)").unwrap();
    let adrp_re = Regex::new(r"^\s*adrp\s+\w+,\s*\S+").unwrap();
    let lo12_br_re = Regex::new(r":lo12:\S+\]").unwrap();
    let lo12_imm_re = Regex::new(r",\s*:lo12:\S+").unwrap();

    let mut asm_input = String::new();
    for raw in inst_lines {
        if adrp_re.is_match(raw) {
            continue;
        }
        let s1 = rip_re.replace_all(raw, "0(%rip)");
        let s2 = lo12_br_re.replace_all(&s1, "#0]");
        let s3 = lo12_imm_re.replace_all(&s2, ", #0");
        asm_input.push_str(&s3);
        asm_input.push('\n');
    }

    let mut child = match Command::new(mca_bin)
        .args([
            &format!("-mtriple={triple}"),
            &format!("-mcpu={cpu}"),
            "-iterations=100",
            "-timeline=false",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return McaScore::Err {
                error: e.to_string(),
            };
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(asm_input.as_bytes());
    }
    let out = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => {
            return McaScore::Err {
                error: e.to_string(),
            };
        }
    };
    if !out.status.success() {
        let msg = String::from_utf8_lossy(&out.stderr)
            .lines()
            .next()
            .unwrap_or("mca failed")
            .to_string();
        return McaScore::Err { error: msg };
    }

    let text = String::from_utf8_lossy(&out.stdout);
    let grab = |pat: &str| -> Option<f64> {
        Regex::new(pat)
            .ok()?
            .captures(&text)?
            .get(1)?
            .as_str()
            .parse()
            .ok()
    };
    let iters = grab(r"Iterations:\s+(\d+)").unwrap_or(100.0);
    let round2 = |v: f64| (v * 100.0).round() / 100.0;
    McaScore::Ok {
        cycles_per_iter: grab(r"Total Cycles:\s+(\d+)").map(|c| round2(c / iters)),
        uops_per_iter: grab(r"Total uOps:\s+(\d+)").map(|u| round2(u / iters)),
        block_rthroughput: grab(r"Block RThroughput:\s+([0-9.]+)"),
        ipc: grab(r"IPC:\s+([0-9.]+)"),
    }
}

fn ensure_remote_repo(root: &Path, remote: &str, explicit_repo: Option<&str>) -> Result<String> {
    if let Some(r) = explicit_repo {
        return Ok(r.to_string());
    }
    let remote_dir = "~/tmp/archmage-mca-repo";
    let status = Command::new("ssh")
        .args([
            remote,
            &format!("mkdir -p {remote_dir} && realpath {remote_dir}"),
        ])
        .output()
        .context("resolving remote repo path")?;
    if !status.status.success() {
        bail!("ssh {remote} mkdir failed");
    }
    let resolved = String::from_utf8_lossy(&status.stdout).trim().to_string();
    let sync = Command::new("rsync")
        .args([
            "-az",
            "--delete",
            "--exclude=.git",
            "--exclude=.jj",
            "--exclude=target",
            "--exclude=benchmarks",
            &format!("{}/", root.display()),
            &format!("{remote}:{resolved}/"),
        ])
        .status()
        .context("rsyncing workspace to remote host")?;
    if !sync.success() {
        bail!("rsync workspace to {remote}:{resolved} failed");
    }
    Ok(resolved)
}

fn compile_asm(
    probe_dir: &Path,
    target: &str,
    rustflags: &str,
    remote: Option<&str>,
) -> Result<String> {
    if let Some(host) = remote {
        let remote_probe = "~/tmp/archmage-mca-probe";
        let st = Command::new("ssh")
            .args([
                host,
                &format!("mkdir -p {remote_probe}/src_sync {remote_probe}/src"),
            ])
            .status()?;
        if !st.success() {
            bail!("ssh {host} mkdir failed");
        }
        let rs = Command::new("rsync")
            .args([
                "-az",
                &probe_dir.join("Cargo.toml").display().to_string(),
                &probe_dir.join("src/lib.rs").display().to_string(),
                &format!("{host}:{remote_probe}/src_sync/"),
            ])
            .status()?;
        if !rs.success() {
            bail!("rsync probe to {host} failed");
        }
        let remote_cmd = format!(
            "export PATH=\"$HOME/.cargo/bin:$HOME/.local/share/mise/shims:$PATH\" && \
             mv {remote_probe}/src_sync/Cargo.toml {remote_probe}/Cargo.toml && \
             mv {remote_probe}/src_sync/lib.rs {remote_probe}/src/lib.rs && \
             rmdir {remote_probe}/src_sync && \
             cd {remote_probe} && \
             RUSTFLAGS=\"{rustflags}\" ~/work/claudehints/scripts/run-heavy --mem 16G -- \
             cargo rustc --release --lib --target {target} -- --emit=asm >&2 && \
             ls -t target/{target}/release/deps/archmage_mca_probe-*.s | head -n1 | xargs cat"
        );
        let out = Command::new("ssh")
            .args([
                host,
                &format!("bash -lc '{}'", remote_cmd.replace('\'', "'\\''")),
            ])
            .output()
            .context("running remote cargo rustc")?;
        for line in String::from_utf8_lossy(&out.stderr).lines() {
            if line.contains("run-heavy:") {
                eprintln!("  [{host} {target}] {line}");
            }
        }
        if !out.status.success() {
            bail!(
                "remote cargo rustc failed for {target}:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        return String::from_utf8(out.stdout).context("remote .s output not UTF-8");
    }

    let tmp_dir = home_tmp()?;
    let st = Command::new("cargo")
        .args([
            "rustc",
            "--release",
            "--lib",
            "--manifest-path",
            &probe_dir.join("Cargo.toml").display().to_string(),
            "--target",
            target,
            "--",
            "--emit=asm",
        ])
        .env("RUSTFLAGS", rustflags)
        .env("TMPDIR", &tmp_dir)
        .status()
        .context("running local cargo rustc")?;
    if !st.success() {
        bail!("cargo rustc --emit=asm failed for {target}");
    }
    let deps_dir = probe_dir.join("target").join(target).join("release/deps");
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in
        fs::read_dir(&deps_dir).with_context(|| format!("reading {}", deps_dir.display()))?
    {
        let entry = entry?;
        let p = entry.path();
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name.starts_with("archmage_mca_probe-") && name.ends_with(".s") {
            let mtime = entry.metadata()?.modified()?;
            if newest.as_ref().is_none_or(|(best, _)| mtime > *best) {
                newest = Some((mtime, p));
            }
        }
    }
    let (_, asm_path) =
        newest.with_context(|| format!("no .s emitted in {}", deps_dir.display()))?;
    fs::read_to_string(&asm_path).with_context(|| format!("reading {}", asm_path.display()))
}

#[derive(Clone, Serialize)]
struct EquivalenceClass {
    tier: String,
    op: String,
    width: String,
    shape: String,
    class_id: String,
    representative: String,
    members: Vec<String>,
    metrics: AsmMetrics,
    mca: BTreeMap<String, McaScore>,
    sample_asm: Vec<String>,
}

#[derive(Clone, Serialize)]
struct DedupEntry {
    class_id: String,
    representative: String,
    members: Vec<String>,
}

#[derive(Serialize)]
struct McaResultsJson {
    total_arms: usize,
    unique_classes: usize,
    dedup_factor: f64,
    unexpected_simd_calls: Vec<(String, String, String, Vec<String>)>,
    classes: Vec<EquivalenceClass>,
}

/// Entry point for `cargo xtask mca`.
pub fn run_mca(args: &[String]) -> Result<()> {
    let root = std::env::current_dir()?;
    let cfg = parse_args(args)?;
    let reg = Registry::load(&root.join("token-registry.toml"))?;
    let flags_map = load_policy_flags(&root)?;
    let mca_bin = find_llvm_mca();

    fs::create_dir_all(&cfg.out_dir)?;
    let probe_dir = home_tmp()?.join("archmage-mca-probe-local");
    fs::create_dir_all(&probe_dir)?;

    let repo_path_for_cargo = if let Some(remote) = &cfg.remote {
        ensure_remote_repo(&root, remote, cfg.remote_repo.as_deref())?
    } else {
        root.display().to_string()
    };

    // Group requested tiers by `(target, rustflags)` so each target compiles once.
    let mut compile_groups: Vec<((String, String), Vec<String>)> = Vec::new();
    for tier in &cfg.tiers {
        let spec = tier_spec(tier)?;
        let key = (spec.target.to_string(), spec.rustflags.to_string());
        if let Some((_, list)) = compile_groups.iter_mut().find(|(k, _)| *k == key) {
            list.push(tier.clone());
        } else {
            compile_groups.push((key, vec![tier.clone()]));
        }
    }

    let mut all_classes: Vec<EquivalenceClass> = Vec::new();
    let mut dedup_map: BTreeMap<String, Vec<DedupEntry>> = BTreeMap::new();
    let mut total_arms = 0usize;
    let mut total_unique = 0usize;

    for ((target, rustflags), group_tiers) in &compile_groups {
        let manifest = generate_probe_crate(
            &probe_dir,
            &repo_path_for_cargo,
            &reg,
            group_tiers,
            &cfg.ops,
            &cfg.widths,
            &flags_map,
        )?;
        let asm_text = compile_asm(&probe_dir, target, rustflags, cfg.remote.as_deref())?;
        fs::write(
            cfg.out_dir.join(format!("{}.s", group_tiers.join("_"))),
            &asm_text,
        )?;
        let asm_funcs = parse_asm_functions(&asm_text);

        type CellKey = (String, String, String, &'static str);
        let mut cells: Vec<(CellKey, Vec<ProbeManifestEntry>)> = Vec::new();
        for item in manifest {
            let key = (
                item.tier.clone(),
                item.op.clone(),
                item.width.clone(),
                item.shape,
            );
            if let Some((_, list)) = cells.iter_mut().find(|(k, _)| *k == key) {
                list.push(item);
            } else {
                cells.push((key, vec![item]));
            }
        }

        for ((tier, op, width, shape), items) in cells {
            let spec = tier_spec(&tier)?;
            let mut classes_for_cell: Vec<EquivalenceClass> = Vec::new();
            for item in items {
                total_arms += 1;
                let raw_lines = asm_funcs
                    .get(&item.symbol)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                let (norm_lines, mca_lines, metrics) = normalize_body(raw_lines);
                let digest = fnv1a_hex12(&norm_lines);
                if let Some(existing) = classes_for_cell.iter_mut().find(|c| c.class_id == digest) {
                    existing.members.push(item.arm);
                } else {
                    let mut mca_scores = BTreeMap::new();
                    if let (Some(bin), Some(triple)) = (&mca_bin, spec.mca_triple) {
                        for &cpu in spec.mca_cpus {
                            mca_scores.insert(
                                cpu.to_string(),
                                run_llvm_mca(bin, triple, cpu, &mca_lines),
                            );
                        }
                    }
                    classes_for_cell.push(EquivalenceClass {
                        tier: tier.clone(),
                        op: op.clone(),
                        width: width.clone(),
                        shape: shape.to_string(),
                        class_id: digest,
                        representative: item.arm.clone(),
                        members: vec![item.arm],
                        metrics,
                        mca: mca_scores,
                        sample_asm: norm_lines.into_iter().take(24).collect(),
                    });
                }
            }

            total_unique += classes_for_cell.len();
            let cell_key = format!("{tier}/{op}/{width}/{shape}");
            dedup_map.insert(
                cell_key,
                classes_for_cell
                    .iter()
                    .map(|c| DedupEntry {
                        class_id: c.class_id.clone(),
                        representative: c.representative.clone(),
                        members: c.members.clone(),
                    })
                    .collect(),
            );
            all_classes.extend(classes_for_cell);
        }
    }

    let dedup_factor = total_arms as f64 / (total_unique.max(1) as f64);
    let mut md = format!(
        "# Static Codegen Equivalence & `llvm-mca` Report\n\n\
         - **Total `(tier, op, width, shape, policy)` arms analyzed:** `{total_arms}`\n\
         - **Unique normalized machine-code classes:** `{total_unique}` (**`{dedup_factor:.2}x` benchmark deduplication factor**)\n\n\
         | Tier | Cell | Class | Rep | Members | Hot Insts | Hot Calls | Stack Ops | Primary MCA (cyc/iter, RThr) | Secondary MCA (cyc/iter, RThr) |\n\
         |---|---|---|---|---:|---:|---:|---:|---|---|\n"
    );

    let mut unexpected_simd_calls = Vec::new();
    for c in &all_classes {
        let spec = tier_spec(&c.tier)?;
        let fmt_mca = |idx: usize| -> String {
            let Some(&cpu) = spec.mca_cpus.get(idx) else {
                return "—".to_string();
            };
            match c.mca.get(cpu) {
                Some(McaScore::Ok {
                    cycles_per_iter,
                    block_rthroughput,
                    ..
                }) => format!(
                    "{cpu}: {}c (RThr {})",
                    cycles_per_iter
                        .map(|v| format!("{v}"))
                        .unwrap_or_else(|| "?".to_string()),
                    block_rthroughput
                        .map(|v| format!("{v}"))
                        .unwrap_or_else(|| "?".to_string())
                ),
                Some(McaScore::Err { error }) => format!("{cpu}: {error}"),
                None => "—".to_string(),
            }
        };

        let members_preview = if c.members.len() <= 4 {
            c.members.join(", ")
        } else {
            format!(
                "{}, +{} more ({})",
                c.members[..3].join(", "),
                c.members.len() - 3,
                c.members.len()
            )
        };
        let _ = writeln!(
            md,
            "| `{}` | `{}/{}/{}` | `{}` | `{}` | {} ({members_preview}) | {} | {} | {} | {} | {} |",
            c.tier,
            c.op,
            c.width,
            c.shape,
            &c.class_id[..8],
            c.representative,
            c.members.len(),
            c.metrics.hot_insts,
            c.metrics.hot_calls.len(),
            c.metrics.stack_ops,
            fmt_mca(0),
            fmt_mca(1),
        );
        if c.tier != "scalar" && !c.metrics.hot_calls.is_empty() {
            unexpected_simd_calls.push((
                c.tier.clone(),
                format!("{}/{}/{}", c.op, c.width, c.shape),
                c.representative.clone(),
                c.metrics.hot_calls.clone(),
            ));
        }
    }

    if unexpected_simd_calls.is_empty() {
        md.push_str(
            "\n## Hot-Path Call Gate\n\n\
             - **PASS:** `0` hot-path `call`/`bl` instructions across all analyzed SIMD kernels.\n",
        );
    } else {
        md.push_str("\n## Hot-Path Call Sites in SIMD Tiers\n\n");
        for (tier, cell, rep, calls) in &unexpected_simd_calls {
            let _ = writeln!(md, "- `{tier}` `{cell}` (`{rep}`): `{calls:?}`");
        }
    }

    fs::write(cfg.out_dir.join("report.md"), &md)?;
    let results_json = McaResultsJson {
        total_arms,
        unique_classes: total_unique,
        dedup_factor: (dedup_factor * 1000.0).round() / 1000.0,
        unexpected_simd_calls: unexpected_simd_calls.clone(),
        classes: all_classes,
    };
    fs::write(
        cfg.out_dir.join("mca-results.json"),
        format!("{}\n", serde_json::to_string_pretty(&results_json)?),
    )?;
    fs::write(
        cfg.out_dir.join("dedup-classes.json"),
        format!("{}\n", serde_json::to_string_pretty(&dedup_map)?),
    )?;

    let mut tsv = String::new();
    for (cell_key, entries) in &dedup_map {
        for entry in entries {
            let _ = writeln!(
                tsv,
                "{cell_key}\t{}\t{}\t{}",
                entry.representative,
                entry.class_id,
                entry.members.join(",")
            );
        }
    }
    fs::write(cfg.out_dir.join("dedup-classes.tsv"), tsv)?;

    let _ = fs::remove_dir_all(&probe_dir);
    println!(
        "Analyzed {total_arms} arms -> {total_unique} unique classes \
         ({dedup_factor:.2}x dedup); SIMD hot calls: {}",
        unexpected_simd_calls.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_icf_aliases_and_ignores_post_ret_cold_panic_differences() {
        let asm = "\
mca_v3_min_f32x8_stream64_a:\n\
\tvminps\t(%rsi), %ymm0, %ymm0\n\
\tretq\n\
.LBB0_2:\n\
\tleaq\t.Lanon.deadbeef.1(%rip), %rdi\n\
\tcallq\tcore::result::unwrap_failed@PLT\n\
.Lfunc_end0:\n\
\t.set mca_v3_min_f32x8_stream64_b, mca_v3_min_f32x8_stream64_a\n\
mca_v3_min_f32x8_stream64_c:\n\
\tvminps\t(%rsi), %ymm0, %ymm0\n\
\tretq\n\
.LBB1_2:\n\
\tleaq\t.Lanon.cafebabe.9(%rip), %rdi\n\
\tcallq\tcore::result::unwrap_failed@PLT\n\
.Lfunc_end1:\n";
        let funcs = parse_asm_functions(asm);
        assert_eq!(funcs.len(), 3);
        let (na, _, ma) = normalize_body(&funcs["mca_v3_min_f32x8_stream64_a"]);
        let (nb, _, mb) = normalize_body(&funcs["mca_v3_min_f32x8_stream64_b"]);
        let (nc, _, mc) = normalize_body(&funcs["mca_v3_min_f32x8_stream64_c"]);
        assert_eq!(fnv1a_hex12(&na), fnv1a_hex12(&nb));
        assert_eq!(fnv1a_hex12(&na), fnv1a_hex12(&nc));
        assert!(ma.hot_calls.is_empty());
        assert!(mb.hot_calls.is_empty());
        assert!(mc.hot_calls.is_empty());
        assert_eq!(ma.cold_calls.len(), 1);
    }
}

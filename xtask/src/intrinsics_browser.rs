//! Intrinsics browser data generator.
//!
//! Produces:
//! - `docs/intrinsics-browser/data/intrinsics.json` — search database
//! - `docs/intrinsics-browser/tokens/*.md` — per-token AI-friendly reference files
//!
//! **Auto-generated** by `cargo xtask generate` — do not edit output files manually.

use crate::registry::Registry;
use anyhow::{Context, Result};
use indoc::formatdoc;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

// ============================================================================
// CSV Parsing (RFC 4180-aware)
// ============================================================================

/// A parsed intrinsic entry from the CSV.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CsvIntrinsic {
    arch: String,
    name: String,
    features: String,
    is_unsafe: bool,
    stability: String,
    file: String,
    doc: String,
    signature: String,
    instruction: String,
}

/// Parse the CSV with proper RFC 4180 quoting support.
fn parse_csv(content: &str) -> Vec<CsvIntrinsic> {
    let mut results = Vec::new();
    let mut lines = content.lines();
    lines.next(); // skip header

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_line(line);
        if fields.len() < 6 {
            continue;
        }
        results.push(CsvIntrinsic {
            arch: fields[0].clone(),
            name: fields[1].clone(),
            features: fields[2].clone(),
            is_unsafe: fields[3] == "True",
            stability: fields[4].clone(),
            file: fields[5].clone(),
            doc: fields.get(6).cloned().unwrap_or_default(),
            signature: fields.get(7).cloned().unwrap_or_default(),
            instruction: fields.get(8).cloned().unwrap_or_default(),
        });
    }
    results
}

/// Parse a single CSV line respecting RFC 4180 quoting.
pub(crate) fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                current.push(ch);
            }
        } else if ch == '"' {
            in_quotes = true;
        } else if ch == ',' {
            fields.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }
    fields.push(current);
    fields
}

// ============================================================================
// Feature → Token Mapping
// ============================================================================

/// Map of feature-set string → token name, built from the registry.
struct FeatureTokenMap {
    /// For each architecture, sorted list of (feature_set, token_name)
    /// where feature_set is the set of features the token provides.
    arch_tokens: HashMap<String, Vec<(HashSet<String>, String)>>,
}

impl FeatureTokenMap {
    fn from_registry(reg: &Registry) -> Self {
        let mut arch_tokens: HashMap<String, Vec<(HashSet<String>, String)>> = HashMap::new();

        for token in &reg.token {
            let arch = match token.arch.as_str() {
                "x86" => "x86_64",
                "arm" => "aarch64",
                "wasm" => "wasm32",
                other => other,
            };
            let features: HashSet<String> = token.features.iter().cloned().collect();
            arch_tokens
                .entry(arch.to_string())
                .or_default()
                .push((features, token.name.clone()));
        }

        // Sort each arch's tokens by feature count (ascending) so we find
        // the minimal (most specific but smallest) token first.
        for tokens in arch_tokens.values_mut() {
            tokens.sort_by_key(|(features, _)| features.len());
        }

        Self { arch_tokens }
    }

    /// Find the minimal token whose features are a superset of `required`.
    fn find_token(&self, arch: &str, required_features: &str) -> Option<String> {
        let arch_key = match arch {
            "x86" | "x86_64" => "x86_64",
            "aarch64" => "aarch64",
            "wasm32" => "wasm32",
            _ => return None,
        };

        let required: HashSet<&str> = required_features.split(',').map(str::trim).collect();

        let tokens = self.arch_tokens.get(arch_key)?;
        for (token_features, token_name) in tokens {
            let token_strs: HashSet<&str> = token_features.iter().map(String::as_str).collect();
            if required.iter().all(|r| token_strs.contains(r)) {
                return Some(token_name.clone());
            }
        }
        None
    }
}

// ============================================================================
// LLVM Timing Data
// ============================================================================

/// Operation timing: [latency, throughput] for each microarchitecture.
/// Keys: "h" = Haswell, "sk" = Skylake-X, "z4" = Zen 4, "sp" = Sapphire Rapids
type TimingEntry = BTreeMap<&'static str, [u32; 2]>;

/// Build the timing category lookup table.
/// Categories map intrinsic name patterns to operation types.
fn build_timing_table() -> BTreeMap<&'static str, TimingEntry> {
    let mut t = BTreeMap::new();

    // Helper to insert a timing entry
    macro_rules! timing {
        ($cat:expr, $h:expr, $sk:expr, $z4:expr, $sp:expr) => {
            t.insert(
                $cat,
                BTreeMap::from([("h", $h), ("sk", $sk), ("z4", $z4), ("sp", $sp)]),
            );
        };
    }

    // Float arithmetic (YMM = 256-bit)
    timing!("Float Add XMM", [3, 1], [4, 1], [3, 1], [4, 1]);
    timing!("Float Add YMM", [3, 1], [4, 1], [3, 1], [4, 1]);
    timing!("Float Add ZMM", [0, 0], [4, 1], [3, 1], [4, 1]);
    timing!("Float Mul XMM", [5, 1], [4, 1], [3, 1], [4, 1]);
    timing!("Float Mul YMM", [5, 1], [4, 1], [3, 1], [4, 1]);
    timing!("Float Mul ZMM", [0, 0], [4, 1], [3, 1], [4, 1]);
    timing!("Float FMA XMM", [5, 1], [4, 1], [4, 1], [4, 1]);
    timing!("Float FMA YMM", [5, 1], [4, 1], [4, 1], [4, 1]);
    timing!("Float FMA ZMM", [0, 0], [4, 1], [4, 1], [4, 1]);
    timing!("Float Div XMM f32", [13, 7], [11, 5], [10, 4], [11, 5]);
    timing!("Float Div YMM f32", [21, 14], [11, 10], [10, 7], [11, 10]);
    timing!("Float Div XMM f64", [20, 14], [14, 8], [13, 7], [14, 8]);
    timing!("Float Div YMM f64", [35, 28], [14, 16], [13, 10], [14, 16]);
    timing!("Float Sqrt XMM f32", [11, 7], [12, 6], [14, 5], [12, 6]);
    timing!("Float Sqrt YMM f32", [19, 14], [12, 12], [14, 7], [12, 12]);
    timing!("Float Sqrt XMM f64", [16, 14], [18, 12], [20, 9], [18, 12]);
    timing!("Float Sqrt YMM f64", [28, 28], [18, 24], [20, 13], [18, 24]);
    timing!("Float Rcp/Rsqrt XMM", [5, 1], [4, 1], [4, 1], [4, 1]);
    timing!("Float Rcp/Rsqrt YMM", [7, 1], [4, 1], [4, 1], [4, 1]);

    // Integer arithmetic
    timing!("Int Add XMM", [1, 1], [1, 1], [1, 1], [1, 1]);
    timing!("Int Add YMM", [1, 1], [1, 1], [1, 1], [1, 1]);
    timing!("Int Add ZMM", [0, 0], [1, 1], [1, 1], [1, 1]);
    timing!("Int Mul 16 XMM", [5, 1], [5, 1], [3, 1], [5, 1]);
    timing!("Int Mul 16 YMM", [5, 1], [5, 1], [3, 1], [5, 1]);
    timing!("Int Mul 32 XMM", [10, 2], [8, 1], [4, 1], [8, 1]);
    timing!("Int Mul 32 YMM", [10, 2], [8, 1], [4, 1], [8, 1]);

    // Shuffle/permute
    timing!("Shuffle XMM", [1, 1], [1, 1], [1, 1], [1, 1]);
    timing!("Shuffle YMM", [1, 1], [1, 1], [1, 1], [1, 1]);
    timing!("Shuffle ZMM", [0, 0], [1, 1], [1, 1], [1, 1]);
    timing!("Perm Cross-Lane YMM", [3, 1], [3, 1], [2, 1], [3, 1]);
    timing!("Perm Cross-Lane ZMM", [0, 0], [3, 1], [2, 1], [3, 1]);

    // Comparison
    timing!("Float Cmp XMM", [3, 1], [4, 1], [3, 1], [4, 1]);
    timing!("Float Cmp YMM", [3, 1], [4, 1], [3, 1], [4, 1]);
    timing!("Int Cmp XMM", [1, 1], [1, 1], [1, 1], [1, 1]);
    timing!("Int Cmp YMM", [1, 1], [1, 1], [1, 1], [1, 1]);

    // Bitwise
    timing!("Bitwise XMM", [1, 1], [1, 1], [1, 1], [1, 1]);
    timing!("Bitwise YMM", [1, 1], [1, 1], [1, 1], [1, 1]);
    timing!("Bitwise ZMM", [0, 0], [1, 1], [1, 1], [1, 1]);

    // Shift
    timing!("Shift XMM", [1, 1], [1, 1], [1, 1], [1, 1]);
    timing!("Shift YMM", [1, 1], [1, 1], [1, 1], [1, 1]);
    timing!("Shift ZMM", [0, 0], [1, 1], [1, 1], [1, 1]);

    // Conversion
    timing!("Float Cvt XMM", [4, 1], [4, 1], [3, 1], [4, 1]);
    timing!("Float Cvt YMM", [4, 1], [4, 1], [3, 1], [4, 1]);

    // Load/Store
    timing!("Load XMM", [5, 1], [5, 1], [4, 1], [5, 1]);
    timing!("Load YMM", [5, 1], [5, 1], [4, 1], [5, 1]);
    timing!("Load ZMM", [0, 0], [6, 1], [5, 1], [6, 1]);
    timing!("Store XMM", [1, 1], [1, 1], [1, 1], [1, 1]);
    timing!("Store YMM", [1, 1], [1, 1], [1, 1], [1, 1]);
    timing!("Store ZMM", [0, 0], [1, 1], [1, 1], [1, 1]);

    // Gather
    timing!("Gather YMM", [12, 4], [10, 1], [7, 1], [10, 1]);
    timing!("Gather ZMM", [0, 0], [12, 1], [8, 1], [12, 1]);

    // Blend/Select
    timing!("Blend XMM", [1, 1], [1, 1], [1, 1], [1, 1]);
    timing!("Blend YMM", [1, 1], [1, 1], [1, 1], [1, 1]);

    // Pack/Unpack
    timing!("Pack XMM", [1, 1], [1, 1], [1, 1], [1, 1]);
    timing!("Pack YMM", [1, 1], [1, 1], [1, 1], [1, 1]);

    // Horizontal ops (slow!)
    timing!("Horizontal XMM", [5, 2], [6, 2], [4, 2], [6, 2]);
    timing!("Horizontal YMM", [5, 2], [6, 2], [4, 2], [6, 2]);

    // AES
    timing!("AES XMM", [7, 1], [4, 1], [4, 1], [4, 1]);
    timing!("AES YMM", [0, 0], [0, 0], [4, 1], [4, 1]);

    // CLMUL
    timing!("CLMUL XMM", [7, 2], [5, 1], [4, 1], [5, 1]);

    // Popcnt
    timing!("Popcnt XMM", [0, 0], [3, 1], [2, 1], [3, 1]);
    timing!("Popcnt YMM", [0, 0], [3, 1], [2, 1], [3, 1]);

    t
}

/// Classify an intrinsic name into a timing category.
fn classify_timing(name: &str) -> Option<&'static str> {
    let width = if name.contains("512") {
        "ZMM"
    } else if name.contains("256") {
        "YMM"
    } else {
        "XMM"
    };

    // Float operations
    if name.contains("_add_p") || name.contains("_add_s") && name.contains("_mm") {
        return Some(match width {
            "ZMM" => "Float Add ZMM",
            "YMM" => "Float Add YMM",
            _ => "Float Add XMM",
        });
    }
    if name.contains("_mul_p") || name.contains("_mul_s") && name.contains("_mm") {
        return Some(match width {
            "ZMM" => "Float Mul ZMM",
            "YMM" => "Float Mul YMM",
            _ => "Float Mul XMM",
        });
    }
    if name.contains("fmadd")
        || name.contains("fmsub")
        || name.contains("fnmadd")
        || name.contains("fnmsub")
    {
        return Some(match width {
            "ZMM" => "Float FMA ZMM",
            "YMM" => "Float FMA YMM",
            _ => "Float FMA XMM",
        });
    }
    if name.contains("_div_p") || name.contains("_div_s") {
        if name.contains("_pd") || name.contains("_sd") {
            return Some(match width {
                "YMM" => "Float Div YMM f64",
                _ => "Float Div XMM f64",
            });
        }
        return Some(match width {
            "YMM" => "Float Div YMM f32",
            _ => "Float Div XMM f32",
        });
    }
    if name.contains("_sqrt_p") || name.contains("_sqrt_s") {
        if name.contains("_pd") || name.contains("_sd") {
            return Some(match width {
                "YMM" => "Float Sqrt YMM f64",
                _ => "Float Sqrt XMM f64",
            });
        }
        return Some(match width {
            "YMM" => "Float Sqrt YMM f32",
            _ => "Float Sqrt XMM f32",
        });
    }
    if name.contains("_rcp_") || name.contains("_rsqrt_") {
        return Some(match width {
            "YMM" => "Float Rcp/Rsqrt YMM",
            _ => "Float Rcp/Rsqrt XMM",
        });
    }
    if name.contains("_cmp_p")
        || name.contains("_cmpeq_p")
        || name.contains("_cmplt_p")
        || name.contains("_cmpgt_p")
        || name.contains("_cmpord_p")
        || name.contains("_cmpneq_p")
    {
        return Some(match width {
            "YMM" => "Float Cmp YMM",
            _ => "Float Cmp XMM",
        });
    }
    if name.contains("_cvt") && name.starts_with("_mm") {
        return Some(match width {
            "YMM" => "Float Cvt YMM",
            _ => "Float Cvt XMM",
        });
    }

    // Integer operations
    if name.contains("_add_epi")
        || name.contains("_sub_epi")
        || name.contains("_adds_epi")
        || name.contains("_subs_epi")
    {
        return Some(match width {
            "ZMM" => "Int Add ZMM",
            "YMM" => "Int Add YMM",
            _ => "Int Add XMM",
        });
    }
    if name.contains("_mullo_epi32") || name.contains("_mul_epu32") || name.contains("_mul_epi32") {
        return Some(match width {
            "YMM" => "Int Mul 32 YMM",
            _ => "Int Mul 32 XMM",
        });
    }
    if name.contains("_mullo_epi16")
        || name.contains("_mulhi_epi16")
        || name.contains("_mulhi_epu16")
    {
        return Some(match width {
            "YMM" => "Int Mul 16 YMM",
            _ => "Int Mul 16 XMM",
        });
    }
    if name.contains("_cmpeq_epi") || name.contains("_cmpgt_epi") || name.contains("_cmplt_epi") {
        return Some(match width {
            "YMM" => "Int Cmp YMM",
            _ => "Int Cmp XMM",
        });
    }

    // Bitwise
    if name.contains("_and_")
        || name.contains("_or_")
        || name.contains("_xor_")
        || name.contains("_andnot_")
    {
        return Some(match width {
            "ZMM" => "Bitwise ZMM",
            "YMM" => "Bitwise YMM",
            _ => "Bitwise XMM",
        });
    }

    // Shifts
    if name.contains("_sll")
        || name.contains("_srl")
        || name.contains("_sra")
        || name.contains("_slli_")
        || name.contains("_srli_")
        || name.contains("_srai_")
    {
        return Some(match width {
            "ZMM" => "Shift ZMM",
            "YMM" => "Shift YMM",
            _ => "Shift XMM",
        });
    }

    // Shuffle/permute
    if name.contains("_shuffle_") || name.contains("_shufflelo_") || name.contains("_shufflehi_") {
        return Some(match width {
            "ZMM" => "Shuffle ZMM",
            "YMM" => "Shuffle YMM",
            _ => "Shuffle XMM",
        });
    }
    if name.contains("_permute") || name.contains("_permutevar") {
        return Some(match width {
            "ZMM" => "Perm Cross-Lane ZMM",
            "YMM" => "Perm Cross-Lane YMM",
            _ => "Shuffle XMM",
        });
    }

    // Load/Store
    if name.contains("_loadu_")
        || name.contains("_load_")
        || name.contains("_lddqu_")
        || name.contains("_loadl_")
        || name.contains("_loadh_")
    {
        return Some(match width {
            "ZMM" => "Load ZMM",
            "YMM" => "Load YMM",
            _ => "Load XMM",
        });
    }
    if name.contains("_storeu_")
        || name.contains("_store_")
        || name.contains("_storel_")
        || name.contains("_storeh_")
        || name.contains("_stream_")
    {
        return Some(match width {
            "ZMM" => "Store ZMM",
            "YMM" => "Store YMM",
            _ => "Store XMM",
        });
    }

    // Gather
    if name.contains("_gather_") || name.contains("_i32gather_") || name.contains("_i64gather_") {
        return Some(match width {
            "ZMM" => "Gather ZMM",
            "YMM" => "Gather YMM",
            _ => "Gather YMM",
        });
    }

    // Blend
    if name.contains("_blend") {
        return Some(match width {
            "YMM" => "Blend YMM",
            _ => "Blend XMM",
        });
    }

    // Pack/Unpack
    if name.contains("_pack") || name.contains("_unpack") {
        return Some(match width {
            "YMM" => "Pack YMM",
            _ => "Pack XMM",
        });
    }

    // Horizontal
    if name.contains("_hadd_") || name.contains("_hsub_") {
        return Some(match width {
            "YMM" => "Horizontal YMM",
            _ => "Horizontal XMM",
        });
    }

    // AES
    if name.contains("_aes") {
        return Some(match width {
            "YMM" => "AES YMM",
            _ => "AES XMM",
        });
    }

    // CLMUL
    if name.contains("_clmulepi") {
        return Some("CLMUL XMM");
    }

    // POPCNT
    if name.contains("_popcnt_") {
        return Some(match width {
            "YMM" => "Popcnt YMM",
            _ => "Popcnt XMM",
        });
    }

    None
}

// ============================================================================
// Safe Variant Lookup
// ============================================================================

/// Build a set of safe_unaligned_simd function names and their signatures.
fn build_safe_variant_map() -> HashMap<String, String> {
    let safe_simd_path = match crate::find_safe_simd_path_simple() {
        Ok(p) => p,
        Err(_) => return HashMap::new(),
    };

    let mut map = HashMap::new();
    let ops = match crate::extract_safe_simd_functions(&safe_simd_path) {
        Ok(ops) => ops,
        Err(_) => return HashMap::new(),
    };

    for op in ops {
        map.insert(op.name.clone(), op.signature.clone());
    }
    map
}

// ============================================================================
// JSON Generation
// ============================================================================

/// Generate the complete intrinsics browser JSON and per-token markdown files.
pub fn generate_intrinsics_browser(reg: &Registry) -> Result<()> {
    let csv_path = PathBuf::from("docs/intrinsics/complete_intrinsics.csv");
    let content = fs::read_to_string(&csv_path).with_context(|| {
        format!(
            "Failed to read {}. Run: python3 xtask/extract_intrinsics.py > {}",
            csv_path.display(),
            csv_path.display()
        )
    })?;

    let intrinsics = parse_csv(&content);
    println!("  Loaded {} intrinsics from CSV", intrinsics.len());

    let feature_map = FeatureTokenMap::from_registry(reg);
    let safe_variants = build_safe_variant_map();
    let timing_table = build_timing_table();

    // Deduplicate: keep first occurrence of each name per arch
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let intrinsics: Vec<CsvIntrinsic> = intrinsics
        .into_iter()
        .filter(|i| seen.insert((i.arch.clone(), i.name.clone())))
        .collect();

    // Build token list for JSON
    let tokens_json: Vec<Value> = reg
        .token
        .iter()
        .map(|t| {
            let arch = match t.arch.as_str() {
                "x86" => "x86_64",
                "arm" => "aarch64",
                "wasm" => "wasm32",
                other => other,
            };
            json!({
                "name": t.name,
                "aliases": t.aliases,
                "arch": arch,
                "display": t.display_name.as_deref().unwrap_or(&t.name),
                "tier": t.short_name.as_deref().unwrap_or(""),
                "features": t.features,
                "parents": t.parents,
                "doc": t.doc.as_deref().unwrap_or("").lines().next().unwrap_or("")
            })
        })
        .collect();

    // Build intrinsics list for JSON
    let intrinsics_json: Vec<Value> = intrinsics
        .iter()
        .map(|i| {
            let token = feature_map.find_token(&i.arch, &i.features);
            let has_safe_variant = safe_variants.contains_key(&i.name);
            let timing_cat = classify_timing(&i.name);

            let mut obj = json!({
                "n": i.name,
                "a": i.arch,
                "f": i.features,
                "u": i.is_unsafe,
                "s": i.stability == "stable",
                "d": i.doc,
                "sig": i.signature,
                "ins": i.instruction,
            });

            if let Some(t) = &token {
                obj["t"] = json!(t);
            }
            if has_safe_variant {
                obj["sv"] = json!(true);
            }
            if let Some(cat) = timing_cat {
                obj["tc"] = json!(cat);
            }

            obj
        })
        .collect();

    // Build safe variants map for JSON
    let safe_json: BTreeMap<&str, &str> = safe_variants
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    // Build timing map for JSON
    let timing_json: BTreeMap<&str, Value> = timing_table
        .iter()
        .map(|(cat, entries)| {
            let vals: BTreeMap<&str, [u32; 2]> = entries
                .iter()
                .filter(|(_, v)| v[0] > 0 || v[1] > 0)
                .map(|(&k, v)| (k, *v))
                .collect();
            (*cat, json!(vals))
        })
        .collect();

    // Get Rust version
    let rust_version = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.split_whitespace().nth(1).unwrap_or("unknown").to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let root_json = json!({
        "tokens": tokens_json,
        "intrinsics": intrinsics_json,
        "safeVariants": safe_json,
        "timing": timing_json,
    });

    // Write JSON
    let out_dir = PathBuf::from("docs/intrinsics-browser/data");
    fs::create_dir_all(&out_dir)?;
    let json_path = out_dir.join("intrinsics.json");
    let json_str = serde_json::to_string_pretty(&root_json)?;
    fs::write(&json_path, &json_str)?;
    println!(
        "  Wrote {} ({} bytes, ~{}KB gzipped est.)",
        json_path.display(),
        json_str.len(),
        json_str.len() / 8
    );

    // Generate per-token markdown files
    generate_per_token_markdown(
        reg,
        &intrinsics,
        &feature_map,
        &safe_variants,
        &timing_table,
    )?;

    Ok(())
}

/// Get current date as YYYY-MM-DD string in UTC (no chrono dependency).
fn chrono_date() -> String {
    let output = std::process::Command::new("date")
        .arg("-u")
        .arg("+%Y-%m-%d")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok());
    output
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

// ============================================================================
// Per-Token Markdown Generation
// ============================================================================

fn generate_per_token_markdown(
    reg: &Registry,
    intrinsics: &[CsvIntrinsic],
    feature_map: &FeatureTokenMap,
    safe_variants: &HashMap<String, String>,
    timing_table: &BTreeMap<&str, TimingEntry>,
) -> Result<()> {
    let tokens_dir = PathBuf::from("docs/intrinsics-browser/tokens");
    fs::create_dir_all(&tokens_dir)?;

    for token_def in &reg.token {
        let token_name = &token_def.name;
        let arch = match token_def.arch.as_str() {
            "x86" => "x86_64",
            "arm" => "aarch64",
            "wasm" => "wasm32",
            other => other,
        };
        let display = token_def.display_name.as_deref().unwrap_or(token_name);
        let aliases_str = if token_def.aliases.is_empty() {
            String::new()
        } else {
            format!(" ({})", token_def.aliases.join(", "))
        };
        let doc_first_line = token_def
            .doc
            .as_deref()
            .unwrap_or("")
            .lines()
            .next()
            .unwrap_or("");

        // Find all intrinsics that map to this token
        let mut token_intrinsics: Vec<&CsvIntrinsic> = intrinsics
            .iter()
            .filter(|i| {
                if i.arch != arch && !(arch == "x86_64" && i.arch == "x86") {
                    return false;
                }
                feature_map.find_token(&i.arch, &i.features).as_deref() == Some(token_name.as_str())
            })
            .collect();
        token_intrinsics.sort_by(|a, b| a.name.cmp(&b.name));

        let safe_count = token_intrinsics.iter().filter(|i| !i.is_unsafe).count();
        let unsafe_count = token_intrinsics.iter().filter(|i| i.is_unsafe).count();
        let stable_count = token_intrinsics
            .iter()
            .filter(|i| i.stability == "stable")
            .count();
        let unstable_count = token_intrinsics
            .iter()
            .filter(|i| i.stability == "unstable" || i.stability == "unknown")
            .count();

        // Build usage example based on architecture
        let usage_example = build_usage_example(token_def, arch);

        // Build safe memory ops table
        let safe_ops_table = build_safe_ops_table(&token_intrinsics, safe_variants);

        // Build intrinsics table by category
        let intrinsics_tables =
            build_intrinsics_tables(&token_intrinsics, safe_variants, timing_table);

        let features_str = token_def.features.join(", ");

        let md = formatdoc! {r#"
            # {token_name}{aliases_str} — {display}

            {doc_first_line}

            **Architecture:** {arch} | **Features:** {features_str}
            **Total intrinsics:** {total} ({safe_count} safe, {unsafe_count} unsafe, {stable_count} stable, {unstable_count} unstable/unknown)

            ## Usage

            ```rust
            {usage_example}
            ```

            {safe_ops_table}

            ## All Intrinsics

            {intrinsics_tables}
        "#,
            total = token_intrinsics.len(),
        };

        let md_path = tokens_dir.join(format!("{token_name}.md"));
        fs::write(&md_path, &md)?;
        println!("  Wrote {} ({} bytes)", md_path.display(), md.len());
    }

    Ok(())
}

fn build_usage_example(token_def: &crate::registry::TokenDef, arch: &str) -> String {
    let primary_name = &token_def.name;

    match arch {
        "x86_64" => {
            if token_def.features.iter().any(|f| f.starts_with("avx512")) {
                formatdoc! {r#"
                    use archmage::prelude::*;

                    if let Some(token) = {primary_name}::summon() {{
                        process(token, &mut data);
                    }}

                    #[arcane(import_intrinsics)]  // Entry point only
                    fn process(token: {primary_name}, data: &mut [f32]) {{
                        for chunk in data.chunks_exact_mut(16) {{
                            process_chunk(token, chunk.try_into().unwrap());
                        }}
                    }}

                    #[rite(import_intrinsics)]  // All inner helpers
                    fn process_chunk(_: {primary_name}, chunk: &mut [f32; 16]) {{
                        let v = _mm512_loadu_ps(chunk.as_ptr());  // safe inside #[rite]
                        let doubled = _mm512_add_ps(v, v);
                        _mm512_storeu_ps(chunk.as_mut_ptr(), doubled);
                    }}
                    // Use #![forbid(unsafe_code)] — import_intrinsics provides safe memory ops."#}
            } else if token_def.features.iter().any(|f| f == "avx2") {
                formatdoc! {r#"
                    use archmage::prelude::*;

                    if let Some(token) = {primary_name}::summon() {{
                        process(token, &mut data);
                    }}

                    #[arcane(import_intrinsics)]  // Entry point only
                    fn process(token: {primary_name}, data: &mut [f32]) {{
                        for chunk in data.chunks_exact_mut(8) {{
                            process_chunk(token, chunk.try_into().unwrap());
                        }}
                    }}

                    #[rite(import_intrinsics)]  // All inner helpers
                    fn process_chunk(_: {primary_name}, chunk: &mut [f32; 8]) {{
                        let v = _mm256_loadu_ps(chunk);  // safe!
                        let doubled = _mm256_add_ps(v, v);    // value intrinsic (safe inside #[rite])
                        _mm256_storeu_ps(chunk, doubled);  // safe!
                    }}
                    // No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate."#}
            } else {
                formatdoc! {r#"
                    use archmage::prelude::*;

                    if let Some(token) = {primary_name}::summon() {{
                        process(token, &mut data);
                    }}

                    #[arcane(import_intrinsics)]  // Entry point only
                    fn process(token: {primary_name}, data: &mut [f32]) {{
                        for chunk in data.chunks_exact_mut(4) {{
                            process_chunk(token, chunk.try_into().unwrap());
                        }}
                    }}

                    #[rite(import_intrinsics)]  // All inner helpers
                    fn process_chunk(_: {primary_name}, chunk: &mut [f32; 4]) {{
                        let v = _mm_loadu_ps(chunk);  // safe!
                        let doubled = _mm_add_ps(v, v);  // value intrinsic (safe inside #[rite])
                        _mm_storeu_ps(chunk, doubled);  // safe!
                    }}
                    // No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate."#}
            }
        }
        "aarch64" => formatdoc! {r#"
            use archmage::prelude::*;

            if let Some(token) = {primary_name}::summon() {{
                process(token, &mut data);
            }}

            #[arcane(import_intrinsics)]  // Entry point only
            fn process(token: {primary_name}, data: &mut [f32]) {{
                for chunk in data.chunks_exact_mut(4) {{
                    process_chunk(token, chunk.try_into().unwrap());
                }}
            }}

            #[rite(import_intrinsics)]  // All inner helpers
            fn process_chunk(_: {primary_name}, chunk: &mut [f32; 4]) {{
                let v = vld1q_f32(chunk);  // safe!
                let doubled = vaddq_f32(v, v);  // value intrinsic (safe inside #[rite])
                vst1q_f32(chunk, doubled);  // safe!
            }}
            // No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate."#},
        "wasm32" => formatdoc! {r#"
            use archmage::prelude::*;

            if let Some(token) = Wasm128Token::summon() {{
                process(token, &mut data);
            }}

            #[arcane(import_intrinsics)]  // Entry point only
            fn process(token: Wasm128Token, data: &mut [f32]) {{
                for chunk in data.chunks_exact_mut(4) {{
                    process_chunk(token, chunk.try_into().unwrap());
                }}
            }}

            #[rite(import_intrinsics)]  // All inner helpers
            fn process_chunk(_: Wasm128Token, chunk: &mut [f32; 4]) {{
                let v = v128_load(chunk);  // safe!
                let doubled = f32x4_add(v, v);  // value intrinsic (safe inside #[rite])
                v128_store(chunk, doubled);  // safe!
            }}
            // No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate."#},
        _ => String::from("// See archmage documentation for usage examples."),
    }
}

fn build_safe_ops_table(
    token_intrinsics: &[&CsvIntrinsic],
    safe_variants: &HashMap<String, String>,
) -> String {
    let ops: Vec<(&CsvIntrinsic, &String)> = token_intrinsics
        .iter()
        .filter_map(|i| safe_variants.get(&i.name).map(|sig| (*i, sig)))
        .collect();

    if ops.is_empty() {
        return String::new();
    }

    let mut out = String::from("## Safe Memory Operations (via import_intrinsics)\n\n");
    out.push_str("| Function | Safe Signature |\n");
    out.push_str("|----------|---------------|\n");
    for (intr, sig) in &ops {
        out.push_str(&format!("| `{}` | `{}` |\n", intr.name, sig));
    }
    out
}

fn build_intrinsics_tables(
    token_intrinsics: &[&CsvIntrinsic],
    safe_variants: &HashMap<String, String>,
    timing_table: &BTreeMap<&str, TimingEntry>,
) -> String {
    if token_intrinsics.is_empty() {
        return String::from("*No intrinsics mapped to this token.*\n");
    }

    // Group by stability+safety
    let mut stable_safe: Vec<&CsvIntrinsic> = Vec::new();
    let mut stable_unsafe: Vec<&CsvIntrinsic> = Vec::new();
    let mut unstable: Vec<&CsvIntrinsic> = Vec::new();

    for i in token_intrinsics {
        if i.stability == "stable" {
            if i.is_unsafe {
                stable_unsafe.push(i);
            } else {
                stable_safe.push(i);
            }
        } else {
            unstable.push(i);
        }
    }

    let mut out = String::new();

    if !stable_safe.is_empty() {
        out.push_str(&format!(
            "### Stable, Safe ({} intrinsics)\n\n",
            stable_safe.len()
        ));
        out.push_str("| Name | Description | Instruction | Timing (H/Z4) |\n");
        out.push_str("|------|-------------|-------------|---------------|\n");
        for i in &stable_safe {
            let timing_str = format_timing(i, timing_table);
            let doc = truncate_doc(&i.doc, 60);
            out.push_str(&format!(
                "| `{}` | {} | {} | {} |\n",
                i.name, doc, i.instruction, timing_str
            ));
        }
        out.push('\n');
    }

    if !stable_unsafe.is_empty() {
        out.push_str(&format!(
            "### Stable, Unsafe ({} intrinsics) — use import_intrinsics for safe versions\n\n",
            stable_unsafe.len()
        ));
        out.push_str("| Name | Description | Safe Variant |\n");
        out.push_str("|------|-------------|--------------|\n");
        for i in &stable_unsafe {
            let doc = truncate_doc(&i.doc, 60);
            let sv = if safe_variants.contains_key(&i.name) {
                format!("`{}` (safe via import_intrinsics)", i.name)
            } else {
                "—".to_string()
            };
            out.push_str(&format!("| `{}` | {} | {} |\n", i.name, doc, sv));
        }
        out.push('\n');
    }

    if !unstable.is_empty() {
        out.push_str(&format!(
            "### Unstable/Nightly ({} intrinsics)\n\n",
            unstable.len()
        ));
        out.push_str("| Name | Description | Instruction |\n");
        out.push_str("|------|-------------|-------------|\n");
        for i in &unstable {
            let doc = truncate_doc(&i.doc, 60);
            out.push_str(&format!("| `{}` | {} | {} |\n", i.name, doc, i.instruction));
        }
        out.push('\n');
    }

    out
}

fn truncate_doc(doc: &str, max_len: usize) -> String {
    // Remove markdown links for the table display, but keep the text
    let cleaned = doc
        .split('[')
        .next()
        .unwrap_or(doc)
        .trim()
        .trim_end_matches('.');
    if cleaned.len() > max_len {
        format!("{}...", &cleaned[..max_len])
    } else {
        cleaned.to_string()
    }
}

fn format_timing(intrinsic: &CsvIntrinsic, timing_table: &BTreeMap<&str, TimingEntry>) -> String {
    let cat = match classify_timing(&intrinsic.name) {
        Some(c) => c,
        None => return "—".to_string(),
    };
    let entry = match timing_table.get(cat) {
        Some(e) => e,
        None => return "—".to_string(),
    };
    let h = entry.get("h").copied().unwrap_or([0, 0]);
    let z4 = entry.get("z4").copied().unwrap_or([0, 0]);
    if h[0] == 0 && z4[0] == 0 {
        return "—".to_string();
    }
    if h[0] == 0 {
        format!("—/{}/{}", z4[0], z4[1])
    } else if z4[0] == 0 {
        format!("{}/{}/ —", h[0], h[1])
    } else {
        format!("{}/{}, {}/{}", h[0], h[1], z4[0], z4[1])
    }
}

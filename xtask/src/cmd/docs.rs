//! Intrinsic CSV database loading, `safe_unaligned_simd` extraction, and reference doc generation.

use anyhow::{Context, Result, bail};
use regex::Regex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::intrinsics_browser::parse_csv_line;

/// Version of safe_unaligned_simd we're generating from
const SAFE_SIMD_VERSION: &str = "0.2.5";

// Feature lookups now come from token-registry.toml via Registry::features_for().

/// Categorize an intrinsic by its operation type
fn categorize_intrinsic(name: &str) -> &'static str {
    // x86 categorization
    if name.contains("loadu")
        || name.contains("load_")
        || name.starts_with("_mm") && name.contains("load")
    {
        "Load"
    } else if name.contains("storeu")
        || name.contains("store_")
        || name.starts_with("_mm") && name.contains("store")
    {
        "Store"
    } else if name.contains("gather") {
        "Gather"
    } else if name.contains("scatter") {
        "Scatter"
    } else if name.contains("compress") || name.contains("expand") {
        "Compress/Expand"
    // NEON categorization
    } else if name.starts_with("vld") {
        "Load"
    } else if name.starts_with("vst") {
        "Store"
    } else {
        "Other"
    }
}

// ============================================================================
// Intrinsic Database (CSV-based)
// ============================================================================

/// Entry from the intrinsics CSV database.
pub(crate) struct IntrinsicEntry {
    /// Architecture column from the CSV: `x86`, `x86_64`, `arm_shared`,
    /// `aarch64`, or `wasm32`.
    pub(crate) arch: String,
    pub(crate) features: String,
    #[allow(dead_code)]
    pub(crate) is_unsafe: bool,
}

/// Load the intrinsic database from docs/intrinsics/complete_intrinsics.csv.
///
/// This CSV is extracted from stdarch source by xtask/extract_intrinsics.py
/// and covers ALL intrinsics: x86 shared, x86_64 specific, arm_shared,
/// aarch64 specific, and wasm32 — including pub use re-exports.
///
/// Format: arch,name,features,unsafe,stability,file
pub(crate) fn load_intrinsic_database() -> Result<HashMap<String, IntrinsicEntry>> {
    let mut db = HashMap::new();

    let csv_path = PathBuf::from("docs/intrinsics/complete_intrinsics.csv");
    let content = fs::read_to_string(&csv_path)
        .with_context(|| format!("Failed to read {}. Run: python3 xtask/extract_intrinsics.py > docs/intrinsics/complete_intrinsics.csv", csv_path.display()))?;

    for line in content.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_line(line);
        if fields.len() >= 4 {
            let arch = fields[0].clone();
            let name = fields[1].clone();
            let features = fields[2].clone();
            let is_unsafe = fields[3] == "True";
            db.insert(
                name,
                IntrinsicEntry {
                    arch,
                    features,
                    is_unsafe,
                },
            );
        }
    }

    if db.is_empty() {
        bail!("Intrinsic database is empty — CSV may be corrupt");
    }

    Ok(db)
}

// ============================================================================
// Reference Documentation Generation
// ============================================================================

/// Safe memory operation extracted from safe_unaligned_simd source.
pub(crate) struct SafeMemOp {
    pub(crate) name: String,
    pub(crate) features: String,
    pub(crate) token: String,
    pub(crate) signature: String,
    pub(crate) doc: String,
}

/// Find safe_unaligned_simd source in cargo cache without walkdir.
pub(crate) fn find_safe_simd_path_simple() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME not set")?;
    let registry_src = PathBuf::from(&home).join(".cargo/registry/src");

    // Track every safe_unaligned_simd-* the cache DOES hold, so the failure can
    // tell "nothing fetched" apart from "a different version is present".
    let mut found_other: Vec<String> = Vec::new();

    for entry in fs::read_dir(&registry_src)
        .with_context(|| format!("Failed to read {}", registry_src.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let index_dir = entry.path();
            for crate_entry in fs::read_dir(&index_dir)? {
                let crate_entry = crate_entry?;
                let name = crate_entry.file_name().to_string_lossy().to_string();
                if name.starts_with("safe_unaligned_simd-") {
                    if name.contains(SAFE_SIMD_VERSION) {
                        return Ok(crate_entry.path());
                    }
                    found_other.push(name);
                }
            }
        }
    }

    // Two very different causes produced the same message before, which is why
    // the monthly CI job read as "run cargo fetch" for six months while the
    // workflow had no fetch step to run. Separate them.
    if found_other.is_empty() {
        bail!(
            "No safe_unaligned_simd source in the cargo cache at {}. \
             Nothing has fetched it: xtask's own dependencies do not include it, \
             so building xtask never pulls it. Run `cargo fetch` at the workspace root first.",
            registry_src.display()
        )
    }

    found_other.sort();
    bail!(
        "safe_unaligned_simd is in the cargo cache, but not at the version xtask wants. \
         Wanted {} (hardcoded as SAFE_SIMD_VERSION in xtask/src/cmd/docs.rs); cache holds: {}. \
         The constant has drifted from Cargo.lock — update it to match the locked version. \
         `cargo fetch` will NOT fix this.",
        SAFE_SIMD_VERSION,
        found_other.join(", ")
    )
}

/// Extract safe memory operations from safe_unaligned_simd source using regex.
pub(crate) fn extract_safe_simd_functions(safe_simd_path: &Path) -> Result<Vec<SafeMemOp>> {
    let mut ops = Vec::new();
    let tf_re =
        Regex::new(r#"#\[target_feature\(enable\s*=\s*"([^"]+)"\)\]"#).expect("invalid tf regex");
    let fn_re = Regex::new(r"pub\s+fn\s+(\w+)(<[^>]+>)?\(([^)]*)\)(?:\s*->\s*(.+?))?\s*\{")
        .expect("invalid fn regex");
    let doc_re = Regex::new(r"///\s*(.*)").expect("invalid doc regex");

    // Walk x86 directory (sorted for deterministic output)
    let x86_dir = safe_simd_path.join("src/x86");
    if x86_dir.exists() {
        let mut paths: Vec<_> = fs::read_dir(&x86_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "rs"))
            .filter(|p| {
                let name = p.file_name().unwrap().to_string_lossy();
                name != "mod.rs" && name != "cell.rs" && !name.starts_with("cell")
            })
            .collect();
        paths.sort();
        for path in paths {
            extract_ops_from_file(&path, "x86_64", &tf_re, &fn_re, &doc_re, &mut ops)?;
        }
    }

    // aarch64
    let aarch64_path = safe_simd_path.join("src/aarch64.rs");
    if aarch64_path.exists() {
        extract_ops_from_file(&aarch64_path, "aarch64", &tf_re, &fn_re, &doc_re, &mut ops)?;
    }

    // wasm32
    let wasm_path = safe_simd_path.join("src/wasm32.rs");
    if wasm_path.exists() {
        extract_ops_from_file(&wasm_path, "wasm32", &tf_re, &fn_re, &doc_re, &mut ops)?;
    }

    Ok(ops)
}

/// Extract safe memory ops from a single source file.
fn extract_ops_from_file(
    path: &Path,
    arch: &str,
    tf_re: &Regex,
    fn_re: &Regex,
    doc_re: &Regex,
    ops: &mut Vec<SafeMemOp>,
) -> Result<()> {
    let content = fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        // Look for target_feature
        if let Some(tf_cap) = tf_re.captures(lines[i]) {
            let features = tf_cap[1].to_string();

            // Collect doc comment lines before the target_feature
            let mut doc_lines = Vec::new();
            let mut k = i.saturating_sub(1);
            while k > 0 {
                if let Some(doc_cap) = doc_re.captures(lines[k]) {
                    doc_lines.push(doc_cap[1].to_string());
                    k -= 1;
                } else {
                    break;
                }
            }
            doc_lines.reverse();
            let doc = doc_lines.first().cloned().unwrap_or_default();

            // Look forward for pub fn
            let mut j = i + 1;
            while j < lines.len() && j < i + 5 {
                if let Some(fn_cap) = fn_re.captures(lines[j]) {
                    let name = fn_cap[1].to_string();
                    let generics = fn_cap.get(2).map(|m| m.as_str()).unwrap_or("");
                    let params = fn_cap[3].to_string();
                    let ret = fn_cap.get(4).map(|m| m.as_str()).unwrap_or("()");
                    let signature = format!("fn {}{}({}) -> {}", name, generics, params, ret);

                    let token = map_features_to_token(&features, arch);

                    ops.push(SafeMemOp {
                        name,
                        features,
                        token,
                        signature,
                        doc,
                    });
                    i = j + 1;
                    break;
                }
                j += 1;
            }
            if j >= i + 5 || j >= lines.len() {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    Ok(())
}

/// Map feature string to archmage token name for documentation.
fn map_features_to_token(features: &str, arch: &str) -> String {
    match arch {
        "x86_64" | "x86" => match features {
            "sse" | "sse2" => "Baseline (always available)".to_string(),
            "avx" => "X64V3Token".to_string(),
            "avx512f" => "X64V4Token".to_string(),
            "avx512f,avx512vl" => "X64V4Token".to_string(),
            "avx512bw" => "X64V4Token".to_string(),
            "avx512bw,avx512vl" => "X64V4Token".to_string(),
            "avx512vbmi2" => "X64V4xToken".to_string(),
            "avx512vbmi2,avx512vl" => "X64V4xToken".to_string(),
            _ => format!("({})", features),
        },
        "aarch64" => "NeonToken / Arm64".to_string(),
        "wasm32" => "Wasm128Token".to_string(),
        _ => "Unknown".to_string(),
    }
}

/// Generate x86 intrinsics-by-token reference doc from CSV database.
fn generate_x86_reference(db: &HashMap<String, IntrinsicEntry>) -> String {
    let mut doc = String::new();

    doc.push_str(
        "# x86 Intrinsics by Archmage Token\n\n\
        Auto-generated reference mapping stdarch intrinsics to archmage tokens.\n\
        Based on Rust 1.96 stdarch. Regenerate: `cargo xtask generate`\n\n",
    );

    // Collect x86 intrinsics and organize by token tier
    struct TierInfo {
        name: &'static str,
        token: &'static str,
        features: &'static [&'static str],
        intrinsics: BTreeMap<String, Vec<String>>, // category -> names
    }

    let tiers: Vec<TierInfo> = vec![
        TierInfo {
            name: "Baseline (SSE, SSE2)",
            token: "None (always available on x86-64)",
            features: &["sse", "sse2"],
            intrinsics: BTreeMap::new(),
        },
        TierInfo {
            name: "X64V2Token (SSE3 → SSE4.2)",
            token: "X64V2Token",
            features: &["sse3", "ssse3", "sse4.1", "sse4.2", "popcnt"],
            intrinsics: BTreeMap::new(),
        },
        TierInfo {
            name: "X64V3Token (AVX2, FMA)",
            token: "X64V3Token",
            features: &["avx", "avx2", "fma", "bmi1", "bmi2", "f16c", "lzcnt"],
            intrinsics: BTreeMap::new(),
        },
        TierInfo {
            name: "X64V3VnniToken (AVX-VNNI + ADX/SHA)",
            token: "X64V3VnniToken",
            features: &["avxvnni", "adx", "sha"],
            intrinsics: BTreeMap::new(),
        },
        TierInfo {
            name: "X64V4Token (AVX-512)",
            token: "X64V4Token / Server64",
            features: &["avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"],
            intrinsics: BTreeMap::new(),
        },
        TierInfo {
            name: "X64V4xToken (Modern Extensions)",
            token: "X64V4xToken",
            features: &[
                "avx512vpopcntdq",
                "avx512ifma",
                "avx512vbmi",
                "avx512vbmi2",
                "avx512bitalg",
                "avx512vnni",
                "avx512bf16",
                "vpclmulqdq",
                "gfni",
                "vaes",
            ],
            intrinsics: BTreeMap::new(),
        },
        TierInfo {
            name: "Avx512Fp16Token",
            token: "Avx512Fp16Token",
            features: &["avx512fp16"],
            intrinsics: BTreeMap::new(),
        },
    ];

    let mut tiers = tiers;

    // Assign each x86 intrinsic to its tier
    for (name, entry) in db {
        if !name.starts_with("_mm") {
            continue; // Only x86 SIMD intrinsics
        }

        let required: HashSet<&str> = entry.features.split(',').map(|s| s.trim()).collect();

        // Find the tier that contains the required features
        let mut assigned = false;
        for tier in tiers.iter_mut() {
            if required.iter().all(|f| tier.features.contains(f)) {
                let category = categorize_intrinsic(name);
                tier.intrinsics
                    .entry(category.to_string())
                    .or_default()
                    .push(name.clone());
                assigned = true;
                break;
            }
        }
        if !assigned {
            // Check if it's a combo feature (e.g., avx512f,avx512vl)
            // Assign to the highest tier that has ALL features
            for tier in tiers.iter_mut().rev() {
                let tier_features: HashSet<&str> = tier.features.iter().copied().collect();
                if required.iter().all(|f| {
                    tier_features.contains(f)
                        || [
                            "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx",
                            "avx2", "fma", "bmi1", "bmi2", "f16c", "lzcnt", "avx512f", "avx512bw",
                            "avx512cd", "avx512dq", "avx512vl",
                        ]
                        .contains(f)
                }) {
                    let category = categorize_intrinsic(name);
                    tier.intrinsics
                        .entry(category.to_string())
                        .or_default()
                        .push(name.clone());
                    break;
                }
            }
        }
    }

    // Output
    for tier in &tiers {
        let count: usize = tier.intrinsics.values().map(|v| v.len()).sum();
        if count == 0 {
            continue;
        }

        doc.push_str(&format!("## {} ({} intrinsics)\n", tier.name, count));
        doc.push_str(&format!("Token: `{}`\n\n", tier.token));

        for (category, names) in &tier.intrinsics {
            let mut sorted = names.clone();
            sorted.sort();
            doc.push_str(&format!("### {}\n\n", category));
            for name in &sorted {
                let safe = db.get(name.as_str()).map(|e| !e.is_unsafe).unwrap_or(false);
                let marker = if safe { "" } else { " (unsafe)" };
                doc.push_str(&format!("- `{}`{}\n", name, marker));
            }
            doc.push('\n');
        }
    }

    doc
}

/// Generate aarch64 intrinsics-by-token reference doc from CSV database.
fn generate_aarch64_reference(db: &HashMap<String, IntrinsicEntry>) -> String {
    let mut doc = String::new();

    doc.push_str(
        "# AArch64 Intrinsics by Archmage Token\n\n\
        Auto-generated reference mapping stdarch intrinsics to archmage tokens.\n\
        Based on Rust 1.96 stdarch. Regenerate: `cargo xtask generate`\n\n",
    );

    // Group by feature set
    let mut by_token: BTreeMap<String, BTreeMap<String, Vec<String>>> = BTreeMap::new();

    for (name, entry) in db {
        // Filter to aarch64 NEON intrinsics (start with 'v')
        if !name.starts_with('v') {
            continue;
        }
        // Skip non-neon
        if !entry.features.contains("neon") {
            continue;
        }

        let token = if entry.features.contains("sha3") {
            "NeonSha3Token"
        } else if entry.features.contains("aes") {
            "NeonAesToken"
        } else {
            "NeonToken / Arm64"
        };

        let category = categorize_intrinsic(name);
        by_token
            .entry(token.to_string())
            .or_default()
            .entry(category.to_string())
            .or_default()
            .push(name.clone());
    }

    for (token, categories) in &by_token {
        let count: usize = categories.values().map(|v| v.len()).sum();
        doc.push_str(&format!("## {} ({} intrinsics)\n\n", token, count));

        for (category, names) in categories {
            let mut sorted = names.clone();
            sorted.sort();
            doc.push_str(&format!("### {}\n\n", category));
            for name in &sorted {
                let safe = db.get(name.as_str()).map(|e| !e.is_unsafe).unwrap_or(false);
                let marker = if safe { "" } else { " (unsafe)" };
                doc.push_str(&format!("- `{}`{}\n", name, marker));
            }
            doc.push('\n');
        }
    }

    doc
}

/// Generate memory operations reference doc.
fn generate_memory_ops_reference(safe_simd_ops: &[SafeMemOp]) -> String {
    let mut doc = String::new();

    doc.push_str(&format!(
        "# Safe Memory Operations Reference\n\n\
         Safe unaligned load/store operations from `safe_unaligned_simd` v{SAFE_SIMD_VERSION},\n\
         organized by the archmage token required to use them inside `#[attune(..., import_intrinsics)]` functions.\n\n\
         Regenerate: `cargo xtask generate`\n\n"
    ));

    // Group by token
    let mut by_token: BTreeMap<&str, Vec<&SafeMemOp>> = BTreeMap::new();
    for op in safe_simd_ops {
        by_token.entry(&op.token).or_default().push(op);
    }

    for (token, ops) in &by_token {
        doc.push_str(&format!("## {} ({} functions)\n\n", token, ops.len()));

        // Sort ops by name for deterministic output across filesystems
        let mut sorted_ops = ops.clone();
        sorted_ops.sort_by_key(|op| &op.name);
        for op in sorted_ops {
            doc.push_str(&format!("### `{}`\n\n", op.name));
            if !op.doc.is_empty() {
                doc.push_str(&format!("{}\n\n", op.doc));
            }
            doc.push_str(&format!("```rust\n{}\n```\n\n", op.signature));
            doc.push_str(&format!("Features: `{}`\n\n", op.features));
        }
    }

    doc
}

/// Generate all reference documentation files.
pub(crate) fn generate_reference_docs_new() -> Result<()> {
    let db = load_intrinsic_database()?;
    println!("Loaded {} intrinsics for reference docs", db.len());

    // Create docs/generated directory
    let docs_gen_dir = PathBuf::from("docs/generated");
    fs::create_dir_all(&docs_gen_dir)?;

    // Generate x86 reference
    let x86_ref = generate_x86_reference(&db);
    let x86_path = docs_gen_dir.join("x86-intrinsics-by-token.md");
    fs::write(&x86_path, &x86_ref)?;
    println!("  Wrote {} ({} bytes)", x86_path.display(), x86_ref.len());

    // Generate aarch64 reference
    let aarch64_ref = generate_aarch64_reference(&db);
    let aarch64_path = docs_gen_dir.join("aarch64-intrinsics-by-token.md");
    fs::write(&aarch64_path, &aarch64_ref)?;
    println!(
        "  Wrote {} ({} bytes)",
        aarch64_path.display(),
        aarch64_ref.len()
    );

    // Generate memory ops reference
    let safe_simd_path = match find_safe_simd_path_simple() {
        Ok(p) => p,
        Err(e) => {
            println!("  Skipping memory-ops-reference.md: {}", e);
            return Ok(());
        }
    };
    let safe_ops = extract_safe_simd_functions(&safe_simd_path)?;
    println!("  Extracted {} safe memory operations", safe_ops.len());

    let mem_ref = generate_memory_ops_reference(&safe_ops);
    let mem_path = docs_gen_dir.join("memory-ops-reference.md");
    fs::write(&mem_path, &mem_ref)?;
    println!("  Wrote {} ({} bytes)", mem_path.display(), mem_ref.len());

    // Copy generated docs into simd_reference book (if it exists)
    let simd_ref_intrinsics = PathBuf::from("docs/simd_reference/src/intrinsics");
    if simd_ref_intrinsics.exists() {
        let copies = [
            ("x86-intrinsics-by-token.md", "x86.md"),
            ("aarch64-intrinsics-by-token.md", "aarch64.md"),
            ("memory-ops-reference.md", "memory-ops.md"),
        ];
        for (src, dst) in &copies {
            let src_path = docs_gen_dir.join(src);
            let dst_path = simd_ref_intrinsics.join(dst);
            fs::copy(&src_path, &dst_path)?;
            println!("  Copied {} → {}", src_path.display(), dst_path.display());
        }
    }

    Ok(())
}

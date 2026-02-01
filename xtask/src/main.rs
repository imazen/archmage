//! Archmage xtask — code generation, validation, and reference docs.
//!
//! Usage:
//!   cargo xtask generate   - Generate SIMD types and reference docs
//!   cargo xtask validate   - Validate magetypes safety

use anyhow::{Context, Result, bail};
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::{Path, PathBuf};

mod registry;
mod simd_types;

/// Version of safe_unaligned_simd we're generating from
const SAFE_SIMD_VERSION: &str = "0.2.3";

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
struct IntrinsicEntry {
    features: String,
    #[allow(dead_code)]
    is_unsafe: bool,
}

/// Load the intrinsic database from docs/intrinsics/complete_intrinsics.csv.
///
/// This CSV is extracted from stdarch source by xtask/extract_intrinsics.py
/// and covers ALL intrinsics: x86 shared, x86_64 specific, arm_shared,
/// aarch64 specific, and wasm32 — including pub use re-exports.
///
/// Format: arch,name,features,unsafe,stability,file
fn load_intrinsic_database() -> Result<HashMap<String, IntrinsicEntry>> {
    let mut db = HashMap::new();

    let csv_path = PathBuf::from("docs/intrinsics/complete_intrinsics.csv");
    let content = fs::read_to_string(&csv_path)
        .with_context(|| format!("Failed to read {}. Run: python3 xtask/extract_intrinsics.py > docs/intrinsics/complete_intrinsics.csv", csv_path.display()))?;

    for line in content.lines().skip(1) {
        let fields: Vec<&str> = line.splitn(6, ',').collect();
        if fields.len() >= 4 {
            let name = fields[1].to_string();
            let features = fields[2].to_string();
            let is_unsafe = fields[3] == "True";
            db.insert(
                name,
                IntrinsicEntry {
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
// Magetypes Validation
// ============================================================================

// File-to-token mappings now come from token-registry.toml [[magetypes_file]] entries.

/// Validate that all intrinsic calls in magetypes generated code are safe
/// under their gating token.
///
/// Every intrinsic found in generated code MUST exist in complete_intrinsics.csv.
/// If an intrinsic is missing from the CSV, that's an error — the CSV must be
/// regenerated with `python3 xtask/extract_intrinsics.py`.
fn validate_magetypes() -> Result<()> {
    let reg = registry::Registry::load(&PathBuf::from("token-registry.toml"))?;
    validate_magetypes_with_registry(&reg)
}

fn validate_magetypes_with_registry(reg: &registry::Registry) -> Result<()> {
    println!("=== Magetypes Safety Validation ===\n");

    let db = load_intrinsic_database()?;
    println!("Loaded {} intrinsics from CSV database", db.len());

    // Compile regex patterns for intrinsic extraction
    // x86: _mm_*, _mm256_*, _mm512_* followed by '(' or '::<' (function call or turbofish)
    let x86_re = Regex::new(r"\b(_mm(?:256|512)?_\w+)\s*(?:\(|::<)").expect("invalid x86 regex");
    // ARM NEON: v*q?_* patterns (NEON intrinsics all start with 'v')
    let arm_re = Regex::new(
        r"\b(v(?:abs|add|and|bic|bsl|ceq|cge|cgt|cle|clt|cnt|cvt|div|dup|eor|ext|fma|fms|get|hadd|ld[1234]|max|min|ml[as]|mov|mul|mvn|neg|not|orn|orr|padd|pmax|pmin|qadd|qsub|reinterpret|rev|rnd|rsqrte|set|shl|shr|sqrt|st[1234]|sub|tbl|tbx|trn|uzp|zip)\w*)\s*\("
    ).expect("invalid arm regex");
    // WASM SIMD128: f32x4_*, i32x4_*, v128_*, u8x16_*, etc.
    let wasm_re = Regex::new(
        r"\b((?:f32x4|f64x2|i8x16|i16x8|i32x4|i64x2|u8x16|u16x8|u32x4|u64x2|v128)_\w+)\s*[\(<]",
    )
    .expect("invalid wasm regex");

    let simd_dir = PathBuf::from("magetypes/src/simd");
    let mut errors: Vec<String> = Vec::new();
    let mut validated = 0usize;

    for mapping in &reg.magetypes_file {
        let file_path = simd_dir.join(&mapping.rel_path);
        if !file_path.exists() {
            errors.push(format!("File not found: {}", file_path.display()));
            continue;
        }

        let content = fs::read_to_string(&file_path)
            .with_context(|| format!("Failed to read {}", file_path.display()))?;

        let token = &mapping.token;
        let features = reg.features_for(token).unwrap_or_default();
        let provided_features: HashSet<&str> = features.into_iter().collect();

        // Select regex based on architecture
        let re = match mapping.arch.as_str() {
            "x86" => &x86_re,
            "arm" => &arm_re,
            "wasm" => &wasm_re,
            _ => continue,
        };

        // Extract all unique intrinsic calls
        let mut intrinsics_found: BTreeSet<String> = BTreeSet::new();
        for cap in re.captures_iter(&content) {
            intrinsics_found.insert(cap[1].to_string());
        }

        println!(
            "{}: {} unique intrinsics, token: {}",
            mapping.rel_path,
            intrinsics_found.len(),
            token
        );

        for intrinsic in &intrinsics_found {
            // 1. Width-prefix validation (x86 only — catches wrong-width intrinsics)
            if mapping.arch == "x86" {
                let intrinsic_width = if intrinsic.starts_with("_mm512_") {
                    3u8
                } else if intrinsic.starts_with("_mm256_") {
                    2
                } else if intrinsic.starts_with("_mm_") {
                    1
                } else {
                    0
                };

                let file_width = if mapping.rel_path.contains("w512") {
                    3u8
                } else if mapping.rel_path.contains("w256") {
                    2
                } else if mapping.rel_path.contains("w128") {
                    1
                } else {
                    0
                };

                if intrinsic_width > 0 && file_width > 0 && intrinsic_width > file_width {
                    errors.push(format!(
                        "WIDTH MISMATCH: {} in {} — intrinsic is {}bit but file only provides {}bit",
                        intrinsic,
                        mapping.rel_path,
                        match intrinsic_width {
                            1 => "128",
                            2 => "256",
                            3 => "512",
                            _ => "?",
                        },
                        match file_width {
                            1 => "128",
                            2 => "256",
                            3 => "512",
                            _ => "?",
                        }
                    ));
                    continue;
                }
            }

            // 2. CSV database lookup — REQUIRED for every intrinsic
            match db.get(intrinsic.as_str()) {
                Some(entry) => {
                    let required: HashSet<&str> =
                        entry.features.split(',').map(|s| s.trim()).collect();
                    let missing: Vec<&str> = required
                        .iter()
                        .copied()
                        .filter(|f| !provided_features.contains(*f))
                        .collect();

                    if missing.is_empty() {
                        validated += 1;
                    } else {
                        errors.push(format!(
                            "FEATURE MISMATCH: {} in {} requires [{}] but {} only provides [{}] — missing: {:?}",
                            intrinsic,
                            mapping.rel_path,
                            entry.features,
                            token,
                            provided_features.iter().copied().collect::<BTreeSet<_>>().iter().copied().collect::<Vec<_>>().join(", "),
                            missing
                        ));
                    }
                }
                None => {
                    errors.push(format!(
                        "UNKNOWN INTRINSIC: {} in {} — not found in complete_intrinsics.csv. \
                         Regenerate CSV: python3 xtask/extract_intrinsics.py > docs/intrinsics/complete_intrinsics.csv",
                        intrinsic, mapping.rel_path
                    ));
                }
            }
        }
    }

    // Summary
    println!("\n=== Validation Summary ===");
    println!(
        "  Validated: {} intrinsic calls (all CSV-verified)",
        validated
    );
    println!("  Errors:    {}", errors.len());

    if !errors.is_empty() {
        println!("\n=== ERRORS ===");
        for e in &errors {
            println!("  {}", e);
        }
    }

    if errors.is_empty() {
        println!("\nAll validations passed!");
        Ok(())
    } else {
        bail!("Validation failed with {} errors", errors.len())
    }
}

// ============================================================================
// Reference Documentation Generation
// ============================================================================

/// Safe memory operation extracted from safe_unaligned_simd source.
struct SafeMemOp {
    name: String,
    features: String,
    token: String,
    signature: String,
    doc: String,
}

/// Find safe_unaligned_simd source in cargo cache without walkdir.
fn find_safe_simd_path_simple() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME not set")?;
    let registry_src = PathBuf::from(&home).join(".cargo/registry/src");

    for entry in fs::read_dir(&registry_src)
        .with_context(|| format!("Failed to read {}", registry_src.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let index_dir = entry.path();
            for crate_entry in fs::read_dir(&index_dir)? {
                let crate_entry = crate_entry?;
                let name = crate_entry.file_name().to_string_lossy().to_string();
                if name.starts_with("safe_unaligned_simd-") && name.contains(SAFE_SIMD_VERSION) {
                    return Ok(crate_entry.path());
                }
            }
        }
    }

    bail!(
        "Could not find safe_unaligned_simd-{} in cargo cache. Run: cargo fetch",
        SAFE_SIMD_VERSION
    )
}

/// Extract safe memory operations from safe_unaligned_simd source using regex.
fn extract_safe_simd_functions(safe_simd_path: &Path) -> Result<Vec<SafeMemOp>> {
    let mut ops = Vec::new();
    let tf_re =
        Regex::new(r#"#\[target_feature\(enable\s*=\s*"([^"]+)"\)\]"#).expect("invalid tf regex");
    let fn_re = Regex::new(r"pub\s+fn\s+(\w+)(<[^>]+>)?\(([^)]*)\)(?:\s*->\s*(.+?))?\s*\{")
        .expect("invalid fn regex");
    let doc_re = Regex::new(r"///\s*(.*)").expect("invalid doc regex");

    // Walk x86 directory
    let x86_dir = safe_simd_path.join("src/x86");
    if x86_dir.exists() {
        for entry in fs::read_dir(&x86_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "rs") {
                let name = path.file_name().unwrap().to_string_lossy().to_string();
                // Skip cell variants and mod.rs
                if name == "mod.rs" || name == "cell.rs" || name.starts_with("cell") {
                    continue;
                }
                extract_ops_from_file(&path, "x86_64", &tf_re, &fn_re, &doc_re, &mut ops)?;
            }
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
            "avx" => "Has256BitSimd / Avx2FmaToken".to_string(),
            "avx512f" => "X64V4Token".to_string(),
            "avx512f,avx512vl" => "X64V4Token".to_string(),
            "avx512bw" => "X64V4Token".to_string(),
            "avx512bw,avx512vl" => "X64V4Token".to_string(),
            "avx512vbmi2" => "Avx512ModernToken".to_string(),
            "avx512vbmi2,avx512vl" => "Avx512ModernToken".to_string(),
            _ => format!("({})", features),
        },
        "aarch64" => "NeonToken / Arm64".to_string(),
        "wasm32" => "Simd128Token".to_string(),
        _ => "Unknown".to_string(),
    }
}

/// Generate x86 intrinsics-by-token reference doc from CSV database.
fn generate_x86_reference(db: &HashMap<String, IntrinsicEntry>) -> String {
    let mut doc = String::new();

    writeln!(doc, "# x86 Intrinsics by Archmage Token").unwrap();
    writeln!(doc).unwrap();
    writeln!(
        doc,
        "Auto-generated reference mapping stdarch intrinsics to archmage tokens."
    )
    .unwrap();
    writeln!(
        doc,
        "Based on Rust 1.92 stdarch. Regenerate: `cargo xtask generate`"
    )
    .unwrap();
    writeln!(doc).unwrap();

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
            name: "Desktop64 / Avx2FmaToken (AVX, AVX2, FMA)",
            token: "Desktop64 / Avx2FmaToken / X64V3Token",
            features: &["avx", "avx2", "fma", "bmi1", "bmi2", "f16c", "lzcnt"],
            intrinsics: BTreeMap::new(),
        },
        TierInfo {
            name: "X64V4Token (AVX-512)",
            token: "X64V4Token / Server64",
            features: &["avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"],
            intrinsics: BTreeMap::new(),
        },
        TierInfo {
            name: "Avx512ModernToken (Modern Extensions)",
            token: "Avx512ModernToken",
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

        writeln!(doc, "## {} ({} intrinsics)", tier.name, count).unwrap();
        writeln!(doc, "Token: `{}`\n", tier.token).unwrap();

        for (category, names) in &tier.intrinsics {
            let mut sorted = names.clone();
            sorted.sort();
            writeln!(doc, "### {}\n", category).unwrap();
            for name in &sorted {
                let safe = db.get(name.as_str()).map(|e| !e.is_unsafe).unwrap_or(false);
                let marker = if safe { "" } else { " (unsafe)" };
                writeln!(doc, "- `{}`{}", name, marker).unwrap();
            }
            writeln!(doc).unwrap();
        }
    }

    doc
}

/// Generate aarch64 intrinsics-by-token reference doc from CSV database.
fn generate_aarch64_reference(db: &HashMap<String, IntrinsicEntry>) -> String {
    let mut doc = String::new();

    writeln!(doc, "# AArch64 Intrinsics by Archmage Token").unwrap();
    writeln!(doc).unwrap();
    writeln!(
        doc,
        "Auto-generated reference mapping stdarch intrinsics to archmage tokens."
    )
    .unwrap();
    writeln!(
        doc,
        "Based on Rust 1.92 stdarch. Regenerate: `cargo xtask generate`"
    )
    .unwrap();
    writeln!(doc).unwrap();

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
        writeln!(doc, "## {} ({} intrinsics)\n", token, count).unwrap();

        for (category, names) in categories {
            let mut sorted = names.clone();
            sorted.sort();
            writeln!(doc, "### {}\n", category).unwrap();
            for name in &sorted {
                let safe = db.get(name.as_str()).map(|e| !e.is_unsafe).unwrap_or(false);
                let marker = if safe { "" } else { " (unsafe)" };
                writeln!(doc, "- `{}`{}", name, marker).unwrap();
            }
            writeln!(doc).unwrap();
        }
    }

    doc
}

/// Generate memory operations reference doc.
fn generate_memory_ops_reference(safe_simd_ops: &[SafeMemOp]) -> String {
    let mut doc = String::new();

    writeln!(doc, "# Safe Memory Operations Reference").unwrap();
    writeln!(doc).unwrap();
    writeln!(
        doc,
        "Safe unaligned load/store operations from `safe_unaligned_simd` v{},",
        SAFE_SIMD_VERSION
    )
    .unwrap();
    writeln!(
        doc,
        "organized by the archmage token required to use them inside `#[arcane]` functions."
    )
    .unwrap();
    writeln!(doc).unwrap();
    writeln!(doc, "Regenerate: `cargo xtask generate`").unwrap();
    writeln!(doc).unwrap();

    // Group by token
    let mut by_token: BTreeMap<&str, Vec<&SafeMemOp>> = BTreeMap::new();
    for op in safe_simd_ops {
        by_token.entry(&op.token).or_default().push(op);
    }

    for (token, ops) in &by_token {
        writeln!(doc, "## {} ({} functions)\n", token, ops.len()).unwrap();

        for op in ops {
            writeln!(doc, "### `{}`\n", op.name).unwrap();
            if !op.doc.is_empty() {
                writeln!(doc, "{}\n", op.doc).unwrap();
            }
            writeln!(doc, "```rust").unwrap();
            writeln!(doc, "{}", op.signature).unwrap();
            writeln!(doc, "```\n").unwrap();
            writeln!(doc, "Features: `{}`\n", op.features).unwrap();
        }
    }

    doc
}

/// Generate all reference documentation files.
fn generate_reference_docs_new() -> Result<()> {
    let db = load_intrinsic_database()?;
    println!("Loaded {} intrinsics for reference docs", db.len());

    // Generate x86 reference
    let x86_ref = generate_x86_reference(&db);
    let x86_path = PathBuf::from("docs/x86-intrinsics-by-token.md");
    fs::write(&x86_path, &x86_ref)?;
    println!("  Wrote {} ({} bytes)", x86_path.display(), x86_ref.len());

    // Generate aarch64 reference
    let aarch64_ref = generate_aarch64_reference(&db);
    let aarch64_path = PathBuf::from("docs/aarch64-intrinsics-by-token.md");
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
    let mem_path = PathBuf::from("docs/memory-ops-reference.md");
    fs::write(&mem_path, &mem_ref)?;
    println!("  Wrote {} ({} bytes)", mem_path.display(), mem_ref.len());

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "Usage: cargo xtask generate            - Generate SIMD types and reference docs"
        );
        eprintln!("       cargo xtask validate            - Validate magetypes safety");
        eprintln!(
            "       cargo xtask validate-registry   - Parse and validate token-registry.toml"
        );
        std::process::exit(1);
    }

    match args[1].as_str() {
        "generate" => generate_all()?,
        "validate" => validate_magetypes()?,
        "validate-registry" => validate_registry()?,
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Parse and validate token-registry.toml
fn validate_registry() -> Result<()> {
    let registry_path = PathBuf::from("token-registry.toml");
    println!("=== Validating Token Registry ===");
    println!("Loading: {}", registry_path.display());

    let reg = registry::Registry::load(&registry_path)?;
    println!("{}", reg);
    println!("All validation checks passed.");
    Ok(())
}

/// Generate all artifacts: SIMD types, macro registry, and documentation
fn generate_all() -> Result<()> {
    // Generate macro registry from token-registry.toml
    println!("=== Generating Macro Registry ===");
    let registry_path = PathBuf::from("token-registry.toml");
    let reg = registry::Registry::load(&registry_path)?;
    let generated = reg.generate_macro_registry();
    let gen_path = PathBuf::from("archmage-macros/src/generated_registry.rs");
    fs::write(&gen_path, &generated)?;
    println!("  Wrote {} ({} bytes)", gen_path.display(), generated.len());

    // Run rustfmt on the generated file so it stays fmt-clean
    let fmt_status = std::process::Command::new("rustfmt")
        .arg(&gen_path)
        .status();
    match fmt_status {
        Ok(s) if s.success() => println!("  Formatted {}", gen_path.display()),
        Ok(s) => println!("  Warning: rustfmt exited with {}", s),
        Err(e) => println!("  Warning: could not run rustfmt: {}", e),
    }

    // Generate SIMD types (wide-like ergonomic types) into magetypes crate
    println!("\n=== Generating SIMD Types (magetypes) ===");

    let simd_dir = PathBuf::from("magetypes/src/simd");
    fs::create_dir_all(&simd_dir)?;
    fs::create_dir_all(simd_dir.join("x86"))?;

    // Generate split files
    let simd_files = simd_types::generate_simd_types_split();
    let mut total_bytes = 0;
    for (rel_path, content) in &simd_files {
        let full_path = simd_dir.join(rel_path);
        fs::write(&full_path, content)?;
        total_bytes += content.len();
        println!("  Wrote {} ({} bytes)", full_path.display(), content.len());
    }
    println!(
        "Total SIMD types: {} files, {} bytes",
        simd_files.len(),
        total_bytes
    );

    // Generate SIMD type tests
    let simd_tests = simd_types::generate_simd_tests();
    let simd_test_path = PathBuf::from("magetypes/tests/generated_simd_types.rs");
    fs::create_dir_all("magetypes/tests")?;
    fs::write(&simd_test_path, &simd_tests)?;
    println!(
        "Wrote {} ({} bytes)",
        simd_test_path.display(),
        simd_tests.len()
    );

    // Generate reference documentation
    println!("\n=== Generating Reference Documentation ===");
    generate_reference_docs_new()?;

    // Run safety validation on generated code (hard failure)
    println!("\n=== Validating Magetypes Safety ===");
    validate_magetypes_with_registry(&reg)?;

    println!("\n=== Generation Complete ===");
    println!("  - SIMD types: magetypes/src/simd/");
    println!("  - Reference docs: docs/*.md");

    Ok(())
}

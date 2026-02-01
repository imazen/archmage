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
// try_new() / summon() Feature Verification
// ============================================================================

/// Validate that every token's try_new() checks exactly the features in the registry.
///
/// Parses src/tokens/*.rs to extract `is_x86_feature_available!("feat")` and
/// `is_aarch64_feature_available!("feat")` calls within each token's try_new() block,
/// then compares against the registry.
///
/// Special cases:
/// - NeonToken: always returns Some (neon is baseline AArch64), skip check
/// - Simd128Token: uses #[cfg(target_feature)] not runtime detection, skip check
/// - x86 tokens: sse/sse2 are baseline, not checked at runtime
/// - AArch64 tokens: neon is baseline, not checked at runtime
fn validate_try_new(reg: &registry::Registry) -> Result<()> {
    println!("=== try_new() Feature Verification ===\n");

    let token_files = [
        ("src/tokens/x86.rs", "x86"),
        ("src/tokens/x86_avx512.rs", "x86"),
        ("src/tokens/arm.rs", "aarch64"),
        ("src/tokens/wasm.rs", "wasm"),
    ];

    // Regex to extract is_x86_feature_available!("feature") calls
    let x86_feature_re =
        Regex::new(r#"is_x86_feature_available!\("([^"]+)"\)"#).expect("invalid regex");
    let arm_feature_re =
        Regex::new(r#"is_aarch64_feature_available!\("([^"]+)"\)"#).expect("invalid regex");

    let mut errors: Vec<String> = Vec::new();
    let mut verified = 0usize;

    for (file_path, arch) in &token_files {
        let path = PathBuf::from(file_path);
        if !path.exists() {
            continue;
        }
        let content =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;

        // Find each "impl SimdToken for <TokenName>" block and extract its try_new() features
        let impl_re = Regex::new(r"impl\s+SimdToken\s+for\s+(\w+)").expect("invalid impl regex");

        for cap in impl_re.captures_iter(&content) {
            let token_name = cap[1].to_string();

            // Look up registry features for this token
            let token_def = match reg.find_token(&token_name) {
                Some(t) => t,
                None => {
                    // Might be an alias type — skip if not a primary token
                    continue;
                }
            };

            // Handle special cases before parsing
            if *arch == "wasm" {
                // Simd128Token uses #[cfg(target_feature)], not runtime detection
                println!(
                    "  {} — compile-time cfg check (no runtime detection)",
                    token_name
                );
                verified += 1;
                continue;
            }

            if token_name == "NeonToken" && token_def.always_available {
                println!("  {} — always available (no runtime checks)", token_name);
                verified += 1;
                continue;
            }

            // Find the try_new() body for this impl block
            let impl_start = cap.get(0).unwrap().start();
            let remaining = &content[impl_start..];

            // Find fn try_new()
            let try_new_pos = match remaining.find("fn try_new()") {
                Some(pos) => pos,
                None => continue,
            };

            // Extract features from the try_new body (up to the next fn or closing brace)
            let try_new_body = &remaining[try_new_pos..];
            // Find end: next "fn " or impl block end
            let body_end = try_new_body
                .find("\n    fn ")
                .or_else(|| try_new_body.find("\n    #["))
                .unwrap_or(try_new_body.len().min(2000));
            let try_new_section = &try_new_body[..body_end];

            let re = match *arch {
                "x86" => &x86_feature_re,
                "aarch64" => &arm_feature_re,
                _ => continue,
            };

            let checked: BTreeSet<String> = re
                .captures_iter(try_new_section)
                .map(|c| c[1].to_string())
                .collect();

            // Determine expected features (strip baselines that aren't runtime-checked)
            let expected: BTreeSet<String> = match *arch {
                "x86" => token_def
                    .features
                    .iter()
                    .filter(|f| *f != "sse" && *f != "sse2")
                    .cloned()
                    .collect(),
                "aarch64" => {
                    // AArch64 tokens only check the extension feature, not neon (baseline)
                    token_def
                        .features
                        .iter()
                        .filter(|f| *f != "neon")
                        .cloned()
                        .collect()
                }
                _ => continue,
            };

            // Compare
            let missing: BTreeSet<&String> = expected.difference(&checked).collect();
            let extra: BTreeSet<&String> = checked.difference(&expected).collect();

            if missing.is_empty() && extra.is_empty() {
                println!("  {} — {} features verified", token_name, checked.len());
                verified += 1;
            } else {
                if !missing.is_empty() {
                    errors.push(format!(
                        "{}: try_new() MISSING checks for: {:?} (registry expects them)",
                        token_name, missing
                    ));
                }
                if !extra.is_empty() {
                    errors.push(format!(
                        "{}: try_new() has EXTRA checks for: {:?} (not in registry)",
                        token_name, extra
                    ));
                }
            }
        }
    }

    println!("\n=== try_new() Verification Summary ===");
    println!("  Verified: {} tokens", verified);
    println!("  Errors:   {}", errors.len());

    if !errors.is_empty() {
        println!("\n=== ERRORS ===");
        for e in &errors {
            println!("  {}", e);
        }
        bail!("try_new() verification failed with {} errors", errors.len());
    }

    println!("\nAll try_new() implementations match registry!");
    Ok(())
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
        eprintln!(
            "       cargo xtask parity              - Check API parity across architectures"
        );
        std::process::exit(1);
    }

    match args[1].as_str() {
        "generate" => generate_all()?,
        "validate" => {
            let reg = registry::Registry::load(&PathBuf::from("token-registry.toml"))?;
            validate_magetypes_with_registry(&reg)?;
            validate_try_new(&reg)?;
        }
        "validate-registry" => validate_registry()?,
        "parity" => check_api_parity()?,
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
    let gen_dir = PathBuf::from("archmage-macros/src/generated");
    fs::create_dir_all(&gen_dir)?;
    let gen_path = gen_dir.join("registry.rs");
    fs::write(&gen_path, &generated)?;
    println!("  Wrote {} ({} bytes)", gen_path.display(), generated.len());

    // Write generated/mod.rs for the macro crate
    let gen_mod = r#"//! Generated code from token-registry.toml.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

mod registry;
pub(crate) use registry::*;
"#;
    let gen_mod_path = gen_dir.join("mod.rs");
    fs::write(&gen_mod_path, gen_mod)?;
    println!("  Wrote {} ({} bytes)", gen_mod_path.display(), gen_mod.len());

    // Run rustfmt on the generated files so they stay fmt-clean
    // archmage-macros uses edition 2021
    for path in [&gen_path, &gen_mod_path] {
        let fmt_status = std::process::Command::new("rustfmt")
            .arg("--edition")
            .arg("2021")
            .arg(path)
            .status();
        match fmt_status {
            Ok(s) if s.success() => println!("  Formatted {}", path.display()),
            Ok(s) => println!("  Warning: rustfmt exited with {}", s),
            Err(e) => println!("  Warning: could not run rustfmt: {}", e),
        }
    }

    // Generate SIMD types (wide-like ergonomic types) into magetypes crate
    println!("\n=== Generating SIMD Types (magetypes) ===");

    let simd_dir = PathBuf::from("magetypes/src/simd");
    fs::create_dir_all(&simd_dir)?;
    fs::create_dir_all(simd_dir.join("generated"))?;
    fs::create_dir_all(simd_dir.join("generated/x86"))?;
    fs::create_dir_all(simd_dir.join("generated/arm"))?;
    fs::create_dir_all(simd_dir.join("generated/wasm"))?;

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

    // Run rustfmt on all generated .rs files for idempotent output
    let mut fmt_paths: Vec<PathBuf> = simd_files
        .iter()
        .map(|(rel, _)| simd_dir.join(rel))
        .collect();
    fmt_paths.push(simd_test_path.clone());
    for path in &fmt_paths {
        let _ = std::process::Command::new("rustfmt")
            .arg("--edition")
            .arg("2024")
            .arg(path)
            .status();
    }
    println!("  Formatted {} generated files", fmt_paths.len());

    // Generate reference documentation
    println!("\n=== Generating Reference Documentation ===");
    generate_reference_docs_new()?;

    // Run safety validation on generated code (hard failure)
    println!("\n=== Validating Magetypes Safety ===");
    validate_magetypes_with_registry(&reg)?;

    // Verify try_new() implementations match registry
    println!();
    validate_try_new(&reg)?;

    println!("\n=== Generation Complete ===");
    println!("  - SIMD types: magetypes/src/simd/");
    println!("  - Reference docs: docs/*.md");

    Ok(())
}

// ============================================================================
// API Parity Detection
// ============================================================================

/// A parsed method from generated SIMD types.
#[derive(Debug, Clone)]
struct ParsedMethod {
    name: String,
    signature: String,
    has_doc: bool,
}

/// A parsed SIMD type with its methods.
#[derive(Debug, Clone)]
struct ParsedType {
    name: String,
    methods: Vec<ParsedMethod>,
}

/// Parse methods from a generated file.
fn parse_simd_methods(content: &str) -> Vec<ParsedType> {
    let mut types = Vec::new();
    let mut current_type: Option<String> = None;
    let mut current_methods = Vec::new();
    let mut in_impl_block = false;
    let mut brace_depth = 0;
    let mut last_line_was_doc = false;

    // Regex to match impl blocks: impl TypeName { or impl TypeName for SomeTrait {
    let impl_re = Regex::new(r"^impl\s+(\w+x\d+)\s*\{").expect("invalid impl regex");
    // Regex to match pub fn declarations
    let fn_re = Regex::new(r"^\s*pub\s+fn\s+(\w+)\s*(<[^>]*>)?\s*\(([^)]*)\)(\s*->\s*.+)?")
        .expect("invalid fn regex");

    for line in content.lines() {
        let trimmed = line.trim();

        // Track doc comments
        if trimmed.starts_with("///") || trimmed.starts_with("#[doc") {
            last_line_was_doc = true;
            continue;
        }

        // Track impl blocks
        if let Some(cap) = impl_re.captures(trimmed) {
            // Save previous type if any
            if let Some(type_name) = current_type.take() {
                if !current_methods.is_empty() {
                    types.push(ParsedType {
                        name: type_name,
                        methods: std::mem::take(&mut current_methods),
                    });
                }
            }
            current_type = Some(cap[1].to_string());
            in_impl_block = true;
            brace_depth = 1;
            last_line_was_doc = false;
            continue;
        }

        // Track brace depth
        if in_impl_block {
            for c in trimmed.chars() {
                match c {
                    '{' => brace_depth += 1,
                    '}' => {
                        brace_depth -= 1;
                        if brace_depth == 0 {
                            in_impl_block = false;
                        }
                    }
                    _ => {}
                }
            }

            // Parse method signatures
            if let Some(cap) = fn_re.captures(trimmed) {
                let name = cap[1].to_string();
                let generics = cap.get(2).map(|m| m.as_str()).unwrap_or("");
                let params = &cap[3];
                let ret = cap.get(4).map(|m| m.as_str()).unwrap_or("");
                let signature = format!("fn {}{}({}){}", name, generics, params, ret);

                current_methods.push(ParsedMethod {
                    name,
                    signature,
                    has_doc: last_line_was_doc,
                });
            }
        }

        last_line_was_doc = false;
    }

    // Save last type
    if let Some(type_name) = current_type {
        if !current_methods.is_empty() {
            types.push(ParsedType {
                name: type_name,
                methods: current_methods,
            });
        }
    }

    types
}

/// Check API parity across architectures.
fn check_api_parity() -> Result<()> {
    println!("=== API Parity Detection ===\n");

    let simd_dir = PathBuf::from("magetypes/src/simd/generated");

    // Parse all architecture files
    let mut x86_types: BTreeMap<String, Vec<ParsedMethod>> = BTreeMap::new();
    let mut arm_types: BTreeMap<String, Vec<ParsedMethod>> = BTreeMap::new();
    let mut wasm_types: BTreeMap<String, Vec<ParsedMethod>> = BTreeMap::new();

    // x86 files
    for width in ["w128", "w256", "w512"] {
        let path = simd_dir.join(format!("x86/{}.rs", width));
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            for parsed_type in parse_simd_methods(&content) {
                x86_types
                    .entry(parsed_type.name.clone())
                    .or_default()
                    .extend(parsed_type.methods);
            }
        }
    }

    // ARM file
    let arm_path = simd_dir.join("arm/w128.rs");
    if arm_path.exists() {
        let content = fs::read_to_string(&arm_path)?;
        for parsed_type in parse_simd_methods(&content) {
            arm_types
                .entry(parsed_type.name.clone())
                .or_default()
                .extend(parsed_type.methods);
        }
    }

    // WASM file
    let wasm_path = simd_dir.join("wasm/w128.rs");
    if wasm_path.exists() {
        let content = fs::read_to_string(&wasm_path)?;
        for parsed_type in parse_simd_methods(&content) {
            wasm_types
                .entry(parsed_type.name.clone())
                .or_default()
                .extend(parsed_type.methods);
        }
    }

    println!("Parsed types:");
    println!("  x86:  {} types", x86_types.len());
    println!("  ARM:  {} types", arm_types.len());
    println!("  WASM: {} types\n", wasm_types.len());

    // Compare W128 types across all three architectures
    let w128_types = ["f32x4", "f64x2", "i8x16", "u8x16", "i16x8", "u16x8", "i32x4", "u32x4", "i64x2", "u64x2"];

    let mut parity_issues = Vec::new();
    let mut doc_issues = Vec::new();

    for type_name in w128_types {
        let x86_methods: BTreeSet<String> = x86_types
            .get(type_name)
            .map(|m| m.iter().map(|p| p.name.clone()).collect())
            .unwrap_or_default();
        let arm_methods: BTreeSet<String> = arm_types
            .get(type_name)
            .map(|m| m.iter().map(|p| p.name.clone()).collect())
            .unwrap_or_default();
        let wasm_methods: BTreeSet<String> = wasm_types
            .get(type_name)
            .map(|m| m.iter().map(|p| p.name.clone()).collect())
            .unwrap_or_default();

        // Find methods missing from each architecture
        let all_methods: BTreeSet<String> = x86_methods
            .union(&arm_methods)
            .cloned()
            .collect::<BTreeSet<_>>()
            .union(&wasm_methods)
            .cloned()
            .collect();

        for method in &all_methods {
            let in_x86 = x86_methods.contains(method);
            let in_arm = arm_methods.contains(method);
            let in_wasm = wasm_methods.contains(method);

            if !in_x86 || !in_arm || !in_wasm {
                let mut missing = Vec::new();
                if !in_x86 {
                    missing.push("x86");
                }
                if !in_arm {
                    missing.push("ARM");
                }
                if !in_wasm {
                    missing.push("WASM");
                }
                parity_issues.push(format!(
                    "  {}::{} — missing from: {}",
                    type_name,
                    method,
                    missing.join(", ")
                ));
            }
        }

        // Check doc coverage
        for methods in [&x86_types, &arm_types, &wasm_types] {
            if let Some(type_methods) = methods.get(type_name) {
                for m in type_methods {
                    if !m.has_doc && !["splat", "load", "store", "to_array", "from_array", "zero", "new"].contains(&m.name.as_str()) {
                        doc_issues.push(format!("  {}::{} — no doc comment", type_name, m.name));
                    }
                }
            }
        }
    }

    // Report results
    println!("=== Parity Report ===\n");

    if parity_issues.is_empty() {
        println!("All W128 types have identical APIs across x86/ARM/WASM!");
    } else {
        println!("Methods with architecture-specific availability:");
        parity_issues.sort();
        parity_issues.dedup();
        for issue in &parity_issues {
            println!("{}", issue);
        }
    }

    println!("\n=== Doc Coverage ===\n");

    if doc_issues.is_empty() {
        println!("All methods have doc comments!");
    } else {
        println!("Methods missing doc comments ({} total):", doc_issues.len());
        doc_issues.sort();
        doc_issues.dedup();
        for issue in doc_issues.iter().take(20) {
            println!("{}", issue);
        }
        if doc_issues.len() > 20 {
            println!("  ... and {} more", doc_issues.len() - 20);
        }
    }

    println!("\n=== Summary ===");
    println!("  Parity issues: {}", parity_issues.len());
    println!("  Doc gaps:      {}", doc_issues.len());

    Ok(())
}

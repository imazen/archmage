//! Cross-architecture API parity detection across generated SIMD types.

use anyhow::{Result, bail};
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

// ============================================================================
// API Parity Detection
// ============================================================================

/// A parsed method from generated SIMD types.
#[derive(Debug, Clone)]
struct ParsedMethod {
    name: String,
    #[allow(dead_code)]
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
            if let Some(type_name) = current_type.take()
                && !current_methods.is_empty()
            {
                types.push(ParsedType {
                    name: type_name,
                    methods: std::mem::take(&mut current_methods),
                });
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
    if let Some(type_name) = current_type
        && !current_methods.is_empty()
    {
        types.push(ParsedType {
            name: type_name,
            methods: current_methods,
        });
    }

    types
}

/// Check API parity across architectures.
///
/// When `strict` is true, fails if there are ANY parity issues.
/// This is used in CI to enforce full cross-platform parity.
pub(crate) fn check_api_parity(strict: bool) -> Result<()> {
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
    let w128_types = [
        "f32x4", "f64x2", "i8x16", "u8x16", "i16x8", "u16x8", "i32x4", "u32x4", "i64x2", "u64x2",
    ];

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
                    if !m.has_doc
                        && ![
                            "splat",
                            "load",
                            "store",
                            "to_array",
                            "from_array",
                            "zero",
                            "new",
                        ]
                        .contains(&m.name.as_str())
                    {
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

    // Known gaps that have no efficient native intrinsic support.
    // These are intentionally allowed and documented.
    let known_gaps: &[&str] = &[
        // Deprecated platform-named raw constructors: the x86 names 0.9.29
        // shipped, forwarding to `from_raw_t`. The cross-platform spellings
        // are `from_raw_t` and `from_raw`.
        "  f32x4::from_m128 — missing from: ARM, WASM",
        "  f64x2::from_m128d — missing from: ARM, WASM",
        "  i8x16::from_m128i — missing from: ARM, WASM",
        "  u8x16::from_m128i — missing from: ARM, WASM",
        "  i16x8::from_m128i — missing from: ARM, WASM",
        "  u16x8::from_m128i — missing from: ARM, WASM",
        "  i32x4::from_m128i — missing from: ARM, WASM",
        "  u32x4::from_m128i — missing from: ARM, WASM",
        "  i64x2::from_m128i — missing from: ARM, WASM",
        "  u64x2::from_m128i — missing from: ARM, WASM",
        // AVX-512 fast-path methods (only available with X64V4Token)
        "  i64x2::abs_fast — missing from: ARM, WASM",
        "  i64x2::min_fast — missing from: ARM, WASM",
        "  i64x2::max_fast — missing from: ARM, WASM",
        "  u64x2::min_fast — missing from: ARM, WASM",
        "  u64x2::max_fast — missing from: ARM, WASM",
        // x86 W128 uses V3 token (no _mm_srai_epi64 until AVX-512)
        "  i64x2::shr_arithmetic — missing from: x86",
    ];

    let actionable_issues: Vec<_> = parity_issues
        .iter()
        .filter(|issue| !known_gaps.contains(&issue.as_str()))
        .collect();

    println!("\n=== Summary ===");
    println!(
        "  Parity issues: {} ({} known gaps, {} actionable)",
        parity_issues.len(),
        parity_issues.len() - actionable_issues.len(),
        actionable_issues.len()
    );
    println!("  Doc gaps:      {}", doc_issues.len());

    if strict && !actionable_issues.is_empty() {
        println!("\nActionable parity issues (not in known_gaps allowlist):");
        for issue in &actionable_issues {
            println!("  {}", issue);
        }
        bail!(
            "{} parity issues found. All architectures must have identical APIs.",
            actionable_issues.len()
        );
    }

    Ok(())
}

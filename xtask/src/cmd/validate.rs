//! Token safety, `summon()` feature verification, compile budgets, and registry validation.

use anyhow::{Context, Result, bail};
use regex::Regex;
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use crate::{registry, soundness};

// ============================================================================
// Magetypes Validation
// ============================================================================
//
// Intrinsic-vs-token verification lives in `soundness.rs` (structure-aware
// scanner over impl-for-token blocks, #[target_feature] fns, and token-typed
// parameters). The former registry-file-mapping-driven validators were removed
// after the impls refactor emptied their file list, making them pass vacuously.

// ============================================================================
// summon() / summon() Feature Verification
// ============================================================================

/// Validate that every token's summon() checks exactly the features in the registry.
///
/// Parses src/tokens/*.rs to extract `is_x86_feature_available!("feat")` and
/// `is_aarch64_feature_available!("feat")` calls within each token's summon() block,
/// then compares against the registry.
///
/// Special cases:
/// - NeonToken: uses runtime detection like other aarch64 tokens
/// - Wasm128Token: uses #[cfg(target_feature)] not runtime detection, skip check
/// - x86 tokens: sse/sse2 are baseline, not checked at runtime
/// - AArch64 tokens: all features including neon are checked at runtime
pub(crate) fn validate_summon(reg: &registry::Registry) -> Result<()> {
    println!("=== summon() Feature Verification ===\n");

    let token_files = [
        ("src/tokens/generated/x86.rs", "x86"),
        ("src/tokens/generated/arm.rs", "aarch64"),
        ("src/tokens/generated/wasm.rs", "wasm"),
    ];

    // Regex to extract is_x86_feature_available!("feature") calls
    let x86_feature_re =
        Regex::new(r#"is_x86_feature_available!\("([^"]+)"\)"#).expect("invalid regex");
    let arm_feature_re =
        Regex::new(r#"is_aarch64_feature_available!\("([^"]+)"\)"#).expect("invalid regex");

    let mut errors: Vec<String> = Vec::new();
    let mut verified = 0usize;
    let impl_re = Regex::new(r"impl\s+SimdToken\s+for\s+(\w+)").expect("invalid impl regex");

    for (file_path, arch) in &token_files {
        let path = PathBuf::from(file_path);
        if !path.exists() {
            continue;
        }
        let content =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;

        // Find each "impl SimdToken for <TokenName>" block and extract its summon() features

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
                // Wasm128Token uses #[cfg(target_feature)], not runtime detection
                println!(
                    "  {} — compile-time cfg check (no runtime detection)",
                    token_name
                );
                verified += 1;
                continue;
            }

            // NeonToken no longer has always_available — uses runtime detection like others

            // Find the summon() body and its cold-path _DETECT function.
            // The detection logic may be in summon() directly (old style) or
            // in a separate #[cold] #[inline(never)] function called from summon().
            let impl_start = cap.get(0).unwrap().start();
            let remaining = &content[impl_start..];

            // Find fn summon()
            let summon_pos = match remaining.find("fn summon()") {
                Some(pos) => pos,
                None => continue,
            };

            // Search the summon() body (inline part)
            let summon_body = &remaining[summon_pos..];
            let body_end = summon_body
                .find("\n    fn ")
                .or_else(|| summon_body.find("\n    #["))
                .unwrap_or(summon_body.len().min(2000));
            let mut summon_section = summon_body[..body_end].to_string();

            // Also search the cold-path _DETECT function if it exists.
            // The detect function is a module-level function named like
            // X64_V3_DETECT, NEON_DETECT, etc.
            let detect_fn_name = {
                let mut result = String::new();
                let mut prev_was_upper = false;
                for (i, c) in token_name.chars().enumerate() {
                    if c.is_uppercase() {
                        if i > 0 && !prev_was_upper {
                            result.push('_');
                        }
                        result.push(c.to_ascii_lowercase());
                        prev_was_upper = true;
                    } else {
                        result.push(c);
                        prev_was_upper = c.is_uppercase();
                    }
                }
                let result = result.trim_end_matches("_token").to_string();
                format!("{result}_detect")
            };
            let detect_marker = format!("fn {detect_fn_name}()");
            if let Some(detect_pos) = content.find(&detect_marker) {
                let detect_end = (detect_pos + 2000).min(content.len());
                summon_section.push_str(&content[detect_pos..detect_end]);
            }

            let re = match *arch {
                "x86" => &x86_feature_re,
                "aarch64" => &arm_feature_re,
                _ => continue,
            };

            let checked: BTreeSet<String> = re
                .captures_iter(&summon_section)
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
                    // AArch64 tokens check ALL features including neon
                    token_def.features.iter().cloned().collect()
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
                        "{}: summon() MISSING checks for: {:?} (registry expects them)",
                        token_name, missing
                    ));
                }
                if !extra.is_empty() {
                    errors.push(format!(
                        "{}: summon() has EXTRA checks for: {:?} (not in registry)",
                        token_name, extra
                    ));
                }
            }
        }
    }

    println!("\n=== summon() Verification Summary ===");
    println!("  Verified: {} tokens", verified);
    println!("  Errors:   {}", errors.len());

    if !errors.is_empty() {
        println!("\n=== ERRORS ===");
        for e in &errors {
            println!("  {}", e);
        }
        bail!("summon() verification failed with {} errors", errors.len());
    }

    println!("\nAll summon() implementations match registry!");
    Ok(())
}

/// Per-target budgets for the generated backend surface that a given
/// `target_arch` actually compiles.
///
/// **Why this check exists.** `magetypes` is a serialization point: dozens of
/// crates sit above it in the build graph and none of them can start until its
/// unit finishes. Its own compile time is therefore fully on their critical
/// path — it does not overlap with anything and does not amortise across
/// cores. A percentage that looks small against a whole parallel build is paid
/// in full, serially, by every downstream crate.
///
/// **The numbers are calibrated, not invented.** Measured on `r5900xt`
/// (32 cores, 60 GB, rustc 1.98.1, idle), `magetypes` unit only with
/// dependencies pre-built, mean of three cold runs, spread +/- 0.02 s:
///
/// | x86-compiled lines | magetypes unit |
/// |---|---|
/// | 17 928 | 2.52 s |
/// | 21 174 | 2.70 s |
///
/// So +18.1% of compiled source cost +5.8% of serial compile time on that
/// host. Source lines are used rather than wall-clock because CI wall-clock is
/// far too noisy to gate on, and because this metric is deterministic — the
/// same tree always produces the same number.
///
/// **These are budgets, not floors.** Exceeding one is not automatically
/// wrong; it means the growth has to be a deliberate, reviewed edit to the
/// number below rather than something that lands unnoticed. Note the
/// asymmetry with `REQUIRED_FILE_FLOORS` in `soundness.rs`, which guards
/// against generated code *disappearing*; this guards against it *accreting*.
///
/// Only files the target actually compiles are counted: `impls/mod.rs`
/// cfg-gates `arm_neon` and `wasm128` away on x86, so a NEON-only change
/// costs an x86 build nothing.
const COMPILE_BUDGETS: &[(&str, &[&str], usize)] = &[
    (
        "x86_64",
        &[
            "magetypes/src/simd/impls/x86_v3.rs",
            "magetypes/src/simd/impls/x86_v3_delegated.rs",
            "magetypes/src/simd/impls/x86_v4.rs",
            "magetypes/src/simd/impls/x86_v4_w512_delegated.rs",
            "magetypes/src/simd/impls/x86_v4_f32_delegated.rs",
            "magetypes/src/simd/impls/x86_v4_f32_overrides.rs",
            "magetypes/src/simd/impls/polyfill.rs",
            "magetypes/src/simd/impls/scalar.rs",
            "magetypes/src/simd/backends/",
        ],
        // Includes `x86_v3_delegated.rs` + `x86_v4_w512_delegated.rs` + `polyfill.rs`; measured 30 846.
        31_500,
    ),
    (
        "aarch64",
        &[
            "magetypes/src/simd/impls/arm_neon.rs",
            "magetypes/src/simd/impls/polyfill.rs",
            "magetypes/src/simd/impls/scalar.rs",
            "magetypes/src/simd/backends/",
        ],
        21_300, // measured 20 687
    ),
    (
        "wasm32",
        &[
            "magetypes/src/simd/impls/wasm128.rs",
            "magetypes/src/simd/impls/polyfill.rs",
            "magetypes/src/simd/impls/scalar.rs",
            "magetypes/src/simd/backends/",
        ],
        20_600, // measured 19 966
    ),
];

/// Check the generated backend surface against [`COMPILE_BUDGETS`].
pub(crate) fn check_compile_budgets() -> Result<()> {
    println!("\n=== Generated backend surface (compile budget) ===");
    let count = |spec: &str| -> Result<usize> {
        let mut files: Vec<PathBuf> = Vec::new();
        if let Some(dir) = spec.strip_suffix('/') {
            for entry in std::fs::read_dir(dir)
                .with_context(|| format!("reading {dir}"))?
                .flatten()
            {
                let p = entry.path();
                if p.extension().is_some_and(|e| e == "rs") {
                    files.push(p);
                }
            }
        } else {
            files.push(PathBuf::from(spec));
        }
        files.sort();
        let mut total = 0usize;
        for f in files {
            let text =
                std::fs::read_to_string(&f).with_context(|| format!("reading {}", f.display()))?;
            total += text.lines().count();
        }
        Ok(total)
    };

    let mut over: Vec<String> = Vec::new();
    for (arch, specs, budget) in COMPILE_BUDGETS {
        let mut lines = 0usize;
        for spec in *specs {
            lines += count(spec)?;
        }
        let pct = (lines as f64 / *budget as f64) * 100.0;
        println!("  {arch:<8} {lines:>6} lines / {budget:>6} budget  ({pct:.1}%)");
        if lines > *budget {
            over.push(format!(
                "  {arch}: {lines} lines exceeds the {budget} budget by {}",
                lines - budget
            ));
        }
    }

    if !over.is_empty() {
        println!();
        anyhow::bail!(
            "generated backend surface over budget:\n{}\n\n\
             magetypes is a serialization point — dozens of crates wait on its unit, so\n\
             its compile time lands in full on every downstream build's critical path.\n\
             If the growth is intended, raise the budget in COMPILE_BUDGETS and say why\n\
             in the commit message. If it is not, the generator emitted more than you\n\
             expected — check what changed before raising the number.",
            over.join("\n")
        );
    }
    println!("  OK: every target's generated surface is within budget");
    Ok(())
}

/// Every check `cargo xtask validate` runs after the soundness scan. `just ci`
/// calls this too, so the local gate matches CI's Validate Token Safety job.
pub(crate) fn validate_after_soundness(reg: &registry::Registry) -> Result<()> {
    soundness::check_stderr_snapshot_portability()?;
    validate_summon(reg)?;
    check_compile_budgets()?;
    crate::cmd::packages::check_package_licenses()?;
    Ok(())
}

/// Parse and validate token-registry.toml
pub(crate) fn validate_registry() -> Result<()> {
    let registry_path = PathBuf::from("token-registry.toml");
    println!("=== Validating Token Registry ===");
    println!("Loading: {}", registry_path.display());

    let reg = registry::Registry::load(&registry_path)?;
    println!("{}", reg);
    println!("All validation checks passed.");
    Ok(())
}

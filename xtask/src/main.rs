//! Archmage xtask — code generation, validation, and reference docs.
//!
//! Usage:
//!   cargo xtask generate   - Generate SIMD types and reference docs
//!   cargo xtask validate   - Validate magetypes safety

use anyhow::{Context, Result};
use std::path::PathBuf;

mod cmd;
mod compare_arch;
mod expand_gen;
mod intrinsics_browser;
mod registry;
mod simd_types;
mod soundness;
mod token_gen;

// Re-export helpers used by sibling xtask modules (`intrinsics_browser`, `cmd::generate`).
pub(crate) use cmd::docs::{extract_safe_simd_functions, find_safe_simd_path_simple};

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
            "       cargo xtask ci | all            - Run ALL checks (must pass before push)"
        );
        eprintln!(
            "       cargo xtask soundness           - Structural & unsafe-surface soundness checks"
        );
        eprintln!(
            "       cargo xtask intrinsics-browser  - Regenerate docs/intrinsics-browser/ from CSV"
        );
        eprintln!("       cargo xtask miri                - Run magetypes under Miri (detects UB)");
        eprintln!(
            "       cargo xtask check-packages [--target T]... - Package the crates and check the packaged sources"
        );
        std::process::exit(1);
    }

    match args[1].as_str() {
        "generate" => cmd::generate::generate_all()?,
        "validate" => {
            let reg = registry::Registry::load(&PathBuf::from("token-registry.toml"))?;
            soundness::verify(&reg)?;
            cmd::validate::validate_after_soundness(&reg)?;
        }
        "validate-registry" => cmd::validate::validate_registry()?,
        "soundness" => cmd::miri::verify_intrinsic_soundness()?,
        "intrinsics-browser" => {
            let reg = registry::Registry::load(&PathBuf::from("token-registry.toml"))?;
            intrinsics_browser::generate_intrinsics_browser(&reg)?;
        }
        "gen-expand" => expand_gen::generate_expand_tests(&PathBuf::from("tests/expand"))?,
        "check-packages" => {
            // `--target T` may repeat; everything else is an error.
            let mut targets = Vec::new();
            let mut rest = args[2..].iter();
            while let Some(arg) = rest.next() {
                anyhow::ensure!(arg == "--target", "unknown check-packages argument: {arg}");
                targets.push(rest.next().context("--target needs a value")?.clone());
            }
            cmd::packages::check_packages(&targets)?
        }
        "miri" => cmd::miri::run_miri()?,
        "nostd" => cmd::packages::run_nostd_checks()?,
        "audit" => cmd::audit::run_safety_audit()?,
        "intrinsics-refresh" => cmd::audit::refresh_intrinsics_database()?,
        "compare-arch-results" => {
            let files: Vec<String> = if args.len() > 2 {
                args[2..].to_vec()
            } else {
                // Auto-discover .jsonl files in arch-results/ directory
                let dir = std::path::Path::new("arch-results");
                if dir.is_dir() {
                    compare_arch::find_jsonl_files(dir)
                } else {
                    Vec::new()
                }
            };
            compare_arch::compare_arch_results(&files)?;
        }
        "ci" | "all" => cmd::ci::run_ci()?,
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            eprintln!("\nAvailable commands:");
            eprintln!("  generate          Regenerate all code from token-registry.toml");
            eprintln!("  validate          Validate magetypes safety + summon() features");
            eprintln!("  validate-registry Parse and validate token-registry.toml");
            eprintln!("  soundness         Verify structural & unsafe-surface soundness");
            eprintln!("  intrinsics-browser  Regenerate docs/intrinsics-browser/ from CSV");
            eprintln!("  miri              Run Miri on magetypes (UB detection)");
            eprintln!("  audit             Scan for safety-critical code");
            eprintln!("  intrinsics-refresh  Re-extract intrinsics from current Rust toolchain");
            eprintln!(
                "  compare-arch-results  Compare JSONL outputs from arch_exercise across architectures"
            );
            eprintln!("  ci | all          Run all CI checks");
            std::process::exit(1);
        }
    }

    Ok(())
}

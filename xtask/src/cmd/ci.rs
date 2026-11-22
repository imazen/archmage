//! `cargo xtask ci` — comprehensive local verification pipeline.

use anyhow::{Context, Result, bail};
use std::fs;
use std::path::PathBuf;

use crate::cmd::miri::miri_toolchain;
use crate::cmd::parity::check_api_parity;
use crate::cmd::validate::validate_after_soundness;
use crate::{registry, soundness};

// ============================================================================
// CI - Comprehensive Check (must pass before push/publish)
// ============================================================================

/// Run all CI checks. Must pass before any push or publish.
///
/// Checks:
/// 1. Regenerate all code
/// 2. Verify clean worktree (no uncommitted changes from generation)
/// 3. Validate magetypes safety
/// 4. Validate summon() feature checks
/// 5. Check API parity (strict - fails on any issues)
/// 6. Intrinsic soundness verification
/// 7. Run clippy (with avx512)
/// 8. Run clippy (default features)
/// 9. Run tests
/// 10. Check formatting
/// 11. Build documentation (catches broken doc links, -Dwarnings)
/// 12. Miri (UB detection)
/// 13. ARM64 cross-compilation + tests (requires cross + Docker)
/// 14. WASM cross-compilation + tests (requires wasmtime)
/// 15. ARM64 clippy
pub(crate) fn run_ci() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    ARCHMAGE CI CHECK                             ║");
    println!("║  All checks must pass before push or publish!                    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    let mut timer = StepTimer::default();

    // Step 1: Generate all code
    timer.begin("Step 1/19: Regenerating code");
    crate::cmd::generate::generate_all()?;
    println!("└─ Code generation complete ─────────────────────────────────────────┘\n");

    // Check without rewriting generated output: formatting it here would hide
    // generator drift that GitHub's raw generation check correctly rejects.
    timer.begin("Checking workspace formatting");
    let fmt = std::process::Command::new("cargo")
        .args(["fmt", "--all", "--", "--check"])
        .status()
        .context("Failed to run cargo fmt")?;
    if !fmt.success() {
        bail!("Workspace formatting differs; run cargo fmt and check generator output");
    }
    println!("  ✓ Workspace formatting is clean");
    println!("└─ Formatting check complete ─────────────────────────────────────────┘\n");

    // Step 2: Check for clean worktree
    timer.begin("Step 2/19: Checking for uncommitted changes");
    let status = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .context("Failed to run git status")?;
    let changes = String::from_utf8_lossy(&status.stdout);
    if !changes.trim().is_empty() {
        println!("Uncommitted changes detected:");
        println!("{}", changes);
        bail!(
            "Working tree is dirty after code generation. \
             Commit these changes before pushing:\n{}",
            changes
        );
    }
    println!("  ✓ Working tree is clean");
    println!("└─ Worktree check passed ────────────────────────────────────────────┘\n");

    // Miri interprets single-threaded for minutes in its own build directory,
    // so it runs alongside the steps below; Step 15 collects the result.
    let miri = BackgroundStep::miri();

    // Step 3: Intrinsic soundness verification (structure-aware scanner)
    timer.begin("Step 3/19: Verifying intrinsic soundness");
    let reg = registry::Registry::load(&PathBuf::from("token-registry.toml"))?;
    soundness::verify(&reg)?;
    println!("└─ Soundness verification passed ────────────────────────────────────┘\n");

    timer.begin("Step 4/19: Validating tokens, delegation, budgets, licenses");
    validate_after_soundness(&reg)?;
    println!("└─ Validation passed ───────────────────────────────────────────────┘\n");

    // Step 5: Parity check (strict mode - fails on any issues)
    timer.begin("Step 5/19: Checking API parity (strict)");
    check_api_parity(true)?;
    println!("└─ Parity check passed ──────────────────────────────────────────────┘\n");

    // Step 6: xtask self-tests — the soundness scanner's own regression
    // tests (planted violations must fail, gated code must pass) plus the
    // registry tests. Without this step a rotted checker passes silently.
    timer.begin("Step 6/19: Testing the verifiers (cargo test -p xtask)");
    let xtask_tests = std::process::Command::new("cargo")
        .args(["test", "-p", "xtask"])
        .status()
        .context("Failed to run xtask self-tests")?;
    if !xtask_tests.success() {
        bail!("xtask self-tests failed — the verification tooling itself is broken");
    }
    println!("└─ Verifier self-tests passed ───────────────────────────────────────┘\n");

    // The no_std build uses its own build directory, so it runs alongside
    // the std clippy and test steps; Step 10 collects it.
    let nostd = BackgroundStep::start(
        "no_std",
        "nostd",
        "target/nostd-ci.log",
        Some("target/nostd"),
    );

    // Step 7: Clippy
    timer.begin("Step 7/19: Running clippy");
    let clippy = std::process::Command::new("cargo")
        .args(["clippy", "--features", "std avx512", "--", "-D", "warnings"])
        .status()
        .context("Failed to run clippy")?;
    if !clippy.success() {
        bail!("Clippy failed with warnings/errors");
    }
    println!("  ✓ Clippy passed (with avx512)");
    println!("└─ Clippy check passed ──────────────────────────────────────────────┘\n");

    // Step 8: Clippy with default features (catches warnings hidden by avx512)
    timer.begin("Step 8/19: Running clippy (default features)");
    let clippy_default = std::process::Command::new("cargo")
        .args(["clippy", "-p", "magetypes", "--", "-D", "warnings"])
        .status()
        .context("Failed to run clippy (default features)")?;
    if !clippy_default.success() {
        bail!("Clippy (default features) failed with warnings/errors");
    }
    // Macro output only meets clippy in a consuming crate.
    let clippy_consumer = std::process::Command::new("cargo")
        .args([
            "clippy",
            "-p",
            "archmage-no-features-test",
            "-p",
            "archmage-clippy-consumer",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ])
        .status()
        .context("Failed to run clippy (macro consumer crate)")?;
    if !clippy_consumer.success() {
        bail!("Clippy (macro consumer crate) failed with warnings/errors");
    }
    println!("  ✓ Clippy passed (default features + macro consumer crate)");
    println!("└─ Clippy check passed (default features) ───────────────────────────┘\n");

    // Proc-macro unit tests are not included by the root package's cargo test.
    for features in ["", "avx512"] {
        let status = std::process::Command::new("cargo")
            .args([
                "test",
                "-p",
                "archmage-macros",
                "--lib",
                "--features",
                features,
            ])
            .status()
            .context("Failed to run macro contract tests")?;
        if !status.success() {
            bail!("Macro contract tests failed (features: {features})");
        }
    }

    // Step 9: Tests
    timer.begin("Step 9/19: Running tests");
    let tests = std::process::Command::new("cargo")
        .args(["test", "--features", "std avx512"])
        .status()
        .context("Failed to run tests")?;
    if !tests.success() {
        bail!("Tests failed");
    }
    let float_contracts = std::process::Command::new("cargo")
        .args([
            "test",
            "-p",
            "magetypes",
            "--features",
            "std avx512",
            "--test",
            "float_inequality",
            "--test",
            "float_negation",
        ])
        .status()
        .context("Failed to run exact float contract tests")?;
    if !float_contracts.success() {
        bail!("Exact float contract tests failed");
    }
    println!("└─ All tests passed ─────────────────────────────────────────────────┘\n");

    // Step 10: no_std compilation and tests
    timer.begin("Step 10/19: no_std compilation + tests");
    if let Some(elapsed) = nostd.finish()? {
        timer.record("no_std (ran concurrently with Steps 7-9)", elapsed);
    }
    println!("└─ no_std checks passed ───────────────────────────────────────────────┘\n");

    // Step 11: No-features integration test (catches silent cfg elimination)
    timer.begin("Step 11/19: No-features integration test");
    let no_features = std::process::Command::new("cargo")
        .args(["test", "-p", "archmage-no-features-test"])
        .status()
        .context("Failed to test archmage-no-features-test")?;
    if !no_features.success() {
        bail!(
            "No-features integration test failed.\n\
             This crate exercises dispatch macros (#[autoversion], incant!) \
             with zero cargo features enabled — catching silent code \
             elimination from #[cfg(feature = ...)] guards."
        );
    }
    println!("  ✓ No-features integration test passed");
    println!("└─ No-features test passed ──────────────────────────────────────────┘\n");

    // Step 12: Format check
    timer.begin("Step 12/19: Checking code formatting");
    let fmt = std::process::Command::new("cargo")
        .args(["fmt", "--", "--check"])
        .status()
        .context("Failed to run cargo fmt")?;
    if !fmt.success() {
        bail!("Code is not formatted. Run 'cargo fmt' first.");
    }
    println!("  ✓ Code is properly formatted");
    println!("└─ Format check passed ──────────────────────────────────────────────┘\n");

    // Step 13: Public-API snapshot check (mirrors CI's "Public API Check" job).
    // The runner lives in the workspace-excluded apidoc/ package; check mode
    // fails if docs/public-api/ is stale instead of regenerating it.
    timer.begin("Step 13/19: Checking public-API snapshots");
    let apidoc = std::process::Command::new("cargo")
        .args(["test", "--manifest-path", "apidoc/Cargo.toml"])
        .env("ZEN_API_DOC", "check")
        .status()
        .context("Failed to run the apidoc snapshot check")?;
    if !apidoc.success() {
        bail!(
            "Public-API snapshots in docs/public-api/ are stale. \
             Run 'just api-doc' and commit the regenerated files."
        );
    }
    println!("  ✓ Public-API snapshots are current");
    println!("└─ Public-API snapshot check passed ─────────────────────────────────┘\n");

    // Step 14: Documentation build (catches broken doc links)
    timer.begin("Step 14/19: Building documentation");
    let doc = std::process::Command::new("cargo")
        .args([
            "doc",
            "-p",
            "archmage",
            "-p",
            "magetypes",
            "-p",
            "magemath",
            "--features",
            "std avx512",
            "--no-deps",
        ])
        .env("RUSTDOCFLAGS", "-Dwarnings")
        .status()
        .context("Failed to run cargo doc")?;
    if !doc.success() {
        bail!("Documentation build failed (broken doc links or rustdoc warnings)");
    }
    println!("  ✓ Documentation builds cleanly");
    println!("└─ Documentation check passed ─────────────────────────────────────────┘\n");

    // Step 13: Miri testing (UB detection)
    timer.begin("Step 15/19: Running Miri (UB detection)");
    if let Some(elapsed) = miri.finish()? {
        timer.record("Miri (ran concurrently with Steps 3-14)", elapsed);
    }
    println!("└─ Miri check complete ──────────────────────────────────────────────┘\n");

    // Step 14: ARM64 cross-compilation + tests
    timer.begin("Step 16/19: ARM64 cross-compilation + tests");
    let cross_ok = std::process::Command::new("cross")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let docker_ok = std::process::Command::new("docker")
        .arg("info")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if cross_ok && docker_ok {
        let arm_test = std::process::Command::new("cross")
            .args([
                "test",
                "--target",
                "aarch64-unknown-linux-gnu",
                "--features",
                "std",
            ])
            .status()
            .context("Failed to run cross test for ARM64")?;
        if !arm_test.success() {
            bail!("ARM64 cross-compilation tests failed");
        }
        println!("  ✓ ARM64 tests passed");
    } else {
        println!("  ⚠ ARM64 cross-testing skipped (requires 'cross' + Docker)");
        if !cross_ok {
            println!(
                "    Install cross: cargo install cross --git https://github.com/cross-rs/cross"
            );
        }
        if !docker_ok {
            println!("    Docker must be running for ARM64 cross-compilation");
        }
    }
    println!("└─ ARM64 cross-compilation complete ───────────────────────────────┘\n");

    // Step 15: WASM cross-compilation + tests
    timer.begin("Step 17/19: WASM cross-compilation + tests");
    let wasmtime_ok = std::process::Command::new("wasmtime")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let wasm_target_ok = std::process::Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("wasm32-wasip1"))
        .unwrap_or(false);

    if wasmtime_ok && wasm_target_ok {
        // Test lib + feature_consistency + wasm_intrinsics_exercise
        let wasm_test = std::process::Command::new("cargo")
            .args([
                "test",
                "--target",
                "wasm32-wasip1",
                "--lib",
                "--test",
                "feature_consistency",
                "--test",
                "wasm_intrinsics_exercise",
            ])
            .env("RUSTFLAGS", "-C target-feature=+simd128")
            .env("CARGO_TARGET_WASM32_WASIP1_RUNNER", "wasmtime")
            .status()
            .context("Failed to run WASM tests")?;
        if !wasm_test.success() {
            bail!("WASM cross-compilation tests failed");
        }
        println!("  ✓ WASM tests passed");
    } else {
        println!("  ⚠ WASM cross-testing skipped (requires wasmtime + wasm32-wasip1 target)");
        if !wasmtime_ok {
            println!("    Install wasmtime: curl https://wasmtime.dev/install.sh -sSf | bash");
        }
        if !wasm_target_ok {
            println!("    Install target: rustup target add wasm32-wasip1");
        }
    }
    println!("└─ WASM cross-compilation complete ───────────────────────────────┘\n");

    // Step 16: ARM64 clippy
    timer.begin("Step 18/19: ARM64 clippy");
    if cross_ok && docker_ok {
        let arm_clippy = std::process::Command::new("cargo")
            .args([
                "clippy",
                "--target",
                "aarch64-unknown-linux-gnu",
                "--features",
                "std",
                "--",
                "-D",
                "warnings",
            ])
            .status()
            .context("Failed to run ARM64 clippy")?;
        if !arm_clippy.success() {
            bail!("ARM64 clippy failed");
        }
        println!("  ✓ ARM64 clippy passed");
    } else {
        println!("  ⚠ ARM64 clippy skipped (requires cross tooling — see Step 11)");
    }
    println!("└─ ARM64 clippy complete ─────────────────────────────────────────┘\n");

    // Step 19: optimized consumer codegen against the last pushed main, on
    // x86_64, AArch64 and WASM. CI runs xtask/codegen.py only for pull
    // requests; work pushed straight to main is compared here instead.
    timer.begin("Step 19/19: Comparing consumer codegen with origin/main");
    let rev = |r: &str| -> Result<String> {
        let out = std::process::Command::new("git")
            .args(["rev-parse", r])
            .output()
            .with_context(|| format!("Failed to resolve {r}"))?;
        if !out.status.success() {
            bail!("git rev-parse {r} failed; fetch origin first (jj git fetch)");
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let (base, head) = (rev("origin/main")?, rev("HEAD")?);
    if base == head {
        println!("  ✓ HEAD is origin/main; nothing to compare");
    } else {
        let codegen = std::process::Command::new("python3")
            .args(["xtask/codegen.py", &base, &head])
            .status()
            .context("Failed to run xtask/codegen.py")?;
        if !codegen.success() {
            bail!(
                "Consumer codegen differs from origin/main without a reviewed exception \
                 (xtask/codegen.py); inspect the printed instructions"
            );
        }
        println!("  ✓ Codegen matches origin/main or differs only as reviewed");
    }
    println!("└─ Codegen comparison complete ────────────────────────────────────┘\n");

    timer.finish();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  ✓ ALL CI CHECKS PASSED - Safe to push/publish                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// An xtask subcommand run as a background child of `just ci` and joined at
/// a later step. Its stdout and stderr go to a log that is printed when it
/// fails and summarized when it passes, so the log of the other steps stays
/// readable. `target_dir` gives the child its own `CARGO_TARGET_DIR`, so it
/// never waits on the main build directory's cargo lock.
enum BackgroundStep {
    /// The step cannot run here (the reason is printed at join time).
    Unavailable { name: String, reason: String },
    Running {
        name: String,
        child: std::process::Child,
        log: PathBuf,
        started: std::time::Instant,
    },
}

impl BackgroundStep {
    fn start(name: &str, subcommand: &str, log: &str, target_dir: Option<&str>) -> Self {
        let log = PathBuf::from(log);
        let _ = fs::create_dir_all("target");
        let spawn = || -> Result<std::process::Child> {
            let out =
                fs::File::create(&log).with_context(|| format!("creating {}", log.display()))?;
            let err = out.try_clone()?;
            let mut command = std::process::Command::new(std::env::current_exe()?);
            command.arg(subcommand).stdout(out).stderr(err);
            if let Some(dir) = target_dir {
                command.env("CARGO_TARGET_DIR", dir);
            }
            Ok(command.spawn()?)
        };
        match spawn() {
            Ok(child) => {
                println!(
                    "  {name} started in the background (log: {})\n",
                    log.display()
                );
                BackgroundStep::Running {
                    name: name.to_string(),
                    child,
                    log,
                    started: std::time::Instant::now(),
                }
            }
            Err(e) => BackgroundStep::Unavailable {
                name: name.to_string(),
                reason: e.to_string(),
            },
        }
    }

    /// Miri, when the pinned toolchain has it.
    fn miri() -> Self {
        let toolchain = miri_toolchain();
        let available = std::process::Command::new("cargo")
            .args([format!("+{toolchain}"), "miri".into(), "--version".into()])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !available {
            return BackgroundStep::Unavailable {
                name: "Miri".to_string(),
                reason: format!(
                    "not available on {toolchain}; install with: rustup toolchain install \
                     {toolchain} --profile minimal --component miri,rust-src"
                ),
            };
        }
        BackgroundStep::start("Miri", "miri", "target/miri-ci.log", None)
    }

    /// Wait for the child; `Ok(Some(elapsed))` on success, `Ok(None)` when the
    /// step could not run, `Err` with the log when it failed.
    fn finish(self) -> Result<Option<std::time::Duration>> {
        match self {
            BackgroundStep::Unavailable { name, reason } => {
                println!("  ⚠ {name} skipped: {reason}");
                Ok(None)
            }
            BackgroundStep::Running {
                name,
                mut child,
                log,
                started,
            } => {
                let status = child
                    .wait()
                    .with_context(|| format!("waiting for the {name} job"))?;
                let elapsed = started.elapsed();
                let text = fs::read_to_string(&log).unwrap_or_default();
                if !status.success() {
                    println!("{text}");
                    bail!(
                        "{name} failed ({status}); its full output is above and in {}",
                        log.display()
                    );
                }
                for line in text
                    .lines()
                    .filter(|l| l.contains("  miri: ") || l.starts_with("  ✓"))
                {
                    println!("{line}");
                }
                println!(
                    "  ✓ {name} passed in {:.0} s, concurrently with the steps above (log: {})",
                    elapsed.as_secs_f64(),
                    log.display()
                );
                Ok(Some(elapsed))
            }
        }
    }
}

/// Wall-clock per `just ci` step, printed as a table at the end so the slow
/// steps are visible without instrumenting a run by hand.
#[derive(Default)]
struct StepTimer {
    done: Vec<(String, std::time::Duration)>,
    current: Option<(String, std::time::Instant)>,
    started: Option<std::time::Instant>,
}

impl StepTimer {
    /// Close the running step, print the banner for the next one, and start it.
    fn begin(&mut self, title: &str) {
        self.close();
        self.started.get_or_insert_with(std::time::Instant::now);
        println!(
            "┌─ {title} {}┐",
            "─".repeat(66usize.saturating_sub(title.len() + 4))
        );
        self.current = Some((title.to_string(), std::time::Instant::now()));
    }

    fn close(&mut self) {
        if let Some((title, start)) = self.current.take() {
            self.done.push((title, start.elapsed()));
        }
    }

    /// Add a duration measured elsewhere (a concurrent job) to the table.
    fn record(&mut self, title: &str, elapsed: std::time::Duration) {
        self.done.push((title.to_string(), elapsed));
    }

    fn finish(&mut self) {
        self.close();
        let total = self.started.map(|s| s.elapsed()).unwrap_or_default();
        let mut sorted: Vec<_> = self.done.iter().collect();
        sorted.sort_by_key(|step| std::cmp::Reverse(step.1));
        println!(
            "=== ci step times (slowest first; total {:.0} s) ===",
            total.as_secs_f64()
        );
        for (title, elapsed) in sorted {
            println!("  {:>6.1} s  {title}", elapsed.as_secs_f64());
        }
        println!();
    }
}

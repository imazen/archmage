//! Crate packaging, license-file verification, and `no_std` target matrix checks.

use anyhow::{Context, Result, bail};
use std::fs;
use std::path::PathBuf;

/// Check that the sub-crates ship both license texts, identical to the repo root.
/// The workspace Cargo.toml lists `license = "MIT OR Apache-2.0"`, and every
/// published sub-crate includes `/LICENSE*`.
///
/// A package's `include` list only reaches files inside its own directory, so
/// archmage-macros and magetypes keep copies; the root crate includes `/LICENSE*`.
pub(crate) fn check_package_licenses() -> Result<()> {
    println!("\n=== License texts in the published sub-crates ===");
    for crate_dir in ["archmage-macros", "magetypes", "magemath"] {
        for name in ["LICENSE-MIT", "LICENSE-APACHE"] {
            let root = std::fs::read(name).with_context(|| format!("reading {name}"))?;
            let copy = PathBuf::from(crate_dir).join(name);
            let packaged = std::fs::read(&copy).with_context(|| {
                format!(
                    "{} is missing; copy {name} from the repository root",
                    copy.display()
                )
            })?;
            anyhow::ensure!(
                packaged == root,
                "{} differs from the root {name}; copy the root file over it",
                copy.display()
            );
        }
    }
    println!("  OK: archmage-macros and magetypes carry LICENSE-MIT and LICENSE-APACHE");
    Ok(())
}

/// Build the three packaged libraries together, without publishing anything.
///
/// Check that `floor` is a semver-compatible minimum version `<= workspace`
/// so `magemath` can release independently against an already-published `0.9.x`
/// floor even after `archmage`/`magetypes` advance to a newer `0.9.(x+k)` on `main`.
fn semver_compatible_floor(floor: &str, workspace: &str) -> bool {
    let parse_xyz = |v: &str| -> Option<(u64, u64, u64)> {
        if v.contains('-') || v.contains('+') {
            return None;
        }
        let mut parts = v.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch = parts.next()?.parse().ok()?;
        if parts.next().is_some() {
            return None;
        }
        Some((major, minor, patch))
    };
    if floor == workspace {
        return true;
    }
    let (Some((fmaj, fmin, fpat)), Some((wmaj, wmin, wpat))) =
        (parse_xyz(floor), parse_xyz(workspace))
    else {
        return false;
    };
    if wmaj == 0 {
        fmaj == 0 && fmin == wmin && fpat <= wpat
    } else {
        fmaj == wmaj && (fmin, fpat) <= (wmin, wpat)
    }
}

/// `cargo package` each crate, extract the archives into
/// `target/package-verification/`, check that the published dependency
/// protocol is what the release needs (archmage pins archmage-macros exactly,
/// magetypes requires the archmage version), and `cargo check` the packaged
/// sources as one workspace, once per `--target` (host when none is given).
pub(crate) fn check_packages(targets: &[String]) -> Result<()> {
    use std::process::Command;
    const CRATES: [&str; 4] = ["archmage", "archmage-macros", "magetypes", "magemath"];

    let metadata: serde_json::Value = serde_json::from_slice(
        &Command::new("cargo")
            .args(["metadata", "--no-deps", "--format-version", "1"])
            .output()
            .context("cargo metadata")?
            .stdout,
    )?;
    let packages = metadata["packages"].as_array().context("packages")?;
    let version = |name: &str| -> Result<String> {
        packages
            .iter()
            .find(|p| p["name"] == name)
            .and_then(|p| p["version"].as_str())
            .map(String::from)
            .with_context(|| format!("{name} is not a workspace member"))
    };
    let target_dir = PathBuf::from(
        metadata["target_directory"]
            .as_str()
            .context("target dir")?,
    );

    let mut package = Command::new("cargo");
    package.args(["package", "--no-verify", "--allow-dirty"]);
    for crate_name in CRATES {
        package.args(["-p", crate_name]);
    }
    println!("+ {package:?}");
    anyhow::ensure!(package.status()?.success(), "cargo package failed");

    let staging = target_dir.join("package-verification").join(format!(
        "sources-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs()
    ));
    fs::create_dir_all(&staging)?;
    let mut members = Vec::new();
    for crate_name in CRATES {
        let member = format!("{crate_name}-{}", version(crate_name)?);
        let archive = target_dir.join("package").join(format!("{member}.crate"));
        let archive_bytes = fs::metadata(&archive)
            .with_context(|| format!("stat {}", archive.display()))?
            .len();
        println!("  packaged {member}.crate: {archive_bytes} bytes");
        if crate_name == "magetypes" {
            const MAGETYPES_MAX_BYTES: u64 = 350 * 1024;
            anyhow::ensure!(
                archive_bytes <= MAGETYPES_MAX_BYTES,
                "magetypes package size ({archive_bytes} bytes) exceeds 350 KiB budget ({MAGETYPES_MAX_BYTES} bytes)"
            );
        }
        let status = Command::new("tar")
            .arg("-xzf")
            .arg(&archive)
            .arg("-C")
            .arg(&staging)
            .status()
            .context("tar")?;
        anyhow::ensure!(status.success(), "extracting {}", archive.display());
        members.push(member);
    }

    // The published dependency protocol, not just local path resolution.
    let manifest = |member: &str| -> Result<toml::Value> {
        Ok(toml::from_str(&fs::read_to_string(
            staging.join(member).join("Cargo.toml"),
        )?)?)
    };
    let archmage = manifest(&members[0])?;
    let magetypes = manifest(&members[2])?;
    let magemath = manifest(&members[3])?;
    anyhow::ensure!(
        archmage["dependencies"]["archmage-macros"]["version"].as_str()
            == Some(&format!("={}", version("archmage-macros")?)),
        "packaged archmage must pin archmage-macros exactly"
    );
    anyhow::ensure!(
        magetypes["dependencies"]["archmage"]["version"].as_str() == Some(&version("archmage")?),
        "packaged magetypes must require the workspace archmage version"
    );
    let dep_archmage = magemath["dependencies"]["archmage"]["version"]
        .as_str()
        .context("magemath dependencies.archmage.version")?;
    let dep_magetypes = magemath["dependencies"]["magetypes"]["version"]
        .as_str()
        .context("magemath dependencies.magetypes.version")?;
    let ws_archmage = version("archmage")?;
    let ws_magetypes = version("magetypes")?;
    anyhow::ensure!(
        dep_archmage == dep_magetypes
            && semver_compatible_floor(dep_archmage, &ws_archmage)
            && semver_compatible_floor(dep_magetypes, &ws_magetypes),
        "packaged magemath must require matching, semver-compatible archmage/magetypes versions \
         <= workspace ({ws_archmage} / {ws_magetypes}), got {dep_archmage} / {dep_magetypes}"
    );

    let mut workspace = format!(
        "[workspace]\nresolver = \"3\"\nmembers = {}\n\n[patch.crates-io]\n",
        serde_json::to_string(&members)?
    );
    for (crate_name, member) in CRATES.iter().zip(&members) {
        workspace.push_str(&format!("{crate_name} = {{ path = {member:?} }}\n"));
    }
    fs::write(staging.join("Cargo.toml"), workspace)?;

    let host = [String::new()];
    for target in if targets.is_empty() {
        &host[..]
    } else {
        targets
    } {
        let mut check = Command::new("cargo");
        check
            .args([
                "check",
                "--workspace",
                "--all-features",
                "--lib",
                "--manifest-path",
            ])
            .arg(staging.join("Cargo.toml"));
        if !target.is_empty() {
            check.args(["--target", target]);
        }
        println!("+ {check:?}");
        anyhow::ensure!(check.status()?.success(), "packaged sources fail to check");
    }
    println!("Verified packaged sources: {}", staging.display());
    Ok(())
}

/// The no_std gate: archmage and magetypes compile and test without `std` on
/// the host, and magetypes compiles for two bare-metal targets, which catch
/// std leaks the host cannot (libstd is always linkable there). `just ci`
/// runs this as a background child in its own build directory
/// (`target/nostd`), so it does not wait on the std build's cargo lock.
pub(crate) fn run_nostd_checks() -> Result<()> {
    // Check archmage compiles under no_std (with macros)
    let nostd_check = std::process::Command::new("cargo")
        .args([
            "check",
            "-p",
            "archmage",
            "--no-default-features",
            "--features",
            "macros avx512",
        ])
        .status()
        .context("Failed to check archmage under no_std")?;
    if !nostd_check.success() {
        bail!("archmage fails to compile under no_std");
    }
    println!("  ✓ archmage compiles under no_std");

    // Check magetypes compiles under no_std
    let nostd_magetypes = std::process::Command::new("cargo")
        .args(["check", "-p", "magetypes", "--no-default-features"])
        .status()
        .context("Failed to check magetypes under no_std")?;
    if !nostd_magetypes.success() {
        bail!("magetypes fails to compile under no_std");
    }
    println!("  ✓ magetypes compiles under no_std");

    // Run magetypes tests under no_std (scalar tests + skip-when-unavailable)
    let nostd_tests = std::process::Command::new("cargo")
        .args(["test", "-p", "magetypes", "--no-default-features"])
        .status()
        .context("Failed to test magetypes under no_std")?;
    if !nostd_tests.success() {
        bail!("magetypes tests fail under no_std");
    }
    println!("  ✓ magetypes tests pass under no_std");

    // Run archmage tests under no_std (with macros)
    let nostd_archmage_tests = std::process::Command::new("cargo")
        .args([
            "test",
            "-p",
            "archmage",
            "--no-default-features",
            "--features",
            "macros avx512",
        ])
        .status()
        .context("Failed to test archmage under no_std")?;
    if !nostd_archmage_tests.success() {
        bail!("archmage tests fail under no_std");
    }
    println!("  ✓ archmage tests pass under no_std");

    // Cross-target no_std compilation (true no_std — catches std leaks the host can't)
    // These are MANDATORY: host-target no_std checks don't catch f64::log2() etc.
    // because libstd is always linkable on the host even without the "std" feature.
    let nostd_targets = [
        ("aarch64-unknown-none", "aarch64 bare metal"),
        ("thumbv7m-none-eabi", "ARM Cortex-M (no SIMD)"),
    ];
    for (target, desc) in &nostd_targets {
        // Auto-install if missing
        let installed = std::process::Command::new("rustup")
            .args(["target", "list", "--installed"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains(target))
            .unwrap_or(false);
        if !installed {
            println!("  Installing target {target}...");
            let install = std::process::Command::new("rustup")
                .args(["target", "add", target])
                .status()
                .context(format!("Failed to install target {target}"))?;
            if !install.success() {
                bail!("Failed to install target {target} — required for no_std CI");
            }
        }
        let check = std::process::Command::new("cargo")
            .args([
                "check",
                "-p",
                "magetypes",
                "--no-default-features",
                "--target",
                target,
            ])
            .status()
            .context(format!("Failed to check magetypes for {target}"))?;
        if !check.success() {
            bail!("magetypes fails to compile for {target} ({desc})");
        }
        println!("  ✓ magetypes compiles for {target} ({desc})");
    }
    println!("no_std checks passed");
    Ok(())
}

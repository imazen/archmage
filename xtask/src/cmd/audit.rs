//! Safety marker audit and stdarch intrinsic database refresh.

use anyhow::{Context, Result, bail};
use regex::Regex;
use std::fs;
use std::path::PathBuf;

use crate::intrinsics_browser::parse_csv_line;

/// Run safety audit - scan for critical code and verify invariants
pub(crate) fn run_safety_audit() -> Result<()> {
    println!("=== Safety Audit ===\n");

    // 1. Scan for CRITICAL/FRAGILE markers without SAFETY justification
    println!("Scanning for safety-critical code markers...\n");

    let critical_pattern =
        Regex::new(r"//\s*(CRITICAL|FRAGILE):\s*(.*)").expect("invalid critical regex");
    let safety_pattern = Regex::new(r"//\s*SAFETY:\s*(.*)").expect("invalid safety regex");
    let unsafe_pattern = Regex::new(r"\bunsafe\s*\{").expect("invalid unsafe regex");
    let forge_pattern =
        Regex::new(r"forge_token_dangerously\s*\(\s*\)").expect("invalid forge regex");
    let from_context_pattern =
        Regex::new(r"from_context\s*\(\s*\)").expect("invalid from_context regex");

    let mut critical_count = 0;
    let mut fragile_count = 0;
    let mut unsafe_without_safety = 0;
    let mut forge_calls = 0;
    let mut from_context_calls = 0;

    let dirs = ["src", "archmage-macros/src", "magetypes/src", "xtask/src"];

    for dir in dirs {
        for entry in walkdir::WalkDir::new(dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
            .filter(|e| !e.path().to_string_lossy().contains("/generated/"))
        {
            let content = fs::read_to_string(entry.path())?;
            let rel_path = entry.path().display();

            // Count markers
            for cap in critical_pattern.captures_iter(&content) {
                let marker = cap.get(1).unwrap().as_str();
                let desc = cap.get(2).unwrap().as_str();
                if marker == "CRITICAL" {
                    critical_count += 1;
                    println!("  CRITICAL in {}: {}", rel_path, desc);
                } else {
                    fragile_count += 1;
                    println!("  FRAGILE in {}: {}", rel_path, desc);
                }
            }

            // Check for unsafe blocks without SAFETY comments
            let lines: Vec<&str> = content.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                if unsafe_pattern.is_match(line) {
                    // Check previous 3 lines for SAFETY comment
                    let has_safety =
                        (i.saturating_sub(3)..i).any(|j| safety_pattern.is_match(lines[j]));
                    if !has_safety && !line.contains("// SAFETY") {
                        unsafe_without_safety += 1;
                    }
                }
            }

            // Count token constructor calls. `from_context()` is the
            // `#[target_feature]`-checked constructor every generated call site
            // uses; `forge_token_dangerously()` is its deprecated alias.
            forge_calls += forge_pattern.find_iter(&content).count();
            from_context_calls += from_context_pattern.find_iter(&content).count();
        }
    }

    println!("\n=== Audit Summary ===");
    println!("  CRITICAL markers: {}", critical_count);
    println!("  FRAGILE markers:  {}", fragile_count);
    println!(
        "  Unsafe blocks without SAFETY comment: {}",
        unsafe_without_safety
    );
    println!("  forge_token_dangerously() calls: {}", forge_calls);
    println!(
        "  from_context() calls:               {}",
        from_context_calls
    );

    // 2. Check intrinsics database freshness
    println!("\n=== Intrinsics Database ===");
    let db_path = PathBuf::from("docs/intrinsics/complete_intrinsics.csv");
    if db_path.exists() {
        let metadata = fs::metadata(&db_path)?;
        let modified = metadata.modified()?;
        let age = std::time::SystemTime::now()
            .duration_since(modified)
            .unwrap_or_default();
        let days = age.as_secs() / 86400;
        println!("  Database age: {} days", days);
        if days > 30 {
            println!("  ⚠ Consider running 'cargo xtask intrinsics-refresh'");
        }
    } else {
        println!("  ⚠ No intrinsics database found!");
        println!("  Run 'cargo xtask intrinsics-refresh' to create it");
    }

    println!("\nSee docs/SAFETY-CRITICAL.md for details on safety-critical areas.");

    Ok(())
}

/// Refresh the intrinsics database from the current Rust toolchain
pub(crate) fn refresh_intrinsics_database() -> Result<()> {
    println!("=== Refreshing Intrinsics Database ===\n");

    let script_path = PathBuf::from("xtask/extract_intrinsics.py");
    if !script_path.exists() {
        bail!(
            "extract_intrinsics.py not found at {}",
            script_path.display()
        );
    }

    // Run the Python script
    let output = std::process::Command::new("python3")
        .arg(&script_path)
        .output()
        .context("Failed to run extract_intrinsics.py")?;

    if !output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        bail!("extract_intrinsics.py failed");
    }

    let new_csv = String::from_utf8_lossy(&output.stdout);
    let new_lines: Vec<&str> = new_csv.lines().collect();
    println!(
        "Extracted {} intrinsics from current toolchain",
        new_lines.len() - 1
    );

    // Load existing database
    let db_path = PathBuf::from("docs/intrinsics/complete_intrinsics.csv");
    let old_csv = if db_path.exists() {
        fs::read_to_string(&db_path)?
    } else {
        String::new()
    };

    // Compare safe/unsafe status
    let parse_csv = |csv: &str| -> std::collections::HashMap<String, bool> {
        csv.lines()
            .skip(1) // header
            .filter_map(|line| {
                let fields = parse_csv_line(line);
                if fields.len() >= 4 {
                    let name = fields[1].clone();
                    let is_unsafe = fields[3] == "True";
                    Some((name, is_unsafe))
                } else {
                    None
                }
            })
            .collect()
    };

    let old_map = parse_csv(&old_csv);
    let new_map = parse_csv(&new_csv);

    // Find intrinsics that became safe
    let mut newly_safe = Vec::new();
    let mut newly_unsafe = Vec::new();

    for (name, is_unsafe) in &new_map {
        if let Some(&was_unsafe) = old_map.get(name) {
            if was_unsafe && !is_unsafe {
                newly_safe.push(name.clone());
            } else if !was_unsafe && *is_unsafe {
                newly_unsafe.push(name.clone());
            }
        }
    }

    if !newly_safe.is_empty() {
        println!("\n✓ Intrinsics that became SAFE ({}):", newly_safe.len());
        for name in &newly_safe {
            println!("  + {}", name);
        }
        println!("\nThese can now be called without unsafe inside #[target_feature] functions!");
    }

    if !newly_unsafe.is_empty() {
        println!(
            "\n⚠ Intrinsics that became UNSAFE ({}):",
            newly_unsafe.len()
        );
        for name in &newly_unsafe {
            println!("  - {}", name);
        }
    }

    // Count safe vs unsafe
    let safe_count = new_map.values().filter(|&&u| !u).count();
    let unsafe_count = new_map.values().filter(|&&u| u).count();
    println!("\nTotal: {} safe, {} unsafe", safe_count, unsafe_count);

    // Save new database
    fs::create_dir_all("docs/intrinsics")?;
    fs::write(&db_path, new_csv.as_ref())?;
    println!("\nSaved to {}", db_path.display());

    Ok(())
}

//! Cross-architecture SIMD result comparison.
//!
//! Parses JSONL files produced by `examples/arch_exercise.rs` from multiple
//! architectures and compares their outputs. Reports divergences with context.
//!
//! Usage: `cargo run -p xtask -- compare-arch-results <file1> <file2> ...`

use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::path::Path;

/// A single test result record from arch_exercise.
#[derive(Debug)]
struct Record {
    arch: String,
    type_name: String,
    op: String,
    input: String,
    output: String,
}

/// Tolerance specification for comparing outputs.
enum Tolerance {
    /// Outputs must be identical strings (hex-encoded, so bit-exact).
    Exact,
    /// Allow known differences (NaN payloads, signed zero, FMA precision).
    /// The string lists which kinds of differences are tolerated.
    Relaxed(&'static str),
}

fn tolerance_for(op: &str) -> Tolerance {
    match op {
        // Exact: these should produce identical bits across all architectures
        "round" | "floor" | "ceil" | "abs" | "sqrt" | "add" | "sub" | "mul" | "div" | "to_i32"
        | "to_i32_round" | "not" => Tolerance::Exact,

        // Relaxed: these have known platform-specific differences
        "neg" => Tolerance::Relaxed("signed-zero (sub(0,x) vs xor sign bit)"),
        "min" | "max" => Tolerance::Relaxed("NaN propagation (SSE returns second operand)"),
        "mul_add" | "mul_sub" => Tolerance::Relaxed("FMA vs separate mul+add"),
        "rcp_approx" | "rsqrt_approx" | "recip" | "rsqrt" => {
            Tolerance::Relaxed("platform-specific approximation precision")
        }
        "reduce_add" => Tolerance::Relaxed("FP associativity (tree vs left-fold)"),
        "reduce_min" | "reduce_max" => Tolerance::Relaxed("NaN propagation + signed zero"),
        _ => Tolerance::Exact,
    }
}

fn parse_record(line: &str) -> Option<Record> {
    // Minimal JSON parsing without serde — just extract string fields.
    // Format: {"arch":"x86_64","type":"f32x4","op":"round","input":[...],"output":[...]}
    let arch = extract_string_field(line, "arch")?;
    let type_name = extract_string_field(line, "type")?;
    let op = extract_string_field(line, "op")?;
    let input = extract_array_field(line, "input")?;
    let output = extract_array_field(line, "output")?;
    Some(Record {
        arch,
        type_name,
        op,
        input,
        output,
    })
}

fn extract_string_field(json: &str, field: &str) -> Option<String> {
    let pattern = format!("\"{field}\":\"");
    let start = json.find(&pattern)? + pattern.len();
    let end = json[start..].find('"')? + start;
    Some(json[start..end].to_string())
}

fn extract_array_field(json: &str, field: &str) -> Option<String> {
    let pattern = format!("\"{field}\":");
    let start = json.find(&pattern)? + pattern.len();
    let bracket_start = json[start..].find('[')? + start;
    let bracket_end = json[bracket_start..].find(']')? + bracket_start + 1;
    Some(json[bracket_start..bracket_end].to_string())
}

/// Run the cross-architecture comparison.
pub fn compare_arch_results(files: &[String]) -> Result<()> {
    if files.is_empty() {
        eprintln!("No input files specified.");
        eprintln!("Usage: cargo xtask compare-arch-results <file1.jsonl> <file2.jsonl> ...");
        std::process::exit(1);
    }

    // Parse all records, grouped by (type, op, input) → {arch → output}
    let mut grouped: BTreeMap<(String, String, String), BTreeMap<String, String>> = BTreeMap::new();
    let mut total_records = 0;

    for file_path in files {
        let content = std::fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read {file_path}"))?;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rec) = parse_record(line) {
                total_records += 1;
                let key = (rec.type_name, rec.op, rec.input);
                grouped.entry(key).or_default().insert(rec.arch, rec.output);
            } else {
                eprintln!("WARNING: Failed to parse line in {file_path}: {line}");
            }
        }
    }

    println!(
        "Parsed {total_records} records across {} unique (type, op, input) keys",
        grouped.len()
    );

    let mut exact_matches = 0;
    let mut relaxed_divergences = 0;
    let mut exact_divergences = 0;
    let mut errors = Vec::new();

    for ((type_name, op, input), arch_outputs) in &grouped {
        if arch_outputs.len() < 2 {
            continue; // need at least 2 architectures to compare
        }

        let outputs: Vec<(&String, &String)> = arch_outputs.iter().collect();
        let first_output = outputs[0].1;
        let all_same = outputs.iter().all(|(_, o)| *o == first_output);

        if all_same {
            exact_matches += 1;
        } else {
            match tolerance_for(op) {
                Tolerance::Exact => {
                    exact_divergences += 1;
                    let mut msg = format!("DIVERGENCE [{type_name}::{op}] input={input}\n");
                    for (arch, output) in &outputs {
                        msg.push_str(&format!("  {arch}: {output}\n"));
                    }
                    errors.push(msg);
                }
                Tolerance::Relaxed(reason) => {
                    relaxed_divergences += 1;
                    eprintln!("  [relaxed] {type_name}::{op} differs across arches ({reason})");
                }
            }
        }
    }

    println!("\n=== Cross-Architecture Comparison Summary ===");
    println!("  Exact matches:       {exact_matches}");
    println!("  Relaxed divergences: {relaxed_divergences} (expected, tolerated)");
    println!("  Exact divergences:   {exact_divergences}");

    if !errors.is_empty() {
        println!("\n=== FAILURES ===");
        for err in &errors {
            println!("{err}");
        }
        bail!("{exact_divergences} exact divergences found")
    } else {
        println!("\nAll cross-architecture results match (or are within tolerance).");
        Ok(())
    }
}

/// Find all .jsonl files in a directory.
pub fn find_jsonl_files(dir: &Path) -> Vec<String> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_jsonl_files(&path));
            } else if path.extension().is_some_and(|ext| ext == "jsonl") {
                files.push(path.to_string_lossy().to_string());
            }
        }
    }
    files
}

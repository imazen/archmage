//! Named implementations, ns per buffer, all-retained paired confidence bounds.
#![forbid(unsafe_code)]
fn main() {
    for path in std::env::args().skip(1) {
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let result = zenbench::SuiteResult::load(std::path::Path::new(&path)).unwrap();
        let placement = raw["cpu_placement"]["status"].as_str().unwrap();
        let verified = matches!(placement, "uniform_hardware" | "enforced_uniform");
        for (index, group) in result.comparisons.iter().enumerate() {
            let clean = !result.unreliable
                && !raw["incomplete"].as_bool().unwrap()
                && group.samples.len() == 50
                && group.samples.iter().all(|r| r.gate_clean == Some(true));
            let std = group
                .benchmarks
                .iter()
                .position(|b| b.name == "std_Rust_baseline" || b.name == "std")
                .unwrap();
            let native_std = group
                .benchmarks
                .iter()
                .position(|b| b.name == "system_C_powf")
                .unwrap_or(std);
            let mut edges = std::collections::BTreeSet::new();
            for (numerator, implementation) in group.benchmarks.iter().enumerate() {
                if implementation.name.starts_with("EMPTY_") {
                    continue;
                }
                for denominator in [std, native_std] {
                    if numerator != denominator {
                        edges.insert((numerator, denominator));
                    }
                }
                for (denominator, reference) in group.benchmarks.iter().enumerate().take(numerator)
                {
                    if !reference.name.starts_with("EMPTY_") {
                        edges.insert((numerator, denominator));
                    }
                }
            }
            for (numerator, denominator) in edges {
                let ratios: Vec<_> = group
                    .samples
                    .iter()
                    .map(|r| {
                        assert!(r.compensated_ns[denominator] > 0);
                        r.compensated_ns[numerator] as f64 / r.compensated_ns[denominator] as f64
                    })
                    .collect();
                let ci = zenbench::MeanCi::from_samples(&ratios, 100_000).unwrap();
                let cost = if ci.upper <= 1.02 {
                    "PASS"
                } else if ci.lower > 1.02 {
                    "FAIL"
                } else {
                    "INCONCLUSIVE"
                };
                let verdict = if !verified {
                    "EXPLORATORY"
                } else if !clean {
                    "INADMISSIBLE"
                } else {
                    cost
                };
                println!(
                    "{}",
                    serde_json::json!({"file":path,"group":group.group_name,
                    "implementation":group.benchmarks[numerator].name,"reference":group.benchmarks[denominator].name,
                    "numerator_mean_ns":raw["comparisons"][index]["benchmarks"][numerator]["summary"]["mean"],
                    "denominator_mean_ns":raw["comparisons"][index]["benchmarks"][denominator]["summary"]["mean"],
                    "comparison_family":if denominator == std || denominator == native_std {"predeclared_std_reference"} else {"descriptive_pairwise"},
                    "pairs":ratios.len(),"resource_clean":clean,"placement":placement,
                    "promotion_eligible":clean&&verified,"ratio_ci95":ci,"cost_bound_only":cost,"verdict":verdict})
                );
            }
        }
    }
}

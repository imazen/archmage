//! Direct paired comparisons; preserve exploratory placement and noisy rounds.
#![forbid(unsafe_code)]
fn main() {
    for path in std::env::args().skip(1) {
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let result = zenbench::SuiteResult::load(std::path::Path::new(&path)).unwrap();
        let placement = raw["cpu_placement"]["status"]
            .as_str()
            .expect("placement metadata required");
        let verified = matches!(placement, "uniform_hardware" | "enforced_uniform");
        for group in &result.comparisons {
            let clean = !result.unreliable
                && !raw["incomplete"].as_bool().unwrap()
                && group.samples.len() >= 50
                && group.samples.iter().all(|r| r.gate_clean == Some(true));
            for candidate in [11, 12] {
                assert_eq!(
                    group.benchmarks[candidate].name,
                    if candidate == 11 {
                        "const_caller"
                    } else {
                        "const_auto"
                    }
                );
                for baseline in 0..group.benchmarks.len() {
                    if [11, 12].contains(&baseline) {
                        continue;
                    }
                    let ratios: Vec<_> = group
                        .samples
                        .iter()
                        .map(|r| {
                            assert!(r.compensated_ns[baseline] > 0);
                            r.compensated_ns[candidate] as f64 / r.compensated_ns[baseline] as f64
                        })
                        .collect();
                    let ci = zenbench::MeanCi::from_samples(&ratios, 100_000).unwrap();
                    let verdict = if !verified {
                        "EXPLORATORY"
                    } else if !clean {
                        "INADMISSIBLE"
                    } else if ci.upper <= 1.02 {
                        "PASS"
                    } else if ci.lower > 1.02 {
                        "FAIL"
                    } else {
                        "INCONCLUSIVE"
                    };
                    println!(
                        "{}",
                        serde_json::json!({"file":path,"group":group.group_name,"baseline":group.benchmarks[baseline].name,"candidate":group.benchmarks[candidate].name,"pairs":ratios.len(),"resource_clean":clean,"placement":placement,"promotion_eligible":clean&&verified,"ratio_ci95":ci,"verdict":verdict})
                    );
                }
            }
        }
    }
}

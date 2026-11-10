//! All retained round ratios: no outlier filtering or independent best samples.
#![forbid(unsafe_code)]
fn main() {
    let mut rows = Vec::new();
    for path in std::env::args().skip(1) {
        let result = zenbench::SuiteResult::load(std::path::Path::new(&path)).unwrap();
        for group in &result.comparisons {
            let clean = !result.unreliable
                && group.samples.len() >= 50
                && group.samples.iter().all(|r| r.gate_clean == Some(true));
            for candidate in 1..group.benchmarks.len() {
                let ratios: Vec<_> = group
                    .samples
                    .iter()
                    .map(|r| r.compensated_ns[candidate] as f64 / r.compensated_ns[0] as f64)
                    .collect();
                let ci = zenbench::MeanCi::from_samples(&ratios, 100_000);
                let verdict = if !clean {
                    "INADMISSIBLE"
                } else if ci.as_ref().unwrap().upper <= 1.02 {
                    "PASS"
                } else if ci.as_ref().unwrap().lower > 1.02 {
                    "FAIL"
                } else {
                    "INCONCLUSIVE"
                };
                rows.push(serde_json::json!({"file":path,"group":group.group_name,"baseline":group.benchmarks[0].name,"candidate":group.benchmarks[candidate].name,"pairs":ratios.len(),"clean":clean,"ratio_ci95":ci,"verdict":verdict}));
            }
        }
    }
    println!("{}", serde_json::to_string_pretty(&rows).unwrap());
}

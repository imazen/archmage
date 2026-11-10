//! All retained paired rounds; no outlier filtering or best-of-run selection.
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    println!(
        "file\tcell\tbaseline\tcandidate\tadmitted\tpairs\tmean_ratio\tci_lower\tci_upper\tgate"
    );
    for file in std::env::args().skip(1) {
        let packet: zenbench::SuiteResult =
            serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
        for group in &packet.comparisons {
            let admitted = !packet.unreliable
                && group.samples.len() >= 50
                && group.samples.iter().all(|r| r.gate_clean == Some(true));
            for (baseline, candidate) in (0..group.benchmarks.len())
                .flat_map(|b| ((b + 1)..group.benchmarks.len()).map(move |c| (b, c)))
            {
                let ratios: Vec<f64> = group
                    .samples
                    .iter()
                    .map(|r| r.compensated_ns[candidate] as f64 / r.compensated_ns[baseline] as f64)
                    .collect();
                let ci = zenbench::MeanCi::from_samples(&ratios, 100_000).expect("paired rounds");
                let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
                let gate = if !admitted {
                    "INADMISSIBLE"
                } else if ci.upper <= 1.02 {
                    "PASS"
                } else if ci.lower > 1.02 {
                    "FAIL"
                } else {
                    "INCONCLUSIVE"
                };
                println!(
                    "{file}\t{}\t{}\t{}\t{admitted}\t{}\t{mean}\t{}\t{}\t{gate}",
                    group.group_name,
                    group.benchmarks[baseline].name,
                    group.benchmarks[candidate].name,
                    ratios.len(),
                    ci.lower,
                    ci.upper
                );
            }
        }
    }
}
#[cfg(target_arch = "wasm32")]
fn main() {
    panic!("analyzer requires a native host");
}

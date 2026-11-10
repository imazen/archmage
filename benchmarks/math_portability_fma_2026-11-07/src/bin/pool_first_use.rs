//! The single preregistered 200-pair extension, retaining every original pair.
#![forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let inputs: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(inputs.len(), 2, "original packet and one appended packet");
    let packets: Vec<zenbench::SuiteResult> = inputs
        .iter()
        .map(|path| serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
        .collect();
    for packet in &packets {
        assert_eq!(packet.comparisons.len(), 1);
        assert!(packet.comparisons[0].group_name.contains("/first-use/"));
    }
    let original = &packets[0].comparisons[0];
    let appended = &packets[1].comparisons[0];
    assert_eq!(original.group_name, appended.group_name);
    assert_eq!(original.benchmarks.len(), appended.benchmarks.len());
    for (a, b) in original.benchmarks.iter().zip(&appended.benchmarks) {
        assert_eq!(a.name, b.name);
    }
    let samples: Vec<_> = original.samples.iter().chain(&appended.samples).collect();
    assert_eq!(
        samples.len(),
        200,
        "one extension, no discarded or extra pairs"
    );
    let admitted = packets.iter().all(|p| !p.unreliable && !p.incomplete)
        && samples.iter().all(|s| s.gate_clean == Some(true));
    let mut comparisons = Vec::new();
    for (baseline, candidate) in [(0, 1), (0, 2), (1, 2)] {
        let ratios: Vec<f64> = samples
            .iter()
            .map(|s| s.compensated_ns[candidate] as f64 / s.compensated_ns[baseline] as f64)
            .collect();
        let ci = zenbench::MeanCi::from_samples(&ratios, 100_000).unwrap();
        let gate = if !admitted {
            "INADMISSIBLE"
        } else if ci.upper <= 1.02 {
            "PASS"
        } else if ci.lower > 1.02 {
            "FAIL"
        } else {
            "INCONCLUSIVE"
        };
        comparisons.push(serde_json::json!({"baseline":original.benchmarks[baseline].name,"candidate":original.benchmarks[candidate].name,"pairs":ratios.len(),"mean_ratio":ratios.iter().sum::<f64>()/ratios.len() as f64,"ci_lower":ci.lower,"ci_upper":ci.upper,"gate":gate}));
    }
    println!("{}",serde_json::to_string_pretty(&serde_json::json!({"inputs":inputs,"cell":original.group_name,"admitted":admitted,"original_pairs":original.samples.len(),"appended_pairs":appended.samples.len(),"comparisons":comparisons})).unwrap());
}
#[cfg(target_arch = "wasm32")]
fn main() {
    panic!("analysis requires a native host");
}

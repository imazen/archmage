//! Paired contrasts for each independently changed option axis.
#![forbid(unsafe_code)]
fn main() {
    for path in std::env::args().skip(1) {
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let result = zenbench::SuiteResult::load(std::path::Path::new(&path)).unwrap();
        let placement = raw["cpu_placement"]["status"].as_str().unwrap();
        let verified = matches!(placement, "uniform_hardware" | "enforced_uniform");
        for (gi, g) in result.comparisons.iter().enumerate() {
            let clean = !result.unreliable
                && !raw["incomplete"].as_bool().unwrap()
                && g.samples.len() == 50
                && g.samples.iter().all(|s| s.gate_clean == Some(true));
            let flags: Vec<_> = g
                .benchmarks
                .iter()
                .map(|b| {
                    b.name
                        .strip_prefix("bits")
                        .and_then(|s| s.split('/').next().unwrap().parse::<u32>().ok())
                })
                .collect();
            let find = |f| flags.iter().position(|&v| v == Some(f)).unwrap();
            let mut edges = Vec::new();
            for (i, &f) in flags.iter().enumerate() {
                if let Some(f) = f {
                    if f & 64 == 0 {
                        edges.push((find(f | 64), i, "LegacyFit/Canonical"));
                    }
                    if f & 16 == 0 {
                        edges.push((find(f | 16), i, "BackendDependent/BitExact"));
                    }
                    if f & 4 == 0 {
                        edges.push((find(f | 4), i, "CallerArray/AutomaticArray"));
                    }
                    if f & 33 == 0 {
                        edges.push((find(f | 32), i, "AssumeValid/Checked"));
                        edges.push((find(f | 1), i, "FullSpecials/FinitePositiveChecked"));
                    }
                }
            }
            let low = g.group_name.contains("/lowp/");
            let token = g.benchmarks.len() - 1;
            edges.push((
                token,
                find(if low { 44 } else { 36 }),
                "ExplicitTokenLaneMap/CallerArrayScalar",
            ));
            edges.push((
                token,
                find(if low { 40 } else { 32 }),
                "ExplicitTokenLaneMap/AutomaticArray",
            ));
            assert_eq!(edges.len(), 54);
            for (n, d, axis) in edges {
                let ratios: Vec<_> = g
                    .samples
                    .iter()
                    .map(|s| {
                        assert!(s.compensated_ns[d] > 0);
                        s.compensated_ns[n] as f64 / s.compensated_ns[d] as f64
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
                    serde_json::json!({"file":path,"group":g.group_name,"axis":axis,"implementation":g.benchmarks[n].name,"reference":g.benchmarks[d].name,
                    "implementation_mean_ns":raw["comparisons"][gi]["benchmarks"][n]["summary"]["mean"],"reference_mean_ns":raw["comparisons"][gi]["benchmarks"][d]["summary"]["mean"],
                    "pairs":ratios.len(),"resource_clean":clean,"placement":placement,"promotion_eligible":clean&&verified,"ratio_ci95":ci,"cost_bound_only":cost,"verdict":verdict})
                );
            }
        }
    }
}

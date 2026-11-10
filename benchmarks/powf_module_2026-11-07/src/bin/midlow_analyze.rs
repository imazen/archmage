//! All-retained paired intervals for named legacy and policy comparisons.
#![forbid(unsafe_code)]
fn main() {
    for path in std::env::args().skip(1) {
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let result = zenbench::SuiteResult::load(std::path::Path::new(&path)).unwrap();
        for (index, g) in result.comparisons.iter().enumerate() {
            let verified = matches!(
                raw["cpu_placement"]["status"].as_str(),
                Some("uniform_hardware" | "enforced_uniform")
            );
            let clean = !result.unreliable
                && !raw["incomplete"].as_bool().unwrap()
                && g.samples.len() == 50
                && g.samples.iter().all(|s| s.gate_clean == Some(true));
            let mut edges = Vec::new();
            for (i, b) in g
                .benchmarks
                .iter()
                .enumerate()
                .filter(|(_, b)| !b.name.starts_with("legacy/"))
            {
                for (j, _) in g.benchmarks.iter().enumerate().filter(|(_, v)| {
                    v.name.starts_with("legacy/")
                        && v.name.contains(if b.name.contains("/lowp/") {
                            "lowp"
                        } else {
                            "midp"
                        })
                }) {
                    edges.push((i, j, "replacement/legacy"));
                }
                for (from, to, edge) in [
                    ("/assume/", "/checked/", "AssumeValid/Checked"),
                    ("/positive-gamma/", "/gamma/", "PositiveGamma/Gamma"),
                    ("/lowp/", "/midp/", "LOWP/MIDP"),
                    ("backend/", "portable/", "BackendDependent/BitExact"),
                    ("/caller", "/auto", "Caller/Automatic"),
                    ("/gamma/", "/nonnegative/", "Gamma/NonNegativeNormal"),
                ] {
                    if b.name.contains(from) {
                        let target = b.name.replace(from, to);
                        if let Some(j) = g.benchmarks.iter().position(|v| v.name == target) {
                            edges.push((i, j, edge));
                            if edge == "BackendDependent/BitExact" {
                                edges.push((j, i, "BitExact/BackendDependent"));
                            }
                        }
                    }
                }
            }
            for (i, j, edge) in edges {
                let ratios: Vec<_> = g
                    .samples
                    .iter()
                    .map(|s| {
                        assert!(s.compensated_ns[j] > 0);
                        s.compensated_ns[i] as f64 / s.compensated_ns[j] as f64
                    })
                    .collect();
                let ci = zenbench::MeanCi::from_samples(&ratios, 100_000).unwrap();
                let deltas: Vec<_> = g
                    .samples
                    .iter()
                    .map(|s| {
                        (s.compensated_ns[i] as f64 - s.compensated_ns[j] as f64)
                            / s.iterations as f64
                    })
                    .collect();
                let delta = zenbench::MeanCi::from_samples(&deltas, 100_000).unwrap();
                println!(
                    "{}",
                    serde_json::json!({"file":path,"group":g.group_name,"edge":edge,"implementation":g.benchmarks[i].name,"reference":g.benchmarks[j].name,"mean_ns":raw["comparisons"][index]["benchmarks"][i]["summary"]["mean"],"reference_mean_ns":raw["comparisons"][index]["benchmarks"][j]["summary"]["mean"],"ci95":ci,"delta_ns_ci95":delta,"clean":clean,"verified":verified,"qualified":clean&&verified,"cost":if ci.upper<=1.02{"PASS"}else if ci.lower>1.02{"FAIL"}else{"INCONCLUSIVE"}})
                );
            }
        }
    }
}

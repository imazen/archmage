#![forbid(unsafe_code)]
use std::{fs::File, io::Write};
use zenbench::{MeanCi, SuiteResult, Summary, Throughput};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let suite = SuiteResult::load(&args[1]).unwrap();
    assert!(!suite.unreliable && !suite.incomplete);
    let mut out = File::create(&args[2]).unwrap();
    writeln!(
        out,
        "group,baseline,candidate,rounds,baseline_mean,candidate_mean,diff,lo,hi,dirty,eligible"
    )
    .unwrap();
    for g in &suite.comparisons {
        let scale = match g.throughput {
            Some(Throughput::Elements(n)) => n as f64,
            _ => panic!("missing throughput"),
        };
        let mut pairs = vec![
            (
                "old-ordinary/correct".to_owned(),
                "ordinary/correct".to_owned(),
            ),
            ("std/control".to_owned(), "ordinary/correct".to_owned()),
            ("std/control".to_owned(), "current/correct".to_owned()),
        ];
        for b in &g.benchmarks {
            if let Some(policy) = b.name.strip_prefix("current/") {
                pairs.push((format!("old-scalar/{policy}"), b.name.clone()));
            }
            if let Some(rest) = b.name.strip_prefix("packed-") {
                pairs.push((format!("old-packed-{rest}"), b.name.clone()));
            }
            if let Some(rest) = b.name.strip_prefix("direct-") {
                pairs.push((format!("old-direct-{rest}"), b.name.clone()));
                if let Some((tier, _)) = rest.split_once('/') {
                    pairs.push((format!("std-context-{tier}/control"), b.name.clone()));
                }
            }
        }
        for (b, c) in pairs {
            let Some(ib) = g.benchmarks.iter().position(|x| x.name == b) else {
                panic!("missing pair {b}")
            };
            let ic = g.benchmarks.iter().position(|x| x.name == c).unwrap();
            let mut bs = Vec::new();
            let mut cs = Vec::new();
            let mut ds = Vec::new();
            let mut dirty = 0;
            for r in &g.samples {
                assert!(r.iterations > 0);
                let x = r.compensated_ns[ib] as f64 / r.iterations as f64 / scale;
                let y = r.compensated_ns[ic] as f64 / r.iterations as f64 / scale;
                bs.push(x);
                cs.push(y);
                ds.push(y - x);
                dirty += usize::from(r.gate_clean != Some(true));
            }
            assert!(ds.len() >= 30);
            let ci = MeanCi::from_samples(&ds, 10000).unwrap();
            writeln!(
                out,
                "{},{},{},{},{},{},{},{},{},{},{}",
                g.group_name,
                b,
                c,
                ds.len(),
                Summary::from_slice(&bs).mean,
                Summary::from_slice(&cs).mean,
                Summary::from_slice(&ds).mean,
                ci.lower,
                ci.upper,
                dirty,
                dirty == 0
            )
            .unwrap();
        }
    }
    println!(
        "PASS {} groups; all-round identical pairs, no extra exclusions, dirty groups rejected",
        suite.comparisons.len()
    );
}

//! Direct matched differences using Zenbench's public bootstrap API.
#![forbid(unsafe_code)]
use std::{
    fs::{self, File},
    io::Write,
    path::Path,
};
use zenbench::{MeanCi, SuiteResult, Summary, Throughput};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let suite = SuiteResult::load(&args[1]).unwrap();
    assert!(!suite.incomplete && !suite.unreliable);
    let dest = Path::new(&args[2]);
    fs::create_dir_all(dest).unwrap();
    let mut direct = File::create(dest.join("direct-pairs.csv")).unwrap();
    let mut original = File::create(dest.join("zenbench-pairs.csv")).unwrap();
    let mut marginal = File::create(dest.join("marginals.csv")).unwrap();
    writeln!(direct,"group,baseline,candidate,rounds,baseline_mean,candidate_mean,diff_mean,ci_lo,ci_hi,quantization_floor,resolution_limited,gate_dirty,policy_eligible,bootstrap_resamples").unwrap();
    writeln!(original,"group,baseline,candidate,retained_n,outliers,baseline_mean,candidate_mean,diff_mean,ci_lo,ci_hi,percent,drift,resolution_limited").unwrap();
    writeln!(marginal, "group,candidate,rounds,mean,median,mad,cv").unwrap();
    let mut cells = 0;
    let mut comparisons = 0;
    let mut limited = 0;
    let mut low_retained = 0;
    for g in &suite.comparisons {
        let scale = match g.throughput {
            Some(Throughput::Elements(n)) => n as f64,
            _ => panic!("missing explicit throughput: {}", g.group_name),
        };
        assert!(g.completed_rounds >= 30);
        cells += g.benchmarks.len();
        let names: Vec<_> = g.benchmarks.iter().map(|b| b.name.as_str()).collect();
        for b in &g.benchmarks {
            writeln!(
                marginal,
                "{},{},{},{},{},{},{}",
                g.group_name,
                b.name,
                b.summary.n,
                b.summary.mean / scale,
                b.summary.median / scale,
                b.summary.mad / scale,
                b.summary.variance.sqrt() / b.summary.mean
            )
            .unwrap();
        }
        for (b, c, a) in &g.analyses {
            low_retained += usize::from(a.diff.n < 30);
            comparisons += 1;
            limited += usize::from(a.resolution_limited);
            writeln!(
                original,
                "{},{},{},{},{},{},{},{},{},{},{},{},{}",
                g.group_name,
                b,
                c,
                a.diff.n,
                a.n_outliers,
                a.baseline.mean / scale,
                a.candidate.mean / scale,
                a.diff.mean / scale,
                a.ci_lower / scale,
                a.ci_upper / scale,
                a.pct_change,
                a.drift_correlation,
                a.resolution_limited
            )
            .unwrap();
        }
        let mut pairs = vec![
            ("branch/baseline", "cached/baseline"),
            ("reference/baseline", "branch/baseline"),
            ("reference/baseline", "cached/baseline"),
        ];
        if names.contains(&"ordinary/baseline") {
            pairs.push(("branch/baseline", "ordinary/baseline"));
        }
        for name in &names {
            if name.starts_with("packed8/") {
                if let Some(&p) = names.iter().find(|n| n.starts_with("packed4/v4x")) {
                    pairs.push((p, name));
                }
            }
        }
        for (b, c) in pairs {
            let ib = names
                .iter()
                .position(|n| *n == b)
                .expect("missing baseline");
            let ic = names
                .iter()
                .position(|n| *n == c)
                .expect("missing candidate");
            let mut bs = Vec::new();
            let mut cs = Vec::new();
            let mut ds = Vec::new();
            let mut iters = Vec::new();
            let mut dirty = 0;
            for r in &g.samples {
                assert!(r.iterations > 0);
                let x = r.compensated_ns[ib] as f64 / r.iterations as f64 / scale;
                let y = r.compensated_ns[ic] as f64 / r.iterations as f64 / scale;
                bs.push(x);
                cs.push(y);
                ds.push(y - x);
                iters.push(r.iterations);
                dirty += usize::from(r.gate_clean != Some(true));
            }
            assert!(ds.len() >= 30);
            let policy_eligible = dirty == 0;
            if !policy_eligible {
                eprintln!(
                    "REJECTED policy group {}: {dirty} dirty rounds; retaining all raw rounds for diagnostic CI",
                    g.group_name
                );
            }
            let ci = MeanCi::from_samples(&ds, 10000).expect("paired bootstrap");
            let bd = Summary::from_slice(&bs);
            let cd = Summary::from_slice(&cs);
            let dd = Summary::from_slice(&ds);
            iters.sort_unstable();
            let floor =
                2. * suite.timer_resolution_ns as f64 / iters[iters.len() / 2] as f64 / scale;
            writeln!(
                direct,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},10000",
                g.group_name,
                b,
                c,
                ds.len(),
                bd.mean,
                cd.mean,
                dd.mean,
                ci.lower,
                ci.upper,
                floor,
                dd.mean.abs() < floor,
                dirty,
                policy_eligible
            )
            .unwrap();
        }
    }
    println!(
        "PASS groups={} cells={cells} original_comparisons={comparisons} original_resolution_limited={limited} original_below30={low_retained}; direct pairs use all identical rounds, no new exclusions",
        suite.comparisons.len()
    );
}

//! Direct paired MeanCi with every retained round; no additional exclusions.
#[cfg(target_arch = "x86_64")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::{fs::File, io::Write};
    use zenbench::{MeanCi, SuiteResult, Summary};
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: paired INPUT.json OUTPUT.csv CANDIDATE|ALL".into());
    }
    let suite = SuiteResult::load(&args[1])?;
    if suite.unreliable {
        return Err("unreliable suite".into());
    }
    if suite.comparisons.is_empty() {
        return Err("empty comparison list".into());
    }
    for g in &suite.comparisons {
        let scale = g
            .group_name
            .split('/')
            .next_back()
            .ok_or("group length")?
            .parse::<f64>()?;
        if !scale.is_finite() || scale <= 0.0 {
            return Err("nonfinite/nonpositive scale".into());
        }
        if g.benchmarks.is_empty() {
            return Err("empty benchmark group".into());
        }
        if g.samples.iter().any(|r| {
            r.compensated_ns.len() != g.benchmarks.len()
                || r.elapsed_ns.len() != g.benchmarks.len()
                || r.execution_order.len() != g.benchmarks.len()
        }) {
            return Err("sample width mismatch".into());
        }
    }
    let mut out = File::create_new(&args[2])?;
    writeln!(
        out,
        "group,baseline,candidate,rounds,baseline_mean_ns_element,candidate_mean_ns_element,time_over_std,diff_mean_ns_element,ci95_lower,ci95_upper,builtin_excluded_pairs,timer_floor_ns_element,method"
    )?;
    for group in &suite.comparisons {
        let names: Vec<_> = group.benchmarks.iter().map(|b| b.name.as_str()).collect();
        let base = names
            .iter()
            .position(|n| *n == "std")
            .ok_or("std baseline missing")?;
        let scale = group
            .group_name
            .split('/')
            .next_back()
            .ok_or("group length")?
            .parse::<f64>()?;
        let candidates: Vec<_> = if args[3] == "ALL" {
            names.iter().copied().filter(|n| *n != "std").collect()
        } else {
            vec![args[3].as_str()]
        };
        if group.samples.len() != group.completed_rounds
            || group.samples.len() < 50
            || group.samples.iter().any(|r| r.gate_clean != Some(true))
        {
            return Err("round/gate failure".into());
        }
        for name in candidates {
            let candidate = names
                .iter()
                .position(|n| *n == name)
                .ok_or("candidate missing")?;
            let mut bs = vec![];
            let mut cs = vec![];
            let mut ds = vec![];
            let mut iters = vec![];
            for sample in &group.samples {
                if sample.iterations == 0 {
                    return Err("zero sample iterations".into());
                }
                let b = sample.compensated_ns[base] as f64 / sample.iterations as f64 / scale;
                let c = sample.compensated_ns[candidate] as f64 / sample.iterations as f64 / scale;
                bs.push(b);
                cs.push(c);
                ds.push(c - b);
                iters.push(sample.iterations);
            }
            let ci = MeanCi::from_samples(&ds, 10000).ok_or("bootstrap")?;
            let b = Summary::from_slice(&bs);
            let c = Summary::from_slice(&cs);
            let d = Summary::from_slice(&ds);
            let trimmed = group
                .analyses
                .iter()
                .find(|(a, z, _)| a == "std" && z == name)
                .ok_or("built-in analysis missing")?
                .2
                .n_outliers;
            iters.sort_unstable();
            let floor =
                2.0 * suite.timer_resolution_ns as f64 / iters[iters.len() / 2] as f64 / scale;
            writeln!(
                out,
                "{},std,{},{},{},{},{},{},{},{},{},{},all retained rounds; paired MeanCi10000; no exclusions",
                group.group_name,
                name,
                ds.len(),
                b.mean,
                c.mean,
                c.mean / b.mean,
                d.mean,
                ci.lower,
                ci.upper,
                trimmed,
                floor
            )?;
            println!(
                "{} {} all{}: delta={} CI[{},{}] ratio={}",
                group.group_name,
                name,
                ds.len(),
                d.mean,
                ci.lower,
                ci.upper,
                c.mean / b.mean
            );
        }
    }
    Ok(())
}
#[cfg(not(target_arch = "x86_64"))]
fn main() {
    panic!("Timing analysis is native x86; numerical kernel is independent")
}

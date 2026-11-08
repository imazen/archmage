//! Named implementations, ns per buffer, all-retained paired confidence bounds.
#![forbid(unsafe_code)]
fn main() {
    for path in std::env::args().skip(1) {
        let raw:serde_json::Value=serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let result=zenbench::SuiteResult::load(std::path::Path::new(&path)).unwrap();
        let placement=raw["cpu_placement"]["status"].as_str().unwrap();
        let verified=matches!(placement,"uniform_hardware"|"enforced_uniform");
        for (index,group) in result.comparisons.iter().enumerate() {
            let clean=!result.unreliable&&!raw["incomplete"].as_bool().unwrap()&&group.samples.len()==50&&group.samples.iter().all(|r|r.gate_clean==Some(true));
            for (numerator,denominator) in [(0,1),(2,3),(0,5),(1,5),(2,4),(3,4),(0,6),(1,6),(2,6),(3,6)] {
                let ratios:Vec<_>=group.samples.iter().map(|r|{
                    assert!(r.compensated_ns[denominator]>0);
                    r.compensated_ns[numerator]as f64/r.compensated_ns[denominator]as f64
                }).collect();
                let ci=zenbench::MeanCi::from_samples(&ratios,100_000).unwrap();
                let cost=if ci.upper<=1.02{"PASS"}else if ci.lower>1.02{"FAIL"}else{"INCONCLUSIVE"};
                let verdict=if !verified{"EXPLORATORY"}else if !clean{"INADMISSIBLE"}else{cost};
                println!("{}",serde_json::json!({"file":path,"group":group.group_name,
                    "candidate":group.benchmarks[numerator].name,"baseline":group.benchmarks[denominator].name,
                    "numerator_mean_ns":raw["comparisons"][index]["benchmarks"][numerator]["summary"]["mean"],
                    "denominator_mean_ns":raw["comparisons"][index]["benchmarks"][denominator]["summary"]["mean"],
                    "pairs":ratios.len(),"resource_clean":clean,"placement":placement,
                    "promotion_eligible":clean&&verified,"ratio_ci95":ci,"cost_bound_only":cost,"verdict":verdict}));
            }
        }
    }
}

//! All-retained paired intervals against explicitly named legacy methods.
#![forbid(unsafe_code)]
fn main() {
 for path in std::env::args().skip(1) {
  let raw:serde_json::Value=serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
  let result=zenbench::SuiteResult::load(std::path::Path::new(&path)).unwrap();
  for (index,g) in result.comparisons.iter().enumerate(){
   let verified=matches!(raw["cpu_placement"]["status"].as_str(),Some("uniform_hardware"|"enforced_uniform"));
   let clean=!result.unreliable && g.samples.len()==50 && g.samples.iter().all(|s|s.gate_clean==Some(true));
   for (i,b) in g.benchmarks.iter().enumerate().filter(|(_,b)|!b.name.starts_with("legacy/")) {
    for (j,legacy) in g.benchmarks.iter().enumerate().filter(|(_,b)|b.name.starts_with("legacy/") && b.name.contains(if g.benchmarks[i].name.contains("/lowp/"){"lowp"}else{"midp"})) {
     let ratios:Vec<_>=g.samples.iter().map(|s|s.compensated_ns[i]as f64/s.compensated_ns[j]as f64).collect();
     let ci=zenbench::MeanCi::from_samples(&ratios,100_000).unwrap();
     println!("{}",serde_json::json!({"group":g.group_name,"implementation":b.name,"legacy":legacy.name,"mean_ns":raw["comparisons"][index]["benchmarks"][i]["summary"]["mean"],"legacy_mean_ns":raw["comparisons"][index]["benchmarks"][j]["summary"]["mean"],"ci95":ci,"clean":clean,"verified":verified,"qualified":clean&&verified,"cost":if ci.upper<=1.02{"PASS"}else if ci.lower>1.02{"FAIL"}else{"INCONCLUSIVE"}}));
    }
   }
  }
 }
}

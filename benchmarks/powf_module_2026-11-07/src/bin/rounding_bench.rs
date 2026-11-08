//! Controlled fused/separate Gamma study. See NONFUSED-PLAN.md.
#![forbid(unsafe_code)]
use archmage::SimdToken;
const LABELS: [&str; 7] = ["MIDP_separate", "MIDP_fused", "LOWP_separate", "LOWP_fused",
    "legacy_pow_lowp", "legacy_pow_midp", "std_f32_powf"];
struct Cell { name:String, width:usize, context:&'static str, size:usize, chain:bool, exponent:u32, checked:bool }
fn normalize(x:f32)->f32 { f32::from_bits(0x3f000000|(x.to_bits()&0x007fffff)) }
include!(concat!(env!("OUT_DIR"),"/rounding_bench.rs"));
fn cells()->Vec<Cell> {
    let native=if cfg!(target_arch="aarch64"){"neon"}else{"v3"};
    let mut cells=Vec::new();
    for width in [4,8,16] { for context in ["scalar",native] { for size in [64,4096] { for chain in [false,true] { for exponent in [0x4019999a,0x3f000000] { for checked in [true,false] {
        let name=format!("rounding/x{width}/{context}/{size}/{}/Gamma/{exponent:08x}/{}",if chain{"chain"}else{"stream"},if checked{"checked"}else{"assumed"});
        cells.push(Cell{name,width,context,size,chain,exponent,checked});
    }}}}}}
    assert_eq!(cells.len(),96); cells
}
fn register_all(mut suite:Option<&mut zenbench::Suite>) {
    for cell in cells() {
        if std::env::var("POWF_FILTER").is_ok_and(|f|f!=cell.name) {continue;}
        match cell.width {4=>dispatch4(suite.as_deref_mut(),&cell),8=>dispatch8(suite.as_deref_mut(),&cell),16=>dispatch16(suite.as_deref_mut(),&cell),_=>unreachable!()}
    }
}
fn main(){
    if std::env::var("POWF_LIST").is_ok(){for cell in cells(){println!("{}",cell.name);}return;}
    if std::env::var("POWF_VERIFY_ONLY").is_ok(){register_all(None);println!("all 96 cells verified");return;}
    assert!(!std::env::args().any(|a|a=="--no-busy-gate"));
    assert!(std::env::var("ZENBENCH_NO_BUSY_GATE").map_or(true,|v|v.is_empty()||v=="0"));
    let output=std::env::var("POWF_RESULT").expect("new POWF_RESULT required");
    assert!(!std::path::Path::new(&output).exists());
    zenbench::run_gated(zenbench::GateConfig::strict(),|s|register_all(Some(s))).save(std::path::Path::new(&output)).unwrap();
}

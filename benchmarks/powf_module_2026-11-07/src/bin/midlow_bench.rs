//! Actual bounded public methods, including token-preserving single-lane tails.
#![forbid(unsafe_code)]
use archmage::SimdToken;
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/midlow_generated.rs"));
fn normalize(x: f32) -> f32 {
    f32::from_bits(0x3f000000 | (x.to_bits() & 0x007fffff))
}
fn main() {
    if std::env::var_os("MIDLOW_VERIFY").is_some() {
        verify();
        return;
    }
    let output = std::env::var("POWF_RESULT").expect("POWF_RESULT");
    assert!(!std::path::Path::new(&output).exists());
    assert!(std::env::var("ZENBENCH_NO_BUSY_GATE").map_or(true, |v| v.is_empty() || v == "0"));
    zenbench::run_gated(zenbench::GateConfig::strict(), register)
        .save(std::path::Path::new(&output))
        .unwrap();
}

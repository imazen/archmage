#![forbid(unsafe_code)]
pub type Kernel<T> = Box<dyn Fn(&[T], &[T], &mut [T]) + Send>;
pub struct Case<T> {
    pub group: &'static str,
    pub name: &'static str,
    pub run: Kernel<T>,
}
mod generated;
pub use generated::{cases_f32, cases_f64};
pub mod fixtures;
pub fn inputs_f32(op: &str) -> (Vec<f32>, Vec<f32>) {
    (
        (0..1024)
            .map(|i| {
                if op.starts_with("rem") {
                    -1024. + 2. * ((i * 719) % 1024) as f32
                } else {
                    -0.5 + ((i * 719) % 1024) as f32 / 1024.
                }
            })
            .collect(),
        (0..1024)
            .map(|i| 0.125 + ((i * 313) % 1024) as f32 / 512.)
            .collect(),
    )
}
pub fn inputs_f64(op: &str) -> (Vec<f64>, Vec<f64>) {
    (
        (0..1024)
            .map(|i| {
                if op.starts_with("rem") {
                    -1024. + 2. * ((i * 719) % 1024) as f64
                } else {
                    -0.5 + ((i * 719) % 1024) as f64 / 1024.
                }
            })
            .collect(),
        (0..1024)
            .map(|i| 0.125 + ((i * 313) % 1024) as f64 / 512.)
            .collect(),
    )
}
pub fn tier_inventory() {
    use archmage::prelude::*;
    let v = [
        ("scalar", ScalarToken::summon().is_some()),
        ("v3", X64V3Token::summon().is_some()),
        ("v4x", X64V4xToken::summon().is_some()),
        ("neon", NeonToken::summon().is_some()),
        ("wasm128", Wasm128Token::summon().is_some()),
    ];
    if let Ok(s) = std::env::var("ALG_EXT_REQUIRE_TIERS") {
        for e in s.split(',') {
            assert!(
                v.iter().any(|(t, ok)| *t == e && *ok),
                "required tier unavailable: {e}"
            )
        }
    }
    for (t, ok) in v {
        println!("TIER {t} supported={ok}")
    }
}
#[test]
fn exact_fixtures() {
    tier_inventory();
    assert!(fixtures::check_all() >= 60);
}

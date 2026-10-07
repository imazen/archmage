#![forbid(unsafe_code)]
use archmage::{autoversion, magetypes};

#[autoversion]
pub fn auto(x: u32) -> u32 {
    x + 1
}

#[magetypes]
pub fn family(_token: Token, x: u32) -> u32 {
    x + 1
}

#[autoversion(cfg(simd_opt), v3, scalar)]
pub fn fallback_when_disabled(x: u32) -> u32 {
    x + 1
}

#[cfg(feature = "simd_opt")]
#[autoversion(v3, scalar)]
pub fn omitted_when_disabled(x: u32) -> u32 {
    x + 1
}

pub fn referenced_outputs() {
    #[cfg(feature = "check_auto_v4")]
    let _: fn(archmage::X64V4Token, u32) -> u32 = auto_v4;
    #[cfg(feature = "check_family_v4")]
    let _: fn(archmage::X64V4Token, u32) -> u32 = family_v4;
    #[cfg(feature = "require_omitted")]
    let _ = omitted_when_disabled(1);
}

#[test]
fn dispatcher_survives_disabled_simd_gate() {
    assert_eq!(fallback_when_disabled(10), 11);
    assert_eq!(auto(10), 11);
}

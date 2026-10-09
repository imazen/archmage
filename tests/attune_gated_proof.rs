#![forbid(unsafe_code)]
#![deny(warnings)]

#[archmage::attune(_scalar)]
fn leaf(x: u32) -> u32 {
    x
}

// No foreign or feature-disabled leaf is declared. Successful compilation
// proves those candidates cannot demand a proof or an existing callee.
#[cfg(not(feature = "avx512"))]
#[archmage::attune(scalar)]
fn disabled(a: archmage::ScalarToken, b: archmage::ScalarToken, x: u32) -> u32 {
    let _ = (a, b);
    archmage::attuned!(leaf(x), [_v4x(cfg(avx512)), _scalar])
}

#[cfg(not(target_arch = "aarch64"))]
#[archmage::attune(scalar)]
fn foreign(a: archmage::ScalarToken, b: archmage::ScalarToken, x: u32) -> u32 {
    let _ = (a, b);
    archmage::attuned!(leaf(x), [_neon, _scalar])
}

#[cfg(target_arch = "aarch64")]
#[archmage::attune(scalar)]
fn foreign(a: archmage::ScalarToken, b: archmage::ScalarToken, x: u32) -> u32 {
    let _ = (a, b);
    archmage::attuned!(leaf(x), [_v3_t, _scalar])
}

#[test]
fn inactive_candidates_do_not_require_parent_proof_selection() {
    #[cfg(not(feature = "avx512"))]
    assert_eq!(disabled(archmage::ScalarToken, archmage::ScalarToken, 7), 7);
    assert_eq!(foreign(archmage::ScalarToken, archmage::ScalarToken, 8), 8);
}

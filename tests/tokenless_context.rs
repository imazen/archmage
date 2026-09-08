//! A tokenless rite caller can prove covered token-first callees without detection.
#![forbid(unsafe_code)]
#![allow(unused_imports, dead_code)]
use archmage::prelude::*;

#[magetypes(rite, v3, neon, wasm128, scalar)]
fn add<const N: usize>(_token: Token, values: &[u32; N]) -> u32 {
    values.iter().sum()
}

#[rite(v3, neon, wasm128, scalar)]
fn inner<const N: usize>(values: &[u32; N]) -> u32 {
    incant!(add::<N>(values), [v3, neon, wasm128, scalar])
}

#[magetypes(v3, neon, wasm128, scalar)]
fn entry(_token: Token, values: &[u32; 3]) -> u32 {
    incant!(inner::<3>(values) without token)
}

#[test]
fn complete_multitier_chain() {
    assert_eq!(incant!(entry(&[1, 2, 3]), [v3, neon, wasm128, scalar]), 6);
    assert_eq!(inner_scalar(&[4, 5]), 9);
}

#[rite]
fn lower_v2(_token: X64V2Token, value: u32) -> u32 {
    value + 1
}
#[rite(v3)]
fn downgrade(value: u32) -> u32 {
    incant!(lower(value), [v2, -scalar])
}

// Stronger and unrelated candidates are intentionally absent: no code should
// reference them or try to summon their tokens from this tokenless context.
#[rite(v3)]
fn skip_upgrades(value: u32) -> u32 {
    incant!(lower(value), [v4, v3_crypto, v2, -scalar])
}
#[rite(v3)]
fn explicit_position(value: u32) -> u32 {
    dispatch_variant!(position(value, Token), [v3, -scalar])
}
#[rite]
fn position_v3(value: u32, _token: X64V3Token) -> u32 {
    value + 2
}
#[arcane]
fn check_x86(_token: X64V3Token) {
    assert_eq!(downgrade(4), 5);
    assert_eq!(skip_upgrades(4), 5);
    let calls = core::cell::Cell::new(0);
    assert_eq!(
        explicit_position({
            calls.set(calls.get() + 1);
            4
        }),
        6
    );
    assert_eq!(calls.get(), 1);
    assert_eq!(gated(4), if cfg!(feature = "std") { 14 } else { 5 });
    assert_eq!(fallback(4), 24);
}
#[cfg(feature = "std")]
#[rite]
fn lower_v3(_token: X64V3Token, value: u32) -> u32 {
    value + 10
}
#[rite(v3)]
fn gated(value: u32) -> u32 {
    incant!(lower(value), [v3(cfg(std)), v2, -scalar])
}
fn missing_default(value: u32) -> u32 {
    value + 20
}
#[rite(v3)]
fn fallback(value: u32) -> u32 {
    incant!(missing(value), [v4, default])
}
#[test]
fn covered_tiers_only() {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = X64V3Token::summon() {
        check_x86(token);
    }
}

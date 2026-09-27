//! Published calling conventions: caller proofs, callee tokens, and fallbacks.
#![forbid(unsafe_code)]
#![allow(deprecated)]
use archmage::{ScalarToken, arcane, autoversion, incant, magetypes, rite};

#[magetypes(rite, v3, neon, wasm128, scalar)]
fn proof_leaf<const N: usize>(x: [u32; N], _proof: Token) -> u32 {
    x.iter().sum::<u32>() + 1
}

#[rite(v3, neon, wasm128, scalar, default)]
fn plain_leaf<const N: usize>(x: [u32; N]) -> u32 {
    x.iter().sum::<u32>() + 2
}

// Tokenless -> tokenful and tokenless -> tokenless; explicit token placement.
#[rite(v3, neon, wasm128, scalar, default)]
fn plain_caller<const N: usize>(x: [u32; N]) -> (u32, u32) {
    (
        incant!(proof_leaf::<N>(x, Token), [v3, neon, wasm128, scalar]),
        incant!(plain_leaf::<N>(x) without token),
    )
}

// Tokenful scalar fallbacks retain runtime dispatch, so their candidates
// must be safe boundaries rather than bare rite functions.
#[magetypes(v3, neon, wasm128, scalar)]
fn boundary_leaf<const N: usize>(x: [u32; N], _proof: Token) -> u32 {
    x.iter().sum::<u32>() + 1
}

// Tokenful -> tokenful and tokenful -> tokenless, through nested dispatch.
#[magetypes(v3, neon, wasm128, scalar)]
fn boundary<const N: usize>(_proof: Token, x: [u32; N]) -> (u32, u32) {
    let direct = incant!(boundary_leaf::<N>(x, Token), [v3, neon, wasm128, scalar]);
    let nested = incant!(plain_caller::<N>(x) without token);
    assert_eq!(direct, nested.0);
    nested
}

#[autoversion(v3, neon, wasm128)]
fn automatic(x: [u32; 3]) -> (u32, u32) {
    incant!(plain_caller::<3>(x) without token)
}

#[autoversion(v3, neon, wasm128)]
fn automatic_legacy(_proof: SimdToken, x: [u32; 3]) -> (u32, u32) {
    incant!(plain_caller::<3>(x) without token)
}

#[arcane]
fn scalar_boundary(_proof: ScalarToken, x: [u32; 3]) -> (u32, u32) {
    incant!(plain_caller::<3>(x) without token)
}

#[test]
fn published_abis_and_nested_calls() {
    let x = [2, 3, 4];
    let expected = (10, 11);
    // Taking these fn pointers pins the actual public signatures.
    let scalar: fn(ScalarToken, [u32; 3]) -> (u32, u32) = boundary_scalar::<3>;
    let default: fn([u32; 3]) -> (u32, u32) = plain_caller_default::<3>;
    assert_eq!(scalar(ScalarToken, x), expected);
    assert_eq!(default(x), expected);
    assert_eq!(scalar_boundary(ScalarToken, x), expected);
    assert_eq!(automatic(x), expected);
    assert_eq!(automatic_legacy(x), expected);
    assert_eq!(automatic_scalar(ScalarToken, x), expected);
    assert_eq!(automatic_legacy_scalar(ScalarToken, x), expected);
    assert_eq!(
        incant!(boundary::<3>(x), [v3, neon, wasm128, scalar]),
        expected
    );
    // Ordinary incant supplies scalar's token and removes default's marker.
    assert_eq!(incant!(proof_leaf::<3>(x, Token), [scalar]), 10);
    assert_eq!(incant!(plain_leaf::<3>(x, Token), [default]), 11);
}

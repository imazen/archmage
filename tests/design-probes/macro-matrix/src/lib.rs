//! Compile-only capability probes. Select error cases individually; see the guide.
#![forbid(unsafe_code)]
#![allow(dead_code)]

#[archmage::autoversion(v4(cfg(avx512)), v3, neon, wasm128, scalar, use(f32xN))]
pub fn portable() -> usize {
    f32xN::splat(1.0).to_array().len()
}

#[archmage::rite(v3, use(f32xN))]
fn generic_helper<T: Copy, const N: usize>(values: [T; N]) -> ([T; N], usize) {
    (values, f32xN::splat(1.0).to_array().len())
}

#[archmage::magetypes(rite, use(f32xN), v3, scalar)]
fn helper() -> usize {
    f32xN::LANES
}

#[cfg(feature = "ungated-autoversion")]
#[archmage::autoversion(use(f32xN))]
fn defaults() -> usize {
    f32xN::splat(1.0).to_array().len()
}

#[cfg(feature = "signature-alias")]
#[archmage::rite(v3, use(f32xN))]
fn signature(value: f32xN) -> f32xN {
    value
}

#[cfg(feature = "tokenless-arcane")]
#[archmage::arcane(v3, use(f32xN))]
fn boundary() -> usize {
    f32xN::LANES
}

#[cfg(feature = "magetypes-without-token")]
#[archmage::magetypes(use(f32xN), v3, scalar)]
fn missing_proof() -> usize {
    f32xN::LANES
}

#[cfg(feature = "rite-placeholder")]
#[archmage::rite(v3, use(f32xN))]
fn placeholder(_: Token) -> usize {
    f32xN::LANES
}

#[cfg(feature = "rite-tier-gate")]
#[archmage::rite(v4(cfg(avx512)), v3, scalar, use(f32xN))]
fn gated_helper() -> usize {
    f32xN::LANES
}

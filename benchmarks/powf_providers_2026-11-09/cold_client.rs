//! Identical legacy consumer, plus explicitly labeled opt-in policy variants.
#![forbid(unsafe_code)]
use std::hint::black_box;
fn main() {
    let x = black_box([0.25_f32, 0.5, 0.75, 1.0]);
    let n = black_box(2.4_f32);
    let v = magetypes::simd::generic::f32x4::from_array_t(archmage::ScalarToken, x);
    black_box(v.pow_midp(n));
    black_box(v.pow_lowp(n));
    black_box(x.map(|v| v.powf(n)));
    #[cfg(feature = "const_policy")]
    black_box(magetypes::powf::try_powf_array_with::<
        { magetypes::powf::PowfOptions::PORTABLE_GAMMA_MIDP.bits() },
        4,
    >(x, n).unwrap());
    #[cfg(feature = "runtime_policy")]
    black_box(magetypes::powf::try_powf_array_runtime(
        x, n, black_box(magetypes::powf::PowfOptions::PORTABLE_GAMMA_MIDP),
    ).unwrap());
}

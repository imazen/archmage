#![forbid(unsafe_code)]
use magemath::{PowBehavior as B, PowRange, PowSign};
#[inline(never)]
fn legacy(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
    magemath::portable::array::pow_f32x8(a, b)
}
#[inline(never)]
fn default_policy(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
    magemath::portable::array::pow_f32x8_with::<0>(a, b)
}
#[inline(never)]
fn precondition(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
    magemath::portable::array::pow_f32x8_with::<
        { B::MIDP.range(PowRange::PositiveNormalPrecondition).to_u64() },
    >(a, b)
}
#[inline(never)]
fn asserted_precondition(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
    magemath::portable::array::pow_f32x8_with::<
        {
            B::MIDP
                .range(PowRange::PositiveNormalPrecondition)
                .debug_assertions(true)
                .to_u64()
        },
    >(a, b)
}
#[cfg(feature = "probe-extra")]
#[inline(never)]
fn signed(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
    magemath::portable::array::pow_f32x8_with::<
        { B::CORRECT.sign(PowSign::SignedMagnitude).to_u64() },
    >(a, b)
}
fn main() {
    std::hint::black_box([
        legacy as fn([f32; 8], [f32; 8]) -> [f32; 8],
        default_policy,
        precondition,
        asserted_precondition,
    ]);
    let a = std::hint::black_box([0.5; 8]);
    let b = std::hint::black_box([2.4; 8]);
    assert_eq!(
        std::hint::black_box(legacy(a, b)),
        std::hint::black_box(default_policy(a, b))
    );
    assert_eq!(
        std::hint::black_box(precondition(a, b)),
        std::hint::black_box(asserted_precondition(a, b))
    );
    #[cfg(feature = "probe-extra")]
    std::hint::black_box(signed(a, b));
    let _ = PowSign::Mathematical;
}

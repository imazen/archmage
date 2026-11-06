#![forbid(unsafe_code)]
use zenmath::scalar::powf_with as pow;
use zenmath::{
    PowBehavior as B, PowConsistency, PowInfinity, PowNaN, PowOutput, PowPrecision, PowRange,
    PowSign, SubnormalInputs, SubnormalOutputs,
};
const DEFAULT: u64 = B::CORRECT.to_u64();
#[test]
fn independent_semantics() {
    const MAG: u64 = B::CORRECT.sign(PowSign::Magnitude).to_u64();
    const SIGNED: u64 = B::CORRECT.sign(PowSign::SignedMagnitude).to_u64();
    for (x, y, mat, mag, signed) in [
        (-2.0, 3.0, 0xc1000000, 0x41000000, 0xc1000000),
        (-2.0, 4.0, 0x41800000, 0x41800000, 0xc1800000),
        (-0.0, 2.0, 0, 0, 0x80000000),
        (-0.0, -3.0, 0xff800000, 0x7f800000, 0xff800000),
        (-1.0, 16_777_216.0, 0x3f800000, 0x3f800000, 0xbf800000),
        (0.0, 0.0, 0x3f800000, 0x3f800000, 0x3f800000),
    ] {
        assert_eq!(pow::<DEFAULT>(x, y).to_bits(), mat);
        assert_eq!(pow::<MAG>(x, y).to_bits(), mag);
        assert_eq!(pow::<SIGNED>(x, y).to_bits(), signed);
    }
    assert!(pow::<DEFAULT>(-4.0, 0.5).is_nan());
    assert_eq!(pow::<MAG>(-4.0, 0.5), 2.0);
    assert_eq!(pow::<SIGNED>(-4.0, 0.5), -2.0);
    assert_eq!(pow::<SIGNED>(-4.0, 0.0), -1.0);
    assert_eq!(pow::<DEFAULT>(1.0, f32::NAN), 1.0);
    assert!(pow::<DEFAULT>(1.0, f32::from_bits(0x7f800001)).is_nan());
    assert_eq!(pow::<SIGNED>(f32::NEG_INFINITY, -2.0).to_bits(), 0x80000000);
}
#[test]
fn transformations_have_explicit_order() {
    const INPUT: u64 = B::CORRECT
        .subnormal_inputs(SubnormalInputs::FlushSignedZero)
        .to_u64();
    const OUTPUT: u64 = B::CORRECT
        .subnormal_outputs(SubnormalOutputs::FlushSignedZero)
        .to_u64();
    assert_eq!(pow::<INPUT>(-f32::from_bits(1), -3.0).to_bits(), 0xff800000);
    assert_eq!(pow::<INPUT>(2.0, f32::from_bits(1)), 1.0);
    assert_eq!(
        pow::<DEFAULT>(-f32::from_bits(1), 1.0).to_bits(),
        0x80000001
    );
    assert_eq!(pow::<OUTPUT>(-f32::from_bits(1), 1.0).to_bits(), 0x80000000);
    const REJECT: u64 = B::CORRECT.infinity(PowInfinity::RejectInputsToNaN).to_u64();
    assert!(pow::<REJECT>(f32::INFINITY, 0.0).is_nan());
    const CLAMP: u64 = B::CORRECT
        .range(PowRange::ClampUnitInterval)
        .infinity(PowInfinity::RejectInputsToNaN)
        .to_u64();
    assert_eq!(
        pow::<CLAMP>(f32::INFINITY, 0.0),
        1.0,
        "clamp precedes infinity rejection"
    );
    assert_eq!(pow::<CLAMP>(-0.0, 3.0).to_bits(), 0x80000000);
    const FINITE: u64 = B::CORRECT.output(PowOutput::SaturateFinite).to_u64();
    const UNIT: u64 = B::CORRECT.output(PowOutput::ClampUnitInterval).to_u64();
    assert_eq!(pow::<FINITE>(-0.0, -3.0).to_bits(), 0xff7fffff);
    assert_eq!(pow::<UNIT>(-2.0, 3.0).to_bits(), 0);
    assert_eq!(pow::<UNIT>(-0.0, 3.0).to_bits(), 0x80000000);
    const CANON: u64 = B::CORRECT.nan(PowNaN::CanonicalResult).to_u64();
    assert_eq!(
        pow::<CANON>(f32::from_bits(0xffc00123), 2.0).to_bits(),
        0x7fc00000
    );
}
#[test]
fn precondition_formula_and_valid_operands() {
    const MID: u64 = B::MIDP
        .range(PowRange::PositiveNormalPrecondition)
        .debug_assertions(true)
        .to_u64();
    const LOW: u64 = B::LOWP.range(PowRange::PositiveNormalPrecondition).to_u64();
    const SEP: u64 = B::MIDP
        .consistency(PowConsistency::PortableSeparate)
        .range(PowRange::PositiveNormalPrecondition)
        .to_u64();
    for (x, y) in [(0.5, 2.4), (2.0, 3.0), (1.0, 0.0)] {
        assert_eq!(
            pow::<MID>(x, y).to_bits(),
            zenmath::scalar::pow_midp_f32(x, y).to_bits()
        );
        assert_eq!(
            pow::<LOW>(x, y).to_bits(),
            zenmath::scalar::pow_lowp_f32(x, y).to_bits()
        );
        assert_eq!(
            pow::<SEP>(x, y).to_bits(),
            zenmath::scalar::unfused::pow_midp_f32(x, y).to_bits()
        );
    }
    const FULL: u64 = B::CORRECT.precision(PowPrecision::Midp).to_u64();
    assert_eq!(pow::<FULL>(-2.0, 3.0).to_bits(), 0xc1000000);
    assert_eq!(pow::<FULL>(0.0, 0.0), 1.0);
    assert_eq!(
        pow::<LOW>(0.0, 0.0).to_bits(),
        0,
        "defined formula on unchecked release/precondition violation"
    );
}
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "PositiveNormalPrecondition")]
fn debug_asserts_original_input() {
    const P: u64 = B::MIDP
        .range(PowRange::PositiveNormalPrecondition)
        .debug_assertions(true)
        .to_u64();
    let _ = pow::<P>(-2.0, 3.0);
}
#[test]
#[cfg(not(debug_assertions))]
fn debug_precondition_has_defined_release_formula() {
    const P: u64 = B::MIDP
        .range(PowRange::PositiveNormalPrecondition)
        .debug_assertions(true)
        .to_u64();
    assert!(pow::<P>(-2.0, 3.0).is_nan());
}

#![forbid(unsafe_code)]
use magemath::portable::{array::pow_f32x8_with as array, slice::pow_f32_with as slice};
use magemath::{
    PowBehavior as B, PowConsistency, PowInfinity, PowNaN, PowOutput, PowRange, PowSign,
    SubnormalInputs, SubnormalOutputs,
};
fn check<const P: u64>() {
    let bases = [
        -4.0,
        -2.0,
        -1.0,
        -0.0,
        0.0,
        f32::from_bits(1),
        0.5,
        1.0,
        2.0,
        4.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0x7f800001),
    ];
    let ys = [
        -3.0,
        -2.0,
        -0.0,
        0.0,
        0.5,
        1.0,
        2.0,
        3.0,
        2.4,
        16_777_215.0,
        16_777_216.0,
        f32::INFINITY,
        f32::NAN,
    ];
    for &x in &bases {
        for &y in &ys {
            let want = zenmath::scalar::powf_with::<P>(x, y);
            for got in array::<P>([x; 8], [y; 8]) {
                assert!(
                    got.to_bits() == want.to_bits() || got.is_nan() && want.is_nan(),
                    "policy{P} {x}^{y}"
                );
            }
        }
    }
    for n in 0..18 {
        let mut x: Vec<_> = (0..n).map(|i| bases[i % bases.len()]).collect();
        let y: Vec<_> = (0..n).map(|i| ys[i % ys.len()]).collect();
        let want: Vec<_> = x
            .iter()
            .zip(&y)
            .map(|(x, y)| zenmath::scalar::powf_with::<P>(*x, *y))
            .collect();
        slice::<P>(&mut x, &y);
        for (got, want) in x.into_iter().zip(want) {
            assert!(got.to_bits() == want.to_bits() || got.is_nan() && want.is_nan());
        }
    }
}
#[test]
fn packed_policies_match_shared_scalar() {
    check::<{ B::CORRECT.to_u64() }>();
    check::<{ B::MIDP.to_u64() }>();
    check::<{ B::LOWP.to_u64() }>();
    check::<
        {
            B::MIDP
                .consistency(PowConsistency::PortableSeparate)
                .to_u64()
        },
    >();
    check::<
        {
            B::LOWP
                .consistency(PowConsistency::PortableSeparate)
                .to_u64()
        },
    >();
    check::<{ B::CORRECT.sign(PowSign::Magnitude).to_u64() }>();
    check::<{ B::CORRECT.sign(PowSign::SignedMagnitude).to_u64() }>();
    check::<
        {
            B::CORRECT
                .subnormal_inputs(SubnormalInputs::FlushSignedZero)
                .to_u64()
        },
    >();
    check::<
        {
            B::CORRECT
                .subnormal_outputs(SubnormalOutputs::FlushSignedZero)
                .to_u64()
        },
    >();
    check::<{ B::CORRECT.nan(PowNaN::CanonicalResult).to_u64() }>();
    check::<{ B::CORRECT.infinity(PowInfinity::RejectInputsToNaN).to_u64() }>();
    check::<{ B::CORRECT.range(PowRange::ClampUnitInterval).to_u64() }>();
    check::<{ B::CORRECT.output(PowOutput::SaturateFinite).to_u64() }>();
    check::<{ B::CORRECT.output(PowOutput::ClampUnitInterval).to_u64() }>();
}
#[test]
fn precondition_tail_padding_is_valid() {
    const P: u64 = B::MIDP
        .range(PowRange::PositiveNormalPrecondition)
        .debug_assertions(true)
        .to_u64();
    for n in 0..18 {
        let mut x = vec![0.5; n];
        let y = vec![2.4; n];
        slice::<P>(&mut x, &y);
        assert!(
            x.iter()
                .all(|x| x.to_bits() == zenmath::scalar::powf_with::<P>(0.5, 2.4).to_bits())
        );
    }
}
#[test]
fn precondition_violations_follow_named_formula_without_debug_flag() {
    const P: u64 = B::LOWP.range(PowRange::PositiveNormalPrecondition).to_u64();
    for (x, y) in [
        (0.0, 0.0),
        (-2.0, 3.0),
        (f32::INFINITY, 0.0),
        (f32::NAN, 2.0),
    ] {
        let want = zenmath::scalar::powf_with::<P>(x, y);
        let got = array::<P>([x; 8], [y; 8])[0];
        assert!(got.to_bits() == want.to_bits() || got.is_nan() && want.is_nan());
    }
}

#[archmage::attune(_v3, _v3_t)]
fn matching_context(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
    const P: u64 = B::CORRECT.sign(PowSign::SignedMagnitude).to_u64();
    magemath::portable::array::pow_f32x8_with_v3::<P>(a, b)
}
#[test]
fn explicit_proof_and_matching_context() {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            const P: u64 = B::CORRECT.sign(PowSign::SignedMagnitude).to_u64();
            let a = [-4.0; 8];
            let b = [0.5; 8];
            assert_eq!(
                magemath::portable::array::pow_f32x8_with_v3_t::<P>(token, a, b),
                [-2.0; 8]
            );
            assert_eq!(matching_context_v3_t(token, a, b), [-2.0; 8]);
            let mut x = [-4.0; 11];
            let y = [0.5; 11];
            magemath::portable::slice::pow_f32_with_v3_t::<P>(token, &mut x, &y);
            assert_eq!(x, [-2.0; 11]);
        }
    }
}

//! Slice entries return the same bits as the scalar entries, element by
//! element, for every length around the 8-lane and 16-lane chunking, and
//! report the exact first rejected index.
use magemath::{
    SliceError,
    cbrtf::{self as c, Cbrtf, CbrtfOptions},
    expf::{self as e, Expf, ExpfOptions},
    log1pf as l1p,
    logf::{self as l, Logf, LogfOptions},
    powf::{self as p, Powf, PowfOptions},
};

fn inputs(len: usize, lo: f32, hi: f32, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            lo + (hi - lo) * (s >> 8) as f32 / 16_777_216.0
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn unary_slices_match_scalar_entries() {
    type UnarySliceFn<E> = fn(&[f32], &mut [f32]) -> Result<(), SliceError<E>>;
    type UnaryScalarFn<E> = fn(f32) -> Result<f32, E>;

    fn check<E1: core::fmt::Debug, E2: core::fmt::Debug>(
        slice_fn: UnarySliceFn<E1>,
        scalar_fn: UnaryScalarFn<E2>,
        lo: f32,
        hi: f32,
        label: &str,
    ) {
        for len in 0..=35 {
            let src = inputs(len, lo, hi, 0x9e37 + len as u32);
            let mut dst = vec![f32::NAN; len];
            slice_fn(&src, &mut dst).unwrap();
            let want: Vec<f32> = src.iter().map(|&x| scalar_fn(x).unwrap()).collect();
            assert_eq!(bits(&dst), bits(&want), "{label} len {len}");
        }
    }

    const EXP_LOWP: u128 = ExpfOptions::new().lowp().bits();
    const LOG_LOWP: u128 = LogfOptions::new().lowp().bits();
    const CBRT_LOWP_FULL: u128 = CbrtfOptions::new().full_domain().lowp().bits();

    check(e::try_exp2_slice, e::try_exp2, -130.0, 130.0, "exp2 midp");
    check(
        Expf::<EXP_LOWP>::exp2_slice,
        Expf::<EXP_LOWP>::exp2,
        -130.0,
        130.0,
        "exp2 lowp",
    );
    check(e::try_exp_slice, e::try_exp, -90.0, 90.0, "exp midp");
    check(l::try_log2_slice, l::try_log2, 1e-6, 1e6, "log2 midp");
    check(
        Logf::<LOG_LOWP>::log2_slice,
        Logf::<LOG_LOWP>::log2,
        1e-6,
        1e6,
        "log2 lowp",
    );
    check(l::try_ln_slice, l::try_ln, 1e-6, 1e6, "ln midp");
    check(l::try_log10_slice, l::try_log10, 1e-6, 1e6, "log10 midp");
    check(
        l1p::try_log1p_slice,
        l1p::try_log1p,
        -0.99,
        100.0,
        "log1p midp",
    );
    check(c::try_cbrt_slice, c::try_cbrt, -1e6, 1e6, "cbrt midp");
    check(
        Cbrtf::<CBRT_LOWP_FULL>::slice,
        Cbrtf::<CBRT_LOWP_FULL>::scalar,
        -1e6,
        1e6,
        "cbrt lowp full",
    );
}

#[test]
fn powf_slices_match_scalar_entries() {
    fn check<const CFG: u128>(lo: f32, hi: f32, n: f32, label: &str) {
        for len in 0..=35 {
            let src = inputs(len, lo, hi, 0x51ed + len as u32);
            let mut dst = vec![f32::NAN; len];
            Powf::<CFG>::slice(&src, &mut dst, n).unwrap();
            let want: Vec<f32> = src
                .iter()
                .map(|&x| Powf::<CFG>::scalar(x, n).unwrap())
                .collect();
            assert_eq!(bits(&dst), bits(&want), "powf {label} len {len}");
        }
    }

    const GM: u128 = PowfOptions::new().gamma().midp().bits();
    const GL: u128 = PowfOptions::new().gamma().lowp().bits();
    const PGM: u128 = PowfOptions::new().positive_gamma().midp().bits();
    const NM: u128 = PowfOptions::new().midp().bits();
    const GH: u128 = PowfOptions::new().gamma().highp().bits();
    const FH: u128 = PowfOptions::new().full_domain().highp().bits();
    const CB_GM: u128 = PowfOptions::new()
        .gamma()
        .midp()
        .backend_dependent()
        .caller()
        .bits();

    check::<GM>(0.0, 4.0, 2.4, "gamma midp");
    check::<GL>(0.0, 4.0, 1.0 / 2.2, "gamma lowp");
    check::<PGM>(0.001, 4.0, 2.2, "pos gamma midp");
    check::<NM>(1e-3, 1e3, -1.5, "normal midp");
    check::<GH>(0.001, 4.0, 2.4, "gamma highp");
    check::<FH>(-4.0, 4.0, 3.0, "full highp");
    check::<CB_GM>(0.0, 4.0, 2.4, "caller backend gamma midp");
}

#[test]
fn caller_execution_with_a_token_matches_upgrade_for_bit_exact_options() {
    const GM: u128 = PowfOptions::new().gamma().midp().bits();
    const CALLER_GM: u128 = PowfOptions::new().gamma().midp().caller().bits();
    let src = inputs(29, 0.0, 4.0, 7);
    let (mut a, mut b) = (vec![0.0; 29], vec![0.0; 29]);
    Powf::<GM>::slice(&src, &mut a, 2.2).unwrap();
    Powf::<CALLER_GM>::slice_with_token(archmage::ScalarToken, &src, &mut b, 2.2).unwrap();
    assert_eq!(bits(&a), bits(&b));
}

#[test]
fn first_rejected_index_is_reported() {
    const GM: u128 = PowfOptions::new().gamma().midp().bits();
    const GH: u128 = PowfOptions::new().gamma().highp().bits();
    for len in [1, 7, 8, 9, 15, 16, 17, 20, 23, 33] {
        for bad in 0..len {
            let mut src = vec![0.5f32; len];
            src[bad] = -1.0;
            if bad + 3 < len {
                src[bad + 3] = f32::NAN;
            }
            let mut dst = vec![0.0; len];
            assert_eq!(
                Powf::<GM>::slice(&src, &mut dst, 2.0),
                Err(SliceError {
                    kind: p::PowfErrorKind::InvalidBase,
                    index: bad
                }),
                "len {len} bad {bad}"
            );
            assert_eq!(
                l::try_log2_slice(&src, &mut dst).unwrap_err().index,
                bad,
                "log len {len} bad {bad}"
            );
        }
    }
    let mut dst = [0.0; 3];
    assert_eq!(
        Powf::<GM>::slice(&[0.5; 3], &mut dst, f32::NAN),
        Err(SliceError {
            kind: p::PowfErrorKind::InvalidExponent,
            index: 0
        })
    );
    let mut src = [0.5f32; 19];
    src[17] = -1.0;
    let mut dst = [0.0; 19];
    assert_eq!(
        Powf::<GH>::slice(&src, &mut dst, 2.0).unwrap_err().index,
        17
    );
}

#[test]
fn powf_slice_reports_earlier_result_error_before_later_base_error() {
    const MIDP: PowfOptions = PowfOptions::new().midp();
    const HIGHP: PowfOptions = PowfOptions::new().highp();
    let mut src = [1.0; 25];
    src[18] = f32::MAX;
    src[21] = f32::NAN;
    let mut dst = [0.0; 25];
    let expected = Err(SliceError {
        kind: p::PowfErrorKind::OutsideApproximationRange,
        index: 18,
    });

    assert_eq!(
        Powf::<{ MIDP.bits() }>::slice(&src, &mut dst, 2.0),
        expected
    );
    assert_eq!(MIDP.runtime().slice(&src, &mut dst, 2.0), expected);
    assert_eq!(
        Powf::<{ HIGHP.bits() }>::slice(&src, &mut dst, 2.0),
        expected
    );
    assert_eq!(HIGHP.runtime().slice(&src, &mut dst, 2.0), expected);
    #[cfg(feature = "std")]
    {
        const STD: PowfOptions = HIGHP.backend_dependent();
        assert_eq!(Powf::<{ STD.bits() }>::slice(&src, &mut dst, 2.0), expected);
        assert_eq!(STD.runtime().slice(&src, &mut dst, 2.0), expected);
    }
}

#[test]
#[should_panic(expected = "slice lengths differ")]
fn length_mismatch_panics() {
    let _ = p::try_powf_slice(&[1.0; 4], &mut [0.0; 3], 2.0);
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[test]
fn vnni_caller_scalar_and_slice_match_vector_bits() {
    use archmage::{SimdToken, X64V3VnniToken};

    #[archmage::attune(wrap)]
    fn check(token: X64V3VnniToken) {
        use magemath::SimdMath;
        use magetypes::simd::generic::f32x4;

        const CBRT: u128 = CbrtfOptions::new().caller().backend_dependent().bits();
        const EXP: u128 = ExpfOptions::new().caller().backend_dependent().bits();
        const LOG: u128 = LogfOptions::new().caller().backend_dependent().bits();
        const LOG1P: u128 = l1p::Log1pfOptions::new()
            .caller()
            .backend_dependent()
            .bits();
        const POW: u128 = PowfOptions::new().caller().backend_dependent().bits();
        const N: f32 = 2.4;

        // Put the rounding witness in both the eight-lane block and the tail.
        let src = [
            1.039_062_5,
            0.31,
            0.68,
            1.72,
            7.0,
            2.61,
            0.103,
            2.345,
            1.039_062_5,
        ];
        let mut dst = [0.0; 9];
        macro_rules! check_family {
            ($api:ty, $scalar:ident, $slice:ident, $vector:ident, $cfg:ident $(, $arg:expr)?) => {
                <$api>::$slice(token, &src, &mut dst $(, $arg)?).unwrap();
                for (&x, &slice) in src.iter().zip(&dst) {
                    let expected = f32x4::splat_t(token, x)
                        .$vector::<$cfg>($($arg)?)
                        .unwrap()
                        .to_array()[0]
                        .to_bits();
                    let scalar = <$api>::$scalar(token, x $(, $arg)?).unwrap().to_bits();
                    assert_eq!(scalar, expected, "{} scalar at {x}", stringify!($vector));
                    assert_eq!(slice.to_bits(), expected, "{} slice at {x}", stringify!($vector));
                }
            };
        }

        check_family!(
            Cbrtf<CBRT>,
            scalar_with_token,
            slice_with_token,
            try_cbrt_with,
            CBRT
        );
        check_family!(
            Expf<EXP>,
            exp2_with_token,
            exp2_slice_with_token,
            try_exp2_with,
            EXP
        );
        check_family!(
            Logf<LOG>,
            log2_with_token,
            log2_slice_with_token,
            try_log2_with,
            LOG
        );
        check_family!(
            l1p::Log1pf<LOG1P>,
            scalar_with_token,
            slice_with_token,
            try_log1p_with,
            LOG1P
        );
        check_family!(
            Powf<POW>,
            scalar_with_token,
            slice_with_token,
            try_powf_with,
            POW,
            N
        );
    }

    if let Some(token) = X64V3VnniToken::summon() {
        check(token);
    }
}

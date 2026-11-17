//! Slice entries return the same bits as the scalar entries, element by
//! element, for every length around the 8-lane and 16-lane chunking, and
//! report the exact first rejected index.
use magemath::{
    SliceError,
    cbrtf::{self as c, Cbrtf, CbrtfOptions},
    expf::{self as e, Expf, ExpfOptions},
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
    check(l::try_log1p_slice, l::try_log1p, -0.99, 100.0, "log1p midp");
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
#[should_panic(expected = "slice lengths differ")]
fn length_mismatch_panics() {
    let _ = p::try_powf_slice(&[1.0; 4], &mut [0.0; 3], 2.0);
}

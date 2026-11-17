//! Slice entries return the same bits as the scalar entries, element by
//! element, for every length around the 8-lane chunking, and report the first
//! rejected index.
use magemath::{Backend, Caller, SliceError, cbrtf as c, expf as e, logf as l, powf as p};

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

    check(
        e::try_exp2_slice_with::<e::presets::Midp>,
        e::try_exp2_with::<e::presets::Midp>,
        -130.0,
        130.0,
        "exp2 midp",
    );
    check(
        e::try_exp2_slice_with::<e::presets::Lowp>,
        e::try_exp2_with::<e::presets::Lowp>,
        -130.0,
        130.0,
        "exp2 lowp",
    );
    check(
        e::try_exp_slice_with::<e::presets::Midp>,
        e::try_exp_with::<e::presets::Midp>,
        -90.0,
        90.0,
        "exp midp",
    );
    check(
        l::try_log2_slice_with::<l::presets::Midp>,
        l::try_log2_with::<l::presets::Midp>,
        1e-6,
        1e6,
        "log2 midp",
    );
    check(
        l::try_log2_slice_with::<l::presets::Lowp>,
        l::try_log2_with::<l::presets::Lowp>,
        1e-6,
        1e6,
        "log2 lowp",
    );
    check(
        l::try_ln_slice_with::<l::presets::Midp>,
        l::try_ln_with::<l::presets::Midp>,
        1e-6,
        1e6,
        "ln midp",
    );
    check(
        l::try_log10_slice_with::<l::presets::Midp>,
        l::try_log10_with::<l::presets::Midp>,
        1e-6,
        1e6,
        "log10 midp",
    );
    check(
        l::try_log1p_slice_with::<l::presets::Midp>,
        l::try_log1p_with::<l::presets::Midp>,
        -0.99,
        100.0,
        "log1p midp",
    );
    check(
        c::try_cbrt_slice_with::<c::presets::Midp>,
        c::try_cbrt_with::<c::presets::Midp>,
        -1e6,
        1e6,
        "cbrt midp",
    );
    check(
        c::try_cbrt_slice_with::<c::presets::LowpFull>,
        c::try_cbrt_with::<c::presets::LowpFull>,
        -1e6,
        1e6,
        "cbrt lowp full",
    );
}

#[test]
fn powf_slices_match_scalar_entries() {
    fn check<P: p::PowfProfile>(lo: f32, hi: f32, n: f32, label: &str) {
        for len in 0..=35 {
            let src = inputs(len, lo, hi, 0x51ed + len as u32);
            let mut dst = vec![f32::NAN; len];
            p::try_powf_slice_with::<P>(&src, &mut dst, n).unwrap();
            let want: Vec<f32> = src
                .iter()
                .map(|&x| p::try_powf_with::<P>(x, n).unwrap())
                .collect();
            assert_eq!(bits(&dst), bits(&want), "powf {label} len {len}");
        }
    }

    check::<p::presets::GammaMidp>(0.0, 4.0, 2.4, "gamma midp");
    check::<p::presets::GammaLowp>(0.0, 4.0, 1.0 / 2.2, "gamma lowp");
    check::<p::presets::PositiveGammaMidp>(0.001, 4.0, 2.2, "pos gamma midp");
    check::<p::presets::NormalMidp>(1e-3, 1e3, -1.5, "normal midp");
    check::<p::presets::GammaHighp>(0.001, 4.0, 2.4, "gamma highp");
    check::<p::presets::FullHighp>(-4.0, 4.0, 3.0, "full highp");
    check::<Caller<Backend<p::presets::GammaMidp>>>(0.0, 4.0, 2.4, "caller backend gamma midp");
}

#[test]
fn caller_execution_with_a_token_matches_automatic_for_bit_exact_presets() {
    let src = inputs(29, 0.0, 4.0, 7);
    let (mut a, mut b) = (vec![0.0; 29], vec![0.0; 29]);
    p::try_powf_slice_with::<p::presets::GammaMidp>(&src, &mut a, 2.2).unwrap();
    p::try_powf_slice_with_token::<Caller<p::presets::GammaMidp>, _>(
        archmage::ScalarToken,
        &src,
        &mut b,
        2.2,
    )
    .unwrap();
    assert_eq!(bits(&a), bits(&b));
}

#[test]
fn first_rejected_index_is_reported() {
    for len in [1, 7, 8, 9, 20, 23] {
        for bad in 0..len {
            let mut src = vec![0.5f32; len];
            src[bad] = -1.0;
            if bad + 3 < len {
                src[bad + 3] = f32::NAN;
            }
            let mut dst = vec![0.0; len];
            assert_eq!(
                p::try_powf_slice_with::<p::presets::GammaMidp>(&src, &mut dst, 2.0),
                Err(SliceError {
                    kind: p::PowfErrorKind::InvalidBase,
                    index: bad
                }),
                "len {len} bad {bad}"
            );
            assert_eq!(
                l::try_log2_slice_with::<l::presets::Midp>(&src, &mut dst)
                    .unwrap_err()
                    .index,
                bad,
                "log len {len} bad {bad}"
            );
        }
    }
    let mut dst = [0.0; 3];
    assert_eq!(
        p::try_powf_slice_with::<p::presets::GammaMidp>(&[0.5; 3], &mut dst, f32::NAN),
        Err(SliceError {
            kind: p::PowfErrorKind::InvalidExponent,
            index: 0
        })
    );
    let mut src = [0.5f32; 11];
    src[9] = -1.0;
    let mut dst = [0.0; 11];
    assert_eq!(
        p::try_powf_slice_with::<p::presets::GammaHighp>(&src, &mut dst, 2.0)
            .unwrap_err()
            .index,
        9
    );
}

#[test]
#[should_panic(expected = "slice lengths differ")]
fn length_mismatch_panics() {
    let _ = p::try_powf_slice_with::<p::presets::GammaMidp>(&[1.0; 4], &mut [0.0; 3], 2.0);
}

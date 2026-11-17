//! Slice entries return the same bits as the scalar entries, element by
//! element, for every length around the 8-lane chunking, and report the first
//! rejected index.
use magemath::{SliceError, cbrtf as c, expf as e, logf as l, powf as p};

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

macro_rules! unary_matches {
    ($m:ident :: $slice:ident, $scalar:ident, $opts:expr, $lo:expr, $hi:expr) => {{
        const O: u32 = $opts;
        for len in 0..=35 {
            let src = inputs(len, $lo, $hi, 0x9e37 + len as u32);
            let mut dst = vec![f32::NAN; len];
            $m::$slice::<O>(&src, &mut dst).unwrap();
            let want: Vec<f32> = src.iter().map(|&x| $m::$scalar::<O>(x).unwrap()).collect();
            assert_eq!(bits(&dst), bits(&want), "{} len {len}", stringify!($slice));
        }
    }};
}

#[test]
fn unary_slices_match_scalar_entries() {
    unary_matches!(
        e::try_exp2_slice_with,
        try_exp2_with,
        e::ExpfOptions::PORTABLE_MIDP.bits(),
        -130.0,
        130.0
    );
    unary_matches!(
        e::try_exp2_slice_with,
        try_exp2_with,
        e::ExpfOptions::PORTABLE_LOWP.bits(),
        -130.0,
        130.0
    );
    unary_matches!(
        e::try_exp_slice_with,
        try_exp_with,
        e::ExpfOptions::PORTABLE_MIDP.bits(),
        -90.0,
        90.0
    );
    unary_matches!(
        l::try_log2_slice_with,
        try_log2_with,
        l::LogfOptions::PORTABLE_MIDP.bits(),
        1e-6,
        1e6
    );
    unary_matches!(
        l::try_log2_slice_with,
        try_log2_with,
        l::LogfOptions::PORTABLE_LOWP.bits(),
        1e-6,
        1e6
    );
    unary_matches!(
        l::try_ln_slice_with,
        try_ln_with,
        l::LogfOptions::PORTABLE_MIDP.bits(),
        1e-6,
        1e6
    );
    unary_matches!(
        l::try_log10_slice_with,
        try_log10_with,
        l::LogfOptions::PORTABLE_MIDP.bits(),
        1e-6,
        1e6
    );
    unary_matches!(
        l::try_log1p_slice_with,
        try_log1p_with,
        l::LogfOptions::PORTABLE_MIDP.bits(),
        -0.99,
        100.0
    );
    unary_matches!(
        c::try_cbrt_slice_with,
        try_cbrt_with,
        c::CbrtfOptions::PORTABLE_MIDP.bits(),
        -1e6,
        1e6
    );
    unary_matches!(
        c::try_cbrt_slice_with,
        try_cbrt_with,
        c::CbrtfOptions::PORTABLE_LOWP_FULL.bits(),
        -1e6,
        1e6
    );
}

macro_rules! pow_matches {
    ($opts:expr, $lo:expr, $hi:expr, $n:expr) => {{
        const O: u32 = $opts;
        for len in 0..=35 {
            let src = inputs(len, $lo, $hi, 0x51ed + len as u32);
            let mut dst = vec![f32::NAN; len];
            p::try_powf_slice_with::<O>(&src, &mut dst, $n).unwrap();
            let want: Vec<f32> = src
                .iter()
                .map(|&x| p::try_powf_with::<O>(x, $n).unwrap())
                .collect();
            assert_eq!(
                bits(&dst),
                bits(&want),
                "powf {} len {len}",
                stringify!($opts)
            );
        }
    }};
}

#[test]
fn powf_slices_match_scalar_entries() {
    use p::PowfOptions as P;
    pow_matches!(P::PORTABLE_GAMMA_MIDP.bits(), 0.0, 4.0, 2.4);
    pow_matches!(P::PORTABLE_GAMMA_LOWP.bits(), 0.0, 4.0, 1.0 / 2.2);
    pow_matches!(P::PORTABLE_POSITIVE_GAMMA_MIDP.bits(), 0.001, 4.0, 2.2);
    pow_matches!(P::PORTABLE_NORMAL_MIDP.bits(), 1e-3, 1e3, -1.5);
    pow_matches!(P::PORTABLE_GAMMA_HIGHP.bits(), 0.001, 4.0, 2.4);
    pow_matches!(P::PORTABLE_HIGHP.bits(), -4.0, 4.0, 3.0);
    pow_matches!(
        P::BACKEND_GAMMA_MIDP
            .with_execution(p::PowfExecution::Caller)
            .bits(),
        0.0,
        4.0,
        2.4
    );
}

#[test]
fn caller_execution_with_a_token_matches_automatic_for_bit_exact_presets() {
    const A: u32 = p::PowfOptions::PORTABLE_GAMMA_MIDP.bits();
    const C: u32 = p::PowfOptions::PORTABLE_GAMMA_MIDP
        .with_execution(p::PowfExecution::Caller)
        .bits();
    let src = inputs(29, 0.0, 4.0, 7);
    let (mut a, mut b) = (vec![0.0; 29], vec![0.0; 29]);
    p::try_powf_slice_with::<A>(&src, &mut a, 2.2).unwrap();
    p::try_powf_slice_with_token::<C, _>(archmage::ScalarToken, &src, &mut b, 2.2).unwrap();
    assert_eq!(bits(&a), bits(&b));
}

#[test]
fn first_rejected_index_is_reported() {
    const G: u32 = p::PowfOptions::PORTABLE_GAMMA_MIDP.bits();
    for len in [1, 7, 8, 9, 20, 23] {
        for bad in 0..len {
            let mut src = vec![0.5f32; len];
            src[bad] = -1.0;
            if bad + 3 < len {
                src[bad + 3] = f32::NAN; // a later failure must not win
            }
            let mut dst = vec![0.0; len];
            assert_eq!(
                p::try_powf_slice_with::<G>(&src, &mut dst, 2.0),
                Err(SliceError {
                    kind: p::PowfErrorKind::InvalidBase,
                    index: bad
                }),
                "len {len} bad {bad}"
            );
            const L: u32 = l::LogfOptions::PORTABLE_MIDP.bits();
            assert_eq!(
                l::try_log2_slice_with::<L>(&src, &mut dst)
                    .unwrap_err()
                    .index,
                bad,
                "log len {len} bad {bad}"
            );
        }
    }
    let mut dst = [0.0; 3];
    assert_eq!(
        p::try_powf_slice_with::<G>(&[0.5; 3], &mut dst, f32::NAN),
        Err(SliceError {
            kind: p::PowfErrorKind::InvalidExponent,
            index: 0
        })
    );
    const H: u32 = p::PowfOptions::PORTABLE_GAMMA_HIGHP.bits();
    let mut src = [0.5f32; 11];
    src[9] = -1.0;
    let mut dst = [0.0; 11];
    assert_eq!(
        p::try_powf_slice_with::<H>(&src, &mut dst, 2.0)
            .unwrap_err()
            .index,
        9
    );
}

#[test]
#[should_panic(expected = "slice lengths differ")]
fn length_mismatch_panics() {
    const G: u32 = p::PowfOptions::PORTABLE_GAMMA_MIDP.bits();
    let _ = p::try_powf_slice_with::<G>(&[1.0; 4], &mut [0.0; 3], 2.0);
}

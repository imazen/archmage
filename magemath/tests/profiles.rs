//! Profiles: typed range-as-data configuration across powf, expf, logf, and cbrtf,
//! root modifiers (`Caller`, `Unchecked`, `Backend`), and custom-range tail behavior.
use archmage::ScalarToken;
use magemath::{
    Backend, Caller, Portability, SliceError, Unchecked, cbrtf as c,
    expf::{self as e, ExpfProfile},
    logf as l,
    powf::{self as p, PowfProfile},
};
use magetypes::simd::generic::f32x8;

fn log_uniform(len: usize, lo: f32, hi: f32, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (lo.ln() + (hi.ln() - lo.ln()) * (s >> 8) as f32 / 16_777_216.0).exp()
        })
        .collect()
}

fn uniform(len: usize, lo: f32, hi: f32, seed: u32) -> Vec<f32> {
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
fn powf_profiles_match_options_across_presets_including_highp_and_full() {
    fn same<P: p::PowfProfile, const B: u32>(lo: f32, hi: f32, ns: &[f32], zero: bool) {
        for (k, &n) in ns.iter().enumerate() {
            let mut src = log_uniform(99, lo, hi, 77 + k as u32);
            if zero {
                src[5] = 0.0;
                src[98] = 0.0;
            }
            let (mut a, mut b) = (vec![0.0; 99], vec![0.0; 99]);
            p::try_powf_slice_with::<P>(&src, &mut a, n).unwrap();
            p::try_powf_slice_with::<p::Options<B>>(&src, &mut b, n).unwrap();
            assert_eq!(bits(&a), bits(&b), "n={n}");
        }
    }

    use p::{PowfDomain, PowfOptions as PO, presets};
    same::<presets::GammaMidp, { PO::PORTABLE_GAMMA_MIDP.bits() }>(
        1.6e-5,
        65_000.0,
        &[0.0, 0.4545, 1.0, 2.2, 2.4, 4.0],
        true,
    );
    same::<presets::GammaLowp, { PO::PORTABLE_GAMMA_LOWP.bits() }>(
        1.6e-5,
        65_000.0,
        &[0.4545, 2.4, 4.0],
        true,
    );
    same::<presets::PositiveGammaMidp, { PO::PORTABLE_POSITIVE_GAMMA_MIDP.bits() }>(
        1.6e-5,
        65_000.0,
        &[0.4545, 2.4],
        false,
    );
    same::<presets::NormalMidp, { PO::PORTABLE_NORMAL_MIDP.bits() }>(
        1e-6,
        1e6,
        &[-3.0, -0.5, 0.0, 1.5, 5.0],
        false,
    );
    same::<presets::GammaHighp, { PO::PORTABLE_GAMMA_HIGHP.bits() }>(
        1.6e-5,
        65_000.0,
        &[0.0, 0.5, 2.4],
        true,
    );
    same::<presets::FullHighp, { PO::PORTABLE_HIGHP.bits() }>(1e-4, 1e4, &[-2.5, 0.0, 2.4], true);
    const NON_NEG: u32 = PO::new().with_domain(PowfDomain::NonNegativeNormal).bits();
    same::<presets::NonNegativeNormalMidp, NON_NEG>(1e-6, 1e6, &[0.0, 1.7], true);
}

struct Tone;
impl p::PowfProfile for Tone {
    const SPEC: p::PowfSpec = p::PowfSpec::new()
        .base_range(1.0 / 1024.0, 16.0)
        .exponent_range(-8.0, 8.0);
}
struct ToneLowp;
impl p::PowfProfile for ToneLowp {
    const SPEC: p::PowfSpec = Tone::SPEC.precision(p::PowfPrecision::Lowp);
}

#[test]
fn custom_powf_range_elides_check_and_stays_accurate() {
    const { assert!(Tone::SPEC.elides_range_check()) };
    let mut worst = (0f64, 0f64);
    for (k, n) in [-8.0f32, -7.99, -1.0, -0.001, 0.0, 0.3, 1.0, 2.2, 7.5, 8.0]
        .into_iter()
        .enumerate()
    {
        let mut src = log_uniform(512, 1.0 / 1024.0, 16.0, 11 + k as u32);
        src[0] = 1.0 / 1024.0;
        src[1] = 16.0;
        let (mut m, mut l) = (vec![0.0; 512], vec![0.0; 512]);
        p::try_powf_slice_with::<Tone>(&src, &mut m, n).unwrap();
        p::try_powf_slice_with::<ToneLowp>(&src, &mut l, n).unwrap();
        for i in 0..512 {
            let t = (src[i] as f64).powf(n as f64);
            assert!(m[i].is_finite() && l[i].is_finite(), "x={} n={n}", src[i]);
            worst.0 = worst.0.max(((m[i] as f64 - t) / t).abs());
            worst.1 = worst.1.max(((l[i] as f64 - t) / t).abs());
        }
    }
    assert!(worst.0 < 2e-5, "midp relative error {}", worst.0);
    assert!(worst.1 < 7e-3, "lowp relative error {}", worst.1);
}

#[test]
fn custom_powf_range_rejects_what_it_excludes() {
    let mut dst = [0.0; 12];
    let mut src = [1.0f32; 12];
    src[10] = 17.0;
    assert_eq!(
        p::try_powf_slice_with::<Tone>(&src, &mut dst, 2.0),
        Err(SliceError {
            kind: p::PowfErrorKind::InvalidBase,
            index: 10
        })
    );
    src[10] = 0.0;
    assert_eq!(
        p::try_powf_slice_with::<Tone>(&src, &mut dst, 2.0)
            .unwrap_err()
            .index,
        10
    );
    assert_eq!(
        p::try_powf_slice_with::<Tone>(&[1.0; 12], &mut dst, 8.5),
        Err(SliceError {
            kind: p::PowfErrorKind::InvalidExponent,
            index: 0
        })
    );
}

/// Wide range keeping the range check while still enforcing `base_range` and `admit_zero`
/// in a single SIMD pass (no pre-scan).
struct Wide;
impl p::PowfProfile for Wide {
    const SPEC: p::PowfSpec = p::PowfSpec::new()
        .base_range(1e-10, 1e10)
        .exponent_range(-10.0, 10.0)
        .admit_zero();
}

#[test]
fn wide_powf_range_single_pass_validation() {
    const { assert!(!Wide::SPEC.elides_range_check()) };
    let mut dst = [0.0; 9];
    let mut src = [2.0f32; 9];
    src[3] = 0.0;
    p::try_powf_slice_with::<Wide>(&src, &mut dst, 3.0).unwrap();
    assert_eq!(dst[3], 0.0);
    assert_eq!(dst[0], 8.0);
    src[8] = 1e10;
    assert_eq!(
        p::try_powf_slice_with::<Wide>(&src, &mut dst, 10.0),
        Err(SliceError {
            kind: p::PowfErrorKind::OutsideApproximationRange,
            index: 8
        })
    );
    src[8] = 2.0;
    src[6] = 1e11;
    src[7] = -1.0;
    assert_eq!(
        p::try_powf_slice_with::<Wide>(&src, &mut dst, 1.0),
        Err(SliceError {
            kind: p::PowfErrorKind::InvalidBase,
            index: 6
        })
    );
    src[6] = 2.0;
    src[7] = 2.0;
    assert_eq!(
        p::try_powf_slice_with::<Wide>(&src, &mut dst, -1.0),
        Err(SliceError {
            kind: p::PowfErrorKind::InvalidBase,
            index: 3
        })
    );
}

/// Ranges that deliberately exclude `1.0` verify `tail[0]` padding on tails < 8.
struct PowExcludingOne;
impl p::PowfProfile for PowExcludingOne {
    const SPEC: p::PowfSpec = p::PowfSpec::new()
        .base_range(2.0, 8.0)
        .exponent_range(0.5, 2.0);
}

struct ExpClamped;
impl e::ExpfProfile for ExpClamped {
    const SPEC: e::ExpfSpec = e::ExpfSpec::new().input_range(2.0, 8.0);
}

struct ExpTight;
impl e::ExpfProfile for ExpTight {
    const SPEC: e::ExpfSpec = e::ExpfSpec::new().input_range(-10.0, 10.0);
}

struct LogExcludingOne;
impl l::LogfProfile for LogExcludingOne {
    const SPEC: l::LogfSpec = l::LogfSpec::new().input_range(2.0, 8.0);
}

struct Log1pSmall;
impl l::LogfProfile for Log1pSmall {
    const SPEC: l::LogfSpec = l::LogfSpec::new().input_range(-0.5, 0.5);
}

struct CbrtExcludingOne;
impl c::CbrtfProfile for CbrtExcludingOne {
    const SPEC: c::CbrtfSpec = c::CbrtfSpec::new().input_range(2.0, 8.0);
}

struct CbrtNearZero;
impl c::CbrtfProfile for CbrtNearZero {
    const SPEC: c::CbrtfSpec = c::CbrtfSpec::new().input_range(-8.0, 8.0);
}

#[test]
fn custom_range_tails_excluding_one_accept_and_reject_exact_indices() {
    const { assert!(ExpClamped::SPEC.elides_clamp()) };
    const { assert!(ExpTight::SPEC.elides_exp_clamp()) };

    for len in 1..=11 {
        let pow_src = uniform(len, 2.0, 8.0, 100 + len as u32);
        let mut pow_dst = vec![0.0; len];
        p::try_powf_slice_with::<PowExcludingOne>(&pow_src, &mut pow_dst, 1.5).unwrap();
        for (i, &x) in pow_src.iter().enumerate() {
            assert_eq!(
                pow_dst[i].to_bits(),
                p::try_powf_with::<PowExcludingOne>(x, 1.5)
                    .unwrap()
                    .to_bits()
            );
        }

        let mut exp_dst = vec![0.0; len];
        e::try_exp2_slice_with::<ExpClamped>(&pow_src, &mut exp_dst).unwrap();
        for (i, &x) in pow_src.iter().enumerate() {
            assert_eq!(
                exp_dst[i].to_bits(),
                e::try_exp2_with::<ExpClamped>(x).unwrap().to_bits()
            );
        }

        let mut log_dst = vec![0.0; len];
        l::try_log2_slice_with::<LogExcludingOne>(&pow_src, &mut log_dst).unwrap();
        for (i, &x) in pow_src.iter().enumerate() {
            assert_eq!(
                log_dst[i].to_bits(),
                l::try_log2_with::<LogExcludingOne>(x).unwrap().to_bits()
            );
        }

        let log1p_src = uniform(len, -0.5, 0.5, 200 + len as u32);
        let mut log1p_dst = vec![0.0; len];
        l::try_log1p_slice_with::<Log1pSmall>(&log1p_src, &mut log1p_dst).unwrap();
        for (i, &x) in log1p_src.iter().enumerate() {
            assert_eq!(
                log1p_dst[i].to_bits(),
                l::try_log1p_with::<Log1pSmall>(x).unwrap().to_bits()
            );
        }

        let mut cbrt_dst = vec![0.0; len];
        c::try_cbrt_slice_with::<CbrtExcludingOne>(&pow_src, &mut cbrt_dst).unwrap();
        for (i, &x) in pow_src.iter().enumerate() {
            assert_eq!(
                cbrt_dst[i].to_bits(),
                c::try_cbrt_with::<CbrtExcludingOne>(x).unwrap().to_bits()
            );
        }
    }

    // Verify rejection on a tail (< 8) when `1.0` (outside [2.0, 8.0]) is placed in the slice.
    for bad in 0..5 {
        let mut src = [3.0f32; 5];
        src[bad] = 1.0;
        let mut dst = [0.0; 5];
        assert_eq!(
            p::try_powf_slice_with::<PowExcludingOne>(&src, &mut dst, 1.5).unwrap_err(),
            SliceError {
                kind: p::PowfErrorKind::InvalidBase,
                index: bad
            }
        );
        assert_eq!(
            e::try_exp2_slice_with::<ExpClamped>(&src, &mut dst).unwrap_err(),
            SliceError {
                kind: e::ExpfErrorKind::InvalidInput,
                index: bad
            }
        );
        assert_eq!(
            l::try_ln_slice_with::<LogExcludingOne>(&src, &mut dst).unwrap_err(),
            SliceError {
                kind: l::LogfErrorKind::InvalidInput,
                index: bad
            }
        );
        assert_eq!(
            c::try_cbrt_slice_with::<CbrtExcludingOne>(&src, &mut dst).unwrap_err(),
            SliceError {
                kind: c::CbrtfErrorKind::InvalidInput,
                index: bad
            }
        );
    }

    // Bounded cbrtf crossing zero still rejects subnormals while accepting ±0.0.
    let cbrt_sub = [0.0, -0.0, f32::from_bits(1), 2.0, 4.0];
    let mut out = [0.0; 5];
    assert_eq!(
        c::try_cbrt_slice_with::<CbrtNearZero>(&cbrt_sub, &mut out).unwrap_err(),
        SliceError {
            kind: c::CbrtfErrorKind::InvalidInput,
            index: 2
        }
    );
}

#[test]
fn root_modifiers_compose_across_all_families_and_five_shapes() {
    type P = Caller<p::presets::GammaMidp>;
    type PU = Unchecked<p::presets::GammaMidp>;
    type EB = Caller<Backend<e::presets::Midp>>;
    type LB = Caller<Backend<l::presets::Midp>>;
    type CB = Caller<Backend<c::presets::Midp>>;

    const { assert!(Portability::BitExact as u8 != Portability::BackendDependent as u8) };

    let s = p::try_powf_with::<P>(0.25, 2.0).unwrap();
    let st = p::try_powf_with_token::<P, _>(ScalarToken, 0.25, 2.0).unwrap();
    let a = p::try_powf_array_with::<P, 4>([0.25; 4], 2.0).unwrap();
    let v = p::try_powf_f32x8_with::<P, _>(ScalarToken, f32x8::splat_t(ScalarToken, 0.25), 2.0)
        .unwrap()
        .to_array();
    let mut sl = [0.0; 11];
    p::try_powf_slice_with_token::<P, _>(ScalarToken, &[0.25; 11], &mut sl, 2.0).unwrap();
    assert_eq!(s.to_bits(), st.to_bits());
    assert_eq!(a[0].to_bits(), s.to_bits());
    assert_eq!(v[0].to_bits(), s.to_bits());
    assert_eq!(sl[10].to_bits(), s.to_bits());

    // Unchecked accepts without returning InvalidBase error.
    assert!(p::try_powf_with::<PU>(-1.0, 2.0).is_ok());

    let es = e::try_exp_with_token::<EB, _>(ScalarToken, 0.5).unwrap();
    let mut edst = [0.0; 9];
    e::try_exp_slice_with_token::<EB, _>(ScalarToken, &[0.5; 9], &mut edst).unwrap();
    assert_eq!(edst[8].to_bits(), es.to_bits());

    let ls = l::try_log1p_with_token::<LB, _>(ScalarToken, 0.5).unwrap();
    let mut ldst = [0.0; 9];
    l::try_log1p_slice_with_token::<LB, _>(ScalarToken, &[0.5; 9], &mut ldst).unwrap();
    assert_eq!(ldst[8].to_bits(), ls.to_bits());

    let cs = c::try_cbrt_with_token::<CB, _>(ScalarToken, 2.5).unwrap();
    let mut cdst = [0.0; 9];
    c::try_cbrt_slice_with_token::<CB, _>(ScalarToken, &[2.5; 9], &mut cdst).unwrap();
    assert_eq!(cdst[8].to_bits(), cs.to_bits());
}

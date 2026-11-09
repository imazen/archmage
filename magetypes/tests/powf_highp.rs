//! Canonical HIGHP values come from an independent exact-rational graph oracle.
use magetypes::powf::{self, PowfErrorKind, PowfExecution, PowfOptions};
mod fixtures {
    include!("common/highp_vectors.rs");
}
const FULL: u32 = PowfOptions::PORTABLE_HIGHP.bits();
const CALLER: u32 = PowfOptions::PORTABLE_HIGHP
    .with_execution(PowfExecution::Caller)
    .bits();
fn equal(actual: f32, expected: u32) {
    if expected & 0x7fffffff > 0x7f800000 {
        assert!(actual.is_nan());
    } else {
        assert_eq!(actual.to_bits(), expected);
    }
}
fn verify<const F: u32, const N: usize>() {
    for &[x, n, y] in fixtures::CASES {
        let result =
            powf::try_powf_array_with::<F, N>([f32::from_bits(x); N], f32::from_bits(n)).unwrap();
        for v in result {
            equal(v, y);
        }
    }
}
#[test]
fn canonical_scalar_and_all_array_widths() {
    verify::<FULL, 1>();
    verify::<CALLER, 1>();
    verify::<FULL, 4>();
    verify::<CALLER, 4>();
    verify::<FULL, 8>();
    verify::<CALLER, 8>();
    #[cfg(feature = "w512")]
    {
        verify::<FULL, 16>();
        verify::<CALLER, 16>();
    }
}
#[test]
fn canonical_checked_masks_and_const_assumptions() {
    const G: u32 = PowfOptions::PORTABLE_GAMMA_HIGHP.bits();
    assert_eq!(
        powf::try_powf_array_with::<G, 4>([0., 1., 4., 16.], 0.).unwrap(),
        [1.; 4]
    );
    let err =
        powf::try_powf_array_with::<G, 4>([0., -0., f32::from_bits(1), f32::NAN], 2.4).unwrap_err();
    assert_eq!(err.kind, PowfErrorKind::InvalidBase);
    assert_eq!(err.lane_mask, 14);
    assert_eq!(
        powf::try_powf_array_with::<G, 4>([1.; 4], f32::from_bits(1))
            .unwrap_err()
            .lane_mask,
        15
    );
    const A: u32 = PowfOptions::PORTABLE_GAMMA_HIGHP
        .with_validation(powf::PowfValidation::AssumeValid)
        .bits();
    for n in [0., 0.5, 2.4, 4.] {
        let xs = [0., 1. / 65536., 1., 65536.];
        assert_eq!(
            powf::try_powf_array_with::<G, 4>(xs, n)
                .unwrap()
                .map(f32::to_bits),
            powf::try_powf_array_with::<A, 4>(xs, n)
                .unwrap()
                .map(f32::to_bits)
        );
    }
    assert!(powf::try_powf_array_runtime([2.; 4], 2.4, PowfOptions::PORTABLE_GAMMA_HIGHP).is_ok());
}

// Different table indices must occupy the same pack. Broadcast-only fixtures
// cannot catch lane ordering, gather or packing mistakes.
#[archmage::magetypes(v3, neon, wasm128, scalar)]
fn vector4(token: Token, x: [f32; 4], n: f32) -> [f32; 4] {
    magetypes::simd::generic::f32x4::from_array_t(token, x)
        .try_powf_with::<CALLER>(n)
        .unwrap()
        .to_array()
}
fn mixed<const N: usize>() {
    let mut cases = fixtures::CASES.to_vec();
    cases.sort_by_key(|r| r[1]);
    for chunk in cases.chunks(N) {
        let x = core::array::from_fn(|i| f32::from_bits(chunk[i % chunk.len()][0]));
        for (i, &[_, n, expected]) in chunk.iter().enumerate() {
            equal(
                powf::try_powf_array_with::<FULL, N>(x, f32::from_bits(n)).unwrap()[i],
                expected,
            );
        }
    }
}
#[test]
fn canonical_mixed_lanes() {
    mixed::<4>();
    mixed::<8>();
    #[cfg(feature = "w512")]
    mixed::<16>();
    let mut cases = fixtures::CASES.to_vec();
    cases.sort_by_key(|r| r[1]);
    for chunk in cases.chunks(4) {
        let x = core::array::from_fn(|i| f32::from_bits(chunk[i % chunk.len()][0]));
        for (i, &[_, n, expected]) in chunk.iter().enumerate() {
            let n = f32::from_bits(n);
            equal(vector4_scalar(archmage::ScalarToken, x, n)[i], expected);
            #[cfg(target_arch = "x86_64")]
            if let Some(t) = <archmage::X64V3Token as archmage::SimdToken>::summon() {
                equal(vector4_v3(t, x, n)[i], expected);
            }
            #[cfg(target_arch = "aarch64")]
            equal(
                vector4_neon(
                    <archmage::NeonToken as archmage::SimdToken>::summon().unwrap(),
                    x,
                    n,
                )[i],
                expected,
            );
            #[cfg(target_arch = "wasm32")]
            if let Some(t) = <archmage::Wasm128Token as archmage::SimdToken>::summon() {
                equal(vector4_wasm128(t, x, n)[i], expected);
            }
        }
    }
}

fn gamma_mixed<const N: usize>() {
    const G: u32 = PowfOptions::PORTABLE_GAMMA_HIGHP.bits();
    const A: u32 = PowfOptions::PORTABLE_GAMMA_HIGHP
        .with_validation(powf::PowfValidation::AssumeValid)
        .bits();
    let mut cases: Vec<_> = fixtures::CASES
        .iter()
        .copied()
        .filter(|r| {
            let x = f32::from_bits(r[0]);
            let n = f32::from_bits(r[1]);
            (r[0] == 0 || (1. / 65536. ..=65536.).contains(&x))
                && (0. ..=4.).contains(&n)
                && (n == 0. || n.is_normal())
        })
        .collect();
    assert!(cases.len() >= 128);
    cases.sort_by_key(|r| r[1]);
    for chunk in cases.chunks(N) {
        let x = core::array::from_fn(|i| f32::from_bits(chunk[i % chunk.len()][0]));
        for (i, &[_, n, expected]) in chunk.iter().enumerate() {
            equal(
                powf::try_powf_array_with::<G, N>(x, f32::from_bits(n)).unwrap()[i],
                expected,
            );
            equal(
                powf::try_powf_array_with::<A, N>(x, f32::from_bits(n)).unwrap()[i],
                expected,
            );
        }
    }
}
#[test]
fn canonical_gamma_mixed_lanes() {
    gamma_mixed::<1>();
    gamma_mixed::<4>();
    gamma_mixed::<8>();
    #[cfg(feature = "w512")]
    gamma_mixed::<16>();
}

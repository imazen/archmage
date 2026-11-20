//! Canonical HIGHP values come from an independent exact-rational graph oracle.
#![forbid(unsafe_code)]
use magemath::{
    Execution, SimdMath, Validation,
    powf::{Powf, PowfErrorKind, PowfOptions},
};
mod fixtures {
    include!("common/highp_vectors.rs");
}

const FULL: u128 = PowfOptions::new().full_domain().highp().bits();
const CALLER_FULL: u128 = PowfOptions::new().full_domain().highp().caller().bits();

fn equal(actual: f32, expected: u32) {
    if expected & 0x7fffffff > 0x7f800000 {
        assert!(actual.is_nan());
    } else {
        assert_eq!(actual.to_bits(), expected);
    }
}

fn verify<const CFG: u128, const N: usize>() {
    for &[x, n, y] in fixtures::CASES {
        let result = Powf::<CFG>::array::<N>([f32::from_bits(x); N], f32::from_bits(n)).unwrap();
        for v in result {
            equal(v, y);
        }
    }
}

#[test]
fn canonical_scalar_and_all_array_widths() {
    verify::<FULL, 1>();
    verify::<CALLER_FULL, 1>();
    verify::<FULL, 4>();
    verify::<CALLER_FULL, 4>();
    verify::<FULL, 8>();
    verify::<CALLER_FULL, 8>();
    #[cfg(feature = "w512")]
    {
        verify::<FULL, 16>();
        verify::<CALLER_FULL, 16>();
    }
}

#[test]
fn canonical_checked_masks_and_const_assumptions() {
    const G: u128 = PowfOptions::new()
        .gamma()
        .reject_subnormals()
        .highp()
        .bits();
    const A: u128 = PowfOptions::new().gamma().highp().unchecked().bits();
    assert_eq!(
        Powf::<G>::array::<4>([0., 1., 4., 16.], 0.).unwrap(),
        [1.; 4]
    );
    let err = Powf::<G>::array::<4>([0., -0., f32::from_bits(1), f32::NAN], 2.4).unwrap_err();
    assert_eq!(err.kind, PowfErrorKind::InvalidBase);
    assert_eq!(err.lane_mask, 14);
    assert_eq!(
        Powf::<G>::array::<4>([1.; 4], f32::from_bits(1))
            .unwrap_err()
            .lane_mask,
        15
    );
    for n in [0., 0.5, 2.4, 4.] {
        let xs = [0., 1. / 65536., 1., 65536.];
        assert_eq!(
            Powf::<G>::array::<4>(xs, n).unwrap().map(f32::to_bits),
            Powf::<A>::array::<4>(xs, n).unwrap().map(f32::to_bits)
        );
    }
    assert!(
        PowfOptions::new()
            .gamma()
            .highp()
            .runtime()
            .array([2.; 4], 2.4)
            .is_ok()
    );
}

// Different table indices must occupy the same pack. Broadcast-only fixtures
// cannot catch lane ordering, gather or packing mistakes.
#[archmage::magetypes(v3, neon, wasm128, scalar)]
fn vector4(token: Token, x: [f32; 4], n: f32) -> [f32; 4] {
    magetypes::simd::generic::f32x4::from_array_t(token, x)
        .powf::<CALLER_FULL>(n)
        .to_array()
}

fn mixed<const N: usize>() {
    let mut cases = fixtures::CASES.to_vec();
    cases.sort_by_key(|r| r[1]);
    for chunk in cases.chunks(N) {
        let x = core::array::from_fn(|i| f32::from_bits(chunk[i % chunk.len()][0]));
        for (i, &[_, n, expected]) in chunk.iter().enumerate() {
            equal(
                Powf::<FULL>::array::<N>(x, f32::from_bits(n)).unwrap()[i],
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
    const G: u128 = PowfOptions::new()
        .gamma()
        .reject_subnormals()
        .highp()
        .bits();
    const A: u128 = PowfOptions::new().gamma().highp().unchecked().bits();
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
                Powf::<G>::array::<N>(x, f32::from_bits(n)).unwrap()[i],
                expected,
            );
            equal(
                Powf::<A>::array::<N>(x, f32::from_bits(n)).unwrap()[i],
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

#[test]
fn every_canonical_highp_encoding_reaches_the_same_graph() {
    const fn opts_for(domain: u32, caller: bool, unchecked: bool) -> PowfOptions {
        let b = match domain {
            0 => PowfOptions::new().highp(),
            1 => PowfOptions::new().admit_zero().highp(),
            2 => PowfOptions::new().gamma().highp(),
            _ => PowfOptions::new().full_domain().highp(),
        };
        b.execution(if caller {
            Execution::Caller
        } else {
            Execution::Upgrade
        })
        .validation(if unchecked {
            Validation::AssumeValid
        } else {
            Validation::Checked
        })
    }

    fn verify_one<const CFG: u128>(opts: PowfOptions) {
        let expected = Powf::<CFG>::array::<4>([2., 4., 8., 16.], 0.5).unwrap();
        let runtime = opts.runtime().array([2., 4., 8., 16.], 0.5).unwrap();
        assert_eq!(expected.map(f32::to_bits), runtime.map(f32::to_bits));
        for (x, value) in [2f32, 4., 8., 16.].into_iter().zip(expected) {
            let row = fixtures::CASES
                .iter()
                .find(|r| r[0] == x.to_bits() && r[1] == 0.5f32.to_bits());
            if x == 4. {
                assert_eq!(value.to_bits(), 2f32.to_bits());
            }
            if x == 16. {
                assert_eq!(value.to_bits(), 4f32.to_bits());
            }
            if let Some(r) = row {
                equal(value, r[2]);
            }
        }
    }

    macro_rules! check {
        ($(($d:literal, $c:literal, $u:literal)),+ $(,)?) => {$(
            {
                const O: PowfOptions = opts_for($d, $c, $u);
                verify_one::<{ O.bits() }>(O);
            }
        )+};
    }

    check!(
        (0, false, false),
        (1, false, false),
        (2, false, false),
        (3, false, false),
        (0, true, false),
        (1, true, false),
        (2, true, false),
        (3, true, false),
        (0, false, true),
        (1, false, true),
        (2, false, true),
        (3, false, true),
        (0, true, true),
        (1, true, true),
        (2, true, true),
        (3, true, true),
    );
}

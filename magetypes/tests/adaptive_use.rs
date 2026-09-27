//! Real macro expansion, feature boundaries, and width-independent row handling.
#![forbid(unsafe_code)]

use archmage::{ScalarToken, autoversion, incant, magetypes, rite};

#[magetypes(
    use(f32xN, f64xN, i8xN, u8xN, i16xN, u16xN, i32xN, u32xN, i64xN, u64xN),
    v4(cfg(avx512)),
    v4x(cfg(avx512)),
    v3,
    neon,
    wasm128,
    scalar
)]
fn families(token: Token) -> usize {
    let n = f32xN::LANES;
    assert_eq!(f32xN::splat(1.0).to_array(), [1.0; f32xN::LANES]);
    assert_eq!(
        f32xN::splat_with_token(token, 1.0).to_array(),
        [1.0; f32xN::LANES]
    );
    let ints: i32xN = f32xN::splat(3.0).to_i32();
    assert_eq!(f32xN::from_i32(ints).to_array(), [3.0; f32xN::LANES]);
    macro_rules! family {
        ($ty:ident, $value:expr, $lanes:expr) => {
            assert_eq!($ty::LANES, $lanes);
            assert_eq!($ty::splat($value).to_array(), [$value; $ty::LANES]);
        };
    }
    family!(f64xN, 2.0, n / 2);
    family!(i8xN, -3, n * 4);
    family!(u8xN, 3, n * 4);
    family!(i16xN, -3, n * 2);
    family!(u16xN, 3, n * 2);
    family!(i32xN, -3, n);
    family!(u32xN, 3, n);
    family!(i64xN, -3, n / 2);
    family!(u64xN, 3, n / 2);
    n
}

// No token parameter: magetypes must pass the known tier to rite.
#[magetypes(rite, use(f32xN))]
fn row(values: &mut [f32]) {
    let (chunks, tail) = f32xN::partition_slice_mut(values);
    for chunk in chunks {
        (f32xN::load(chunk) + f32xN::splat(2.0)).store(chunk);
    }
    for value in tail {
        *value += 2.0;
    }
}

#[magetypes]
fn apply(_token: Token, values: &mut [f32]) {
    incant!(row(values) without token);
}

#[autoversion(use(f32xN), v3, neon, wasm128, default)]
fn auto(values: &mut [f32]) {
    let (chunks, tail) = f32xN::partition_slice_mut(values);
    for chunk in chunks {
        (f32xN::load(chunk) + f32xN::splat(2.0)).store(chunk);
    }
    for value in tail {
        *value += 2.0;
    }
}

#[autoversion(use(f32xN), cfg(avx512), v3, scalar)]
fn gated() -> usize {
    f32xN::splat(1.0).to_array().len()
}

#[rite(default, use(f32xN, f32x8))]
fn tokenless_scalar() -> (usize, usize) {
    (f32xN::LANES, f32x8::LANES)
}

#[rite(v3, neon, wasm128, default, use(f32xN))]
fn multi() -> usize {
    f32xN::splat(1.0).to_array().len()
}

#[magetypes(rite, default, use(f32xN))]
fn default_only() -> usize {
    f32xN::zero().to_array().len()
}

fn exercise(f: impl Fn(&mut [f32])) {
    for width in 0..=65 {
        for offset in 0..4 {
            let stride = width + 3;
            let mut data = vec![-999.0; offset + 3 * stride];
            for row in 0..3 {
                let start = offset + row * stride;
                for (i, value) in data[start..start + width].iter_mut().enumerate() {
                    *value = i as f32 - 20.0;
                }
            }
            let mut expected = data.clone();
            for row in 0..3 {
                let start = offset + row * stride;
                for value in &mut expected[start..start + width] {
                    *value += 2.0;
                }
                f(&mut data[start..start + width]);
            }
            assert_eq!(data, expected, "width={width} offset={offset}");
        }
    }
}

#[test]
fn scalar_and_dispatch() {
    assert_eq!(families_scalar(ScalarToken), 4);
    assert_eq!(tokenless_scalar(), (4, 8));
    assert_eq!(multi_default(), 4);
    assert_eq!(default_only_default(), 4);
    assert_eq!(magetypes::simd::generic::f32x4::<ScalarToken>::LANES, 4);
    exercise(|v| apply_scalar(ScalarToken, v));
    exercise(|v| incant!(apply(v)));
    exercise(auto);
    let lanes = incant!(families());
    assert!(matches!(lanes, 4 | 8 | 16));
    #[cfg(not(feature = "avx512"))]
    assert_eq!(gated(), 4);
    #[cfg(feature = "avx512")]
    assert!(matches!(gated(), 4 | 8));
}

// Mandatory native/SDE execution is selected by the test caller. The normal
// portable suite above covers runtime dispatch without silently skipping tests.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use archmage::{SimdToken, X64V3Token, arcane};

    #[arcane(use(f32xN, f32x8))]
    fn boundary(_: X64V3Token) -> usize {
        assert_eq!(f32x8::splat(1.0).to_array(), [1.0; 8]);
        assert_eq!(multi_v3(), 8);
        assert_eq!(inferred(X64V3Token::from_context()), 8);
        f32xN::splat(1.0).to_array().len()
    }

    #[rite(use(f32xN))]
    fn inferred(_: X64V3Token) -> usize {
        f32xN::splat(1.0).to_array().len()
    }

    struct Methods;
    impl Methods {
        #[arcane(use(f32xN))]
        fn sibling(&self, _: X64V3Token) -> usize {
            f32xN::LANES
        }
        #[arcane(_self = Methods, use(f32xN))]
        fn nested(&self, _: X64V3Token) -> usize {
            f32xN::splat(1.0).to_array().len()
        }
    }

    #[archmage::token_target_features(v3, use(f32xN))]
    fn descriptive_helper() -> usize {
        f32xN::splat(1.0).to_array().len()
    }
    #[archmage::token_target_features_boundary(use(f32xN))]
    fn descriptive(_: X64V3Token) -> usize {
        assert_eq!(f32xN::LANES, descriptive_helper());
        f32xN::LANES
    }
    #[archmage::simd_fn(use(f32xN))]
    fn legacy(_: X64V3Token) -> usize {
        f32xN::splat(1.0).to_array().len()
    }

    pub(super) fn v3_required() {
        let token = X64V3Token::summon().expect("run with V3 CPU or SDE -hsw");
        assert_eq!(families_v3(token), magetypes::simd::v3::f32xN::LANES);
        assert_eq!(boundary(token), 8);
        assert_eq!(Methods.sibling(token), 8);
        assert_eq!(Methods.nested(token), 8);
        assert_eq!(descriptive(token), 8);
        assert_eq!(legacy(token), 8);
        exercise(|v| apply_v3(token, v));
    }
}

#[cfg(target_arch = "aarch64")]
fn neon_required() {
    use archmage::SimdToken;
    let token = archmage::NeonToken::summon().expect("NEON required");
    assert_eq!(families_neon(token), magetypes::simd::neon::f32xN::LANES);
    exercise(|v| apply_neon(token, v));
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod wide {
    use super::*;
    use archmage::SimdToken;

    #[magetypes(use(f32xN, i32xN), v4x, -scalar)]
    fn extended(token: Token) -> usize {
        let v: i32xN = f32xN::splat_with_token(token, 3.0).to_i32();
        assert_eq!(v.to_array(), [3; 16]);
        f32xN::LANES
    }

    pub(super) fn v4_required() {
        let token = archmage::X64V4Token::summon().expect("run with V4 CPU or SDE -skx");
        assert_eq!(families_v4(token), magetypes::simd::v4::f32xN::LANES);
        exercise(|v| apply_v4(token, v));
    }

    pub(super) fn v4x_required() {
        let token = archmage::X64V4xToken::summon().expect("run with V4x CPU or SDE -spr");
        assert_eq!(families_v4x(token), 16);
        assert_eq!(extended_v4x(token), magetypes::simd::v4x::f32xN::LANES);
    }
}

/// The caller selects a mandatory hardware path; the default executes scalar.
/// No detection-based early return can turn a requested hardware test into a pass.
#[test]
fn selected_backend() {
    match std::env::var("ARCHMAGE_ADAPTIVE_TEST_TIER")
        .as_deref()
        .unwrap_or("scalar")
    {
        "scalar" => {
            assert_eq!(families_scalar(ScalarToken), 4);
            exercise(|v| apply_scalar(ScalarToken, v));
        }
        #[cfg(target_arch = "x86_64")]
        "v3" => x86::v3_required(),
        #[cfg(target_arch = "aarch64")]
        "neon" => neon_required(),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4" => wide::v4_required(),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4x" => wide::v4x_required(),
        tier => panic!("requested adaptive test tier {tier} is not compiled for this target"),
    }
}

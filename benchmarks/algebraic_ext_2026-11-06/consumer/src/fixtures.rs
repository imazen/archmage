use algebraic_magetypes_prototype::{AlgebraicOps, AlgebraicOpsWithToken, AlgebraicReduce};
use archmage::prelude::*;
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn fixture_f32x4(t: Token) {
    let aa: [f32; 4] = core::array::from_fn(|i| {
        let a = (4 * i + 7) as f32;
        if i % 2 == 0 { a } else { -a }
    });
    let bb: [f32; 4] = core::array::from_fn(|i| [2., 4., 8., 16.][i % 4]);
    let a = f32x4::from_array_t(t, aa);
    let b = f32x4::from_array_t(t, bb);
    let expected: [f32; 4] = core::array::from_fn(|i| aa[i] + bb[i]);
    assert_eq!(a.algebraic_add(b).to_array(), expected);
    assert_eq!(a.algebraic_add_t(t, b).to_array(), expected);
    let expected: [f32; 4] = core::array::from_fn(|i| aa[i] - bb[i]);
    assert_eq!(a.algebraic_sub(b).to_array(), expected);
    assert_eq!(a.algebraic_sub_t(t, b).to_array(), expected);
    let expected: [f32; 4] = core::array::from_fn(|i| aa[i] * bb[i]);
    assert_eq!(a.algebraic_mul(b).to_array(), expected);
    assert_eq!(a.algebraic_mul_t(t, b).to_array(), expected);
    let expected: [f32; 4] = core::array::from_fn(|i| aa[i] / bb[i]);
    assert_eq!(a.algebraic_div(b).to_array(), expected);
    assert_eq!(a.algebraic_div_t(t, b).to_array(), expected);
    let expected: [f32; 4] = core::array::from_fn(|i| aa[i] % bb[i]);
    assert_eq!(a.algebraic_rem(b).to_array(), expected);
    assert_eq!(a.algebraic_rem_t(t, b).to_array(), expected);
    assert_eq!(a.algebraic_reduce_add(), aa.into_iter().sum::<f32>());
}
pub fn check_f32x4() -> usize {
    let mut n = 0;
    if let Some(t) = ScalarToken::summon() {
        fixture_f32x4_scalar_t(t);
        n += 10;
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        fixture_f32x4_v3_t(t);
        n += 10;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        fixture_f32x4_v4x_t(t);
        n += 10;
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        fixture_f32x4_neon_t(t);
        n += 10;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        fixture_f32x4_wasm128_t(t);
        n += 10;
    }
    n
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fixture_f32x8(t: Token) {
    let aa: [f32; 8] = core::array::from_fn(|i| {
        let a = (4 * i + 7) as f32;
        if i % 2 == 0 { a } else { -a }
    });
    let bb: [f32; 8] = core::array::from_fn(|i| [2., 4., 8., 16.][i % 4]);
    let a = f32x8::from_array_t(t, aa);
    let b = f32x8::from_array_t(t, bb);
    let expected: [f32; 8] = core::array::from_fn(|i| aa[i] + bb[i]);
    assert_eq!(a.algebraic_add(b).to_array(), expected);
    assert_eq!(a.algebraic_add_t(t, b).to_array(), expected);
    let expected: [f32; 8] = core::array::from_fn(|i| aa[i] - bb[i]);
    assert_eq!(a.algebraic_sub(b).to_array(), expected);
    assert_eq!(a.algebraic_sub_t(t, b).to_array(), expected);
    let expected: [f32; 8] = core::array::from_fn(|i| aa[i] * bb[i]);
    assert_eq!(a.algebraic_mul(b).to_array(), expected);
    assert_eq!(a.algebraic_mul_t(t, b).to_array(), expected);
    let expected: [f32; 8] = core::array::from_fn(|i| aa[i] / bb[i]);
    assert_eq!(a.algebraic_div(b).to_array(), expected);
    assert_eq!(a.algebraic_div_t(t, b).to_array(), expected);
    let expected: [f32; 8] = core::array::from_fn(|i| aa[i] % bb[i]);
    assert_eq!(a.algebraic_rem(b).to_array(), expected);
    assert_eq!(a.algebraic_rem_t(t, b).to_array(), expected);
    assert_eq!(a.algebraic_reduce_add(), aa.into_iter().sum::<f32>());
}
pub fn check_f32x8() -> usize {
    let mut n = 0;
    if let Some(t) = ScalarToken::summon() {
        fixture_f32x8_scalar_t(t);
        n += 10;
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        fixture_f32x8_v3_t(t);
        n += 10;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        fixture_f32x8_v4x_t(t);
        n += 10;
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        fixture_f32x8_neon_t(t);
        n += 10;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        fixture_f32x8_wasm128_t(t);
        n += 10;
    }
    n
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn fixture_f32x16(t: Token) {
    let aa: [f32; 16] = core::array::from_fn(|i| {
        let a = (4 * i + 7) as f32;
        if i % 2 == 0 { a } else { -a }
    });
    let bb: [f32; 16] = core::array::from_fn(|i| [2., 4., 8., 16.][i % 4]);
    let a = f32x16::from_array_t(t, aa);
    let b = f32x16::from_array_t(t, bb);
    let expected: [f32; 16] = core::array::from_fn(|i| aa[i] + bb[i]);
    assert_eq!(a.algebraic_add(b).to_array(), expected);
    assert_eq!(a.algebraic_add_t(t, b).to_array(), expected);
    let expected: [f32; 16] = core::array::from_fn(|i| aa[i] - bb[i]);
    assert_eq!(a.algebraic_sub(b).to_array(), expected);
    assert_eq!(a.algebraic_sub_t(t, b).to_array(), expected);
    let expected: [f32; 16] = core::array::from_fn(|i| aa[i] * bb[i]);
    assert_eq!(a.algebraic_mul(b).to_array(), expected);
    assert_eq!(a.algebraic_mul_t(t, b).to_array(), expected);
    let expected: [f32; 16] = core::array::from_fn(|i| aa[i] / bb[i]);
    assert_eq!(a.algebraic_div(b).to_array(), expected);
    assert_eq!(a.algebraic_div_t(t, b).to_array(), expected);
    let expected: [f32; 16] = core::array::from_fn(|i| aa[i] % bb[i]);
    assert_eq!(a.algebraic_rem(b).to_array(), expected);
    assert_eq!(a.algebraic_rem_t(t, b).to_array(), expected);
    assert_eq!(a.algebraic_reduce_add(), aa.into_iter().sum::<f32>());
}
pub fn check_f32x16() -> usize {
    let mut n = 0;
    if let Some(t) = ScalarToken::summon() {
        fixture_f32x16_scalar_t(t);
        n += 10;
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        fixture_f32x16_v3_t(t);
        n += 10;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        fixture_f32x16_v4x_t(t);
        n += 10;
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        fixture_f32x16_neon_t(t);
        n += 10;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        fixture_f32x16_wasm128_t(t);
        n += 10;
    }
    n
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x2))]
fn fixture_f64x2(t: Token) {
    let aa: [f64; 2] = core::array::from_fn(|i| {
        let a = (4 * i + 7) as f64;
        if i % 2 == 0 { a } else { -a }
    });
    let bb: [f64; 2] = core::array::from_fn(|i| [2., 4., 8., 16.][i % 4]);
    let a = f64x2::from_array_t(t, aa);
    let b = f64x2::from_array_t(t, bb);
    let expected: [f64; 2] = core::array::from_fn(|i| aa[i] + bb[i]);
    assert_eq!(a.algebraic_add(b).to_array(), expected);
    assert_eq!(a.algebraic_add_t(t, b).to_array(), expected);
    let expected: [f64; 2] = core::array::from_fn(|i| aa[i] - bb[i]);
    assert_eq!(a.algebraic_sub(b).to_array(), expected);
    assert_eq!(a.algebraic_sub_t(t, b).to_array(), expected);
    let expected: [f64; 2] = core::array::from_fn(|i| aa[i] * bb[i]);
    assert_eq!(a.algebraic_mul(b).to_array(), expected);
    assert_eq!(a.algebraic_mul_t(t, b).to_array(), expected);
    let expected: [f64; 2] = core::array::from_fn(|i| aa[i] / bb[i]);
    assert_eq!(a.algebraic_div(b).to_array(), expected);
    assert_eq!(a.algebraic_div_t(t, b).to_array(), expected);
    let expected: [f64; 2] = core::array::from_fn(|i| aa[i] % bb[i]);
    assert_eq!(a.algebraic_rem(b).to_array(), expected);
    assert_eq!(a.algebraic_rem_t(t, b).to_array(), expected);
    assert_eq!(a.algebraic_reduce_add(), aa.into_iter().sum::<f64>());
}
pub fn check_f64x2() -> usize {
    let mut n = 0;
    if let Some(t) = ScalarToken::summon() {
        fixture_f64x2_scalar_t(t);
        n += 10;
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        fixture_f64x2_v3_t(t);
        n += 10;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        fixture_f64x2_v4x_t(t);
        n += 10;
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        fixture_f64x2_neon_t(t);
        n += 10;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        fixture_f64x2_wasm128_t(t);
        n += 10;
    }
    n
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
fn fixture_f64x4(t: Token) {
    let aa: [f64; 4] = core::array::from_fn(|i| {
        let a = (4 * i + 7) as f64;
        if i % 2 == 0 { a } else { -a }
    });
    let bb: [f64; 4] = core::array::from_fn(|i| [2., 4., 8., 16.][i % 4]);
    let a = f64x4::from_array_t(t, aa);
    let b = f64x4::from_array_t(t, bb);
    let expected: [f64; 4] = core::array::from_fn(|i| aa[i] + bb[i]);
    assert_eq!(a.algebraic_add(b).to_array(), expected);
    assert_eq!(a.algebraic_add_t(t, b).to_array(), expected);
    let expected: [f64; 4] = core::array::from_fn(|i| aa[i] - bb[i]);
    assert_eq!(a.algebraic_sub(b).to_array(), expected);
    assert_eq!(a.algebraic_sub_t(t, b).to_array(), expected);
    let expected: [f64; 4] = core::array::from_fn(|i| aa[i] * bb[i]);
    assert_eq!(a.algebraic_mul(b).to_array(), expected);
    assert_eq!(a.algebraic_mul_t(t, b).to_array(), expected);
    let expected: [f64; 4] = core::array::from_fn(|i| aa[i] / bb[i]);
    assert_eq!(a.algebraic_div(b).to_array(), expected);
    assert_eq!(a.algebraic_div_t(t, b).to_array(), expected);
    let expected: [f64; 4] = core::array::from_fn(|i| aa[i] % bb[i]);
    assert_eq!(a.algebraic_rem(b).to_array(), expected);
    assert_eq!(a.algebraic_rem_t(t, b).to_array(), expected);
    assert_eq!(a.algebraic_reduce_add(), aa.into_iter().sum::<f64>());
}
pub fn check_f64x4() -> usize {
    let mut n = 0;
    if let Some(t) = ScalarToken::summon() {
        fixture_f64x4_scalar_t(t);
        n += 10;
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        fixture_f64x4_v3_t(t);
        n += 10;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        fixture_f64x4_v4x_t(t);
        n += 10;
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        fixture_f64x4_neon_t(t);
        n += 10;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        fixture_f64x4_wasm128_t(t);
        n += 10;
    }
    n
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x8))]
fn fixture_f64x8(t: Token) {
    let aa: [f64; 8] = core::array::from_fn(|i| {
        let a = (4 * i + 7) as f64;
        if i % 2 == 0 { a } else { -a }
    });
    let bb: [f64; 8] = core::array::from_fn(|i| [2., 4., 8., 16.][i % 4]);
    let a = f64x8::from_array_t(t, aa);
    let b = f64x8::from_array_t(t, bb);
    let expected: [f64; 8] = core::array::from_fn(|i| aa[i] + bb[i]);
    assert_eq!(a.algebraic_add(b).to_array(), expected);
    assert_eq!(a.algebraic_add_t(t, b).to_array(), expected);
    let expected: [f64; 8] = core::array::from_fn(|i| aa[i] - bb[i]);
    assert_eq!(a.algebraic_sub(b).to_array(), expected);
    assert_eq!(a.algebraic_sub_t(t, b).to_array(), expected);
    let expected: [f64; 8] = core::array::from_fn(|i| aa[i] * bb[i]);
    assert_eq!(a.algebraic_mul(b).to_array(), expected);
    assert_eq!(a.algebraic_mul_t(t, b).to_array(), expected);
    let expected: [f64; 8] = core::array::from_fn(|i| aa[i] / bb[i]);
    assert_eq!(a.algebraic_div(b).to_array(), expected);
    assert_eq!(a.algebraic_div_t(t, b).to_array(), expected);
    let expected: [f64; 8] = core::array::from_fn(|i| aa[i] % bb[i]);
    assert_eq!(a.algebraic_rem(b).to_array(), expected);
    assert_eq!(a.algebraic_rem_t(t, b).to_array(), expected);
    assert_eq!(a.algebraic_reduce_add(), aa.into_iter().sum::<f64>());
}
pub fn check_f64x8() -> usize {
    let mut n = 0;
    if let Some(t) = ScalarToken::summon() {
        fixture_f64x8_scalar_t(t);
        n += 10;
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        fixture_f64x8_v3_t(t);
        n += 10;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        fixture_f64x8_v4x_t(t);
        n += 10;
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        fixture_f64x8_neon_t(t);
        n += 10;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        fixture_f64x8_wasm128_t(t);
        n += 10;
    }
    n
}
pub fn check_all() -> usize {
    check_f32x4() + check_f32x8() + check_f32x16() + check_f64x2() + check_f64x4() + check_f64x8()
}

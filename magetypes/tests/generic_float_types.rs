//! Tests for generic f32x4<T>, f64x2<T>, f64x4<T> wrapper types.
//! (f32x8<T> tests are in generic_f32x8.rs)

#![cfg(target_arch = "x86_64")]

use archmage::{ScalarToken, SimdToken, X64V3Token};
use magetypes::simd::backends::{F32x4Backend, F64x2Backend, F64x4Backend};
use magetypes::simd::generic::{f32x4, f64x2, f64x4};

fn v3() -> X64V3Token {
    X64V3Token::summon().expect("Tests require AVX2+FMA (x86-64-v3)")
}

// ============================================================================
// f32x4<T> tests
// ============================================================================

#[test]
fn f32x4_size_matches_m128() {
    assert_eq!(
        core::mem::size_of::<f32x4<X64V3Token>>(),
        core::mem::size_of::<core::arch::x86_64::__m128>()
    );
}

#[test]
fn f32x4_scalar_size() {
    assert_eq!(
        core::mem::size_of::<f32x4<ScalarToken>>(),
        core::mem::size_of::<[f32; 4]>()
    );
}

#[test]
fn f32x4_basic_arithmetic() {
    let t = v3();
    let a = f32x4::from_array(t, [1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::splat(t, 10.0);
    let c = a + b;
    assert_eq!(c.to_array(), [11.0, 12.0, 13.0, 14.0]);

    let d = a * b;
    assert_eq!(d.to_array(), [10.0, 20.0, 30.0, 40.0]);

    let e = b - a;
    assert_eq!(e.to_array(), [9.0, 8.0, 7.0, 6.0]);
}

#[test]
fn f32x4_load_store_roundtrip() {
    let t = v3();
    let data = [1.5, 2.5, 3.5, 4.5];
    let v = f32x4::load(t, &data);
    let mut out = [0.0f32; 4];
    v.store(&mut out);
    assert_eq!(out, data);
}

#[test]
fn f32x4_math() {
    let t = v3();
    let a = f32x4::from_array(t, [1.0, 4.0, 9.0, 16.0]);
    assert_eq!(a.sqrt().to_array(), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!((-a).to_array(), [-1.0, -4.0, -9.0, -16.0]);
    assert_eq!((-a).abs().to_array(), [1.0, 4.0, 9.0, 16.0]);
}

#[test]
fn f32x4_floor_ceil_round() {
    let t = v3();
    let a = f32x4::from_array(t, [1.3, 2.7, -0.5, -1.8]);
    assert_eq!(a.floor().to_array(), [1.0, 2.0, -1.0, -2.0]);
    assert_eq!(a.ceil().to_array(), [2.0, 3.0, 0.0, -1.0]);
    // round ties to even
    let b = f32x4::from_array(t, [1.5, 2.5, 3.5, 4.5]);
    let r = b.round().to_array();
    assert_eq!(r, [2.0, 2.0, 4.0, 4.0]);
}

#[test]
fn f32x4_mul_add() {
    let t = v3();
    let a = f32x4::from_array(t, [1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::from_array(t, [5.0, 6.0, 7.0, 8.0]);
    let c = f32x4::from_array(t, [10.0, 20.0, 30.0, 40.0]);
    let r = a.mul_add(b, c);
    assert_eq!(r.to_array(), [15.0, 32.0, 51.0, 72.0]);
}

#[test]
fn f32x4_reductions() {
    let t = v3();
    let a = f32x4::from_array(t, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(a.reduce_add(), 10.0);
    assert_eq!(a.reduce_min(), 1.0);
    assert_eq!(a.reduce_max(), 4.0);
}

#[test]
fn f32x4_comparisons() {
    let t = v3();
    let a = f32x4::from_array(t, [1.0, 5.0, 3.0, 7.0]);
    let b = f32x4::from_array(t, [2.0, 5.0, 1.0, 8.0]);

    let lt_mask = a.simd_lt(b);
    let blended = f32x4::blend(lt_mask, a, b);
    // Where a < b, pick a; else pick b (the smaller one)
    assert_eq!(blended.to_array(), [1.0, 5.0, 1.0, 7.0]);
}

#[test]
fn f32x4_scalar_broadcast_ops() {
    let t = v3();
    let a = f32x4::from_array(t, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!((a + 10.0).to_array(), [11.0, 12.0, 13.0, 14.0]);
    assert_eq!((a * 2.0).to_array(), [2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn f32x4_indexing() {
    let t = v3();
    let a = f32x4::from_array(t, [10.0, 20.0, 30.0, 40.0]);
    assert_eq!(a[0], 10.0);
    assert_eq!(a[3], 40.0);
}

#[test]
fn f32x4_raw_m128_roundtrip() {
    let t = v3();
    let a = f32x4::from_array(t, [1.0, 2.0, 3.0, 4.0]);
    let raw = a.raw();
    let b = f32x4::from_m128(t, raw);
    assert_eq!(a.to_array(), b.to_array());
}

#[test]
fn f32x4_generic_fn() {
    fn sum_generic<T: F32x4Backend>(token: T, data: &[f32; 4]) -> f32 {
        let v = f32x4::<T>::load(token, data);
        v.reduce_add()
    }
    let t = v3();
    assert_eq!(sum_generic(t, &[1.0, 2.0, 3.0, 4.0]), 10.0);
    assert_eq!(sum_generic(ScalarToken, &[1.0, 2.0, 3.0, 4.0]), 10.0);
}

// ============================================================================
// f64x2<T> tests
// ============================================================================

#[test]
fn f64x2_size_matches_m128d() {
    assert_eq!(
        core::mem::size_of::<f64x2<X64V3Token>>(),
        core::mem::size_of::<core::arch::x86_64::__m128d>()
    );
}

#[test]
fn f64x2_scalar_size() {
    assert_eq!(
        core::mem::size_of::<f64x2<ScalarToken>>(),
        core::mem::size_of::<[f64; 2]>()
    );
}

#[test]
fn f64x2_basic_arithmetic() {
    let t = v3();
    let a = f64x2::from_array(t, [1.0, 2.0]);
    let b = f64x2::splat(t, 10.0);
    assert_eq!((a + b).to_array(), [11.0, 12.0]);
    assert_eq!((a * b).to_array(), [10.0, 20.0]);
    assert_eq!((b - a).to_array(), [9.0, 8.0]);
    assert_eq!((b / a).to_array(), [10.0, 5.0]);
}

#[test]
fn f64x2_load_store_roundtrip() {
    let t = v3();
    let data = [3.14, 2.72];
    let v = f64x2::load(t, &data);
    let mut out = [0.0f64; 2];
    v.store(&mut out);
    assert_eq!(out, data);
}

#[test]
fn f64x2_math() {
    let t = v3();
    let a = f64x2::from_array(t, [4.0, 9.0]);
    assert_eq!(a.sqrt().to_array(), [2.0, 3.0]);
    assert_eq!((-a).to_array(), [-4.0, -9.0]);
    assert_eq!((-a).abs().to_array(), [4.0, 9.0]);
}

#[test]
fn f64x2_floor_ceil_round() {
    let t = v3();
    let a = f64x2::from_array(t, [1.3, -1.8]);
    assert_eq!(a.floor().to_array(), [1.0, -2.0]);
    assert_eq!(a.ceil().to_array(), [2.0, -1.0]);
}

#[test]
fn f64x2_mul_add() {
    let t = v3();
    let a = f64x2::from_array(t, [1.0, 2.0]);
    let b = f64x2::from_array(t, [3.0, 4.0]);
    let c = f64x2::from_array(t, [10.0, 20.0]);
    assert_eq!(a.mul_add(b, c).to_array(), [13.0, 28.0]);
}

#[test]
fn f64x2_reductions() {
    let t = v3();
    let a = f64x2::from_array(t, [3.0, 7.0]);
    assert_eq!(a.reduce_add(), 10.0);
    assert_eq!(a.reduce_min(), 3.0);
    assert_eq!(a.reduce_max(), 7.0);
}

#[test]
fn f64x2_raw_m128d_roundtrip() {
    let t = v3();
    let a = f64x2::from_array(t, [1.5, 2.5]);
    let raw = a.raw();
    let b = f64x2::from_m128d(t, raw);
    assert_eq!(a.to_array(), b.to_array());
}

#[test]
fn f64x2_generic_fn() {
    fn sum_generic<T: F64x2Backend>(token: T, data: &[f64; 2]) -> f64 {
        let v = f64x2::<T>::load(token, data);
        v.reduce_add()
    }
    let t = v3();
    assert_eq!(sum_generic(t, &[3.0, 7.0]), 10.0);
    assert_eq!(sum_generic(ScalarToken, &[3.0, 7.0]), 10.0);
}

// ============================================================================
// f64x4<T> tests
// ============================================================================

#[test]
fn f64x4_size_matches_m256d() {
    assert_eq!(
        core::mem::size_of::<f64x4<X64V3Token>>(),
        core::mem::size_of::<core::arch::x86_64::__m256d>()
    );
}

#[test]
fn f64x4_scalar_size() {
    assert_eq!(
        core::mem::size_of::<f64x4<ScalarToken>>(),
        core::mem::size_of::<[f64; 4]>()
    );
}

#[test]
fn f64x4_basic_arithmetic() {
    let t = v3();
    let a = f64x4::from_array(t, [1.0, 2.0, 3.0, 4.0]);
    let b = f64x4::splat(t, 10.0);
    assert_eq!((a + b).to_array(), [11.0, 12.0, 13.0, 14.0]);
    assert_eq!((a * b).to_array(), [10.0, 20.0, 30.0, 40.0]);
    assert_eq!((b - a).to_array(), [9.0, 8.0, 7.0, 6.0]);
}

#[test]
fn f64x4_load_store_roundtrip() {
    let t = v3();
    let data = [1.1, 2.2, 3.3, 4.4];
    let v = f64x4::load(t, &data);
    let mut out = [0.0f64; 4];
    v.store(&mut out);
    assert_eq!(out, data);
}

#[test]
fn f64x4_math() {
    let t = v3();
    let a = f64x4::from_array(t, [1.0, 4.0, 9.0, 16.0]);
    assert_eq!(a.sqrt().to_array(), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!((-a).to_array(), [-1.0, -4.0, -9.0, -16.0]);
    assert_eq!((-a).abs().to_array(), [1.0, 4.0, 9.0, 16.0]);
}

#[test]
fn f64x4_floor_ceil_round() {
    let t = v3();
    let a = f64x4::from_array(t, [1.3, 2.7, -0.5, -1.8]);
    assert_eq!(a.floor().to_array(), [1.0, 2.0, -1.0, -2.0]);
    assert_eq!(a.ceil().to_array(), [2.0, 3.0, 0.0, -1.0]);
}

#[test]
fn f64x4_mul_add() {
    let t = v3();
    let a = f64x4::from_array(t, [1.0, 2.0, 3.0, 4.0]);
    let b = f64x4::from_array(t, [5.0, 6.0, 7.0, 8.0]);
    let c = f64x4::from_array(t, [10.0, 20.0, 30.0, 40.0]);
    assert_eq!(a.mul_add(b, c).to_array(), [15.0, 32.0, 51.0, 72.0]);
}

#[test]
fn f64x4_reductions() {
    let t = v3();
    let a = f64x4::from_array(t, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(a.reduce_add(), 10.0);
    assert_eq!(a.reduce_min(), 1.0);
    assert_eq!(a.reduce_max(), 4.0);
}

#[test]
fn f64x4_comparisons() {
    let t = v3();
    let a = f64x4::from_array(t, [1.0, 5.0, 3.0, 7.0]);
    let b = f64x4::from_array(t, [2.0, 5.0, 1.0, 8.0]);
    let lt_mask = a.simd_lt(b);
    let blended = f64x4::blend(lt_mask, a, b);
    assert_eq!(blended.to_array(), [1.0, 5.0, 1.0, 7.0]);
}

#[test]
fn f64x4_raw_m256d_roundtrip() {
    let t = v3();
    let a = f64x4::from_array(t, [1.0, 2.0, 3.0, 4.0]);
    let raw = a.raw();
    let b = f64x4::from_m256d(t, raw);
    assert_eq!(a.to_array(), b.to_array());
}

#[test]
fn f64x4_scalar_ops() {
    let a = f64x4::from_array(ScalarToken, [1.0, 2.0, 3.0, 4.0]);
    let b = f64x4::splat(ScalarToken, 10.0);
    assert_eq!((a + b).to_array(), [11.0, 12.0, 13.0, 14.0]);
    assert_eq!(a.reduce_add(), 10.0);
    assert_eq!(
        a.sqrt().to_array(),
        [1.0, f64::sqrt(2.0), f64::sqrt(3.0), 2.0]
    );
}

#[test]
fn f64x4_generic_fn() {
    fn sum_generic<T: F64x4Backend>(token: T, data: &[f64; 4]) -> f64 {
        let v = f64x4::<T>::load(token, data);
        v.reduce_add()
    }
    let t = v3();
    assert_eq!(sum_generic(t, &[1.0, 2.0, 3.0, 4.0]), 10.0);
    assert_eq!(sum_generic(ScalarToken, &[1.0, 2.0, 3.0, 4.0]), 10.0);
}

// ============================================================================
// Cross-backend consistency
// ============================================================================

#[test]
fn all_types_scalar_x86_agree() {
    let t = v3();

    // f32x4
    let a4 = f32x4::from_array(t, [1.0, 2.0, 3.0, 4.0]);
    let b4 = f32x4::from_array(ScalarToken, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(a4.reduce_add(), b4.reduce_add());
    assert_eq!((a4 * a4).to_array(), (b4 * b4).to_array());

    // f64x2
    let a2 = f64x2::from_array(t, [1.0, 2.0]);
    let b2 = f64x2::from_array(ScalarToken, [1.0, 2.0]);
    assert_eq!(a2.reduce_add(), b2.reduce_add());
    assert_eq!((a2 * a2).to_array(), (b2 * b2).to_array());

    // f64x4
    let a4d = f64x4::from_array(t, [1.0, 2.0, 3.0, 4.0]);
    let b4d = f64x4::from_array(ScalarToken, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(a4d.reduce_add(), b4d.reduce_add());
    assert_eq!((a4d * a4d).to_array(), (b4d * b4d).to_array());
}

//! Tests for `#[magetypes(define(...))]` — auto-inject `type f32x8 =
//! ::magetypes::simd::generic::f32x8<Token>;` aliases at the top of each
//! variant body. Eliminates the boilerplate users previously had to write
//! manually inside every `#[magetypes]` function.
//!
//! The aliases are scoped to the function body, so they don't affect
//! outer scope. `Token` in the alias RHS is substituted per tier.

use archmage::{ScalarToken, incant, magetypes};

// ============================================================================
// Basic: one type injected, used in the body without a manual alias line
// ============================================================================

#[magetypes(define(f32x8), v3, scalar)]
fn scale_impl(token: Token, plane: &mut [f32], factor: f32) {
    // `f32x8` is in scope via the `define` preamble.
    let factor_v = f32x8::splat(token, factor);
    let (chunks, tail) = f32x8::partition_slice_mut(token, plane);
    for chunk in chunks {
        (f32x8::load(token, chunk) * factor_v).store(chunk);
    }
    for v in tail {
        *v *= factor;
    }
}

pub fn scale(plane: &mut [f32], factor: f32) {
    incant!(scale_impl(plane, factor), [v3, scalar])
}

#[test]
fn define_injects_single_type() {
    let mut v = vec![1.0f32; 19];
    scale(&mut v, 3.0);
    assert!(v.iter().all(|x| (x - 3.0).abs() < 1e-6));
}

// ============================================================================
// Multiple types injected together
// ============================================================================

#[magetypes(define(f32x4, f32x8), v3, scalar)]
fn mixed_widths_impl(token: Token, data_4: &[f32; 4], data_8: &[f32; 8]) -> f32 {
    let v4 = f32x4::load(token, data_4);
    let v8 = f32x8::load(token, data_8);
    v4.reduce_add() + v8.reduce_add()
}

pub fn mixed_widths(data_4: &[f32; 4], data_8: &[f32; 8]) -> f32 {
    incant!(mixed_widths_impl(data_4, data_8), [v3, scalar])
}

#[test]
fn define_injects_multiple_types() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // reduce_add: a -> 10.0, b -> 36.0, total 46.0
    assert_eq!(mixed_widths(&a, &b), 46.0);
}

// ============================================================================
// Integer types work the same way
// ============================================================================

#[magetypes(define(u8x16, i16x8), v3, scalar)]
fn integer_ops_impl(token: Token, bytes: &[u8; 16]) -> i32 {
    let v: u8x16 = u8x16::load(token, bytes);
    // Existence-of-type test: both u8x16 and i16x8 are injected by define
    // and both compile. i16x8 is unused here; u8x16 we peek at via to_array.
    let _ = i16x8::zero(token);
    v.to_array().iter().filter(|&&b| b != 0).count() as i32
}

pub fn integer_ops(bytes: &[u8; 16]) -> i32 {
    incant!(integer_ops_impl(bytes), [v3, scalar])
}

#[test]
fn define_injects_integer_types() {
    assert_eq!(integer_ops(&[0u8; 16]), 0);
    assert_eq!(integer_ops(&[1u8; 16]), 16);
}

// ============================================================================
// Combined with rite flag — define works for rite-flavored magetypes too
// ============================================================================

#[magetypes(rite, define(f32x8), v3, scalar)]
fn rite_with_define_impl(token: Token, data: &[f32; 8]) -> f32 {
    let v = f32x8::load(token, data);
    v.reduce_add()
}

#[test]
fn define_works_with_rite_flag_scalar_variant() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert_eq!(rite_with_define_impl_scalar(ScalarToken, &data), 36.0);
}

// ============================================================================
// define(...) is scoped to the function body — no collision with outer scope
// ============================================================================

// An outer type alias named `f32x8` — our define(f32x8) shadows this locally.
#[allow(dead_code)]
type F32x8Outer = u32;

#[magetypes(define(f32x8), v3, scalar)]
fn scope_isolation_impl(token: Token, data: &[f32; 8]) -> f32 {
    let v = f32x8::load(token, data);
    v.reduce_add()
}

#[test]
fn define_is_function_local() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert_eq!(scope_isolation_impl_scalar(ScalarToken, &data), 36.0);
    // Outer alias still exists:
    let _x: F32x8Outer = 42u32;
}

// ============================================================================
// Empty define(...) list is valid — it's just a no-op
// ============================================================================
// Intentionally accepts empty — `define()` should parse cleanly even if
// semantically useless, so users can comment out items without syntax errors.

#[magetypes(define(), v3, scalar)]
fn empty_define_impl(_t: Token, x: f32) -> f32 {
    x * 2.0
}

#[test]
fn empty_define_list() {
    assert_eq!(empty_define_impl_scalar(ScalarToken, 5.0), 10.0);
}

// ============================================================================
// Order independence: define before or after tier list both work
// ============================================================================

#[magetypes(v3, define(f32x8), scalar)]
fn order_define_middle_impl(token: Token, data: &[f32; 8]) -> f32 {
    f32x8::load(token, data).reduce_add()
}

#[magetypes(v3, scalar, define(f32x8))]
fn order_define_last_impl(token: Token, data: &[f32; 8]) -> f32 {
    f32x8::load(token, data).reduce_add()
}

#[test]
fn define_can_appear_anywhere_in_attr() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert_eq!(order_define_middle_impl_scalar(ScalarToken, &data), 36.0);
    assert_eq!(order_define_last_impl_scalar(ScalarToken, &data), 36.0);
}

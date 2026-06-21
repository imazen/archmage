//! Scalar-vs-native backend parity tests.
//!
//! **Auto-generated** by `cargo xtask generate` — do not edit manually.
//!
//! For every SIMD math operation, this tests that the ScalarToken backend
//! produces identical results to the native hardware backend on edge-case
//! inputs. Catches divergences like ties-away-from-zero vs ties-to-even
//! (issue #20).

#![allow(unused_imports)]
#![allow(clippy::approx_constant)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::float_cmp)]

// ============================================================================
// Comparison helpers
// ============================================================================

/// Bit-exact f32 comparison, treating all NaN values as equal.
fn assert_f32_exact(scalar: &[f32], native: &[f32], op: &str, input: &[f32]) {
    assert_eq!(scalar.len(), native.len(), "{op}: length mismatch");
    for i in 0..scalar.len() {
        let s = scalar[i];
        let n = native[i];
        if s.is_nan() && n.is_nan() {
            continue;
        }
        if s.to_bits() == n.to_bits() {
            continue;
        }
        panic!(
            "{op} divergence at lane {i}: scalar={s} (0x{:08x}) native={n} (0x{:08x}) input={input:?}",
            s.to_bits(),
            n.to_bits()
        );
    }
}

/// Bit-exact f64 comparison, treating all NaN values as equal.
fn assert_f64_exact(scalar: &[f64], native: &[f64], op: &str, input: &[f64]) {
    assert_eq!(scalar.len(), native.len(), "{op}: length mismatch");
    for i in 0..scalar.len() {
        let s = scalar[i];
        let n = native[i];
        if s.is_nan() && n.is_nan() {
            continue;
        }
        if s.to_bits() == n.to_bits() {
            continue;
        }
        panic!(
            "{op} divergence at lane {i}: scalar={s} (0x{:016x}) native={n} (0x{:016x}) input={input:?}",
            s.to_bits(),
            n.to_bits()
        );
    }
}

/// f32 comparison with ULP tolerance, treating NaN==NaN and ±0 as equal.
fn assert_f32_ulps(scalar: &[f32], native: &[f32], op: &str, input: &[f32], max_ulps: u32) {
    assert_eq!(scalar.len(), native.len(), "{op}: length mismatch");
    for i in 0..scalar.len() {
        let s = scalar[i];
        let n = native[i];
        if s.is_nan() && n.is_nan() {
            continue;
        }
        if s.to_bits() == n.to_bits() {
            continue;
        }
        // Allow ±0 difference (FMA vs mul+add can produce different zero signs)
        if s == 0.0 && n == 0.0 {
            continue;
        }
        // Both must be finite and same sign for ULP comparison to make sense
        if s.is_nan() || n.is_nan() || s.is_infinite() || n.is_infinite() {
            panic!(
                "{op} divergence at lane {i}: scalar={s} native={n} (NaN/Inf mismatch) input={input:?}"
            );
        }
        let ulps = (s.to_bits() as i64 - n.to_bits() as i64).unsigned_abs() as u32;
        if ulps > max_ulps {
            panic!(
                "{op} divergence at lane {i}: scalar={s} native={n} ({ulps} ulps, max={max_ulps}) input={input:?}"
            );
        }
    }
}

/// f64 comparison with ULP tolerance, treating all NaN values as equal.
fn assert_f64_ulps(scalar: &[f64], native: &[f64], op: &str, input: &[f64], max_ulps: u64) {
    assert_eq!(scalar.len(), native.len(), "{op}: length mismatch");
    for i in 0..scalar.len() {
        let s = scalar[i];
        let n = native[i];
        if s.is_nan() && n.is_nan() {
            continue;
        }
        if s.to_bits() == n.to_bits() {
            continue;
        }
        if s.is_nan() || n.is_nan() || s.is_infinite() || n.is_infinite() {
            panic!(
                "{op} divergence at lane {i}: scalar={s} native={n} (NaN/Inf mismatch) input={input:?}"
            );
        }
        let ulps = (s.to_bits() as i128 - n.to_bits() as i128).unsigned_abs() as u64;
        if ulps > max_ulps {
            panic!(
                "{op} divergence at lane {i}: scalar={s} native={n} ({ulps} ulps, max={max_ulps}) input={input:?}"
            );
        }
    }
}

/// f32 comparison with relative tolerance (for approximate operations).
fn assert_f32_approx(scalar: &[f32], native: &[f32], op: &str, input: &[f32], rel_tol: f32) {
    assert_eq!(scalar.len(), native.len(), "{op}: length mismatch");
    for i in 0..scalar.len() {
        let s = scalar[i];
        let n = native[i];
        if s.is_nan() && n.is_nan() {
            continue;
        }
        if s.to_bits() == n.to_bits() {
            continue;
        }
        if s.is_nan() || n.is_nan() {
            panic!("{op} NaN mismatch at lane {i}: scalar={s} native={n} input={input:?}");
        }
        if s.is_infinite() && n.is_infinite() && s.signum() == n.signum() {
            continue;
        }
        let denom = s.abs().max(n.abs()).max(f32::MIN_POSITIVE);
        let rel_err = (s - n).abs() / denom;
        if rel_err > rel_tol {
            panic!(
                "{op} divergence at lane {i}: scalar={s} native={n} (rel_err={rel_err}, max={rel_tol}) input={input:?}"
            );
        }
    }
}

/// f32 comparison that treats ±0 as equal (for ops where signed zero differs).
fn assert_f32_signed_zero_tolerant(scalar: &[f32], native: &[f32], op: &str, input: &[f32]) {
    assert_eq!(scalar.len(), native.len(), "{op}: length mismatch");
    for i in 0..scalar.len() {
        let s = scalar[i];
        let n = native[i];
        if s.is_nan() && n.is_nan() {
            continue;
        }
        if s.to_bits() == n.to_bits() {
            continue;
        }
        // Allow ±0 difference
        if s == 0.0 && n == 0.0 {
            continue;
        }
        panic!(
            "{op} divergence at lane {i}: scalar={s} (0x{:08x}) native={n} (0x{:08x}) input={input:?}",
            s.to_bits(),
            n.to_bits()
        );
    }
}

/// f64 comparison that treats ±0 as equal.
fn assert_f64_signed_zero_tolerant(scalar: &[f64], native: &[f64], op: &str, input: &[f64]) {
    assert_eq!(scalar.len(), native.len(), "{op}: length mismatch");
    for i in 0..scalar.len() {
        let s = scalar[i];
        let n = native[i];
        if s.is_nan() && n.is_nan() {
            continue;
        }
        if s.to_bits() == n.to_bits() {
            continue;
        }
        if s == 0.0 && n == 0.0 {
            continue;
        }
        panic!(
            "{op} divergence at lane {i}: scalar={s} (0x{:016x}) native={n} (0x{:016x}) input={input:?}",
            s.to_bits(),
            n.to_bits()
        );
    }
}

/// f32 FMA comparison: allows both relative and absolute tolerance.
/// FMA (one rounding) vs separate mul+add (two roundings) can produce large relative
/// errors near zero due to catastrophic cancellation, but the absolute error is small.
fn assert_f32_fma(scalar: &[f32], native: &[f32], op: &str, input: &[f32]) {
    assert_eq!(scalar.len(), native.len(), "{op}: length mismatch");
    for i in 0..scalar.len() {
        let s = scalar[i];
        let n = native[i];
        if s.is_nan() && n.is_nan() {
            continue;
        }
        if s.to_bits() == n.to_bits() {
            continue;
        }
        if s == 0.0 && n == 0.0 {
            continue;
        } // ±0
        if s.is_nan() || n.is_nan() {
            panic!("{op} NaN mismatch at lane {i}: scalar={s} native={n} input={input:?}");
        }
        if s.is_infinite() && n.is_infinite() && s.signum() == n.signum() {
            continue;
        }
        let abs_err = (s - n).abs();
        // Allow absolute error up to 1e-6 (handles near-zero cancellation)
        if abs_err < 1e-6 {
            continue;
        }
        // Allow relative error up to 1e-4 for larger values
        let denom = s.abs().max(n.abs());
        let rel_err = abs_err / denom;
        if rel_err > 1e-4 {
            panic!(
                "{op} divergence at lane {i}: scalar={s} native={n} (abs_err={abs_err}, rel_err={rel_err}) input={input:?}"
            );
        }
    }
}

/// f64 FMA comparison.
fn assert_f64_fma(scalar: &[f64], native: &[f64], op: &str, input: &[f64]) {
    assert_eq!(scalar.len(), native.len(), "{op}: length mismatch");
    for i in 0..scalar.len() {
        let s = scalar[i];
        let n = native[i];
        if s.is_nan() && n.is_nan() {
            continue;
        }
        if s.to_bits() == n.to_bits() {
            continue;
        }
        if s == 0.0 && n == 0.0 {
            continue;
        }
        if s.is_nan() || n.is_nan() {
            panic!("{op} NaN mismatch at lane {i}: scalar={s} native={n} input={input:?}");
        }
        if s.is_infinite() && n.is_infinite() && s.signum() == n.signum() {
            continue;
        }
        let abs_err = (s - n).abs();
        if abs_err < 1e-12 {
            continue;
        }
        let denom = s.abs().max(n.abs());
        let rel_err = abs_err / denom;
        if rel_err > 1e-10 {
            panic!(
                "{op} divergence at lane {i}: scalar={s} native={n} (abs_err={abs_err}, rel_err={rel_err}) input={input:?}"
            );
        }
    }
}

/// Check if a f32 slice contains NaN or infinity.
fn has_nan_or_inf_f32(data: &[f32]) -> bool {
    data.iter().any(|x| x.is_nan() || x.is_infinite())
}

/// Check if a f64 slice contains NaN or infinity.
fn has_nan_or_inf_f64(data: &[f64]) -> bool {
    data.iter().any(|x| x.is_nan() || x.is_infinite())
}

/// Bit-exact i32 comparison.
fn assert_i32_exact(scalar: &[i32], native: &[i32], op: &str, input: &[i32]) {
    assert_eq!(
        scalar, native,
        "{op} divergence: scalar={scalar:?} native={native:?} input={input:?}"
    );
}

/// Bit-exact u32 comparison.
fn assert_u32_exact(scalar: &[u32], native: &[u32], op: &str, input: &[u32]) {
    assert_eq!(
        scalar, native,
        "{op} divergence: scalar={scalar:?} native={native:?} input={input:?}"
    );
}

/// Bit-exact i16 comparison.
fn assert_i16_exact(scalar: &[i16], native: &[i16], op: &str, input: &[i16]) {
    assert_eq!(
        scalar, native,
        "{op} divergence: scalar={scalar:?} native={native:?} input={input:?}"
    );
}

/// Bit-exact u16 comparison.
fn assert_u16_exact(scalar: &[u16], native: &[u16], op: &str, input: &[u16]) {
    assert_eq!(
        scalar, native,
        "{op} divergence: scalar={scalar:?} native={native:?} input={input:?}"
    );
}

/// Bit-exact i8 comparison.
fn assert_i8_exact(scalar: &[i8], native: &[i8], op: &str, input: &[i8]) {
    assert_eq!(
        scalar, native,
        "{op} divergence: scalar={scalar:?} native={native:?} input={input:?}"
    );
}

/// Bit-exact u8 comparison.
fn assert_u8_exact(scalar: &[u8], native: &[u8], op: &str, input: &[u8]) {
    assert_eq!(
        scalar, native,
        "{op} divergence: scalar={scalar:?} native={native:?} input={input:?}"
    );
}

// ============================================================================
// Edge-case input constants
// ============================================================================

// Designed to catch rounding, NaN propagation, sign, overflow, and boundary errors.
// Length 32 is divisible by 4, 8, and 16 for clean chunking across all vector widths.

const F32_EDGE_A: [f32; 32] = [
    // Rounding ties (the exact bug class from issue #20)
    0.5,
    -0.5,
    1.5,
    -1.5,
    2.5,
    -2.5,
    3.5,
    -3.5,
    // Zeros and signs
    0.0,
    -0.0,
    1.0,
    -1.0,
    // Special values
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::EPSILON,
    // Extremes
    f32::MIN,
    f32::MAX,
    f32::MIN_POSITIVE,
    -f32::MIN_POSITIVE,
    // Near rounding boundary (2^23 = 8388608)
    8388607.5,
    8388608.5,
    -8388607.5,
    -8388608.5,
    // Near i32 overflow boundary
    2147483520.0,
    -2147483520.0,
    2147483648.0,
    -2147483648.0,
    // Miscellaneous
    0.1,
    0.9,
    100.0,
    -100.0,
];

const F32_EDGE_B: [f32; 32] = [
    // Second operand for binary operations
    1.0,
    -1.0,
    2.0,
    -2.0,
    0.5,
    -0.5,
    0.25,
    -0.25,
    3.0,
    -3.0,
    0.0,
    -0.0,
    f32::INFINITY,
    f32::NAN,
    1.0,
    f32::NEG_INFINITY,
    f32::EPSILON,
    f32::MIN_POSITIVE,
    -f32::MIN_POSITIVE,
    f32::EPSILON,
    1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    42.0,
    -42.0,
    0.001,
    -0.001,
];

// Third operand for ternary ops (mul_add, mul_sub)
const F32_EDGE_C: [f32; 32] = [
    0.5,
    -0.5,
    0.5,
    -0.5,
    1.0,
    -1.0,
    1.0,
    -1.0,
    0.0,
    -0.0,
    0.0,
    -0.0,
    f32::NAN,
    1.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    0.0,
    0.0,
    0.0,
    0.0,
    100.0,
    -100.0,
    100.0,
    -100.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.1,
    -0.1,
    0.1,
    -0.1,
];

const F64_EDGE_A: [f64; 16] = [
    // Rounding ties
    0.5,
    -0.5,
    1.5,
    -1.5,
    2.5,
    -2.5,
    3.5,
    -3.5,
    // Special values
    0.0,
    -0.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    f64::MAX,
    f64::MIN,
    f64::MIN_POSITIVE,
];

const F64_EDGE_B: [f64; 16] = [
    1.0,
    -1.0,
    2.0,
    -2.0,
    0.5,
    -0.5,
    0.25,
    -0.25,
    3.0,
    -3.0,
    f64::NAN,
    1.0,
    f64::INFINITY,
    f64::EPSILON,
    -f64::MIN_POSITIVE,
    42.0,
];

const F64_EDGE_C: [f64; 16] = [
    0.5,
    -0.5,
    0.5,
    -0.5,
    1.0,
    -1.0,
    1.0,
    -1.0,
    0.0,
    -0.0,
    1.0,
    f64::NAN,
    0.0,
    100.0,
    -100.0,
    0.1,
];

const I32_EDGE_A: [i32; 32] = [
    0,
    1,
    -1,
    2,
    -2,
    42,
    -42,
    127,
    -128,
    255,
    -256,
    1000,
    -1000,
    i32::MAX,
    i32::MIN,
    i32::MAX - 1,
    i32::MIN + 1,
    0x7F,
    0xFF,
    0x7FFF,
    0xFFFF_u32 as i32,
    0x7FFF_FFFF,
    -0x7FFF_FFFF,
    100,
    -100,
    1024,
    -1024,
    0,
    0,
    0,
    0,
    0,
];

const I32_EDGE_B: [i32; 32] = [
    1, -1, 2, -2, 3, -3, 1, -1, 42, -42, 0, 0, 1, -1, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, 1, 1, 1,
    0, 0, 0, 0,
];

const I16_EDGE_A: [i16; 32] = [
    0,
    1,
    -1,
    2,
    -2,
    42,
    -42,
    127,
    -128,
    255,
    -256,
    1000,
    -1000,
    i16::MAX,
    i16::MIN,
    i16::MAX - 1,
    i16::MIN + 1,
    0x7F,
    -0x7F,
    0x7FFF,
    -0x7FFF,
    100,
    -100,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
];

const I16_EDGE_B: [i16; 32] = [
    1, -1, 2, -2, 3, -3, 1, -1, 42, -42, 0, 0, 1, -1, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, 1, 1, 1,
    0, 0, 0, 0,
];

// ============================================================================
// Test suite macro
// ============================================================================

macro_rules! scalar_vs_native {
    ($native_token:ty) => {
        // Re-export file-level items so nested modules can reach them via super::
        use super::*;
        use archmage::{ScalarToken, SimdToken};
        use magetypes::simd::generic;

mod f32x4_parity {
    use super::*;

    fn run_unary(
        op: &str,
        f: impl Fn(generic::f32x4<ScalarToken>, generic::f32x4<$native_token>) -> (
            [f32; 4], [f32; 4]
        ),
        cmp: impl Fn(&[f32], &[f32], &str, &[f32]),
    ) {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for chunk in super::F32_EDGE_A.chunks_exact(4) {
                let input: [f32; 4] = chunk.try_into().unwrap();
                let vs = generic::f32x4::<ScalarToken>::from_array(token_s, input);
                let vn = generic::f32x4::<$native_token>::from_array(token_n, input);
                let (s, n) = f(vs, vn);
                cmp(&s, &n, op, &input);
            }
        }
    }

    /// Like run_unary but skips chunks containing NaN or infinity.
    fn run_unary_finite(
        op: &str,
        f: impl Fn(generic::f32x4<ScalarToken>, generic::f32x4<$native_token>) -> (
            [f32; 4], [f32; 4]
        ),
        cmp: impl Fn(&[f32], &[f32], &str, &[f32]),
    ) {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for chunk in super::F32_EDGE_A.chunks_exact(4) {
                let input: [f32; 4] = chunk.try_into().unwrap();
                if super::has_nan_or_inf_f32(&input) { continue; }
                let vs = generic::f32x4::<ScalarToken>::from_array(token_s, input);
                let vn = generic::f32x4::<$native_token>::from_array(token_n, input);
                let (s, n) = f(vs, vn);
                cmp(&s, &n, op, &input);
            }
        }
    }

    fn run_binary(
        op: &str,
        f: impl Fn(
            generic::f32x4<ScalarToken>, generic::f32x4<ScalarToken>,
            generic::f32x4<$native_token>, generic::f32x4<$native_token>,
        ) -> ([f32; 4], [f32; 4]),
        cmp: impl Fn(&[f32], &[f32], &str, &[f32]),
    ) {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for (ca, cb) in super::F32_EDGE_A.chunks_exact(4).zip(super::F32_EDGE_B.chunks_exact(4)) {
                let a: [f32; 4] = ca.try_into().unwrap();
                let b: [f32; 4] = cb.try_into().unwrap();
                let as_ = generic::f32x4::<ScalarToken>::from_array(token_s, a);
                let bs = generic::f32x4::<ScalarToken>::from_array(token_s, b);
                let an = generic::f32x4::<$native_token>::from_array(token_n, a);
                let bn = generic::f32x4::<$native_token>::from_array(token_n, b);
                let (s, n) = f(as_, bs, an, bn);
                cmp(&s, &n, op, &a);
            }
        }
    }

    /// Like run_binary but skips chunks where either input contains NaN or infinity.
    fn run_binary_finite(
        op: &str,
        f: impl Fn(
            generic::f32x4<ScalarToken>, generic::f32x4<ScalarToken>,
            generic::f32x4<$native_token>, generic::f32x4<$native_token>,
        ) -> ([f32; 4], [f32; 4]),
        cmp: impl Fn(&[f32], &[f32], &str, &[f32]),
    ) {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for (ca, cb) in super::F32_EDGE_A.chunks_exact(4).zip(super::F32_EDGE_B.chunks_exact(4)) {
                let a: [f32; 4] = ca.try_into().unwrap();
                let b: [f32; 4] = cb.try_into().unwrap();
                if super::has_nan_or_inf_f32(&a) || super::has_nan_or_inf_f32(&b) { continue; }
                let as_ = generic::f32x4::<ScalarToken>::from_array(token_s, a);
                let bs = generic::f32x4::<ScalarToken>::from_array(token_s, b);
                let an = generic::f32x4::<$native_token>::from_array(token_n, a);
                let bn = generic::f32x4::<$native_token>::from_array(token_n, b);
                let (s, n) = f(as_, bs, an, bn);
                cmp(&s, &n, op, &a);
            }
        }
    }

#[test]
fn round() {
    run_unary("f32x4::round", |vs, vn| {
        (vs.round().to_array(), vn.round().to_array())
    }, super::assert_f32_exact);
}

#[test]
fn floor() {
    run_unary("f32x4::floor", |vs, vn| {
        (vs.floor().to_array(), vn.floor().to_array())
    }, super::assert_f32_exact);
}

#[test]
fn ceil() {
    run_unary("f32x4::ceil", |vs, vn| {
        (vs.ceil().to_array(), vn.ceil().to_array())
    }, super::assert_f32_exact);
}

#[test]
fn abs() {
    run_unary("f32x4::abs", |vs, vn| {
        (vs.abs().to_array(), vn.abs().to_array())
    }, super::assert_f32_exact);
}

#[test]
fn sqrt() {
    run_unary("f32x4::sqrt", |vs, vn| {
        (vs.sqrt().to_array(), vn.sqrt().to_array())
    }, super::assert_f32_exact);
}

#[test]
fn not() {
    run_unary("f32x4::not", |vs, vn| {
        (vs.not().to_array(), vn.not().to_array())
    }, super::assert_f32_exact);
}

#[test]
fn neg() {
    run_unary("f32x4::neg", |vs, vn| {
        ((-vs).to_array(), (-vn).to_array())
    }, super::assert_f32_signed_zero_tolerant);
}

#[test]
fn add() {
    run_binary("f32x4::add", |as_, bs, an, bn| {
        ((as_ + bs).to_array(), (an + bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn sub() {
    run_binary("f32x4::sub", |as_, bs, an, bn| {
        ((as_ - bs).to_array(), (an - bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn mul() {
    run_binary("f32x4::mul", |as_, bs, an, bn| {
        ((as_ * bs).to_array(), (an * bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn div() {
    run_binary("f32x4::div", |as_, bs, an, bn| {
        ((as_ / bs).to_array(), (an / bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn bitand() {
    run_binary("f32x4::bitand", |as_, bs, an, bn| {
        ((as_ & bs).to_array(), (an & bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn bitor() {
    run_binary("f32x4::bitor", |as_, bs, an, bn| {
        ((as_ | bs).to_array(), (an | bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn bitxor() {
    run_binary("f32x4::bitxor", |as_, bs, an, bn| {
        ((as_ ^ bs).to_array(), (an ^ bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn min() {
    run_binary_finite("f32x4::min", |as_, bs, an, bn| {
        (as_.min(bs).to_array(), an.min(bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn max() {
    run_binary_finite("f32x4::max", |as_, bs, an, bn| {
        (as_.max(bs).to_array(), an.max(bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn mul_add() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for ((ca, cb), cc) in super::F32_EDGE_A.chunks_exact(4)
            .zip(super::F32_EDGE_B.chunks_exact(4))
            .zip(super::F32_EDGE_C.chunks_exact(4))
        {
            let a: [f32; 4] = ca.try_into().unwrap();
            let b: [f32; 4] = cb.try_into().unwrap();
            let c: [f32; 4] = cc.try_into().unwrap();
            // Skip chunks with NaN/Inf/extreme values
            if [a.as_slice(), b.as_slice(), c.as_slice()].iter()
                .any(|arr| arr.iter().any(|x| x.is_nan() || x.is_infinite() || x.abs() > 1e30))
            { continue; }
            let as_ = generic::f32x4::<ScalarToken>::from_array(token_s, a);
            let bs = generic::f32x4::<ScalarToken>::from_array(token_s, b);
            let cs = generic::f32x4::<ScalarToken>::from_array(token_s, c);
            let an = generic::f32x4::<$native_token>::from_array(token_n, a);
            let bn = generic::f32x4::<$native_token>::from_array(token_n, b);
            let cn = generic::f32x4::<$native_token>::from_array(token_n, c);
            let s = as_.mul_add(bs, cs).to_array();
            let n = an.mul_add(bn, cn).to_array();
            // FMA vs mul+add can differ by many ULPs with catastrophic cancellation.
            // Just check that NaN/Inf agreement is maintained and finite values
            // are in the same ballpark (1e-4 relative tolerance).
            // FMA vs mul+add can differ significantly with catastrophic cancellation.
            // Use signed-zero-tolerant comparison (allows ±0 and NaN agreement).
            super::assert_f32_fma(&s, &n, "f32x4::mul_add", &a);
        }
    }
}

#[test]
fn mul_sub() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for ((ca, cb), cc) in super::F32_EDGE_A.chunks_exact(4)
            .zip(super::F32_EDGE_B.chunks_exact(4))
            .zip(super::F32_EDGE_C.chunks_exact(4))
        {
            let a: [f32; 4] = ca.try_into().unwrap();
            let b: [f32; 4] = cb.try_into().unwrap();
            let c: [f32; 4] = cc.try_into().unwrap();
            // Skip chunks with NaN/Inf/extreme values
            if [a.as_slice(), b.as_slice(), c.as_slice()].iter()
                .any(|arr| arr.iter().any(|x| x.is_nan() || x.is_infinite() || x.abs() > 1e30))
            { continue; }
            let as_ = generic::f32x4::<ScalarToken>::from_array(token_s, a);
            let bs = generic::f32x4::<ScalarToken>::from_array(token_s, b);
            let cs = generic::f32x4::<ScalarToken>::from_array(token_s, c);
            let an = generic::f32x4::<$native_token>::from_array(token_n, a);
            let bn = generic::f32x4::<$native_token>::from_array(token_n, b);
            let cn = generic::f32x4::<$native_token>::from_array(token_n, c);
            let s = as_.mul_sub(bs, cs).to_array();
            let n = an.mul_sub(bn, cn).to_array();
            // FMA vs mul+add can differ by many ULPs with catastrophic cancellation.
            // Just check that NaN/Inf agreement is maintained and finite values
            // are in the same ballpark (1e-4 relative tolerance).
            // FMA vs mul+add can differ significantly with catastrophic cancellation.
            // Use signed-zero-tolerant comparison (allows ±0 and NaN agreement).
            super::assert_f32_fma(&s, &n, "f32x4::mul_sub", &a);
        }
    }
}

#[test]
fn reduce_add() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in super::F32_EDGE_A.chunks_exact(4) {
            let input: [f32; 4] = chunk.try_into().unwrap();
            // Skip chunks with NaN/Inf or extreme magnitudes (catastrophic cancellation
            // from different FP associativity between tree and left-fold reduction)
            if input.iter().any(|x| x.is_nan() || x.is_infinite() || x.abs() > 1e9) { continue; }
            let s = generic::f32x4::<ScalarToken>::from_array(token_s, input).reduce_add();
            let n = generic::f32x4::<$native_token>::from_array(token_n, input).reduce_add();
            if s.is_nan() && n.is_nan() { continue; }
            if s.to_bits() != n.to_bits() {
                // Allow relative tolerance for FP associativity
                let denom = s.abs().max(n.abs()).max(f32::MIN_POSITIVE);
                let rel_err = (s - n).abs() / denom;
                assert!(rel_err < 1e-6,
                    "f32x4::reduce_add divergence: scalar={s} native={n} (rel_err={rel_err}) input={input:?}");
            }
        }
    }
}

#[test]
fn reduce_min() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in super::F32_EDGE_A.chunks_exact(4) {
            let input: [f32; 4] = chunk.try_into().unwrap();
            if super::has_nan_or_inf_f32(&input) { continue; }
            let s = generic::f32x4::<ScalarToken>::from_array(token_s, input).reduce_min();
            let n = generic::f32x4::<$native_token>::from_array(token_n, input).reduce_min();
            // Allow ±0 difference (hardware min may return different zero sign)
            if s == 0.0 && n == 0.0 { continue; }
            assert_eq!(s.to_bits(), n.to_bits(),
                "f32x4::reduce_min divergence: scalar={s} native={n} input={input:?}");
        }
    }
}

#[test]
fn reduce_max() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in super::F32_EDGE_A.chunks_exact(4) {
            let input: [f32; 4] = chunk.try_into().unwrap();
            if super::has_nan_or_inf_f32(&input) { continue; }
            let s = generic::f32x4::<ScalarToken>::from_array(token_s, input).reduce_max();
            let n = generic::f32x4::<$native_token>::from_array(token_n, input).reduce_max();
            if s == 0.0 && n == 0.0 { continue; }
            assert_eq!(s.to_bits(), n.to_bits(),
                "f32x4::reduce_max divergence: scalar={s} native={n} input={input:?}");
        }
    }
}

#[test]
fn simd_eq() {
    run_binary("f32x4::simd_eq", |as_, bs, an, bn| {
        (as_.simd_eq(bs).to_array(), an.simd_eq(bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn simd_lt() {
    run_binary("f32x4::simd_lt", |as_, bs, an, bn| {
        (as_.simd_lt(bs).to_array(), an.simd_lt(bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn simd_le() {
    run_binary("f32x4::simd_le", |as_, bs, an, bn| {
        (as_.simd_le(bs).to_array(), an.simd_le(bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn simd_gt() {
    run_binary("f32x4::simd_gt", |as_, bs, an, bn| {
        (as_.simd_gt(bs).to_array(), an.simd_gt(bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn simd_ge() {
    run_binary("f32x4::simd_ge", |as_, bs, an, bn| {
        (as_.simd_ge(bs).to_array(), an.simd_ge(bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn simd_ne() {
    run_binary_finite("f32x4::simd_ne", |as_, bs, an, bn| {
        (as_.simd_ne(bs).to_array(), an.simd_ne(bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn rcp_approx() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        // Use only positive, non-special values for reciprocal/rsqrt
        let safe_inputs: Vec<f32> = super::F32_EDGE_A.iter()
            .filter(|&&x| x.is_finite() && x > 0.01 && x < 1e30)
            .copied()
            .collect();
        for chunk in safe_inputs.chunks_exact(4) {
            let input: [f32; 4] = chunk.try_into().unwrap();
            let s = generic::f32x4::<ScalarToken>::from_array(token_s, input).rcp_approx().to_array();
            let n = generic::f32x4::<$native_token>::from_array(token_n, input).rcp_approx().to_array();
            super::assert_f32_approx(&s, &n, "f32x4::rcp_approx", &input, 4e-3);
        }
    }
}

#[test]
fn rsqrt_approx() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        // Use only positive, non-special values for reciprocal/rsqrt
        let safe_inputs: Vec<f32> = super::F32_EDGE_A.iter()
            .filter(|&&x| x.is_finite() && x > 0.01 && x < 1e30)
            .copied()
            .collect();
        for chunk in safe_inputs.chunks_exact(4) {
            let input: [f32; 4] = chunk.try_into().unwrap();
            let s = generic::f32x4::<ScalarToken>::from_array(token_s, input).rsqrt_approx().to_array();
            let n = generic::f32x4::<$native_token>::from_array(token_n, input).rsqrt_approx().to_array();
            super::assert_f32_approx(&s, &n, "f32x4::rsqrt_approx", &input, 4e-3);
        }
    }
}

#[test]
fn recip() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        // Use only positive, non-special values for reciprocal/rsqrt
        let safe_inputs: Vec<f32> = super::F32_EDGE_A.iter()
            .filter(|&&x| x.is_finite() && x > 0.01 && x < 1e30)
            .copied()
            .collect();
        for chunk in safe_inputs.chunks_exact(4) {
            let input: [f32; 4] = chunk.try_into().unwrap();
            let s = generic::f32x4::<ScalarToken>::from_array(token_s, input).recip().to_array();
            let n = generic::f32x4::<$native_token>::from_array(token_n, input).recip().to_array();
            super::assert_f32_approx(&s, &n, "f32x4::recip", &input, 1e-5);
        }
    }
}

#[test]
fn rsqrt() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        // Use only positive, non-special values for reciprocal/rsqrt
        let safe_inputs: Vec<f32> = super::F32_EDGE_A.iter()
            .filter(|&&x| x.is_finite() && x > 0.01 && x < 1e30)
            .copied()
            .collect();
        for chunk in safe_inputs.chunks_exact(4) {
            let input: [f32; 4] = chunk.try_into().unwrap();
            let s = generic::f32x4::<ScalarToken>::from_array(token_s, input).rsqrt().to_array();
            let n = generic::f32x4::<$native_token>::from_array(token_n, input).rsqrt().to_array();
            super::assert_f32_approx(&s, &n, "f32x4::rsqrt", &input, 1e-5);
        }
    }
}

        }

mod f32x8_parity {
    use super::*;

    fn run_unary(
        op: &str,
        f: impl Fn(generic::f32x8<ScalarToken>, generic::f32x8<$native_token>) -> (
            [f32; 8], [f32; 8]
        ),
        cmp: impl Fn(&[f32], &[f32], &str, &[f32]),
    ) {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for chunk in super::F32_EDGE_A.chunks_exact(8) {
                let input: [f32; 8] = chunk.try_into().unwrap();
                let vs = generic::f32x8::<ScalarToken>::from_array(token_s, input);
                let vn = generic::f32x8::<$native_token>::from_array(token_n, input);
                let (s, n) = f(vs, vn);
                cmp(&s, &n, op, &input);
            }
        }
    }

    /// Like run_unary but skips chunks containing NaN or infinity.
    fn run_unary_finite(
        op: &str,
        f: impl Fn(generic::f32x8<ScalarToken>, generic::f32x8<$native_token>) -> (
            [f32; 8], [f32; 8]
        ),
        cmp: impl Fn(&[f32], &[f32], &str, &[f32]),
    ) {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for chunk in super::F32_EDGE_A.chunks_exact(8) {
                let input: [f32; 8] = chunk.try_into().unwrap();
                if super::has_nan_or_inf_f32(&input) { continue; }
                let vs = generic::f32x8::<ScalarToken>::from_array(token_s, input);
                let vn = generic::f32x8::<$native_token>::from_array(token_n, input);
                let (s, n) = f(vs, vn);
                cmp(&s, &n, op, &input);
            }
        }
    }

    fn run_binary(
        op: &str,
        f: impl Fn(
            generic::f32x8<ScalarToken>, generic::f32x8<ScalarToken>,
            generic::f32x8<$native_token>, generic::f32x8<$native_token>,
        ) -> ([f32; 8], [f32; 8]),
        cmp: impl Fn(&[f32], &[f32], &str, &[f32]),
    ) {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for (ca, cb) in super::F32_EDGE_A.chunks_exact(8).zip(super::F32_EDGE_B.chunks_exact(8)) {
                let a: [f32; 8] = ca.try_into().unwrap();
                let b: [f32; 8] = cb.try_into().unwrap();
                let as_ = generic::f32x8::<ScalarToken>::from_array(token_s, a);
                let bs = generic::f32x8::<ScalarToken>::from_array(token_s, b);
                let an = generic::f32x8::<$native_token>::from_array(token_n, a);
                let bn = generic::f32x8::<$native_token>::from_array(token_n, b);
                let (s, n) = f(as_, bs, an, bn);
                cmp(&s, &n, op, &a);
            }
        }
    }

    /// Like run_binary but skips chunks where either input contains NaN or infinity.
    fn run_binary_finite(
        op: &str,
        f: impl Fn(
            generic::f32x8<ScalarToken>, generic::f32x8<ScalarToken>,
            generic::f32x8<$native_token>, generic::f32x8<$native_token>,
        ) -> ([f32; 8], [f32; 8]),
        cmp: impl Fn(&[f32], &[f32], &str, &[f32]),
    ) {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for (ca, cb) in super::F32_EDGE_A.chunks_exact(8).zip(super::F32_EDGE_B.chunks_exact(8)) {
                let a: [f32; 8] = ca.try_into().unwrap();
                let b: [f32; 8] = cb.try_into().unwrap();
                if super::has_nan_or_inf_f32(&a) || super::has_nan_or_inf_f32(&b) { continue; }
                let as_ = generic::f32x8::<ScalarToken>::from_array(token_s, a);
                let bs = generic::f32x8::<ScalarToken>::from_array(token_s, b);
                let an = generic::f32x8::<$native_token>::from_array(token_n, a);
                let bn = generic::f32x8::<$native_token>::from_array(token_n, b);
                let (s, n) = f(as_, bs, an, bn);
                cmp(&s, &n, op, &a);
            }
        }
    }

#[test]
fn round() {
    run_unary("f32x8::round", |vs, vn| {
        (vs.round().to_array(), vn.round().to_array())
    }, super::assert_f32_exact);
}

#[test]
fn floor() {
    run_unary("f32x8::floor", |vs, vn| {
        (vs.floor().to_array(), vn.floor().to_array())
    }, super::assert_f32_exact);
}

#[test]
fn ceil() {
    run_unary("f32x8::ceil", |vs, vn| {
        (vs.ceil().to_array(), vn.ceil().to_array())
    }, super::assert_f32_exact);
}

#[test]
fn abs() {
    run_unary("f32x8::abs", |vs, vn| {
        (vs.abs().to_array(), vn.abs().to_array())
    }, super::assert_f32_exact);
}

#[test]
fn sqrt() {
    run_unary("f32x8::sqrt", |vs, vn| {
        (vs.sqrt().to_array(), vn.sqrt().to_array())
    }, super::assert_f32_exact);
}

#[test]
fn not() {
    run_unary("f32x8::not", |vs, vn| {
        (vs.not().to_array(), vn.not().to_array())
    }, super::assert_f32_exact);
}

#[test]
fn neg() {
    run_unary("f32x8::neg", |vs, vn| {
        ((-vs).to_array(), (-vn).to_array())
    }, super::assert_f32_signed_zero_tolerant);
}

#[test]
fn add() {
    run_binary("f32x8::add", |as_, bs, an, bn| {
        ((as_ + bs).to_array(), (an + bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn sub() {
    run_binary("f32x8::sub", |as_, bs, an, bn| {
        ((as_ - bs).to_array(), (an - bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn mul() {
    run_binary("f32x8::mul", |as_, bs, an, bn| {
        ((as_ * bs).to_array(), (an * bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn div() {
    run_binary("f32x8::div", |as_, bs, an, bn| {
        ((as_ / bs).to_array(), (an / bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn bitand() {
    run_binary("f32x8::bitand", |as_, bs, an, bn| {
        ((as_ & bs).to_array(), (an & bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn bitor() {
    run_binary("f32x8::bitor", |as_, bs, an, bn| {
        ((as_ | bs).to_array(), (an | bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn bitxor() {
    run_binary("f32x8::bitxor", |as_, bs, an, bn| {
        ((as_ ^ bs).to_array(), (an ^ bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn min() {
    run_binary_finite("f32x8::min", |as_, bs, an, bn| {
        (as_.min(bs).to_array(), an.min(bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn max() {
    run_binary_finite("f32x8::max", |as_, bs, an, bn| {
        (as_.max(bs).to_array(), an.max(bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn mul_add() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for ((ca, cb), cc) in super::F32_EDGE_A.chunks_exact(8)
            .zip(super::F32_EDGE_B.chunks_exact(8))
            .zip(super::F32_EDGE_C.chunks_exact(8))
        {
            let a: [f32; 8] = ca.try_into().unwrap();
            let b: [f32; 8] = cb.try_into().unwrap();
            let c: [f32; 8] = cc.try_into().unwrap();
            // Skip chunks with NaN/Inf/extreme values
            if [a.as_slice(), b.as_slice(), c.as_slice()].iter()
                .any(|arr| arr.iter().any(|x| x.is_nan() || x.is_infinite() || x.abs() > 1e30))
            { continue; }
            let as_ = generic::f32x8::<ScalarToken>::from_array(token_s, a);
            let bs = generic::f32x8::<ScalarToken>::from_array(token_s, b);
            let cs = generic::f32x8::<ScalarToken>::from_array(token_s, c);
            let an = generic::f32x8::<$native_token>::from_array(token_n, a);
            let bn = generic::f32x8::<$native_token>::from_array(token_n, b);
            let cn = generic::f32x8::<$native_token>::from_array(token_n, c);
            let s = as_.mul_add(bs, cs).to_array();
            let n = an.mul_add(bn, cn).to_array();
            // FMA vs mul+add can differ by many ULPs with catastrophic cancellation.
            // Just check that NaN/Inf agreement is maintained and finite values
            // are in the same ballpark (1e-4 relative tolerance).
            // FMA vs mul+add can differ significantly with catastrophic cancellation.
            // Use signed-zero-tolerant comparison (allows ±0 and NaN agreement).
            super::assert_f32_fma(&s, &n, "f32x8::mul_add", &a);
        }
    }
}

#[test]
fn mul_sub() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for ((ca, cb), cc) in super::F32_EDGE_A.chunks_exact(8)
            .zip(super::F32_EDGE_B.chunks_exact(8))
            .zip(super::F32_EDGE_C.chunks_exact(8))
        {
            let a: [f32; 8] = ca.try_into().unwrap();
            let b: [f32; 8] = cb.try_into().unwrap();
            let c: [f32; 8] = cc.try_into().unwrap();
            // Skip chunks with NaN/Inf/extreme values
            if [a.as_slice(), b.as_slice(), c.as_slice()].iter()
                .any(|arr| arr.iter().any(|x| x.is_nan() || x.is_infinite() || x.abs() > 1e30))
            { continue; }
            let as_ = generic::f32x8::<ScalarToken>::from_array(token_s, a);
            let bs = generic::f32x8::<ScalarToken>::from_array(token_s, b);
            let cs = generic::f32x8::<ScalarToken>::from_array(token_s, c);
            let an = generic::f32x8::<$native_token>::from_array(token_n, a);
            let bn = generic::f32x8::<$native_token>::from_array(token_n, b);
            let cn = generic::f32x8::<$native_token>::from_array(token_n, c);
            let s = as_.mul_sub(bs, cs).to_array();
            let n = an.mul_sub(bn, cn).to_array();
            // FMA vs mul+add can differ by many ULPs with catastrophic cancellation.
            // Just check that NaN/Inf agreement is maintained and finite values
            // are in the same ballpark (1e-4 relative tolerance).
            // FMA vs mul+add can differ significantly with catastrophic cancellation.
            // Use signed-zero-tolerant comparison (allows ±0 and NaN agreement).
            super::assert_f32_fma(&s, &n, "f32x8::mul_sub", &a);
        }
    }
}

#[test]
fn reduce_add() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in super::F32_EDGE_A.chunks_exact(8) {
            let input: [f32; 8] = chunk.try_into().unwrap();
            // Skip chunks with NaN/Inf or extreme magnitudes (catastrophic cancellation
            // from different FP associativity between tree and left-fold reduction)
            if input.iter().any(|x| x.is_nan() || x.is_infinite() || x.abs() > 1e9) { continue; }
            let s = generic::f32x8::<ScalarToken>::from_array(token_s, input).reduce_add();
            let n = generic::f32x8::<$native_token>::from_array(token_n, input).reduce_add();
            if s.is_nan() && n.is_nan() { continue; }
            if s.to_bits() != n.to_bits() {
                // Allow relative tolerance for FP associativity
                let denom = s.abs().max(n.abs()).max(f32::MIN_POSITIVE);
                let rel_err = (s - n).abs() / denom;
                assert!(rel_err < 1e-6,
                    "f32x8::reduce_add divergence: scalar={s} native={n} (rel_err={rel_err}) input={input:?}");
            }
        }
    }
}

#[test]
fn reduce_min() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in super::F32_EDGE_A.chunks_exact(8) {
            let input: [f32; 8] = chunk.try_into().unwrap();
            if super::has_nan_or_inf_f32(&input) { continue; }
            let s = generic::f32x8::<ScalarToken>::from_array(token_s, input).reduce_min();
            let n = generic::f32x8::<$native_token>::from_array(token_n, input).reduce_min();
            // Allow ±0 difference (hardware min may return different zero sign)
            if s == 0.0 && n == 0.0 { continue; }
            assert_eq!(s.to_bits(), n.to_bits(),
                "f32x8::reduce_min divergence: scalar={s} native={n} input={input:?}");
        }
    }
}

#[test]
fn reduce_max() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in super::F32_EDGE_A.chunks_exact(8) {
            let input: [f32; 8] = chunk.try_into().unwrap();
            if super::has_nan_or_inf_f32(&input) { continue; }
            let s = generic::f32x8::<ScalarToken>::from_array(token_s, input).reduce_max();
            let n = generic::f32x8::<$native_token>::from_array(token_n, input).reduce_max();
            if s == 0.0 && n == 0.0 { continue; }
            assert_eq!(s.to_bits(), n.to_bits(),
                "f32x8::reduce_max divergence: scalar={s} native={n} input={input:?}");
        }
    }
}

#[test]
fn simd_eq() {
    run_binary("f32x8::simd_eq", |as_, bs, an, bn| {
        (as_.simd_eq(bs).to_array(), an.simd_eq(bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn simd_lt() {
    run_binary("f32x8::simd_lt", |as_, bs, an, bn| {
        (as_.simd_lt(bs).to_array(), an.simd_lt(bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn simd_le() {
    run_binary("f32x8::simd_le", |as_, bs, an, bn| {
        (as_.simd_le(bs).to_array(), an.simd_le(bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn simd_gt() {
    run_binary("f32x8::simd_gt", |as_, bs, an, bn| {
        (as_.simd_gt(bs).to_array(), an.simd_gt(bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn simd_ge() {
    run_binary("f32x8::simd_ge", |as_, bs, an, bn| {
        (as_.simd_ge(bs).to_array(), an.simd_ge(bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn simd_ne() {
    run_binary_finite("f32x8::simd_ne", |as_, bs, an, bn| {
        (as_.simd_ne(bs).to_array(), an.simd_ne(bn).to_array())
    }, super::assert_f32_exact);
}

#[test]
fn rcp_approx() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        // Use only positive, non-special values for reciprocal/rsqrt
        let safe_inputs: Vec<f32> = super::F32_EDGE_A.iter()
            .filter(|&&x| x.is_finite() && x > 0.01 && x < 1e30)
            .copied()
            .collect();
        for chunk in safe_inputs.chunks_exact(8) {
            let input: [f32; 8] = chunk.try_into().unwrap();
            let s = generic::f32x8::<ScalarToken>::from_array(token_s, input).rcp_approx().to_array();
            let n = generic::f32x8::<$native_token>::from_array(token_n, input).rcp_approx().to_array();
            super::assert_f32_approx(&s, &n, "f32x8::rcp_approx", &input, 4e-3);
        }
    }
}

#[test]
fn rsqrt_approx() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        // Use only positive, non-special values for reciprocal/rsqrt
        let safe_inputs: Vec<f32> = super::F32_EDGE_A.iter()
            .filter(|&&x| x.is_finite() && x > 0.01 && x < 1e30)
            .copied()
            .collect();
        for chunk in safe_inputs.chunks_exact(8) {
            let input: [f32; 8] = chunk.try_into().unwrap();
            let s = generic::f32x8::<ScalarToken>::from_array(token_s, input).rsqrt_approx().to_array();
            let n = generic::f32x8::<$native_token>::from_array(token_n, input).rsqrt_approx().to_array();
            super::assert_f32_approx(&s, &n, "f32x8::rsqrt_approx", &input, 4e-3);
        }
    }
}

#[test]
fn recip() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        // Use only positive, non-special values for reciprocal/rsqrt
        let safe_inputs: Vec<f32> = super::F32_EDGE_A.iter()
            .filter(|&&x| x.is_finite() && x > 0.01 && x < 1e30)
            .copied()
            .collect();
        for chunk in safe_inputs.chunks_exact(8) {
            let input: [f32; 8] = chunk.try_into().unwrap();
            let s = generic::f32x8::<ScalarToken>::from_array(token_s, input).recip().to_array();
            let n = generic::f32x8::<$native_token>::from_array(token_n, input).recip().to_array();
            super::assert_f32_approx(&s, &n, "f32x8::recip", &input, 1e-5);
        }
    }
}

#[test]
fn rsqrt() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        // Use only positive, non-special values for reciprocal/rsqrt
        let safe_inputs: Vec<f32> = super::F32_EDGE_A.iter()
            .filter(|&&x| x.is_finite() && x > 0.01 && x < 1e30)
            .copied()
            .collect();
        for chunk in safe_inputs.chunks_exact(8) {
            let input: [f32; 8] = chunk.try_into().unwrap();
            let s = generic::f32x8::<ScalarToken>::from_array(token_s, input).rsqrt().to_array();
            let n = generic::f32x8::<$native_token>::from_array(token_n, input).rsqrt().to_array();
            super::assert_f32_approx(&s, &n, "f32x8::rsqrt", &input, 1e-5);
        }
    }
}

        }

mod f64x2_parity {
    use super::*;

    fn run_unary(
        op: &str,
        f: impl Fn(generic::f64x2<ScalarToken>, generic::f64x2<$native_token>) -> (
            [f64; 2], [f64; 2]
        ),
        cmp: impl Fn(&[f64], &[f64], &str, &[f64]),
    ) {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for chunk in super::F64_EDGE_A.chunks_exact(2) {
                let input: [f64; 2] = chunk.try_into().unwrap();
                let vs = generic::f64x2::<ScalarToken>::from_array(token_s, input);
                let vn = generic::f64x2::<$native_token>::from_array(token_n, input);
                let (s, n) = f(vs, vn);
                cmp(&s, &n, op, &input);
            }
        }
    }

    /// Like run_unary but skips chunks containing NaN or infinity.
    fn run_unary_finite(
        op: &str,
        f: impl Fn(generic::f64x2<ScalarToken>, generic::f64x2<$native_token>) -> (
            [f64; 2], [f64; 2]
        ),
        cmp: impl Fn(&[f64], &[f64], &str, &[f64]),
    ) {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for chunk in super::F64_EDGE_A.chunks_exact(2) {
                let input: [f64; 2] = chunk.try_into().unwrap();
                if super::has_nan_or_inf_f64(&input) { continue; }
                let vs = generic::f64x2::<ScalarToken>::from_array(token_s, input);
                let vn = generic::f64x2::<$native_token>::from_array(token_n, input);
                let (s, n) = f(vs, vn);
                cmp(&s, &n, op, &input);
            }
        }
    }

    fn run_binary(
        op: &str,
        f: impl Fn(
            generic::f64x2<ScalarToken>, generic::f64x2<ScalarToken>,
            generic::f64x2<$native_token>, generic::f64x2<$native_token>,
        ) -> ([f64; 2], [f64; 2]),
        cmp: impl Fn(&[f64], &[f64], &str, &[f64]),
    ) {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for (ca, cb) in super::F64_EDGE_A.chunks_exact(2).zip(super::F64_EDGE_B.chunks_exact(2)) {
                let a: [f64; 2] = ca.try_into().unwrap();
                let b: [f64; 2] = cb.try_into().unwrap();
                let as_ = generic::f64x2::<ScalarToken>::from_array(token_s, a);
                let bs = generic::f64x2::<ScalarToken>::from_array(token_s, b);
                let an = generic::f64x2::<$native_token>::from_array(token_n, a);
                let bn = generic::f64x2::<$native_token>::from_array(token_n, b);
                let (s, n) = f(as_, bs, an, bn);
                cmp(&s, &n, op, &a);
            }
        }
    }

    /// Like run_binary but skips chunks where either input contains NaN or infinity.
    fn run_binary_finite(
        op: &str,
        f: impl Fn(
            generic::f64x2<ScalarToken>, generic::f64x2<ScalarToken>,
            generic::f64x2<$native_token>, generic::f64x2<$native_token>,
        ) -> ([f64; 2], [f64; 2]),
        cmp: impl Fn(&[f64], &[f64], &str, &[f64]),
    ) {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for (ca, cb) in super::F64_EDGE_A.chunks_exact(2).zip(super::F64_EDGE_B.chunks_exact(2)) {
                let a: [f64; 2] = ca.try_into().unwrap();
                let b: [f64; 2] = cb.try_into().unwrap();
                if super::has_nan_or_inf_f64(&a) || super::has_nan_or_inf_f64(&b) { continue; }
                let as_ = generic::f64x2::<ScalarToken>::from_array(token_s, a);
                let bs = generic::f64x2::<ScalarToken>::from_array(token_s, b);
                let an = generic::f64x2::<$native_token>::from_array(token_n, a);
                let bn = generic::f64x2::<$native_token>::from_array(token_n, b);
                let (s, n) = f(as_, bs, an, bn);
                cmp(&s, &n, op, &a);
            }
        }
    }

#[test]
fn round() {
    run_unary("f64x2::round", |vs, vn| {
        (vs.round().to_array(), vn.round().to_array())
    }, super::assert_f64_exact);
}

#[test]
fn floor() {
    run_unary("f64x2::floor", |vs, vn| {
        (vs.floor().to_array(), vn.floor().to_array())
    }, super::assert_f64_exact);
}

#[test]
fn ceil() {
    run_unary("f64x2::ceil", |vs, vn| {
        (vs.ceil().to_array(), vn.ceil().to_array())
    }, super::assert_f64_exact);
}

#[test]
fn abs() {
    run_unary("f64x2::abs", |vs, vn| {
        (vs.abs().to_array(), vn.abs().to_array())
    }, super::assert_f64_exact);
}

#[test]
fn sqrt() {
    run_unary("f64x2::sqrt", |vs, vn| {
        (vs.sqrt().to_array(), vn.sqrt().to_array())
    }, super::assert_f64_exact);
}

#[test]
fn not() {
    run_unary("f64x2::not", |vs, vn| {
        (vs.not().to_array(), vn.not().to_array())
    }, super::assert_f64_exact);
}

#[test]
fn neg() {
    run_unary("f64x2::neg", |vs, vn| {
        ((-vs).to_array(), (-vn).to_array())
    }, super::assert_f64_signed_zero_tolerant);
}

#[test]
fn add() {
    run_binary("f64x2::add", |as_, bs, an, bn| {
        ((as_ + bs).to_array(), (an + bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn sub() {
    run_binary("f64x2::sub", |as_, bs, an, bn| {
        ((as_ - bs).to_array(), (an - bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn mul() {
    run_binary("f64x2::mul", |as_, bs, an, bn| {
        ((as_ * bs).to_array(), (an * bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn div() {
    run_binary("f64x2::div", |as_, bs, an, bn| {
        ((as_ / bs).to_array(), (an / bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn bitand() {
    run_binary("f64x2::bitand", |as_, bs, an, bn| {
        ((as_ & bs).to_array(), (an & bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn bitor() {
    run_binary("f64x2::bitor", |as_, bs, an, bn| {
        ((as_ | bs).to_array(), (an | bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn bitxor() {
    run_binary("f64x2::bitxor", |as_, bs, an, bn| {
        ((as_ ^ bs).to_array(), (an ^ bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn min() {
    run_binary_finite("f64x2::min", |as_, bs, an, bn| {
        (as_.min(bs).to_array(), an.min(bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn max() {
    run_binary_finite("f64x2::max", |as_, bs, an, bn| {
        (as_.max(bs).to_array(), an.max(bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn mul_add() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for ((ca, cb), cc) in super::F64_EDGE_A.chunks_exact(2)
            .zip(super::F64_EDGE_B.chunks_exact(2))
            .zip(super::F64_EDGE_C.chunks_exact(2))
        {
            let a: [f64; 2] = ca.try_into().unwrap();
            let b: [f64; 2] = cb.try_into().unwrap();
            let c: [f64; 2] = cc.try_into().unwrap();
            // Skip chunks with NaN/Inf/extreme values
            if [a.as_slice(), b.as_slice(), c.as_slice()].iter()
                .any(|arr| arr.iter().any(|x| x.is_nan() || x.is_infinite() || x.abs() > 1e30))
            { continue; }
            let as_ = generic::f64x2::<ScalarToken>::from_array(token_s, a);
            let bs = generic::f64x2::<ScalarToken>::from_array(token_s, b);
            let cs = generic::f64x2::<ScalarToken>::from_array(token_s, c);
            let an = generic::f64x2::<$native_token>::from_array(token_n, a);
            let bn = generic::f64x2::<$native_token>::from_array(token_n, b);
            let cn = generic::f64x2::<$native_token>::from_array(token_n, c);
            let s = as_.mul_add(bs, cs).to_array();
            let n = an.mul_add(bn, cn).to_array();
            // FMA vs mul+add can differ by many ULPs with catastrophic cancellation.
            // Just check that NaN/Inf agreement is maintained and finite values
            // are in the same ballpark (1e-4 relative tolerance).
            // FMA vs mul+add can differ significantly with catastrophic cancellation.
            // Use signed-zero-tolerant comparison (allows ±0 and NaN agreement).
            super::assert_f64_fma(&s, &n, "f64x2::mul_add", &a);
        }
    }
}

#[test]
fn mul_sub() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for ((ca, cb), cc) in super::F64_EDGE_A.chunks_exact(2)
            .zip(super::F64_EDGE_B.chunks_exact(2))
            .zip(super::F64_EDGE_C.chunks_exact(2))
        {
            let a: [f64; 2] = ca.try_into().unwrap();
            let b: [f64; 2] = cb.try_into().unwrap();
            let c: [f64; 2] = cc.try_into().unwrap();
            // Skip chunks with NaN/Inf/extreme values
            if [a.as_slice(), b.as_slice(), c.as_slice()].iter()
                .any(|arr| arr.iter().any(|x| x.is_nan() || x.is_infinite() || x.abs() > 1e30))
            { continue; }
            let as_ = generic::f64x2::<ScalarToken>::from_array(token_s, a);
            let bs = generic::f64x2::<ScalarToken>::from_array(token_s, b);
            let cs = generic::f64x2::<ScalarToken>::from_array(token_s, c);
            let an = generic::f64x2::<$native_token>::from_array(token_n, a);
            let bn = generic::f64x2::<$native_token>::from_array(token_n, b);
            let cn = generic::f64x2::<$native_token>::from_array(token_n, c);
            let s = as_.mul_sub(bs, cs).to_array();
            let n = an.mul_sub(bn, cn).to_array();
            // FMA vs mul+add can differ by many ULPs with catastrophic cancellation.
            // Just check that NaN/Inf agreement is maintained and finite values
            // are in the same ballpark (1e-4 relative tolerance).
            // FMA vs mul+add can differ significantly with catastrophic cancellation.
            // Use signed-zero-tolerant comparison (allows ±0 and NaN agreement).
            super::assert_f64_fma(&s, &n, "f64x2::mul_sub", &a);
        }
    }
}

#[test]
fn reduce_add() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in super::F64_EDGE_A.chunks_exact(2) {
            let input: [f64; 2] = chunk.try_into().unwrap();
            // Skip chunks with NaN/Inf or extreme magnitudes (catastrophic cancellation
            // from different FP associativity between tree and left-fold reduction)
            if input.iter().any(|x| x.is_nan() || x.is_infinite() || x.abs() > 1e9) { continue; }
            let s = generic::f64x2::<ScalarToken>::from_array(token_s, input).reduce_add();
            let n = generic::f64x2::<$native_token>::from_array(token_n, input).reduce_add();
            if s.is_nan() && n.is_nan() { continue; }
            if s.to_bits() != n.to_bits() {
                // Allow relative tolerance for FP associativity
                let denom = s.abs().max(n.abs()).max(f64::MIN_POSITIVE);
                let rel_err = (s - n).abs() / denom;
                assert!(rel_err < 1e-6,
                    "f64x2::reduce_add divergence: scalar={s} native={n} (rel_err={rel_err}) input={input:?}");
            }
        }
    }
}

#[test]
fn reduce_min() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in super::F64_EDGE_A.chunks_exact(2) {
            let input: [f64; 2] = chunk.try_into().unwrap();
            if super::has_nan_or_inf_f64(&input) { continue; }
            let s = generic::f64x2::<ScalarToken>::from_array(token_s, input).reduce_min();
            let n = generic::f64x2::<$native_token>::from_array(token_n, input).reduce_min();
            // Allow ±0 difference (hardware min may return different zero sign)
            if s == 0.0 && n == 0.0 { continue; }
            assert_eq!(s.to_bits(), n.to_bits(),
                "f64x2::reduce_min divergence: scalar={s} native={n} input={input:?}");
        }
    }
}

#[test]
fn reduce_max() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in super::F64_EDGE_A.chunks_exact(2) {
            let input: [f64; 2] = chunk.try_into().unwrap();
            if super::has_nan_or_inf_f64(&input) { continue; }
            let s = generic::f64x2::<ScalarToken>::from_array(token_s, input).reduce_max();
            let n = generic::f64x2::<$native_token>::from_array(token_n, input).reduce_max();
            if s == 0.0 && n == 0.0 { continue; }
            assert_eq!(s.to_bits(), n.to_bits(),
                "f64x2::reduce_max divergence: scalar={s} native={n} input={input:?}");
        }
    }
}

#[test]
fn simd_eq() {
    run_binary("f64x2::simd_eq", |as_, bs, an, bn| {
        (as_.simd_eq(bs).to_array(), an.simd_eq(bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn simd_lt() {
    run_binary("f64x2::simd_lt", |as_, bs, an, bn| {
        (as_.simd_lt(bs).to_array(), an.simd_lt(bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn simd_le() {
    run_binary("f64x2::simd_le", |as_, bs, an, bn| {
        (as_.simd_le(bs).to_array(), an.simd_le(bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn simd_gt() {
    run_binary("f64x2::simd_gt", |as_, bs, an, bn| {
        (as_.simd_gt(bs).to_array(), an.simd_gt(bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn simd_ge() {
    run_binary("f64x2::simd_ge", |as_, bs, an, bn| {
        (as_.simd_ge(bs).to_array(), an.simd_ge(bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn simd_ne() {
    run_binary_finite("f64x2::simd_ne", |as_, bs, an, bn| {
        (as_.simd_ne(bs).to_array(), an.simd_ne(bn).to_array())
    }, super::assert_f64_exact);
}

        }

mod f64x4_parity {
    use super::*;

    fn run_unary(
        op: &str,
        f: impl Fn(generic::f64x4<ScalarToken>, generic::f64x4<$native_token>) -> (
            [f64; 4], [f64; 4]
        ),
        cmp: impl Fn(&[f64], &[f64], &str, &[f64]),
    ) {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for chunk in super::F64_EDGE_A.chunks_exact(4) {
                let input: [f64; 4] = chunk.try_into().unwrap();
                let vs = generic::f64x4::<ScalarToken>::from_array(token_s, input);
                let vn = generic::f64x4::<$native_token>::from_array(token_n, input);
                let (s, n) = f(vs, vn);
                cmp(&s, &n, op, &input);
            }
        }
    }

    /// Like run_unary but skips chunks containing NaN or infinity.
    fn run_unary_finite(
        op: &str,
        f: impl Fn(generic::f64x4<ScalarToken>, generic::f64x4<$native_token>) -> (
            [f64; 4], [f64; 4]
        ),
        cmp: impl Fn(&[f64], &[f64], &str, &[f64]),
    ) {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for chunk in super::F64_EDGE_A.chunks_exact(4) {
                let input: [f64; 4] = chunk.try_into().unwrap();
                if super::has_nan_or_inf_f64(&input) { continue; }
                let vs = generic::f64x4::<ScalarToken>::from_array(token_s, input);
                let vn = generic::f64x4::<$native_token>::from_array(token_n, input);
                let (s, n) = f(vs, vn);
                cmp(&s, &n, op, &input);
            }
        }
    }

    fn run_binary(
        op: &str,
        f: impl Fn(
            generic::f64x4<ScalarToken>, generic::f64x4<ScalarToken>,
            generic::f64x4<$native_token>, generic::f64x4<$native_token>,
        ) -> ([f64; 4], [f64; 4]),
        cmp: impl Fn(&[f64], &[f64], &str, &[f64]),
    ) {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for (ca, cb) in super::F64_EDGE_A.chunks_exact(4).zip(super::F64_EDGE_B.chunks_exact(4)) {
                let a: [f64; 4] = ca.try_into().unwrap();
                let b: [f64; 4] = cb.try_into().unwrap();
                let as_ = generic::f64x4::<ScalarToken>::from_array(token_s, a);
                let bs = generic::f64x4::<ScalarToken>::from_array(token_s, b);
                let an = generic::f64x4::<$native_token>::from_array(token_n, a);
                let bn = generic::f64x4::<$native_token>::from_array(token_n, b);
                let (s, n) = f(as_, bs, an, bn);
                cmp(&s, &n, op, &a);
            }
        }
    }

    /// Like run_binary but skips chunks where either input contains NaN or infinity.
    fn run_binary_finite(
        op: &str,
        f: impl Fn(
            generic::f64x4<ScalarToken>, generic::f64x4<ScalarToken>,
            generic::f64x4<$native_token>, generic::f64x4<$native_token>,
        ) -> ([f64; 4], [f64; 4]),
        cmp: impl Fn(&[f64], &[f64], &str, &[f64]),
    ) {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for (ca, cb) in super::F64_EDGE_A.chunks_exact(4).zip(super::F64_EDGE_B.chunks_exact(4)) {
                let a: [f64; 4] = ca.try_into().unwrap();
                let b: [f64; 4] = cb.try_into().unwrap();
                if super::has_nan_or_inf_f64(&a) || super::has_nan_or_inf_f64(&b) { continue; }
                let as_ = generic::f64x4::<ScalarToken>::from_array(token_s, a);
                let bs = generic::f64x4::<ScalarToken>::from_array(token_s, b);
                let an = generic::f64x4::<$native_token>::from_array(token_n, a);
                let bn = generic::f64x4::<$native_token>::from_array(token_n, b);
                let (s, n) = f(as_, bs, an, bn);
                cmp(&s, &n, op, &a);
            }
        }
    }

#[test]
fn round() {
    run_unary("f64x4::round", |vs, vn| {
        (vs.round().to_array(), vn.round().to_array())
    }, super::assert_f64_exact);
}

#[test]
fn floor() {
    run_unary("f64x4::floor", |vs, vn| {
        (vs.floor().to_array(), vn.floor().to_array())
    }, super::assert_f64_exact);
}

#[test]
fn ceil() {
    run_unary("f64x4::ceil", |vs, vn| {
        (vs.ceil().to_array(), vn.ceil().to_array())
    }, super::assert_f64_exact);
}

#[test]
fn abs() {
    run_unary("f64x4::abs", |vs, vn| {
        (vs.abs().to_array(), vn.abs().to_array())
    }, super::assert_f64_exact);
}

#[test]
fn sqrt() {
    run_unary("f64x4::sqrt", |vs, vn| {
        (vs.sqrt().to_array(), vn.sqrt().to_array())
    }, super::assert_f64_exact);
}

#[test]
fn not() {
    run_unary("f64x4::not", |vs, vn| {
        (vs.not().to_array(), vn.not().to_array())
    }, super::assert_f64_exact);
}

#[test]
fn neg() {
    run_unary("f64x4::neg", |vs, vn| {
        ((-vs).to_array(), (-vn).to_array())
    }, super::assert_f64_signed_zero_tolerant);
}

#[test]
fn add() {
    run_binary("f64x4::add", |as_, bs, an, bn| {
        ((as_ + bs).to_array(), (an + bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn sub() {
    run_binary("f64x4::sub", |as_, bs, an, bn| {
        ((as_ - bs).to_array(), (an - bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn mul() {
    run_binary("f64x4::mul", |as_, bs, an, bn| {
        ((as_ * bs).to_array(), (an * bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn div() {
    run_binary("f64x4::div", |as_, bs, an, bn| {
        ((as_ / bs).to_array(), (an / bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn bitand() {
    run_binary("f64x4::bitand", |as_, bs, an, bn| {
        ((as_ & bs).to_array(), (an & bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn bitor() {
    run_binary("f64x4::bitor", |as_, bs, an, bn| {
        ((as_ | bs).to_array(), (an | bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn bitxor() {
    run_binary("f64x4::bitxor", |as_, bs, an, bn| {
        ((as_ ^ bs).to_array(), (an ^ bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn min() {
    run_binary_finite("f64x4::min", |as_, bs, an, bn| {
        (as_.min(bs).to_array(), an.min(bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn max() {
    run_binary_finite("f64x4::max", |as_, bs, an, bn| {
        (as_.max(bs).to_array(), an.max(bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn mul_add() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for ((ca, cb), cc) in super::F64_EDGE_A.chunks_exact(4)
            .zip(super::F64_EDGE_B.chunks_exact(4))
            .zip(super::F64_EDGE_C.chunks_exact(4))
        {
            let a: [f64; 4] = ca.try_into().unwrap();
            let b: [f64; 4] = cb.try_into().unwrap();
            let c: [f64; 4] = cc.try_into().unwrap();
            // Skip chunks with NaN/Inf/extreme values
            if [a.as_slice(), b.as_slice(), c.as_slice()].iter()
                .any(|arr| arr.iter().any(|x| x.is_nan() || x.is_infinite() || x.abs() > 1e30))
            { continue; }
            let as_ = generic::f64x4::<ScalarToken>::from_array(token_s, a);
            let bs = generic::f64x4::<ScalarToken>::from_array(token_s, b);
            let cs = generic::f64x4::<ScalarToken>::from_array(token_s, c);
            let an = generic::f64x4::<$native_token>::from_array(token_n, a);
            let bn = generic::f64x4::<$native_token>::from_array(token_n, b);
            let cn = generic::f64x4::<$native_token>::from_array(token_n, c);
            let s = as_.mul_add(bs, cs).to_array();
            let n = an.mul_add(bn, cn).to_array();
            // FMA vs mul+add can differ by many ULPs with catastrophic cancellation.
            // Just check that NaN/Inf agreement is maintained and finite values
            // are in the same ballpark (1e-4 relative tolerance).
            // FMA vs mul+add can differ significantly with catastrophic cancellation.
            // Use signed-zero-tolerant comparison (allows ±0 and NaN agreement).
            super::assert_f64_fma(&s, &n, "f64x4::mul_add", &a);
        }
    }
}

#[test]
fn mul_sub() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for ((ca, cb), cc) in super::F64_EDGE_A.chunks_exact(4)
            .zip(super::F64_EDGE_B.chunks_exact(4))
            .zip(super::F64_EDGE_C.chunks_exact(4))
        {
            let a: [f64; 4] = ca.try_into().unwrap();
            let b: [f64; 4] = cb.try_into().unwrap();
            let c: [f64; 4] = cc.try_into().unwrap();
            // Skip chunks with NaN/Inf/extreme values
            if [a.as_slice(), b.as_slice(), c.as_slice()].iter()
                .any(|arr| arr.iter().any(|x| x.is_nan() || x.is_infinite() || x.abs() > 1e30))
            { continue; }
            let as_ = generic::f64x4::<ScalarToken>::from_array(token_s, a);
            let bs = generic::f64x4::<ScalarToken>::from_array(token_s, b);
            let cs = generic::f64x4::<ScalarToken>::from_array(token_s, c);
            let an = generic::f64x4::<$native_token>::from_array(token_n, a);
            let bn = generic::f64x4::<$native_token>::from_array(token_n, b);
            let cn = generic::f64x4::<$native_token>::from_array(token_n, c);
            let s = as_.mul_sub(bs, cs).to_array();
            let n = an.mul_sub(bn, cn).to_array();
            // FMA vs mul+add can differ by many ULPs with catastrophic cancellation.
            // Just check that NaN/Inf agreement is maintained and finite values
            // are in the same ballpark (1e-4 relative tolerance).
            // FMA vs mul+add can differ significantly with catastrophic cancellation.
            // Use signed-zero-tolerant comparison (allows ±0 and NaN agreement).
            super::assert_f64_fma(&s, &n, "f64x4::mul_sub", &a);
        }
    }
}

#[test]
fn reduce_add() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in super::F64_EDGE_A.chunks_exact(4) {
            let input: [f64; 4] = chunk.try_into().unwrap();
            // Skip chunks with NaN/Inf or extreme magnitudes (catastrophic cancellation
            // from different FP associativity between tree and left-fold reduction)
            if input.iter().any(|x| x.is_nan() || x.is_infinite() || x.abs() > 1e9) { continue; }
            let s = generic::f64x4::<ScalarToken>::from_array(token_s, input).reduce_add();
            let n = generic::f64x4::<$native_token>::from_array(token_n, input).reduce_add();
            if s.is_nan() && n.is_nan() { continue; }
            if s.to_bits() != n.to_bits() {
                // Allow relative tolerance for FP associativity
                let denom = s.abs().max(n.abs()).max(f64::MIN_POSITIVE);
                let rel_err = (s - n).abs() / denom;
                assert!(rel_err < 1e-6,
                    "f64x4::reduce_add divergence: scalar={s} native={n} (rel_err={rel_err}) input={input:?}");
            }
        }
    }
}

#[test]
fn reduce_min() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in super::F64_EDGE_A.chunks_exact(4) {
            let input: [f64; 4] = chunk.try_into().unwrap();
            if super::has_nan_or_inf_f64(&input) { continue; }
            let s = generic::f64x4::<ScalarToken>::from_array(token_s, input).reduce_min();
            let n = generic::f64x4::<$native_token>::from_array(token_n, input).reduce_min();
            // Allow ±0 difference (hardware min may return different zero sign)
            if s == 0.0 && n == 0.0 { continue; }
            assert_eq!(s.to_bits(), n.to_bits(),
                "f64x4::reduce_min divergence: scalar={s} native={n} input={input:?}");
        }
    }
}

#[test]
fn reduce_max() {
    let token_s = ScalarToken;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in super::F64_EDGE_A.chunks_exact(4) {
            let input: [f64; 4] = chunk.try_into().unwrap();
            if super::has_nan_or_inf_f64(&input) { continue; }
            let s = generic::f64x4::<ScalarToken>::from_array(token_s, input).reduce_max();
            let n = generic::f64x4::<$native_token>::from_array(token_n, input).reduce_max();
            if s == 0.0 && n == 0.0 { continue; }
            assert_eq!(s.to_bits(), n.to_bits(),
                "f64x4::reduce_max divergence: scalar={s} native={n} input={input:?}");
        }
    }
}

#[test]
fn simd_eq() {
    run_binary("f64x4::simd_eq", |as_, bs, an, bn| {
        (as_.simd_eq(bs).to_array(), an.simd_eq(bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn simd_lt() {
    run_binary("f64x4::simd_lt", |as_, bs, an, bn| {
        (as_.simd_lt(bs).to_array(), an.simd_lt(bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn simd_le() {
    run_binary("f64x4::simd_le", |as_, bs, an, bn| {
        (as_.simd_le(bs).to_array(), an.simd_le(bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn simd_gt() {
    run_binary("f64x4::simd_gt", |as_, bs, an, bn| {
        (as_.simd_gt(bs).to_array(), an.simd_gt(bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn simd_ge() {
    run_binary("f64x4::simd_ge", |as_, bs, an, bn| {
        (as_.simd_ge(bs).to_array(), an.simd_ge(bn).to_array())
    }, super::assert_f64_exact);
}

#[test]
fn simd_ne() {
    run_binary_finite("f64x4::simd_ne", |as_, bs, an, bn| {
        (as_.simd_ne(bs).to_array(), an.simd_ne(bn).to_array())
    }, super::assert_f64_exact);
}

        }

mod i32x4_parity {
    use super::*;

#[test]
fn add() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A;
    let edge_b = super::I32_EDGE_B;
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(4).zip(edge_b.chunks_exact(4)) {
            let a: [i32; 4] = ca.try_into().unwrap();
            let b: [i32; 4] = cb.try_into().unwrap();
            let as_ = generic::i32x4::<ScalarToken>::from_array(token_s, a);
            let bs = generic::i32x4::<ScalarToken>::from_array(token_s, b);
            let an = generic::i32x4::<$native_token>::from_array(token_n, a);
            let bn = generic::i32x4::<$native_token>::from_array(token_n, b);
            let s = (as_ + bs).to_array();
            let n = (an + bn).to_array();
            super::assert_i32_exact(&s, &n, "i32x4::add", &a);
        }
    }
}

#[test]
fn sub() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A;
    let edge_b = super::I32_EDGE_B;
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(4).zip(edge_b.chunks_exact(4)) {
            let a: [i32; 4] = ca.try_into().unwrap();
            let b: [i32; 4] = cb.try_into().unwrap();
            let as_ = generic::i32x4::<ScalarToken>::from_array(token_s, a);
            let bs = generic::i32x4::<ScalarToken>::from_array(token_s, b);
            let an = generic::i32x4::<$native_token>::from_array(token_n, a);
            let bn = generic::i32x4::<$native_token>::from_array(token_n, b);
            let s = (as_ - bs).to_array();
            let n = (an - bn).to_array();
            super::assert_i32_exact(&s, &n, "i32x4::sub", &a);
        }
    }
}

#[test]
fn min() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A;
    let edge_b = super::I32_EDGE_B;
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(4).zip(edge_b.chunks_exact(4)) {
            let a: [i32; 4] = ca.try_into().unwrap();
            let b: [i32; 4] = cb.try_into().unwrap();
            let s = generic::i32x4::<ScalarToken>::from_array(token_s, a).min(
                generic::i32x4::<ScalarToken>::from_array(token_s, b)
            ).to_array();
            let n = generic::i32x4::<$native_token>::from_array(token_n, a).min(
                generic::i32x4::<$native_token>::from_array(token_n, b)
            ).to_array();
            super::assert_i32_exact(&s, &n, "i32x4::min", &a);
        }
    }
}

#[test]
fn max() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A;
    let edge_b = super::I32_EDGE_B;
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(4).zip(edge_b.chunks_exact(4)) {
            let a: [i32; 4] = ca.try_into().unwrap();
            let b: [i32; 4] = cb.try_into().unwrap();
            let s = generic::i32x4::<ScalarToken>::from_array(token_s, a).max(
                generic::i32x4::<ScalarToken>::from_array(token_s, b)
            ).to_array();
            let n = generic::i32x4::<$native_token>::from_array(token_n, a).max(
                generic::i32x4::<$native_token>::from_array(token_n, b)
            ).to_array();
            super::assert_i32_exact(&s, &n, "i32x4::max", &a);
        }
    }
}

#[test]
fn abs() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in edge_a.chunks_exact(4) {
            let input: [i32; 4] = chunk.try_into().unwrap();
            let s = generic::i32x4::<ScalarToken>::from_array(token_s, input).abs().to_array();
            let n = generic::i32x4::<$native_token>::from_array(token_n, input).abs().to_array();
            super::assert_i32_exact(&s, &n, "i32x4::abs", &input);
        }
    }
}

#[test]
fn neg() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in edge_a.chunks_exact(4) {
            let input: [i32; 4] = chunk.try_into().unwrap();
            let s = (-generic::i32x4::<ScalarToken>::from_array(token_s, input)).to_array();
            let n = (-generic::i32x4::<$native_token>::from_array(token_n, input)).to_array();
            super::assert_i32_exact(&s, &n, "i32x4::neg", &input);
        }
    }
}

#[test]
fn not() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in edge_a.chunks_exact(4) {
            let input: [i32; 4] = chunk.try_into().unwrap();
            let s = generic::i32x4::<ScalarToken>::from_array(token_s, input).not().to_array();
            let n = generic::i32x4::<$native_token>::from_array(token_n, input).not().to_array();
            super::assert_i32_exact(&s, &n, "i32x4::not", &input);
        }
    }
}

        }

mod i32x8_parity {
    use super::*;

#[test]
fn add() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A;
    let edge_b = super::I32_EDGE_B;
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(8).zip(edge_b.chunks_exact(8)) {
            let a: [i32; 8] = ca.try_into().unwrap();
            let b: [i32; 8] = cb.try_into().unwrap();
            let as_ = generic::i32x8::<ScalarToken>::from_array(token_s, a);
            let bs = generic::i32x8::<ScalarToken>::from_array(token_s, b);
            let an = generic::i32x8::<$native_token>::from_array(token_n, a);
            let bn = generic::i32x8::<$native_token>::from_array(token_n, b);
            let s = (as_ + bs).to_array();
            let n = (an + bn).to_array();
            super::assert_i32_exact(&s, &n, "i32x8::add", &a);
        }
    }
}

#[test]
fn sub() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A;
    let edge_b = super::I32_EDGE_B;
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(8).zip(edge_b.chunks_exact(8)) {
            let a: [i32; 8] = ca.try_into().unwrap();
            let b: [i32; 8] = cb.try_into().unwrap();
            let as_ = generic::i32x8::<ScalarToken>::from_array(token_s, a);
            let bs = generic::i32x8::<ScalarToken>::from_array(token_s, b);
            let an = generic::i32x8::<$native_token>::from_array(token_n, a);
            let bn = generic::i32x8::<$native_token>::from_array(token_n, b);
            let s = (as_ - bs).to_array();
            let n = (an - bn).to_array();
            super::assert_i32_exact(&s, &n, "i32x8::sub", &a);
        }
    }
}

#[test]
fn min() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A;
    let edge_b = super::I32_EDGE_B;
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(8).zip(edge_b.chunks_exact(8)) {
            let a: [i32; 8] = ca.try_into().unwrap();
            let b: [i32; 8] = cb.try_into().unwrap();
            let s = generic::i32x8::<ScalarToken>::from_array(token_s, a).min(
                generic::i32x8::<ScalarToken>::from_array(token_s, b)
            ).to_array();
            let n = generic::i32x8::<$native_token>::from_array(token_n, a).min(
                generic::i32x8::<$native_token>::from_array(token_n, b)
            ).to_array();
            super::assert_i32_exact(&s, &n, "i32x8::min", &a);
        }
    }
}

#[test]
fn max() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A;
    let edge_b = super::I32_EDGE_B;
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(8).zip(edge_b.chunks_exact(8)) {
            let a: [i32; 8] = ca.try_into().unwrap();
            let b: [i32; 8] = cb.try_into().unwrap();
            let s = generic::i32x8::<ScalarToken>::from_array(token_s, a).max(
                generic::i32x8::<ScalarToken>::from_array(token_s, b)
            ).to_array();
            let n = generic::i32x8::<$native_token>::from_array(token_n, a).max(
                generic::i32x8::<$native_token>::from_array(token_n, b)
            ).to_array();
            super::assert_i32_exact(&s, &n, "i32x8::max", &a);
        }
    }
}

#[test]
fn abs() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in edge_a.chunks_exact(8) {
            let input: [i32; 8] = chunk.try_into().unwrap();
            let s = generic::i32x8::<ScalarToken>::from_array(token_s, input).abs().to_array();
            let n = generic::i32x8::<$native_token>::from_array(token_n, input).abs().to_array();
            super::assert_i32_exact(&s, &n, "i32x8::abs", &input);
        }
    }
}

#[test]
fn neg() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in edge_a.chunks_exact(8) {
            let input: [i32; 8] = chunk.try_into().unwrap();
            let s = (-generic::i32x8::<ScalarToken>::from_array(token_s, input)).to_array();
            let n = (-generic::i32x8::<$native_token>::from_array(token_n, input)).to_array();
            super::assert_i32_exact(&s, &n, "i32x8::neg", &input);
        }
    }
}

#[test]
fn not() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in edge_a.chunks_exact(8) {
            let input: [i32; 8] = chunk.try_into().unwrap();
            let s = generic::i32x8::<ScalarToken>::from_array(token_s, input).not().to_array();
            let n = generic::i32x8::<$native_token>::from_array(token_n, input).not().to_array();
            super::assert_i32_exact(&s, &n, "i32x8::not", &input);
        }
    }
}

        }

mod i16x8_parity {
    use super::*;

#[test]
fn add() {
    let token_s = ScalarToken;
    let edge_a = super::I16_EDGE_A;
    let edge_b = super::I16_EDGE_B;
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(8).zip(edge_b.chunks_exact(8)) {
            let a: [i16; 8] = ca.try_into().unwrap();
            let b: [i16; 8] = cb.try_into().unwrap();
            let as_ = generic::i16x8::<ScalarToken>::from_array(token_s, a);
            let bs = generic::i16x8::<ScalarToken>::from_array(token_s, b);
            let an = generic::i16x8::<$native_token>::from_array(token_n, a);
            let bn = generic::i16x8::<$native_token>::from_array(token_n, b);
            let s = (as_ + bs).to_array();
            let n = (an + bn).to_array();
            super::assert_i16_exact(&s, &n, "i16x8::add", &a);
        }
    }
}

#[test]
fn sub() {
    let token_s = ScalarToken;
    let edge_a = super::I16_EDGE_A;
    let edge_b = super::I16_EDGE_B;
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(8).zip(edge_b.chunks_exact(8)) {
            let a: [i16; 8] = ca.try_into().unwrap();
            let b: [i16; 8] = cb.try_into().unwrap();
            let as_ = generic::i16x8::<ScalarToken>::from_array(token_s, a);
            let bs = generic::i16x8::<ScalarToken>::from_array(token_s, b);
            let an = generic::i16x8::<$native_token>::from_array(token_n, a);
            let bn = generic::i16x8::<$native_token>::from_array(token_n, b);
            let s = (as_ - bs).to_array();
            let n = (an - bn).to_array();
            super::assert_i16_exact(&s, &n, "i16x8::sub", &a);
        }
    }
}

#[test]
fn min() {
    let token_s = ScalarToken;
    let edge_a = super::I16_EDGE_A;
    let edge_b = super::I16_EDGE_B;
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(8).zip(edge_b.chunks_exact(8)) {
            let a: [i16; 8] = ca.try_into().unwrap();
            let b: [i16; 8] = cb.try_into().unwrap();
            let s = generic::i16x8::<ScalarToken>::from_array(token_s, a).min(
                generic::i16x8::<ScalarToken>::from_array(token_s, b)
            ).to_array();
            let n = generic::i16x8::<$native_token>::from_array(token_n, a).min(
                generic::i16x8::<$native_token>::from_array(token_n, b)
            ).to_array();
            super::assert_i16_exact(&s, &n, "i16x8::min", &a);
        }
    }
}

#[test]
fn max() {
    let token_s = ScalarToken;
    let edge_a = super::I16_EDGE_A;
    let edge_b = super::I16_EDGE_B;
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(8).zip(edge_b.chunks_exact(8)) {
            let a: [i16; 8] = ca.try_into().unwrap();
            let b: [i16; 8] = cb.try_into().unwrap();
            let s = generic::i16x8::<ScalarToken>::from_array(token_s, a).max(
                generic::i16x8::<ScalarToken>::from_array(token_s, b)
            ).to_array();
            let n = generic::i16x8::<$native_token>::from_array(token_n, a).max(
                generic::i16x8::<$native_token>::from_array(token_n, b)
            ).to_array();
            super::assert_i16_exact(&s, &n, "i16x8::max", &a);
        }
    }
}

#[test]
fn abs() {
    let token_s = ScalarToken;
    let edge_a = super::I16_EDGE_A;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in edge_a.chunks_exact(8) {
            let input: [i16; 8] = chunk.try_into().unwrap();
            let s = generic::i16x8::<ScalarToken>::from_array(token_s, input).abs().to_array();
            let n = generic::i16x8::<$native_token>::from_array(token_n, input).abs().to_array();
            super::assert_i16_exact(&s, &n, "i16x8::abs", &input);
        }
    }
}

#[test]
fn neg() {
    let token_s = ScalarToken;
    let edge_a = super::I16_EDGE_A;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in edge_a.chunks_exact(8) {
            let input: [i16; 8] = chunk.try_into().unwrap();
            let s = (-generic::i16x8::<ScalarToken>::from_array(token_s, input)).to_array();
            let n = (-generic::i16x8::<$native_token>::from_array(token_n, input)).to_array();
            super::assert_i16_exact(&s, &n, "i16x8::neg", &input);
        }
    }
}

#[test]
fn not() {
    let token_s = ScalarToken;
    let edge_a = super::I16_EDGE_A;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in edge_a.chunks_exact(8) {
            let input: [i16; 8] = chunk.try_into().unwrap();
            let s = generic::i16x8::<ScalarToken>::from_array(token_s, input).not().to_array();
            let n = generic::i16x8::<$native_token>::from_array(token_n, input).not().to_array();
            super::assert_i16_exact(&s, &n, "i16x8::not", &input);
        }
    }
}

        }

mod i16x16_parity {
    use super::*;

#[test]
fn add() {
    let token_s = ScalarToken;
    let edge_a = super::I16_EDGE_A;
    let edge_b = super::I16_EDGE_B;
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(16).zip(edge_b.chunks_exact(16)) {
            let a: [i16; 16] = ca.try_into().unwrap();
            let b: [i16; 16] = cb.try_into().unwrap();
            let as_ = generic::i16x16::<ScalarToken>::from_array(token_s, a);
            let bs = generic::i16x16::<ScalarToken>::from_array(token_s, b);
            let an = generic::i16x16::<$native_token>::from_array(token_n, a);
            let bn = generic::i16x16::<$native_token>::from_array(token_n, b);
            let s = (as_ + bs).to_array();
            let n = (an + bn).to_array();
            super::assert_i16_exact(&s, &n, "i16x16::add", &a);
        }
    }
}

#[test]
fn sub() {
    let token_s = ScalarToken;
    let edge_a = super::I16_EDGE_A;
    let edge_b = super::I16_EDGE_B;
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(16).zip(edge_b.chunks_exact(16)) {
            let a: [i16; 16] = ca.try_into().unwrap();
            let b: [i16; 16] = cb.try_into().unwrap();
            let as_ = generic::i16x16::<ScalarToken>::from_array(token_s, a);
            let bs = generic::i16x16::<ScalarToken>::from_array(token_s, b);
            let an = generic::i16x16::<$native_token>::from_array(token_n, a);
            let bn = generic::i16x16::<$native_token>::from_array(token_n, b);
            let s = (as_ - bs).to_array();
            let n = (an - bn).to_array();
            super::assert_i16_exact(&s, &n, "i16x16::sub", &a);
        }
    }
}

#[test]
fn min() {
    let token_s = ScalarToken;
    let edge_a = super::I16_EDGE_A;
    let edge_b = super::I16_EDGE_B;
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(16).zip(edge_b.chunks_exact(16)) {
            let a: [i16; 16] = ca.try_into().unwrap();
            let b: [i16; 16] = cb.try_into().unwrap();
            let s = generic::i16x16::<ScalarToken>::from_array(token_s, a).min(
                generic::i16x16::<ScalarToken>::from_array(token_s, b)
            ).to_array();
            let n = generic::i16x16::<$native_token>::from_array(token_n, a).min(
                generic::i16x16::<$native_token>::from_array(token_n, b)
            ).to_array();
            super::assert_i16_exact(&s, &n, "i16x16::min", &a);
        }
    }
}

#[test]
fn max() {
    let token_s = ScalarToken;
    let edge_a = super::I16_EDGE_A;
    let edge_b = super::I16_EDGE_B;
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(16).zip(edge_b.chunks_exact(16)) {
            let a: [i16; 16] = ca.try_into().unwrap();
            let b: [i16; 16] = cb.try_into().unwrap();
            let s = generic::i16x16::<ScalarToken>::from_array(token_s, a).max(
                generic::i16x16::<ScalarToken>::from_array(token_s, b)
            ).to_array();
            let n = generic::i16x16::<$native_token>::from_array(token_n, a).max(
                generic::i16x16::<$native_token>::from_array(token_n, b)
            ).to_array();
            super::assert_i16_exact(&s, &n, "i16x16::max", &a);
        }
    }
}

#[test]
fn abs() {
    let token_s = ScalarToken;
    let edge_a = super::I16_EDGE_A;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in edge_a.chunks_exact(16) {
            let input: [i16; 16] = chunk.try_into().unwrap();
            let s = generic::i16x16::<ScalarToken>::from_array(token_s, input).abs().to_array();
            let n = generic::i16x16::<$native_token>::from_array(token_n, input).abs().to_array();
            super::assert_i16_exact(&s, &n, "i16x16::abs", &input);
        }
    }
}

#[test]
fn neg() {
    let token_s = ScalarToken;
    let edge_a = super::I16_EDGE_A;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in edge_a.chunks_exact(16) {
            let input: [i16; 16] = chunk.try_into().unwrap();
            let s = (-generic::i16x16::<ScalarToken>::from_array(token_s, input)).to_array();
            let n = (-generic::i16x16::<$native_token>::from_array(token_n, input)).to_array();
            super::assert_i16_exact(&s, &n, "i16x16::neg", &input);
        }
    }
}

#[test]
fn not() {
    let token_s = ScalarToken;
    let edge_a = super::I16_EDGE_A;
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in edge_a.chunks_exact(16) {
            let input: [i16; 16] = chunk.try_into().unwrap();
            let s = generic::i16x16::<ScalarToken>::from_array(token_s, input).not().to_array();
            let n = generic::i16x16::<$native_token>::from_array(token_n, input).not().to_array();
            super::assert_i16_exact(&s, &n, "i16x16::not", &input);
        }
    }
}

        }

mod u32x4_parity {
    use super::*;

#[test]
fn add() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A.map(|x| x as u32);
    let edge_b = super::I32_EDGE_B.map(|x| x as u32);
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(4).zip(edge_b.chunks_exact(4)) {
            let a: [u32; 4] = ca.try_into().unwrap();
            let b: [u32; 4] = cb.try_into().unwrap();
            let as_ = generic::u32x4::<ScalarToken>::from_array(token_s, a);
            let bs = generic::u32x4::<ScalarToken>::from_array(token_s, b);
            let an = generic::u32x4::<$native_token>::from_array(token_n, a);
            let bn = generic::u32x4::<$native_token>::from_array(token_n, b);
            let s = (as_ + bs).to_array();
            let n = (an + bn).to_array();
            super::assert_u32_exact(&s, &n, "u32x4::add", &a);
        }
    }
}

#[test]
fn sub() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A.map(|x| x as u32);
    let edge_b = super::I32_EDGE_B.map(|x| x as u32);
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(4).zip(edge_b.chunks_exact(4)) {
            let a: [u32; 4] = ca.try_into().unwrap();
            let b: [u32; 4] = cb.try_into().unwrap();
            let as_ = generic::u32x4::<ScalarToken>::from_array(token_s, a);
            let bs = generic::u32x4::<ScalarToken>::from_array(token_s, b);
            let an = generic::u32x4::<$native_token>::from_array(token_n, a);
            let bn = generic::u32x4::<$native_token>::from_array(token_n, b);
            let s = (as_ - bs).to_array();
            let n = (an - bn).to_array();
            super::assert_u32_exact(&s, &n, "u32x4::sub", &a);
        }
    }
}

#[test]
fn min() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A.map(|x| x as u32);
    let edge_b = super::I32_EDGE_B.map(|x| x as u32);
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(4).zip(edge_b.chunks_exact(4)) {
            let a: [u32; 4] = ca.try_into().unwrap();
            let b: [u32; 4] = cb.try_into().unwrap();
            let s = generic::u32x4::<ScalarToken>::from_array(token_s, a).min(
                generic::u32x4::<ScalarToken>::from_array(token_s, b)
            ).to_array();
            let n = generic::u32x4::<$native_token>::from_array(token_n, a).min(
                generic::u32x4::<$native_token>::from_array(token_n, b)
            ).to_array();
            super::assert_u32_exact(&s, &n, "u32x4::min", &a);
        }
    }
}

#[test]
fn max() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A.map(|x| x as u32);
    let edge_b = super::I32_EDGE_B.map(|x| x as u32);
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(4).zip(edge_b.chunks_exact(4)) {
            let a: [u32; 4] = ca.try_into().unwrap();
            let b: [u32; 4] = cb.try_into().unwrap();
            let s = generic::u32x4::<ScalarToken>::from_array(token_s, a).max(
                generic::u32x4::<ScalarToken>::from_array(token_s, b)
            ).to_array();
            let n = generic::u32x4::<$native_token>::from_array(token_n, a).max(
                generic::u32x4::<$native_token>::from_array(token_n, b)
            ).to_array();
            super::assert_u32_exact(&s, &n, "u32x4::max", &a);
        }
    }
}

#[test]
fn not() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A.map(|x| x as u32);
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in edge_a.chunks_exact(4) {
            let input: [u32; 4] = chunk.try_into().unwrap();
            let s = generic::u32x4::<ScalarToken>::from_array(token_s, input).not().to_array();
            let n = generic::u32x4::<$native_token>::from_array(token_n, input).not().to_array();
            super::assert_u32_exact(&s, &n, "u32x4::not", &input);
        }
    }
}

        }

mod u32x8_parity {
    use super::*;

#[test]
fn add() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A.map(|x| x as u32);
    let edge_b = super::I32_EDGE_B.map(|x| x as u32);
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(8).zip(edge_b.chunks_exact(8)) {
            let a: [u32; 8] = ca.try_into().unwrap();
            let b: [u32; 8] = cb.try_into().unwrap();
            let as_ = generic::u32x8::<ScalarToken>::from_array(token_s, a);
            let bs = generic::u32x8::<ScalarToken>::from_array(token_s, b);
            let an = generic::u32x8::<$native_token>::from_array(token_n, a);
            let bn = generic::u32x8::<$native_token>::from_array(token_n, b);
            let s = (as_ + bs).to_array();
            let n = (an + bn).to_array();
            super::assert_u32_exact(&s, &n, "u32x8::add", &a);
        }
    }
}

#[test]
fn sub() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A.map(|x| x as u32);
    let edge_b = super::I32_EDGE_B.map(|x| x as u32);
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(8).zip(edge_b.chunks_exact(8)) {
            let a: [u32; 8] = ca.try_into().unwrap();
            let b: [u32; 8] = cb.try_into().unwrap();
            let as_ = generic::u32x8::<ScalarToken>::from_array(token_s, a);
            let bs = generic::u32x8::<ScalarToken>::from_array(token_s, b);
            let an = generic::u32x8::<$native_token>::from_array(token_n, a);
            let bn = generic::u32x8::<$native_token>::from_array(token_n, b);
            let s = (as_ - bs).to_array();
            let n = (an - bn).to_array();
            super::assert_u32_exact(&s, &n, "u32x8::sub", &a);
        }
    }
}

#[test]
fn min() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A.map(|x| x as u32);
    let edge_b = super::I32_EDGE_B.map(|x| x as u32);
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(8).zip(edge_b.chunks_exact(8)) {
            let a: [u32; 8] = ca.try_into().unwrap();
            let b: [u32; 8] = cb.try_into().unwrap();
            let s = generic::u32x8::<ScalarToken>::from_array(token_s, a).min(
                generic::u32x8::<ScalarToken>::from_array(token_s, b)
            ).to_array();
            let n = generic::u32x8::<$native_token>::from_array(token_n, a).min(
                generic::u32x8::<$native_token>::from_array(token_n, b)
            ).to_array();
            super::assert_u32_exact(&s, &n, "u32x8::min", &a);
        }
    }
}

#[test]
fn max() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A.map(|x| x as u32);
    let edge_b = super::I32_EDGE_B.map(|x| x as u32);
    if let Some(token_n) = <$native_token>::summon() {
        for (ca, cb) in edge_a.chunks_exact(8).zip(edge_b.chunks_exact(8)) {
            let a: [u32; 8] = ca.try_into().unwrap();
            let b: [u32; 8] = cb.try_into().unwrap();
            let s = generic::u32x8::<ScalarToken>::from_array(token_s, a).max(
                generic::u32x8::<ScalarToken>::from_array(token_s, b)
            ).to_array();
            let n = generic::u32x8::<$native_token>::from_array(token_n, a).max(
                generic::u32x8::<$native_token>::from_array(token_n, b)
            ).to_array();
            super::assert_u32_exact(&s, &n, "u32x8::max", &a);
        }
    }
}

#[test]
fn not() {
    let token_s = ScalarToken;
    let edge_a = super::I32_EDGE_A.map(|x| x as u32);
    if let Some(token_n) = <$native_token>::summon() {
        for chunk in edge_a.chunks_exact(8) {
            let input: [u32; 8] = chunk.try_into().unwrap();
            let s = generic::u32x8::<ScalarToken>::from_array(token_s, input).not().to_array();
            let n = generic::u32x8::<$native_token>::from_array(token_n, input).not().to_array();
            super::assert_u32_exact(&s, &n, "u32x8::not", &input);
        }
    }
}

        }

mod convert_f32x4_parity {
    use super::*;

    /// The exact bug from issue #20: to_i32_round must use ties-to-even.
    #[test]
    fn to_i32_round() {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            // Focus on values that are safe to convert to i32
            let safe_f32: Vec<f32> = super::F32_EDGE_A.iter()
                .filter(|&&x| x.is_finite() && x.abs() < 2147483520.0)
                .copied()
                .collect();
            for chunk in safe_f32.chunks_exact(4) {
                let input: [f32; 4] = chunk.try_into().unwrap();
                let s = generic::f32x4::<ScalarToken>::from_array(token_s, input)
                    .to_i32_round().to_array();
                let n = generic::f32x4::<$native_token>::from_array(token_n, input)
                    .to_i32_round().to_array();
                super::assert_i32_exact(&s, &n, "f32x4::to_i32_round", &input.map(|x| x as i32));
            }
        }
    }

    #[test]
    fn to_i32() {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            let safe_f32: Vec<f32> = super::F32_EDGE_A.iter()
                .filter(|&&x| x.is_finite() && x.abs() < 2147483520.0)
                .copied()
                .collect();
            for chunk in safe_f32.chunks_exact(4) {
                let input: [f32; 4] = chunk.try_into().unwrap();
                let s = generic::f32x4::<ScalarToken>::from_array(token_s, input)
                    .to_i32().to_array();
                let n = generic::f32x4::<$native_token>::from_array(token_n, input)
                    .to_i32().to_array();
                super::assert_i32_exact(&s, &n, "f32x4::to_i32", &input.map(|x| x as i32));
            }
        }
    }

    #[test]
    fn from_i32_roundtrip() {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for chunk in super::I32_EDGE_A.chunks_exact(4) {
                let input: [i32; 4] = chunk.try_into().unwrap();
                let s = generic::f32x4::<ScalarToken>::from_i32(
                    token_s,
                    generic::i32x4::<ScalarToken>::from_array(token_s, input)
                ).to_array();
                let n = generic::f32x4::<$native_token>::from_i32(
                    token_n,
                    generic::i32x4::<$native_token>::from_array(token_n, input)
                ).to_array();
                super::assert_f32_exact(&s, &n, "f32x4::from_i32", &s);
            }
        }
    }

    #[test]
    fn bitcast_roundtrip() {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for chunk in super::F32_EDGE_A.chunks_exact(4) {
                let input: [f32; 4] = chunk.try_into().unwrap();
                // f32 → i32 bitcast → f32 bitcast should be identity
                let s = generic::f32x4::<ScalarToken>::from_i32_bitcast(
                    token_s,
                    generic::f32x4::<ScalarToken>::from_array(token_s, input).bitcast_to_i32()
                ).to_array();
                let n = generic::f32x4::<$native_token>::from_i32_bitcast(
                    token_n,
                    generic::f32x4::<$native_token>::from_array(token_n, input).bitcast_to_i32()
                ).to_array();
                // Compare bit patterns (NaN payload must survive roundtrip)
                for i in 0..4 {
                    assert_eq!(s[i].to_bits(), n[i].to_bits(),
                        "f32x4::bitcast_roundtrip divergence at lane {i}: scalar={} native={}",
                        s[i], n[i]);
                }
            }
        }
    }
}

mod convert_f32x8_parity {
    use super::*;

    /// The exact bug from issue #20: to_i32_round must use ties-to-even.
    #[test]
    fn to_i32_round() {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            // Focus on values that are safe to convert to i32
            let safe_f32: Vec<f32> = super::F32_EDGE_A.iter()
                .filter(|&&x| x.is_finite() && x.abs() < 2147483520.0)
                .copied()
                .collect();
            for chunk in safe_f32.chunks_exact(8) {
                let input: [f32; 8] = chunk.try_into().unwrap();
                let s = generic::f32x8::<ScalarToken>::from_array(token_s, input)
                    .to_i32_round().to_array();
                let n = generic::f32x8::<$native_token>::from_array(token_n, input)
                    .to_i32_round().to_array();
                super::assert_i32_exact(&s, &n, "f32x8::to_i32_round", &input.map(|x| x as i32));
            }
        }
    }

    #[test]
    fn to_i32() {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            let safe_f32: Vec<f32> = super::F32_EDGE_A.iter()
                .filter(|&&x| x.is_finite() && x.abs() < 2147483520.0)
                .copied()
                .collect();
            for chunk in safe_f32.chunks_exact(8) {
                let input: [f32; 8] = chunk.try_into().unwrap();
                let s = generic::f32x8::<ScalarToken>::from_array(token_s, input)
                    .to_i32().to_array();
                let n = generic::f32x8::<$native_token>::from_array(token_n, input)
                    .to_i32().to_array();
                super::assert_i32_exact(&s, &n, "f32x8::to_i32", &input.map(|x| x as i32));
            }
        }
    }

    #[test]
    fn from_i32_roundtrip() {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for chunk in super::I32_EDGE_A.chunks_exact(8) {
                let input: [i32; 8] = chunk.try_into().unwrap();
                let s = generic::f32x8::<ScalarToken>::from_i32(
                    token_s,
                    generic::i32x8::<ScalarToken>::from_array(token_s, input)
                ).to_array();
                let n = generic::f32x8::<$native_token>::from_i32(
                    token_n,
                    generic::i32x8::<$native_token>::from_array(token_n, input)
                ).to_array();
                super::assert_f32_exact(&s, &n, "f32x8::from_i32", &s);
            }
        }
    }

    #[test]
    fn bitcast_roundtrip() {
        let token_s = ScalarToken;
        if let Some(token_n) = <$native_token>::summon() {
            for chunk in super::F32_EDGE_A.chunks_exact(8) {
                let input: [f32; 8] = chunk.try_into().unwrap();
                // f32 → i32 bitcast → f32 bitcast should be identity
                let s = generic::f32x8::<ScalarToken>::from_i32_bitcast(
                    token_s,
                    generic::f32x8::<ScalarToken>::from_array(token_s, input).bitcast_to_i32()
                ).to_array();
                let n = generic::f32x8::<$native_token>::from_i32_bitcast(
                    token_n,
                    generic::f32x8::<$native_token>::from_array(token_n, input).bitcast_to_i32()
                ).to_array();
                // Compare bit patterns (NaN payload must survive roundtrip)
                for i in 0..8 {
                    assert_eq!(s[i].to_bits(), n[i].to_bits(),
                        "f32x8::bitcast_roundtrip divergence at lane {i}: scalar={} native={}",
                        s[i], n[i]);
                }
            }
        }
    }
}

    };
}

// ============================================================================
// Architecture-specific invocations
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod x86 {
    scalar_vs_native!(archmage::X64V3Token);
}

#[cfg(target_arch = "aarch64")]
mod arm {
    scalar_vs_native!(archmage::NeonToken);
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    scalar_vs_native!(archmage::Wasm128Token);
}

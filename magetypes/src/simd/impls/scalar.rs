//! Backend implementations for ScalarToken (fallback).
//!
//! All operations are plain array math. Always available on all platforms.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

use crate::simd::backends::*;

// Helpers to avoid trait method name shadowing inside the impl block.
// Inside `impl XxxBackend`, names like `sqrt`, `floor`, etc. resolve to
// the trait's associated functions instead of f32's inherent methods.
#[inline(always)]
fn f32_sqrt(x: f32) -> f32 {
    x.sqrt()
}
#[inline(always)]
fn f32_floor(x: f32) -> f32 {
    x.floor()
}
#[inline(always)]
fn f32_ceil(x: f32) -> f32 {
    x.ceil()
}
#[inline(always)]
fn f32_round(x: f32) -> f32 {
    x.round()
}

// Helpers to avoid trait method name shadowing inside the impl block.
// Inside `impl XxxBackend`, names like `sqrt`, `floor`, etc. resolve to
// the trait's associated functions instead of f64's inherent methods.
#[inline(always)]
fn f64_sqrt(x: f64) -> f64 {
    x.sqrt()
}
#[inline(always)]
fn f64_floor(x: f64) -> f64 {
    x.floor()
}
#[inline(always)]
fn f64_ceil(x: f64) -> f64 {
    x.ceil()
}
#[inline(always)]
fn f64_round(x: f64) -> f64 {
    x.round()
}

impl F32x4Backend for archmage::ScalarToken {
    type Repr = [f32; 4];

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: f32) -> [f32; 4] {
        [v; 4]
    }

    #[inline(always)]
    fn zero() -> [f32; 4] {
        [0.0f32; 4]
    }

    #[inline(always)]
    fn load(data: &[f32; 4]) -> [f32; 4] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [f32; 4]) -> [f32; 4] {
        arr
    }

    #[inline(always)]
    fn store(repr: [f32; 4], out: &mut [f32; 4]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [f32; 4]) -> [f32; 4] {
        repr
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
    }

    #[inline(always)]
    fn sub(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
    }

    #[inline(always)]
    fn mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
    }

    #[inline(always)]
    fn div(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        [a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]]
    }

    #[inline(always)]
    fn neg(a: [f32; 4]) -> [f32; 4] {
        [-a[0], -a[1], -a[2], -a[3]]
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        [
            a[0].min(b[0]),
            a[1].min(b[1]),
            a[2].min(b[2]),
            a[3].min(b[3]),
        ]
    }

    #[inline(always)]
    fn max(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        [
            a[0].max(b[0]),
            a[1].max(b[1]),
            a[2].max(b[2]),
            a[3].max(b[3]),
        ]
    }

    #[inline(always)]
    fn sqrt(a: [f32; 4]) -> [f32; 4] {
        let mut r = [0.0f32; 4];
        for i in 0..4 {
            r[i] = f32_sqrt(a[i]);
        }
        r
    }

    #[inline(always)]
    fn abs(a: [f32; 4]) -> [f32; 4] {
        [
            f32::from_bits(a[0].to_bits() & 0x7FFF_FFFF),
            f32::from_bits(a[1].to_bits() & 0x7FFF_FFFF),
            f32::from_bits(a[2].to_bits() & 0x7FFF_FFFF),
            f32::from_bits(a[3].to_bits() & 0x7FFF_FFFF),
        ]
    }

    #[inline(always)]
    fn floor(a: [f32; 4]) -> [f32; 4] {
        let mut r = [0.0f32; 4];
        for i in 0..4 {
            r[i] = f32_floor(a[i]);
        }
        r
    }

    #[inline(always)]
    fn ceil(a: [f32; 4]) -> [f32; 4] {
        let mut r = [0.0f32; 4];
        for i in 0..4 {
            r[i] = f32_ceil(a[i]);
        }
        r
    }

    #[inline(always)]
    fn round(a: [f32; 4]) -> [f32; 4] {
        let mut r = [0.0f32; 4];
        for i in 0..4 {
            r[i] = f32_round(a[i]);
        }
        r
    }

    #[inline(always)]
    fn mul_add(a: [f32; 4], b: [f32; 4], c: [f32; 4]) -> [f32; 4] {
        [
            a[0] * b[0] + c[0],
            a[1] * b[1] + c[1],
            a[2] * b[2] + c[2],
            a[3] * b[3] + c[3],
        ]
    }

    #[inline(always)]
    fn mul_sub(a: [f32; 4], b: [f32; 4], c: [f32; 4]) -> [f32; 4] {
        [
            a[0] * b[0] - c[0],
            a[1] * b[1] - c[1],
            a[2] * b[2] - c[2],
            a[3] * b[3] - c[3],
        ]
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        let mut r = [0.0f32; 4];
        for i in 0..4 {
            r[i] = if a[i] == b[i] {
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_ne(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        let mut r = [0.0f32; 4];
        for i in 0..4 {
            r[i] = if a[i] != b[i] {
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_lt(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        let mut r = [0.0f32; 4];
        for i in 0..4 {
            r[i] = if a[i] < b[i] {
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_le(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        let mut r = [0.0f32; 4];
        for i in 0..4 {
            r[i] = if a[i] <= b[i] {
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_gt(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        let mut r = [0.0f32; 4];
        for i in 0..4 {
            r[i] = if a[i] > b[i] {
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_ge(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        let mut r = [0.0f32; 4];
        for i in 0..4 {
            r[i] = if a[i] >= b[i] {
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn blend(mask: [f32; 4], if_true: [f32; 4], if_false: [f32; 4]) -> [f32; 4] {
        let mut r = [0.0f32; 4];
        for i in 0..4 {
            // Check sign bit of mask (all-1s has sign bit set)
            r[i] = if (mask[i].to_bits() >> 31) != 0 {
                if_true[i]
            } else {
                if_false[i]
            };
        }
        r
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: [f32; 4]) -> f32 {
        a[0] + a[1] + a[2] + a[3]
    }

    #[inline(always)]
    fn reduce_min(a: [f32; 4]) -> f32 {
        let mut m = a[0];
        for &v in &a[1..] {
            m = m.min(v);
        }
        m
    }

    #[inline(always)]
    fn reduce_max(a: [f32; 4]) -> f32 {
        let mut m = a[0];
        for &v in &a[1..] {
            m = m.max(v);
        }
        m
    }

    // ====== Approximations ======

    #[inline(always)]
    fn rcp_approx(a: [f32; 4]) -> [f32; 4] {
        [1.0 / a[0], 1.0 / a[1], 1.0 / a[2], 1.0 / a[3]]
    }

    #[inline(always)]
    fn rsqrt_approx(a: [f32; 4]) -> [f32; 4] {
        let mut r = [0.0f32; 4];
        for i in 0..4 {
            r[i] = 1.0 / f32_sqrt(a[i]);
        }
        r
    }

    // Override defaults: scalar doesn't need Newton-Raphson (already full precision)
    // Use FQS because ScalarToken implements multiple backend traits.
    #[inline(always)]
    fn recip(a: [f32; 4]) -> [f32; 4] {
        <archmage::ScalarToken as F32x4Backend>::rcp_approx(a)
    }

    #[inline(always)]
    fn rsqrt(a: [f32; 4]) -> [f32; 4] {
        <archmage::ScalarToken as F32x4Backend>::rsqrt_approx(a)
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: [f32; 4]) -> [f32; 4] {
        [
            f32::from_bits(!a[0].to_bits()),
            f32::from_bits(!a[1].to_bits()),
            f32::from_bits(!a[2].to_bits()),
            f32::from_bits(!a[3].to_bits()),
        ]
    }

    #[inline(always)]
    fn bitand(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        [
            f32::from_bits(a[0].to_bits() & b[0].to_bits()),
            f32::from_bits(a[1].to_bits() & b[1].to_bits()),
            f32::from_bits(a[2].to_bits() & b[2].to_bits()),
            f32::from_bits(a[3].to_bits() & b[3].to_bits()),
        ]
    }

    #[inline(always)]
    fn bitor(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        [
            f32::from_bits(a[0].to_bits() | b[0].to_bits()),
            f32::from_bits(a[1].to_bits() | b[1].to_bits()),
            f32::from_bits(a[2].to_bits() | b[2].to_bits()),
            f32::from_bits(a[3].to_bits() | b[3].to_bits()),
        ]
    }

    #[inline(always)]
    fn bitxor(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        [
            f32::from_bits(a[0].to_bits() ^ b[0].to_bits()),
            f32::from_bits(a[1].to_bits() ^ b[1].to_bits()),
            f32::from_bits(a[2].to_bits() ^ b[2].to_bits()),
            f32::from_bits(a[3].to_bits() ^ b[3].to_bits()),
        ]
    }
}

impl F32x8Backend for archmage::ScalarToken {
    type Repr = [f32; 8];

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: f32) -> [f32; 8] {
        [v; 8]
    }

    #[inline(always)]
    fn zero() -> [f32; 8] {
        [0.0f32; 8]
    }

    #[inline(always)]
    fn load(data: &[f32; 8]) -> [f32; 8] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [f32; 8]) -> [f32; 8] {
        arr
    }

    #[inline(always)]
    fn store(repr: [f32; 8], out: &mut [f32; 8]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [f32; 8]) -> [f32; 8] {
        repr
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        [
            a[0] + b[0],
            a[1] + b[1],
            a[2] + b[2],
            a[3] + b[3],
            a[4] + b[4],
            a[5] + b[5],
            a[6] + b[6],
            a[7] + b[7],
        ]
    }

    #[inline(always)]
    fn sub(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        [
            a[0] - b[0],
            a[1] - b[1],
            a[2] - b[2],
            a[3] - b[3],
            a[4] - b[4],
            a[5] - b[5],
            a[6] - b[6],
            a[7] - b[7],
        ]
    }

    #[inline(always)]
    fn mul(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        [
            a[0] * b[0],
            a[1] * b[1],
            a[2] * b[2],
            a[3] * b[3],
            a[4] * b[4],
            a[5] * b[5],
            a[6] * b[6],
            a[7] * b[7],
        ]
    }

    #[inline(always)]
    fn div(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        [
            a[0] / b[0],
            a[1] / b[1],
            a[2] / b[2],
            a[3] / b[3],
            a[4] / b[4],
            a[5] / b[5],
            a[6] / b[6],
            a[7] / b[7],
        ]
    }

    #[inline(always)]
    fn neg(a: [f32; 8]) -> [f32; 8] {
        [-a[0], -a[1], -a[2], -a[3], -a[4], -a[5], -a[6], -a[7]]
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        [
            a[0].min(b[0]),
            a[1].min(b[1]),
            a[2].min(b[2]),
            a[3].min(b[3]),
            a[4].min(b[4]),
            a[5].min(b[5]),
            a[6].min(b[6]),
            a[7].min(b[7]),
        ]
    }

    #[inline(always)]
    fn max(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        [
            a[0].max(b[0]),
            a[1].max(b[1]),
            a[2].max(b[2]),
            a[3].max(b[3]),
            a[4].max(b[4]),
            a[5].max(b[5]),
            a[6].max(b[6]),
            a[7].max(b[7]),
        ]
    }

    #[inline(always)]
    fn sqrt(a: [f32; 8]) -> [f32; 8] {
        let mut r = [0.0f32; 8];
        for i in 0..8 {
            r[i] = f32_sqrt(a[i]);
        }
        r
    }

    #[inline(always)]
    fn abs(a: [f32; 8]) -> [f32; 8] {
        [
            f32::from_bits(a[0].to_bits() & 0x7FFF_FFFF),
            f32::from_bits(a[1].to_bits() & 0x7FFF_FFFF),
            f32::from_bits(a[2].to_bits() & 0x7FFF_FFFF),
            f32::from_bits(a[3].to_bits() & 0x7FFF_FFFF),
            f32::from_bits(a[4].to_bits() & 0x7FFF_FFFF),
            f32::from_bits(a[5].to_bits() & 0x7FFF_FFFF),
            f32::from_bits(a[6].to_bits() & 0x7FFF_FFFF),
            f32::from_bits(a[7].to_bits() & 0x7FFF_FFFF),
        ]
    }

    #[inline(always)]
    fn floor(a: [f32; 8]) -> [f32; 8] {
        let mut r = [0.0f32; 8];
        for i in 0..8 {
            r[i] = f32_floor(a[i]);
        }
        r
    }

    #[inline(always)]
    fn ceil(a: [f32; 8]) -> [f32; 8] {
        let mut r = [0.0f32; 8];
        for i in 0..8 {
            r[i] = f32_ceil(a[i]);
        }
        r
    }

    #[inline(always)]
    fn round(a: [f32; 8]) -> [f32; 8] {
        let mut r = [0.0f32; 8];
        for i in 0..8 {
            r[i] = f32_round(a[i]);
        }
        r
    }

    #[inline(always)]
    fn mul_add(a: [f32; 8], b: [f32; 8], c: [f32; 8]) -> [f32; 8] {
        [
            a[0] * b[0] + c[0],
            a[1] * b[1] + c[1],
            a[2] * b[2] + c[2],
            a[3] * b[3] + c[3],
            a[4] * b[4] + c[4],
            a[5] * b[5] + c[5],
            a[6] * b[6] + c[6],
            a[7] * b[7] + c[7],
        ]
    }

    #[inline(always)]
    fn mul_sub(a: [f32; 8], b: [f32; 8], c: [f32; 8]) -> [f32; 8] {
        [
            a[0] * b[0] - c[0],
            a[1] * b[1] - c[1],
            a[2] * b[2] - c[2],
            a[3] * b[3] - c[3],
            a[4] * b[4] - c[4],
            a[5] * b[5] - c[5],
            a[6] * b[6] - c[6],
            a[7] * b[7] - c[7],
        ]
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        let mut r = [0.0f32; 8];
        for i in 0..8 {
            r[i] = if a[i] == b[i] {
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_ne(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        let mut r = [0.0f32; 8];
        for i in 0..8 {
            r[i] = if a[i] != b[i] {
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_lt(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        let mut r = [0.0f32; 8];
        for i in 0..8 {
            r[i] = if a[i] < b[i] {
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_le(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        let mut r = [0.0f32; 8];
        for i in 0..8 {
            r[i] = if a[i] <= b[i] {
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_gt(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        let mut r = [0.0f32; 8];
        for i in 0..8 {
            r[i] = if a[i] > b[i] {
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_ge(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        let mut r = [0.0f32; 8];
        for i in 0..8 {
            r[i] = if a[i] >= b[i] {
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn blend(mask: [f32; 8], if_true: [f32; 8], if_false: [f32; 8]) -> [f32; 8] {
        let mut r = [0.0f32; 8];
        for i in 0..8 {
            // Check sign bit of mask (all-1s has sign bit set)
            r[i] = if (mask[i].to_bits() >> 31) != 0 {
                if_true[i]
            } else {
                if_false[i]
            };
        }
        r
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: [f32; 8]) -> f32 {
        a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7]
    }

    #[inline(always)]
    fn reduce_min(a: [f32; 8]) -> f32 {
        let mut m = a[0];
        for &v in &a[1..] {
            m = m.min(v);
        }
        m
    }

    #[inline(always)]
    fn reduce_max(a: [f32; 8]) -> f32 {
        let mut m = a[0];
        for &v in &a[1..] {
            m = m.max(v);
        }
        m
    }

    // ====== Approximations ======

    #[inline(always)]
    fn rcp_approx(a: [f32; 8]) -> [f32; 8] {
        [
            1.0 / a[0],
            1.0 / a[1],
            1.0 / a[2],
            1.0 / a[3],
            1.0 / a[4],
            1.0 / a[5],
            1.0 / a[6],
            1.0 / a[7],
        ]
    }

    #[inline(always)]
    fn rsqrt_approx(a: [f32; 8]) -> [f32; 8] {
        let mut r = [0.0f32; 8];
        for i in 0..8 {
            r[i] = 1.0 / f32_sqrt(a[i]);
        }
        r
    }

    // Override defaults: scalar doesn't need Newton-Raphson (already full precision)
    // Use FQS because ScalarToken implements multiple backend traits.
    #[inline(always)]
    fn recip(a: [f32; 8]) -> [f32; 8] {
        <archmage::ScalarToken as F32x8Backend>::rcp_approx(a)
    }

    #[inline(always)]
    fn rsqrt(a: [f32; 8]) -> [f32; 8] {
        <archmage::ScalarToken as F32x8Backend>::rsqrt_approx(a)
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: [f32; 8]) -> [f32; 8] {
        [
            f32::from_bits(!a[0].to_bits()),
            f32::from_bits(!a[1].to_bits()),
            f32::from_bits(!a[2].to_bits()),
            f32::from_bits(!a[3].to_bits()),
            f32::from_bits(!a[4].to_bits()),
            f32::from_bits(!a[5].to_bits()),
            f32::from_bits(!a[6].to_bits()),
            f32::from_bits(!a[7].to_bits()),
        ]
    }

    #[inline(always)]
    fn bitand(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        [
            f32::from_bits(a[0].to_bits() & b[0].to_bits()),
            f32::from_bits(a[1].to_bits() & b[1].to_bits()),
            f32::from_bits(a[2].to_bits() & b[2].to_bits()),
            f32::from_bits(a[3].to_bits() & b[3].to_bits()),
            f32::from_bits(a[4].to_bits() & b[4].to_bits()),
            f32::from_bits(a[5].to_bits() & b[5].to_bits()),
            f32::from_bits(a[6].to_bits() & b[6].to_bits()),
            f32::from_bits(a[7].to_bits() & b[7].to_bits()),
        ]
    }

    #[inline(always)]
    fn bitor(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        [
            f32::from_bits(a[0].to_bits() | b[0].to_bits()),
            f32::from_bits(a[1].to_bits() | b[1].to_bits()),
            f32::from_bits(a[2].to_bits() | b[2].to_bits()),
            f32::from_bits(a[3].to_bits() | b[3].to_bits()),
            f32::from_bits(a[4].to_bits() | b[4].to_bits()),
            f32::from_bits(a[5].to_bits() | b[5].to_bits()),
            f32::from_bits(a[6].to_bits() | b[6].to_bits()),
            f32::from_bits(a[7].to_bits() | b[7].to_bits()),
        ]
    }

    #[inline(always)]
    fn bitxor(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        [
            f32::from_bits(a[0].to_bits() ^ b[0].to_bits()),
            f32::from_bits(a[1].to_bits() ^ b[1].to_bits()),
            f32::from_bits(a[2].to_bits() ^ b[2].to_bits()),
            f32::from_bits(a[3].to_bits() ^ b[3].to_bits()),
            f32::from_bits(a[4].to_bits() ^ b[4].to_bits()),
            f32::from_bits(a[5].to_bits() ^ b[5].to_bits()),
            f32::from_bits(a[6].to_bits() ^ b[6].to_bits()),
            f32::from_bits(a[7].to_bits() ^ b[7].to_bits()),
        ]
    }
}

impl F64x2Backend for archmage::ScalarToken {
    type Repr = [f64; 2];

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: f64) -> [f64; 2] {
        [v; 2]
    }

    #[inline(always)]
    fn zero() -> [f64; 2] {
        [0.0f64; 2]
    }

    #[inline(always)]
    fn load(data: &[f64; 2]) -> [f64; 2] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [f64; 2]) -> [f64; 2] {
        arr
    }

    #[inline(always)]
    fn store(repr: [f64; 2], out: &mut [f64; 2]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [f64; 2]) -> [f64; 2] {
        repr
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        [a[0] + b[0], a[1] + b[1]]
    }

    #[inline(always)]
    fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        [a[0] - b[0], a[1] - b[1]]
    }

    #[inline(always)]
    fn mul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        [a[0] * b[0], a[1] * b[1]]
    }

    #[inline(always)]
    fn div(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        [a[0] / b[0], a[1] / b[1]]
    }

    #[inline(always)]
    fn neg(a: [f64; 2]) -> [f64; 2] {
        [-a[0], -a[1]]
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        [a[0].min(b[0]), a[1].min(b[1])]
    }

    #[inline(always)]
    fn max(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        [a[0].max(b[0]), a[1].max(b[1])]
    }

    #[inline(always)]
    fn sqrt(a: [f64; 2]) -> [f64; 2] {
        let mut r = [0.0f64; 2];
        for i in 0..2 {
            r[i] = f64_sqrt(a[i]);
        }
        r
    }

    #[inline(always)]
    fn abs(a: [f64; 2]) -> [f64; 2] {
        [
            f64::from_bits(a[0].to_bits() & 0x7FFF_FFFF_FFFF_FFFF),
            f64::from_bits(a[1].to_bits() & 0x7FFF_FFFF_FFFF_FFFF),
        ]
    }

    #[inline(always)]
    fn floor(a: [f64; 2]) -> [f64; 2] {
        let mut r = [0.0f64; 2];
        for i in 0..2 {
            r[i] = f64_floor(a[i]);
        }
        r
    }

    #[inline(always)]
    fn ceil(a: [f64; 2]) -> [f64; 2] {
        let mut r = [0.0f64; 2];
        for i in 0..2 {
            r[i] = f64_ceil(a[i]);
        }
        r
    }

    #[inline(always)]
    fn round(a: [f64; 2]) -> [f64; 2] {
        let mut r = [0.0f64; 2];
        for i in 0..2 {
            r[i] = f64_round(a[i]);
        }
        r
    }

    #[inline(always)]
    fn mul_add(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> [f64; 2] {
        [a[0] * b[0] + c[0], a[1] * b[1] + c[1]]
    }

    #[inline(always)]
    fn mul_sub(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> [f64; 2] {
        [a[0] * b[0] - c[0], a[1] * b[1] - c[1]]
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        let mut r = [0.0f64; 2];
        for i in 0..2 {
            r[i] = if a[i] == b[i] {
                f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_ne(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        let mut r = [0.0f64; 2];
        for i in 0..2 {
            r[i] = if a[i] != b[i] {
                f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_lt(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        let mut r = [0.0f64; 2];
        for i in 0..2 {
            r[i] = if a[i] < b[i] {
                f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_le(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        let mut r = [0.0f64; 2];
        for i in 0..2 {
            r[i] = if a[i] <= b[i] {
                f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_gt(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        let mut r = [0.0f64; 2];
        for i in 0..2 {
            r[i] = if a[i] > b[i] {
                f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_ge(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        let mut r = [0.0f64; 2];
        for i in 0..2 {
            r[i] = if a[i] >= b[i] {
                f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn blend(mask: [f64; 2], if_true: [f64; 2], if_false: [f64; 2]) -> [f64; 2] {
        let mut r = [0.0f64; 2];
        for i in 0..2 {
            // Check sign bit of mask (all-1s has sign bit set)
            r[i] = if (mask[i].to_bits() >> 63) != 0 {
                if_true[i]
            } else {
                if_false[i]
            };
        }
        r
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: [f64; 2]) -> f64 {
        a[0] + a[1]
    }

    #[inline(always)]
    fn reduce_min(a: [f64; 2]) -> f64 {
        let mut m = a[0];
        for &v in &a[1..] {
            m = m.min(v);
        }
        m
    }

    #[inline(always)]
    fn reduce_max(a: [f64; 2]) -> f64 {
        let mut m = a[0];
        for &v in &a[1..] {
            m = m.max(v);
        }
        m
    }

    // ====== Approximations ======

    #[inline(always)]
    fn rcp_approx(a: [f64; 2]) -> [f64; 2] {
        [1.0 / a[0], 1.0 / a[1]]
    }

    #[inline(always)]
    fn rsqrt_approx(a: [f64; 2]) -> [f64; 2] {
        let mut r = [0.0f64; 2];
        for i in 0..2 {
            r[i] = 1.0 / f64_sqrt(a[i]);
        }
        r
    }

    // Override defaults: scalar doesn't need Newton-Raphson (already full precision)
    // Use FQS because ScalarToken implements multiple backend traits.
    #[inline(always)]
    fn recip(a: [f64; 2]) -> [f64; 2] {
        <archmage::ScalarToken as F64x2Backend>::rcp_approx(a)
    }

    #[inline(always)]
    fn rsqrt(a: [f64; 2]) -> [f64; 2] {
        <archmage::ScalarToken as F64x2Backend>::rsqrt_approx(a)
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: [f64; 2]) -> [f64; 2] {
        [
            f64::from_bits(!a[0].to_bits()),
            f64::from_bits(!a[1].to_bits()),
        ]
    }

    #[inline(always)]
    fn bitand(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        [
            f64::from_bits(a[0].to_bits() & b[0].to_bits()),
            f64::from_bits(a[1].to_bits() & b[1].to_bits()),
        ]
    }

    #[inline(always)]
    fn bitor(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        [
            f64::from_bits(a[0].to_bits() | b[0].to_bits()),
            f64::from_bits(a[1].to_bits() | b[1].to_bits()),
        ]
    }

    #[inline(always)]
    fn bitxor(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        [
            f64::from_bits(a[0].to_bits() ^ b[0].to_bits()),
            f64::from_bits(a[1].to_bits() ^ b[1].to_bits()),
        ]
    }
}

impl F64x4Backend for archmage::ScalarToken {
    type Repr = [f64; 4];

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: f64) -> [f64; 4] {
        [v; 4]
    }

    #[inline(always)]
    fn zero() -> [f64; 4] {
        [0.0f64; 4]
    }

    #[inline(always)]
    fn load(data: &[f64; 4]) -> [f64; 4] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [f64; 4]) -> [f64; 4] {
        arr
    }

    #[inline(always)]
    fn store(repr: [f64; 4], out: &mut [f64; 4]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [f64; 4]) -> [f64; 4] {
        repr
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
    }

    #[inline(always)]
    fn sub(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
    }

    #[inline(always)]
    fn mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
    }

    #[inline(always)]
    fn div(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        [a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]]
    }

    #[inline(always)]
    fn neg(a: [f64; 4]) -> [f64; 4] {
        [-a[0], -a[1], -a[2], -a[3]]
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        [
            a[0].min(b[0]),
            a[1].min(b[1]),
            a[2].min(b[2]),
            a[3].min(b[3]),
        ]
    }

    #[inline(always)]
    fn max(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        [
            a[0].max(b[0]),
            a[1].max(b[1]),
            a[2].max(b[2]),
            a[3].max(b[3]),
        ]
    }

    #[inline(always)]
    fn sqrt(a: [f64; 4]) -> [f64; 4] {
        let mut r = [0.0f64; 4];
        for i in 0..4 {
            r[i] = f64_sqrt(a[i]);
        }
        r
    }

    #[inline(always)]
    fn abs(a: [f64; 4]) -> [f64; 4] {
        [
            f64::from_bits(a[0].to_bits() & 0x7FFF_FFFF_FFFF_FFFF),
            f64::from_bits(a[1].to_bits() & 0x7FFF_FFFF_FFFF_FFFF),
            f64::from_bits(a[2].to_bits() & 0x7FFF_FFFF_FFFF_FFFF),
            f64::from_bits(a[3].to_bits() & 0x7FFF_FFFF_FFFF_FFFF),
        ]
    }

    #[inline(always)]
    fn floor(a: [f64; 4]) -> [f64; 4] {
        let mut r = [0.0f64; 4];
        for i in 0..4 {
            r[i] = f64_floor(a[i]);
        }
        r
    }

    #[inline(always)]
    fn ceil(a: [f64; 4]) -> [f64; 4] {
        let mut r = [0.0f64; 4];
        for i in 0..4 {
            r[i] = f64_ceil(a[i]);
        }
        r
    }

    #[inline(always)]
    fn round(a: [f64; 4]) -> [f64; 4] {
        let mut r = [0.0f64; 4];
        for i in 0..4 {
            r[i] = f64_round(a[i]);
        }
        r
    }

    #[inline(always)]
    fn mul_add(a: [f64; 4], b: [f64; 4], c: [f64; 4]) -> [f64; 4] {
        [
            a[0] * b[0] + c[0],
            a[1] * b[1] + c[1],
            a[2] * b[2] + c[2],
            a[3] * b[3] + c[3],
        ]
    }

    #[inline(always)]
    fn mul_sub(a: [f64; 4], b: [f64; 4], c: [f64; 4]) -> [f64; 4] {
        [
            a[0] * b[0] - c[0],
            a[1] * b[1] - c[1],
            a[2] * b[2] - c[2],
            a[3] * b[3] - c[3],
        ]
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        let mut r = [0.0f64; 4];
        for i in 0..4 {
            r[i] = if a[i] == b[i] {
                f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_ne(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        let mut r = [0.0f64; 4];
        for i in 0..4 {
            r[i] = if a[i] != b[i] {
                f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_lt(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        let mut r = [0.0f64; 4];
        for i in 0..4 {
            r[i] = if a[i] < b[i] {
                f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_le(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        let mut r = [0.0f64; 4];
        for i in 0..4 {
            r[i] = if a[i] <= b[i] {
                f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_gt(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        let mut r = [0.0f64; 4];
        for i in 0..4 {
            r[i] = if a[i] > b[i] {
                f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn simd_ge(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        let mut r = [0.0f64; 4];
        for i in 0..4 {
            r[i] = if a[i] >= b[i] {
                f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)
            } else {
                0.0
            };
        }
        r
    }

    #[inline(always)]
    fn blend(mask: [f64; 4], if_true: [f64; 4], if_false: [f64; 4]) -> [f64; 4] {
        let mut r = [0.0f64; 4];
        for i in 0..4 {
            // Check sign bit of mask (all-1s has sign bit set)
            r[i] = if (mask[i].to_bits() >> 63) != 0 {
                if_true[i]
            } else {
                if_false[i]
            };
        }
        r
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: [f64; 4]) -> f64 {
        a[0] + a[1] + a[2] + a[3]
    }

    #[inline(always)]
    fn reduce_min(a: [f64; 4]) -> f64 {
        let mut m = a[0];
        for &v in &a[1..] {
            m = m.min(v);
        }
        m
    }

    #[inline(always)]
    fn reduce_max(a: [f64; 4]) -> f64 {
        let mut m = a[0];
        for &v in &a[1..] {
            m = m.max(v);
        }
        m
    }

    // ====== Approximations ======

    #[inline(always)]
    fn rcp_approx(a: [f64; 4]) -> [f64; 4] {
        [1.0 / a[0], 1.0 / a[1], 1.0 / a[2], 1.0 / a[3]]
    }

    #[inline(always)]
    fn rsqrt_approx(a: [f64; 4]) -> [f64; 4] {
        let mut r = [0.0f64; 4];
        for i in 0..4 {
            r[i] = 1.0 / f64_sqrt(a[i]);
        }
        r
    }

    // Override defaults: scalar doesn't need Newton-Raphson (already full precision)
    // Use FQS because ScalarToken implements multiple backend traits.
    #[inline(always)]
    fn recip(a: [f64; 4]) -> [f64; 4] {
        <archmage::ScalarToken as F64x4Backend>::rcp_approx(a)
    }

    #[inline(always)]
    fn rsqrt(a: [f64; 4]) -> [f64; 4] {
        <archmage::ScalarToken as F64x4Backend>::rsqrt_approx(a)
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: [f64; 4]) -> [f64; 4] {
        [
            f64::from_bits(!a[0].to_bits()),
            f64::from_bits(!a[1].to_bits()),
            f64::from_bits(!a[2].to_bits()),
            f64::from_bits(!a[3].to_bits()),
        ]
    }

    #[inline(always)]
    fn bitand(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        [
            f64::from_bits(a[0].to_bits() & b[0].to_bits()),
            f64::from_bits(a[1].to_bits() & b[1].to_bits()),
            f64::from_bits(a[2].to_bits() & b[2].to_bits()),
            f64::from_bits(a[3].to_bits() & b[3].to_bits()),
        ]
    }

    #[inline(always)]
    fn bitor(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        [
            f64::from_bits(a[0].to_bits() | b[0].to_bits()),
            f64::from_bits(a[1].to_bits() | b[1].to_bits()),
            f64::from_bits(a[2].to_bits() | b[2].to_bits()),
            f64::from_bits(a[3].to_bits() | b[3].to_bits()),
        ]
    }

    #[inline(always)]
    fn bitxor(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        [
            f64::from_bits(a[0].to_bits() ^ b[0].to_bits()),
            f64::from_bits(a[1].to_bits() ^ b[1].to_bits()),
            f64::from_bits(a[2].to_bits() ^ b[2].to_bits()),
            f64::from_bits(a[3].to_bits() ^ b[3].to_bits()),
        ]
    }
}

impl I32x4Backend for archmage::ScalarToken {
    type Repr = [i32; 4];

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: i32) -> [i32; 4] {
        [v; 4]
    }

    #[inline(always)]
    fn zero() -> [i32; 4] {
        [0i32; 4]
    }

    #[inline(always)]
    fn load(data: &[i32; 4]) -> [i32; 4] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [i32; 4]) -> [i32; 4] {
        arr
    }

    #[inline(always)]
    fn store(repr: [i32; 4], out: &mut [i32; 4]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [i32; 4]) -> [i32; 4] {
        repr
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
        [
            a[0].wrapping_add(b[0]),
            a[1].wrapping_add(b[1]),
            a[2].wrapping_add(b[2]),
            a[3].wrapping_add(b[3]),
        ]
    }

    #[inline(always)]
    fn sub(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
        [
            a[0].wrapping_sub(b[0]),
            a[1].wrapping_sub(b[1]),
            a[2].wrapping_sub(b[2]),
            a[3].wrapping_sub(b[3]),
        ]
    }

    #[inline(always)]
    fn mul(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
        [
            a[0].wrapping_mul(b[0]),
            a[1].wrapping_mul(b[1]),
            a[2].wrapping_mul(b[2]),
            a[3].wrapping_mul(b[3]),
        ]
    }

    #[inline(always)]
    fn neg(a: [i32; 4]) -> [i32; 4] {
        [
            a[0].wrapping_neg(),
            a[1].wrapping_neg(),
            a[2].wrapping_neg(),
            a[3].wrapping_neg(),
        ]
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
        [
            if a[0] < b[0] { a[0] } else { b[0] },
            if a[1] < b[1] { a[1] } else { b[1] },
            if a[2] < b[2] { a[2] } else { b[2] },
            if a[3] < b[3] { a[3] } else { b[3] },
        ]
    }

    #[inline(always)]
    fn max(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
        [
            if a[0] > b[0] { a[0] } else { b[0] },
            if a[1] > b[1] { a[1] } else { b[1] },
            if a[2] > b[2] { a[2] } else { b[2] },
            if a[3] > b[3] { a[3] } else { b[3] },
        ]
    }

    #[inline(always)]
    fn abs(a: [i32; 4]) -> [i32; 4] {
        [
            a[0].wrapping_abs(),
            a[1].wrapping_abs(),
            a[2].wrapping_abs(),
            a[3].wrapping_abs(),
        ]
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
        let mut r = [0i32; 4];
        for i in 0..4 {
            r[i] = if a[i] == b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_ne(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
        let mut r = [0i32; 4];
        for i in 0..4 {
            r[i] = if a[i] != b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_lt(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
        let mut r = [0i32; 4];
        for i in 0..4 {
            r[i] = if a[i] < b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_le(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
        let mut r = [0i32; 4];
        for i in 0..4 {
            r[i] = if a[i] <= b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_gt(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
        let mut r = [0i32; 4];
        for i in 0..4 {
            r[i] = if a[i] > b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_ge(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
        let mut r = [0i32; 4];
        for i in 0..4 {
            r[i] = if a[i] >= b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn blend(mask: [i32; 4], if_true: [i32; 4], if_false: [i32; 4]) -> [i32; 4] {
        let mut r = [0i32; 4];
        for i in 0..4 {
            r[i] = if mask[i] != 0 {
                if_true[i]
            } else {
                if_false[i]
            };
        }
        r
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: [i32; 4]) -> i32 {
        a[0].wrapping_add(a[1].wrapping_add(a[2].wrapping_add(a[3])))
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: [i32; 4]) -> [i32; 4] {
        [!a[0], !a[1], !a[2], !a[3]]
    }

    #[inline(always)]
    fn bitand(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
        [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]
    }

    #[inline(always)]
    fn bitor(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
        [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]
    }

    #[inline(always)]
    fn bitxor(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
        [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
    }

    // ====== Shifts ======

    #[inline(always)]
    fn shl_const<const N: i32>(a: [i32; 4]) -> [i32; 4] {
        [a[0] << N, a[1] << N, a[2] << N, a[3] << N]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [i32; 4]) -> [i32; 4] {
        [a[0] >> N, a[1] >> N, a[2] >> N, a[3] >> N]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [i32; 4]) -> [i32; 4] {
        [
            ((a[0] as u32) >> N) as i32,
            ((a[1] as u32) >> N) as i32,
            ((a[2] as u32) >> N) as i32,
            ((a[3] as u32) >> N) as i32,
        ]
    }

    // ====== Boolean ======

    #[inline(always)]
    fn all_true(a: [i32; 4]) -> bool {
        a[0] != 0 && a[1] != 0 && a[2] != 0 && a[3] != 0
    }

    #[inline(always)]
    fn any_true(a: [i32; 4]) -> bool {
        a[0] != 0 || a[1] != 0 || a[2] != 0 || a[3] != 0
    }

    #[inline(always)]
    fn bitmask(a: [i32; 4]) -> u32 {
        ((a[0] as u32) >> 31)
            | (((a[1] as u32) >> 31) << 1)
            | (((a[2] as u32) >> 31) << 2)
            | (((a[3] as u32) >> 31) << 3)
    }
}

impl I32x8Backend for archmage::ScalarToken {
    type Repr = [i32; 8];

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: i32) -> [i32; 8] {
        [v; 8]
    }

    #[inline(always)]
    fn zero() -> [i32; 8] {
        [0i32; 8]
    }

    #[inline(always)]
    fn load(data: &[i32; 8]) -> [i32; 8] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [i32; 8]) -> [i32; 8] {
        arr
    }

    #[inline(always)]
    fn store(repr: [i32; 8], out: &mut [i32; 8]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [i32; 8]) -> [i32; 8] {
        repr
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
        [
            a[0].wrapping_add(b[0]),
            a[1].wrapping_add(b[1]),
            a[2].wrapping_add(b[2]),
            a[3].wrapping_add(b[3]),
            a[4].wrapping_add(b[4]),
            a[5].wrapping_add(b[5]),
            a[6].wrapping_add(b[6]),
            a[7].wrapping_add(b[7]),
        ]
    }

    #[inline(always)]
    fn sub(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
        [
            a[0].wrapping_sub(b[0]),
            a[1].wrapping_sub(b[1]),
            a[2].wrapping_sub(b[2]),
            a[3].wrapping_sub(b[3]),
            a[4].wrapping_sub(b[4]),
            a[5].wrapping_sub(b[5]),
            a[6].wrapping_sub(b[6]),
            a[7].wrapping_sub(b[7]),
        ]
    }

    #[inline(always)]
    fn mul(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
        [
            a[0].wrapping_mul(b[0]),
            a[1].wrapping_mul(b[1]),
            a[2].wrapping_mul(b[2]),
            a[3].wrapping_mul(b[3]),
            a[4].wrapping_mul(b[4]),
            a[5].wrapping_mul(b[5]),
            a[6].wrapping_mul(b[6]),
            a[7].wrapping_mul(b[7]),
        ]
    }

    #[inline(always)]
    fn neg(a: [i32; 8]) -> [i32; 8] {
        [
            a[0].wrapping_neg(),
            a[1].wrapping_neg(),
            a[2].wrapping_neg(),
            a[3].wrapping_neg(),
            a[4].wrapping_neg(),
            a[5].wrapping_neg(),
            a[6].wrapping_neg(),
            a[7].wrapping_neg(),
        ]
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
        [
            if a[0] < b[0] { a[0] } else { b[0] },
            if a[1] < b[1] { a[1] } else { b[1] },
            if a[2] < b[2] { a[2] } else { b[2] },
            if a[3] < b[3] { a[3] } else { b[3] },
            if a[4] < b[4] { a[4] } else { b[4] },
            if a[5] < b[5] { a[5] } else { b[5] },
            if a[6] < b[6] { a[6] } else { b[6] },
            if a[7] < b[7] { a[7] } else { b[7] },
        ]
    }

    #[inline(always)]
    fn max(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
        [
            if a[0] > b[0] { a[0] } else { b[0] },
            if a[1] > b[1] { a[1] } else { b[1] },
            if a[2] > b[2] { a[2] } else { b[2] },
            if a[3] > b[3] { a[3] } else { b[3] },
            if a[4] > b[4] { a[4] } else { b[4] },
            if a[5] > b[5] { a[5] } else { b[5] },
            if a[6] > b[6] { a[6] } else { b[6] },
            if a[7] > b[7] { a[7] } else { b[7] },
        ]
    }

    #[inline(always)]
    fn abs(a: [i32; 8]) -> [i32; 8] {
        [
            a[0].wrapping_abs(),
            a[1].wrapping_abs(),
            a[2].wrapping_abs(),
            a[3].wrapping_abs(),
            a[4].wrapping_abs(),
            a[5].wrapping_abs(),
            a[6].wrapping_abs(),
            a[7].wrapping_abs(),
        ]
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = if a[i] == b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_ne(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = if a[i] != b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_lt(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = if a[i] < b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_le(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = if a[i] <= b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_gt(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = if a[i] > b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_ge(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = if a[i] >= b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn blend(mask: [i32; 8], if_true: [i32; 8], if_false: [i32; 8]) -> [i32; 8] {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = if mask[i] != 0 {
                if_true[i]
            } else {
                if_false[i]
            };
        }
        r
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: [i32; 8]) -> i32 {
        a[0].wrapping_add(a[1].wrapping_add(a[2].wrapping_add(
            a[3].wrapping_add(a[4].wrapping_add(a[5].wrapping_add(a[6].wrapping_add(a[7])))),
        )))
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: [i32; 8]) -> [i32; 8] {
        [!a[0], !a[1], !a[2], !a[3], !a[4], !a[5], !a[6], !a[7]]
    }

    #[inline(always)]
    fn bitand(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
        [
            a[0] & b[0],
            a[1] & b[1],
            a[2] & b[2],
            a[3] & b[3],
            a[4] & b[4],
            a[5] & b[5],
            a[6] & b[6],
            a[7] & b[7],
        ]
    }

    #[inline(always)]
    fn bitor(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
        [
            a[0] | b[0],
            a[1] | b[1],
            a[2] | b[2],
            a[3] | b[3],
            a[4] | b[4],
            a[5] | b[5],
            a[6] | b[6],
            a[7] | b[7],
        ]
    }

    #[inline(always)]
    fn bitxor(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
        [
            a[0] ^ b[0],
            a[1] ^ b[1],
            a[2] ^ b[2],
            a[3] ^ b[3],
            a[4] ^ b[4],
            a[5] ^ b[5],
            a[6] ^ b[6],
            a[7] ^ b[7],
        ]
    }

    // ====== Shifts ======

    #[inline(always)]
    fn shl_const<const N: i32>(a: [i32; 8]) -> [i32; 8] {
        [
            a[0] << N,
            a[1] << N,
            a[2] << N,
            a[3] << N,
            a[4] << N,
            a[5] << N,
            a[6] << N,
            a[7] << N,
        ]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [i32; 8]) -> [i32; 8] {
        [
            a[0] >> N,
            a[1] >> N,
            a[2] >> N,
            a[3] >> N,
            a[4] >> N,
            a[5] >> N,
            a[6] >> N,
            a[7] >> N,
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [i32; 8]) -> [i32; 8] {
        [
            ((a[0] as u32) >> N) as i32,
            ((a[1] as u32) >> N) as i32,
            ((a[2] as u32) >> N) as i32,
            ((a[3] as u32) >> N) as i32,
            ((a[4] as u32) >> N) as i32,
            ((a[5] as u32) >> N) as i32,
            ((a[6] as u32) >> N) as i32,
            ((a[7] as u32) >> N) as i32,
        ]
    }

    // ====== Boolean ======

    #[inline(always)]
    fn all_true(a: [i32; 8]) -> bool {
        a[0] != 0
            && a[1] != 0
            && a[2] != 0
            && a[3] != 0
            && a[4] != 0
            && a[5] != 0
            && a[6] != 0
            && a[7] != 0
    }

    #[inline(always)]
    fn any_true(a: [i32; 8]) -> bool {
        a[0] != 0
            || a[1] != 0
            || a[2] != 0
            || a[3] != 0
            || a[4] != 0
            || a[5] != 0
            || a[6] != 0
            || a[7] != 0
    }

    #[inline(always)]
    fn bitmask(a: [i32; 8]) -> u32 {
        ((a[0] as u32) >> 31)
            | (((a[1] as u32) >> 31) << 1)
            | (((a[2] as u32) >> 31) << 2)
            | (((a[3] as u32) >> 31) << 3)
            | (((a[4] as u32) >> 31) << 4)
            | (((a[5] as u32) >> 31) << 5)
            | (((a[6] as u32) >> 31) << 6)
            | (((a[7] as u32) >> 31) << 7)
    }
}

impl U32x4Backend for archmage::ScalarToken {
    type Repr = [u32; 4];

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: u32) -> [u32; 4] {
        [v; 4]
    }

    #[inline(always)]
    fn zero() -> [u32; 4] {
        [0u32; 4]
    }

    #[inline(always)]
    fn load(data: &[u32; 4]) -> [u32; 4] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [u32; 4]) -> [u32; 4] {
        arr
    }

    #[inline(always)]
    fn store(repr: [u32; 4], out: &mut [u32; 4]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [u32; 4]) -> [u32; 4] {
        repr
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        [
            a[0].wrapping_add(b[0]),
            a[1].wrapping_add(b[1]),
            a[2].wrapping_add(b[2]),
            a[3].wrapping_add(b[3]),
        ]
    }

    #[inline(always)]
    fn sub(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        [
            a[0].wrapping_sub(b[0]),
            a[1].wrapping_sub(b[1]),
            a[2].wrapping_sub(b[2]),
            a[3].wrapping_sub(b[3]),
        ]
    }

    #[inline(always)]
    fn mul(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        [
            a[0].wrapping_mul(b[0]),
            a[1].wrapping_mul(b[1]),
            a[2].wrapping_mul(b[2]),
            a[3].wrapping_mul(b[3]),
        ]
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        [
            if a[0] < b[0] { a[0] } else { b[0] },
            if a[1] < b[1] { a[1] } else { b[1] },
            if a[2] < b[2] { a[2] } else { b[2] },
            if a[3] < b[3] { a[3] } else { b[3] },
        ]
    }

    #[inline(always)]
    fn max(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        [
            if a[0] > b[0] { a[0] } else { b[0] },
            if a[1] > b[1] { a[1] } else { b[1] },
            if a[2] > b[2] { a[2] } else { b[2] },
            if a[3] > b[3] { a[3] } else { b[3] },
        ]
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        let mut r = [0u32; 4];
        for i in 0..4 {
            r[i] = if a[i] == b[i] { u32::MAX } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_ne(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        let mut r = [0u32; 4];
        for i in 0..4 {
            r[i] = if a[i] != b[i] { u32::MAX } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_lt(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        let mut r = [0u32; 4];
        for i in 0..4 {
            r[i] = if a[i] < b[i] { u32::MAX } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_le(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        let mut r = [0u32; 4];
        for i in 0..4 {
            r[i] = if a[i] <= b[i] { u32::MAX } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_gt(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        let mut r = [0u32; 4];
        for i in 0..4 {
            r[i] = if a[i] > b[i] { u32::MAX } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_ge(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        let mut r = [0u32; 4];
        for i in 0..4 {
            r[i] = if a[i] >= b[i] { u32::MAX } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn blend(mask: [u32; 4], if_true: [u32; 4], if_false: [u32; 4]) -> [u32; 4] {
        let mut r = [0u32; 4];
        for i in 0..4 {
            r[i] = if mask[i] != 0 {
                if_true[i]
            } else {
                if_false[i]
            };
        }
        r
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: [u32; 4]) -> u32 {
        a[0].wrapping_add(a[1].wrapping_add(a[2].wrapping_add(a[3])))
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: [u32; 4]) -> [u32; 4] {
        [!a[0], !a[1], !a[2], !a[3]]
    }

    #[inline(always)]
    fn bitand(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]
    }

    #[inline(always)]
    fn bitor(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]
    }

    #[inline(always)]
    fn bitxor(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
    }

    // ====== Shifts ======

    #[inline(always)]
    fn shl_const<const N: i32>(a: [u32; 4]) -> [u32; 4] {
        [a[0] << N, a[1] << N, a[2] << N, a[3] << N]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [u32; 4]) -> [u32; 4] {
        [a[0] >> N, a[1] >> N, a[2] >> N, a[3] >> N]
    }

    // ====== Boolean ======

    #[inline(always)]
    fn all_true(a: [u32; 4]) -> bool {
        a[0] != 0 && a[1] != 0 && a[2] != 0 && a[3] != 0
    }

    #[inline(always)]
    fn any_true(a: [u32; 4]) -> bool {
        a[0] != 0 || a[1] != 0 || a[2] != 0 || a[3] != 0
    }

    #[inline(always)]
    fn bitmask(a: [u32; 4]) -> u32 {
        (a[0] >> 31) | ((a[1] >> 31) << 1) | ((a[2] >> 31) << 2) | ((a[3] >> 31) << 3)
    }
}

impl U32x8Backend for archmage::ScalarToken {
    type Repr = [u32; 8];

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: u32) -> [u32; 8] {
        [v; 8]
    }

    #[inline(always)]
    fn zero() -> [u32; 8] {
        [0u32; 8]
    }

    #[inline(always)]
    fn load(data: &[u32; 8]) -> [u32; 8] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [u32; 8]) -> [u32; 8] {
        arr
    }

    #[inline(always)]
    fn store(repr: [u32; 8], out: &mut [u32; 8]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [u32; 8]) -> [u32; 8] {
        repr
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
        [
            a[0].wrapping_add(b[0]),
            a[1].wrapping_add(b[1]),
            a[2].wrapping_add(b[2]),
            a[3].wrapping_add(b[3]),
            a[4].wrapping_add(b[4]),
            a[5].wrapping_add(b[5]),
            a[6].wrapping_add(b[6]),
            a[7].wrapping_add(b[7]),
        ]
    }

    #[inline(always)]
    fn sub(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
        [
            a[0].wrapping_sub(b[0]),
            a[1].wrapping_sub(b[1]),
            a[2].wrapping_sub(b[2]),
            a[3].wrapping_sub(b[3]),
            a[4].wrapping_sub(b[4]),
            a[5].wrapping_sub(b[5]),
            a[6].wrapping_sub(b[6]),
            a[7].wrapping_sub(b[7]),
        ]
    }

    #[inline(always)]
    fn mul(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
        [
            a[0].wrapping_mul(b[0]),
            a[1].wrapping_mul(b[1]),
            a[2].wrapping_mul(b[2]),
            a[3].wrapping_mul(b[3]),
            a[4].wrapping_mul(b[4]),
            a[5].wrapping_mul(b[5]),
            a[6].wrapping_mul(b[6]),
            a[7].wrapping_mul(b[7]),
        ]
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
        [
            if a[0] < b[0] { a[0] } else { b[0] },
            if a[1] < b[1] { a[1] } else { b[1] },
            if a[2] < b[2] { a[2] } else { b[2] },
            if a[3] < b[3] { a[3] } else { b[3] },
            if a[4] < b[4] { a[4] } else { b[4] },
            if a[5] < b[5] { a[5] } else { b[5] },
            if a[6] < b[6] { a[6] } else { b[6] },
            if a[7] < b[7] { a[7] } else { b[7] },
        ]
    }

    #[inline(always)]
    fn max(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
        [
            if a[0] > b[0] { a[0] } else { b[0] },
            if a[1] > b[1] { a[1] } else { b[1] },
            if a[2] > b[2] { a[2] } else { b[2] },
            if a[3] > b[3] { a[3] } else { b[3] },
            if a[4] > b[4] { a[4] } else { b[4] },
            if a[5] > b[5] { a[5] } else { b[5] },
            if a[6] > b[6] { a[6] } else { b[6] },
            if a[7] > b[7] { a[7] } else { b[7] },
        ]
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
        let mut r = [0u32; 8];
        for i in 0..8 {
            r[i] = if a[i] == b[i] { u32::MAX } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_ne(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
        let mut r = [0u32; 8];
        for i in 0..8 {
            r[i] = if a[i] != b[i] { u32::MAX } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_lt(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
        let mut r = [0u32; 8];
        for i in 0..8 {
            r[i] = if a[i] < b[i] { u32::MAX } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_le(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
        let mut r = [0u32; 8];
        for i in 0..8 {
            r[i] = if a[i] <= b[i] { u32::MAX } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_gt(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
        let mut r = [0u32; 8];
        for i in 0..8 {
            r[i] = if a[i] > b[i] { u32::MAX } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_ge(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
        let mut r = [0u32; 8];
        for i in 0..8 {
            r[i] = if a[i] >= b[i] { u32::MAX } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn blend(mask: [u32; 8], if_true: [u32; 8], if_false: [u32; 8]) -> [u32; 8] {
        let mut r = [0u32; 8];
        for i in 0..8 {
            r[i] = if mask[i] != 0 {
                if_true[i]
            } else {
                if_false[i]
            };
        }
        r
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: [u32; 8]) -> u32 {
        a[0].wrapping_add(a[1].wrapping_add(a[2].wrapping_add(
            a[3].wrapping_add(a[4].wrapping_add(a[5].wrapping_add(a[6].wrapping_add(a[7])))),
        )))
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: [u32; 8]) -> [u32; 8] {
        [!a[0], !a[1], !a[2], !a[3], !a[4], !a[5], !a[6], !a[7]]
    }

    #[inline(always)]
    fn bitand(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
        [
            a[0] & b[0],
            a[1] & b[1],
            a[2] & b[2],
            a[3] & b[3],
            a[4] & b[4],
            a[5] & b[5],
            a[6] & b[6],
            a[7] & b[7],
        ]
    }

    #[inline(always)]
    fn bitor(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
        [
            a[0] | b[0],
            a[1] | b[1],
            a[2] | b[2],
            a[3] | b[3],
            a[4] | b[4],
            a[5] | b[5],
            a[6] | b[6],
            a[7] | b[7],
        ]
    }

    #[inline(always)]
    fn bitxor(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
        [
            a[0] ^ b[0],
            a[1] ^ b[1],
            a[2] ^ b[2],
            a[3] ^ b[3],
            a[4] ^ b[4],
            a[5] ^ b[5],
            a[6] ^ b[6],
            a[7] ^ b[7],
        ]
    }

    // ====== Shifts ======

    #[inline(always)]
    fn shl_const<const N: i32>(a: [u32; 8]) -> [u32; 8] {
        [
            a[0] << N,
            a[1] << N,
            a[2] << N,
            a[3] << N,
            a[4] << N,
            a[5] << N,
            a[6] << N,
            a[7] << N,
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [u32; 8]) -> [u32; 8] {
        [
            a[0] >> N,
            a[1] >> N,
            a[2] >> N,
            a[3] >> N,
            a[4] >> N,
            a[5] >> N,
            a[6] >> N,
            a[7] >> N,
        ]
    }

    // ====== Boolean ======

    #[inline(always)]
    fn all_true(a: [u32; 8]) -> bool {
        a[0] != 0
            && a[1] != 0
            && a[2] != 0
            && a[3] != 0
            && a[4] != 0
            && a[5] != 0
            && a[6] != 0
            && a[7] != 0
    }

    #[inline(always)]
    fn any_true(a: [u32; 8]) -> bool {
        a[0] != 0
            || a[1] != 0
            || a[2] != 0
            || a[3] != 0
            || a[4] != 0
            || a[5] != 0
            || a[6] != 0
            || a[7] != 0
    }

    #[inline(always)]
    fn bitmask(a: [u32; 8]) -> u32 {
        (a[0] >> 31)
            | ((a[1] >> 31) << 1)
            | ((a[2] >> 31) << 2)
            | ((a[3] >> 31) << 3)
            | ((a[4] >> 31) << 4)
            | ((a[5] >> 31) << 5)
            | ((a[6] >> 31) << 6)
            | ((a[7] >> 31) << 7)
    }
}

impl I64x2Backend for archmage::ScalarToken {
    type Repr = [i64; 2];

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: i64) -> [i64; 2] {
        [v; 2]
    }

    #[inline(always)]
    fn zero() -> [i64; 2] {
        [0i64; 2]
    }

    #[inline(always)]
    fn load(data: &[i64; 2]) -> [i64; 2] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [i64; 2]) -> [i64; 2] {
        arr
    }

    #[inline(always)]
    fn store(repr: [i64; 2], out: &mut [i64; 2]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [i64; 2]) -> [i64; 2] {
        repr
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
        [a[0].wrapping_add(b[0]), a[1].wrapping_add(b[1])]
    }

    #[inline(always)]
    fn sub(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
        [a[0].wrapping_sub(b[0]), a[1].wrapping_sub(b[1])]
    }

    #[inline(always)]
    fn neg(a: [i64; 2]) -> [i64; 2] {
        [a[0].wrapping_neg(), a[1].wrapping_neg()]
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
        [
            if a[0] < b[0] { a[0] } else { b[0] },
            if a[1] < b[1] { a[1] } else { b[1] },
        ]
    }

    #[inline(always)]
    fn max(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
        [
            if a[0] > b[0] { a[0] } else { b[0] },
            if a[1] > b[1] { a[1] } else { b[1] },
        ]
    }

    #[inline(always)]
    fn abs(a: [i64; 2]) -> [i64; 2] {
        [a[0].wrapping_abs(), a[1].wrapping_abs()]
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
        let mut r = [0i64; 2];
        for i in 0..2 {
            r[i] = if a[i] == b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_ne(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
        let mut r = [0i64; 2];
        for i in 0..2 {
            r[i] = if a[i] != b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_lt(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
        let mut r = [0i64; 2];
        for i in 0..2 {
            r[i] = if a[i] < b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_le(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
        let mut r = [0i64; 2];
        for i in 0..2 {
            r[i] = if a[i] <= b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_gt(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
        let mut r = [0i64; 2];
        for i in 0..2 {
            r[i] = if a[i] > b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_ge(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
        let mut r = [0i64; 2];
        for i in 0..2 {
            r[i] = if a[i] >= b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn blend(mask: [i64; 2], if_true: [i64; 2], if_false: [i64; 2]) -> [i64; 2] {
        let mut r = [0i64; 2];
        for i in 0..2 {
            r[i] = if mask[i] != 0 {
                if_true[i]
            } else {
                if_false[i]
            };
        }
        r
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: [i64; 2]) -> i64 {
        a[0].wrapping_add(a[1])
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: [i64; 2]) -> [i64; 2] {
        [!a[0], !a[1]]
    }

    #[inline(always)]
    fn bitand(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
        [a[0] & b[0], a[1] & b[1]]
    }

    #[inline(always)]
    fn bitor(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
        [a[0] | b[0], a[1] | b[1]]
    }

    #[inline(always)]
    fn bitxor(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
        [a[0] ^ b[0], a[1] ^ b[1]]
    }

    // ====== Shifts ======

    #[inline(always)]
    fn shl_const<const N: i32>(a: [i64; 2]) -> [i64; 2] {
        [a[0] << N, a[1] << N]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [i64; 2]) -> [i64; 2] {
        [a[0] >> N, a[1] >> N]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [i64; 2]) -> [i64; 2] {
        [((a[0] as u64) >> N) as i64, ((a[1] as u64) >> N) as i64]
    }

    // ====== Boolean ======

    #[inline(always)]
    fn all_true(a: [i64; 2]) -> bool {
        a[0] != 0 && a[1] != 0
    }

    #[inline(always)]
    fn any_true(a: [i64; 2]) -> bool {
        a[0] != 0 || a[1] != 0
    }

    #[inline(always)]
    fn bitmask(a: [i64; 2]) -> u32 {
        ((a[0] as u64) >> 63) as u32 | ((((a[1] as u64) >> 63) as u32) << 1)
    }
}

impl I64x4Backend for archmage::ScalarToken {
    type Repr = [i64; 4];

    // ====== Construction ======

    #[inline(always)]
    fn splat(v: i64) -> [i64; 4] {
        [v; 4]
    }

    #[inline(always)]
    fn zero() -> [i64; 4] {
        [0i64; 4]
    }

    #[inline(always)]
    fn load(data: &[i64; 4]) -> [i64; 4] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [i64; 4]) -> [i64; 4] {
        arr
    }

    #[inline(always)]
    fn store(repr: [i64; 4], out: &mut [i64; 4]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [i64; 4]) -> [i64; 4] {
        repr
    }

    // ====== Arithmetic ======

    #[inline(always)]
    fn add(a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
        [
            a[0].wrapping_add(b[0]),
            a[1].wrapping_add(b[1]),
            a[2].wrapping_add(b[2]),
            a[3].wrapping_add(b[3]),
        ]
    }

    #[inline(always)]
    fn sub(a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
        [
            a[0].wrapping_sub(b[0]),
            a[1].wrapping_sub(b[1]),
            a[2].wrapping_sub(b[2]),
            a[3].wrapping_sub(b[3]),
        ]
    }

    #[inline(always)]
    fn neg(a: [i64; 4]) -> [i64; 4] {
        [
            a[0].wrapping_neg(),
            a[1].wrapping_neg(),
            a[2].wrapping_neg(),
            a[3].wrapping_neg(),
        ]
    }

    // ====== Math ======

    #[inline(always)]
    fn min(a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
        [
            if a[0] < b[0] { a[0] } else { b[0] },
            if a[1] < b[1] { a[1] } else { b[1] },
            if a[2] < b[2] { a[2] } else { b[2] },
            if a[3] < b[3] { a[3] } else { b[3] },
        ]
    }

    #[inline(always)]
    fn max(a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
        [
            if a[0] > b[0] { a[0] } else { b[0] },
            if a[1] > b[1] { a[1] } else { b[1] },
            if a[2] > b[2] { a[2] } else { b[2] },
            if a[3] > b[3] { a[3] } else { b[3] },
        ]
    }

    #[inline(always)]
    fn abs(a: [i64; 4]) -> [i64; 4] {
        [
            a[0].wrapping_abs(),
            a[1].wrapping_abs(),
            a[2].wrapping_abs(),
            a[3].wrapping_abs(),
        ]
    }

    // ====== Comparisons ======

    #[inline(always)]
    fn simd_eq(a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
        let mut r = [0i64; 4];
        for i in 0..4 {
            r[i] = if a[i] == b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_ne(a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
        let mut r = [0i64; 4];
        for i in 0..4 {
            r[i] = if a[i] != b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_lt(a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
        let mut r = [0i64; 4];
        for i in 0..4 {
            r[i] = if a[i] < b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_le(a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
        let mut r = [0i64; 4];
        for i in 0..4 {
            r[i] = if a[i] <= b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_gt(a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
        let mut r = [0i64; 4];
        for i in 0..4 {
            r[i] = if a[i] > b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn simd_ge(a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
        let mut r = [0i64; 4];
        for i in 0..4 {
            r[i] = if a[i] >= b[i] { -1 } else { 0 };
        }
        r
    }

    #[inline(always)]
    fn blend(mask: [i64; 4], if_true: [i64; 4], if_false: [i64; 4]) -> [i64; 4] {
        let mut r = [0i64; 4];
        for i in 0..4 {
            r[i] = if mask[i] != 0 {
                if_true[i]
            } else {
                if_false[i]
            };
        }
        r
    }

    // ====== Reductions ======

    #[inline(always)]
    fn reduce_add(a: [i64; 4]) -> i64 {
        a[0].wrapping_add(a[1].wrapping_add(a[2].wrapping_add(a[3])))
    }

    // ====== Bitwise ======

    #[inline(always)]
    fn not(a: [i64; 4]) -> [i64; 4] {
        [!a[0], !a[1], !a[2], !a[3]]
    }

    #[inline(always)]
    fn bitand(a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
        [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]
    }

    #[inline(always)]
    fn bitor(a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
        [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]
    }

    #[inline(always)]
    fn bitxor(a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
        [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
    }

    // ====== Shifts ======

    #[inline(always)]
    fn shl_const<const N: i32>(a: [i64; 4]) -> [i64; 4] {
        [a[0] << N, a[1] << N, a[2] << N, a[3] << N]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [i64; 4]) -> [i64; 4] {
        [a[0] >> N, a[1] >> N, a[2] >> N, a[3] >> N]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [i64; 4]) -> [i64; 4] {
        [
            ((a[0] as u64) >> N) as i64,
            ((a[1] as u64) >> N) as i64,
            ((a[2] as u64) >> N) as i64,
            ((a[3] as u64) >> N) as i64,
        ]
    }

    // ====== Boolean ======

    #[inline(always)]
    fn all_true(a: [i64; 4]) -> bool {
        a[0] != 0 && a[1] != 0 && a[2] != 0 && a[3] != 0
    }

    #[inline(always)]
    fn any_true(a: [i64; 4]) -> bool {
        a[0] != 0 || a[1] != 0 || a[2] != 0 || a[3] != 0
    }

    #[inline(always)]
    fn bitmask(a: [i64; 4]) -> u32 {
        ((a[0] as u64) >> 63) as u32
            | ((((a[1] as u64) >> 63) as u32) << 1)
            | ((((a[2] as u64) >> 63) as u32) << 2)
            | ((((a[3] as u64) >> 63) as u32) << 3)
    }
}

impl F32x4Convert for archmage::ScalarToken {
    #[inline(always)]
    fn bitcast_f32_to_i32(a: [f32; 4]) -> [i32; 4] {
        [
            a[0].to_bits() as i32,
            a[1].to_bits() as i32,
            a[2].to_bits() as i32,
            a[3].to_bits() as i32,
        ]
    }

    #[inline(always)]
    fn bitcast_i32_to_f32(a: [i32; 4]) -> [f32; 4] {
        [
            f32::from_bits(a[0] as u32),
            f32::from_bits(a[1] as u32),
            f32::from_bits(a[2] as u32),
            f32::from_bits(a[3] as u32),
        ]
    }

    #[inline(always)]
    fn convert_f32_to_i32(a: [f32; 4]) -> [i32; 4] {
        [a[0] as i32, a[1] as i32, a[2] as i32, a[3] as i32]
    }

    #[inline(always)]
    fn convert_f32_to_i32_round(a: [f32; 4]) -> [i32; 4] {
        [
            f32_round(a[0]) as i32,
            f32_round(a[1]) as i32,
            f32_round(a[2]) as i32,
            f32_round(a[3]) as i32,
        ]
    }

    #[inline(always)]
    fn convert_i32_to_f32(a: [i32; 4]) -> [f32; 4] {
        [a[0] as f32, a[1] as f32, a[2] as f32, a[3] as f32]
    }
}

impl F32x8Convert for archmage::ScalarToken {
    #[inline(always)]
    fn bitcast_f32_to_i32(a: [f32; 8]) -> [i32; 8] {
        [
            a[0].to_bits() as i32,
            a[1].to_bits() as i32,
            a[2].to_bits() as i32,
            a[3].to_bits() as i32,
            a[4].to_bits() as i32,
            a[5].to_bits() as i32,
            a[6].to_bits() as i32,
            a[7].to_bits() as i32,
        ]
    }

    #[inline(always)]
    fn bitcast_i32_to_f32(a: [i32; 8]) -> [f32; 8] {
        [
            f32::from_bits(a[0] as u32),
            f32::from_bits(a[1] as u32),
            f32::from_bits(a[2] as u32),
            f32::from_bits(a[3] as u32),
            f32::from_bits(a[4] as u32),
            f32::from_bits(a[5] as u32),
            f32::from_bits(a[6] as u32),
            f32::from_bits(a[7] as u32),
        ]
    }

    #[inline(always)]
    fn convert_f32_to_i32(a: [f32; 8]) -> [i32; 8] {
        [
            a[0] as i32,
            a[1] as i32,
            a[2] as i32,
            a[3] as i32,
            a[4] as i32,
            a[5] as i32,
            a[6] as i32,
            a[7] as i32,
        ]
    }

    #[inline(always)]
    fn convert_f32_to_i32_round(a: [f32; 8]) -> [i32; 8] {
        [
            f32_round(a[0]) as i32,
            f32_round(a[1]) as i32,
            f32_round(a[2]) as i32,
            f32_round(a[3]) as i32,
            f32_round(a[4]) as i32,
            f32_round(a[5]) as i32,
            f32_round(a[6]) as i32,
            f32_round(a[7]) as i32,
        ]
    }

    #[inline(always)]
    fn convert_i32_to_f32(a: [i32; 8]) -> [f32; 8] {
        [
            a[0] as f32,
            a[1] as f32,
            a[2] as f32,
            a[3] as f32,
            a[4] as f32,
            a[5] as f32,
            a[6] as f32,
            a[7] as f32,
        ]
    }
}

impl U32x4Bitcast for archmage::ScalarToken {
    #[inline(always)]
    fn bitcast_u32_to_i32(a: [u32; 4]) -> [i32; 4] {
        [a[0] as i32, a[1] as i32, a[2] as i32, a[3] as i32]
    }

    #[inline(always)]
    fn bitcast_i32_to_u32(a: [i32; 4]) -> [u32; 4] {
        [a[0] as u32, a[1] as u32, a[2] as u32, a[3] as u32]
    }
}

impl U32x8Bitcast for archmage::ScalarToken {
    #[inline(always)]
    fn bitcast_u32_to_i32(a: [u32; 8]) -> [i32; 8] {
        [
            a[0] as i32,
            a[1] as i32,
            a[2] as i32,
            a[3] as i32,
            a[4] as i32,
            a[5] as i32,
            a[6] as i32,
            a[7] as i32,
        ]
    }

    #[inline(always)]
    fn bitcast_i32_to_u32(a: [i32; 8]) -> [u32; 8] {
        [
            a[0] as u32,
            a[1] as u32,
            a[2] as u32,
            a[3] as u32,
            a[4] as u32,
            a[5] as u32,
            a[6] as u32,
            a[7] as u32,
        ]
    }
}

impl I64x2Bitcast for archmage::ScalarToken {
    #[inline(always)]
    fn bitcast_i64_to_f64(a: [i64; 2]) -> [f64; 2] {
        [f64::from_bits(a[0] as u64), f64::from_bits(a[1] as u64)]
    }

    #[inline(always)]
    fn bitcast_f64_to_i64(a: [f64; 2]) -> [i64; 2] {
        [a[0].to_bits() as i64, a[1].to_bits() as i64]
    }
}

impl I64x4Bitcast for archmage::ScalarToken {
    #[inline(always)]
    fn bitcast_i64_to_f64(a: [i64; 4]) -> [f64; 4] {
        [
            f64::from_bits(a[0] as u64),
            f64::from_bits(a[1] as u64),
            f64::from_bits(a[2] as u64),
            f64::from_bits(a[3] as u64),
        ]
    }

    #[inline(always)]
    fn bitcast_f64_to_i64(a: [f64; 4]) -> [i64; 4] {
        [
            a[0].to_bits() as i64,
            a[1].to_bits() as i64,
            a[2].to_bits() as i64,
            a[3].to_bits() as i64,
        ]
    }
}

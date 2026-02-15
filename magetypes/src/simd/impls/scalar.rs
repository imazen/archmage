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

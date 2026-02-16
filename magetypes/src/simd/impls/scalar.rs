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

impl I8x16Backend for archmage::ScalarToken {
    type Repr = [i8; 16];

    #[inline(always)]
    fn splat(v: i8) -> [i8; 16] {
        [v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v]
    }

    #[inline(always)]
    fn zero() -> [i8; 16] {
        [0i8; 16]
    }

    #[inline(always)]
    fn load(data: &[i8; 16]) -> [i8; 16] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [i8; 16]) -> [i8; 16] {
        arr
    }

    #[inline(always)]
    fn store(repr: [i8; 16], out: &mut [i8; 16]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [i8; 16]) -> [i8; 16] {
        repr
    }

    #[inline(always)]
    fn add(a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
        [
            a[0].wrapping_add(b[0]),
            a[1].wrapping_add(b[1]),
            a[2].wrapping_add(b[2]),
            a[3].wrapping_add(b[3]),
            a[4].wrapping_add(b[4]),
            a[5].wrapping_add(b[5]),
            a[6].wrapping_add(b[6]),
            a[7].wrapping_add(b[7]),
            a[8].wrapping_add(b[8]),
            a[9].wrapping_add(b[9]),
            a[10].wrapping_add(b[10]),
            a[11].wrapping_add(b[11]),
            a[12].wrapping_add(b[12]),
            a[13].wrapping_add(b[13]),
            a[14].wrapping_add(b[14]),
            a[15].wrapping_add(b[15]),
        ]
    }

    #[inline(always)]
    fn sub(a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
        [
            a[0].wrapping_sub(b[0]),
            a[1].wrapping_sub(b[1]),
            a[2].wrapping_sub(b[2]),
            a[3].wrapping_sub(b[3]),
            a[4].wrapping_sub(b[4]),
            a[5].wrapping_sub(b[5]),
            a[6].wrapping_sub(b[6]),
            a[7].wrapping_sub(b[7]),
            a[8].wrapping_sub(b[8]),
            a[9].wrapping_sub(b[9]),
            a[10].wrapping_sub(b[10]),
            a[11].wrapping_sub(b[11]),
            a[12].wrapping_sub(b[12]),
            a[13].wrapping_sub(b[13]),
            a[14].wrapping_sub(b[14]),
            a[15].wrapping_sub(b[15]),
        ]
    }

    #[inline(always)]
    fn neg(a: [i8; 16]) -> [i8; 16] {
        [
            a[0].wrapping_neg(),
            a[1].wrapping_neg(),
            a[2].wrapping_neg(),
            a[3].wrapping_neg(),
            a[4].wrapping_neg(),
            a[5].wrapping_neg(),
            a[6].wrapping_neg(),
            a[7].wrapping_neg(),
            a[8].wrapping_neg(),
            a[9].wrapping_neg(),
            a[10].wrapping_neg(),
            a[11].wrapping_neg(),
            a[12].wrapping_neg(),
            a[13].wrapping_neg(),
            a[14].wrapping_neg(),
            a[15].wrapping_neg(),
        ]
    }

    #[inline(always)]
    fn min(a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
        [
            if a[0] < b[0] { a[0] } else { b[0] },
            if a[1] < b[1] { a[1] } else { b[1] },
            if a[2] < b[2] { a[2] } else { b[2] },
            if a[3] < b[3] { a[3] } else { b[3] },
            if a[4] < b[4] { a[4] } else { b[4] },
            if a[5] < b[5] { a[5] } else { b[5] },
            if a[6] < b[6] { a[6] } else { b[6] },
            if a[7] < b[7] { a[7] } else { b[7] },
            if a[8] < b[8] { a[8] } else { b[8] },
            if a[9] < b[9] { a[9] } else { b[9] },
            if a[10] < b[10] { a[10] } else { b[10] },
            if a[11] < b[11] { a[11] } else { b[11] },
            if a[12] < b[12] { a[12] } else { b[12] },
            if a[13] < b[13] { a[13] } else { b[13] },
            if a[14] < b[14] { a[14] } else { b[14] },
            if a[15] < b[15] { a[15] } else { b[15] },
        ]
    }

    #[inline(always)]
    fn max(a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
        [
            if a[0] > b[0] { a[0] } else { b[0] },
            if a[1] > b[1] { a[1] } else { b[1] },
            if a[2] > b[2] { a[2] } else { b[2] },
            if a[3] > b[3] { a[3] } else { b[3] },
            if a[4] > b[4] { a[4] } else { b[4] },
            if a[5] > b[5] { a[5] } else { b[5] },
            if a[6] > b[6] { a[6] } else { b[6] },
            if a[7] > b[7] { a[7] } else { b[7] },
            if a[8] > b[8] { a[8] } else { b[8] },
            if a[9] > b[9] { a[9] } else { b[9] },
            if a[10] > b[10] { a[10] } else { b[10] },
            if a[11] > b[11] { a[11] } else { b[11] },
            if a[12] > b[12] { a[12] } else { b[12] },
            if a[13] > b[13] { a[13] } else { b[13] },
            if a[14] > b[14] { a[14] } else { b[14] },
            if a[15] > b[15] { a[15] } else { b[15] },
        ]
    }

    #[inline(always)]
    fn abs(a: [i8; 16]) -> [i8; 16] {
        [
            a[0].wrapping_abs(),
            a[1].wrapping_abs(),
            a[2].wrapping_abs(),
            a[3].wrapping_abs(),
            a[4].wrapping_abs(),
            a[5].wrapping_abs(),
            a[6].wrapping_abs(),
            a[7].wrapping_abs(),
            a[8].wrapping_abs(),
            a[9].wrapping_abs(),
            a[10].wrapping_abs(),
            a[11].wrapping_abs(),
            a[12].wrapping_abs(),
            a[13].wrapping_abs(),
            a[14].wrapping_abs(),
            a[15].wrapping_abs(),
        ]
    }

    #[inline(always)]
    fn simd_eq(a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
        [
            if a[0] == b[0] { -1 } else { 0 },
            if a[1] == b[1] { -1 } else { 0 },
            if a[2] == b[2] { -1 } else { 0 },
            if a[3] == b[3] { -1 } else { 0 },
            if a[4] == b[4] { -1 } else { 0 },
            if a[5] == b[5] { -1 } else { 0 },
            if a[6] == b[6] { -1 } else { 0 },
            if a[7] == b[7] { -1 } else { 0 },
            if a[8] == b[8] { -1 } else { 0 },
            if a[9] == b[9] { -1 } else { 0 },
            if a[10] == b[10] { -1 } else { 0 },
            if a[11] == b[11] { -1 } else { 0 },
            if a[12] == b[12] { -1 } else { 0 },
            if a[13] == b[13] { -1 } else { 0 },
            if a[14] == b[14] { -1 } else { 0 },
            if a[15] == b[15] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ne(a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
        [
            if a[0] != b[0] { -1 } else { 0 },
            if a[1] != b[1] { -1 } else { 0 },
            if a[2] != b[2] { -1 } else { 0 },
            if a[3] != b[3] { -1 } else { 0 },
            if a[4] != b[4] { -1 } else { 0 },
            if a[5] != b[5] { -1 } else { 0 },
            if a[6] != b[6] { -1 } else { 0 },
            if a[7] != b[7] { -1 } else { 0 },
            if a[8] != b[8] { -1 } else { 0 },
            if a[9] != b[9] { -1 } else { 0 },
            if a[10] != b[10] { -1 } else { 0 },
            if a[11] != b[11] { -1 } else { 0 },
            if a[12] != b[12] { -1 } else { 0 },
            if a[13] != b[13] { -1 } else { 0 },
            if a[14] != b[14] { -1 } else { 0 },
            if a[15] != b[15] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_lt(a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
        [
            if a[0] < b[0] { -1 } else { 0 },
            if a[1] < b[1] { -1 } else { 0 },
            if a[2] < b[2] { -1 } else { 0 },
            if a[3] < b[3] { -1 } else { 0 },
            if a[4] < b[4] { -1 } else { 0 },
            if a[5] < b[5] { -1 } else { 0 },
            if a[6] < b[6] { -1 } else { 0 },
            if a[7] < b[7] { -1 } else { 0 },
            if a[8] < b[8] { -1 } else { 0 },
            if a[9] < b[9] { -1 } else { 0 },
            if a[10] < b[10] { -1 } else { 0 },
            if a[11] < b[11] { -1 } else { 0 },
            if a[12] < b[12] { -1 } else { 0 },
            if a[13] < b[13] { -1 } else { 0 },
            if a[14] < b[14] { -1 } else { 0 },
            if a[15] < b[15] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_le(a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
        [
            if a[0] <= b[0] { -1 } else { 0 },
            if a[1] <= b[1] { -1 } else { 0 },
            if a[2] <= b[2] { -1 } else { 0 },
            if a[3] <= b[3] { -1 } else { 0 },
            if a[4] <= b[4] { -1 } else { 0 },
            if a[5] <= b[5] { -1 } else { 0 },
            if a[6] <= b[6] { -1 } else { 0 },
            if a[7] <= b[7] { -1 } else { 0 },
            if a[8] <= b[8] { -1 } else { 0 },
            if a[9] <= b[9] { -1 } else { 0 },
            if a[10] <= b[10] { -1 } else { 0 },
            if a[11] <= b[11] { -1 } else { 0 },
            if a[12] <= b[12] { -1 } else { 0 },
            if a[13] <= b[13] { -1 } else { 0 },
            if a[14] <= b[14] { -1 } else { 0 },
            if a[15] <= b[15] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_gt(a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
        [
            if a[0] > b[0] { -1 } else { 0 },
            if a[1] > b[1] { -1 } else { 0 },
            if a[2] > b[2] { -1 } else { 0 },
            if a[3] > b[3] { -1 } else { 0 },
            if a[4] > b[4] { -1 } else { 0 },
            if a[5] > b[5] { -1 } else { 0 },
            if a[6] > b[6] { -1 } else { 0 },
            if a[7] > b[7] { -1 } else { 0 },
            if a[8] > b[8] { -1 } else { 0 },
            if a[9] > b[9] { -1 } else { 0 },
            if a[10] > b[10] { -1 } else { 0 },
            if a[11] > b[11] { -1 } else { 0 },
            if a[12] > b[12] { -1 } else { 0 },
            if a[13] > b[13] { -1 } else { 0 },
            if a[14] > b[14] { -1 } else { 0 },
            if a[15] > b[15] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ge(a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
        [
            if a[0] >= b[0] { -1 } else { 0 },
            if a[1] >= b[1] { -1 } else { 0 },
            if a[2] >= b[2] { -1 } else { 0 },
            if a[3] >= b[3] { -1 } else { 0 },
            if a[4] >= b[4] { -1 } else { 0 },
            if a[5] >= b[5] { -1 } else { 0 },
            if a[6] >= b[6] { -1 } else { 0 },
            if a[7] >= b[7] { -1 } else { 0 },
            if a[8] >= b[8] { -1 } else { 0 },
            if a[9] >= b[9] { -1 } else { 0 },
            if a[10] >= b[10] { -1 } else { 0 },
            if a[11] >= b[11] { -1 } else { 0 },
            if a[12] >= b[12] { -1 } else { 0 },
            if a[13] >= b[13] { -1 } else { 0 },
            if a[14] >= b[14] { -1 } else { 0 },
            if a[15] >= b[15] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn blend(mask: [i8; 16], if_true: [i8; 16], if_false: [i8; 16]) -> [i8; 16] {
        [
            if mask[0] != 0 {
                if_true[0]
            } else {
                if_false[0]
            },
            if mask[1] != 0 {
                if_true[1]
            } else {
                if_false[1]
            },
            if mask[2] != 0 {
                if_true[2]
            } else {
                if_false[2]
            },
            if mask[3] != 0 {
                if_true[3]
            } else {
                if_false[3]
            },
            if mask[4] != 0 {
                if_true[4]
            } else {
                if_false[4]
            },
            if mask[5] != 0 {
                if_true[5]
            } else {
                if_false[5]
            },
            if mask[6] != 0 {
                if_true[6]
            } else {
                if_false[6]
            },
            if mask[7] != 0 {
                if_true[7]
            } else {
                if_false[7]
            },
            if mask[8] != 0 {
                if_true[8]
            } else {
                if_false[8]
            },
            if mask[9] != 0 {
                if_true[9]
            } else {
                if_false[9]
            },
            if mask[10] != 0 {
                if_true[10]
            } else {
                if_false[10]
            },
            if mask[11] != 0 {
                if_true[11]
            } else {
                if_false[11]
            },
            if mask[12] != 0 {
                if_true[12]
            } else {
                if_false[12]
            },
            if mask[13] != 0 {
                if_true[13]
            } else {
                if_false[13]
            },
            if mask[14] != 0 {
                if_true[14]
            } else {
                if_false[14]
            },
            if mask[15] != 0 {
                if_true[15]
            } else {
                if_false[15]
            },
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [i8; 16]) -> i8 {
        a.iter().copied().fold(0i8, i8::wrapping_add)
    }

    #[inline(always)]
    fn not(a: [i8; 16]) -> [i8; 16] {
        [
            !a[0], !a[1], !a[2], !a[3], !a[4], !a[5], !a[6], !a[7], !a[8], !a[9], !a[10], !a[11],
            !a[12], !a[13], !a[14], !a[15],
        ]
    }

    #[inline(always)]
    fn bitand(a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
        [
            a[0] & b[0],
            a[1] & b[1],
            a[2] & b[2],
            a[3] & b[3],
            a[4] & b[4],
            a[5] & b[5],
            a[6] & b[6],
            a[7] & b[7],
            a[8] & b[8],
            a[9] & b[9],
            a[10] & b[10],
            a[11] & b[11],
            a[12] & b[12],
            a[13] & b[13],
            a[14] & b[14],
            a[15] & b[15],
        ]
    }

    #[inline(always)]
    fn bitor(a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
        [
            a[0] | b[0],
            a[1] | b[1],
            a[2] | b[2],
            a[3] | b[3],
            a[4] | b[4],
            a[5] | b[5],
            a[6] | b[6],
            a[7] | b[7],
            a[8] | b[8],
            a[9] | b[9],
            a[10] | b[10],
            a[11] | b[11],
            a[12] | b[12],
            a[13] | b[13],
            a[14] | b[14],
            a[15] | b[15],
        ]
    }

    #[inline(always)]
    fn bitxor(a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
        [
            a[0] ^ b[0],
            a[1] ^ b[1],
            a[2] ^ b[2],
            a[3] ^ b[3],
            a[4] ^ b[4],
            a[5] ^ b[5],
            a[6] ^ b[6],
            a[7] ^ b[7],
            a[8] ^ b[8],
            a[9] ^ b[9],
            a[10] ^ b[10],
            a[11] ^ b[11],
            a[12] ^ b[12],
            a[13] ^ b[13],
            a[14] ^ b[14],
            a[15] ^ b[15],
        ]
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [i8; 16]) -> [i8; 16] {
        [
            a[0].wrapping_shl(N as u32),
            a[1].wrapping_shl(N as u32),
            a[2].wrapping_shl(N as u32),
            a[3].wrapping_shl(N as u32),
            a[4].wrapping_shl(N as u32),
            a[5].wrapping_shl(N as u32),
            a[6].wrapping_shl(N as u32),
            a[7].wrapping_shl(N as u32),
            a[8].wrapping_shl(N as u32),
            a[9].wrapping_shl(N as u32),
            a[10].wrapping_shl(N as u32),
            a[11].wrapping_shl(N as u32),
            a[12].wrapping_shl(N as u32),
            a[13].wrapping_shl(N as u32),
            a[14].wrapping_shl(N as u32),
            a[15].wrapping_shl(N as u32),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [i8; 16]) -> [i8; 16] {
        [
            (a[0] as u8).wrapping_shr(N as u32) as i8,
            (a[1] as u8).wrapping_shr(N as u32) as i8,
            (a[2] as u8).wrapping_shr(N as u32) as i8,
            (a[3] as u8).wrapping_shr(N as u32) as i8,
            (a[4] as u8).wrapping_shr(N as u32) as i8,
            (a[5] as u8).wrapping_shr(N as u32) as i8,
            (a[6] as u8).wrapping_shr(N as u32) as i8,
            (a[7] as u8).wrapping_shr(N as u32) as i8,
            (a[8] as u8).wrapping_shr(N as u32) as i8,
            (a[9] as u8).wrapping_shr(N as u32) as i8,
            (a[10] as u8).wrapping_shr(N as u32) as i8,
            (a[11] as u8).wrapping_shr(N as u32) as i8,
            (a[12] as u8).wrapping_shr(N as u32) as i8,
            (a[13] as u8).wrapping_shr(N as u32) as i8,
            (a[14] as u8).wrapping_shr(N as u32) as i8,
            (a[15] as u8).wrapping_shr(N as u32) as i8,
        ]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [i8; 16]) -> [i8; 16] {
        [
            a[0].wrapping_shr(N as u32),
            a[1].wrapping_shr(N as u32),
            a[2].wrapping_shr(N as u32),
            a[3].wrapping_shr(N as u32),
            a[4].wrapping_shr(N as u32),
            a[5].wrapping_shr(N as u32),
            a[6].wrapping_shr(N as u32),
            a[7].wrapping_shr(N as u32),
            a[8].wrapping_shr(N as u32),
            a[9].wrapping_shr(N as u32),
            a[10].wrapping_shr(N as u32),
            a[11].wrapping_shr(N as u32),
            a[12].wrapping_shr(N as u32),
            a[13].wrapping_shr(N as u32),
            a[14].wrapping_shr(N as u32),
            a[15].wrapping_shr(N as u32),
        ]
    }

    #[inline(always)]
    fn all_true(a: [i8; 16]) -> bool {
        a[0] != 0
            && a[1] != 0
            && a[2] != 0
            && a[3] != 0
            && a[4] != 0
            && a[5] != 0
            && a[6] != 0
            && a[7] != 0
            && a[8] != 0
            && a[9] != 0
            && a[10] != 0
            && a[11] != 0
            && a[12] != 0
            && a[13] != 0
            && a[14] != 0
            && a[15] != 0
    }

    #[inline(always)]
    fn any_true(a: [i8; 16]) -> bool {
        a[0] != 0
            || a[1] != 0
            || a[2] != 0
            || a[3] != 0
            || a[4] != 0
            || a[5] != 0
            || a[6] != 0
            || a[7] != 0
            || a[8] != 0
            || a[9] != 0
            || a[10] != 0
            || a[11] != 0
            || a[12] != 0
            || a[13] != 0
            || a[14] != 0
            || a[15] != 0
    }

    #[inline(always)]
    fn bitmask(a: [i8; 16]) -> u32 {
        ((a[0] >> 7) as u32 & 1) << 0
            | ((a[1] >> 7) as u32 & 1) << 1
            | ((a[2] >> 7) as u32 & 1) << 2
            | ((a[3] >> 7) as u32 & 1) << 3
            | ((a[4] >> 7) as u32 & 1) << 4
            | ((a[5] >> 7) as u32 & 1) << 5
            | ((a[6] >> 7) as u32 & 1) << 6
            | ((a[7] >> 7) as u32 & 1) << 7
            | ((a[8] >> 7) as u32 & 1) << 8
            | ((a[9] >> 7) as u32 & 1) << 9
            | ((a[10] >> 7) as u32 & 1) << 10
            | ((a[11] >> 7) as u32 & 1) << 11
            | ((a[12] >> 7) as u32 & 1) << 12
            | ((a[13] >> 7) as u32 & 1) << 13
            | ((a[14] >> 7) as u32 & 1) << 14
            | ((a[15] >> 7) as u32 & 1) << 15
    }
}

impl I8x32Backend for archmage::ScalarToken {
    type Repr = [i8; 32];

    #[inline(always)]
    fn splat(v: i8) -> [i8; 32] {
        [
            v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v,
            v, v, v,
        ]
    }

    #[inline(always)]
    fn zero() -> [i8; 32] {
        [0i8; 32]
    }

    #[inline(always)]
    fn load(data: &[i8; 32]) -> [i8; 32] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [i8; 32]) -> [i8; 32] {
        arr
    }

    #[inline(always)]
    fn store(repr: [i8; 32], out: &mut [i8; 32]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [i8; 32]) -> [i8; 32] {
        repr
    }

    #[inline(always)]
    fn add(a: [i8; 32], b: [i8; 32]) -> [i8; 32] {
        [
            a[0].wrapping_add(b[0]),
            a[1].wrapping_add(b[1]),
            a[2].wrapping_add(b[2]),
            a[3].wrapping_add(b[3]),
            a[4].wrapping_add(b[4]),
            a[5].wrapping_add(b[5]),
            a[6].wrapping_add(b[6]),
            a[7].wrapping_add(b[7]),
            a[8].wrapping_add(b[8]),
            a[9].wrapping_add(b[9]),
            a[10].wrapping_add(b[10]),
            a[11].wrapping_add(b[11]),
            a[12].wrapping_add(b[12]),
            a[13].wrapping_add(b[13]),
            a[14].wrapping_add(b[14]),
            a[15].wrapping_add(b[15]),
            a[16].wrapping_add(b[16]),
            a[17].wrapping_add(b[17]),
            a[18].wrapping_add(b[18]),
            a[19].wrapping_add(b[19]),
            a[20].wrapping_add(b[20]),
            a[21].wrapping_add(b[21]),
            a[22].wrapping_add(b[22]),
            a[23].wrapping_add(b[23]),
            a[24].wrapping_add(b[24]),
            a[25].wrapping_add(b[25]),
            a[26].wrapping_add(b[26]),
            a[27].wrapping_add(b[27]),
            a[28].wrapping_add(b[28]),
            a[29].wrapping_add(b[29]),
            a[30].wrapping_add(b[30]),
            a[31].wrapping_add(b[31]),
        ]
    }

    #[inline(always)]
    fn sub(a: [i8; 32], b: [i8; 32]) -> [i8; 32] {
        [
            a[0].wrapping_sub(b[0]),
            a[1].wrapping_sub(b[1]),
            a[2].wrapping_sub(b[2]),
            a[3].wrapping_sub(b[3]),
            a[4].wrapping_sub(b[4]),
            a[5].wrapping_sub(b[5]),
            a[6].wrapping_sub(b[6]),
            a[7].wrapping_sub(b[7]),
            a[8].wrapping_sub(b[8]),
            a[9].wrapping_sub(b[9]),
            a[10].wrapping_sub(b[10]),
            a[11].wrapping_sub(b[11]),
            a[12].wrapping_sub(b[12]),
            a[13].wrapping_sub(b[13]),
            a[14].wrapping_sub(b[14]),
            a[15].wrapping_sub(b[15]),
            a[16].wrapping_sub(b[16]),
            a[17].wrapping_sub(b[17]),
            a[18].wrapping_sub(b[18]),
            a[19].wrapping_sub(b[19]),
            a[20].wrapping_sub(b[20]),
            a[21].wrapping_sub(b[21]),
            a[22].wrapping_sub(b[22]),
            a[23].wrapping_sub(b[23]),
            a[24].wrapping_sub(b[24]),
            a[25].wrapping_sub(b[25]),
            a[26].wrapping_sub(b[26]),
            a[27].wrapping_sub(b[27]),
            a[28].wrapping_sub(b[28]),
            a[29].wrapping_sub(b[29]),
            a[30].wrapping_sub(b[30]),
            a[31].wrapping_sub(b[31]),
        ]
    }

    #[inline(always)]
    fn neg(a: [i8; 32]) -> [i8; 32] {
        [
            a[0].wrapping_neg(),
            a[1].wrapping_neg(),
            a[2].wrapping_neg(),
            a[3].wrapping_neg(),
            a[4].wrapping_neg(),
            a[5].wrapping_neg(),
            a[6].wrapping_neg(),
            a[7].wrapping_neg(),
            a[8].wrapping_neg(),
            a[9].wrapping_neg(),
            a[10].wrapping_neg(),
            a[11].wrapping_neg(),
            a[12].wrapping_neg(),
            a[13].wrapping_neg(),
            a[14].wrapping_neg(),
            a[15].wrapping_neg(),
            a[16].wrapping_neg(),
            a[17].wrapping_neg(),
            a[18].wrapping_neg(),
            a[19].wrapping_neg(),
            a[20].wrapping_neg(),
            a[21].wrapping_neg(),
            a[22].wrapping_neg(),
            a[23].wrapping_neg(),
            a[24].wrapping_neg(),
            a[25].wrapping_neg(),
            a[26].wrapping_neg(),
            a[27].wrapping_neg(),
            a[28].wrapping_neg(),
            a[29].wrapping_neg(),
            a[30].wrapping_neg(),
            a[31].wrapping_neg(),
        ]
    }

    #[inline(always)]
    fn min(a: [i8; 32], b: [i8; 32]) -> [i8; 32] {
        [
            if a[0] < b[0] { a[0] } else { b[0] },
            if a[1] < b[1] { a[1] } else { b[1] },
            if a[2] < b[2] { a[2] } else { b[2] },
            if a[3] < b[3] { a[3] } else { b[3] },
            if a[4] < b[4] { a[4] } else { b[4] },
            if a[5] < b[5] { a[5] } else { b[5] },
            if a[6] < b[6] { a[6] } else { b[6] },
            if a[7] < b[7] { a[7] } else { b[7] },
            if a[8] < b[8] { a[8] } else { b[8] },
            if a[9] < b[9] { a[9] } else { b[9] },
            if a[10] < b[10] { a[10] } else { b[10] },
            if a[11] < b[11] { a[11] } else { b[11] },
            if a[12] < b[12] { a[12] } else { b[12] },
            if a[13] < b[13] { a[13] } else { b[13] },
            if a[14] < b[14] { a[14] } else { b[14] },
            if a[15] < b[15] { a[15] } else { b[15] },
            if a[16] < b[16] { a[16] } else { b[16] },
            if a[17] < b[17] { a[17] } else { b[17] },
            if a[18] < b[18] { a[18] } else { b[18] },
            if a[19] < b[19] { a[19] } else { b[19] },
            if a[20] < b[20] { a[20] } else { b[20] },
            if a[21] < b[21] { a[21] } else { b[21] },
            if a[22] < b[22] { a[22] } else { b[22] },
            if a[23] < b[23] { a[23] } else { b[23] },
            if a[24] < b[24] { a[24] } else { b[24] },
            if a[25] < b[25] { a[25] } else { b[25] },
            if a[26] < b[26] { a[26] } else { b[26] },
            if a[27] < b[27] { a[27] } else { b[27] },
            if a[28] < b[28] { a[28] } else { b[28] },
            if a[29] < b[29] { a[29] } else { b[29] },
            if a[30] < b[30] { a[30] } else { b[30] },
            if a[31] < b[31] { a[31] } else { b[31] },
        ]
    }

    #[inline(always)]
    fn max(a: [i8; 32], b: [i8; 32]) -> [i8; 32] {
        [
            if a[0] > b[0] { a[0] } else { b[0] },
            if a[1] > b[1] { a[1] } else { b[1] },
            if a[2] > b[2] { a[2] } else { b[2] },
            if a[3] > b[3] { a[3] } else { b[3] },
            if a[4] > b[4] { a[4] } else { b[4] },
            if a[5] > b[5] { a[5] } else { b[5] },
            if a[6] > b[6] { a[6] } else { b[6] },
            if a[7] > b[7] { a[7] } else { b[7] },
            if a[8] > b[8] { a[8] } else { b[8] },
            if a[9] > b[9] { a[9] } else { b[9] },
            if a[10] > b[10] { a[10] } else { b[10] },
            if a[11] > b[11] { a[11] } else { b[11] },
            if a[12] > b[12] { a[12] } else { b[12] },
            if a[13] > b[13] { a[13] } else { b[13] },
            if a[14] > b[14] { a[14] } else { b[14] },
            if a[15] > b[15] { a[15] } else { b[15] },
            if a[16] > b[16] { a[16] } else { b[16] },
            if a[17] > b[17] { a[17] } else { b[17] },
            if a[18] > b[18] { a[18] } else { b[18] },
            if a[19] > b[19] { a[19] } else { b[19] },
            if a[20] > b[20] { a[20] } else { b[20] },
            if a[21] > b[21] { a[21] } else { b[21] },
            if a[22] > b[22] { a[22] } else { b[22] },
            if a[23] > b[23] { a[23] } else { b[23] },
            if a[24] > b[24] { a[24] } else { b[24] },
            if a[25] > b[25] { a[25] } else { b[25] },
            if a[26] > b[26] { a[26] } else { b[26] },
            if a[27] > b[27] { a[27] } else { b[27] },
            if a[28] > b[28] { a[28] } else { b[28] },
            if a[29] > b[29] { a[29] } else { b[29] },
            if a[30] > b[30] { a[30] } else { b[30] },
            if a[31] > b[31] { a[31] } else { b[31] },
        ]
    }

    #[inline(always)]
    fn abs(a: [i8; 32]) -> [i8; 32] {
        [
            a[0].wrapping_abs(),
            a[1].wrapping_abs(),
            a[2].wrapping_abs(),
            a[3].wrapping_abs(),
            a[4].wrapping_abs(),
            a[5].wrapping_abs(),
            a[6].wrapping_abs(),
            a[7].wrapping_abs(),
            a[8].wrapping_abs(),
            a[9].wrapping_abs(),
            a[10].wrapping_abs(),
            a[11].wrapping_abs(),
            a[12].wrapping_abs(),
            a[13].wrapping_abs(),
            a[14].wrapping_abs(),
            a[15].wrapping_abs(),
            a[16].wrapping_abs(),
            a[17].wrapping_abs(),
            a[18].wrapping_abs(),
            a[19].wrapping_abs(),
            a[20].wrapping_abs(),
            a[21].wrapping_abs(),
            a[22].wrapping_abs(),
            a[23].wrapping_abs(),
            a[24].wrapping_abs(),
            a[25].wrapping_abs(),
            a[26].wrapping_abs(),
            a[27].wrapping_abs(),
            a[28].wrapping_abs(),
            a[29].wrapping_abs(),
            a[30].wrapping_abs(),
            a[31].wrapping_abs(),
        ]
    }

    #[inline(always)]
    fn simd_eq(a: [i8; 32], b: [i8; 32]) -> [i8; 32] {
        [
            if a[0] == b[0] { -1 } else { 0 },
            if a[1] == b[1] { -1 } else { 0 },
            if a[2] == b[2] { -1 } else { 0 },
            if a[3] == b[3] { -1 } else { 0 },
            if a[4] == b[4] { -1 } else { 0 },
            if a[5] == b[5] { -1 } else { 0 },
            if a[6] == b[6] { -1 } else { 0 },
            if a[7] == b[7] { -1 } else { 0 },
            if a[8] == b[8] { -1 } else { 0 },
            if a[9] == b[9] { -1 } else { 0 },
            if a[10] == b[10] { -1 } else { 0 },
            if a[11] == b[11] { -1 } else { 0 },
            if a[12] == b[12] { -1 } else { 0 },
            if a[13] == b[13] { -1 } else { 0 },
            if a[14] == b[14] { -1 } else { 0 },
            if a[15] == b[15] { -1 } else { 0 },
            if a[16] == b[16] { -1 } else { 0 },
            if a[17] == b[17] { -1 } else { 0 },
            if a[18] == b[18] { -1 } else { 0 },
            if a[19] == b[19] { -1 } else { 0 },
            if a[20] == b[20] { -1 } else { 0 },
            if a[21] == b[21] { -1 } else { 0 },
            if a[22] == b[22] { -1 } else { 0 },
            if a[23] == b[23] { -1 } else { 0 },
            if a[24] == b[24] { -1 } else { 0 },
            if a[25] == b[25] { -1 } else { 0 },
            if a[26] == b[26] { -1 } else { 0 },
            if a[27] == b[27] { -1 } else { 0 },
            if a[28] == b[28] { -1 } else { 0 },
            if a[29] == b[29] { -1 } else { 0 },
            if a[30] == b[30] { -1 } else { 0 },
            if a[31] == b[31] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ne(a: [i8; 32], b: [i8; 32]) -> [i8; 32] {
        [
            if a[0] != b[0] { -1 } else { 0 },
            if a[1] != b[1] { -1 } else { 0 },
            if a[2] != b[2] { -1 } else { 0 },
            if a[3] != b[3] { -1 } else { 0 },
            if a[4] != b[4] { -1 } else { 0 },
            if a[5] != b[5] { -1 } else { 0 },
            if a[6] != b[6] { -1 } else { 0 },
            if a[7] != b[7] { -1 } else { 0 },
            if a[8] != b[8] { -1 } else { 0 },
            if a[9] != b[9] { -1 } else { 0 },
            if a[10] != b[10] { -1 } else { 0 },
            if a[11] != b[11] { -1 } else { 0 },
            if a[12] != b[12] { -1 } else { 0 },
            if a[13] != b[13] { -1 } else { 0 },
            if a[14] != b[14] { -1 } else { 0 },
            if a[15] != b[15] { -1 } else { 0 },
            if a[16] != b[16] { -1 } else { 0 },
            if a[17] != b[17] { -1 } else { 0 },
            if a[18] != b[18] { -1 } else { 0 },
            if a[19] != b[19] { -1 } else { 0 },
            if a[20] != b[20] { -1 } else { 0 },
            if a[21] != b[21] { -1 } else { 0 },
            if a[22] != b[22] { -1 } else { 0 },
            if a[23] != b[23] { -1 } else { 0 },
            if a[24] != b[24] { -1 } else { 0 },
            if a[25] != b[25] { -1 } else { 0 },
            if a[26] != b[26] { -1 } else { 0 },
            if a[27] != b[27] { -1 } else { 0 },
            if a[28] != b[28] { -1 } else { 0 },
            if a[29] != b[29] { -1 } else { 0 },
            if a[30] != b[30] { -1 } else { 0 },
            if a[31] != b[31] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_lt(a: [i8; 32], b: [i8; 32]) -> [i8; 32] {
        [
            if a[0] < b[0] { -1 } else { 0 },
            if a[1] < b[1] { -1 } else { 0 },
            if a[2] < b[2] { -1 } else { 0 },
            if a[3] < b[3] { -1 } else { 0 },
            if a[4] < b[4] { -1 } else { 0 },
            if a[5] < b[5] { -1 } else { 0 },
            if a[6] < b[6] { -1 } else { 0 },
            if a[7] < b[7] { -1 } else { 0 },
            if a[8] < b[8] { -1 } else { 0 },
            if a[9] < b[9] { -1 } else { 0 },
            if a[10] < b[10] { -1 } else { 0 },
            if a[11] < b[11] { -1 } else { 0 },
            if a[12] < b[12] { -1 } else { 0 },
            if a[13] < b[13] { -1 } else { 0 },
            if a[14] < b[14] { -1 } else { 0 },
            if a[15] < b[15] { -1 } else { 0 },
            if a[16] < b[16] { -1 } else { 0 },
            if a[17] < b[17] { -1 } else { 0 },
            if a[18] < b[18] { -1 } else { 0 },
            if a[19] < b[19] { -1 } else { 0 },
            if a[20] < b[20] { -1 } else { 0 },
            if a[21] < b[21] { -1 } else { 0 },
            if a[22] < b[22] { -1 } else { 0 },
            if a[23] < b[23] { -1 } else { 0 },
            if a[24] < b[24] { -1 } else { 0 },
            if a[25] < b[25] { -1 } else { 0 },
            if a[26] < b[26] { -1 } else { 0 },
            if a[27] < b[27] { -1 } else { 0 },
            if a[28] < b[28] { -1 } else { 0 },
            if a[29] < b[29] { -1 } else { 0 },
            if a[30] < b[30] { -1 } else { 0 },
            if a[31] < b[31] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_le(a: [i8; 32], b: [i8; 32]) -> [i8; 32] {
        [
            if a[0] <= b[0] { -1 } else { 0 },
            if a[1] <= b[1] { -1 } else { 0 },
            if a[2] <= b[2] { -1 } else { 0 },
            if a[3] <= b[3] { -1 } else { 0 },
            if a[4] <= b[4] { -1 } else { 0 },
            if a[5] <= b[5] { -1 } else { 0 },
            if a[6] <= b[6] { -1 } else { 0 },
            if a[7] <= b[7] { -1 } else { 0 },
            if a[8] <= b[8] { -1 } else { 0 },
            if a[9] <= b[9] { -1 } else { 0 },
            if a[10] <= b[10] { -1 } else { 0 },
            if a[11] <= b[11] { -1 } else { 0 },
            if a[12] <= b[12] { -1 } else { 0 },
            if a[13] <= b[13] { -1 } else { 0 },
            if a[14] <= b[14] { -1 } else { 0 },
            if a[15] <= b[15] { -1 } else { 0 },
            if a[16] <= b[16] { -1 } else { 0 },
            if a[17] <= b[17] { -1 } else { 0 },
            if a[18] <= b[18] { -1 } else { 0 },
            if a[19] <= b[19] { -1 } else { 0 },
            if a[20] <= b[20] { -1 } else { 0 },
            if a[21] <= b[21] { -1 } else { 0 },
            if a[22] <= b[22] { -1 } else { 0 },
            if a[23] <= b[23] { -1 } else { 0 },
            if a[24] <= b[24] { -1 } else { 0 },
            if a[25] <= b[25] { -1 } else { 0 },
            if a[26] <= b[26] { -1 } else { 0 },
            if a[27] <= b[27] { -1 } else { 0 },
            if a[28] <= b[28] { -1 } else { 0 },
            if a[29] <= b[29] { -1 } else { 0 },
            if a[30] <= b[30] { -1 } else { 0 },
            if a[31] <= b[31] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_gt(a: [i8; 32], b: [i8; 32]) -> [i8; 32] {
        [
            if a[0] > b[0] { -1 } else { 0 },
            if a[1] > b[1] { -1 } else { 0 },
            if a[2] > b[2] { -1 } else { 0 },
            if a[3] > b[3] { -1 } else { 0 },
            if a[4] > b[4] { -1 } else { 0 },
            if a[5] > b[5] { -1 } else { 0 },
            if a[6] > b[6] { -1 } else { 0 },
            if a[7] > b[7] { -1 } else { 0 },
            if a[8] > b[8] { -1 } else { 0 },
            if a[9] > b[9] { -1 } else { 0 },
            if a[10] > b[10] { -1 } else { 0 },
            if a[11] > b[11] { -1 } else { 0 },
            if a[12] > b[12] { -1 } else { 0 },
            if a[13] > b[13] { -1 } else { 0 },
            if a[14] > b[14] { -1 } else { 0 },
            if a[15] > b[15] { -1 } else { 0 },
            if a[16] > b[16] { -1 } else { 0 },
            if a[17] > b[17] { -1 } else { 0 },
            if a[18] > b[18] { -1 } else { 0 },
            if a[19] > b[19] { -1 } else { 0 },
            if a[20] > b[20] { -1 } else { 0 },
            if a[21] > b[21] { -1 } else { 0 },
            if a[22] > b[22] { -1 } else { 0 },
            if a[23] > b[23] { -1 } else { 0 },
            if a[24] > b[24] { -1 } else { 0 },
            if a[25] > b[25] { -1 } else { 0 },
            if a[26] > b[26] { -1 } else { 0 },
            if a[27] > b[27] { -1 } else { 0 },
            if a[28] > b[28] { -1 } else { 0 },
            if a[29] > b[29] { -1 } else { 0 },
            if a[30] > b[30] { -1 } else { 0 },
            if a[31] > b[31] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ge(a: [i8; 32], b: [i8; 32]) -> [i8; 32] {
        [
            if a[0] >= b[0] { -1 } else { 0 },
            if a[1] >= b[1] { -1 } else { 0 },
            if a[2] >= b[2] { -1 } else { 0 },
            if a[3] >= b[3] { -1 } else { 0 },
            if a[4] >= b[4] { -1 } else { 0 },
            if a[5] >= b[5] { -1 } else { 0 },
            if a[6] >= b[6] { -1 } else { 0 },
            if a[7] >= b[7] { -1 } else { 0 },
            if a[8] >= b[8] { -1 } else { 0 },
            if a[9] >= b[9] { -1 } else { 0 },
            if a[10] >= b[10] { -1 } else { 0 },
            if a[11] >= b[11] { -1 } else { 0 },
            if a[12] >= b[12] { -1 } else { 0 },
            if a[13] >= b[13] { -1 } else { 0 },
            if a[14] >= b[14] { -1 } else { 0 },
            if a[15] >= b[15] { -1 } else { 0 },
            if a[16] >= b[16] { -1 } else { 0 },
            if a[17] >= b[17] { -1 } else { 0 },
            if a[18] >= b[18] { -1 } else { 0 },
            if a[19] >= b[19] { -1 } else { 0 },
            if a[20] >= b[20] { -1 } else { 0 },
            if a[21] >= b[21] { -1 } else { 0 },
            if a[22] >= b[22] { -1 } else { 0 },
            if a[23] >= b[23] { -1 } else { 0 },
            if a[24] >= b[24] { -1 } else { 0 },
            if a[25] >= b[25] { -1 } else { 0 },
            if a[26] >= b[26] { -1 } else { 0 },
            if a[27] >= b[27] { -1 } else { 0 },
            if a[28] >= b[28] { -1 } else { 0 },
            if a[29] >= b[29] { -1 } else { 0 },
            if a[30] >= b[30] { -1 } else { 0 },
            if a[31] >= b[31] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn blend(mask: [i8; 32], if_true: [i8; 32], if_false: [i8; 32]) -> [i8; 32] {
        [
            if mask[0] != 0 {
                if_true[0]
            } else {
                if_false[0]
            },
            if mask[1] != 0 {
                if_true[1]
            } else {
                if_false[1]
            },
            if mask[2] != 0 {
                if_true[2]
            } else {
                if_false[2]
            },
            if mask[3] != 0 {
                if_true[3]
            } else {
                if_false[3]
            },
            if mask[4] != 0 {
                if_true[4]
            } else {
                if_false[4]
            },
            if mask[5] != 0 {
                if_true[5]
            } else {
                if_false[5]
            },
            if mask[6] != 0 {
                if_true[6]
            } else {
                if_false[6]
            },
            if mask[7] != 0 {
                if_true[7]
            } else {
                if_false[7]
            },
            if mask[8] != 0 {
                if_true[8]
            } else {
                if_false[8]
            },
            if mask[9] != 0 {
                if_true[9]
            } else {
                if_false[9]
            },
            if mask[10] != 0 {
                if_true[10]
            } else {
                if_false[10]
            },
            if mask[11] != 0 {
                if_true[11]
            } else {
                if_false[11]
            },
            if mask[12] != 0 {
                if_true[12]
            } else {
                if_false[12]
            },
            if mask[13] != 0 {
                if_true[13]
            } else {
                if_false[13]
            },
            if mask[14] != 0 {
                if_true[14]
            } else {
                if_false[14]
            },
            if mask[15] != 0 {
                if_true[15]
            } else {
                if_false[15]
            },
            if mask[16] != 0 {
                if_true[16]
            } else {
                if_false[16]
            },
            if mask[17] != 0 {
                if_true[17]
            } else {
                if_false[17]
            },
            if mask[18] != 0 {
                if_true[18]
            } else {
                if_false[18]
            },
            if mask[19] != 0 {
                if_true[19]
            } else {
                if_false[19]
            },
            if mask[20] != 0 {
                if_true[20]
            } else {
                if_false[20]
            },
            if mask[21] != 0 {
                if_true[21]
            } else {
                if_false[21]
            },
            if mask[22] != 0 {
                if_true[22]
            } else {
                if_false[22]
            },
            if mask[23] != 0 {
                if_true[23]
            } else {
                if_false[23]
            },
            if mask[24] != 0 {
                if_true[24]
            } else {
                if_false[24]
            },
            if mask[25] != 0 {
                if_true[25]
            } else {
                if_false[25]
            },
            if mask[26] != 0 {
                if_true[26]
            } else {
                if_false[26]
            },
            if mask[27] != 0 {
                if_true[27]
            } else {
                if_false[27]
            },
            if mask[28] != 0 {
                if_true[28]
            } else {
                if_false[28]
            },
            if mask[29] != 0 {
                if_true[29]
            } else {
                if_false[29]
            },
            if mask[30] != 0 {
                if_true[30]
            } else {
                if_false[30]
            },
            if mask[31] != 0 {
                if_true[31]
            } else {
                if_false[31]
            },
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [i8; 32]) -> i8 {
        a.iter().copied().fold(0i8, i8::wrapping_add)
    }

    #[inline(always)]
    fn not(a: [i8; 32]) -> [i8; 32] {
        [
            !a[0], !a[1], !a[2], !a[3], !a[4], !a[5], !a[6], !a[7], !a[8], !a[9], !a[10], !a[11],
            !a[12], !a[13], !a[14], !a[15], !a[16], !a[17], !a[18], !a[19], !a[20], !a[21], !a[22],
            !a[23], !a[24], !a[25], !a[26], !a[27], !a[28], !a[29], !a[30], !a[31],
        ]
    }

    #[inline(always)]
    fn bitand(a: [i8; 32], b: [i8; 32]) -> [i8; 32] {
        [
            a[0] & b[0],
            a[1] & b[1],
            a[2] & b[2],
            a[3] & b[3],
            a[4] & b[4],
            a[5] & b[5],
            a[6] & b[6],
            a[7] & b[7],
            a[8] & b[8],
            a[9] & b[9],
            a[10] & b[10],
            a[11] & b[11],
            a[12] & b[12],
            a[13] & b[13],
            a[14] & b[14],
            a[15] & b[15],
            a[16] & b[16],
            a[17] & b[17],
            a[18] & b[18],
            a[19] & b[19],
            a[20] & b[20],
            a[21] & b[21],
            a[22] & b[22],
            a[23] & b[23],
            a[24] & b[24],
            a[25] & b[25],
            a[26] & b[26],
            a[27] & b[27],
            a[28] & b[28],
            a[29] & b[29],
            a[30] & b[30],
            a[31] & b[31],
        ]
    }

    #[inline(always)]
    fn bitor(a: [i8; 32], b: [i8; 32]) -> [i8; 32] {
        [
            a[0] | b[0],
            a[1] | b[1],
            a[2] | b[2],
            a[3] | b[3],
            a[4] | b[4],
            a[5] | b[5],
            a[6] | b[6],
            a[7] | b[7],
            a[8] | b[8],
            a[9] | b[9],
            a[10] | b[10],
            a[11] | b[11],
            a[12] | b[12],
            a[13] | b[13],
            a[14] | b[14],
            a[15] | b[15],
            a[16] | b[16],
            a[17] | b[17],
            a[18] | b[18],
            a[19] | b[19],
            a[20] | b[20],
            a[21] | b[21],
            a[22] | b[22],
            a[23] | b[23],
            a[24] | b[24],
            a[25] | b[25],
            a[26] | b[26],
            a[27] | b[27],
            a[28] | b[28],
            a[29] | b[29],
            a[30] | b[30],
            a[31] | b[31],
        ]
    }

    #[inline(always)]
    fn bitxor(a: [i8; 32], b: [i8; 32]) -> [i8; 32] {
        [
            a[0] ^ b[0],
            a[1] ^ b[1],
            a[2] ^ b[2],
            a[3] ^ b[3],
            a[4] ^ b[4],
            a[5] ^ b[5],
            a[6] ^ b[6],
            a[7] ^ b[7],
            a[8] ^ b[8],
            a[9] ^ b[9],
            a[10] ^ b[10],
            a[11] ^ b[11],
            a[12] ^ b[12],
            a[13] ^ b[13],
            a[14] ^ b[14],
            a[15] ^ b[15],
            a[16] ^ b[16],
            a[17] ^ b[17],
            a[18] ^ b[18],
            a[19] ^ b[19],
            a[20] ^ b[20],
            a[21] ^ b[21],
            a[22] ^ b[22],
            a[23] ^ b[23],
            a[24] ^ b[24],
            a[25] ^ b[25],
            a[26] ^ b[26],
            a[27] ^ b[27],
            a[28] ^ b[28],
            a[29] ^ b[29],
            a[30] ^ b[30],
            a[31] ^ b[31],
        ]
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [i8; 32]) -> [i8; 32] {
        [
            a[0].wrapping_shl(N as u32),
            a[1].wrapping_shl(N as u32),
            a[2].wrapping_shl(N as u32),
            a[3].wrapping_shl(N as u32),
            a[4].wrapping_shl(N as u32),
            a[5].wrapping_shl(N as u32),
            a[6].wrapping_shl(N as u32),
            a[7].wrapping_shl(N as u32),
            a[8].wrapping_shl(N as u32),
            a[9].wrapping_shl(N as u32),
            a[10].wrapping_shl(N as u32),
            a[11].wrapping_shl(N as u32),
            a[12].wrapping_shl(N as u32),
            a[13].wrapping_shl(N as u32),
            a[14].wrapping_shl(N as u32),
            a[15].wrapping_shl(N as u32),
            a[16].wrapping_shl(N as u32),
            a[17].wrapping_shl(N as u32),
            a[18].wrapping_shl(N as u32),
            a[19].wrapping_shl(N as u32),
            a[20].wrapping_shl(N as u32),
            a[21].wrapping_shl(N as u32),
            a[22].wrapping_shl(N as u32),
            a[23].wrapping_shl(N as u32),
            a[24].wrapping_shl(N as u32),
            a[25].wrapping_shl(N as u32),
            a[26].wrapping_shl(N as u32),
            a[27].wrapping_shl(N as u32),
            a[28].wrapping_shl(N as u32),
            a[29].wrapping_shl(N as u32),
            a[30].wrapping_shl(N as u32),
            a[31].wrapping_shl(N as u32),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [i8; 32]) -> [i8; 32] {
        [
            (a[0] as u8).wrapping_shr(N as u32) as i8,
            (a[1] as u8).wrapping_shr(N as u32) as i8,
            (a[2] as u8).wrapping_shr(N as u32) as i8,
            (a[3] as u8).wrapping_shr(N as u32) as i8,
            (a[4] as u8).wrapping_shr(N as u32) as i8,
            (a[5] as u8).wrapping_shr(N as u32) as i8,
            (a[6] as u8).wrapping_shr(N as u32) as i8,
            (a[7] as u8).wrapping_shr(N as u32) as i8,
            (a[8] as u8).wrapping_shr(N as u32) as i8,
            (a[9] as u8).wrapping_shr(N as u32) as i8,
            (a[10] as u8).wrapping_shr(N as u32) as i8,
            (a[11] as u8).wrapping_shr(N as u32) as i8,
            (a[12] as u8).wrapping_shr(N as u32) as i8,
            (a[13] as u8).wrapping_shr(N as u32) as i8,
            (a[14] as u8).wrapping_shr(N as u32) as i8,
            (a[15] as u8).wrapping_shr(N as u32) as i8,
            (a[16] as u8).wrapping_shr(N as u32) as i8,
            (a[17] as u8).wrapping_shr(N as u32) as i8,
            (a[18] as u8).wrapping_shr(N as u32) as i8,
            (a[19] as u8).wrapping_shr(N as u32) as i8,
            (a[20] as u8).wrapping_shr(N as u32) as i8,
            (a[21] as u8).wrapping_shr(N as u32) as i8,
            (a[22] as u8).wrapping_shr(N as u32) as i8,
            (a[23] as u8).wrapping_shr(N as u32) as i8,
            (a[24] as u8).wrapping_shr(N as u32) as i8,
            (a[25] as u8).wrapping_shr(N as u32) as i8,
            (a[26] as u8).wrapping_shr(N as u32) as i8,
            (a[27] as u8).wrapping_shr(N as u32) as i8,
            (a[28] as u8).wrapping_shr(N as u32) as i8,
            (a[29] as u8).wrapping_shr(N as u32) as i8,
            (a[30] as u8).wrapping_shr(N as u32) as i8,
            (a[31] as u8).wrapping_shr(N as u32) as i8,
        ]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [i8; 32]) -> [i8; 32] {
        [
            a[0].wrapping_shr(N as u32),
            a[1].wrapping_shr(N as u32),
            a[2].wrapping_shr(N as u32),
            a[3].wrapping_shr(N as u32),
            a[4].wrapping_shr(N as u32),
            a[5].wrapping_shr(N as u32),
            a[6].wrapping_shr(N as u32),
            a[7].wrapping_shr(N as u32),
            a[8].wrapping_shr(N as u32),
            a[9].wrapping_shr(N as u32),
            a[10].wrapping_shr(N as u32),
            a[11].wrapping_shr(N as u32),
            a[12].wrapping_shr(N as u32),
            a[13].wrapping_shr(N as u32),
            a[14].wrapping_shr(N as u32),
            a[15].wrapping_shr(N as u32),
            a[16].wrapping_shr(N as u32),
            a[17].wrapping_shr(N as u32),
            a[18].wrapping_shr(N as u32),
            a[19].wrapping_shr(N as u32),
            a[20].wrapping_shr(N as u32),
            a[21].wrapping_shr(N as u32),
            a[22].wrapping_shr(N as u32),
            a[23].wrapping_shr(N as u32),
            a[24].wrapping_shr(N as u32),
            a[25].wrapping_shr(N as u32),
            a[26].wrapping_shr(N as u32),
            a[27].wrapping_shr(N as u32),
            a[28].wrapping_shr(N as u32),
            a[29].wrapping_shr(N as u32),
            a[30].wrapping_shr(N as u32),
            a[31].wrapping_shr(N as u32),
        ]
    }

    #[inline(always)]
    fn all_true(a: [i8; 32]) -> bool {
        a[0] != 0
            && a[1] != 0
            && a[2] != 0
            && a[3] != 0
            && a[4] != 0
            && a[5] != 0
            && a[6] != 0
            && a[7] != 0
            && a[8] != 0
            && a[9] != 0
            && a[10] != 0
            && a[11] != 0
            && a[12] != 0
            && a[13] != 0
            && a[14] != 0
            && a[15] != 0
            && a[16] != 0
            && a[17] != 0
            && a[18] != 0
            && a[19] != 0
            && a[20] != 0
            && a[21] != 0
            && a[22] != 0
            && a[23] != 0
            && a[24] != 0
            && a[25] != 0
            && a[26] != 0
            && a[27] != 0
            && a[28] != 0
            && a[29] != 0
            && a[30] != 0
            && a[31] != 0
    }

    #[inline(always)]
    fn any_true(a: [i8; 32]) -> bool {
        a[0] != 0
            || a[1] != 0
            || a[2] != 0
            || a[3] != 0
            || a[4] != 0
            || a[5] != 0
            || a[6] != 0
            || a[7] != 0
            || a[8] != 0
            || a[9] != 0
            || a[10] != 0
            || a[11] != 0
            || a[12] != 0
            || a[13] != 0
            || a[14] != 0
            || a[15] != 0
            || a[16] != 0
            || a[17] != 0
            || a[18] != 0
            || a[19] != 0
            || a[20] != 0
            || a[21] != 0
            || a[22] != 0
            || a[23] != 0
            || a[24] != 0
            || a[25] != 0
            || a[26] != 0
            || a[27] != 0
            || a[28] != 0
            || a[29] != 0
            || a[30] != 0
            || a[31] != 0
    }

    #[inline(always)]
    fn bitmask(a: [i8; 32]) -> u32 {
        ((a[0] >> 7) as u32 & 1) << 0
            | ((a[1] >> 7) as u32 & 1) << 1
            | ((a[2] >> 7) as u32 & 1) << 2
            | ((a[3] >> 7) as u32 & 1) << 3
            | ((a[4] >> 7) as u32 & 1) << 4
            | ((a[5] >> 7) as u32 & 1) << 5
            | ((a[6] >> 7) as u32 & 1) << 6
            | ((a[7] >> 7) as u32 & 1) << 7
            | ((a[8] >> 7) as u32 & 1) << 8
            | ((a[9] >> 7) as u32 & 1) << 9
            | ((a[10] >> 7) as u32 & 1) << 10
            | ((a[11] >> 7) as u32 & 1) << 11
            | ((a[12] >> 7) as u32 & 1) << 12
            | ((a[13] >> 7) as u32 & 1) << 13
            | ((a[14] >> 7) as u32 & 1) << 14
            | ((a[15] >> 7) as u32 & 1) << 15
            | ((a[16] >> 7) as u32 & 1) << 16
            | ((a[17] >> 7) as u32 & 1) << 17
            | ((a[18] >> 7) as u32 & 1) << 18
            | ((a[19] >> 7) as u32 & 1) << 19
            | ((a[20] >> 7) as u32 & 1) << 20
            | ((a[21] >> 7) as u32 & 1) << 21
            | ((a[22] >> 7) as u32 & 1) << 22
            | ((a[23] >> 7) as u32 & 1) << 23
            | ((a[24] >> 7) as u32 & 1) << 24
            | ((a[25] >> 7) as u32 & 1) << 25
            | ((a[26] >> 7) as u32 & 1) << 26
            | ((a[27] >> 7) as u32 & 1) << 27
            | ((a[28] >> 7) as u32 & 1) << 28
            | ((a[29] >> 7) as u32 & 1) << 29
            | ((a[30] >> 7) as u32 & 1) << 30
            | ((a[31] >> 7) as u32 & 1) << 31
    }
}

impl U8x16Backend for archmage::ScalarToken {
    type Repr = [u8; 16];

    #[inline(always)]
    fn splat(v: u8) -> [u8; 16] {
        [v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v]
    }

    #[inline(always)]
    fn zero() -> [u8; 16] {
        [0u8; 16]
    }

    #[inline(always)]
    fn load(data: &[u8; 16]) -> [u8; 16] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [u8; 16]) -> [u8; 16] {
        arr
    }

    #[inline(always)]
    fn store(repr: [u8; 16], out: &mut [u8; 16]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [u8; 16]) -> [u8; 16] {
        repr
    }

    #[inline(always)]
    fn add(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        [
            a[0].wrapping_add(b[0]),
            a[1].wrapping_add(b[1]),
            a[2].wrapping_add(b[2]),
            a[3].wrapping_add(b[3]),
            a[4].wrapping_add(b[4]),
            a[5].wrapping_add(b[5]),
            a[6].wrapping_add(b[6]),
            a[7].wrapping_add(b[7]),
            a[8].wrapping_add(b[8]),
            a[9].wrapping_add(b[9]),
            a[10].wrapping_add(b[10]),
            a[11].wrapping_add(b[11]),
            a[12].wrapping_add(b[12]),
            a[13].wrapping_add(b[13]),
            a[14].wrapping_add(b[14]),
            a[15].wrapping_add(b[15]),
        ]
    }

    #[inline(always)]
    fn sub(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        [
            a[0].wrapping_sub(b[0]),
            a[1].wrapping_sub(b[1]),
            a[2].wrapping_sub(b[2]),
            a[3].wrapping_sub(b[3]),
            a[4].wrapping_sub(b[4]),
            a[5].wrapping_sub(b[5]),
            a[6].wrapping_sub(b[6]),
            a[7].wrapping_sub(b[7]),
            a[8].wrapping_sub(b[8]),
            a[9].wrapping_sub(b[9]),
            a[10].wrapping_sub(b[10]),
            a[11].wrapping_sub(b[11]),
            a[12].wrapping_sub(b[12]),
            a[13].wrapping_sub(b[13]),
            a[14].wrapping_sub(b[14]),
            a[15].wrapping_sub(b[15]),
        ]
    }

    #[inline(always)]
    fn min(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        [
            if a[0] < b[0] { a[0] } else { b[0] },
            if a[1] < b[1] { a[1] } else { b[1] },
            if a[2] < b[2] { a[2] } else { b[2] },
            if a[3] < b[3] { a[3] } else { b[3] },
            if a[4] < b[4] { a[4] } else { b[4] },
            if a[5] < b[5] { a[5] } else { b[5] },
            if a[6] < b[6] { a[6] } else { b[6] },
            if a[7] < b[7] { a[7] } else { b[7] },
            if a[8] < b[8] { a[8] } else { b[8] },
            if a[9] < b[9] { a[9] } else { b[9] },
            if a[10] < b[10] { a[10] } else { b[10] },
            if a[11] < b[11] { a[11] } else { b[11] },
            if a[12] < b[12] { a[12] } else { b[12] },
            if a[13] < b[13] { a[13] } else { b[13] },
            if a[14] < b[14] { a[14] } else { b[14] },
            if a[15] < b[15] { a[15] } else { b[15] },
        ]
    }

    #[inline(always)]
    fn max(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        [
            if a[0] > b[0] { a[0] } else { b[0] },
            if a[1] > b[1] { a[1] } else { b[1] },
            if a[2] > b[2] { a[2] } else { b[2] },
            if a[3] > b[3] { a[3] } else { b[3] },
            if a[4] > b[4] { a[4] } else { b[4] },
            if a[5] > b[5] { a[5] } else { b[5] },
            if a[6] > b[6] { a[6] } else { b[6] },
            if a[7] > b[7] { a[7] } else { b[7] },
            if a[8] > b[8] { a[8] } else { b[8] },
            if a[9] > b[9] { a[9] } else { b[9] },
            if a[10] > b[10] { a[10] } else { b[10] },
            if a[11] > b[11] { a[11] } else { b[11] },
            if a[12] > b[12] { a[12] } else { b[12] },
            if a[13] > b[13] { a[13] } else { b[13] },
            if a[14] > b[14] { a[14] } else { b[14] },
            if a[15] > b[15] { a[15] } else { b[15] },
        ]
    }

    #[inline(always)]
    fn simd_eq(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        [
            if a[0] == b[0] { 0xFF } else { 0 },
            if a[1] == b[1] { 0xFF } else { 0 },
            if a[2] == b[2] { 0xFF } else { 0 },
            if a[3] == b[3] { 0xFF } else { 0 },
            if a[4] == b[4] { 0xFF } else { 0 },
            if a[5] == b[5] { 0xFF } else { 0 },
            if a[6] == b[6] { 0xFF } else { 0 },
            if a[7] == b[7] { 0xFF } else { 0 },
            if a[8] == b[8] { 0xFF } else { 0 },
            if a[9] == b[9] { 0xFF } else { 0 },
            if a[10] == b[10] { 0xFF } else { 0 },
            if a[11] == b[11] { 0xFF } else { 0 },
            if a[12] == b[12] { 0xFF } else { 0 },
            if a[13] == b[13] { 0xFF } else { 0 },
            if a[14] == b[14] { 0xFF } else { 0 },
            if a[15] == b[15] { 0xFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ne(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        [
            if a[0] != b[0] { 0xFF } else { 0 },
            if a[1] != b[1] { 0xFF } else { 0 },
            if a[2] != b[2] { 0xFF } else { 0 },
            if a[3] != b[3] { 0xFF } else { 0 },
            if a[4] != b[4] { 0xFF } else { 0 },
            if a[5] != b[5] { 0xFF } else { 0 },
            if a[6] != b[6] { 0xFF } else { 0 },
            if a[7] != b[7] { 0xFF } else { 0 },
            if a[8] != b[8] { 0xFF } else { 0 },
            if a[9] != b[9] { 0xFF } else { 0 },
            if a[10] != b[10] { 0xFF } else { 0 },
            if a[11] != b[11] { 0xFF } else { 0 },
            if a[12] != b[12] { 0xFF } else { 0 },
            if a[13] != b[13] { 0xFF } else { 0 },
            if a[14] != b[14] { 0xFF } else { 0 },
            if a[15] != b[15] { 0xFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_lt(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        [
            if a[0] < b[0] { 0xFF } else { 0 },
            if a[1] < b[1] { 0xFF } else { 0 },
            if a[2] < b[2] { 0xFF } else { 0 },
            if a[3] < b[3] { 0xFF } else { 0 },
            if a[4] < b[4] { 0xFF } else { 0 },
            if a[5] < b[5] { 0xFF } else { 0 },
            if a[6] < b[6] { 0xFF } else { 0 },
            if a[7] < b[7] { 0xFF } else { 0 },
            if a[8] < b[8] { 0xFF } else { 0 },
            if a[9] < b[9] { 0xFF } else { 0 },
            if a[10] < b[10] { 0xFF } else { 0 },
            if a[11] < b[11] { 0xFF } else { 0 },
            if a[12] < b[12] { 0xFF } else { 0 },
            if a[13] < b[13] { 0xFF } else { 0 },
            if a[14] < b[14] { 0xFF } else { 0 },
            if a[15] < b[15] { 0xFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_le(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        [
            if a[0] <= b[0] { 0xFF } else { 0 },
            if a[1] <= b[1] { 0xFF } else { 0 },
            if a[2] <= b[2] { 0xFF } else { 0 },
            if a[3] <= b[3] { 0xFF } else { 0 },
            if a[4] <= b[4] { 0xFF } else { 0 },
            if a[5] <= b[5] { 0xFF } else { 0 },
            if a[6] <= b[6] { 0xFF } else { 0 },
            if a[7] <= b[7] { 0xFF } else { 0 },
            if a[8] <= b[8] { 0xFF } else { 0 },
            if a[9] <= b[9] { 0xFF } else { 0 },
            if a[10] <= b[10] { 0xFF } else { 0 },
            if a[11] <= b[11] { 0xFF } else { 0 },
            if a[12] <= b[12] { 0xFF } else { 0 },
            if a[13] <= b[13] { 0xFF } else { 0 },
            if a[14] <= b[14] { 0xFF } else { 0 },
            if a[15] <= b[15] { 0xFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_gt(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        [
            if a[0] > b[0] { 0xFF } else { 0 },
            if a[1] > b[1] { 0xFF } else { 0 },
            if a[2] > b[2] { 0xFF } else { 0 },
            if a[3] > b[3] { 0xFF } else { 0 },
            if a[4] > b[4] { 0xFF } else { 0 },
            if a[5] > b[5] { 0xFF } else { 0 },
            if a[6] > b[6] { 0xFF } else { 0 },
            if a[7] > b[7] { 0xFF } else { 0 },
            if a[8] > b[8] { 0xFF } else { 0 },
            if a[9] > b[9] { 0xFF } else { 0 },
            if a[10] > b[10] { 0xFF } else { 0 },
            if a[11] > b[11] { 0xFF } else { 0 },
            if a[12] > b[12] { 0xFF } else { 0 },
            if a[13] > b[13] { 0xFF } else { 0 },
            if a[14] > b[14] { 0xFF } else { 0 },
            if a[15] > b[15] { 0xFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ge(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        [
            if a[0] >= b[0] { 0xFF } else { 0 },
            if a[1] >= b[1] { 0xFF } else { 0 },
            if a[2] >= b[2] { 0xFF } else { 0 },
            if a[3] >= b[3] { 0xFF } else { 0 },
            if a[4] >= b[4] { 0xFF } else { 0 },
            if a[5] >= b[5] { 0xFF } else { 0 },
            if a[6] >= b[6] { 0xFF } else { 0 },
            if a[7] >= b[7] { 0xFF } else { 0 },
            if a[8] >= b[8] { 0xFF } else { 0 },
            if a[9] >= b[9] { 0xFF } else { 0 },
            if a[10] >= b[10] { 0xFF } else { 0 },
            if a[11] >= b[11] { 0xFF } else { 0 },
            if a[12] >= b[12] { 0xFF } else { 0 },
            if a[13] >= b[13] { 0xFF } else { 0 },
            if a[14] >= b[14] { 0xFF } else { 0 },
            if a[15] >= b[15] { 0xFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn blend(mask: [u8; 16], if_true: [u8; 16], if_false: [u8; 16]) -> [u8; 16] {
        [
            if mask[0] != 0 {
                if_true[0]
            } else {
                if_false[0]
            },
            if mask[1] != 0 {
                if_true[1]
            } else {
                if_false[1]
            },
            if mask[2] != 0 {
                if_true[2]
            } else {
                if_false[2]
            },
            if mask[3] != 0 {
                if_true[3]
            } else {
                if_false[3]
            },
            if mask[4] != 0 {
                if_true[4]
            } else {
                if_false[4]
            },
            if mask[5] != 0 {
                if_true[5]
            } else {
                if_false[5]
            },
            if mask[6] != 0 {
                if_true[6]
            } else {
                if_false[6]
            },
            if mask[7] != 0 {
                if_true[7]
            } else {
                if_false[7]
            },
            if mask[8] != 0 {
                if_true[8]
            } else {
                if_false[8]
            },
            if mask[9] != 0 {
                if_true[9]
            } else {
                if_false[9]
            },
            if mask[10] != 0 {
                if_true[10]
            } else {
                if_false[10]
            },
            if mask[11] != 0 {
                if_true[11]
            } else {
                if_false[11]
            },
            if mask[12] != 0 {
                if_true[12]
            } else {
                if_false[12]
            },
            if mask[13] != 0 {
                if_true[13]
            } else {
                if_false[13]
            },
            if mask[14] != 0 {
                if_true[14]
            } else {
                if_false[14]
            },
            if mask[15] != 0 {
                if_true[15]
            } else {
                if_false[15]
            },
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [u8; 16]) -> u8 {
        a.iter().copied().fold(0u8, u8::wrapping_add)
    }

    #[inline(always)]
    fn not(a: [u8; 16]) -> [u8; 16] {
        [
            !a[0], !a[1], !a[2], !a[3], !a[4], !a[5], !a[6], !a[7], !a[8], !a[9], !a[10], !a[11],
            !a[12], !a[13], !a[14], !a[15],
        ]
    }

    #[inline(always)]
    fn bitand(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        [
            a[0] & b[0],
            a[1] & b[1],
            a[2] & b[2],
            a[3] & b[3],
            a[4] & b[4],
            a[5] & b[5],
            a[6] & b[6],
            a[7] & b[7],
            a[8] & b[8],
            a[9] & b[9],
            a[10] & b[10],
            a[11] & b[11],
            a[12] & b[12],
            a[13] & b[13],
            a[14] & b[14],
            a[15] & b[15],
        ]
    }

    #[inline(always)]
    fn bitor(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        [
            a[0] | b[0],
            a[1] | b[1],
            a[2] | b[2],
            a[3] | b[3],
            a[4] | b[4],
            a[5] | b[5],
            a[6] | b[6],
            a[7] | b[7],
            a[8] | b[8],
            a[9] | b[9],
            a[10] | b[10],
            a[11] | b[11],
            a[12] | b[12],
            a[13] | b[13],
            a[14] | b[14],
            a[15] | b[15],
        ]
    }

    #[inline(always)]
    fn bitxor(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
        [
            a[0] ^ b[0],
            a[1] ^ b[1],
            a[2] ^ b[2],
            a[3] ^ b[3],
            a[4] ^ b[4],
            a[5] ^ b[5],
            a[6] ^ b[6],
            a[7] ^ b[7],
            a[8] ^ b[8],
            a[9] ^ b[9],
            a[10] ^ b[10],
            a[11] ^ b[11],
            a[12] ^ b[12],
            a[13] ^ b[13],
            a[14] ^ b[14],
            a[15] ^ b[15],
        ]
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [u8; 16]) -> [u8; 16] {
        [
            a[0].wrapping_shl(N as u32),
            a[1].wrapping_shl(N as u32),
            a[2].wrapping_shl(N as u32),
            a[3].wrapping_shl(N as u32),
            a[4].wrapping_shl(N as u32),
            a[5].wrapping_shl(N as u32),
            a[6].wrapping_shl(N as u32),
            a[7].wrapping_shl(N as u32),
            a[8].wrapping_shl(N as u32),
            a[9].wrapping_shl(N as u32),
            a[10].wrapping_shl(N as u32),
            a[11].wrapping_shl(N as u32),
            a[12].wrapping_shl(N as u32),
            a[13].wrapping_shl(N as u32),
            a[14].wrapping_shl(N as u32),
            a[15].wrapping_shl(N as u32),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [u8; 16]) -> [u8; 16] {
        [
            a[0].wrapping_shr(N as u32),
            a[1].wrapping_shr(N as u32),
            a[2].wrapping_shr(N as u32),
            a[3].wrapping_shr(N as u32),
            a[4].wrapping_shr(N as u32),
            a[5].wrapping_shr(N as u32),
            a[6].wrapping_shr(N as u32),
            a[7].wrapping_shr(N as u32),
            a[8].wrapping_shr(N as u32),
            a[9].wrapping_shr(N as u32),
            a[10].wrapping_shr(N as u32),
            a[11].wrapping_shr(N as u32),
            a[12].wrapping_shr(N as u32),
            a[13].wrapping_shr(N as u32),
            a[14].wrapping_shr(N as u32),
            a[15].wrapping_shr(N as u32),
        ]
    }

    #[inline(always)]
    fn all_true(a: [u8; 16]) -> bool {
        a[0] != 0
            && a[1] != 0
            && a[2] != 0
            && a[3] != 0
            && a[4] != 0
            && a[5] != 0
            && a[6] != 0
            && a[7] != 0
            && a[8] != 0
            && a[9] != 0
            && a[10] != 0
            && a[11] != 0
            && a[12] != 0
            && a[13] != 0
            && a[14] != 0
            && a[15] != 0
    }

    #[inline(always)]
    fn any_true(a: [u8; 16]) -> bool {
        a[0] != 0
            || a[1] != 0
            || a[2] != 0
            || a[3] != 0
            || a[4] != 0
            || a[5] != 0
            || a[6] != 0
            || a[7] != 0
            || a[8] != 0
            || a[9] != 0
            || a[10] != 0
            || a[11] != 0
            || a[12] != 0
            || a[13] != 0
            || a[14] != 0
            || a[15] != 0
    }

    #[inline(always)]
    fn bitmask(a: [u8; 16]) -> u32 {
        ((a[0] >> 7) as u32 & 1) << 0
            | ((a[1] >> 7) as u32 & 1) << 1
            | ((a[2] >> 7) as u32 & 1) << 2
            | ((a[3] >> 7) as u32 & 1) << 3
            | ((a[4] >> 7) as u32 & 1) << 4
            | ((a[5] >> 7) as u32 & 1) << 5
            | ((a[6] >> 7) as u32 & 1) << 6
            | ((a[7] >> 7) as u32 & 1) << 7
            | ((a[8] >> 7) as u32 & 1) << 8
            | ((a[9] >> 7) as u32 & 1) << 9
            | ((a[10] >> 7) as u32 & 1) << 10
            | ((a[11] >> 7) as u32 & 1) << 11
            | ((a[12] >> 7) as u32 & 1) << 12
            | ((a[13] >> 7) as u32 & 1) << 13
            | ((a[14] >> 7) as u32 & 1) << 14
            | ((a[15] >> 7) as u32 & 1) << 15
    }
}

impl U8x32Backend for archmage::ScalarToken {
    type Repr = [u8; 32];

    #[inline(always)]
    fn splat(v: u8) -> [u8; 32] {
        [
            v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v,
            v, v, v,
        ]
    }

    #[inline(always)]
    fn zero() -> [u8; 32] {
        [0u8; 32]
    }

    #[inline(always)]
    fn load(data: &[u8; 32]) -> [u8; 32] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [u8; 32]) -> [u8; 32] {
        arr
    }

    #[inline(always)]
    fn store(repr: [u8; 32], out: &mut [u8; 32]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [u8; 32]) -> [u8; 32] {
        repr
    }

    #[inline(always)]
    fn add(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
        [
            a[0].wrapping_add(b[0]),
            a[1].wrapping_add(b[1]),
            a[2].wrapping_add(b[2]),
            a[3].wrapping_add(b[3]),
            a[4].wrapping_add(b[4]),
            a[5].wrapping_add(b[5]),
            a[6].wrapping_add(b[6]),
            a[7].wrapping_add(b[7]),
            a[8].wrapping_add(b[8]),
            a[9].wrapping_add(b[9]),
            a[10].wrapping_add(b[10]),
            a[11].wrapping_add(b[11]),
            a[12].wrapping_add(b[12]),
            a[13].wrapping_add(b[13]),
            a[14].wrapping_add(b[14]),
            a[15].wrapping_add(b[15]),
            a[16].wrapping_add(b[16]),
            a[17].wrapping_add(b[17]),
            a[18].wrapping_add(b[18]),
            a[19].wrapping_add(b[19]),
            a[20].wrapping_add(b[20]),
            a[21].wrapping_add(b[21]),
            a[22].wrapping_add(b[22]),
            a[23].wrapping_add(b[23]),
            a[24].wrapping_add(b[24]),
            a[25].wrapping_add(b[25]),
            a[26].wrapping_add(b[26]),
            a[27].wrapping_add(b[27]),
            a[28].wrapping_add(b[28]),
            a[29].wrapping_add(b[29]),
            a[30].wrapping_add(b[30]),
            a[31].wrapping_add(b[31]),
        ]
    }

    #[inline(always)]
    fn sub(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
        [
            a[0].wrapping_sub(b[0]),
            a[1].wrapping_sub(b[1]),
            a[2].wrapping_sub(b[2]),
            a[3].wrapping_sub(b[3]),
            a[4].wrapping_sub(b[4]),
            a[5].wrapping_sub(b[5]),
            a[6].wrapping_sub(b[6]),
            a[7].wrapping_sub(b[7]),
            a[8].wrapping_sub(b[8]),
            a[9].wrapping_sub(b[9]),
            a[10].wrapping_sub(b[10]),
            a[11].wrapping_sub(b[11]),
            a[12].wrapping_sub(b[12]),
            a[13].wrapping_sub(b[13]),
            a[14].wrapping_sub(b[14]),
            a[15].wrapping_sub(b[15]),
            a[16].wrapping_sub(b[16]),
            a[17].wrapping_sub(b[17]),
            a[18].wrapping_sub(b[18]),
            a[19].wrapping_sub(b[19]),
            a[20].wrapping_sub(b[20]),
            a[21].wrapping_sub(b[21]),
            a[22].wrapping_sub(b[22]),
            a[23].wrapping_sub(b[23]),
            a[24].wrapping_sub(b[24]),
            a[25].wrapping_sub(b[25]),
            a[26].wrapping_sub(b[26]),
            a[27].wrapping_sub(b[27]),
            a[28].wrapping_sub(b[28]),
            a[29].wrapping_sub(b[29]),
            a[30].wrapping_sub(b[30]),
            a[31].wrapping_sub(b[31]),
        ]
    }

    #[inline(always)]
    fn min(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
        [
            if a[0] < b[0] { a[0] } else { b[0] },
            if a[1] < b[1] { a[1] } else { b[1] },
            if a[2] < b[2] { a[2] } else { b[2] },
            if a[3] < b[3] { a[3] } else { b[3] },
            if a[4] < b[4] { a[4] } else { b[4] },
            if a[5] < b[5] { a[5] } else { b[5] },
            if a[6] < b[6] { a[6] } else { b[6] },
            if a[7] < b[7] { a[7] } else { b[7] },
            if a[8] < b[8] { a[8] } else { b[8] },
            if a[9] < b[9] { a[9] } else { b[9] },
            if a[10] < b[10] { a[10] } else { b[10] },
            if a[11] < b[11] { a[11] } else { b[11] },
            if a[12] < b[12] { a[12] } else { b[12] },
            if a[13] < b[13] { a[13] } else { b[13] },
            if a[14] < b[14] { a[14] } else { b[14] },
            if a[15] < b[15] { a[15] } else { b[15] },
            if a[16] < b[16] { a[16] } else { b[16] },
            if a[17] < b[17] { a[17] } else { b[17] },
            if a[18] < b[18] { a[18] } else { b[18] },
            if a[19] < b[19] { a[19] } else { b[19] },
            if a[20] < b[20] { a[20] } else { b[20] },
            if a[21] < b[21] { a[21] } else { b[21] },
            if a[22] < b[22] { a[22] } else { b[22] },
            if a[23] < b[23] { a[23] } else { b[23] },
            if a[24] < b[24] { a[24] } else { b[24] },
            if a[25] < b[25] { a[25] } else { b[25] },
            if a[26] < b[26] { a[26] } else { b[26] },
            if a[27] < b[27] { a[27] } else { b[27] },
            if a[28] < b[28] { a[28] } else { b[28] },
            if a[29] < b[29] { a[29] } else { b[29] },
            if a[30] < b[30] { a[30] } else { b[30] },
            if a[31] < b[31] { a[31] } else { b[31] },
        ]
    }

    #[inline(always)]
    fn max(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
        [
            if a[0] > b[0] { a[0] } else { b[0] },
            if a[1] > b[1] { a[1] } else { b[1] },
            if a[2] > b[2] { a[2] } else { b[2] },
            if a[3] > b[3] { a[3] } else { b[3] },
            if a[4] > b[4] { a[4] } else { b[4] },
            if a[5] > b[5] { a[5] } else { b[5] },
            if a[6] > b[6] { a[6] } else { b[6] },
            if a[7] > b[7] { a[7] } else { b[7] },
            if a[8] > b[8] { a[8] } else { b[8] },
            if a[9] > b[9] { a[9] } else { b[9] },
            if a[10] > b[10] { a[10] } else { b[10] },
            if a[11] > b[11] { a[11] } else { b[11] },
            if a[12] > b[12] { a[12] } else { b[12] },
            if a[13] > b[13] { a[13] } else { b[13] },
            if a[14] > b[14] { a[14] } else { b[14] },
            if a[15] > b[15] { a[15] } else { b[15] },
            if a[16] > b[16] { a[16] } else { b[16] },
            if a[17] > b[17] { a[17] } else { b[17] },
            if a[18] > b[18] { a[18] } else { b[18] },
            if a[19] > b[19] { a[19] } else { b[19] },
            if a[20] > b[20] { a[20] } else { b[20] },
            if a[21] > b[21] { a[21] } else { b[21] },
            if a[22] > b[22] { a[22] } else { b[22] },
            if a[23] > b[23] { a[23] } else { b[23] },
            if a[24] > b[24] { a[24] } else { b[24] },
            if a[25] > b[25] { a[25] } else { b[25] },
            if a[26] > b[26] { a[26] } else { b[26] },
            if a[27] > b[27] { a[27] } else { b[27] },
            if a[28] > b[28] { a[28] } else { b[28] },
            if a[29] > b[29] { a[29] } else { b[29] },
            if a[30] > b[30] { a[30] } else { b[30] },
            if a[31] > b[31] { a[31] } else { b[31] },
        ]
    }

    #[inline(always)]
    fn simd_eq(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
        [
            if a[0] == b[0] { 0xFF } else { 0 },
            if a[1] == b[1] { 0xFF } else { 0 },
            if a[2] == b[2] { 0xFF } else { 0 },
            if a[3] == b[3] { 0xFF } else { 0 },
            if a[4] == b[4] { 0xFF } else { 0 },
            if a[5] == b[5] { 0xFF } else { 0 },
            if a[6] == b[6] { 0xFF } else { 0 },
            if a[7] == b[7] { 0xFF } else { 0 },
            if a[8] == b[8] { 0xFF } else { 0 },
            if a[9] == b[9] { 0xFF } else { 0 },
            if a[10] == b[10] { 0xFF } else { 0 },
            if a[11] == b[11] { 0xFF } else { 0 },
            if a[12] == b[12] { 0xFF } else { 0 },
            if a[13] == b[13] { 0xFF } else { 0 },
            if a[14] == b[14] { 0xFF } else { 0 },
            if a[15] == b[15] { 0xFF } else { 0 },
            if a[16] == b[16] { 0xFF } else { 0 },
            if a[17] == b[17] { 0xFF } else { 0 },
            if a[18] == b[18] { 0xFF } else { 0 },
            if a[19] == b[19] { 0xFF } else { 0 },
            if a[20] == b[20] { 0xFF } else { 0 },
            if a[21] == b[21] { 0xFF } else { 0 },
            if a[22] == b[22] { 0xFF } else { 0 },
            if a[23] == b[23] { 0xFF } else { 0 },
            if a[24] == b[24] { 0xFF } else { 0 },
            if a[25] == b[25] { 0xFF } else { 0 },
            if a[26] == b[26] { 0xFF } else { 0 },
            if a[27] == b[27] { 0xFF } else { 0 },
            if a[28] == b[28] { 0xFF } else { 0 },
            if a[29] == b[29] { 0xFF } else { 0 },
            if a[30] == b[30] { 0xFF } else { 0 },
            if a[31] == b[31] { 0xFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ne(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
        [
            if a[0] != b[0] { 0xFF } else { 0 },
            if a[1] != b[1] { 0xFF } else { 0 },
            if a[2] != b[2] { 0xFF } else { 0 },
            if a[3] != b[3] { 0xFF } else { 0 },
            if a[4] != b[4] { 0xFF } else { 0 },
            if a[5] != b[5] { 0xFF } else { 0 },
            if a[6] != b[6] { 0xFF } else { 0 },
            if a[7] != b[7] { 0xFF } else { 0 },
            if a[8] != b[8] { 0xFF } else { 0 },
            if a[9] != b[9] { 0xFF } else { 0 },
            if a[10] != b[10] { 0xFF } else { 0 },
            if a[11] != b[11] { 0xFF } else { 0 },
            if a[12] != b[12] { 0xFF } else { 0 },
            if a[13] != b[13] { 0xFF } else { 0 },
            if a[14] != b[14] { 0xFF } else { 0 },
            if a[15] != b[15] { 0xFF } else { 0 },
            if a[16] != b[16] { 0xFF } else { 0 },
            if a[17] != b[17] { 0xFF } else { 0 },
            if a[18] != b[18] { 0xFF } else { 0 },
            if a[19] != b[19] { 0xFF } else { 0 },
            if a[20] != b[20] { 0xFF } else { 0 },
            if a[21] != b[21] { 0xFF } else { 0 },
            if a[22] != b[22] { 0xFF } else { 0 },
            if a[23] != b[23] { 0xFF } else { 0 },
            if a[24] != b[24] { 0xFF } else { 0 },
            if a[25] != b[25] { 0xFF } else { 0 },
            if a[26] != b[26] { 0xFF } else { 0 },
            if a[27] != b[27] { 0xFF } else { 0 },
            if a[28] != b[28] { 0xFF } else { 0 },
            if a[29] != b[29] { 0xFF } else { 0 },
            if a[30] != b[30] { 0xFF } else { 0 },
            if a[31] != b[31] { 0xFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_lt(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
        [
            if a[0] < b[0] { 0xFF } else { 0 },
            if a[1] < b[1] { 0xFF } else { 0 },
            if a[2] < b[2] { 0xFF } else { 0 },
            if a[3] < b[3] { 0xFF } else { 0 },
            if a[4] < b[4] { 0xFF } else { 0 },
            if a[5] < b[5] { 0xFF } else { 0 },
            if a[6] < b[6] { 0xFF } else { 0 },
            if a[7] < b[7] { 0xFF } else { 0 },
            if a[8] < b[8] { 0xFF } else { 0 },
            if a[9] < b[9] { 0xFF } else { 0 },
            if a[10] < b[10] { 0xFF } else { 0 },
            if a[11] < b[11] { 0xFF } else { 0 },
            if a[12] < b[12] { 0xFF } else { 0 },
            if a[13] < b[13] { 0xFF } else { 0 },
            if a[14] < b[14] { 0xFF } else { 0 },
            if a[15] < b[15] { 0xFF } else { 0 },
            if a[16] < b[16] { 0xFF } else { 0 },
            if a[17] < b[17] { 0xFF } else { 0 },
            if a[18] < b[18] { 0xFF } else { 0 },
            if a[19] < b[19] { 0xFF } else { 0 },
            if a[20] < b[20] { 0xFF } else { 0 },
            if a[21] < b[21] { 0xFF } else { 0 },
            if a[22] < b[22] { 0xFF } else { 0 },
            if a[23] < b[23] { 0xFF } else { 0 },
            if a[24] < b[24] { 0xFF } else { 0 },
            if a[25] < b[25] { 0xFF } else { 0 },
            if a[26] < b[26] { 0xFF } else { 0 },
            if a[27] < b[27] { 0xFF } else { 0 },
            if a[28] < b[28] { 0xFF } else { 0 },
            if a[29] < b[29] { 0xFF } else { 0 },
            if a[30] < b[30] { 0xFF } else { 0 },
            if a[31] < b[31] { 0xFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_le(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
        [
            if a[0] <= b[0] { 0xFF } else { 0 },
            if a[1] <= b[1] { 0xFF } else { 0 },
            if a[2] <= b[2] { 0xFF } else { 0 },
            if a[3] <= b[3] { 0xFF } else { 0 },
            if a[4] <= b[4] { 0xFF } else { 0 },
            if a[5] <= b[5] { 0xFF } else { 0 },
            if a[6] <= b[6] { 0xFF } else { 0 },
            if a[7] <= b[7] { 0xFF } else { 0 },
            if a[8] <= b[8] { 0xFF } else { 0 },
            if a[9] <= b[9] { 0xFF } else { 0 },
            if a[10] <= b[10] { 0xFF } else { 0 },
            if a[11] <= b[11] { 0xFF } else { 0 },
            if a[12] <= b[12] { 0xFF } else { 0 },
            if a[13] <= b[13] { 0xFF } else { 0 },
            if a[14] <= b[14] { 0xFF } else { 0 },
            if a[15] <= b[15] { 0xFF } else { 0 },
            if a[16] <= b[16] { 0xFF } else { 0 },
            if a[17] <= b[17] { 0xFF } else { 0 },
            if a[18] <= b[18] { 0xFF } else { 0 },
            if a[19] <= b[19] { 0xFF } else { 0 },
            if a[20] <= b[20] { 0xFF } else { 0 },
            if a[21] <= b[21] { 0xFF } else { 0 },
            if a[22] <= b[22] { 0xFF } else { 0 },
            if a[23] <= b[23] { 0xFF } else { 0 },
            if a[24] <= b[24] { 0xFF } else { 0 },
            if a[25] <= b[25] { 0xFF } else { 0 },
            if a[26] <= b[26] { 0xFF } else { 0 },
            if a[27] <= b[27] { 0xFF } else { 0 },
            if a[28] <= b[28] { 0xFF } else { 0 },
            if a[29] <= b[29] { 0xFF } else { 0 },
            if a[30] <= b[30] { 0xFF } else { 0 },
            if a[31] <= b[31] { 0xFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_gt(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
        [
            if a[0] > b[0] { 0xFF } else { 0 },
            if a[1] > b[1] { 0xFF } else { 0 },
            if a[2] > b[2] { 0xFF } else { 0 },
            if a[3] > b[3] { 0xFF } else { 0 },
            if a[4] > b[4] { 0xFF } else { 0 },
            if a[5] > b[5] { 0xFF } else { 0 },
            if a[6] > b[6] { 0xFF } else { 0 },
            if a[7] > b[7] { 0xFF } else { 0 },
            if a[8] > b[8] { 0xFF } else { 0 },
            if a[9] > b[9] { 0xFF } else { 0 },
            if a[10] > b[10] { 0xFF } else { 0 },
            if a[11] > b[11] { 0xFF } else { 0 },
            if a[12] > b[12] { 0xFF } else { 0 },
            if a[13] > b[13] { 0xFF } else { 0 },
            if a[14] > b[14] { 0xFF } else { 0 },
            if a[15] > b[15] { 0xFF } else { 0 },
            if a[16] > b[16] { 0xFF } else { 0 },
            if a[17] > b[17] { 0xFF } else { 0 },
            if a[18] > b[18] { 0xFF } else { 0 },
            if a[19] > b[19] { 0xFF } else { 0 },
            if a[20] > b[20] { 0xFF } else { 0 },
            if a[21] > b[21] { 0xFF } else { 0 },
            if a[22] > b[22] { 0xFF } else { 0 },
            if a[23] > b[23] { 0xFF } else { 0 },
            if a[24] > b[24] { 0xFF } else { 0 },
            if a[25] > b[25] { 0xFF } else { 0 },
            if a[26] > b[26] { 0xFF } else { 0 },
            if a[27] > b[27] { 0xFF } else { 0 },
            if a[28] > b[28] { 0xFF } else { 0 },
            if a[29] > b[29] { 0xFF } else { 0 },
            if a[30] > b[30] { 0xFF } else { 0 },
            if a[31] > b[31] { 0xFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ge(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
        [
            if a[0] >= b[0] { 0xFF } else { 0 },
            if a[1] >= b[1] { 0xFF } else { 0 },
            if a[2] >= b[2] { 0xFF } else { 0 },
            if a[3] >= b[3] { 0xFF } else { 0 },
            if a[4] >= b[4] { 0xFF } else { 0 },
            if a[5] >= b[5] { 0xFF } else { 0 },
            if a[6] >= b[6] { 0xFF } else { 0 },
            if a[7] >= b[7] { 0xFF } else { 0 },
            if a[8] >= b[8] { 0xFF } else { 0 },
            if a[9] >= b[9] { 0xFF } else { 0 },
            if a[10] >= b[10] { 0xFF } else { 0 },
            if a[11] >= b[11] { 0xFF } else { 0 },
            if a[12] >= b[12] { 0xFF } else { 0 },
            if a[13] >= b[13] { 0xFF } else { 0 },
            if a[14] >= b[14] { 0xFF } else { 0 },
            if a[15] >= b[15] { 0xFF } else { 0 },
            if a[16] >= b[16] { 0xFF } else { 0 },
            if a[17] >= b[17] { 0xFF } else { 0 },
            if a[18] >= b[18] { 0xFF } else { 0 },
            if a[19] >= b[19] { 0xFF } else { 0 },
            if a[20] >= b[20] { 0xFF } else { 0 },
            if a[21] >= b[21] { 0xFF } else { 0 },
            if a[22] >= b[22] { 0xFF } else { 0 },
            if a[23] >= b[23] { 0xFF } else { 0 },
            if a[24] >= b[24] { 0xFF } else { 0 },
            if a[25] >= b[25] { 0xFF } else { 0 },
            if a[26] >= b[26] { 0xFF } else { 0 },
            if a[27] >= b[27] { 0xFF } else { 0 },
            if a[28] >= b[28] { 0xFF } else { 0 },
            if a[29] >= b[29] { 0xFF } else { 0 },
            if a[30] >= b[30] { 0xFF } else { 0 },
            if a[31] >= b[31] { 0xFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn blend(mask: [u8; 32], if_true: [u8; 32], if_false: [u8; 32]) -> [u8; 32] {
        [
            if mask[0] != 0 {
                if_true[0]
            } else {
                if_false[0]
            },
            if mask[1] != 0 {
                if_true[1]
            } else {
                if_false[1]
            },
            if mask[2] != 0 {
                if_true[2]
            } else {
                if_false[2]
            },
            if mask[3] != 0 {
                if_true[3]
            } else {
                if_false[3]
            },
            if mask[4] != 0 {
                if_true[4]
            } else {
                if_false[4]
            },
            if mask[5] != 0 {
                if_true[5]
            } else {
                if_false[5]
            },
            if mask[6] != 0 {
                if_true[6]
            } else {
                if_false[6]
            },
            if mask[7] != 0 {
                if_true[7]
            } else {
                if_false[7]
            },
            if mask[8] != 0 {
                if_true[8]
            } else {
                if_false[8]
            },
            if mask[9] != 0 {
                if_true[9]
            } else {
                if_false[9]
            },
            if mask[10] != 0 {
                if_true[10]
            } else {
                if_false[10]
            },
            if mask[11] != 0 {
                if_true[11]
            } else {
                if_false[11]
            },
            if mask[12] != 0 {
                if_true[12]
            } else {
                if_false[12]
            },
            if mask[13] != 0 {
                if_true[13]
            } else {
                if_false[13]
            },
            if mask[14] != 0 {
                if_true[14]
            } else {
                if_false[14]
            },
            if mask[15] != 0 {
                if_true[15]
            } else {
                if_false[15]
            },
            if mask[16] != 0 {
                if_true[16]
            } else {
                if_false[16]
            },
            if mask[17] != 0 {
                if_true[17]
            } else {
                if_false[17]
            },
            if mask[18] != 0 {
                if_true[18]
            } else {
                if_false[18]
            },
            if mask[19] != 0 {
                if_true[19]
            } else {
                if_false[19]
            },
            if mask[20] != 0 {
                if_true[20]
            } else {
                if_false[20]
            },
            if mask[21] != 0 {
                if_true[21]
            } else {
                if_false[21]
            },
            if mask[22] != 0 {
                if_true[22]
            } else {
                if_false[22]
            },
            if mask[23] != 0 {
                if_true[23]
            } else {
                if_false[23]
            },
            if mask[24] != 0 {
                if_true[24]
            } else {
                if_false[24]
            },
            if mask[25] != 0 {
                if_true[25]
            } else {
                if_false[25]
            },
            if mask[26] != 0 {
                if_true[26]
            } else {
                if_false[26]
            },
            if mask[27] != 0 {
                if_true[27]
            } else {
                if_false[27]
            },
            if mask[28] != 0 {
                if_true[28]
            } else {
                if_false[28]
            },
            if mask[29] != 0 {
                if_true[29]
            } else {
                if_false[29]
            },
            if mask[30] != 0 {
                if_true[30]
            } else {
                if_false[30]
            },
            if mask[31] != 0 {
                if_true[31]
            } else {
                if_false[31]
            },
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [u8; 32]) -> u8 {
        a.iter().copied().fold(0u8, u8::wrapping_add)
    }

    #[inline(always)]
    fn not(a: [u8; 32]) -> [u8; 32] {
        [
            !a[0], !a[1], !a[2], !a[3], !a[4], !a[5], !a[6], !a[7], !a[8], !a[9], !a[10], !a[11],
            !a[12], !a[13], !a[14], !a[15], !a[16], !a[17], !a[18], !a[19], !a[20], !a[21], !a[22],
            !a[23], !a[24], !a[25], !a[26], !a[27], !a[28], !a[29], !a[30], !a[31],
        ]
    }

    #[inline(always)]
    fn bitand(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
        [
            a[0] & b[0],
            a[1] & b[1],
            a[2] & b[2],
            a[3] & b[3],
            a[4] & b[4],
            a[5] & b[5],
            a[6] & b[6],
            a[7] & b[7],
            a[8] & b[8],
            a[9] & b[9],
            a[10] & b[10],
            a[11] & b[11],
            a[12] & b[12],
            a[13] & b[13],
            a[14] & b[14],
            a[15] & b[15],
            a[16] & b[16],
            a[17] & b[17],
            a[18] & b[18],
            a[19] & b[19],
            a[20] & b[20],
            a[21] & b[21],
            a[22] & b[22],
            a[23] & b[23],
            a[24] & b[24],
            a[25] & b[25],
            a[26] & b[26],
            a[27] & b[27],
            a[28] & b[28],
            a[29] & b[29],
            a[30] & b[30],
            a[31] & b[31],
        ]
    }

    #[inline(always)]
    fn bitor(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
        [
            a[0] | b[0],
            a[1] | b[1],
            a[2] | b[2],
            a[3] | b[3],
            a[4] | b[4],
            a[5] | b[5],
            a[6] | b[6],
            a[7] | b[7],
            a[8] | b[8],
            a[9] | b[9],
            a[10] | b[10],
            a[11] | b[11],
            a[12] | b[12],
            a[13] | b[13],
            a[14] | b[14],
            a[15] | b[15],
            a[16] | b[16],
            a[17] | b[17],
            a[18] | b[18],
            a[19] | b[19],
            a[20] | b[20],
            a[21] | b[21],
            a[22] | b[22],
            a[23] | b[23],
            a[24] | b[24],
            a[25] | b[25],
            a[26] | b[26],
            a[27] | b[27],
            a[28] | b[28],
            a[29] | b[29],
            a[30] | b[30],
            a[31] | b[31],
        ]
    }

    #[inline(always)]
    fn bitxor(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
        [
            a[0] ^ b[0],
            a[1] ^ b[1],
            a[2] ^ b[2],
            a[3] ^ b[3],
            a[4] ^ b[4],
            a[5] ^ b[5],
            a[6] ^ b[6],
            a[7] ^ b[7],
            a[8] ^ b[8],
            a[9] ^ b[9],
            a[10] ^ b[10],
            a[11] ^ b[11],
            a[12] ^ b[12],
            a[13] ^ b[13],
            a[14] ^ b[14],
            a[15] ^ b[15],
            a[16] ^ b[16],
            a[17] ^ b[17],
            a[18] ^ b[18],
            a[19] ^ b[19],
            a[20] ^ b[20],
            a[21] ^ b[21],
            a[22] ^ b[22],
            a[23] ^ b[23],
            a[24] ^ b[24],
            a[25] ^ b[25],
            a[26] ^ b[26],
            a[27] ^ b[27],
            a[28] ^ b[28],
            a[29] ^ b[29],
            a[30] ^ b[30],
            a[31] ^ b[31],
        ]
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [u8; 32]) -> [u8; 32] {
        [
            a[0].wrapping_shl(N as u32),
            a[1].wrapping_shl(N as u32),
            a[2].wrapping_shl(N as u32),
            a[3].wrapping_shl(N as u32),
            a[4].wrapping_shl(N as u32),
            a[5].wrapping_shl(N as u32),
            a[6].wrapping_shl(N as u32),
            a[7].wrapping_shl(N as u32),
            a[8].wrapping_shl(N as u32),
            a[9].wrapping_shl(N as u32),
            a[10].wrapping_shl(N as u32),
            a[11].wrapping_shl(N as u32),
            a[12].wrapping_shl(N as u32),
            a[13].wrapping_shl(N as u32),
            a[14].wrapping_shl(N as u32),
            a[15].wrapping_shl(N as u32),
            a[16].wrapping_shl(N as u32),
            a[17].wrapping_shl(N as u32),
            a[18].wrapping_shl(N as u32),
            a[19].wrapping_shl(N as u32),
            a[20].wrapping_shl(N as u32),
            a[21].wrapping_shl(N as u32),
            a[22].wrapping_shl(N as u32),
            a[23].wrapping_shl(N as u32),
            a[24].wrapping_shl(N as u32),
            a[25].wrapping_shl(N as u32),
            a[26].wrapping_shl(N as u32),
            a[27].wrapping_shl(N as u32),
            a[28].wrapping_shl(N as u32),
            a[29].wrapping_shl(N as u32),
            a[30].wrapping_shl(N as u32),
            a[31].wrapping_shl(N as u32),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [u8; 32]) -> [u8; 32] {
        [
            a[0].wrapping_shr(N as u32),
            a[1].wrapping_shr(N as u32),
            a[2].wrapping_shr(N as u32),
            a[3].wrapping_shr(N as u32),
            a[4].wrapping_shr(N as u32),
            a[5].wrapping_shr(N as u32),
            a[6].wrapping_shr(N as u32),
            a[7].wrapping_shr(N as u32),
            a[8].wrapping_shr(N as u32),
            a[9].wrapping_shr(N as u32),
            a[10].wrapping_shr(N as u32),
            a[11].wrapping_shr(N as u32),
            a[12].wrapping_shr(N as u32),
            a[13].wrapping_shr(N as u32),
            a[14].wrapping_shr(N as u32),
            a[15].wrapping_shr(N as u32),
            a[16].wrapping_shr(N as u32),
            a[17].wrapping_shr(N as u32),
            a[18].wrapping_shr(N as u32),
            a[19].wrapping_shr(N as u32),
            a[20].wrapping_shr(N as u32),
            a[21].wrapping_shr(N as u32),
            a[22].wrapping_shr(N as u32),
            a[23].wrapping_shr(N as u32),
            a[24].wrapping_shr(N as u32),
            a[25].wrapping_shr(N as u32),
            a[26].wrapping_shr(N as u32),
            a[27].wrapping_shr(N as u32),
            a[28].wrapping_shr(N as u32),
            a[29].wrapping_shr(N as u32),
            a[30].wrapping_shr(N as u32),
            a[31].wrapping_shr(N as u32),
        ]
    }

    #[inline(always)]
    fn all_true(a: [u8; 32]) -> bool {
        a[0] != 0
            && a[1] != 0
            && a[2] != 0
            && a[3] != 0
            && a[4] != 0
            && a[5] != 0
            && a[6] != 0
            && a[7] != 0
            && a[8] != 0
            && a[9] != 0
            && a[10] != 0
            && a[11] != 0
            && a[12] != 0
            && a[13] != 0
            && a[14] != 0
            && a[15] != 0
            && a[16] != 0
            && a[17] != 0
            && a[18] != 0
            && a[19] != 0
            && a[20] != 0
            && a[21] != 0
            && a[22] != 0
            && a[23] != 0
            && a[24] != 0
            && a[25] != 0
            && a[26] != 0
            && a[27] != 0
            && a[28] != 0
            && a[29] != 0
            && a[30] != 0
            && a[31] != 0
    }

    #[inline(always)]
    fn any_true(a: [u8; 32]) -> bool {
        a[0] != 0
            || a[1] != 0
            || a[2] != 0
            || a[3] != 0
            || a[4] != 0
            || a[5] != 0
            || a[6] != 0
            || a[7] != 0
            || a[8] != 0
            || a[9] != 0
            || a[10] != 0
            || a[11] != 0
            || a[12] != 0
            || a[13] != 0
            || a[14] != 0
            || a[15] != 0
            || a[16] != 0
            || a[17] != 0
            || a[18] != 0
            || a[19] != 0
            || a[20] != 0
            || a[21] != 0
            || a[22] != 0
            || a[23] != 0
            || a[24] != 0
            || a[25] != 0
            || a[26] != 0
            || a[27] != 0
            || a[28] != 0
            || a[29] != 0
            || a[30] != 0
            || a[31] != 0
    }

    #[inline(always)]
    fn bitmask(a: [u8; 32]) -> u32 {
        ((a[0] >> 7) as u32 & 1) << 0
            | ((a[1] >> 7) as u32 & 1) << 1
            | ((a[2] >> 7) as u32 & 1) << 2
            | ((a[3] >> 7) as u32 & 1) << 3
            | ((a[4] >> 7) as u32 & 1) << 4
            | ((a[5] >> 7) as u32 & 1) << 5
            | ((a[6] >> 7) as u32 & 1) << 6
            | ((a[7] >> 7) as u32 & 1) << 7
            | ((a[8] >> 7) as u32 & 1) << 8
            | ((a[9] >> 7) as u32 & 1) << 9
            | ((a[10] >> 7) as u32 & 1) << 10
            | ((a[11] >> 7) as u32 & 1) << 11
            | ((a[12] >> 7) as u32 & 1) << 12
            | ((a[13] >> 7) as u32 & 1) << 13
            | ((a[14] >> 7) as u32 & 1) << 14
            | ((a[15] >> 7) as u32 & 1) << 15
            | ((a[16] >> 7) as u32 & 1) << 16
            | ((a[17] >> 7) as u32 & 1) << 17
            | ((a[18] >> 7) as u32 & 1) << 18
            | ((a[19] >> 7) as u32 & 1) << 19
            | ((a[20] >> 7) as u32 & 1) << 20
            | ((a[21] >> 7) as u32 & 1) << 21
            | ((a[22] >> 7) as u32 & 1) << 22
            | ((a[23] >> 7) as u32 & 1) << 23
            | ((a[24] >> 7) as u32 & 1) << 24
            | ((a[25] >> 7) as u32 & 1) << 25
            | ((a[26] >> 7) as u32 & 1) << 26
            | ((a[27] >> 7) as u32 & 1) << 27
            | ((a[28] >> 7) as u32 & 1) << 28
            | ((a[29] >> 7) as u32 & 1) << 29
            | ((a[30] >> 7) as u32 & 1) << 30
            | ((a[31] >> 7) as u32 & 1) << 31
    }
}

impl I16x8Backend for archmage::ScalarToken {
    type Repr = [i16; 8];

    #[inline(always)]
    fn splat(v: i16) -> [i16; 8] {
        [v, v, v, v, v, v, v, v]
    }

    #[inline(always)]
    fn zero() -> [i16; 8] {
        [0i16; 8]
    }

    #[inline(always)]
    fn load(data: &[i16; 8]) -> [i16; 8] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [i16; 8]) -> [i16; 8] {
        arr
    }

    #[inline(always)]
    fn store(repr: [i16; 8], out: &mut [i16; 8]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [i16; 8]) -> [i16; 8] {
        repr
    }

    #[inline(always)]
    fn add(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
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
    fn sub(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
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
    fn mul(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
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
    fn neg(a: [i16; 8]) -> [i16; 8] {
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

    #[inline(always)]
    fn min(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
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
    fn max(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
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
    fn abs(a: [i16; 8]) -> [i16; 8] {
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

    #[inline(always)]
    fn simd_eq(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
        [
            if a[0] == b[0] { -1 } else { 0 },
            if a[1] == b[1] { -1 } else { 0 },
            if a[2] == b[2] { -1 } else { 0 },
            if a[3] == b[3] { -1 } else { 0 },
            if a[4] == b[4] { -1 } else { 0 },
            if a[5] == b[5] { -1 } else { 0 },
            if a[6] == b[6] { -1 } else { 0 },
            if a[7] == b[7] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ne(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
        [
            if a[0] != b[0] { -1 } else { 0 },
            if a[1] != b[1] { -1 } else { 0 },
            if a[2] != b[2] { -1 } else { 0 },
            if a[3] != b[3] { -1 } else { 0 },
            if a[4] != b[4] { -1 } else { 0 },
            if a[5] != b[5] { -1 } else { 0 },
            if a[6] != b[6] { -1 } else { 0 },
            if a[7] != b[7] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_lt(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
        [
            if a[0] < b[0] { -1 } else { 0 },
            if a[1] < b[1] { -1 } else { 0 },
            if a[2] < b[2] { -1 } else { 0 },
            if a[3] < b[3] { -1 } else { 0 },
            if a[4] < b[4] { -1 } else { 0 },
            if a[5] < b[5] { -1 } else { 0 },
            if a[6] < b[6] { -1 } else { 0 },
            if a[7] < b[7] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_le(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
        [
            if a[0] <= b[0] { -1 } else { 0 },
            if a[1] <= b[1] { -1 } else { 0 },
            if a[2] <= b[2] { -1 } else { 0 },
            if a[3] <= b[3] { -1 } else { 0 },
            if a[4] <= b[4] { -1 } else { 0 },
            if a[5] <= b[5] { -1 } else { 0 },
            if a[6] <= b[6] { -1 } else { 0 },
            if a[7] <= b[7] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_gt(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
        [
            if a[0] > b[0] { -1 } else { 0 },
            if a[1] > b[1] { -1 } else { 0 },
            if a[2] > b[2] { -1 } else { 0 },
            if a[3] > b[3] { -1 } else { 0 },
            if a[4] > b[4] { -1 } else { 0 },
            if a[5] > b[5] { -1 } else { 0 },
            if a[6] > b[6] { -1 } else { 0 },
            if a[7] > b[7] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ge(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
        [
            if a[0] >= b[0] { -1 } else { 0 },
            if a[1] >= b[1] { -1 } else { 0 },
            if a[2] >= b[2] { -1 } else { 0 },
            if a[3] >= b[3] { -1 } else { 0 },
            if a[4] >= b[4] { -1 } else { 0 },
            if a[5] >= b[5] { -1 } else { 0 },
            if a[6] >= b[6] { -1 } else { 0 },
            if a[7] >= b[7] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn blend(mask: [i16; 8], if_true: [i16; 8], if_false: [i16; 8]) -> [i16; 8] {
        [
            if mask[0] != 0 {
                if_true[0]
            } else {
                if_false[0]
            },
            if mask[1] != 0 {
                if_true[1]
            } else {
                if_false[1]
            },
            if mask[2] != 0 {
                if_true[2]
            } else {
                if_false[2]
            },
            if mask[3] != 0 {
                if_true[3]
            } else {
                if_false[3]
            },
            if mask[4] != 0 {
                if_true[4]
            } else {
                if_false[4]
            },
            if mask[5] != 0 {
                if_true[5]
            } else {
                if_false[5]
            },
            if mask[6] != 0 {
                if_true[6]
            } else {
                if_false[6]
            },
            if mask[7] != 0 {
                if_true[7]
            } else {
                if_false[7]
            },
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [i16; 8]) -> i16 {
        a.iter().copied().fold(0i16, i16::wrapping_add)
    }

    #[inline(always)]
    fn not(a: [i16; 8]) -> [i16; 8] {
        [!a[0], !a[1], !a[2], !a[3], !a[4], !a[5], !a[6], !a[7]]
    }

    #[inline(always)]
    fn bitand(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
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
    fn bitor(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
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
    fn bitxor(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
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

    #[inline(always)]
    fn shl_const<const N: i32>(a: [i16; 8]) -> [i16; 8] {
        [
            a[0].wrapping_shl(N as u32),
            a[1].wrapping_shl(N as u32),
            a[2].wrapping_shl(N as u32),
            a[3].wrapping_shl(N as u32),
            a[4].wrapping_shl(N as u32),
            a[5].wrapping_shl(N as u32),
            a[6].wrapping_shl(N as u32),
            a[7].wrapping_shl(N as u32),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [i16; 8]) -> [i16; 8] {
        [
            (a[0] as u16).wrapping_shr(N as u32) as i16,
            (a[1] as u16).wrapping_shr(N as u32) as i16,
            (a[2] as u16).wrapping_shr(N as u32) as i16,
            (a[3] as u16).wrapping_shr(N as u32) as i16,
            (a[4] as u16).wrapping_shr(N as u32) as i16,
            (a[5] as u16).wrapping_shr(N as u32) as i16,
            (a[6] as u16).wrapping_shr(N as u32) as i16,
            (a[7] as u16).wrapping_shr(N as u32) as i16,
        ]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [i16; 8]) -> [i16; 8] {
        [
            a[0].wrapping_shr(N as u32),
            a[1].wrapping_shr(N as u32),
            a[2].wrapping_shr(N as u32),
            a[3].wrapping_shr(N as u32),
            a[4].wrapping_shr(N as u32),
            a[5].wrapping_shr(N as u32),
            a[6].wrapping_shr(N as u32),
            a[7].wrapping_shr(N as u32),
        ]
    }

    #[inline(always)]
    fn all_true(a: [i16; 8]) -> bool {
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
    fn any_true(a: [i16; 8]) -> bool {
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
    fn bitmask(a: [i16; 8]) -> u32 {
        ((a[0] >> 15) as u32 & 1) << 0
            | ((a[1] >> 15) as u32 & 1) << 1
            | ((a[2] >> 15) as u32 & 1) << 2
            | ((a[3] >> 15) as u32 & 1) << 3
            | ((a[4] >> 15) as u32 & 1) << 4
            | ((a[5] >> 15) as u32 & 1) << 5
            | ((a[6] >> 15) as u32 & 1) << 6
            | ((a[7] >> 15) as u32 & 1) << 7
    }
}

impl I16x16Backend for archmage::ScalarToken {
    type Repr = [i16; 16];

    #[inline(always)]
    fn splat(v: i16) -> [i16; 16] {
        [v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v]
    }

    #[inline(always)]
    fn zero() -> [i16; 16] {
        [0i16; 16]
    }

    #[inline(always)]
    fn load(data: &[i16; 16]) -> [i16; 16] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [i16; 16]) -> [i16; 16] {
        arr
    }

    #[inline(always)]
    fn store(repr: [i16; 16], out: &mut [i16; 16]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [i16; 16]) -> [i16; 16] {
        repr
    }

    #[inline(always)]
    fn add(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
        [
            a[0].wrapping_add(b[0]),
            a[1].wrapping_add(b[1]),
            a[2].wrapping_add(b[2]),
            a[3].wrapping_add(b[3]),
            a[4].wrapping_add(b[4]),
            a[5].wrapping_add(b[5]),
            a[6].wrapping_add(b[6]),
            a[7].wrapping_add(b[7]),
            a[8].wrapping_add(b[8]),
            a[9].wrapping_add(b[9]),
            a[10].wrapping_add(b[10]),
            a[11].wrapping_add(b[11]),
            a[12].wrapping_add(b[12]),
            a[13].wrapping_add(b[13]),
            a[14].wrapping_add(b[14]),
            a[15].wrapping_add(b[15]),
        ]
    }

    #[inline(always)]
    fn sub(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
        [
            a[0].wrapping_sub(b[0]),
            a[1].wrapping_sub(b[1]),
            a[2].wrapping_sub(b[2]),
            a[3].wrapping_sub(b[3]),
            a[4].wrapping_sub(b[4]),
            a[5].wrapping_sub(b[5]),
            a[6].wrapping_sub(b[6]),
            a[7].wrapping_sub(b[7]),
            a[8].wrapping_sub(b[8]),
            a[9].wrapping_sub(b[9]),
            a[10].wrapping_sub(b[10]),
            a[11].wrapping_sub(b[11]),
            a[12].wrapping_sub(b[12]),
            a[13].wrapping_sub(b[13]),
            a[14].wrapping_sub(b[14]),
            a[15].wrapping_sub(b[15]),
        ]
    }

    #[inline(always)]
    fn mul(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
        [
            a[0].wrapping_mul(b[0]),
            a[1].wrapping_mul(b[1]),
            a[2].wrapping_mul(b[2]),
            a[3].wrapping_mul(b[3]),
            a[4].wrapping_mul(b[4]),
            a[5].wrapping_mul(b[5]),
            a[6].wrapping_mul(b[6]),
            a[7].wrapping_mul(b[7]),
            a[8].wrapping_mul(b[8]),
            a[9].wrapping_mul(b[9]),
            a[10].wrapping_mul(b[10]),
            a[11].wrapping_mul(b[11]),
            a[12].wrapping_mul(b[12]),
            a[13].wrapping_mul(b[13]),
            a[14].wrapping_mul(b[14]),
            a[15].wrapping_mul(b[15]),
        ]
    }

    #[inline(always)]
    fn neg(a: [i16; 16]) -> [i16; 16] {
        [
            a[0].wrapping_neg(),
            a[1].wrapping_neg(),
            a[2].wrapping_neg(),
            a[3].wrapping_neg(),
            a[4].wrapping_neg(),
            a[5].wrapping_neg(),
            a[6].wrapping_neg(),
            a[7].wrapping_neg(),
            a[8].wrapping_neg(),
            a[9].wrapping_neg(),
            a[10].wrapping_neg(),
            a[11].wrapping_neg(),
            a[12].wrapping_neg(),
            a[13].wrapping_neg(),
            a[14].wrapping_neg(),
            a[15].wrapping_neg(),
        ]
    }

    #[inline(always)]
    fn min(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
        [
            if a[0] < b[0] { a[0] } else { b[0] },
            if a[1] < b[1] { a[1] } else { b[1] },
            if a[2] < b[2] { a[2] } else { b[2] },
            if a[3] < b[3] { a[3] } else { b[3] },
            if a[4] < b[4] { a[4] } else { b[4] },
            if a[5] < b[5] { a[5] } else { b[5] },
            if a[6] < b[6] { a[6] } else { b[6] },
            if a[7] < b[7] { a[7] } else { b[7] },
            if a[8] < b[8] { a[8] } else { b[8] },
            if a[9] < b[9] { a[9] } else { b[9] },
            if a[10] < b[10] { a[10] } else { b[10] },
            if a[11] < b[11] { a[11] } else { b[11] },
            if a[12] < b[12] { a[12] } else { b[12] },
            if a[13] < b[13] { a[13] } else { b[13] },
            if a[14] < b[14] { a[14] } else { b[14] },
            if a[15] < b[15] { a[15] } else { b[15] },
        ]
    }

    #[inline(always)]
    fn max(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
        [
            if a[0] > b[0] { a[0] } else { b[0] },
            if a[1] > b[1] { a[1] } else { b[1] },
            if a[2] > b[2] { a[2] } else { b[2] },
            if a[3] > b[3] { a[3] } else { b[3] },
            if a[4] > b[4] { a[4] } else { b[4] },
            if a[5] > b[5] { a[5] } else { b[5] },
            if a[6] > b[6] { a[6] } else { b[6] },
            if a[7] > b[7] { a[7] } else { b[7] },
            if a[8] > b[8] { a[8] } else { b[8] },
            if a[9] > b[9] { a[9] } else { b[9] },
            if a[10] > b[10] { a[10] } else { b[10] },
            if a[11] > b[11] { a[11] } else { b[11] },
            if a[12] > b[12] { a[12] } else { b[12] },
            if a[13] > b[13] { a[13] } else { b[13] },
            if a[14] > b[14] { a[14] } else { b[14] },
            if a[15] > b[15] { a[15] } else { b[15] },
        ]
    }

    #[inline(always)]
    fn abs(a: [i16; 16]) -> [i16; 16] {
        [
            a[0].wrapping_abs(),
            a[1].wrapping_abs(),
            a[2].wrapping_abs(),
            a[3].wrapping_abs(),
            a[4].wrapping_abs(),
            a[5].wrapping_abs(),
            a[6].wrapping_abs(),
            a[7].wrapping_abs(),
            a[8].wrapping_abs(),
            a[9].wrapping_abs(),
            a[10].wrapping_abs(),
            a[11].wrapping_abs(),
            a[12].wrapping_abs(),
            a[13].wrapping_abs(),
            a[14].wrapping_abs(),
            a[15].wrapping_abs(),
        ]
    }

    #[inline(always)]
    fn simd_eq(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
        [
            if a[0] == b[0] { -1 } else { 0 },
            if a[1] == b[1] { -1 } else { 0 },
            if a[2] == b[2] { -1 } else { 0 },
            if a[3] == b[3] { -1 } else { 0 },
            if a[4] == b[4] { -1 } else { 0 },
            if a[5] == b[5] { -1 } else { 0 },
            if a[6] == b[6] { -1 } else { 0 },
            if a[7] == b[7] { -1 } else { 0 },
            if a[8] == b[8] { -1 } else { 0 },
            if a[9] == b[9] { -1 } else { 0 },
            if a[10] == b[10] { -1 } else { 0 },
            if a[11] == b[11] { -1 } else { 0 },
            if a[12] == b[12] { -1 } else { 0 },
            if a[13] == b[13] { -1 } else { 0 },
            if a[14] == b[14] { -1 } else { 0 },
            if a[15] == b[15] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ne(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
        [
            if a[0] != b[0] { -1 } else { 0 },
            if a[1] != b[1] { -1 } else { 0 },
            if a[2] != b[2] { -1 } else { 0 },
            if a[3] != b[3] { -1 } else { 0 },
            if a[4] != b[4] { -1 } else { 0 },
            if a[5] != b[5] { -1 } else { 0 },
            if a[6] != b[6] { -1 } else { 0 },
            if a[7] != b[7] { -1 } else { 0 },
            if a[8] != b[8] { -1 } else { 0 },
            if a[9] != b[9] { -1 } else { 0 },
            if a[10] != b[10] { -1 } else { 0 },
            if a[11] != b[11] { -1 } else { 0 },
            if a[12] != b[12] { -1 } else { 0 },
            if a[13] != b[13] { -1 } else { 0 },
            if a[14] != b[14] { -1 } else { 0 },
            if a[15] != b[15] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_lt(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
        [
            if a[0] < b[0] { -1 } else { 0 },
            if a[1] < b[1] { -1 } else { 0 },
            if a[2] < b[2] { -1 } else { 0 },
            if a[3] < b[3] { -1 } else { 0 },
            if a[4] < b[4] { -1 } else { 0 },
            if a[5] < b[5] { -1 } else { 0 },
            if a[6] < b[6] { -1 } else { 0 },
            if a[7] < b[7] { -1 } else { 0 },
            if a[8] < b[8] { -1 } else { 0 },
            if a[9] < b[9] { -1 } else { 0 },
            if a[10] < b[10] { -1 } else { 0 },
            if a[11] < b[11] { -1 } else { 0 },
            if a[12] < b[12] { -1 } else { 0 },
            if a[13] < b[13] { -1 } else { 0 },
            if a[14] < b[14] { -1 } else { 0 },
            if a[15] < b[15] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_le(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
        [
            if a[0] <= b[0] { -1 } else { 0 },
            if a[1] <= b[1] { -1 } else { 0 },
            if a[2] <= b[2] { -1 } else { 0 },
            if a[3] <= b[3] { -1 } else { 0 },
            if a[4] <= b[4] { -1 } else { 0 },
            if a[5] <= b[5] { -1 } else { 0 },
            if a[6] <= b[6] { -1 } else { 0 },
            if a[7] <= b[7] { -1 } else { 0 },
            if a[8] <= b[8] { -1 } else { 0 },
            if a[9] <= b[9] { -1 } else { 0 },
            if a[10] <= b[10] { -1 } else { 0 },
            if a[11] <= b[11] { -1 } else { 0 },
            if a[12] <= b[12] { -1 } else { 0 },
            if a[13] <= b[13] { -1 } else { 0 },
            if a[14] <= b[14] { -1 } else { 0 },
            if a[15] <= b[15] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_gt(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
        [
            if a[0] > b[0] { -1 } else { 0 },
            if a[1] > b[1] { -1 } else { 0 },
            if a[2] > b[2] { -1 } else { 0 },
            if a[3] > b[3] { -1 } else { 0 },
            if a[4] > b[4] { -1 } else { 0 },
            if a[5] > b[5] { -1 } else { 0 },
            if a[6] > b[6] { -1 } else { 0 },
            if a[7] > b[7] { -1 } else { 0 },
            if a[8] > b[8] { -1 } else { 0 },
            if a[9] > b[9] { -1 } else { 0 },
            if a[10] > b[10] { -1 } else { 0 },
            if a[11] > b[11] { -1 } else { 0 },
            if a[12] > b[12] { -1 } else { 0 },
            if a[13] > b[13] { -1 } else { 0 },
            if a[14] > b[14] { -1 } else { 0 },
            if a[15] > b[15] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ge(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
        [
            if a[0] >= b[0] { -1 } else { 0 },
            if a[1] >= b[1] { -1 } else { 0 },
            if a[2] >= b[2] { -1 } else { 0 },
            if a[3] >= b[3] { -1 } else { 0 },
            if a[4] >= b[4] { -1 } else { 0 },
            if a[5] >= b[5] { -1 } else { 0 },
            if a[6] >= b[6] { -1 } else { 0 },
            if a[7] >= b[7] { -1 } else { 0 },
            if a[8] >= b[8] { -1 } else { 0 },
            if a[9] >= b[9] { -1 } else { 0 },
            if a[10] >= b[10] { -1 } else { 0 },
            if a[11] >= b[11] { -1 } else { 0 },
            if a[12] >= b[12] { -1 } else { 0 },
            if a[13] >= b[13] { -1 } else { 0 },
            if a[14] >= b[14] { -1 } else { 0 },
            if a[15] >= b[15] { -1 } else { 0 },
        ]
    }

    #[inline(always)]
    fn blend(mask: [i16; 16], if_true: [i16; 16], if_false: [i16; 16]) -> [i16; 16] {
        [
            if mask[0] != 0 {
                if_true[0]
            } else {
                if_false[0]
            },
            if mask[1] != 0 {
                if_true[1]
            } else {
                if_false[1]
            },
            if mask[2] != 0 {
                if_true[2]
            } else {
                if_false[2]
            },
            if mask[3] != 0 {
                if_true[3]
            } else {
                if_false[3]
            },
            if mask[4] != 0 {
                if_true[4]
            } else {
                if_false[4]
            },
            if mask[5] != 0 {
                if_true[5]
            } else {
                if_false[5]
            },
            if mask[6] != 0 {
                if_true[6]
            } else {
                if_false[6]
            },
            if mask[7] != 0 {
                if_true[7]
            } else {
                if_false[7]
            },
            if mask[8] != 0 {
                if_true[8]
            } else {
                if_false[8]
            },
            if mask[9] != 0 {
                if_true[9]
            } else {
                if_false[9]
            },
            if mask[10] != 0 {
                if_true[10]
            } else {
                if_false[10]
            },
            if mask[11] != 0 {
                if_true[11]
            } else {
                if_false[11]
            },
            if mask[12] != 0 {
                if_true[12]
            } else {
                if_false[12]
            },
            if mask[13] != 0 {
                if_true[13]
            } else {
                if_false[13]
            },
            if mask[14] != 0 {
                if_true[14]
            } else {
                if_false[14]
            },
            if mask[15] != 0 {
                if_true[15]
            } else {
                if_false[15]
            },
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [i16; 16]) -> i16 {
        a.iter().copied().fold(0i16, i16::wrapping_add)
    }

    #[inline(always)]
    fn not(a: [i16; 16]) -> [i16; 16] {
        [
            !a[0], !a[1], !a[2], !a[3], !a[4], !a[5], !a[6], !a[7], !a[8], !a[9], !a[10], !a[11],
            !a[12], !a[13], !a[14], !a[15],
        ]
    }

    #[inline(always)]
    fn bitand(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
        [
            a[0] & b[0],
            a[1] & b[1],
            a[2] & b[2],
            a[3] & b[3],
            a[4] & b[4],
            a[5] & b[5],
            a[6] & b[6],
            a[7] & b[7],
            a[8] & b[8],
            a[9] & b[9],
            a[10] & b[10],
            a[11] & b[11],
            a[12] & b[12],
            a[13] & b[13],
            a[14] & b[14],
            a[15] & b[15],
        ]
    }

    #[inline(always)]
    fn bitor(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
        [
            a[0] | b[0],
            a[1] | b[1],
            a[2] | b[2],
            a[3] | b[3],
            a[4] | b[4],
            a[5] | b[5],
            a[6] | b[6],
            a[7] | b[7],
            a[8] | b[8],
            a[9] | b[9],
            a[10] | b[10],
            a[11] | b[11],
            a[12] | b[12],
            a[13] | b[13],
            a[14] | b[14],
            a[15] | b[15],
        ]
    }

    #[inline(always)]
    fn bitxor(a: [i16; 16], b: [i16; 16]) -> [i16; 16] {
        [
            a[0] ^ b[0],
            a[1] ^ b[1],
            a[2] ^ b[2],
            a[3] ^ b[3],
            a[4] ^ b[4],
            a[5] ^ b[5],
            a[6] ^ b[6],
            a[7] ^ b[7],
            a[8] ^ b[8],
            a[9] ^ b[9],
            a[10] ^ b[10],
            a[11] ^ b[11],
            a[12] ^ b[12],
            a[13] ^ b[13],
            a[14] ^ b[14],
            a[15] ^ b[15],
        ]
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [i16; 16]) -> [i16; 16] {
        [
            a[0].wrapping_shl(N as u32),
            a[1].wrapping_shl(N as u32),
            a[2].wrapping_shl(N as u32),
            a[3].wrapping_shl(N as u32),
            a[4].wrapping_shl(N as u32),
            a[5].wrapping_shl(N as u32),
            a[6].wrapping_shl(N as u32),
            a[7].wrapping_shl(N as u32),
            a[8].wrapping_shl(N as u32),
            a[9].wrapping_shl(N as u32),
            a[10].wrapping_shl(N as u32),
            a[11].wrapping_shl(N as u32),
            a[12].wrapping_shl(N as u32),
            a[13].wrapping_shl(N as u32),
            a[14].wrapping_shl(N as u32),
            a[15].wrapping_shl(N as u32),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [i16; 16]) -> [i16; 16] {
        [
            (a[0] as u16).wrapping_shr(N as u32) as i16,
            (a[1] as u16).wrapping_shr(N as u32) as i16,
            (a[2] as u16).wrapping_shr(N as u32) as i16,
            (a[3] as u16).wrapping_shr(N as u32) as i16,
            (a[4] as u16).wrapping_shr(N as u32) as i16,
            (a[5] as u16).wrapping_shr(N as u32) as i16,
            (a[6] as u16).wrapping_shr(N as u32) as i16,
            (a[7] as u16).wrapping_shr(N as u32) as i16,
            (a[8] as u16).wrapping_shr(N as u32) as i16,
            (a[9] as u16).wrapping_shr(N as u32) as i16,
            (a[10] as u16).wrapping_shr(N as u32) as i16,
            (a[11] as u16).wrapping_shr(N as u32) as i16,
            (a[12] as u16).wrapping_shr(N as u32) as i16,
            (a[13] as u16).wrapping_shr(N as u32) as i16,
            (a[14] as u16).wrapping_shr(N as u32) as i16,
            (a[15] as u16).wrapping_shr(N as u32) as i16,
        ]
    }

    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(a: [i16; 16]) -> [i16; 16] {
        [
            a[0].wrapping_shr(N as u32),
            a[1].wrapping_shr(N as u32),
            a[2].wrapping_shr(N as u32),
            a[3].wrapping_shr(N as u32),
            a[4].wrapping_shr(N as u32),
            a[5].wrapping_shr(N as u32),
            a[6].wrapping_shr(N as u32),
            a[7].wrapping_shr(N as u32),
            a[8].wrapping_shr(N as u32),
            a[9].wrapping_shr(N as u32),
            a[10].wrapping_shr(N as u32),
            a[11].wrapping_shr(N as u32),
            a[12].wrapping_shr(N as u32),
            a[13].wrapping_shr(N as u32),
            a[14].wrapping_shr(N as u32),
            a[15].wrapping_shr(N as u32),
        ]
    }

    #[inline(always)]
    fn all_true(a: [i16; 16]) -> bool {
        a[0] != 0
            && a[1] != 0
            && a[2] != 0
            && a[3] != 0
            && a[4] != 0
            && a[5] != 0
            && a[6] != 0
            && a[7] != 0
            && a[8] != 0
            && a[9] != 0
            && a[10] != 0
            && a[11] != 0
            && a[12] != 0
            && a[13] != 0
            && a[14] != 0
            && a[15] != 0
    }

    #[inline(always)]
    fn any_true(a: [i16; 16]) -> bool {
        a[0] != 0
            || a[1] != 0
            || a[2] != 0
            || a[3] != 0
            || a[4] != 0
            || a[5] != 0
            || a[6] != 0
            || a[7] != 0
            || a[8] != 0
            || a[9] != 0
            || a[10] != 0
            || a[11] != 0
            || a[12] != 0
            || a[13] != 0
            || a[14] != 0
            || a[15] != 0
    }

    #[inline(always)]
    fn bitmask(a: [i16; 16]) -> u32 {
        ((a[0] >> 15) as u32 & 1) << 0
            | ((a[1] >> 15) as u32 & 1) << 1
            | ((a[2] >> 15) as u32 & 1) << 2
            | ((a[3] >> 15) as u32 & 1) << 3
            | ((a[4] >> 15) as u32 & 1) << 4
            | ((a[5] >> 15) as u32 & 1) << 5
            | ((a[6] >> 15) as u32 & 1) << 6
            | ((a[7] >> 15) as u32 & 1) << 7
            | ((a[8] >> 15) as u32 & 1) << 8
            | ((a[9] >> 15) as u32 & 1) << 9
            | ((a[10] >> 15) as u32 & 1) << 10
            | ((a[11] >> 15) as u32 & 1) << 11
            | ((a[12] >> 15) as u32 & 1) << 12
            | ((a[13] >> 15) as u32 & 1) << 13
            | ((a[14] >> 15) as u32 & 1) << 14
            | ((a[15] >> 15) as u32 & 1) << 15
    }
}

impl U16x8Backend for archmage::ScalarToken {
    type Repr = [u16; 8];

    #[inline(always)]
    fn splat(v: u16) -> [u16; 8] {
        [v, v, v, v, v, v, v, v]
    }

    #[inline(always)]
    fn zero() -> [u16; 8] {
        [0u16; 8]
    }

    #[inline(always)]
    fn load(data: &[u16; 8]) -> [u16; 8] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [u16; 8]) -> [u16; 8] {
        arr
    }

    #[inline(always)]
    fn store(repr: [u16; 8], out: &mut [u16; 8]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [u16; 8]) -> [u16; 8] {
        repr
    }

    #[inline(always)]
    fn add(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
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
    fn sub(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
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
    fn mul(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
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
    fn min(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
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
    fn max(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
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
    fn simd_eq(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
        [
            if a[0] == b[0] { 0xFFFF } else { 0 },
            if a[1] == b[1] { 0xFFFF } else { 0 },
            if a[2] == b[2] { 0xFFFF } else { 0 },
            if a[3] == b[3] { 0xFFFF } else { 0 },
            if a[4] == b[4] { 0xFFFF } else { 0 },
            if a[5] == b[5] { 0xFFFF } else { 0 },
            if a[6] == b[6] { 0xFFFF } else { 0 },
            if a[7] == b[7] { 0xFFFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ne(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
        [
            if a[0] != b[0] { 0xFFFF } else { 0 },
            if a[1] != b[1] { 0xFFFF } else { 0 },
            if a[2] != b[2] { 0xFFFF } else { 0 },
            if a[3] != b[3] { 0xFFFF } else { 0 },
            if a[4] != b[4] { 0xFFFF } else { 0 },
            if a[5] != b[5] { 0xFFFF } else { 0 },
            if a[6] != b[6] { 0xFFFF } else { 0 },
            if a[7] != b[7] { 0xFFFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_lt(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
        [
            if a[0] < b[0] { 0xFFFF } else { 0 },
            if a[1] < b[1] { 0xFFFF } else { 0 },
            if a[2] < b[2] { 0xFFFF } else { 0 },
            if a[3] < b[3] { 0xFFFF } else { 0 },
            if a[4] < b[4] { 0xFFFF } else { 0 },
            if a[5] < b[5] { 0xFFFF } else { 0 },
            if a[6] < b[6] { 0xFFFF } else { 0 },
            if a[7] < b[7] { 0xFFFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_le(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
        [
            if a[0] <= b[0] { 0xFFFF } else { 0 },
            if a[1] <= b[1] { 0xFFFF } else { 0 },
            if a[2] <= b[2] { 0xFFFF } else { 0 },
            if a[3] <= b[3] { 0xFFFF } else { 0 },
            if a[4] <= b[4] { 0xFFFF } else { 0 },
            if a[5] <= b[5] { 0xFFFF } else { 0 },
            if a[6] <= b[6] { 0xFFFF } else { 0 },
            if a[7] <= b[7] { 0xFFFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_gt(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
        [
            if a[0] > b[0] { 0xFFFF } else { 0 },
            if a[1] > b[1] { 0xFFFF } else { 0 },
            if a[2] > b[2] { 0xFFFF } else { 0 },
            if a[3] > b[3] { 0xFFFF } else { 0 },
            if a[4] > b[4] { 0xFFFF } else { 0 },
            if a[5] > b[5] { 0xFFFF } else { 0 },
            if a[6] > b[6] { 0xFFFF } else { 0 },
            if a[7] > b[7] { 0xFFFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ge(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
        [
            if a[0] >= b[0] { 0xFFFF } else { 0 },
            if a[1] >= b[1] { 0xFFFF } else { 0 },
            if a[2] >= b[2] { 0xFFFF } else { 0 },
            if a[3] >= b[3] { 0xFFFF } else { 0 },
            if a[4] >= b[4] { 0xFFFF } else { 0 },
            if a[5] >= b[5] { 0xFFFF } else { 0 },
            if a[6] >= b[6] { 0xFFFF } else { 0 },
            if a[7] >= b[7] { 0xFFFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn blend(mask: [u16; 8], if_true: [u16; 8], if_false: [u16; 8]) -> [u16; 8] {
        [
            if mask[0] != 0 {
                if_true[0]
            } else {
                if_false[0]
            },
            if mask[1] != 0 {
                if_true[1]
            } else {
                if_false[1]
            },
            if mask[2] != 0 {
                if_true[2]
            } else {
                if_false[2]
            },
            if mask[3] != 0 {
                if_true[3]
            } else {
                if_false[3]
            },
            if mask[4] != 0 {
                if_true[4]
            } else {
                if_false[4]
            },
            if mask[5] != 0 {
                if_true[5]
            } else {
                if_false[5]
            },
            if mask[6] != 0 {
                if_true[6]
            } else {
                if_false[6]
            },
            if mask[7] != 0 {
                if_true[7]
            } else {
                if_false[7]
            },
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [u16; 8]) -> u16 {
        a.iter().copied().fold(0u16, u16::wrapping_add)
    }

    #[inline(always)]
    fn not(a: [u16; 8]) -> [u16; 8] {
        [!a[0], !a[1], !a[2], !a[3], !a[4], !a[5], !a[6], !a[7]]
    }

    #[inline(always)]
    fn bitand(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
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
    fn bitor(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
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
    fn bitxor(a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
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

    #[inline(always)]
    fn shl_const<const N: i32>(a: [u16; 8]) -> [u16; 8] {
        [
            a[0].wrapping_shl(N as u32),
            a[1].wrapping_shl(N as u32),
            a[2].wrapping_shl(N as u32),
            a[3].wrapping_shl(N as u32),
            a[4].wrapping_shl(N as u32),
            a[5].wrapping_shl(N as u32),
            a[6].wrapping_shl(N as u32),
            a[7].wrapping_shl(N as u32),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [u16; 8]) -> [u16; 8] {
        [
            a[0].wrapping_shr(N as u32),
            a[1].wrapping_shr(N as u32),
            a[2].wrapping_shr(N as u32),
            a[3].wrapping_shr(N as u32),
            a[4].wrapping_shr(N as u32),
            a[5].wrapping_shr(N as u32),
            a[6].wrapping_shr(N as u32),
            a[7].wrapping_shr(N as u32),
        ]
    }

    #[inline(always)]
    fn all_true(a: [u16; 8]) -> bool {
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
    fn any_true(a: [u16; 8]) -> bool {
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
    fn bitmask(a: [u16; 8]) -> u32 {
        ((a[0] >> 15) as u32 & 1) << 0
            | ((a[1] >> 15) as u32 & 1) << 1
            | ((a[2] >> 15) as u32 & 1) << 2
            | ((a[3] >> 15) as u32 & 1) << 3
            | ((a[4] >> 15) as u32 & 1) << 4
            | ((a[5] >> 15) as u32 & 1) << 5
            | ((a[6] >> 15) as u32 & 1) << 6
            | ((a[7] >> 15) as u32 & 1) << 7
    }
}

impl U16x16Backend for archmage::ScalarToken {
    type Repr = [u16; 16];

    #[inline(always)]
    fn splat(v: u16) -> [u16; 16] {
        [v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v]
    }

    #[inline(always)]
    fn zero() -> [u16; 16] {
        [0u16; 16]
    }

    #[inline(always)]
    fn load(data: &[u16; 16]) -> [u16; 16] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [u16; 16]) -> [u16; 16] {
        arr
    }

    #[inline(always)]
    fn store(repr: [u16; 16], out: &mut [u16; 16]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [u16; 16]) -> [u16; 16] {
        repr
    }

    #[inline(always)]
    fn add(a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
        [
            a[0].wrapping_add(b[0]),
            a[1].wrapping_add(b[1]),
            a[2].wrapping_add(b[2]),
            a[3].wrapping_add(b[3]),
            a[4].wrapping_add(b[4]),
            a[5].wrapping_add(b[5]),
            a[6].wrapping_add(b[6]),
            a[7].wrapping_add(b[7]),
            a[8].wrapping_add(b[8]),
            a[9].wrapping_add(b[9]),
            a[10].wrapping_add(b[10]),
            a[11].wrapping_add(b[11]),
            a[12].wrapping_add(b[12]),
            a[13].wrapping_add(b[13]),
            a[14].wrapping_add(b[14]),
            a[15].wrapping_add(b[15]),
        ]
    }

    #[inline(always)]
    fn sub(a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
        [
            a[0].wrapping_sub(b[0]),
            a[1].wrapping_sub(b[1]),
            a[2].wrapping_sub(b[2]),
            a[3].wrapping_sub(b[3]),
            a[4].wrapping_sub(b[4]),
            a[5].wrapping_sub(b[5]),
            a[6].wrapping_sub(b[6]),
            a[7].wrapping_sub(b[7]),
            a[8].wrapping_sub(b[8]),
            a[9].wrapping_sub(b[9]),
            a[10].wrapping_sub(b[10]),
            a[11].wrapping_sub(b[11]),
            a[12].wrapping_sub(b[12]),
            a[13].wrapping_sub(b[13]),
            a[14].wrapping_sub(b[14]),
            a[15].wrapping_sub(b[15]),
        ]
    }

    #[inline(always)]
    fn mul(a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
        [
            a[0].wrapping_mul(b[0]),
            a[1].wrapping_mul(b[1]),
            a[2].wrapping_mul(b[2]),
            a[3].wrapping_mul(b[3]),
            a[4].wrapping_mul(b[4]),
            a[5].wrapping_mul(b[5]),
            a[6].wrapping_mul(b[6]),
            a[7].wrapping_mul(b[7]),
            a[8].wrapping_mul(b[8]),
            a[9].wrapping_mul(b[9]),
            a[10].wrapping_mul(b[10]),
            a[11].wrapping_mul(b[11]),
            a[12].wrapping_mul(b[12]),
            a[13].wrapping_mul(b[13]),
            a[14].wrapping_mul(b[14]),
            a[15].wrapping_mul(b[15]),
        ]
    }

    #[inline(always)]
    fn min(a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
        [
            if a[0] < b[0] { a[0] } else { b[0] },
            if a[1] < b[1] { a[1] } else { b[1] },
            if a[2] < b[2] { a[2] } else { b[2] },
            if a[3] < b[3] { a[3] } else { b[3] },
            if a[4] < b[4] { a[4] } else { b[4] },
            if a[5] < b[5] { a[5] } else { b[5] },
            if a[6] < b[6] { a[6] } else { b[6] },
            if a[7] < b[7] { a[7] } else { b[7] },
            if a[8] < b[8] { a[8] } else { b[8] },
            if a[9] < b[9] { a[9] } else { b[9] },
            if a[10] < b[10] { a[10] } else { b[10] },
            if a[11] < b[11] { a[11] } else { b[11] },
            if a[12] < b[12] { a[12] } else { b[12] },
            if a[13] < b[13] { a[13] } else { b[13] },
            if a[14] < b[14] { a[14] } else { b[14] },
            if a[15] < b[15] { a[15] } else { b[15] },
        ]
    }

    #[inline(always)]
    fn max(a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
        [
            if a[0] > b[0] { a[0] } else { b[0] },
            if a[1] > b[1] { a[1] } else { b[1] },
            if a[2] > b[2] { a[2] } else { b[2] },
            if a[3] > b[3] { a[3] } else { b[3] },
            if a[4] > b[4] { a[4] } else { b[4] },
            if a[5] > b[5] { a[5] } else { b[5] },
            if a[6] > b[6] { a[6] } else { b[6] },
            if a[7] > b[7] { a[7] } else { b[7] },
            if a[8] > b[8] { a[8] } else { b[8] },
            if a[9] > b[9] { a[9] } else { b[9] },
            if a[10] > b[10] { a[10] } else { b[10] },
            if a[11] > b[11] { a[11] } else { b[11] },
            if a[12] > b[12] { a[12] } else { b[12] },
            if a[13] > b[13] { a[13] } else { b[13] },
            if a[14] > b[14] { a[14] } else { b[14] },
            if a[15] > b[15] { a[15] } else { b[15] },
        ]
    }

    #[inline(always)]
    fn simd_eq(a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
        [
            if a[0] == b[0] { 0xFFFF } else { 0 },
            if a[1] == b[1] { 0xFFFF } else { 0 },
            if a[2] == b[2] { 0xFFFF } else { 0 },
            if a[3] == b[3] { 0xFFFF } else { 0 },
            if a[4] == b[4] { 0xFFFF } else { 0 },
            if a[5] == b[5] { 0xFFFF } else { 0 },
            if a[6] == b[6] { 0xFFFF } else { 0 },
            if a[7] == b[7] { 0xFFFF } else { 0 },
            if a[8] == b[8] { 0xFFFF } else { 0 },
            if a[9] == b[9] { 0xFFFF } else { 0 },
            if a[10] == b[10] { 0xFFFF } else { 0 },
            if a[11] == b[11] { 0xFFFF } else { 0 },
            if a[12] == b[12] { 0xFFFF } else { 0 },
            if a[13] == b[13] { 0xFFFF } else { 0 },
            if a[14] == b[14] { 0xFFFF } else { 0 },
            if a[15] == b[15] { 0xFFFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ne(a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
        [
            if a[0] != b[0] { 0xFFFF } else { 0 },
            if a[1] != b[1] { 0xFFFF } else { 0 },
            if a[2] != b[2] { 0xFFFF } else { 0 },
            if a[3] != b[3] { 0xFFFF } else { 0 },
            if a[4] != b[4] { 0xFFFF } else { 0 },
            if a[5] != b[5] { 0xFFFF } else { 0 },
            if a[6] != b[6] { 0xFFFF } else { 0 },
            if a[7] != b[7] { 0xFFFF } else { 0 },
            if a[8] != b[8] { 0xFFFF } else { 0 },
            if a[9] != b[9] { 0xFFFF } else { 0 },
            if a[10] != b[10] { 0xFFFF } else { 0 },
            if a[11] != b[11] { 0xFFFF } else { 0 },
            if a[12] != b[12] { 0xFFFF } else { 0 },
            if a[13] != b[13] { 0xFFFF } else { 0 },
            if a[14] != b[14] { 0xFFFF } else { 0 },
            if a[15] != b[15] { 0xFFFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_lt(a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
        [
            if a[0] < b[0] { 0xFFFF } else { 0 },
            if a[1] < b[1] { 0xFFFF } else { 0 },
            if a[2] < b[2] { 0xFFFF } else { 0 },
            if a[3] < b[3] { 0xFFFF } else { 0 },
            if a[4] < b[4] { 0xFFFF } else { 0 },
            if a[5] < b[5] { 0xFFFF } else { 0 },
            if a[6] < b[6] { 0xFFFF } else { 0 },
            if a[7] < b[7] { 0xFFFF } else { 0 },
            if a[8] < b[8] { 0xFFFF } else { 0 },
            if a[9] < b[9] { 0xFFFF } else { 0 },
            if a[10] < b[10] { 0xFFFF } else { 0 },
            if a[11] < b[11] { 0xFFFF } else { 0 },
            if a[12] < b[12] { 0xFFFF } else { 0 },
            if a[13] < b[13] { 0xFFFF } else { 0 },
            if a[14] < b[14] { 0xFFFF } else { 0 },
            if a[15] < b[15] { 0xFFFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_le(a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
        [
            if a[0] <= b[0] { 0xFFFF } else { 0 },
            if a[1] <= b[1] { 0xFFFF } else { 0 },
            if a[2] <= b[2] { 0xFFFF } else { 0 },
            if a[3] <= b[3] { 0xFFFF } else { 0 },
            if a[4] <= b[4] { 0xFFFF } else { 0 },
            if a[5] <= b[5] { 0xFFFF } else { 0 },
            if a[6] <= b[6] { 0xFFFF } else { 0 },
            if a[7] <= b[7] { 0xFFFF } else { 0 },
            if a[8] <= b[8] { 0xFFFF } else { 0 },
            if a[9] <= b[9] { 0xFFFF } else { 0 },
            if a[10] <= b[10] { 0xFFFF } else { 0 },
            if a[11] <= b[11] { 0xFFFF } else { 0 },
            if a[12] <= b[12] { 0xFFFF } else { 0 },
            if a[13] <= b[13] { 0xFFFF } else { 0 },
            if a[14] <= b[14] { 0xFFFF } else { 0 },
            if a[15] <= b[15] { 0xFFFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_gt(a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
        [
            if a[0] > b[0] { 0xFFFF } else { 0 },
            if a[1] > b[1] { 0xFFFF } else { 0 },
            if a[2] > b[2] { 0xFFFF } else { 0 },
            if a[3] > b[3] { 0xFFFF } else { 0 },
            if a[4] > b[4] { 0xFFFF } else { 0 },
            if a[5] > b[5] { 0xFFFF } else { 0 },
            if a[6] > b[6] { 0xFFFF } else { 0 },
            if a[7] > b[7] { 0xFFFF } else { 0 },
            if a[8] > b[8] { 0xFFFF } else { 0 },
            if a[9] > b[9] { 0xFFFF } else { 0 },
            if a[10] > b[10] { 0xFFFF } else { 0 },
            if a[11] > b[11] { 0xFFFF } else { 0 },
            if a[12] > b[12] { 0xFFFF } else { 0 },
            if a[13] > b[13] { 0xFFFF } else { 0 },
            if a[14] > b[14] { 0xFFFF } else { 0 },
            if a[15] > b[15] { 0xFFFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ge(a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
        [
            if a[0] >= b[0] { 0xFFFF } else { 0 },
            if a[1] >= b[1] { 0xFFFF } else { 0 },
            if a[2] >= b[2] { 0xFFFF } else { 0 },
            if a[3] >= b[3] { 0xFFFF } else { 0 },
            if a[4] >= b[4] { 0xFFFF } else { 0 },
            if a[5] >= b[5] { 0xFFFF } else { 0 },
            if a[6] >= b[6] { 0xFFFF } else { 0 },
            if a[7] >= b[7] { 0xFFFF } else { 0 },
            if a[8] >= b[8] { 0xFFFF } else { 0 },
            if a[9] >= b[9] { 0xFFFF } else { 0 },
            if a[10] >= b[10] { 0xFFFF } else { 0 },
            if a[11] >= b[11] { 0xFFFF } else { 0 },
            if a[12] >= b[12] { 0xFFFF } else { 0 },
            if a[13] >= b[13] { 0xFFFF } else { 0 },
            if a[14] >= b[14] { 0xFFFF } else { 0 },
            if a[15] >= b[15] { 0xFFFF } else { 0 },
        ]
    }

    #[inline(always)]
    fn blend(mask: [u16; 16], if_true: [u16; 16], if_false: [u16; 16]) -> [u16; 16] {
        [
            if mask[0] != 0 {
                if_true[0]
            } else {
                if_false[0]
            },
            if mask[1] != 0 {
                if_true[1]
            } else {
                if_false[1]
            },
            if mask[2] != 0 {
                if_true[2]
            } else {
                if_false[2]
            },
            if mask[3] != 0 {
                if_true[3]
            } else {
                if_false[3]
            },
            if mask[4] != 0 {
                if_true[4]
            } else {
                if_false[4]
            },
            if mask[5] != 0 {
                if_true[5]
            } else {
                if_false[5]
            },
            if mask[6] != 0 {
                if_true[6]
            } else {
                if_false[6]
            },
            if mask[7] != 0 {
                if_true[7]
            } else {
                if_false[7]
            },
            if mask[8] != 0 {
                if_true[8]
            } else {
                if_false[8]
            },
            if mask[9] != 0 {
                if_true[9]
            } else {
                if_false[9]
            },
            if mask[10] != 0 {
                if_true[10]
            } else {
                if_false[10]
            },
            if mask[11] != 0 {
                if_true[11]
            } else {
                if_false[11]
            },
            if mask[12] != 0 {
                if_true[12]
            } else {
                if_false[12]
            },
            if mask[13] != 0 {
                if_true[13]
            } else {
                if_false[13]
            },
            if mask[14] != 0 {
                if_true[14]
            } else {
                if_false[14]
            },
            if mask[15] != 0 {
                if_true[15]
            } else {
                if_false[15]
            },
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [u16; 16]) -> u16 {
        a.iter().copied().fold(0u16, u16::wrapping_add)
    }

    #[inline(always)]
    fn not(a: [u16; 16]) -> [u16; 16] {
        [
            !a[0], !a[1], !a[2], !a[3], !a[4], !a[5], !a[6], !a[7], !a[8], !a[9], !a[10], !a[11],
            !a[12], !a[13], !a[14], !a[15],
        ]
    }

    #[inline(always)]
    fn bitand(a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
        [
            a[0] & b[0],
            a[1] & b[1],
            a[2] & b[2],
            a[3] & b[3],
            a[4] & b[4],
            a[5] & b[5],
            a[6] & b[6],
            a[7] & b[7],
            a[8] & b[8],
            a[9] & b[9],
            a[10] & b[10],
            a[11] & b[11],
            a[12] & b[12],
            a[13] & b[13],
            a[14] & b[14],
            a[15] & b[15],
        ]
    }

    #[inline(always)]
    fn bitor(a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
        [
            a[0] | b[0],
            a[1] | b[1],
            a[2] | b[2],
            a[3] | b[3],
            a[4] | b[4],
            a[5] | b[5],
            a[6] | b[6],
            a[7] | b[7],
            a[8] | b[8],
            a[9] | b[9],
            a[10] | b[10],
            a[11] | b[11],
            a[12] | b[12],
            a[13] | b[13],
            a[14] | b[14],
            a[15] | b[15],
        ]
    }

    #[inline(always)]
    fn bitxor(a: [u16; 16], b: [u16; 16]) -> [u16; 16] {
        [
            a[0] ^ b[0],
            a[1] ^ b[1],
            a[2] ^ b[2],
            a[3] ^ b[3],
            a[4] ^ b[4],
            a[5] ^ b[5],
            a[6] ^ b[6],
            a[7] ^ b[7],
            a[8] ^ b[8],
            a[9] ^ b[9],
            a[10] ^ b[10],
            a[11] ^ b[11],
            a[12] ^ b[12],
            a[13] ^ b[13],
            a[14] ^ b[14],
            a[15] ^ b[15],
        ]
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [u16; 16]) -> [u16; 16] {
        [
            a[0].wrapping_shl(N as u32),
            a[1].wrapping_shl(N as u32),
            a[2].wrapping_shl(N as u32),
            a[3].wrapping_shl(N as u32),
            a[4].wrapping_shl(N as u32),
            a[5].wrapping_shl(N as u32),
            a[6].wrapping_shl(N as u32),
            a[7].wrapping_shl(N as u32),
            a[8].wrapping_shl(N as u32),
            a[9].wrapping_shl(N as u32),
            a[10].wrapping_shl(N as u32),
            a[11].wrapping_shl(N as u32),
            a[12].wrapping_shl(N as u32),
            a[13].wrapping_shl(N as u32),
            a[14].wrapping_shl(N as u32),
            a[15].wrapping_shl(N as u32),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [u16; 16]) -> [u16; 16] {
        [
            a[0].wrapping_shr(N as u32),
            a[1].wrapping_shr(N as u32),
            a[2].wrapping_shr(N as u32),
            a[3].wrapping_shr(N as u32),
            a[4].wrapping_shr(N as u32),
            a[5].wrapping_shr(N as u32),
            a[6].wrapping_shr(N as u32),
            a[7].wrapping_shr(N as u32),
            a[8].wrapping_shr(N as u32),
            a[9].wrapping_shr(N as u32),
            a[10].wrapping_shr(N as u32),
            a[11].wrapping_shr(N as u32),
            a[12].wrapping_shr(N as u32),
            a[13].wrapping_shr(N as u32),
            a[14].wrapping_shr(N as u32),
            a[15].wrapping_shr(N as u32),
        ]
    }

    #[inline(always)]
    fn all_true(a: [u16; 16]) -> bool {
        a[0] != 0
            && a[1] != 0
            && a[2] != 0
            && a[3] != 0
            && a[4] != 0
            && a[5] != 0
            && a[6] != 0
            && a[7] != 0
            && a[8] != 0
            && a[9] != 0
            && a[10] != 0
            && a[11] != 0
            && a[12] != 0
            && a[13] != 0
            && a[14] != 0
            && a[15] != 0
    }

    #[inline(always)]
    fn any_true(a: [u16; 16]) -> bool {
        a[0] != 0
            || a[1] != 0
            || a[2] != 0
            || a[3] != 0
            || a[4] != 0
            || a[5] != 0
            || a[6] != 0
            || a[7] != 0
            || a[8] != 0
            || a[9] != 0
            || a[10] != 0
            || a[11] != 0
            || a[12] != 0
            || a[13] != 0
            || a[14] != 0
            || a[15] != 0
    }

    #[inline(always)]
    fn bitmask(a: [u16; 16]) -> u32 {
        ((a[0] >> 15) as u32 & 1) << 0
            | ((a[1] >> 15) as u32 & 1) << 1
            | ((a[2] >> 15) as u32 & 1) << 2
            | ((a[3] >> 15) as u32 & 1) << 3
            | ((a[4] >> 15) as u32 & 1) << 4
            | ((a[5] >> 15) as u32 & 1) << 5
            | ((a[6] >> 15) as u32 & 1) << 6
            | ((a[7] >> 15) as u32 & 1) << 7
            | ((a[8] >> 15) as u32 & 1) << 8
            | ((a[9] >> 15) as u32 & 1) << 9
            | ((a[10] >> 15) as u32 & 1) << 10
            | ((a[11] >> 15) as u32 & 1) << 11
            | ((a[12] >> 15) as u32 & 1) << 12
            | ((a[13] >> 15) as u32 & 1) << 13
            | ((a[14] >> 15) as u32 & 1) << 14
            | ((a[15] >> 15) as u32 & 1) << 15
    }
}

impl U64x2Backend for archmage::ScalarToken {
    type Repr = [u64; 2];

    #[inline(always)]
    fn splat(v: u64) -> [u64; 2] {
        [v, v]
    }

    #[inline(always)]
    fn zero() -> [u64; 2] {
        [0u64; 2]
    }

    #[inline(always)]
    fn load(data: &[u64; 2]) -> [u64; 2] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [u64; 2]) -> [u64; 2] {
        arr
    }

    #[inline(always)]
    fn store(repr: [u64; 2], out: &mut [u64; 2]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [u64; 2]) -> [u64; 2] {
        repr
    }

    #[inline(always)]
    fn add(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [a[0].wrapping_add(b[0]), a[1].wrapping_add(b[1])]
    }

    #[inline(always)]
    fn sub(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [a[0].wrapping_sub(b[0]), a[1].wrapping_sub(b[1])]
    }

    #[inline(always)]
    fn min(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [
            if a[0] < b[0] { a[0] } else { b[0] },
            if a[1] < b[1] { a[1] } else { b[1] },
        ]
    }

    #[inline(always)]
    fn max(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [
            if a[0] > b[0] { a[0] } else { b[0] },
            if a[1] > b[1] { a[1] } else { b[1] },
        ]
    }

    #[inline(always)]
    fn simd_eq(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [
            if a[0] == b[0] { u64::MAX } else { 0 },
            if a[1] == b[1] { u64::MAX } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ne(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [
            if a[0] != b[0] { u64::MAX } else { 0 },
            if a[1] != b[1] { u64::MAX } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_lt(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [
            if a[0] < b[0] { u64::MAX } else { 0 },
            if a[1] < b[1] { u64::MAX } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_le(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [
            if a[0] <= b[0] { u64::MAX } else { 0 },
            if a[1] <= b[1] { u64::MAX } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_gt(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [
            if a[0] > b[0] { u64::MAX } else { 0 },
            if a[1] > b[1] { u64::MAX } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ge(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [
            if a[0] >= b[0] { u64::MAX } else { 0 },
            if a[1] >= b[1] { u64::MAX } else { 0 },
        ]
    }

    #[inline(always)]
    fn blend(mask: [u64; 2], if_true: [u64; 2], if_false: [u64; 2]) -> [u64; 2] {
        [
            if mask[0] != 0 {
                if_true[0]
            } else {
                if_false[0]
            },
            if mask[1] != 0 {
                if_true[1]
            } else {
                if_false[1]
            },
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [u64; 2]) -> u64 {
        a.iter().copied().fold(0u64, u64::wrapping_add)
    }

    #[inline(always)]
    fn not(a: [u64; 2]) -> [u64; 2] {
        [!a[0], !a[1]]
    }

    #[inline(always)]
    fn bitand(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [a[0] & b[0], a[1] & b[1]]
    }

    #[inline(always)]
    fn bitor(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [a[0] | b[0], a[1] | b[1]]
    }

    #[inline(always)]
    fn bitxor(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        [a[0] ^ b[0], a[1] ^ b[1]]
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [u64; 2]) -> [u64; 2] {
        [a[0].wrapping_shl(N as u32), a[1].wrapping_shl(N as u32)]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [u64; 2]) -> [u64; 2] {
        [a[0].wrapping_shr(N as u32), a[1].wrapping_shr(N as u32)]
    }

    #[inline(always)]
    fn all_true(a: [u64; 2]) -> bool {
        a[0] != 0 && a[1] != 0
    }

    #[inline(always)]
    fn any_true(a: [u64; 2]) -> bool {
        a[0] != 0 || a[1] != 0
    }

    #[inline(always)]
    fn bitmask(a: [u64; 2]) -> u32 {
        ((a[0] >> 63) as u32 & 1) << 0 | ((a[1] >> 63) as u32 & 1) << 1
    }
}

impl U64x4Backend for archmage::ScalarToken {
    type Repr = [u64; 4];

    #[inline(always)]
    fn splat(v: u64) -> [u64; 4] {
        [v, v, v, v]
    }

    #[inline(always)]
    fn zero() -> [u64; 4] {
        [0u64; 4]
    }

    #[inline(always)]
    fn load(data: &[u64; 4]) -> [u64; 4] {
        *data
    }

    #[inline(always)]
    fn from_array(arr: [u64; 4]) -> [u64; 4] {
        arr
    }

    #[inline(always)]
    fn store(repr: [u64; 4], out: &mut [u64; 4]) {
        *out = repr;
    }

    #[inline(always)]
    fn to_array(repr: [u64; 4]) -> [u64; 4] {
        repr
    }

    #[inline(always)]
    fn add(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        [
            a[0].wrapping_add(b[0]),
            a[1].wrapping_add(b[1]),
            a[2].wrapping_add(b[2]),
            a[3].wrapping_add(b[3]),
        ]
    }

    #[inline(always)]
    fn sub(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        [
            a[0].wrapping_sub(b[0]),
            a[1].wrapping_sub(b[1]),
            a[2].wrapping_sub(b[2]),
            a[3].wrapping_sub(b[3]),
        ]
    }

    #[inline(always)]
    fn min(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        [
            if a[0] < b[0] { a[0] } else { b[0] },
            if a[1] < b[1] { a[1] } else { b[1] },
            if a[2] < b[2] { a[2] } else { b[2] },
            if a[3] < b[3] { a[3] } else { b[3] },
        ]
    }

    #[inline(always)]
    fn max(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        [
            if a[0] > b[0] { a[0] } else { b[0] },
            if a[1] > b[1] { a[1] } else { b[1] },
            if a[2] > b[2] { a[2] } else { b[2] },
            if a[3] > b[3] { a[3] } else { b[3] },
        ]
    }

    #[inline(always)]
    fn simd_eq(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        [
            if a[0] == b[0] { u64::MAX } else { 0 },
            if a[1] == b[1] { u64::MAX } else { 0 },
            if a[2] == b[2] { u64::MAX } else { 0 },
            if a[3] == b[3] { u64::MAX } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ne(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        [
            if a[0] != b[0] { u64::MAX } else { 0 },
            if a[1] != b[1] { u64::MAX } else { 0 },
            if a[2] != b[2] { u64::MAX } else { 0 },
            if a[3] != b[3] { u64::MAX } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_lt(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        [
            if a[0] < b[0] { u64::MAX } else { 0 },
            if a[1] < b[1] { u64::MAX } else { 0 },
            if a[2] < b[2] { u64::MAX } else { 0 },
            if a[3] < b[3] { u64::MAX } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_le(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        [
            if a[0] <= b[0] { u64::MAX } else { 0 },
            if a[1] <= b[1] { u64::MAX } else { 0 },
            if a[2] <= b[2] { u64::MAX } else { 0 },
            if a[3] <= b[3] { u64::MAX } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_gt(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        [
            if a[0] > b[0] { u64::MAX } else { 0 },
            if a[1] > b[1] { u64::MAX } else { 0 },
            if a[2] > b[2] { u64::MAX } else { 0 },
            if a[3] > b[3] { u64::MAX } else { 0 },
        ]
    }

    #[inline(always)]
    fn simd_ge(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        [
            if a[0] >= b[0] { u64::MAX } else { 0 },
            if a[1] >= b[1] { u64::MAX } else { 0 },
            if a[2] >= b[2] { u64::MAX } else { 0 },
            if a[3] >= b[3] { u64::MAX } else { 0 },
        ]
    }

    #[inline(always)]
    fn blend(mask: [u64; 4], if_true: [u64; 4], if_false: [u64; 4]) -> [u64; 4] {
        [
            if mask[0] != 0 {
                if_true[0]
            } else {
                if_false[0]
            },
            if mask[1] != 0 {
                if_true[1]
            } else {
                if_false[1]
            },
            if mask[2] != 0 {
                if_true[2]
            } else {
                if_false[2]
            },
            if mask[3] != 0 {
                if_true[3]
            } else {
                if_false[3]
            },
        ]
    }

    #[inline(always)]
    fn reduce_add(a: [u64; 4]) -> u64 {
        a.iter().copied().fold(0u64, u64::wrapping_add)
    }

    #[inline(always)]
    fn not(a: [u64; 4]) -> [u64; 4] {
        [!a[0], !a[1], !a[2], !a[3]]
    }

    #[inline(always)]
    fn bitand(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]
    }

    #[inline(always)]
    fn bitor(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]
    }

    #[inline(always)]
    fn bitxor(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
    }

    #[inline(always)]
    fn shl_const<const N: i32>(a: [u64; 4]) -> [u64; 4] {
        [
            a[0].wrapping_shl(N as u32),
            a[1].wrapping_shl(N as u32),
            a[2].wrapping_shl(N as u32),
            a[3].wrapping_shl(N as u32),
        ]
    }

    #[inline(always)]
    fn shr_logical_const<const N: i32>(a: [u64; 4]) -> [u64; 4] {
        [
            a[0].wrapping_shr(N as u32),
            a[1].wrapping_shr(N as u32),
            a[2].wrapping_shr(N as u32),
            a[3].wrapping_shr(N as u32),
        ]
    }

    #[inline(always)]
    fn all_true(a: [u64; 4]) -> bool {
        a[0] != 0 && a[1] != 0 && a[2] != 0 && a[3] != 0
    }

    #[inline(always)]
    fn any_true(a: [u64; 4]) -> bool {
        a[0] != 0 || a[1] != 0 || a[2] != 0 || a[3] != 0
    }

    #[inline(always)]
    fn bitmask(a: [u64; 4]) -> u32 {
        ((a[0] >> 63) as u32 & 1) << 0
            | ((a[1] >> 63) as u32 & 1) << 1
            | ((a[2] >> 63) as u32 & 1) << 2
            | ((a[3] >> 63) as u32 & 1) << 3
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

impl I8x16Bitcast for archmage::ScalarToken {
    #[inline(always)]
    fn bitcast_i8_to_u8(a: [i8; 16]) -> [u8; 16] {
        let mut out = [0u8; 16];
        let mut i = 0;
        while i < 16 {
            out[i] = a[i] as u8;
            i += 1;
        }
        out
    }
    #[inline(always)]
    fn bitcast_u8_to_i8(a: [u8; 16]) -> [i8; 16] {
        let mut out = [0i8; 16];
        let mut i = 0;
        while i < 16 {
            out[i] = a[i] as i8;
            i += 1;
        }
        out
    }
}

impl I8x32Bitcast for archmage::ScalarToken {
    #[inline(always)]
    fn bitcast_i8_to_u8(a: [i8; 32]) -> [u8; 32] {
        let mut out = [0u8; 32];
        let mut i = 0;
        while i < 32 {
            out[i] = a[i] as u8;
            i += 1;
        }
        out
    }
    #[inline(always)]
    fn bitcast_u8_to_i8(a: [u8; 32]) -> [i8; 32] {
        let mut out = [0i8; 32];
        let mut i = 0;
        while i < 32 {
            out[i] = a[i] as i8;
            i += 1;
        }
        out
    }
}

impl I16x8Bitcast for archmage::ScalarToken {
    #[inline(always)]
    fn bitcast_i16_to_u16(a: [i16; 8]) -> [u16; 8] {
        let mut out = [0u16; 8];
        let mut i = 0;
        while i < 8 {
            out[i] = a[i] as u16;
            i += 1;
        }
        out
    }
    #[inline(always)]
    fn bitcast_u16_to_i16(a: [u16; 8]) -> [i16; 8] {
        let mut out = [0i16; 8];
        let mut i = 0;
        while i < 8 {
            out[i] = a[i] as i16;
            i += 1;
        }
        out
    }
}

impl I16x16Bitcast for archmage::ScalarToken {
    #[inline(always)]
    fn bitcast_i16_to_u16(a: [i16; 16]) -> [u16; 16] {
        let mut out = [0u16; 16];
        let mut i = 0;
        while i < 16 {
            out[i] = a[i] as u16;
            i += 1;
        }
        out
    }
    #[inline(always)]
    fn bitcast_u16_to_i16(a: [u16; 16]) -> [i16; 16] {
        let mut out = [0i16; 16];
        let mut i = 0;
        while i < 16 {
            out[i] = a[i] as i16;
            i += 1;
        }
        out
    }
}

impl U64x2Bitcast for archmage::ScalarToken {
    #[inline(always)]
    fn bitcast_u64_to_i64(a: [u64; 2]) -> [i64; 2] {
        [a[0] as i64, a[1] as i64]
    }
    #[inline(always)]
    fn bitcast_i64_to_u64(a: [i64; 2]) -> [u64; 2] {
        [a[0] as u64, a[1] as u64]
    }
}

impl U64x4Bitcast for archmage::ScalarToken {
    #[inline(always)]
    fn bitcast_u64_to_i64(a: [u64; 4]) -> [i64; 4] {
        [a[0] as i64, a[1] as i64, a[2] as i64, a[3] as i64]
    }
    #[inline(always)]
    fn bitcast_i64_to_u64(a: [i64; 4]) -> [u64; 4] {
        [a[0] as u64, a[1] as u64, a[2] as u64, a[3] as u64]
    }
}

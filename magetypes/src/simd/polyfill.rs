//! Polyfill module for emulating wider SIMD types on narrower hardware.
//!
//! This module provides types like `f32x8` that work on SSE hardware by
//! internally using two `f32x4` operations. This allows writing code
//! targeting AVX2 widths while still running (slower) on SSE-only systems.
//!
//! Polyfills are mainly used internally by [`WidthDispatch`](crate::WidthDispatch).
//! Direct usage is rarely needed.

#[cfg(target_arch = "x86_64")]
pub mod sse {
    //! Polyfilled 256-bit types using SSE (128-bit) operations.
    //!
    //! These types emulate AVX2-width vectors using pairs of SSE vectors.

    use crate::simd::x86::w128::{f32x4, f64x2, i32x4};
    use archmage::Sse41Token;
    use core::ops::{Add, Div, Mul, Neg, Sub};

    /// Emulated 8-wide f32 vector using two SSE f32x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct f32x8 {
        lo: f32x4,
        hi: f32x4,
    }

    impl f32x8 {
        pub const LANES: usize = 8;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: Sse41Token, data: &[f32; 8]) -> Self {
            let lo_arr: &[f32; 4] = data[0..4].try_into().unwrap();
            let hi_arr: &[f32; 4] = data[4..8].try_into().unwrap();
            Self {
                lo: f32x4::load(token, lo_arr),
                hi: f32x4::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: Sse41Token, v: f32) -> Self {
            Self {
                lo: f32x4::splat(token, v),
                hi: f32x4::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: Sse41Token) -> Self {
            Self {
                lo: f32x4::zero(token),
                hi: f32x4::zero(token),
            }
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [f32; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            let lo_arr: &mut [f32; 4] = lo.try_into().unwrap();
            let hi_arr: &mut [f32; 4] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [f32; 8] {
            let mut out = [0.0f32; 8];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Square root
        #[inline(always)]
        pub fn sqrt(self) -> Self {
            Self {
                lo: self.lo.sqrt(),
                hi: self.hi.sqrt(),
            }
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Floor
        #[inline(always)]
        pub fn floor(self) -> Self {
            Self {
                lo: self.lo.floor(),
                hi: self.hi.floor(),
            }
        }

        /// Ceil
        #[inline(always)]
        pub fn ceil(self) -> Self {
            Self {
                lo: self.lo.ceil(),
                hi: self.hi.ceil(),
            }
        }

        /// Round
        #[inline(always)]
        pub fn round(self) -> Self {
            Self {
                lo: self.lo.round(),
                hi: self.hi.round(),
            }
        }

        /// Fused multiply-add: self * a + b
        /// Note: SSE doesn't have native FMA, this uses separate mul+add
        #[inline(always)]
        pub fn mul_add(self, a: Self, b: Self) -> Self {
            Self {
                lo: self.lo.mul_add(a.lo, b.lo),
                hi: self.hi.mul_add(a.hi, b.hi),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> f32 {
            self.lo.reduce_add() + self.hi.reduce_add()
        }

        /// Reduce: max of all lanes
        #[inline(always)]
        pub fn reduce_max(self) -> f32 {
            self.lo.reduce_max().max(self.hi.reduce_max())
        }

        /// Reduce: min of all lanes
        #[inline(always)]
        pub fn reduce_min(self) -> f32 {
            self.lo.reduce_min().min(self.hi.reduce_min())
        }
    }

    impl Add for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl Div for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            Self {
                lo: self.lo / rhs.lo,
                hi: self.hi / rhs.hi,
            }
        }
    }

    impl Neg for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self {
                lo: -self.lo,
                hi: -self.hi,
            }
        }
    }

    impl core::ops::AddAssign for f32x8 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for f32x8 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for f32x8 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl core::ops::DivAssign for f32x8 {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }

    /// Emulated 4-wide f64 vector using two SSE f64x2 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct f64x4 {
        lo: f64x2,
        hi: f64x2,
    }

    impl f64x4 {
        pub const LANES: usize = 4;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: Sse41Token, data: &[f64; 4]) -> Self {
            let lo_arr: &[f64; 2] = data[0..2].try_into().unwrap();
            let hi_arr: &[f64; 2] = data[2..4].try_into().unwrap();
            Self {
                lo: f64x2::load(token, lo_arr),
                hi: f64x2::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: Sse41Token, v: f64) -> Self {
            Self {
                lo: f64x2::splat(token, v),
                hi: f64x2::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: Sse41Token) -> Self {
            Self {
                lo: f64x2::zero(token),
                hi: f64x2::zero(token),
            }
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [f64; 4]) {
            let (lo, hi) = out.split_at_mut(2);
            let lo_arr: &mut [f64; 2] = lo.try_into().unwrap();
            let hi_arr: &mut [f64; 2] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [f64; 4] {
            let mut out = [0.0f64; 4];
            self.store(&mut out);
            out
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> f64 {
            self.lo.reduce_add() + self.hi.reduce_add()
        }
    }

    impl Add for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl Div for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            Self {
                lo: self.lo / rhs.lo,
                hi: self.hi / rhs.hi,
            }
        }
    }

    impl Neg for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self {
                lo: -self.lo,
                hi: -self.hi,
            }
        }
    }

    /// Emulated 8-wide i32 vector using two SSE i32x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i32x8 {
        lo: i32x4,
        hi: i32x4,
    }

    impl i32x8 {
        pub const LANES: usize = 8;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: Sse41Token, data: &[i32; 8]) -> Self {
            let lo_arr: &[i32; 4] = data[0..4].try_into().unwrap();
            let hi_arr: &[i32; 4] = data[4..8].try_into().unwrap();
            Self {
                lo: i32x4::load(token, lo_arr),
                hi: i32x4::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: Sse41Token, v: i32) -> Self {
            Self {
                lo: i32x4::splat(token, v),
                hi: i32x4::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: Sse41Token) -> Self {
            Self {
                lo: i32x4::zero(token),
                hi: i32x4::zero(token),
            }
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i32; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            let lo_arr: &mut [i32; 4] = lo.try_into().unwrap();
            let hi_arr: &mut [i32; 4] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i32; 8] {
            let mut out = [0i32; 8];
            self.store(&mut out);
            out
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i32 {
            self.lo.reduce_add() + self.hi.reduce_add()
        }
    }

    impl Add for i32x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i32x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    /// Token type alias for SSE polyfill module
    pub type Token = Sse41Token;

    /// Lane count constants for polyfilled types
    pub const LANES_F32: usize = 8;
    pub const LANES_F64: usize = 4;
    pub const LANES_32: usize = 8;

    /// Alias for the polyfilled type (same API as avx2::f32xN)
    pub type f32xN = f32x8;
    pub type f64xN = f64x4;
    pub type i32xN = i32x8;

    // ========================================================================
    // 512-bit polyfills using 4x 128-bit SSE vectors
    // ========================================================================

    /// Emulated 16-wide f32 vector using four SSE f32x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct f32x16 {
        a: f32x4,
        b: f32x4,
        c: f32x4,
        d: f32x4,
    }

    impl f32x16 {
        pub const LANES: usize = 16;

        #[inline(always)]
        pub fn load(token: Sse41Token, data: &[f32; 16]) -> Self {
            Self {
                a: f32x4::load(token, data[0..4].try_into().unwrap()),
                b: f32x4::load(token, data[4..8].try_into().unwrap()),
                c: f32x4::load(token, data[8..12].try_into().unwrap()),
                d: f32x4::load(token, data[12..16].try_into().unwrap()),
            }
        }

        #[inline(always)]
        pub fn splat(token: Sse41Token, v: f32) -> Self {
            let part = f32x4::splat(token, v);
            Self {
                a: part,
                b: part,
                c: part,
                d: part,
            }
        }

        #[inline(always)]
        pub fn zero(token: Sse41Token) -> Self {
            let part = f32x4::zero(token);
            Self {
                a: part,
                b: part,
                c: part,
                d: part,
            }
        }

        #[inline(always)]
        pub fn store(self, out: &mut [f32; 16]) {
            let (ab, cd) = out.split_at_mut(8);
            let (a, b) = ab.split_at_mut(4);
            let (c, d) = cd.split_at_mut(4);
            self.a.store(a.try_into().unwrap());
            self.b.store(b.try_into().unwrap());
            self.c.store(c.try_into().unwrap());
            self.d.store(d.try_into().unwrap());
        }

        #[inline(always)]
        pub fn to_array(self) -> [f32; 16] {
            let mut out = [0.0f32; 16];
            self.store(&mut out);
            out
        }
    }

    impl Add for f32x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                a: self.a + rhs.a,
                b: self.b + rhs.b,
                c: self.c + rhs.c,
                d: self.d + rhs.d,
            }
        }
    }

    impl Sub for f32x16 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                a: self.a - rhs.a,
                b: self.b - rhs.b,
                c: self.c - rhs.c,
                d: self.d - rhs.d,
            }
        }
    }

    impl Mul for f32x16 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                a: self.a * rhs.a,
                b: self.b * rhs.b,
                c: self.c * rhs.c,
                d: self.d * rhs.d,
            }
        }
    }

    impl Div for f32x16 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            Self {
                a: self.a / rhs.a,
                b: self.b / rhs.b,
                c: self.c / rhs.c,
                d: self.d / rhs.d,
            }
        }
    }

    impl Neg for f32x16 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self {
                a: -self.a,
                b: -self.b,
                c: -self.c,
                d: -self.d,
            }
        }
    }

    /// Emulated 8-wide f64 vector using four SSE f64x2 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct f64x8 {
        a: f64x2,
        b: f64x2,
        c: f64x2,
        d: f64x2,
    }

    impl f64x8 {
        pub const LANES: usize = 8;

        #[inline(always)]
        pub fn load(token: Sse41Token, data: &[f64; 8]) -> Self {
            Self {
                a: f64x2::load(token, data[0..2].try_into().unwrap()),
                b: f64x2::load(token, data[2..4].try_into().unwrap()),
                c: f64x2::load(token, data[4..6].try_into().unwrap()),
                d: f64x2::load(token, data[6..8].try_into().unwrap()),
            }
        }

        #[inline(always)]
        pub fn splat(token: Sse41Token, v: f64) -> Self {
            let part = f64x2::splat(token, v);
            Self {
                a: part,
                b: part,
                c: part,
                d: part,
            }
        }

        #[inline(always)]
        pub fn zero(token: Sse41Token) -> Self {
            let part = f64x2::zero(token);
            Self {
                a: part,
                b: part,
                c: part,
                d: part,
            }
        }

        #[inline(always)]
        pub fn store(self, out: &mut [f64; 8]) {
            let (ab, cd) = out.split_at_mut(4);
            let (a, b) = ab.split_at_mut(2);
            let (c, d) = cd.split_at_mut(2);
            self.a.store(a.try_into().unwrap());
            self.b.store(b.try_into().unwrap());
            self.c.store(c.try_into().unwrap());
            self.d.store(d.try_into().unwrap());
        }

        #[inline(always)]
        pub fn to_array(self) -> [f64; 8] {
            let mut out = [0.0f64; 8];
            self.store(&mut out);
            out
        }
    }

    impl Add for f64x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                a: self.a + rhs.a,
                b: self.b + rhs.b,
                c: self.c + rhs.c,
                d: self.d + rhs.d,
            }
        }
    }

    impl Sub for f64x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                a: self.a - rhs.a,
                b: self.b - rhs.b,
                c: self.c - rhs.c,
                d: self.d - rhs.d,
            }
        }
    }

    impl Mul for f64x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                a: self.a * rhs.a,
                b: self.b * rhs.b,
                c: self.c * rhs.c,
                d: self.d * rhs.d,
            }
        }
    }

    impl Div for f64x8 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            Self {
                a: self.a / rhs.a,
                b: self.b / rhs.b,
                c: self.c / rhs.c,
                d: self.d / rhs.d,
            }
        }
    }

    impl Neg for f64x8 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self {
                a: -self.a,
                b: -self.b,
                c: -self.c,
                d: -self.d,
            }
        }
    }

    /// Emulated 16-wide i32 vector using four SSE i32x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i32x16 {
        a: i32x4,
        b: i32x4,
        c: i32x4,
        d: i32x4,
    }

    impl i32x16 {
        pub const LANES: usize = 16;

        #[inline(always)]
        pub fn load(token: Sse41Token, data: &[i32; 16]) -> Self {
            Self {
                a: i32x4::load(token, data[0..4].try_into().unwrap()),
                b: i32x4::load(token, data[4..8].try_into().unwrap()),
                c: i32x4::load(token, data[8..12].try_into().unwrap()),
                d: i32x4::load(token, data[12..16].try_into().unwrap()),
            }
        }

        #[inline(always)]
        pub fn splat(token: Sse41Token, v: i32) -> Self {
            let part = i32x4::splat(token, v);
            Self {
                a: part,
                b: part,
                c: part,
                d: part,
            }
        }

        #[inline(always)]
        pub fn zero(token: Sse41Token) -> Self {
            let part = i32x4::zero(token);
            Self {
                a: part,
                b: part,
                c: part,
                d: part,
            }
        }

        #[inline(always)]
        pub fn store(self, out: &mut [i32; 16]) {
            let (ab, cd) = out.split_at_mut(8);
            let (a, b) = ab.split_at_mut(4);
            let (c, d) = cd.split_at_mut(4);
            self.a.store(a.try_into().unwrap());
            self.b.store(b.try_into().unwrap());
            self.c.store(c.try_into().unwrap());
            self.d.store(d.try_into().unwrap());
        }

        #[inline(always)]
        pub fn to_array(self) -> [i32; 16] {
            let mut out = [0i32; 16];
            self.store(&mut out);
            out
        }
    }

    impl Add for i32x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                a: self.a + rhs.a,
                b: self.b + rhs.b,
                c: self.c + rhs.c,
                d: self.d + rhs.d,
            }
        }
    }

    impl Sub for i32x16 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                a: self.a - rhs.a,
                b: self.b - rhs.b,
                c: self.c - rhs.c,
                d: self.d - rhs.d,
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    //! AVX2-based 512-bit polyfills (more efficient than SSE-based on AVX2 hardware).
    //!
    //! Polyfilled 512-bit types using AVX2 (256-bit) operations.
    //! These types emulate AVX-512-width vectors using pairs of AVX2 vectors.

    use crate::simd::x86::w256::{f32x8, f64x4, i32x8};
    use archmage::Avx2FmaToken;
    use core::ops::{Add, Div, Mul, Neg, Sub};

    /// Emulated 16-wide f32 vector using two AVX2 f32x8 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct f32x16 {
        lo: f32x8,
        hi: f32x8,
    }

    impl f32x16 {
        pub const LANES: usize = 16;

        #[inline(always)]
        pub fn load(token: Avx2FmaToken, data: &[f32; 16]) -> Self {
            Self {
                lo: f32x8::load(token, data[0..8].try_into().unwrap()),
                hi: f32x8::load(token, data[8..16].try_into().unwrap()),
            }
        }

        #[inline(always)]
        pub fn splat(token: Avx2FmaToken, v: f32) -> Self {
            let part = f32x8::splat(token, v);
            Self { lo: part, hi: part }
        }

        #[inline(always)]
        pub fn zero(token: Avx2FmaToken) -> Self {
            let part = f32x8::zero(token);
            Self { lo: part, hi: part }
        }

        #[inline(always)]
        pub fn store(self, out: &mut [f32; 16]) {
            let (lo, hi) = out.split_at_mut(8);
            self.lo.store(lo.try_into().unwrap());
            self.hi.store(hi.try_into().unwrap());
        }

        #[inline(always)]
        pub fn to_array(self) -> [f32; 16] {
            let mut out = [0.0f32; 16];
            self.store(&mut out);
            out
        }
    }

    impl Add for f32x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for f32x16 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for f32x16 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl Div for f32x16 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            Self {
                lo: self.lo / rhs.lo,
                hi: self.hi / rhs.hi,
            }
        }
    }

    impl Neg for f32x16 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self {
                lo: -self.lo,
                hi: -self.hi,
            }
        }
    }

    /// Emulated 8-wide f64 vector using two AVX2 f64x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct f64x8 {
        lo: f64x4,
        hi: f64x4,
    }

    impl f64x8 {
        pub const LANES: usize = 8;

        #[inline(always)]
        pub fn load(token: Avx2FmaToken, data: &[f64; 8]) -> Self {
            Self {
                lo: f64x4::load(token, data[0..4].try_into().unwrap()),
                hi: f64x4::load(token, data[4..8].try_into().unwrap()),
            }
        }

        #[inline(always)]
        pub fn splat(token: Avx2FmaToken, v: f64) -> Self {
            let part = f64x4::splat(token, v);
            Self { lo: part, hi: part }
        }

        #[inline(always)]
        pub fn zero(token: Avx2FmaToken) -> Self {
            let part = f64x4::zero(token);
            Self { lo: part, hi: part }
        }

        #[inline(always)]
        pub fn store(self, out: &mut [f64; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            self.lo.store(lo.try_into().unwrap());
            self.hi.store(hi.try_into().unwrap());
        }

        #[inline(always)]
        pub fn to_array(self) -> [f64; 8] {
            let mut out = [0.0f64; 8];
            self.store(&mut out);
            out
        }
    }

    impl Add for f64x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for f64x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for f64x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl Div for f64x8 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            Self {
                lo: self.lo / rhs.lo,
                hi: self.hi / rhs.hi,
            }
        }
    }

    impl Neg for f64x8 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self {
                lo: -self.lo,
                hi: -self.hi,
            }
        }
    }

    /// Emulated 16-wide i32 vector using two AVX2 i32x8 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i32x16 {
        lo: i32x8,
        hi: i32x8,
    }

    impl i32x16 {
        pub const LANES: usize = 16;

        #[inline(always)]
        pub fn load(token: Avx2FmaToken, data: &[i32; 16]) -> Self {
            Self {
                lo: i32x8::load(token, data[0..8].try_into().unwrap()),
                hi: i32x8::load(token, data[8..16].try_into().unwrap()),
            }
        }

        #[inline(always)]
        pub fn splat(token: Avx2FmaToken, v: i32) -> Self {
            let part = i32x8::splat(token, v);
            Self { lo: part, hi: part }
        }

        #[inline(always)]
        pub fn zero(token: Avx2FmaToken) -> Self {
            let part = i32x8::zero(token);
            Self { lo: part, hi: part }
        }

        #[inline(always)]
        pub fn store(self, out: &mut [i32; 16]) {
            let (lo, hi) = out.split_at_mut(8);
            self.lo.store(lo.try_into().unwrap());
            self.hi.store(hi.try_into().unwrap());
        }

        #[inline(always)]
        pub fn to_array(self) -> [i32; 16] {
            let mut out = [0i32; 16];
            self.store(&mut out);
            out
        }
    }

    impl Add for i32x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i32x16 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    /// Token type alias
    pub type Token = Avx2FmaToken;
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    //! Polyfilled 256-bit types using NEON (128-bit) operations.
    //!
    //! These types emulate AVX2-width vectors using pairs of NEON vectors.
    //! Allows writing code targeting 256-bit widths on ARM hardware.

    use crate::simd::arm::w128::{f32x4, f64x2, i32x4};
    use archmage::NeonToken;
    use core::ops::{Add, Div, Mul, Neg, Sub};

    /// Emulated 8-wide f32 vector using two NEON f32x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct f32x8 {
        lo: f32x4,
        hi: f32x4,
    }

    impl f32x8 {
        pub const LANES: usize = 8;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: NeonToken, data: &[f32; 8]) -> Self {
            let lo_arr: &[f32; 4] = data[0..4].try_into().unwrap();
            let hi_arr: &[f32; 4] = data[4..8].try_into().unwrap();
            Self {
                lo: f32x4::load(token, lo_arr),
                hi: f32x4::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: NeonToken, v: f32) -> Self {
            Self {
                lo: f32x4::splat(token, v),
                hi: f32x4::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: NeonToken) -> Self {
            Self {
                lo: f32x4::zero(token),
                hi: f32x4::zero(token),
            }
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [f32; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            let lo_arr: &mut [f32; 4] = lo.try_into().unwrap();
            let hi_arr: &mut [f32; 4] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [f32; 8] {
            let mut out = [0.0f32; 8];
            self.store(&mut out);
            out
        }

        /// Element-wise minimum
        #[inline(always)]
        pub fn min(self, other: Self) -> Self {
            Self {
                lo: self.lo.min(other.lo),
                hi: self.hi.min(other.hi),
            }
        }

        /// Element-wise maximum
        #[inline(always)]
        pub fn max(self, other: Self) -> Self {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.max(other.hi),
            }
        }

        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {
            self.max(lo).min(hi)
        }

        /// Square root
        #[inline(always)]
        pub fn sqrt(self) -> Self {
            Self {
                lo: self.lo.sqrt(),
                hi: self.hi.sqrt(),
            }
        }

        /// Absolute value
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self {
                lo: self.lo.abs(),
                hi: self.hi.abs(),
            }
        }

        /// Floor
        #[inline(always)]
        pub fn floor(self) -> Self {
            Self {
                lo: self.lo.floor(),
                hi: self.hi.floor(),
            }
        }

        /// Ceil
        #[inline(always)]
        pub fn ceil(self) -> Self {
            Self {
                lo: self.lo.ceil(),
                hi: self.hi.ceil(),
            }
        }

        /// Round
        #[inline(always)]
        pub fn round(self) -> Self {
            Self {
                lo: self.lo.round(),
                hi: self.hi.round(),
            }
        }

        /// Fused multiply-add: self * a + b
        /// NEON has native FMA, so this is efficient
        #[inline(always)]
        pub fn mul_add(self, a: Self, b: Self) -> Self {
            Self {
                lo: self.lo.mul_add(a.lo, b.lo),
                hi: self.hi.mul_add(a.hi, b.hi),
            }
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> f32 {
            self.lo.reduce_add() + self.hi.reduce_add()
        }

        /// Reduce: max of all lanes
        #[inline(always)]
        pub fn reduce_max(self) -> f32 {
            self.lo.reduce_max().max(self.hi.reduce_max())
        }

        /// Reduce: min of all lanes
        #[inline(always)]
        pub fn reduce_min(self) -> f32 {
            self.lo.reduce_min().min(self.hi.reduce_min())
        }
    }

    impl Add for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl Div for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            Self {
                lo: self.lo / rhs.lo,
                hi: self.hi / rhs.hi,
            }
        }
    }

    impl Neg for f32x8 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self {
                lo: -self.lo,
                hi: -self.hi,
            }
        }
    }

    impl core::ops::AddAssign for f32x8 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for f32x8 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for f32x8 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl core::ops::DivAssign for f32x8 {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }

    /// Emulated 4-wide f64 vector using two NEON f64x2 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct f64x4 {
        lo: f64x2,
        hi: f64x2,
    }

    impl f64x4 {
        pub const LANES: usize = 4;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: NeonToken, data: &[f64; 4]) -> Self {
            let lo_arr: &[f64; 2] = data[0..2].try_into().unwrap();
            let hi_arr: &[f64; 2] = data[2..4].try_into().unwrap();
            Self {
                lo: f64x2::load(token, lo_arr),
                hi: f64x2::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: NeonToken, v: f64) -> Self {
            Self {
                lo: f64x2::splat(token, v),
                hi: f64x2::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: NeonToken) -> Self {
            Self {
                lo: f64x2::zero(token),
                hi: f64x2::zero(token),
            }
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [f64; 4]) {
            let (lo, hi) = out.split_at_mut(2);
            let lo_arr: &mut [f64; 2] = lo.try_into().unwrap();
            let hi_arr: &mut [f64; 2] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [f64; 4] {
            let mut out = [0.0f64; 4];
            self.store(&mut out);
            out
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> f64 {
            self.lo.reduce_add() + self.hi.reduce_add()
        }
    }

    impl Add for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    impl Mul for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                lo: self.lo * rhs.lo,
                hi: self.hi * rhs.hi,
            }
        }
    }

    impl Div for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            Self {
                lo: self.lo / rhs.lo,
                hi: self.hi / rhs.hi,
            }
        }
    }

    impl Neg for f64x4 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self {
                lo: -self.lo,
                hi: -self.hi,
            }
        }
    }

    /// Emulated 8-wide i32 vector using two NEON i32x4 vectors.
    #[derive(Clone, Copy, Debug)]
    pub struct i32x8 {
        lo: i32x4,
        hi: i32x4,
    }

    impl i32x8 {
        pub const LANES: usize = 8;

        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(token: NeonToken, data: &[i32; 8]) -> Self {
            let lo_arr: &[i32; 4] = data[0..4].try_into().unwrap();
            let hi_arr: &[i32; 4] = data[4..8].try_into().unwrap();
            Self {
                lo: i32x4::load(token, lo_arr),
                hi: i32x4::load(token, hi_arr),
            }
        }

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(token: NeonToken, v: i32) -> Self {
            Self {
                lo: i32x4::splat(token, v),
                hi: i32x4::splat(token, v),
            }
        }

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(token: NeonToken) -> Self {
            Self {
                lo: i32x4::zero(token),
                hi: i32x4::zero(token),
            }
        }

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [i32; 8]) {
            let (lo, hi) = out.split_at_mut(4);
            let lo_arr: &mut [i32; 4] = lo.try_into().unwrap();
            let hi_arr: &mut [i32; 4] = hi.try_into().unwrap();
            self.lo.store(lo_arr);
            self.hi.store(hi_arr);
        }

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [i32; 8] {
            let mut out = [0i32; 8];
            self.store(&mut out);
            out
        }

        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> i32 {
            self.lo.reduce_add() + self.hi.reduce_add()
        }
    }

    impl Add for i32x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                lo: self.lo + rhs.lo,
                hi: self.hi + rhs.hi,
            }
        }
    }

    impl Sub for i32x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                lo: self.lo - rhs.lo,
                hi: self.hi - rhs.hi,
            }
        }
    }

    /// Token type alias for NEON polyfill module
    pub type Token = NeonToken;

    /// Lane count constants for polyfilled types
    pub const LANES_F32: usize = 8;
    pub const LANES_F64: usize = 4;
    pub const LANES_32: usize = 8;

    /// Alias for the polyfilled type (same API as avx2::f32xN)
    pub type f32xN = f32x8;
    pub type f64xN = f64x4;
    pub type i32xN = i32x8;
}

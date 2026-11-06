//! Private four-lane binary64 arithmetic. No detection or Magetypes dependency.
use core::ops::{Add, Mul, Sub};
pub(super) trait Backend: Copy {
    type Raw: Copy;
    fn load(self, a: [f64; 4]) -> Self::Raw;
    fn store(self, a: Self::Raw) -> [f64; 4];
    fn add(self, a: Self::Raw, b: Self::Raw) -> Self::Raw;
    fn sub(self, a: Self::Raw, b: Self::Raw) -> Self::Raw;
    fn mul(self, a: Self::Raw, b: Self::Raw) -> Self::Raw;
    fn fma(self, a: Self::Raw, b: Self::Raw, c: Self::Raw) -> Self::Raw;
    fn floor(self, a: Self::Raw) -> Self::Raw;
}
#[derive(Clone, Copy)]
pub(super) struct V<T: Backend> {
    raw: T::Raw,
    token: T,
}
impl<T: Backend> V<T> {
    #[inline(always)]
    pub fn from_array_t(token: T, a: [f64; 4]) -> Self {
        Self {
            raw: token.load(a),
            token,
        }
    }
    #[inline(always)]
    pub fn splat_t(token: T, x: f64) -> Self {
        Self::from_array_t(token, [x; 4])
    }
    #[inline(always)]
    pub fn to_array(self) -> [f64; 4] {
        self.token.store(self.raw)
    }
    #[inline(always)]
    pub fn mul_add_fused(self, b: Self, c: Self) -> Self {
        Self {
            raw: self.token.fma(self.raw, b.raw, c.raw),
            token: self.token,
        }
    }
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self {
            raw: self.token.floor(self.raw),
            token: self.token,
        }
    }
}
macro_rules! op {
    ($tr:ident, $method:ident) => {
        impl<T: Backend> $tr for V<T> {
            type Output = Self;
            #[inline(always)]
            fn $method(self, b: Self) -> Self {
                Self {
                    raw: self.token.$method(self.raw, b.raw),
                    token: self.token,
                }
            }
        }
    };
}
op!(Add, add);
op!(Sub, sub);
op!(Mul, mul);
impl Backend for archmage::ScalarToken {
    type Raw = [f64; 4];
    #[inline(always)]
    fn load(self, a: Self::Raw) -> Self::Raw {
        a
    }
    #[inline(always)]
    fn store(self, a: Self::Raw) -> Self::Raw {
        a
    }
    #[inline(always)]
    fn add(self, a: Self::Raw, b: Self::Raw) -> Self::Raw {
        core::array::from_fn(|i| a[i] + b[i])
    }
    #[inline(always)]
    fn sub(self, a: Self::Raw, b: Self::Raw) -> Self::Raw {
        core::array::from_fn(|i| a[i] - b[i])
    }
    #[inline(always)]
    fn mul(self, a: Self::Raw, b: Self::Raw) -> Self::Raw {
        core::array::from_fn(|i| a[i] * b[i])
    }
    #[inline(always)]
    fn fma(self, a: Self::Raw, b: Self::Raw, c: Self::Raw) -> Self::Raw {
        super::soft_fma::fma_soft_lanes(a, b, c)
    }
    #[inline(always)]
    fn floor(self, a: Self::Raw) -> Self::Raw {
        a.map(crate::scalar::floor)
    }
}

#[cfg(target_arch = "x86_64")]
mod v3 {
    use super::Backend;
    use archmage::X64V3Token;
    use archmage::intrinsics::x86_64::*;
    type Raw = __m256d;
    #[archmage::attune(wrap)]
    fn load(_token: X64V3Token, a: [f64; 4]) -> Raw {
        _mm256_loadu_pd(&a)
    }
    #[archmage::attune(wrap)]
    fn store(_token: X64V3Token, a: Raw) -> [f64; 4] {
        let mut out = [0.; 4];
        _mm256_storeu_pd(&mut out, a);
        out
    }
    #[archmage::attune(wrap)]
    fn add(_token: X64V3Token, a: Raw, b: Raw) -> Raw {
        _mm256_add_pd(a, b)
    }
    #[archmage::attune(wrap)]
    fn sub(_token: X64V3Token, a: Raw, b: Raw) -> Raw {
        _mm256_sub_pd(a, b)
    }
    #[archmage::attune(wrap)]
    fn mul(_token: X64V3Token, a: Raw, b: Raw) -> Raw {
        _mm256_mul_pd(a, b)
    }
    #[archmage::attune(wrap)]
    fn fma(_token: X64V3Token, a: Raw, b: Raw, c: Raw) -> Raw {
        _mm256_fmadd_pd(a, b, c)
    }
    #[archmage::attune(wrap)]
    fn floor(_token: X64V3Token, a: Raw) -> Raw {
        _mm256_floor_pd(a)
    }
    impl Backend for X64V3Token {
        type Raw = Raw;
        #[inline(always)]
        fn load(self, a: [f64; 4]) -> Raw {
            load(self, a)
        }
        #[inline(always)]
        fn store(self, a: Raw) -> [f64; 4] {
            store(self, a)
        }
        #[inline(always)]
        fn add(self, a: Raw, b: Raw) -> Raw {
            add(self, a, b)
        }
        #[inline(always)]
        fn sub(self, a: Raw, b: Raw) -> Raw {
            sub(self, a, b)
        }
        #[inline(always)]
        fn mul(self, a: Raw, b: Raw) -> Raw {
            mul(self, a, b)
        }
        #[inline(always)]
        fn fma(self, a: Raw, b: Raw, c: Raw) -> Raw {
            fma(self, a, b, c)
        }
        #[inline(always)]
        fn floor(self, a: Raw) -> Raw {
            floor(self, a)
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::Backend;
    use archmage::NeonToken;
    use archmage::intrinsics::aarch64::*;
    type Raw = [float64x2_t; 2];
    #[archmage::attune(wrap)]
    fn load(_token: NeonToken, a: [f64; 4]) -> Raw {
        [vld1q_f64(&[a[0], a[1]]), vld1q_f64(&[a[2], a[3]])]
    }
    #[archmage::attune(wrap)]
    fn store(_token: NeonToken, a: Raw) -> [f64; 4] {
        let mut lo = [0.; 2];
        let mut hi = [0.; 2];
        vst1q_f64(&mut lo, a[0]);
        vst1q_f64(&mut hi, a[1]);
        [lo[0], lo[1], hi[0], hi[1]]
    }
    #[archmage::attune(wrap)]
    fn add(_token: NeonToken, a: Raw, b: Raw) -> Raw {
        [vaddq_f64(a[0], b[0]), vaddq_f64(a[1], b[1])]
    }
    #[archmage::attune(wrap)]
    fn sub(_token: NeonToken, a: Raw, b: Raw) -> Raw {
        [vsubq_f64(a[0], b[0]), vsubq_f64(a[1], b[1])]
    }
    #[archmage::attune(wrap)]
    fn mul(_token: NeonToken, a: Raw, b: Raw) -> Raw {
        [vmulq_f64(a[0], b[0]), vmulq_f64(a[1], b[1])]
    }
    #[archmage::attune(wrap)]
    fn fma(_token: NeonToken, a: Raw, b: Raw, c: Raw) -> Raw {
        [vfmaq_f64(c[0], a[0], b[0]), vfmaq_f64(c[1], a[1], b[1])]
    }
    #[archmage::attune(wrap)]
    fn floor(_token: NeonToken, a: Raw) -> Raw {
        [vrndmq_f64(a[0]), vrndmq_f64(a[1])]
    }
    impl Backend for NeonToken {
        type Raw = Raw;
        #[inline(always)]
        fn load(self, a: [f64; 4]) -> Raw {
            load(self, a)
        }
        #[inline(always)]
        fn store(self, a: Raw) -> [f64; 4] {
            store(self, a)
        }
        #[inline(always)]
        fn add(self, a: Raw, b: Raw) -> Raw {
            add(self, a, b)
        }
        #[inline(always)]
        fn sub(self, a: Raw, b: Raw) -> Raw {
            sub(self, a, b)
        }
        #[inline(always)]
        fn mul(self, a: Raw, b: Raw) -> Raw {
            mul(self, a, b)
        }
        #[inline(always)]
        fn fma(self, a: Raw, b: Raw, c: Raw) -> Raw {
            fma(self, a, b, c)
        }
        #[inline(always)]
        fn floor(self, a: Raw) -> Raw {
            floor(self, a)
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::Backend;
    use archmage::Wasm128Token;
    use archmage::intrinsics::wasm32::*;
    type Raw = [v128; 2];
    #[archmage::attune(wrap)]
    fn load(_token: Wasm128Token, a: [f64; 4]) -> Raw {
        [f64x2(a[0], a[1]), f64x2(a[2], a[3])]
    }
    #[archmage::attune(wrap)]
    fn store(_token: Wasm128Token, a: Raw) -> [f64; 4] {
        [
            f64x2_extract_lane::<0>(a[0]),
            f64x2_extract_lane::<1>(a[0]),
            f64x2_extract_lane::<0>(a[1]),
            f64x2_extract_lane::<1>(a[1]),
        ]
    }
    #[archmage::attune(wrap)]
    fn add(_token: Wasm128Token, a: Raw, b: Raw) -> Raw {
        [f64x2_add(a[0], b[0]), f64x2_add(a[1], b[1])]
    }
    #[archmage::attune(wrap)]
    fn sub(_token: Wasm128Token, a: Raw, b: Raw) -> Raw {
        [f64x2_sub(a[0], b[0]), f64x2_sub(a[1], b[1])]
    }
    #[archmage::attune(wrap)]
    fn mul(_token: Wasm128Token, a: Raw, b: Raw) -> Raw {
        [f64x2_mul(a[0], b[0]), f64x2_mul(a[1], b[1])]
    }
    #[archmage::attune(wrap)]
    fn floor(_token: Wasm128Token, a: Raw) -> Raw {
        [f64x2_floor(a[0]), f64x2_floor(a[1])]
    }
    #[archmage::attune(wrap)]
    fn fma(_token: Wasm128Token, a: Raw, b: Raw, c: Raw) -> Raw {
        let aa = store(_token, a);
        let bb = store(_token, b);
        let cc = store(_token, c);
        load(_token, super::super::soft_fma::fma_soft_lanes(aa, bb, cc))
    }
    impl Backend for Wasm128Token {
        type Raw = Raw;
        #[inline(always)]
        fn load(self, a: [f64; 4]) -> Raw {
            load(self, a)
        }
        #[inline(always)]
        fn store(self, a: Raw) -> [f64; 4] {
            store(self, a)
        }
        #[inline(always)]
        fn add(self, a: Raw, b: Raw) -> Raw {
            add(self, a, b)
        }
        #[inline(always)]
        fn sub(self, a: Raw, b: Raw) -> Raw {
            sub(self, a, b)
        }
        #[inline(always)]
        fn mul(self, a: Raw, b: Raw) -> Raw {
            mul(self, a, b)
        }
        #[inline(always)]
        fn fma(self, a: Raw, b: Raw, c: Raw) -> Raw {
            fma(self, a, b, c)
        }
        #[inline(always)]
        fn floor(self, a: Raw) -> Raw {
            floor(self, a)
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl Backend for archmage::X64V4xToken {
    type Raw = <archmage::X64V3Token as Backend>::Raw;
    #[inline(always)]
    fn load(self, a: [f64; 4]) -> Self::Raw {
        self.v3().load(a)
    }
    #[inline(always)]
    fn store(self, a: Self::Raw) -> [f64; 4] {
        self.v3().store(a)
    }
    #[inline(always)]
    fn add(self, a: Self::Raw, b: Self::Raw) -> Self::Raw {
        self.v3().add(a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Raw, b: Self::Raw) -> Self::Raw {
        self.v3().sub(a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Raw, b: Self::Raw) -> Self::Raw {
        self.v3().mul(a, b)
    }
    #[inline(always)]
    fn fma(self, a: Self::Raw, b: Self::Raw, c: Self::Raw) -> Self::Raw {
        self.v3().fma(a, b, c)
    }
    #[inline(always)]
    fn floor(self, a: Self::Raw) -> Self::Raw {
        self.v3().floor(a)
    }
}

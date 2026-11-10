//! Study adapter using existing public vector FMA primitives; no new vector API.
#![allow(dead_code)] // Width and precision use depends on the selected consumer.
use archmage::ScalarToken;

#[inline(always)]
pub(crate) fn fmaf_soft_lanes<const N: usize>(a: [f32; N], b: [f32; N], c: [f32; N]) -> [f32; N] {
    let mut out = [0.; N];
    macro_rules! width {
        ($n:literal,$v:ident) => {
            if N == $n {
                use crate::simd::generic::$v as V;
                let r = V::from_array_t(ScalarToken, core::array::from_fn(|i| a[i]))
                    .mul_add_fused(
                        V::from_array_t(ScalarToken, core::array::from_fn(|i| b[i])),
                        V::from_array_t(ScalarToken, core::array::from_fn(|i| c[i])),
                    )
                    .to_array();
                out.copy_from_slice(&r);
                return out;
            }
        };
    }
    width!(4, f32x4);
    width!(8, f32x8);
    #[cfg(feature = "w512")]
    width!(16, f32x16);
    use crate::simd::generic::f32x4 as V;
    for i in 0..N {
        out[i] = V::splat_t(ScalarToken, a[i])
            .mul_add_fused(V::splat_t(ScalarToken, b[i]), V::splat_t(ScalarToken, c[i]))
            .to_array()[0];
    }
    out
}

#[inline(always)]
pub(crate) fn fma_lanes<const N: usize>(a: [f64; N], b: [f64; N], c: [f64; N]) -> [f64; N] {
    let mut out = [0.; N];
    macro_rules! width {
        ($n:literal,$v:ident) => {
            if N == $n {
                use crate::simd::generic::$v as V;
                let r = V::from_array_t(ScalarToken, core::array::from_fn(|i| a[i]))
                    .mul_add_fused(
                        V::from_array_t(ScalarToken, core::array::from_fn(|i| b[i])),
                        V::from_array_t(ScalarToken, core::array::from_fn(|i| c[i])),
                    )
                    .to_array();
                out.copy_from_slice(&r);
                return out;
            }
        };
    }
    width!(2, f64x2);
    width!(4, f64x4);
    #[cfg(feature = "w512")]
    width!(8, f64x8);
    use crate::simd::generic::f64x2 as V;
    for i in 0..N {
        out[i] = V::splat_t(ScalarToken, a[i])
            .mul_add_fused(V::splat_t(ScalarToken, b[i]), V::splat_t(ScalarToken, c[i]))
            .to_array()[0];
    }
    out
}

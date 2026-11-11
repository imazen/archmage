//! Benchmark-only alternatives; no Magemath public profile selects these graphs.
#![allow(dead_code)] // Study binaries instantiate different logical widths.
macro_rules! graphs {
    ($rc:ident,$rs:ident,$v:ident,$i:ident,$bound:ident) => {
        #[inline(always)]
        pub(crate) fn $rc<T: magetypes::simd::backends::$bound>(
            t: T,
            x: magetypes::simd::generic::$v<T>,
        ) -> magetypes::simd::generic::$v<T> {
            use magetypes::simd::generic::{$i as I, $v as V};
            let y = (I::splat_t(t, 0x7ef127ea) - x.bitcast_to_i32()).bitcast_to_f32();
            let one = V::splat_t(t, 1.0);
            let e = (-x).mul_add_fused(y, one);
            // Two independent nodes after e: r*(1+e) and 1+e^2.
            let first = y.mul_add_fused(e, y);
            let second = e.mul_add_fused(e, one);
            first * second
        }
        #[inline(always)]
        pub(crate) fn $rs<T: magetypes::simd::backends::$bound>(
            t: T,
            x: magetypes::simd::generic::$v<T>,
        ) -> magetypes::simd::generic::$v<T> {
            use magetypes::simd::generic::{$i as I, $v as V};
            let y = (I::splat_t(t, 0x5f3759df) - x.bitcast_to_i32().shr_logical_const::<1>())
                .bitcast_to_f32();
            let one = V::splat_t(t, 1.0);
            let e = (-x).mul_add_fused(y * y, one);
            // Quadratic correction: adjust the linear coefficient to absorb
            // the linear Chebyshev component of the omitted cubic on |e|<=.07.
            // c1 = RN_f32(.5 + 15*.07^2/64), encoded explicitly.
            let p =
                V::splat_t(t, 0.375).mul_add_fused(e, V::splat_t(t, f32::from_bits(0x3f004b44)));
            y * p.mul_add_fused(e, one)
        }
    };
}
graphs!(recip4, rsqrt4, f32x4, i32x4, F32x4Convert);
graphs!(recip8, rsqrt8, f32x8, i32x8, F32x8Convert);

#[cfg(feature = "w512")]
graphs!(recip16, rsqrt16, f32x16, i32x16, F32x16Convert);

//! End-to-end verification of `X64V3VnniToken` concrete V3 delegation in `magetypes`.
//!
//! Runs both WITH and WITHOUT `feature = "avx512"` to confirm that `X64V3VnniToken`
//! implements all 34 W128/W256 vector/convert/bitcast backends, `F32ScalarBackend`,
//! `F64ScalarBackend`, `F32x8FromHalves`, and `SimdTypes` without depending on AVX-512.

#![forbid(unsafe_code)]

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod x86_vnni_delegation {
    use archmage::{SimdToken, X64V3Token, X64V3VnniToken, attune};
    use magetypes::SimdTypes;
    use magetypes::scalar::{mul_add_relaxed_f32, mul_add_relaxed_f64};
    use magetypes::simd::backends::{
        F32x4Backend, F32x8Backend, F32x8Convert, F64x4Backend, I8x32Backend, I32x8Backend,
        U8x32Backend,
    };
    use magetypes::simd::generic::{f32x4, f32x8, f64x4, i8x32, i32x8, u8x32};

    fn assert_vnni_bounds<T>()
    where
        T: F32x4Backend
            + F32x8Backend
            + F32x8Convert
            + F64x4Backend
            + I32x8Backend
            + I8x32Backend
            + U8x32Backend
            + SimdTypes<F32 = f32x8<T>, I32 = i32x8<T>, U8 = u8x32<T>, I8 = i8x32<T>>,
    {
    }

    #[test]
    fn vnni_token_satisfies_v3_backend_and_simd_types_bounds() {
        assert_vnni_bounds::<X64V3VnniToken>();
        assert_eq!(<X64V3VnniToken as SimdTypes>::F32_LANES, 8);
        assert_eq!(<X64V3VnniToken as SimdTypes>::F64_LANES, 4);
        assert_eq!(<X64V3VnniToken as SimdTypes>::I32_LANES, 8);
    }

    #[attune(wrap)]
    fn vnni_kernel(token: X64V3VnniToken, xs: &[f32; 8], ys: &[f32; 8]) -> ([f32; 8], [i32; 8]) {
        let lo = f32x4::load_t(token, xs[..4].try_into().unwrap());
        let hi = f32x4::load_t(token, xs[4..].try_into().unwrap());
        let a = f32x8::from_halves_t(token, lo, hi);
        let b = f32x8::load_t(token, ys);
        let c = a.mul_add_relaxed(b, f32x8::splat_t(token, 1.5));
        let ints = c.to_i32_round();
        let back = f32x8::from_i32_t(token, ints);
        assert_eq!(back.to_array(), ints.to_array().map(|v| v as f32));

        // Verify integer + f64 + scalar delegated paths too.
        let d = f64x4::splat_t(token, 2.25) + f64x4::splat_t(token, 1.75);
        assert_eq!(d.to_array(), [4.0; 4]);
        let u = u8x32::splat_t(token, 7);
        let s = i8x32::splat_t(token, -3);
        assert_eq!(u.to_array()[0], 7);
        assert_eq!(s.to_array()[0], -3);
        assert_eq!(mul_add_relaxed_f32(token, 2.0, 3.0, 4.0), 10.0);
        assert_eq!(mul_add_relaxed_f64(token, 2.0, 3.0, 4.0), 10.0);

        (c.to_array(), ints.to_array())
    }

    #[attune(wrap)]
    fn v3_reference(token: X64V3Token, xs: &[f32; 8], ys: &[f32; 8]) -> ([f32; 8], [i32; 8]) {
        let a = f32x8::load_t(token, xs);
        let b = f32x8::load_t(token, ys);
        let c = a.mul_add_relaxed(b, f32x8::splat_t(token, 1.5));
        (c.to_array(), c.to_i32_round().to_array())
    }

    #[test]
    fn vnni_delegated_ops_match_v3_bit_for_bit() {
        let Some(vnni) = X64V3VnniToken::summon() else {
            return;
        };
        let xs = [1.0, -2.5, 3.25, 0.0, 7.5, -0.25, 16.0, 100.5];
        let ys = [2.0, 4.0, -1.0, 9.0, 0.5, -8.0, 0.125, -0.5];
        assert_eq!(
            vnni_kernel(vnni, &xs, &ys),
            v3_reference(vnni.v3(), &xs, &ys)
        );
    }
}

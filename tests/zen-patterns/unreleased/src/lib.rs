//! Integration patterns that depend on features introduced in 0.9.32
//! (`magemath` with caller-proof SIMD tails, `#[attune(..., suppress_const_test)]`,
//! `f32` `min_fast`/`max_fast`/`to_i32_fast`, 3-channel RGB block helpers,
//! `load_padded_t`/`store_prefix`, `magetypes::numerics` const-configured
//! conversions/reciprocals/extrema, `u16`/`u32`/`f16`/`bf16` layouts,
//! width-polymorphic `RGBA` + alpha preservation across `x4`/`x8`/`x16`,
//! `trunc`, and `mul_neg_add_{fused,relaxed}`).
//!
//! Patterns that already compile on published 0.9.31 live in `../released`.

pub mod attune_suppress_const {
    //! Code-generator proof wrappers moving from `#[arcane(suppress_const_test)]`
    //! to `#[attune]` (`wrap` and explicit `_t` suffix).

    #[archmage::attune(wrap, suppress_const_test)]
    pub fn dot4_wrap(token: ::archmage::X64V3Token, a: &[f32; 4], b: &[f32; 4]) -> f32 {
        let va = magetypes::simd::generic::f32x4::load_t(token, a);
        let vb = magetypes::simd::generic::f32x4::load_t(token, b);
        (va * vb).reduce_add()
    }

    #[archmage::attune(suppress_const_test)]
    pub fn scale4_v3_t(token: ::archmage::X64V3Token, x: &mut [f32; 4], k: f32) {
        let v = magetypes::simd::generic::f32x4::load_t(token, x);
        (v * magetypes::simd::generic::f32x4::splat_t(token, k)).store(x);
    }

    #[archmage::attune(wrap, suppress_const_test)]
    pub fn dot4_scalar_wrap(token: ::archmage::ScalarToken, a: &[f32; 4], b: &[f32; 4]) -> f32 {
        let va = magetypes::simd::generic::f32x4::load_t(token, a);
        let vb = magetypes::simd::generic::f32x4::load_t(token, b);
        (va * vb).reduce_add()
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn suppress_const_test_scalar_wrapper_runs_everywhere() {
            let d = super::dot4_scalar_wrap(
                archmage::ScalarToken,
                &[0.5, -1.0, 1.5, 2.0],
                &[2.0, 4.0, 6.0, 8.0],
            );
            assert_eq!(d, 22.0);
        }

        #[cfg(target_arch = "x86_64")]
        #[test]
        fn suppress_const_test_wrappers_execute_and_preserve_results() {
            use archmage::SimdToken;
            let _lock = archmage::testing::lock_token_testing();
            let token = archmage::X64V3Token::summon().expect("x86-64 test host without AVX2+FMA");
            let mut x = [1.0, -2.0, 3.0, 4.0];
            super::scale4_v3_t(token, &mut x, 0.5);
            assert_eq!(x, [0.5, -1.0, 1.5, 2.0]);
            let d = super::dot4_wrap(token, &x, &[2.0, 4.0, 6.0, 8.0]);
            assert_eq!(d, 22.0);
        }
    }
}

pub mod magemath_tails {
    //! Vector chunks (`SimdMath` inline methods + `.token()`) and `*_with_token`
    //! scalar tails sharing one fluent `const CFG: u128` configuration inside
    //! an `#[attune(_*_t, ...)]` family.

    use magemath::SimdMath;
    use magemath::cbrtf::{Cbrtf, CbrtfError, CbrtfOptions};
    use magemath::expf::{Expf, ExpfError, ExpfOptions};
    use magemath::logf::{Logf, LogfError, LogfOptions};
    use magemath::powf::{Powf, PowfError, PowfOptions};

    pub const POWF_CFG: u128 = PowfOptions::new().gamma().midp().caller().bits();
    pub const EXPF_CFG: u128 = ExpfOptions::new().midp().caller().bits();
    pub const LOGF_CFG: u128 = LogfOptions::new().midp().caller().bits();
    pub const CBRTF_CFG: u128 = CbrtfOptions::new()
        .reject_subnormals()
        .midp()
        .caller()
        .bits();

    #[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
    fn powf_slice_inner(
        token: Token,
        src: &[f32],
        dst: &mut [f32],
        exp: f32,
    ) -> Result<(), PowfError> {
        assert_eq!(src.len(), dst.len(), "powf slice lengths differ");
        let (sc, st) = f32x8::partition_slice_t(token, src);
        let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
        for (i, o) in sc.iter().zip(dc.iter_mut()) {
            let loaded = f32x8::load_t(token, i);
            let _same_tok: Token = loaded.token();
            loaded.try_powf_with::<POWF_CFG>(exp)?.store(o);
        }
        for (&x, o) in st.iter().zip(dt.iter_mut()) {
            *o = Powf::<POWF_CFG>::scalar_with_token(token, x, exp)?;
        }
        Ok(())
    }

    pub fn powf_slice(src: &[f32], dst: &mut [f32], exp: f32) -> Result<(), PowfError> {
        archmage::attuned!(
            powf_slice_inner(src, dst, exp),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        )
    }

    #[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
    fn exp2_slice_inner(token: Token, src: &[f32], dst: &mut [f32]) -> Result<(), ExpfError> {
        assert_eq!(src.len(), dst.len(), "exp2 slice lengths differ");
        let (sc, st) = f32x8::partition_slice_t(token, src);
        let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
        for (i, o) in sc.iter().zip(dc.iter_mut()) {
            f32x8::load_t(token, i)
                .try_exp2_with::<EXPF_CFG>()?
                .store(o);
        }
        for (&x, o) in st.iter().zip(dt.iter_mut()) {
            *o = Expf::<EXPF_CFG>::exp2_with_token(token, x)?;
        }
        Ok(())
    }

    pub fn exp2_slice(src: &[f32], dst: &mut [f32]) -> Result<(), ExpfError> {
        archmage::attuned!(
            exp2_slice_inner(src, dst),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        )
    }

    #[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
    fn log2_slice_inner(token: Token, src: &[f32], dst: &mut [f32]) -> Result<(), LogfError> {
        assert_eq!(src.len(), dst.len(), "log2 slice lengths differ");
        let (sc, st) = f32x8::partition_slice_t(token, src);
        let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
        for (i, o) in sc.iter().zip(dc.iter_mut()) {
            f32x8::load_t(token, i)
                .try_log2_with::<LOGF_CFG>()?
                .store(o);
        }
        for (&x, o) in st.iter().zip(dt.iter_mut()) {
            *o = Logf::<LOGF_CFG>::log2_with_token(token, x)?;
        }
        Ok(())
    }

    pub fn log2_slice(src: &[f32], dst: &mut [f32]) -> Result<(), LogfError> {
        archmage::attuned!(
            log2_slice_inner(src, dst),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        )
    }

    #[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
    fn cbrt_slice_inner(token: Token, src: &[f32], dst: &mut [f32]) -> Result<(), CbrtfError> {
        assert_eq!(src.len(), dst.len(), "cbrt slice lengths differ");
        let (sc, st) = f32x8::partition_slice_t(token, src);
        let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
        for (i, o) in sc.iter().zip(dc.iter_mut()) {
            f32x8::load_t(token, i)
                .try_cbrt_with::<CBRTF_CFG>()?
                .store(o);
        }
        for (&x, o) in st.iter().zip(dt.iter_mut()) {
            *o = Cbrtf::<CBRTF_CFG>::scalar_with_token(token, x)?;
        }
        Ok(())
    }

    pub fn cbrt_slice(src: &[f32], dst: &mut [f32]) -> Result<(), CbrtfError> {
        archmage::attuned!(
            cbrt_slice_inner(src, dst),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use archmage::ScalarToken;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
        use magemath::cbrtf::CbrtfErrorKind;
        use magemath::expf::ExpfErrorKind;
        use magemath::logf::LogfErrorKind;
        use magemath::powf::PowfErrorKind;

        #[test]
        fn vector_chunks_and_scalar_tails_match_bit_for_bit_across_tiers() {
            // 29 = 3*8 + 5, so lanes 0..5 repeat in the 5-element scalar tail.
            let pos: Vec<f32> = (0..29).map(|i| 0.05 + (i % 8) as f32 * 0.37).collect();
            let pow_in: Vec<f32> = (0..29)
                .map(|i| {
                    if i % 8 == 0 {
                        0.0
                    } else {
                        0.05 + (i % 8) as f32 * 0.37
                    }
                })
                .collect();
            let exp_in: Vec<f32> = (0..29).map(|i| -4.5 + (i % 8) as f32 * 1.25).collect();

            let mut pow_ref = vec![0.0; 29];
            let mut exp_ref = vec![0.0; 29];
            let mut log_ref = vec![0.0; 29];
            let mut cbrt_ref = vec![0.0; 29];
            powf_slice_inner_scalar_t(ScalarToken, &pow_in, &mut pow_ref, 2.2).unwrap();
            exp2_slice_inner_scalar_t(ScalarToken, &exp_in, &mut exp_ref).unwrap();
            log2_slice_inner_scalar_t(ScalarToken, &pos, &mut log_ref).unwrap();
            cbrt_slice_inner_scalar_t(ScalarToken, &pos, &mut cbrt_ref).unwrap();

            // Scalar chunks and tail must agree bit-for-bit before testing hardware tiers.
            for i in 0..5 {
                assert_eq!(pow_ref[i].to_bits(), pow_ref[24 + i].to_bits());
                assert_eq!(exp_ref[i].to_bits(), exp_ref[24 + i].to_bits());
                assert_eq!(log_ref[i].to_bits(), log_ref[24 + i].to_bits());
                assert_eq!(cbrt_ref[i].to_bits(), cbrt_ref[24 + i].to_bits());
            }

            let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_| {
                let (mut p, mut e, mut l, mut c) =
                    (vec![0.0; 29], vec![0.0; 29], vec![0.0; 29], vec![0.0; 29]);
                powf_slice(&pow_in, &mut p, 2.2).unwrap();
                exp2_slice(&exp_in, &mut e).unwrap();
                log2_slice(&pos, &mut l).unwrap();
                cbrt_slice(&pos, &mut c).unwrap();

                // BitExact profiles: every SIMD tier and scalar tail match the reference bits.
                for i in 0..29 {
                    assert_eq!(p[i].to_bits(), pow_ref[i].to_bits(), "powf idx {i}");
                    assert_eq!(e[i].to_bits(), exp_ref[i].to_bits(), "exp2 idx {i}");
                    assert_eq!(l[i].to_bits(), log_ref[i].to_bits(), "log2 idx {i}");
                    assert_eq!(c[i].to_bits(), cbrt_ref[i].to_bits(), "cbrt idx {i}");
                }
            });
        }

        #[test]
        fn whole_vector_domain_rejections_carry_exact_lane_masks() {
            let mut out = [0.0; 9];
            let mut pow_bad = [0.5; 9];
            pow_bad[3] = -0.25;
            assert_eq!(
                powf_slice(&pow_bad, &mut out, 2.2),
                Err(PowfError {
                    kind: PowfErrorKind::InvalidBase,
                    lane_mask: 1 << 3,
                })
            );

            let mut exp_bad = [0.0; 9];
            exp_bad[8] = f32::NAN;
            assert_eq!(
                exp2_slice(&exp_bad, &mut out),
                Err(ExpfError {
                    kind: ExpfErrorKind::InvalidInput,
                    lane_mask: 1,
                })
            );

            let mut log_bad = [1.0; 9];
            log_bad[5] = 0.0;
            assert_eq!(
                log2_slice(&log_bad, &mut out),
                Err(LogfError {
                    kind: LogfErrorKind::InvalidInput,
                    lane_mask: 1 << 5,
                })
            );

            let mut cbrt_bad = [1.0; 9];
            cbrt_bad[2] = 1e-42; // subnormal rejected by the explicit profile
            assert_eq!(
                cbrt_slice(&cbrt_bad, &mut out),
                Err(CbrtfError {
                    kind: CbrtfErrorKind::InvalidInput,
                    lane_mask: 1 << 2,
                })
            );
        }
    }
}

pub mod fast_rgb_ops {
    //! Generic `F32x8Backend + F32x8Convert` RGB kernel combining native
    //! `load_rgb_u8_t`, `interleave_3ch` / `deinterleave_3ch`, `max_fast` /
    //! `min_fast`, `to_i32_fast`, and `store_rgb_u8`, paired with
    //! `magetypes::scalar` tail helpers inside `#[archmage::attune]`.

    use magetypes::scalar;
    use magetypes::simd::backends::{F32x8Backend, F32x8Convert};
    use magetypes::simd::generic::f32x8;

    #[inline(always)]
    pub fn process_rgb8_block<T: F32x8Backend + F32x8Convert>(
        token: T,
        rgb: &[u8; 24],
        scale: f32,
        bias: f32,
    ) -> ([u8; 24], [i32; 8]) {
        let (r, g, b) = f32x8::load_rgb_u8_t(token, rgb);
        // Round-trip planar -> AoS -> planar inside the generic kernel.
        let [r, g, b] = f32x8::deinterleave_3ch(f32x8::interleave_3ch([r, g, b]));
        let s = f32x8::splat_t(token, scale);
        let o = f32x8::splat_t(token, bias);
        let lo = f32x8::zero_t(token);
        let hi = f32x8::splat_t(token, 255.0);

        let r = r.mul_add_portable(s, o).max_fast(lo).min_fast(hi);
        let g = g.mul_add_portable(s, o).max_fast(lo).min_fast(hi);
        let b = b.mul_add_portable(s, o).max_fast(lo).min_fast(hi);

        let quant_r = r.to_i32_fast().to_array();
        let mut out = [0; 24];
        f32x8::store_rgb_u8(r, g, b, &mut out);
        (out, quant_r)
    }

    #[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
    fn transform_rgb_slice_inner(
        token: Token,
        src: &[u8],
        dst: &mut [u8],
        quant_r_out: &mut [i32],
        scale: f32,
        bias: f32,
    ) {
        assert_eq!(src.len(), dst.len());
        assert_eq!(src.len() % 3, 0);
        assert_eq!(quant_r_out.len(), src.len() / 3);

        let (src_chunks, src_tail) = src.as_chunks::<24>();
        let (dst_chunks, dst_tail) = dst.as_chunks_mut::<24>();
        let (q_chunks, q_tail) = quant_r_out.as_chunks_mut::<8>();
        for ((s, d), q) in src_chunks
            .iter()
            .zip(dst_chunks.iter_mut())
            .zip(q_chunks.iter_mut())
        {
            let (packed, qr) = process_rgb8_block(token, s, scale, bias);
            *d = packed;
            *q = qr;
        }

        for ((s_px, d_px), q_slot) in src_tail
            .chunks_exact(3)
            .zip(dst_tail.chunks_exact_mut(3))
            .zip(q_tail.iter_mut())
        {
            let r_raw = scalar::mul_add_fused_f32(s_px[0] as f32, scale, bias);
            let g_raw = scalar::mul_add_fused_f32(s_px[1] as f32, scale, bias);
            let b_raw = scalar::mul_add_fused_f32(s_px[2] as f32, scale, bias);
            let r = scalar::min_fast_f32(token, scalar::max_fast_f32(token, r_raw, 0.0), 255.0);
            let g = scalar::min_fast_f32(token, scalar::max_fast_f32(token, g_raw, 0.0), 255.0);
            let b = scalar::min_fast_f32(token, scalar::max_fast_f32(token, b_raw, 0.0), 255.0);
            *q_slot = scalar::to_i32_fast(token, r);
            d_px[0] = scalar::round_f32(r) as u8;
            d_px[1] = scalar::round_f32(g) as u8;
            d_px[2] = scalar::round_f32(b) as u8;
        }
    }

    pub fn transform_rgb_slice(
        src: &[u8],
        dst: &mut [u8],
        quant_r_out: &mut [i32],
        scale: f32,
        bias: f32,
    ) {
        archmage::attuned!(
            transform_rgb_slice_inner(src, dst, quant_r_out, scale, bias),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }

    #[cfg(test)]
    mod tests {
        use archmage::ScalarToken;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        use super::*;

        #[test]
        fn fast_rgb_kernel_matches_scalar_across_tiers() {
            let _lock = archmage::testing::lock_token_testing();
            let pixels = 11;
            let src: Vec<u8> = (0..(pixels * 3))
                .map(|i| ((i * 29 + 7) & 0xff) as u8)
                .collect();
            let mut ref_dst = vec![0u8; pixels * 3];
            let mut ref_q = vec![0i32; pixels];
            transform_rgb_slice_inner_scalar_t(
                ScalarToken,
                &src,
                &mut ref_dst,
                &mut ref_q,
                1.25,
                -10.0,
            );

            let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_| {
                let mut dst = vec![0u8; pixels * 3];
                let mut q = vec![0i32; pixels];
                transform_rgb_slice(&src, &mut dst, &mut q, 1.25, -10.0);
                assert_eq!(dst, ref_dst);
                assert_eq!(q, ref_q);
            });
        }

        #[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8, f64x4))]
        fn verify_three_tier_minmax_inner(token: Token, a: &[f32; 8], b: &[f32; 8]) {
            let va = f32x8::load_t(token, a);
            let vb = f32x8::load_t(token, b);

            let r_min = va.min(vb).to_array();
            let r_max = va.max(vb).to_array();
            let r_minimum = va.minimum(vb).to_array();
            let r_maximum = va.maximum(vb).to_array();
            let r_min_fast = va.min_fast(vb).to_array();
            let r_max_fast = va.max_fast(vb).to_array();

            for i in 0..8 {
                let (x, y) = (a[i], b[i]);
                if x.is_nan() && y.is_nan() {
                    assert!(r_min[i].is_nan());
                    assert!(r_max[i].is_nan());
                } else {
                    assert_eq!(r_min[i], x.min(y));
                    assert_eq!(r_max[i], x.max(y));
                }
                let exp_min = scalar::minimum_f32(x, y);
                let exp_max = scalar::maximum_f32(x, y);
                if exp_min.is_nan() {
                    assert!(r_minimum[i].is_nan());
                } else {
                    assert_eq!(r_minimum[i].to_bits(), exp_min.to_bits());
                }
                if exp_max.is_nan() {
                    assert!(r_maximum[i].is_nan());
                } else {
                    assert_eq!(r_maximum[i].to_bits(), exp_max.to_bits());
                }

                let s_min_fast = scalar::min_fast_f32(token, x, y);
                let s_max_fast = scalar::max_fast_f32(token, x, y);
                if r_min_fast[i].is_nan() {
                    assert!(s_min_fast.is_nan());
                } else {
                    assert_eq!(r_min_fast[i].to_bits(), s_min_fast.to_bits());
                }
                if r_max_fast[i].is_nan() {
                    assert!(s_max_fast.is_nan());
                } else {
                    assert_eq!(r_max_fast[i].to_bits(), s_max_fast.to_bits());
                }
            }

            let d0 = f64x4::from_array_t(token, [f64::NAN, 5.0, -0.0, 0.0]);
            let d1 = f64x4::from_array_t(token, [5.0, f64::NAN, 0.0, -0.0]);
            assert_eq!(d0.min(d1).to_array()[..2], [5.0, 5.0]);
            assert_eq!(d0.max(d1).to_array()[..2], [5.0, 5.0]);
            let dmn = d0.minimum(d1).to_array();
            let dmx = d0.maximum(d1).to_array();
            assert!(dmn[0].is_nan() && dmn[1].is_nan());
            assert_eq!(dmn[2].to_bits(), scalar::minimum_f64(-0.0, 0.0).to_bits());
            assert_eq!(dmx[3].to_bits(), scalar::maximum_f64(0.0, -0.0).to_bits());
            assert_eq!(
                d0.min_fast(d1).to_array()[2].to_bits(),
                scalar::min_fast_f64(token, -0.0, 0.0).to_bits()
            );
        }

        #[test]
        fn three_tier_minmax_and_scalar_tails_across_tiers() {
            let _lock = archmage::testing::lock_token_testing();
            let a = [
                f32::NAN,
                5.0,
                -0.0,
                0.0,
                f32::NEG_INFINITY,
                f32::INFINITY,
                -3.5,
                7.0,
            ];
            let b = [5.0, f32::NAN, 0.0, -0.0, f32::NAN, 12.0, 2.0, f32::NAN];
            let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_| {
                archmage::attuned!(
                    verify_three_tier_minmax_inner(&a, &b),
                    [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
                );
            });
        }
    }
}

pub mod padded_tails {
    //! Single-vector padded tail handling using `partition_slice_t`,
    //! `partition_slice_mut_t`, `load_padded_t`, and `store_prefix`.

    #[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
    fn affine_clamp_padded_inner(
        token: Token,
        src: &[f32],
        dst: &mut [f32],
        scale: f32,
        bias: f32,
    ) {
        assert_eq!(src.len(), dst.len());
        let s = f32x8::splat_t(token, scale);
        let b = f32x8::splat_t(token, bias);
        let lo = f32x8::zero_t(token);
        let hi = f32x8::splat_t(token, 255.0);

        let (src_chunks, src_tail) = f32x8::partition_slice_t(token, src);
        let (dst_chunks, dst_tail) = f32x8::partition_slice_mut_t(token, dst);
        for (sc, dc) in src_chunks.iter().zip(dst_chunks) {
            f32x8::load_t(token, sc)
                .mul_add_relaxed(s, b)
                .max_fast(lo)
                .min_fast(hi)
                .store(dc);
        }
        if !src_tail.is_empty() {
            f32x8::load_padded_t(token, src_tail, 0.0)
                .mul_add_relaxed(s, b)
                .max_fast(lo)
                .min_fast(hi)
                .store_prefix(dst_tail);
        }
    }

    pub fn affine_clamp_padded(src: &[f32], dst: &mut [f32], scale: f32, bias: f32) {
        archmage::attuned!(
            affine_clamp_padded_inner(src, dst, scale, bias),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }

    #[cfg(test)]
    mod tests {
        use archmage::ScalarToken;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        use super::*;

        #[test]
        fn padded_tails_match_scalar_across_lengths_and_tiers() {
            let _lock = archmage::testing::lock_token_testing();
            for len in [0usize, 1, 7, 8, 13, 24] {
                let src: Vec<f32> = (0..len).map(|i| (i as f32) * 19.5 - 25.0).collect();
                let mut ref_dst = vec![0.0f32; len];
                affine_clamp_padded_inner_scalar_t(ScalarToken, &src, &mut ref_dst, 1.5, 10.0);

                let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_| {
                    let mut dst = vec![-123.0f32; len];
                    affine_clamp_padded(&src, &mut dst, 1.5, 10.0);
                    assert_eq!(dst, ref_dst, "len={len}");
                });
            }
        }
    }
}

pub mod numerics_pixel_layouts {
    //! Stage 2 pixel and half-float consumer patterns mirroring `garb`,
    //! `linear-srgb`, and `zenpixels-convert`:
    //! - `IntConvert` + `from_u16_t` / `to_u16` / `to_u16_with` + `load_rgb_u16_t`
    //! - `HalfConvert` + `from_f16_t` / `from_f16_with_t` / `to_f16_bits[_with]`
    //!   and `from_bf16_t` / `to_bf16[_with]` + matching `magetypes::scalar` tails
    //! - Width-polymorphic `load_rgba_u8_t` / `store_rgba_u8[_with]` across
    //!   `f32x4`, `f32x8`, and `f32x16` (`from_u8_t`, `to_u8`, `{de,}interleave_4ch`)
    //! - Bit-preserving `broadcast_rgba_alpha` + `with_rgba_alpha` unpremultiply
    //! - `store_rgb_u8_with`

    use magetypes::numerics::{HalfConvert, IntConvert, RecipOptions};
    use magetypes::scalar;

    pub const U16_TRUNC_IN_RANGE: u32 = IntConvert::new().trunc().in_range().bits();
    pub const U8_RNE_IN_RANGE: u32 = IntConvert::new().nearest_even().in_range().bits();
    pub const F16_ALL_NORM: u32 = HalfConvert::new().all_floats().normalize().bits();
    pub const BF16_TRUNC_FINITE: u32 = HalfConvert::new().trunc().bits();
    pub const RECIP_CEIL: u32 = RecipOptions::new().nr1_ceil().bits();

    #[archmage::attune(_*_t, +v4(cfg(avx512)), define(f32x4, f32x8, f32x16))]
    fn unpremul_rgba_preserve_alpha_inner(token: Token, src: &[f32], dst: &mut [f32]) {
        assert_eq!(src.len(), dst.len());
        assert_eq!(src.len() % 4, 0);
        let one = f32x16::splat_t(token, 1.0);
        let zero = f32x16::zero_t(token);
        let eps = f32x16::splat_t(token, 1e-6);

        let (sc, st) = f32x16::partition_slice_t(token, src);
        let (dc, dt) = f32x16::partition_slice_mut_t(token, dst);
        for (i, o) in sc.iter().zip(dc.iter_mut()) {
            let px = f32x16::load_t(token, i);
            let alpha = px.broadcast_rgba_alpha();
            let safe_a = alpha.max_fast(eps);
            let inv_a = safe_a.recip_with::<RECIP_CEIL>();
            let active = alpha.simd_gt(zero);
            let rgb = f32x16::blend(active, (px * inv_a).min_fast(one), zero);
            rgb.with_rgba_alpha(px).store(o);
        }

        let (sc4, _) = f32x4::partition_slice_t(token, st);
        let (dc4, _) = f32x4::partition_slice_mut_t(token, dt);
        let one4 = f32x4::splat_t(token, 1.0);
        let zero4 = f32x4::zero_t(token);
        let eps4 = f32x4::splat_t(token, 1e-6);
        for (i, o) in sc4.iter().zip(dc4.iter_mut()) {
            let px = f32x4::load_t(token, i);
            let alpha = px.broadcast_rgba_alpha();
            let safe_a = alpha.max_fast(eps4);
            let inv_a = safe_a.recip_with::<RECIP_CEIL>();
            let active = alpha.simd_gt(zero4);
            let rgb = f32x4::blend(active, (px * inv_a).min_fast(one4), zero4);
            rgb.with_rgba_alpha(px).store(o);
        }
    }

    pub fn unpremul_rgba_preserve_alpha(src: &[f32], dst: &mut [f32]) {
        archmage::attuned!(
            unpremul_rgba_preserve_alpha_inner(src, dst),
            [_v4(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }

    #[archmage::attune(_*_t, +v4(cfg(avx512)), define(f32x4, f32x8, f32x16))]
    fn pixel_and_half_pipeline_inner(
        token: Token,
        u16_src: &[u16],
        u16_dst: &mut [u16],
        f16_dst: &mut [u16],
        bf16_dst: &mut [u16],
        rgba16_src: &[u8; 64],
        rgba16_dst: &mut [u8; 64],
        rgb16_block: &[u16; 24],
        rgb8_block_out: &mut [u8; 24],
    ) {
        assert_eq!(u16_src.len(), u16_dst.len());
        assert_eq!(u16_src.len(), f16_dst.len());
        assert_eq!(u16_src.len(), bf16_dst.len());

        // 16-wide u16 -> f32 -> f16 / bf16 -> f32 -> u16_with + scalar tails.
        let scale = f32x16::splat_t(token, 1.0 / 65535.0);
        let up = f32x16::splat_t(token, 65535.0);
        let half_bias = f32x16::splat_t(token, 0.5);
        let (s16, st) = u16_src.as_chunks::<16>();
        let (u16c, u16t) = u16_dst.as_chunks_mut::<16>();
        let (f16c, f16t) = f16_dst.as_chunks_mut::<16>();
        let (bf16c, bf16t) = bf16_dst.as_chunks_mut::<16>();

        for (((s, u_out), h_out), b_out) in s16
            .iter()
            .zip(u16c.iter_mut())
            .zip(f16c.iter_mut())
            .zip(bf16c.iter_mut())
        {
            let norm = f32x16::from_u16_t(token, s) * scale;
            let h_bits = norm.to_f16_bits_with::<F16_ALL_NORM>();
            let b_bits = norm.to_bf16_with::<BF16_TRUNC_FINITE>();
            *h_out = h_bits;
            *b_out = b_bits;
            let back = (f32x16::from_f16_with_t::<F16_ALL_NORM>(token, &h_bits)
                + f32x16::from_bf16_t(token, &b_bits))
                * f32x16::splat_t(token, 0.5);
            *u_out = back
                .mul_add_fused(up, half_bias)
                .clamp(f32x16::zero_t(token), up)
                .to_u16_with::<U16_TRUNC_IN_RANGE>();
        }

        for (((&s, u_out), h_out), b_out) in st
            .iter()
            .zip(u16t.iter_mut())
            .zip(f16t.iter_mut())
            .zip(bf16t.iter_mut())
        {
            let norm = (s as f32) * (1.0 / 65535.0);
            let h_bits = scalar::to_f16_bits_with::<F16_ALL_NORM>(norm);
            let b_bits = scalar::to_bf16_with::<BF16_TRUNC_FINITE>(norm);
            *h_out = h_bits;
            *b_out = b_bits;
            let back =
                (scalar::from_f16_with::<F16_ALL_NORM>(h_bits) + scalar::from_bf16(b_bits)) * 0.5;
            let scaled = scalar::mul_add_fused_f32(back, 65535.0, 0.5).clamp(0.0, 65535.0);
            *u_out = scalar::to_u16_with::<U16_TRUNC_IN_RANGE>(scaled);
        }

        // Width-polymorphic RGBA across f32x16, f32x8, and f32x4 + interleave_4ch.
        let (r16, g16, b16, a16) = f32x16::load_rgba_u8_t(token, rgba16_src);
        let [r16, g16, b16, a16] =
            f32x16::deinterleave_4ch(f32x16::interleave_4ch([r16, g16, b16, a16]));
        f32x16::store_rgba_u8_with::<U8_RNE_IN_RANGE>(b16, g16, r16, a16, rgba16_dst);

        let raw16 = f32x16::from_u8_t(token, rgba16_src.first_chunk::<16>().unwrap()).to_u8();
        assert_eq!(&raw16, rgba16_src.first_chunk::<16>().unwrap());

        let (r8, g8, b8, a8) =
            f32x8::load_rgba_u8_t(token, rgba16_src.first_chunk::<32>().unwrap());
        let mut tmp8 = [0u8; 32];
        f32x8::store_rgba_u8(b8, g8, r8, a8, &mut tmp8);
        assert_eq!(&tmp8, rgba16_dst.first_chunk::<32>().unwrap());

        let (r4, g4, b4, a4) =
            f32x4::load_rgba_u8_t(token, rgba16_src.first_chunk::<16>().unwrap());
        let mut tmp4 = [0u8; 16];
        f32x4::store_rgba_u8_with::<U8_RNE_IN_RANGE>(b4, g4, r4, a4, &mut tmp4);
        assert_eq!(&tmp4, rgba16_dst.first_chunk::<16>().unwrap());

        // 3-channel RGB16 load -> downscale -> store_rgb_u8_with (x8 and x4).
        let down = f32x8::splat_t(token, 1.0 / 257.0);
        let (rr8, rg8, rb8) = f32x8::load_rgb_u16_t(token, rgb16_block);
        f32x8::store_rgb_u8_with::<U8_RNE_IN_RANGE>(
            rr8 * down,
            rg8 * down,
            rb8 * down,
            rgb8_block_out,
        );
        let down4 = f32x4::splat_t(token, 1.0 / 257.0);
        let (rr4, rg4, rb4) =
            f32x4::load_rgb_u16_t(token, rgb16_block.first_chunk::<12>().unwrap());
        let mut rgb4_out = [0u8; 12];
        f32x4::store_rgb_u8_with::<U8_RNE_IN_RANGE>(
            rr4 * down4,
            rg4 * down4,
            rb4 * down4,
            &mut rgb4_out,
        );
        assert_eq!(&rgb4_out, rgb8_block_out.first_chunk::<12>().unwrap());
    }

    pub fn pixel_and_half_pipeline(
        u16_src: &[u16],
        u16_dst: &mut [u16],
        f16_dst: &mut [u16],
        bf16_dst: &mut [u16],
        rgba16_src: &[u8; 64],
        rgba16_dst: &mut [u8; 64],
        rgb16_block: &[u16; 24],
        rgb8_block_out: &mut [u8; 24],
    ) {
        archmage::attuned!(
            pixel_and_half_pipeline_inner(
                u16_src,
                u16_dst,
                f16_dst,
                bf16_dst,
                rgba16_src,
                rgba16_dst,
                rgb16_block,
                rgb8_block_out
            ),
            [_v4(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use archmage::ScalarToken;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        #[test]
        fn stage2_pixel_and_half_layouts_match_across_tiers() {
            let _lock = archmage::testing::lock_token_testing();
            // 21 = 16 + 5 exercises both the 16-lane vector loop and 5-lane scalar tail.
            let u16_src: Vec<u16> = (0..21).map(|i| (i * 3120) as u16).collect();
            let rgba16_src: [u8; 64] = core::array::from_fn(|i| ((i * 37 + 11) & 0xff) as u8);
            let rgb16_block: [u16; 24] = core::array::from_fn(|i| (i * 2730) as u16);

            let (mut ref_u16, mut ref_f16, mut ref_bf16) =
                (vec![0u16; 21], vec![0u16; 21], vec![0u16; 21]);
            let mut ref_rgba = [0u8; 64];
            let mut ref_rgb8 = [0u8; 24];
            pixel_and_half_pipeline_inner_scalar_t(
                ScalarToken,
                &u16_src,
                &mut ref_u16,
                &mut ref_f16,
                &mut ref_bf16,
                &rgba16_src,
                &mut ref_rgba,
                &rgb16_block,
                &mut ref_rgb8,
            );
            // Verify vector lane 0..5 matches repeated values if run through scalar helpers.
            for i in 0..21 {
                let norm = (u16_src[i] as f32) * (1.0 / 65535.0);
                assert_eq!(ref_f16[i], scalar::to_f16_bits_with::<F16_ALL_NORM>(norm));
                assert_eq!(ref_bf16[i], scalar::to_bf16_with::<BF16_TRUNC_FINITE>(norm));
            }

            let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_| {
                let (mut u16_dst, mut f16_dst, mut bf16_dst) =
                    (vec![0u16; 21], vec![0u16; 21], vec![0u16; 21]);
                let mut rgba_dst = [0u8; 64];
                let mut rgb8_dst = [0u8; 24];
                pixel_and_half_pipeline(
                    &u16_src,
                    &mut u16_dst,
                    &mut f16_dst,
                    &mut bf16_dst,
                    &rgba16_src,
                    &mut rgba_dst,
                    &rgb16_block,
                    &mut rgb8_dst,
                );
                assert_eq!(u16_dst, ref_u16);
                assert_eq!(f16_dst, ref_f16);
                assert_eq!(bf16_dst, ref_bf16);
                assert_eq!(rgba_dst, ref_rgba);
                assert_eq!(rgb8_dst, ref_rgb8);
            });
        }

        #[test]
        fn unpremul_preserves_exact_alpha_bits_including_negative_zero_and_nan() {
            let _lock = archmage::testing::lock_token_testing();
            let custom_nan = f32::from_bits(0x7fc1_2345);
            let mut src = [0.0f32; 20];
            // Pixel 0: premultiplied 50% alpha
            src[0..4].copy_from_slice(&[0.25, 0.5, 0.125, 0.5]);
            // Pixel 1: -0.0 alpha (must preserve -0.0 bit pattern!)
            src[4..8].copy_from_slice(&[0.0, 0.0, 0.0, -0.0]);
            // Pixel 2..4: normal + tail pixel with custom NaN alpha payload
            src[8..12].copy_from_slice(&[0.2, 0.4, 0.6, 0.8]);
            src[12..16].copy_from_slice(&[1.0, 0.5, 0.25, 1.0]);
            src[16..20].copy_from_slice(&[0.1, 0.2, 0.3, custom_nan]);

            let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_| {
                let mut dst = [0.0f32; 20];
                unpremul_rgba_preserve_alpha(&src, &mut dst);
                assert_eq!(dst[3].to_bits(), 0.5f32.to_bits());
                assert_eq!(dst[7].to_bits(), (-0.0f32).to_bits());
                assert_eq!(dst[19].to_bits(), custom_nan.to_bits());
                assert!((dst[0] - 0.5).abs() <= 1e-5);
                assert!((dst[1] - 1.0).abs() <= 1e-5);
                assert_eq!(&dst[4..7], &[0.0, 0.0, 0.0]);
            });
        }
    }
}

pub mod unsigned_extrema_and_nmadd {
    //! Stage 2 unsigned 32-bit conversions, `RecipOptions` (`Nr1` / `Nr1Ceil`),
    //! `ExtremaOptions` (`*_with`), `trunc`, and `mul_neg_add_{fused,relaxed}`
    //! across `f32x8`, `f64x4`, and `magetypes::scalar`.

    use magetypes::numerics::{ExtremaOptions, IntConvert, RecipOptions};

    pub const U32_SAT: u32 = IntConvert::new().trunc().saturate_all().bits();
    pub const I32_RNE_SAT: u32 = IntConvert::new().nearest_even().saturate_all().bits();
    pub const EXTREMA_DROP_NAN: u32 = ExtremaOptions::new().drop_nan().bits();
    pub const EXTREMA_PROP_NAN: u32 = ExtremaOptions::new().propagate_nan().bits();
    pub const RECIP_NR1: u32 = RecipOptions::new().nr1().all_floats().bits();
    pub const RECIP_CEIL: u32 = RecipOptions::new().nr1_ceil().bits();

    #[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8, u32x8, f64x4))]
    fn evaluate_numerics_inner(
        token: Token,
        u32_in: &[u32; 8],
        f64_in: &[f64; 4],
    ) -> ([u32; 8], [i32; 8], [f32; 8], [f64; 4]) {
        // Unsigned u32 > i32::MAX widening & narrowing round-trip
        let uv = u32x8::load_t(token, u32_in);
        let fv = f32x8::from_u32_t(token, uv);
        let u_back = fv.to_u32_with::<U32_SAT>().to_array();
        let i_sat = fv.to_i32_with::<I32_RNE_SAT>().to_array();

        // Nr1Ceil upper-bound guarantee (`x * recip_ceil(x) >= 1.0`) + trunc + nmadd
        let pos = f32x8::from_array_t(token, [1.5, 3.0, 7.25, 19.0, 63.5, 127.0, 255.0, 1023.0]);
        let r_ceil = pos.recip_with::<RECIP_CEIL>();
        for (x, r) in pos.to_array().into_iter().zip(r_ceil.to_array()) {
            assert!(r >= 1.0 / x, "Nr1Ceil undershot 1/{x}: got {r}");
        }
        // Residual `1.0 - pos * r_ceil` via mul_neg_add_fused & mul_neg_add_relaxed + trunc
        let residual = pos
            .mul_neg_add_fused(r_ceil, f32x8::splat_t(token, 1.0))
            .mul_neg_add_relaxed(f32x8::splat_t(token, 2.0), pos.trunc());

        // Extrema *_with policies (`DropNan` vs `PropagateNan`) + reductions
        let nan_vec = f32x8::from_array_t(token, [f32::NAN, 2.0, -4.0, 8.0, 1.0, 3.0, 5.0, 7.0]);
        let rhs_vec = f32x8::splat_t(token, 4.0);
        assert_eq!(
            nan_vec.min_with::<EXTREMA_DROP_NAN>(rhs_vec).to_array()[0],
            4.0
        );
        assert!(nan_vec.min_with::<EXTREMA_PROP_NAN>(rhs_vec).to_array()[0].is_nan());
        assert_eq!(
            nan_vec
                .clamp_with::<EXTREMA_DROP_NAN>(f32x8::zero_t(token), rhs_vec)
                .reduce_max_with::<EXTREMA_DROP_NAN>(),
            4.0
        );
        assert_eq!(pos.reduce_min_with::<EXTREMA_DROP_NAN>(), 1.5);

        // f64 2-step NR recip_with & rsqrt_with + trunc + mul_neg_add_fused
        let dv = f64x4::load_t(token, f64_in);
        let dr = dv.recip_with::<RECIP_NR1>();
        let drs = dv.rsqrt_with::<RECIP_NR1>();
        let d_out = dv
            .mul_neg_add_fused(dr, f64x4::splat_t(token, 2.0))
            .mul_neg_add_relaxed(drs, dv.trunc())
            .to_array();

        (u_back, i_sat, residual.to_array(), d_out)
    }

    pub fn evaluate_numerics(
        u32_in: &[u32; 8],
        f64_in: &[f64; 4],
    ) -> ([u32; 8], [i32; 8], [f32; 8], [f64; 4]) {
        archmage::attuned!(
            evaluate_numerics_inner(u32_in, f64_in),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use archmage::ScalarToken;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
        use magetypes::numerics::HalfConvert;
        use magetypes::scalar;

        #[test]
        fn option_family_tags_reject_cross_family_bit_patterns() {
            assert!(IntConvert::from_bits(U32_SAT).is_some());
            assert!(HalfConvert::from_bits(U32_SAT).is_none());
            assert!(RecipOptions::from_bits(U32_SAT).is_none());
            assert!(ExtremaOptions::from_bits(U32_SAT).is_none());

            assert!(RecipOptions::from_bits(RECIP_CEIL).is_some());
            assert!(IntConvert::from_bits(RECIP_CEIL).is_none());
        }

        #[test]
        fn unsigned_recip_extrema_and_nmadd_validate_across_tiers() {
            let _lock = archmage::testing::lock_token_testing();
            // Powers/multiples of 256 up to 3_000_000_000 fit exactly in 24-bit f32 mantissa!
            let u32_in: [u32; 8] = [
                0,
                255,
                65_535,
                1_048_576,
                2_147_483_392,
                2_500_000_000 & !0xff,
                3_000_000_000 & !0xfff,
                4_000_000_000 & !0xfff,
            ];
            let f64_in: [f64; 4] = [2.25, 9.0, 16.5, 64.125];

            let (ref_u, ref_i, _, ref_d) =
                evaluate_numerics_inner_scalar_t(ScalarToken, &u32_in, &f64_in);
            for i in 0..8 {
                assert_eq!(ref_u[i], u32_in[i]);
                assert_eq!(
                    ref_u[i],
                    scalar::to_u32_with::<U32_SAT>(scalar::from_u32(u32_in[i]))
                );
            }

            let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_| {
                let (got_u, got_i, got_res, got_d) = evaluate_numerics(&u32_in, &f64_in);
                assert_eq!(got_u, ref_u);
                assert_eq!(got_i, ref_i);
                for (g, r) in got_d.into_iter().zip(ref_d) {
                    assert!((g - r).abs() <= 1e-12, "f64 mismatch: got {g}, ref {r}");
                }
                for r in got_res {
                    assert!(r.is_finite());
                }
            });

            // Scalar *_with tail parity checks
            assert_eq!(scalar::trunc_f32(-3.75), -3.0);
            assert_eq!(scalar::trunc_f64(-3.75), -3.0);
            assert_eq!(scalar::mul_neg_add_fused_f32(2.0, 3.0, 10.0), 4.0);
            assert_eq!(
                scalar::mul_neg_add_relaxed_f64(ScalarToken, 2.0, 3.0, 10.0),
                4.0
            );
            assert!(scalar::recip_f32_with::<RECIP_CEIL>(3.0) >= 1.0f32 / 3.0);
            assert!((scalar::rsqrt_f64_with::<RECIP_NR1>(4.0) - 0.5).abs() <= 1e-15);
            assert_eq!(scalar::min_f32_with::<EXTREMA_DROP_NAN>(f32::NAN, 2.5), 2.5);
            assert_eq!(
                scalar::clamp_f32_with::<EXTREMA_DROP_NAN>(f32::NAN, 1.0, 5.0),
                1.0
            );
        }
    }
}

/// Compile-fail proofs that `pub(crate) trait F32x*Internal` and `F*ScalarInternal`
/// seal low-level repr/plumbing hooks from external crates on both concrete tokens
/// and generic backend bounds (`E0624` / `E0599`, never merely `#[doc(hidden)]`).
///
/// Generic `F32x4Backend` callers cannot reach internal repr methods:
/// ```compile_fail,E0624
/// use magetypes::simd::backends::F32x4Backend;
/// fn try_leak<T: F32x4Backend>(t: T) {
///     let _ = t.from_u16_lanes([0; 4]);
/// }
/// ```
///
/// Concrete token callers cannot reach internal repr methods either:
/// ```compile_fail
/// use archmage::ScalarToken;
/// use magetypes::simd::backends::F32x4Backend;
/// let _ = ScalarToken.from_u16_lanes([0; 4]);
/// ```
///
/// Generic `F32ScalarBackend` callers cannot reach internal scalar dispatch methods:
/// ```compile_fail,E0624
/// use magetypes::scalar::F32ScalarBackend;
/// fn try_leak_scalar<T: F32ScalarBackend>(t: T) -> f32 {
///     t.min_fast_f32(1.0, 2.0)
/// }
/// ```
///
/// Concrete token callers cannot reach internal scalar dispatch methods either:
/// ```compile_fail
/// use archmage::ScalarToken;
/// use magetypes::scalar::F32ScalarBackend;
/// let _ = ScalarToken.min_fast_f32(1.0, 2.0);
/// ```
pub mod sealed_backend_privacy {}

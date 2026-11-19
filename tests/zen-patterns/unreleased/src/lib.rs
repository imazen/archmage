//! Integration patterns that depend on features introduced in 0.9.32
//! (`magemath` with caller-proof SIMD tails, `#[attune(..., suppress_const_test)]`,
//! `f32` `min_fast`/`max_fast`/`to_i32_fast`, and 3-channel RGB block helpers).
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

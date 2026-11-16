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
    //! Vector chunks and `*_with_token` scalar tails sharing one `const` profile
    //! and the caller's token inside an `#[attune(_*_t, ...)]` family.

    use magemath::cbrtf::{
        CbrtfError, CbrtfExecution, CbrtfOptions, try_cbrt_f32x8_with, try_cbrt_with_token,
    };
    use magemath::expf::{
        ExpfError, ExpfExecution, ExpfOptions, try_exp2_f32x8_with, try_exp2_with_token,
    };
    use magemath::logf::{
        LogfError, LogfExecution, LogfOptions, try_log2_f32x8_with, try_log2_with_token,
    };
    use magemath::powf::{
        PowfError, PowfExecution, PowfOptions, try_powf_f32x8_with, try_powf_with_token,
    };

    pub const POWF_FLAGS: u32 = PowfOptions::PORTABLE_GAMMA_MIDP
        .with_execution(PowfExecution::Caller)
        .bits();
    pub const EXPF_FLAGS: u32 = ExpfOptions::PORTABLE_MIDP
        .with_execution(ExpfExecution::Caller)
        .bits();
    pub const LOGF_FLAGS: u32 = LogfOptions::PORTABLE_MIDP
        .with_execution(LogfExecution::Caller)
        .bits();
    pub const CBRTF_FLAGS: u32 = CbrtfOptions::PORTABLE_MIDP
        .with_execution(CbrtfExecution::Caller)
        .bits();

    #[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
    fn powf_slice_inner(
        token: Token,
        src: &[f32],
        dst: &mut [f32],
        exp: f32,
    ) -> Result<(), PowfError> {
        let (sc, st) = f32x8::partition_slice_t(token, src);
        let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
        for (i, o) in sc.iter().zip(dc.iter_mut()) {
            let v = try_powf_f32x8_with::<POWF_FLAGS, _>(token, f32x8::load_t(token, i), exp)?;
            v.store(o);
        }
        for (&x, o) in st.iter().zip(dt.iter_mut()) {
            *o = try_powf_with_token::<POWF_FLAGS, _>(token, x, exp)?;
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
        let (sc, st) = f32x8::partition_slice_t(token, src);
        let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
        for (i, o) in sc.iter().zip(dc.iter_mut()) {
            let v = try_exp2_f32x8_with::<EXPF_FLAGS, _>(token, f32x8::load_t(token, i))?;
            v.store(o);
        }
        for (&x, o) in st.iter().zip(dt.iter_mut()) {
            *o = try_exp2_with_token::<EXPF_FLAGS, _>(token, x)?;
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
        let (sc, st) = f32x8::partition_slice_t(token, src);
        let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
        for (i, o) in sc.iter().zip(dc.iter_mut()) {
            let v = try_log2_f32x8_with::<LOGF_FLAGS, _>(token, f32x8::load_t(token, i))?;
            v.store(o);
        }
        for (&x, o) in st.iter().zip(dt.iter_mut()) {
            *o = try_log2_with_token::<LOGF_FLAGS, _>(token, x)?;
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
        let (sc, st) = f32x8::partition_slice_t(token, src);
        let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
        for (i, o) in sc.iter().zip(dc.iter_mut()) {
            let v = try_cbrt_f32x8_with::<CBRTF_FLAGS, _>(token, f32x8::load_t(token, i))?;
            v.store(o);
        }
        for (&x, o) in st.iter().zip(dt.iter_mut()) {
            *o = try_cbrt_with_token::<CBRTF_FLAGS, _>(token, x)?;
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
            cbrt_bad[2] = 1e-42; // subnormal rejected by Bounded
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
    //! `load_8_rgb_u8_t`, `interleave_3ch` / `deinterleave_3ch`, `max_fast` /
    //! `min_fast`, `to_i32_fast`, and `store_8_rgb_u8` inside `#[archmage::attune]`.

    use magetypes::simd::backends::{F32x8Backend, F32x8Convert};
    use magetypes::simd::generic::f32x8;

    #[inline(always)]
    pub fn process_rgb8_block<T: F32x8Backend + F32x8Convert>(
        token: T,
        rgb: &[u8; 24],
        scale: f32,
        bias: f32,
    ) -> ([u8; 24], [i32; 8]) {
        let (r, g, b) = f32x8::load_8_rgb_u8_t(token, rgb);
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
        (f32x8::store_8_rgb_u8(r, g, b), quant_r)
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
            let r = (s_px[0] as f32 * scale + bias).clamp(0.0, 255.0);
            let g = (s_px[1] as f32 * scale + bias).clamp(0.0, 255.0);
            let b = (s_px[2] as f32 * scale + bias).clamp(0.0, 255.0);
            *q_slot = r as i32;
            d_px[0] = r.round_ties_even() as u8;
            d_px[1] = g.round_ties_even() as u8;
            d_px[2] = b.round_ties_even() as u8;
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
    }
}

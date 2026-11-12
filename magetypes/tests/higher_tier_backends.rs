//! Integration tests for higher-tier `magetypes` SIMD backends (#155, #161):
//! - x86_64: `X64V3CryptoToken`, `X64V3GfniCryptoToken`, `X64V3VnniToken`
//! - aarch64: `Arm64V2Token`, `Arm64V3Token`
//! - wasm32: `Wasm128RelaxedToken` (including per-function `relaxed-simd` FMA overrides)
//!
//! Verifies:
//! 1. Generic kernels over `F32x8Convert + F32x8FromHalves + F16Convert` (and `w512` `F32x16FromHalves`)
//!    compile and execute identically across `#[magetypes(...)]` + `incant!` for all sub-tiers.
//! 2. `implementation_name()` reports distinct per-tier backend identities across W128/W256/W512.
//! 3. Zero-`unsafe` `.raw()`, turbofish-free `Type::from_raw_t(token, raw)`, and `Type::<Token>::from_raw(raw)`
//!    interop lets higher-tier `#[arcane]` functions mix `magetypes` vectors directly with
//!    tier-specific intrinsics (`vpdpbusd`, `vaes`, `gf2p8mul`, `vqrdmlah` / `vdot`, `vbcax` / `veor3`,
//!    and `relaxed-simd` FMA) without token downcasting.

#![forbid(unsafe_code)]
#![allow(unexpected_cfgs)]

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use archmage::SimdToken;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use archmage::arcane;
use archmage::{ScalarToken, incant, magetypes};
use magetypes::simd::backends::{F32x8Convert, I8x16Backend, I32x8Backend, U8x32Backend};
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use magetypes::simd::generic::i32x4;
use magetypes::simd::generic::{F16Convert, F32x8FromHalves, f32x4, f32x8, i32x8};

#[cfg(feature = "w512")]
use magetypes::simd::backends::F32x16Convert;
#[cfg(feature = "w512")]
use magetypes::simd::generic::{F32x16FromHalves, f32x16};

fn run_generic_pipeline<T>(token: T) -> ([f32; 8], [i32; 8], [f32; 8])
where
    T: F32x8Convert + F32x8FromHalves + F16Convert + I32x8Backend + U8x32Backend + I8x16Backend,
{
    let lo = f32x4::<T>::from_array_t(token, [1.5, -2.25, 3.75, -4.5]);
    let hi = f32x4::<T>::from_array_t(token, [0.5, 6.25, -7.0, 8.125]);
    let combined = f32x8::<T>::from_halves_t(token, lo, hi);
    assert_eq!(combined.low().to_array(), lo.to_array());
    assert_eq!(combined.high().to_array(), hi.to_array());

    let scale = f32x8::<T>::splat_t(token, 2.0);
    let bias = f32x8::<T>::splat_t(token, 0.5);
    let affine = combined.mul_add_relaxed(scale, bias);
    let rounded = affine.to_i32_round();
    let shifted = (rounded + i32x8::<T>::splat_t(token, 3)).shl_const::<1>();

    // Exercise F16Convert slice encode + decode directly on the higher-tier token.
    let affine_arr = affine.to_array();
    let mut f16_bits = [0u16; 8];
    let mut decoded = [0.0f32; 8];
    token.f32_to_f16_slice(&affine_arr, &mut f16_bits);
    token.f16_to_f32_slice(&f16_bits, &mut decoded);

    (affine_arr, shifted.to_array(), decoded)
}

#[cfg(feature = "w512")]
fn run_generic_w512<T>(token: T) -> [f32; 16]
where
    T: F32x16Convert + F32x16FromHalves,
{
    let lo = f32x8::<T>::from_array_t(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let hi = f32x8::<T>::from_array_t(token, [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
    let wide = f32x16::<T>::from_halves_t(token, lo, hi);
    assert_eq!(wide.low().to_array(), lo.to_array());
    assert_eq!(wide.high().to_array(), hi.to_array());
    let two = f32x16::<T>::splat_t(token, 2.0);
    let one = f32x16::<T>::splat_t(token, 1.0);
    wide.mul_add_relaxed(two, one).to_array()
}

#[magetypes(
    v3_vnni,
    v3_gfni_crypto,
    v3_crypto,
    v3,
    arm_v3,
    arm_v2,
    neon,
    wasm128_relaxed,
    wasm128,
    scalar
)]
fn tiered_pipeline(token: Token) -> ([f32; 8], [i32; 8], [f32; 8]) {
    run_generic_pipeline(token)
}

fn dispatch_tiered_pipeline() -> ([f32; 8], [i32; 8], [f32; 8]) {
    incant!(
        tiered_pipeline(),
        [
            v3_vnni,
            v3_gfni_crypto,
            v3_crypto,
            v3,
            arm_v3,
            arm_v2,
            neon,
            wasm128_relaxed,
            wasm128,
            scalar
        ]
    )
}

#[test]
fn generic_pipeline_matches_scalar_across_all_active_tiers() {
    let expected = tiered_pipeline_scalar(ScalarToken);
    let dispatched = dispatch_tiered_pipeline();
    assert_eq!(dispatched.1, expected.1);
    for i in 0..8 {
        assert!(
            (dispatched.0[i] - expected.0[i]).abs() <= 1e-6,
            "lane {i}: got {}, expected {}",
            dispatched.0[i],
            expected.0[i]
        );
        // Exact dyadic values representable in binary16: roundtrip must be exact!
        assert_eq!(dispatched.2[i], expected.0[i], "f16 roundtrip lane {i}");
    }

    #[cfg(feature = "w512")]
    {
        let expected_w512 = run_generic_w512(ScalarToken);
        for (i, &val) in expected_w512.iter().enumerate() {
            assert_eq!(val, (i as f32 + 1.0) * 2.0 + 1.0);
        }
    }
}

// ============================================================================
// x86_64 higher-tier backends: X64V3CryptoToken, X64V3GfniCryptoToken, X64V3VnniToken
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_higher_tiers {
    use super::*;
    use archmage::{X64V3CryptoToken, X64V3GfniCryptoToken, X64V3VnniToken};
    use core::arch::x86_64::*;
    use magetypes::simd::generic::{i8x32, i64x4, u8x32};

    #[test]
    fn implementation_names_x86_higher_tiers() {
        assert_eq!(
            f32x8::<X64V3CryptoToken>::implementation_name(),
            "x86::v3_crypto::f32x8"
        );
        assert_eq!(
            f32x8::<X64V3GfniCryptoToken>::implementation_name(),
            "x86::v3_gfni_crypto::f32x8"
        );
        assert_eq!(
            f32x8::<X64V3VnniToken>::implementation_name(),
            "x86::v3_vnni::f32x8"
        );
        assert_eq!(
            i32x4::<X64V3VnniToken>::implementation_name(),
            "x86::v3_vnni::i32x4"
        );
        #[cfg(feature = "w512")]
        {
            assert_eq!(
                f32x16::<X64V3CryptoToken>::implementation_name(),
                "polyfill::v3_crypto_512::f32x16"
            );
            assert_eq!(
                f32x16::<X64V3GfniCryptoToken>::implementation_name(),
                "polyfill::v3_gfni_crypto_512::f32x16"
            );
            assert_eq!(
                f32x16::<X64V3VnniToken>::implementation_name(),
                "polyfill::v3_vnni_512::f32x16"
            );
        }
    }

    #[arcane]
    fn v3_crypto_vaes_with_magetypes(token: X64V3CryptoToken) -> [i64; 4] {
        let state = i64x4::from_array_t(token, [1, 2, 3, 4]);
        let key = i64x4::from_array_t(token, [5, 6, 7, 8]);
        // Turbofish-free `from_raw_t(token, ...)` + explicit `from_raw(...)`
        let enc = i64x4::from_raw_t(token, _mm256_aesenc_epi128(state.raw(), key.raw()));
        let dec = i64x4::<X64V3CryptoToken>::from_raw(enc.raw());
        assert_eq!(enc.to_array(), dec.to_array());
        enc.to_array()
    }

    #[arcane]
    fn v3_gfni_with_magetypes(token: X64V3GfniCryptoToken) -> [u8; 32] {
        let a = u8x32::splat_t(token, 0x57);
        let b = u8x32::splat_t(token, 0x13);
        // 0x57 * 0x13 in GF(2^8) with AES polynomial 0x11b = 0xfe
        u8x32::from_raw_t(token, _mm256_gf2p8mul_epi8(a.raw(), b.raw())).to_array()
    }

    #[arcane]
    fn v3_vnni_quantized_layer(token: X64V3VnniToken) -> [f32; 8] {
        // Mix magetypes vectors + raw AVX-VNNI dot product + magetypes FMA post-scale
        // with zero `.v3()` token extraction, zero turbofish, and zero `unsafe`!
        let activations = u8x32::from_array_t(
            token,
            [
                1, 2, 3, 4, 2, 2, 2, 2, 3, 0, 3, 0, 4, 4, 4, 4, 5, 1, 1, 1, 0, 10, 0, 10, 2, 4, 6,
                8, 1, 1, 1, 1,
            ],
        );
        let weights = i8x32::from_array_t(
            token,
            [
                1, 1, 1, 1, -1, 2, -1, 2, 4, 5, -2, 7, 1, -1, 1, -1, 2, 2, 2, 2, 3, -3, 3, 5, 1, 0,
                -1, 2, 10, 20, 30, -40,
            ],
        );
        let acc = i32x8::from_array_t(token, [0, 10, 20, 30, 40, 50, 60, 70]);
        let dots = i32x8::from_raw_t(
            token,
            _mm256_dpbusd_avx_epi32(acc.raw(), activations.raw(), weights.raw()),
        );
        let scale = f32x8::splat_t(token, 0.25);
        let bias = f32x8::splat_t(token, 1.0);
        f32x8::from_i32_t(token, dots)
            .mul_add_relaxed(scale, bias)
            .to_array()
    }

    #[test]
    fn x86_higher_tier_raw_and_intrinsic_interop() {
        let scalar_ref = tiered_pipeline_scalar(ScalarToken);

        if let Some(t) = X64V3CryptoToken::summon() {
            assert_eq!(tiered_pipeline_v3_crypto(t).1, scalar_ref.1);
            let out = v3_crypto_vaes_with_magetypes(t);
            assert_ne!(out, [1, 2, 3, 4]);
            #[cfg(feature = "w512")]
            assert_eq!(run_generic_w512(t), run_generic_w512(ScalarToken));
        }

        if let Some(t) = X64V3GfniCryptoToken::summon() {
            assert_eq!(tiered_pipeline_v3_gfni_crypto(t).1, scalar_ref.1);
            assert_eq!(v3_gfni_with_magetypes(t), [0xfe; 32]);
            #[cfg(feature = "w512")]
            assert_eq!(run_generic_w512(t), run_generic_w512(ScalarToken));
        }

        if let Some(t) = X64V3VnniToken::summon() {
            assert_eq!(tiered_pipeline_v3_vnni(t).1, scalar_ref.1);
            let out = v3_vnni_quantized_layer(t);
            assert_eq!(out, [3.5, 4.5, 7.5, 8.5, 15.0, 18.5, 19.0, 23.5]);
            #[cfg(feature = "w512")]
            assert_eq!(run_generic_w512(t), run_generic_w512(ScalarToken));
        }
    }
}

// ============================================================================
// aarch64 higher-tier backends: Arm64V2Token, Arm64V3Token
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod arm_higher_tiers {
    use super::*;
    use archmage::{Arm64V2Token, Arm64V3Token};
    use core::arch::aarch64::*;
    use magetypes::simd::generic::i8x16;

    #[test]
    fn implementation_names_arm_higher_tiers() {
        assert_eq!(
            f32x4::<Arm64V2Token>::implementation_name(),
            "arm::arm_v2::f32x4"
        );
        assert_eq!(
            f32x8::<Arm64V2Token>::implementation_name(),
            "polyfill::arm_v2::f32x8"
        );
        assert_eq!(
            f32x4::<Arm64V3Token>::implementation_name(),
            "arm::arm_v3::f32x4"
        );
        assert_eq!(
            f32x8::<Arm64V3Token>::implementation_name(),
            "polyfill::arm_v3::f32x8"
        );
        #[cfg(feature = "w512")]
        {
            assert_eq!(
                f32x16::<Arm64V2Token>::implementation_name(),
                "polyfill::arm_v2_512::f32x16"
            );
            assert_eq!(
                f32x16::<Arm64V3Token>::implementation_name(),
                "polyfill::arm_v3_512::f32x16"
            );
        }
    }

    #[arcane]
    fn arm_v2_rdm_with_magetypes(token: Arm64V2Token) -> [f32; 4] {
        // RDM `vqrdmlahq_s32` (ARMv8.1-A, guaranteed by Arm64V2Token) + magetypes FMA
        // Exercises turbofish-free `i32x4::from_raw_t(token, ...)`.
        let acc = i32x4::splat_t(token, 100);
        let a = i32x4::splat_t(token, 1 << 29);
        let b = i32x4::splat_t(token, 4);
        // 2 * (2^29 * 4) >> 32 = 1, so acc + 1 = 101
        let rdm = i32x4::from_raw_t(token, vqrdmlahq_s32(acc.raw(), a.raw(), b.raw()));
        let scale = f32x4::splat_t(token, 2.0);
        let bias = f32x4::splat_t(token, 0.5);
        f32x4::from_i32_t(token, rdm)
            .mul_add_relaxed(scale, bias)
            .to_array()
    }

    #[rustversion::since(1.98)]
    #[arcane]
    fn arm_v2_dotprod_with_magetypes(token: Arm64V2Token) -> [i32; 4] {
        let acc = i32x4::from_array_t(token, [10, 20, 30, 40]);
        let a = i8x16::from_array_t(token, [1, 2, 3, 4, 2, 2, 2, 2, -1, -2, -3, -4, 5, 0, -5, 2]);
        let b = i8x16::from_array_t(token, [1, 1, 1, 1, 3, -1, 3, -1, 2, 2, 2, 2, 4, 9, 4, 3]);
        i32x4::from_raw_t(token, vdotq_s32(acc.raw(), a.raw(), b.raw())).to_array()
    }

    #[rustversion::since(1.98)]
    fn assert_arm_v2_dotprod(token: Arm64V2Token) {
        assert_eq!(arm_v2_dotprod_with_magetypes(token), [20, 28, 10, 46]);
    }

    #[rustversion::before(1.98)]
    fn assert_arm_v2_dotprod(_token: Arm64V2Token) {
        let _ = i8x16::<Arm64V2Token>::implementation_name();
    }

    #[arcane]
    fn arm_v3_sha3_with_magetypes(token: Arm64V3Token) -> [i32; 4] {
        // SHA3 `veor3q_s32` & `vbcaxq_s32` (ARMv8.2+SHA3, guaranteed by Arm64V3Token)
        let a = i32x4::from_array_t(token, [0x0f, 0xf0, 0xaa, 0x55]);
        let b = i32x4::from_array_t(token, [0x33, 0x33, 0x0f, 0xf0]);
        let c = i32x4::from_array_t(token, [0x55, 0xaa, 0xe1, 0x1e]);
        let xor3 = i32x4::from_raw_t(token, veor3q_s32(a.raw(), b.raw(), c.raw()));
        let bcax = i32x4::<Arm64V3Token>::from_raw(vbcaxq_s32(a.raw(), b.raw(), c.raw()));
        (xor3 + bcax).to_array()
    }

    #[test]
    fn arm_higher_tier_raw_and_intrinsic_interop() {
        let scalar_ref = tiered_pipeline_scalar(ScalarToken);
        println!(
            "aarch64 tiers: arm_v2={}, arm_v3={}",
            Arm64V2Token::summon().is_some(),
            Arm64V3Token::summon().is_some()
        );

        if let Some(v2) = Arm64V2Token::summon() {
            assert_eq!(tiered_pipeline_arm_v2(v2).1, scalar_ref.1);
            assert_eq!(arm_v2_rdm_with_magetypes(v2), [202.5; 4]);
            assert_arm_v2_dotprod(v2);
            #[cfg(feature = "w512")]
            assert_eq!(run_generic_w512(v2), run_generic_w512(ScalarToken));
        }

        if let Some(v3) = Arm64V3Token::summon() {
            assert_eq!(tiered_pipeline_arm_v3(v3).1, scalar_ref.1);
            let expected: [i32; 4] = [
                (0x0f ^ 0x33 ^ 0x55) + (0x0f ^ (0x33 & !0x55)),
                (0xf0 ^ 0x33 ^ 0xaa) + (0xf0 ^ (0x33 & !0xaa)),
                (0xaa ^ 0x0f ^ 0xe1) + (0xaa ^ (0x0f & !0xe1)),
                (0x55 ^ 0xf0 ^ 0x1e) + (0x55 ^ (0xf0 & !0x1e)),
            ];
            assert_eq!(arm_v3_sha3_with_magetypes(v3), expected);
            #[cfg(feature = "w512")]
            assert_eq!(run_generic_w512(v3), run_generic_w512(ScalarToken));
        }
    }
}

// ============================================================================
// wasm32 higher-tier backend: Wasm128RelaxedToken + per-function relaxed FMA
// ============================================================================

#[cfg(target_arch = "wasm32")]
mod wasm_higher_tiers {
    use super::*;
    use archmage::Wasm128RelaxedToken;
    #[cfg(any(target_feature = "relaxed-simd", test_per_fn_wasm_relaxed))]
    use magetypes::simd::generic::{f64x2, f64x4, u8x16};

    #[test]
    fn implementation_names_wasm_relaxed() {
        assert_eq!(
            f32x4::<Wasm128RelaxedToken>::implementation_name(),
            "wasm::wasm128_relaxed::f32x4"
        );
        assert_eq!(
            f32x8::<Wasm128RelaxedToken>::implementation_name(),
            "polyfill::wasm128_relaxed::f32x8"
        );
        #[cfg(feature = "w512")]
        assert_eq!(
            f32x16::<Wasm128RelaxedToken>::implementation_name(),
            "polyfill::wasm128_relaxed_512::f32x16"
        );
    }

    // Exported / per-function `#[archmage::rite(wasm128_relaxed)]` kernel:
    // compiles and emits `f32x4.relaxed_madd` / `f64x2.relaxed_madd` EVEN when
    // the crate is built with ONLY `-C target-feature=+simd128` (no crate-global
    // `+relaxed-simd`), while staying gated off in strict-only WASM test runs so
    // `WebAssembly.validate` succeeds on engines with `relaxed-simd` disabled.
    #[allow(deprecated)]
    #[cfg(any(target_feature = "relaxed-simd", test_per_fn_wasm_relaxed))]
    #[archmage::rite(wasm128_relaxed)]
    fn per_function_wasm_relaxed_fma() -> ([f32; 4], [f32; 8], [f64; 2], [f64; 4]) {
        let token = Wasm128RelaxedToken::from_context();

        // Turbofish-free `from_raw_t` roundtrip
        let raw_in = u8x16::splat_t(token, 42);
        let raw_rt = u8x16::from_raw_t(token, raw_in.raw());
        assert_eq!(raw_rt.to_array(), [42u8; 16]);

        let a4 = f32x4::from_array_t(token, [1.5, -2.0, 3.25, 4.0]);
        let b4 = f32x4::from_array_t(token, [2.0, 3.0, -4.0, 0.5]);
        let c4 = f32x4::from_array_t(token, [0.5, 1.0, 2.0, -1.0]);
        let fma4 = a4.mul_add_relaxed(b4, c4).mul_sub_relaxed(b4, c4);
        let fma4_p = a4.mul_add_portable(b4, c4).mul_sub_portable(b4, c4);
        assert_eq!(fma4.to_array(), fma4_p.to_array());

        let a8 = f32x8::from_halves_t(token, a4, b4);
        let b8 = f32x8::splat_t(token, 2.0);
        let c8 = f32x8::splat_t(token, 1.0);
        let fma8 = a8.mul_add_relaxed(b8, c8);

        let d2 = f64x2::from_array_t(token, [2.5, -3.5]);
        let e2 = f64x2::from_array_t(token, [4.0, 2.0]);
        let g2 = f64x2::from_array_t(token, [1.0, -0.5]);
        let fma_d2 = d2.mul_add_relaxed(e2, g2);

        let d4 = f64x4::from_array_t(token, [1.0, 2.0, 3.0, 4.0]);
        let fma_d4 = d4.mul_sub_relaxed(f64x4::splat_t(token, 3.0), f64x4::splat_t(token, 0.5));

        #[cfg(feature = "w512")]
        {
            assert_eq!(run_generic_w512(token), run_generic_w512(ScalarToken));
        }

        (
            fma4.to_array(),
            fma8.to_array(),
            fma_d2.to_array(),
            fma_d4.to_array(),
        )
    }

    #[test]
    fn wasm_relaxed_dispatch_and_per_fn_fma() {
        let scalar_ref = tiered_pipeline_scalar(ScalarToken);
        if let Some(t) = Wasm128RelaxedToken::summon() {
            assert_eq!(tiered_pipeline_wasm128_relaxed(t).1, scalar_ref.1);
        }

        #[cfg(any(target_feature = "relaxed-simd", test_per_fn_wasm_relaxed))]
        {
            let (fma4, fma8, fma_d2, fma_d4) = per_function_wasm_relaxed_fma();
            // a4.mul_add(b4, c4) = [3.5, -5.0, -11.0, 1.0]
            // .mul_sub(b4, c4)   = [6.5, -16.0, 42.0, 1.5]
            assert_eq!(fma4, [6.5, -16.0, 42.0, 1.5]);
            assert_eq!(fma8, [4.0, -3.0, 7.5, 9.0, 5.0, 7.0, -7.0, 2.0]);
            assert_eq!(fma_d2, [11.0, -7.5]);
            assert_eq!(fma_d4, [2.5, 5.5, 8.5, 11.5]);
        }
    }
}

//! x86 token stubs for non-x86 architectures.
//!
//! These types exist so cross-platform code can reference them without cfg guards.
//! `summon()` always returns `None` on non-x86.

use super::{CompositeToken, SimdToken};
use super::{Has128BitSimd, Has256BitSimd, HasFma};
#[cfg(feature = "avx512")]
use super::{Has512BitSimd, HasAvx512bw, HasAvx512dq, HasAvx512f, HasAvx512vbmi2, HasAvx512vl};
use super::{HasAvx, HasAvx2, HasSse, HasSse2, HasSse41, HasSse42};

macro_rules! define_x86_stub {
    ($name:ident, $display:literal) => {
        #[doc = concat!("Stub for ", $display, " token (not available on this architecture).")]
        #[derive(Clone, Copy, Debug)]
        pub struct $name {
            _private: (),
        }

        impl SimdToken for $name {
            const NAME: &'static str = $display;

            #[inline]
            fn try_new() -> Option<Self> {
                None // Not available on this architecture
            }

            #[inline(always)]
            unsafe fn forge_token_dangerously() -> Self {
                Self { _private: () }
            }
        }
    };
}

// Define non-AVX-512 x86 token stubs
define_x86_stub!(SseToken, "SSE");
define_x86_stub!(Sse2Token, "SSE2");
define_x86_stub!(Sse41Token, "SSE4.1");
define_x86_stub!(Sse42Token, "SSE4.2");
define_x86_stub!(AvxToken, "AVX");
define_x86_stub!(Avx2Token, "AVX2");
define_x86_stub!(FmaToken, "FMA");
define_x86_stub!(X64V2Token, "x86-64-v2");
define_x86_stub!(X64V3Token, "x86-64-v3");

// AVX-512 token stubs (requires "avx512" feature)
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512fToken, "AVX-512F");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512bwToken, "AVX-512BW");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512fVlToken, "AVX-512F+VL");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512bwVlToken, "AVX-512BW+VL");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512Vbmi2Token, "AVX-512VBMI2");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512Vbmi2VlToken, "AVX-512VBMI2+VL");
#[cfg(feature = "avx512")]
define_x86_stub!(X64V4Token, "AVX-512");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512Token, "AVX-512");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512ModernToken, "AVX-512Modern");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512Fp16Token, "AVX-512FP16");

/// Stub for AVX2+FMA combined token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct Avx2FmaToken {
    _private: (),
}

impl SimdToken for Avx2FmaToken {
    const NAME: &'static str = "AVX2+FMA";

    #[inline]
    fn try_new() -> Option<Self> {
        None
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl CompositeToken for Avx2FmaToken {
    type Components = (Avx2Token, FmaToken);

    fn components(&self) -> Self::Components {
        (unsafe { Avx2Token::forge_token_dangerously() }, unsafe {
            FmaToken::forge_token_dangerously()
        })
    }
}

/// Alias for x86-64-v3 (AVX2 + FMA) - stub on non-x86 architectures.
pub type Desktop64 = X64V3Token;

// ============================================================================
// Marker trait implementations for non-AVX-512 tokens
// ============================================================================

impl Has128BitSimd for SseToken {}
impl Has128BitSimd for Sse2Token {}
impl Has128BitSimd for Sse41Token {}
impl Has128BitSimd for Sse42Token {}
impl Has128BitSimd for X64V2Token {}
impl Has128BitSimd for AvxToken {}
impl Has256BitSimd for AvxToken {}
impl Has128BitSimd for Avx2Token {}
impl Has256BitSimd for Avx2Token {}
impl Has128BitSimd for Avx2FmaToken {}
impl Has256BitSimd for Avx2FmaToken {}
impl Has128BitSimd for X64V3Token {}
impl Has256BitSimd for X64V3Token {}

impl HasFma for FmaToken {}
impl HasFma for Avx2FmaToken {}
impl HasFma for X64V3Token {}

impl HasSse for SseToken {}
impl HasSse for Sse2Token {}
impl HasSse for Sse41Token {}
impl HasSse for Sse42Token {}
impl HasSse for AvxToken {}
impl HasSse for Avx2Token {}
impl HasSse for Avx2FmaToken {}
impl HasSse for FmaToken {}
impl HasSse for X64V2Token {}
impl HasSse for X64V3Token {}

impl HasSse2 for Sse2Token {}
impl HasSse2 for Sse41Token {}
impl HasSse2 for Sse42Token {}
impl HasSse2 for AvxToken {}
impl HasSse2 for Avx2Token {}
impl HasSse2 for Avx2FmaToken {}
impl HasSse2 for FmaToken {}
impl HasSse2 for X64V2Token {}
impl HasSse2 for X64V3Token {}

impl HasSse41 for Sse41Token {}
impl HasSse41 for Sse42Token {}
impl HasSse41 for AvxToken {}
impl HasSse41 for Avx2Token {}
impl HasSse41 for Avx2FmaToken {}
impl HasSse41 for X64V2Token {}
impl HasSse41 for X64V3Token {}

impl HasSse42 for Sse42Token {}
impl HasSse42 for AvxToken {}
impl HasSse42 for Avx2Token {}
impl HasSse42 for Avx2FmaToken {}
impl HasSse42 for X64V2Token {}
impl HasSse42 for X64V3Token {}

impl HasAvx for AvxToken {}
impl HasAvx for Avx2Token {}
impl HasAvx for Avx2FmaToken {}
impl HasAvx for X64V3Token {}

impl HasAvx2 for Avx2Token {}
impl HasAvx2 for Avx2FmaToken {}
impl HasAvx2 for X64V3Token {}

// ============================================================================
// AVX-512 marker trait implementations (requires "avx512" feature)
// ============================================================================

#[cfg(feature = "avx512")]
mod avx512_impls {
    use super::*;

    impl Has128BitSimd for Avx512fToken {}
    impl Has256BitSimd for Avx512fToken {}
    impl Has512BitSimd for Avx512fToken {}
    impl Has128BitSimd for Avx512bwToken {}
    impl Has256BitSimd for Avx512bwToken {}
    impl Has512BitSimd for Avx512bwToken {}
    impl Has128BitSimd for X64V4Token {}
    impl Has256BitSimd for X64V4Token {}
    impl Has512BitSimd for X64V4Token {}
    impl Has128BitSimd for Avx512fVlToken {}
    impl Has256BitSimd for Avx512fVlToken {}
    impl Has512BitSimd for Avx512fVlToken {}
    impl Has128BitSimd for Avx512bwVlToken {}
    impl Has256BitSimd for Avx512bwVlToken {}
    impl Has512BitSimd for Avx512bwVlToken {}
    impl Has128BitSimd for Avx512Vbmi2Token {}
    impl Has256BitSimd for Avx512Vbmi2Token {}
    impl Has512BitSimd for Avx512Vbmi2Token {}
    impl Has128BitSimd for Avx512Vbmi2VlToken {}
    impl Has256BitSimd for Avx512Vbmi2VlToken {}
    impl Has512BitSimd for Avx512Vbmi2VlToken {}

    impl HasFma for X64V4Token {}
    impl HasFma for Avx512fToken {}
    impl HasFma for Avx512fVlToken {}
    impl HasFma for Avx512bwToken {}
    impl HasFma for Avx512bwVlToken {}
    impl HasFma for Avx512Vbmi2Token {}
    impl HasFma for Avx512Vbmi2VlToken {}

    impl HasSse for Avx512fToken {}
    impl HasSse for Avx512fVlToken {}
    impl HasSse for Avx512bwToken {}
    impl HasSse for Avx512bwVlToken {}
    impl HasSse for Avx512Vbmi2Token {}
    impl HasSse for Avx512Vbmi2VlToken {}
    impl HasSse for X64V4Token {}

    impl HasSse2 for Avx512fToken {}
    impl HasSse2 for Avx512fVlToken {}
    impl HasSse2 for Avx512bwToken {}
    impl HasSse2 for Avx512bwVlToken {}
    impl HasSse2 for Avx512Vbmi2Token {}
    impl HasSse2 for Avx512Vbmi2VlToken {}
    impl HasSse2 for X64V4Token {}

    impl HasSse41 for Avx512fToken {}
    impl HasSse41 for Avx512fVlToken {}
    impl HasSse41 for Avx512bwToken {}
    impl HasSse41 for Avx512bwVlToken {}
    impl HasSse41 for Avx512Vbmi2Token {}
    impl HasSse41 for Avx512Vbmi2VlToken {}
    impl HasSse41 for X64V4Token {}

    impl HasSse42 for Avx512fToken {}
    impl HasSse42 for Avx512fVlToken {}
    impl HasSse42 for Avx512bwToken {}
    impl HasSse42 for Avx512bwVlToken {}
    impl HasSse42 for Avx512Vbmi2Token {}
    impl HasSse42 for Avx512Vbmi2VlToken {}
    impl HasSse42 for X64V4Token {}

    impl HasAvx for Avx512fToken {}
    impl HasAvx for Avx512fVlToken {}
    impl HasAvx for Avx512bwToken {}
    impl HasAvx for Avx512bwVlToken {}
    impl HasAvx for Avx512Vbmi2Token {}
    impl HasAvx for Avx512Vbmi2VlToken {}
    impl HasAvx for X64V4Token {}

    impl HasAvx2 for Avx512fToken {}
    impl HasAvx2 for Avx512fVlToken {}
    impl HasAvx2 for Avx512bwToken {}
    impl HasAvx2 for Avx512bwVlToken {}
    impl HasAvx2 for Avx512Vbmi2Token {}
    impl HasAvx2 for Avx512Vbmi2VlToken {}
    impl HasAvx2 for X64V4Token {}

    impl HasAvx512f for Avx512fToken {}
    impl HasAvx512f for Avx512fVlToken {}
    impl HasAvx512f for Avx512bwToken {}
    impl HasAvx512f for Avx512bwVlToken {}
    impl HasAvx512f for Avx512Vbmi2Token {}
    impl HasAvx512f for Avx512Vbmi2VlToken {}
    impl HasAvx512f for X64V4Token {}

    impl HasAvx512vl for Avx512fVlToken {}
    impl HasAvx512vl for Avx512bwVlToken {}
    impl HasAvx512vl for Avx512Vbmi2VlToken {}
    impl HasAvx512vl for X64V4Token {}

    impl HasAvx512bw for Avx512bwToken {}
    impl HasAvx512bw for Avx512bwVlToken {}
    impl HasAvx512bw for Avx512Vbmi2Token {}
    impl HasAvx512bw for Avx512Vbmi2VlToken {}
    impl HasAvx512bw for X64V4Token {}

    impl HasAvx512dq for X64V4Token {}

    impl HasAvx512vbmi2 for Avx512Vbmi2Token {}
    impl HasAvx512vbmi2 for Avx512Vbmi2VlToken {}
    impl HasAvx512vbmi2 for Avx512ModernToken {}

    // Avx512Token
    impl Has128BitSimd for Avx512Token {}
    impl Has256BitSimd for Avx512Token {}
    impl Has512BitSimd for Avx512Token {}
    impl HasFma for Avx512Token {}
    impl HasSse for Avx512Token {}
    impl HasSse2 for Avx512Token {}
    impl HasSse41 for Avx512Token {}
    impl HasSse42 for Avx512Token {}
    impl HasAvx for Avx512Token {}
    impl HasAvx2 for Avx512Token {}
    impl HasAvx512f for Avx512Token {}
    impl HasAvx512vl for Avx512Token {}
    impl HasAvx512bw for Avx512Token {}
    impl HasAvx512dq for Avx512Token {}

    // Avx512ModernToken (Ice Lake / Zen 4)
    impl Has128BitSimd for Avx512ModernToken {}
    impl Has256BitSimd for Avx512ModernToken {}
    impl Has512BitSimd for Avx512ModernToken {}
    impl HasFma for Avx512ModernToken {}
    impl HasSse for Avx512ModernToken {}
    impl HasSse2 for Avx512ModernToken {}
    impl HasSse41 for Avx512ModernToken {}
    impl HasSse42 for Avx512ModernToken {}
    impl HasAvx for Avx512ModernToken {}
    impl HasAvx2 for Avx512ModernToken {}
    impl HasAvx512f for Avx512ModernToken {}
    impl HasAvx512vl for Avx512ModernToken {}
    impl HasAvx512bw for Avx512ModernToken {}
    impl HasAvx512dq for Avx512ModernToken {}

    // Avx512Fp16Token (Sapphire Rapids+)
    impl Has128BitSimd for Avx512Fp16Token {}
    impl Has256BitSimd for Avx512Fp16Token {}
    impl Has512BitSimd for Avx512Fp16Token {}
    impl HasFma for Avx512Fp16Token {}
    impl HasSse for Avx512Fp16Token {}
    impl HasSse2 for Avx512Fp16Token {}
    impl HasSse41 for Avx512Fp16Token {}
    impl HasSse42 for Avx512Fp16Token {}
    impl HasAvx for Avx512Fp16Token {}
    impl HasAvx2 for Avx512Fp16Token {}
    impl HasAvx512f for Avx512Fp16Token {}
    impl HasAvx512vl for Avx512Fp16Token {}
    impl HasAvx512bw for Avx512Fp16Token {}
    impl HasAvx512dq for Avx512Fp16Token {}
}

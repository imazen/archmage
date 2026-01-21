//! x86 token stubs for non-x86 architectures.
//!
//! These types exist so cross-platform code can reference them without cfg guards.
//! `summon()` always returns `None` on non-x86.

use super::{CompositeToken, SimdToken};
use super::{Has128BitSimd, Has256BitSimd, Has512BitSimd, HasFma};
use super::{
    HasAvx, HasAvx2, HasAvx512bw, HasAvx512f, HasAvx512vbmi2, HasAvx512vl, HasSse, HasSse2,
    HasSse41, HasSse42,
};

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

// Define all x86 token stubs
define_x86_stub!(SseToken, "SSE");
define_x86_stub!(Sse2Token, "SSE2");
define_x86_stub!(Sse41Token, "SSE4.1");
define_x86_stub!(Sse42Token, "SSE4.2");
define_x86_stub!(AvxToken, "AVX");
define_x86_stub!(Avx2Token, "AVX2");
define_x86_stub!(FmaToken, "FMA");
define_x86_stub!(Avx512fToken, "AVX-512F");
define_x86_stub!(Avx512bwToken, "AVX-512BW");
define_x86_stub!(Avx512fVlToken, "AVX-512F+VL");
define_x86_stub!(Avx512bwVlToken, "AVX-512BW+VL");
define_x86_stub!(Avx512Vbmi2Token, "AVX-512VBMI2");
define_x86_stub!(Avx512Vbmi2VlToken, "AVX-512VBMI2+VL");
define_x86_stub!(X64V2Token, "x86-64-v2");
define_x86_stub!(X64V3Token, "x86-64-v3");
define_x86_stub!(X64V4Token, "x86-64-v4");

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
        (
            unsafe { Avx2Token::forge_token_dangerously() },
            unsafe { FmaToken::forge_token_dangerously() },
        )
    }
}

/// Alias for x86-64-v3 (AVX2 + FMA) - stub on non-x86 architectures.
pub type Desktop64 = X64V3Token;
/// Alias for x86-64-v4 (AVX-512) - stub on non-x86 architectures.
pub type Server64 = X64V4Token;

// Implement marker traits for stubs
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

impl HasFma for FmaToken {}
impl HasFma for Avx2FmaToken {}
impl HasFma for X64V3Token {}
impl HasFma for X64V4Token {}
impl HasFma for Avx512fToken {}
impl HasFma for Avx512fVlToken {}
impl HasFma for Avx512bwToken {}
impl HasFma for Avx512bwVlToken {}
impl HasFma for Avx512Vbmi2Token {}
impl HasFma for Avx512Vbmi2VlToken {}

impl HasSse for SseToken {}
impl HasSse for Sse2Token {}
impl HasSse for Sse41Token {}
impl HasSse for Sse42Token {}
impl HasSse for AvxToken {}
impl HasSse for Avx2Token {}
impl HasSse for Avx2FmaToken {}
impl HasSse for FmaToken {}
impl HasSse for Avx512fToken {}
impl HasSse for Avx512fVlToken {}
impl HasSse for Avx512bwToken {}
impl HasSse for Avx512bwVlToken {}
impl HasSse for Avx512Vbmi2Token {}
impl HasSse for Avx512Vbmi2VlToken {}
impl HasSse for X64V2Token {}
impl HasSse for X64V3Token {}
impl HasSse for X64V4Token {}

impl HasSse2 for Sse2Token {}
impl HasSse2 for Sse41Token {}
impl HasSse2 for Sse42Token {}
impl HasSse2 for AvxToken {}
impl HasSse2 for Avx2Token {}
impl HasSse2 for Avx2FmaToken {}
impl HasSse2 for FmaToken {}
impl HasSse2 for Avx512fToken {}
impl HasSse2 for Avx512fVlToken {}
impl HasSse2 for Avx512bwToken {}
impl HasSse2 for Avx512bwVlToken {}
impl HasSse2 for Avx512Vbmi2Token {}
impl HasSse2 for Avx512Vbmi2VlToken {}
impl HasSse2 for X64V2Token {}
impl HasSse2 for X64V3Token {}
impl HasSse2 for X64V4Token {}

impl HasSse41 for Sse41Token {}
impl HasSse41 for Sse42Token {}
impl HasSse41 for AvxToken {}
impl HasSse41 for Avx2Token {}
impl HasSse41 for Avx2FmaToken {}
impl HasSse41 for Avx512fToken {}
impl HasSse41 for Avx512fVlToken {}
impl HasSse41 for Avx512bwToken {}
impl HasSse41 for Avx512bwVlToken {}
impl HasSse41 for Avx512Vbmi2Token {}
impl HasSse41 for Avx512Vbmi2VlToken {}
impl HasSse41 for X64V2Token {}
impl HasSse41 for X64V3Token {}
impl HasSse41 for X64V4Token {}

impl HasSse42 for Sse42Token {}
impl HasSse42 for AvxToken {}
impl HasSse42 for Avx2Token {}
impl HasSse42 for Avx2FmaToken {}
impl HasSse42 for Avx512fToken {}
impl HasSse42 for Avx512fVlToken {}
impl HasSse42 for Avx512bwToken {}
impl HasSse42 for Avx512bwVlToken {}
impl HasSse42 for Avx512Vbmi2Token {}
impl HasSse42 for Avx512Vbmi2VlToken {}
impl HasSse42 for X64V2Token {}
impl HasSse42 for X64V3Token {}
impl HasSse42 for X64V4Token {}

impl HasAvx for AvxToken {}
impl HasAvx for Avx2Token {}
impl HasAvx for Avx2FmaToken {}
impl HasAvx for Avx512fToken {}
impl HasAvx for Avx512fVlToken {}
impl HasAvx for Avx512bwToken {}
impl HasAvx for Avx512bwVlToken {}
impl HasAvx for Avx512Vbmi2Token {}
impl HasAvx for Avx512Vbmi2VlToken {}
impl HasAvx for X64V3Token {}
impl HasAvx for X64V4Token {}

impl HasAvx2 for Avx2Token {}
impl HasAvx2 for Avx2FmaToken {}
impl HasAvx2 for Avx512fToken {}
impl HasAvx2 for Avx512fVlToken {}
impl HasAvx2 for Avx512bwToken {}
impl HasAvx2 for Avx512bwVlToken {}
impl HasAvx2 for Avx512Vbmi2Token {}
impl HasAvx2 for Avx512Vbmi2VlToken {}
impl HasAvx2 for X64V3Token {}
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

impl HasAvx512vbmi2 for Avx512Vbmi2Token {}
impl HasAvx512vbmi2 for Avx512Vbmi2VlToken {}

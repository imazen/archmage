//! x86_64 SIMD capability tokens
//!
//! Provides tokens for SSE2, SSE4.1, AVX, AVX2, AVX-512, and FMA.
//!
//! Token construction uses [`crate::is_x86_feature_available!`] which combines
//! compile-time and runtime detection. When compiled with a target feature
//! enabled (e.g., in a `#[multiversed]` function), the runtime check is
//! completely eliminated.

use super::{CompositeToken, SimdToken};

// Re-export AVX-512 tokens from the dedicated module
#[cfg(feature = "avx512")]
pub use super::x86_avx512::{Avx512Fp16Token, Avx512ModernToken, Avx512Token, X64V4Token};

// ============================================================================
// SSE4.1 Token
// ============================================================================

/// Proof that SSE4.1 is available.
///
/// SSE4.1 adds blend, round, and other useful instructions.
#[derive(Clone, Copy, Debug)]
pub struct Sse41Token {
    _private: (),
}

impl SimdToken for Sse41Token {
    const NAME: &'static str = "SSE4.1";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("sse4.1") {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

// ============================================================================
// AVX Token
// ============================================================================

/// Proof that AVX is available.
///
/// AVX provides 256-bit floating-point vectors and VEX encoding.
#[derive(Clone, Copy, Debug)]
pub struct AvxToken {
    _private: (),
}

impl SimdToken for AvxToken {
    const NAME: &'static str = "AVX";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // Explicitly check all implied features for robustness against broken emulators
        if crate::is_x86_feature_available!("avx")
            && crate::is_x86_feature_available!("sse4.1")
        {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl AvxToken {
    /// Get an SSE4.1 token (AVX implies SSE4.1)
    #[inline(always)]
    pub fn sse41(self) -> Sse41Token {
        unsafe { Sse41Token::forge_token_dangerously() }
    }
}

// ============================================================================
// AVX2 Token
// ============================================================================

/// Proof that AVX2 is available.
///
/// AVX2 provides 256-bit integer operations and gather instructions.
/// This is the most commonly targeted feature level for SIMD code.
#[derive(Clone, Copy, Debug)]
pub struct Avx2Token {
    _private: (),
}

impl SimdToken for Avx2Token {
    const NAME: &'static str = "AVX2";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // Explicitly check all implied features for robustness against broken emulators
        if crate::is_x86_feature_available!("avx2")
            && crate::is_x86_feature_available!("avx")
            && crate::is_x86_feature_available!("sse4.1")
        {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl Avx2Token {
    /// Get an AVX token (AVX2 implies AVX)
    #[inline(always)]
    pub fn avx(self) -> AvxToken {
        unsafe { AvxToken::forge_token_dangerously() }
    }

    /// Get an SSE4.1 token (AVX2 implies SSE4.1)
    #[inline(always)]
    pub fn sse41(self) -> Sse41Token {
        unsafe { Sse41Token::forge_token_dangerously() }
    }
}

// ============================================================================
// FMA Token
// ============================================================================

/// Proof that FMA (Fused Multiply-Add) is available.
///
/// FMA is independent of AVX2 in the feature hierarchy but almost always
/// available together on modern CPUs. Use `Avx2FmaToken` for the common case.
#[derive(Clone, Copy, Debug)]
pub struct FmaToken {
    _private: (),
}

impl SimdToken for FmaToken {
    const NAME: &'static str = "FMA";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("fma") {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

// ============================================================================
// Combined AVX2 + FMA Token
// ============================================================================

/// Combined proof that both AVX2 and FMA are available.
///
/// This is the most common token for floating-point SIMD work.
/// Almost all CPUs with AVX2 also have FMA (Haswell and later).
#[derive(Clone, Copy, Debug)]
pub struct Avx2FmaToken {
    avx2: Avx2Token,
    fma: FmaToken,
}

impl SimdToken for Avx2FmaToken {
    const NAME: &'static str = "AVX2+FMA";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // Explicitly check all implied features for robustness against broken emulators
        if crate::is_x86_feature_available!("avx2")
            && crate::is_x86_feature_available!("fma")
            && crate::is_x86_feature_available!("avx")
            && crate::is_x86_feature_available!("sse4.1")
        {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self {
            avx2: unsafe { Avx2Token::forge_token_dangerously() },
            fma: unsafe { FmaToken::forge_token_dangerously() },
        }
    }
}

impl CompositeToken for Avx2FmaToken {
    type Components = (Avx2Token, FmaToken);

    #[inline(always)]
    fn components(&self) -> Self::Components {
        (self.avx2, self.fma)
    }
}

impl Avx2FmaToken {
    /// Get the AVX2 component token
    #[inline(always)]
    pub fn avx2(&self) -> Avx2Token {
        self.avx2
    }

    /// Get the FMA component token
    #[inline(always)]
    pub fn fma(&self) -> FmaToken {
        self.fma
    }

    /// Get an AVX token
    #[inline(always)]
    pub fn avx(&self) -> AvxToken {
        self.avx2.avx()
    }

    /// Get an SSE4.1 token
    #[inline(always)]
    pub fn sse41(&self) -> Sse41Token {
        self.avx2.sse41()
    }
}

// ============================================================================
// AVX-512 Tokens
// ============================================================================

/// Proof that AVX-512F (Foundation) is available.
///
/// AVX-512F is the base AVX-512 feature set with 512-bit vectors.
#[cfg(feature = "avx512")]
#[derive(Clone, Copy, Debug)]
pub struct Avx512fToken {
    _private: (),
}

#[cfg(feature = "avx512")]
impl SimdToken for Avx512fToken {
    const NAME: &'static str = "AVX-512F";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("avx512f") {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

#[cfg(feature = "avx512")]
impl Avx512fToken {
    /// Get an AVX2 token (AVX-512F implies AVX2)
    #[inline(always)]
    pub fn avx2(self) -> Avx2Token {
        unsafe { Avx2Token::forge_token_dangerously() }
    }

    /// Get an FMA token (AVX-512F implies FMA)
    #[inline(always)]
    pub fn fma(self) -> FmaToken {
        unsafe { FmaToken::forge_token_dangerously() }
    }

    /// Get a combined AVX2+FMA token
    #[inline(always)]
    pub fn avx2_fma(self) -> Avx2FmaToken {
        unsafe { Avx2FmaToken::forge_token_dangerously() }
    }
}

/// Proof that AVX-512BW (Byte and Word) is available.
#[cfg(feature = "avx512")]
#[derive(Clone, Copy, Debug)]
pub struct Avx512bwToken {
    _private: (),
}

#[cfg(feature = "avx512")]
impl SimdToken for Avx512bwToken {
    const NAME: &'static str = "AVX-512BW";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("avx512bw") {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

#[cfg(feature = "avx512")]
impl Avx512bwToken {
    /// Get an AVX-512F token (AVX-512BW implies AVX-512F)
    #[inline(always)]
    pub fn avx512f(self) -> Avx512fToken {
        unsafe { Avx512fToken::forge_token_dangerously() }
    }
}

// ============================================================================
// AVX-512 + VL Tokens (for 128/256-bit variants of AVX-512 instructions)
// ============================================================================

/// Proof that AVX-512F + AVX-512VL are available.
///
/// AVX-512VL (Vector Length) extensions allow AVX-512 instructions to operate
/// on 128-bit and 256-bit vectors, not just 512-bit. This is required for
/// functions like `_mm_loadu_epi32` and `_mm256_loadu_epi32`.
#[cfg(feature = "avx512")]
#[derive(Clone, Copy, Debug)]
pub struct Avx512fVlToken {
    _private: (),
}

#[cfg(feature = "avx512")]
impl SimdToken for Avx512fVlToken {
    const NAME: &'static str = "AVX-512F+VL";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("avx512f")
            && crate::is_x86_feature_available!("avx512vl")
        {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

#[cfg(feature = "avx512")]
impl Avx512fVlToken {
    /// Get an AVX-512F token
    #[inline(always)]
    pub fn avx512f(self) -> Avx512fToken {
        unsafe { Avx512fToken::forge_token_dangerously() }
    }

    /// Get an AVX2 token
    #[inline(always)]
    pub fn avx2(self) -> Avx2Token {
        unsafe { Avx2Token::forge_token_dangerously() }
    }
}

/// Proof that AVX-512BW + AVX-512VL are available.
///
/// Required for byte/word operations on 128-bit and 256-bit vectors,
/// such as `_mm_loadu_epi8`, `_mm256_loadu_epi16`, etc.
#[cfg(feature = "avx512")]
#[derive(Clone, Copy, Debug)]
pub struct Avx512bwVlToken {
    _private: (),
}

#[cfg(feature = "avx512")]
impl SimdToken for Avx512bwVlToken {
    const NAME: &'static str = "AVX-512BW+VL";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("avx512bw")
            && crate::is_x86_feature_available!("avx512vl")
        {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

#[cfg(feature = "avx512")]
impl Avx512bwVlToken {
    /// Get an AVX-512BW token
    #[inline(always)]
    pub fn avx512bw(self) -> Avx512bwToken {
        unsafe { Avx512bwToken::forge_token_dangerously() }
    }

    /// Get an AVX-512F+VL token
    #[inline(always)]
    pub fn avx512f_vl(self) -> Avx512fVlToken {
        unsafe { Avx512fVlToken::forge_token_dangerously() }
    }
}

/// Proof that AVX-512 VBMI2 is available.
///
/// AVX-512 VBMI2 (Vector Byte Manipulation Instructions 2) provides
/// compress/expand operations for byte and word elements.
/// Available on Ice Lake+, Zen 4+.
#[cfg(feature = "avx512")]
#[derive(Clone, Copy, Debug)]
pub struct Avx512Vbmi2Token {
    _private: (),
}

#[cfg(feature = "avx512")]
impl SimdToken for Avx512Vbmi2Token {
    const NAME: &'static str = "AVX-512VBMI2";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("avx512vbmi2") {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

#[cfg(feature = "avx512")]
impl Avx512Vbmi2Token {
    /// Get an AVX-512BW token (VBMI2 implies BW)
    #[inline(always)]
    pub fn avx512bw(self) -> Avx512bwToken {
        unsafe { Avx512bwToken::forge_token_dangerously() }
    }
}

/// Proof that AVX-512 VBMI2 + VL are available.
///
/// Required for compress/expand operations on 128-bit and 256-bit vectors.
#[cfg(feature = "avx512")]
#[derive(Clone, Copy, Debug)]
pub struct Avx512Vbmi2VlToken {
    _private: (),
}

#[cfg(feature = "avx512")]
impl SimdToken for Avx512Vbmi2VlToken {
    const NAME: &'static str = "AVX-512VBMI2+VL";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("avx512vbmi2")
            && crate::is_x86_feature_available!("avx512vl")
        {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

#[cfg(feature = "avx512")]
impl Avx512Vbmi2VlToken {
    /// Get an AVX-512 VBMI2 token
    #[inline(always)]
    pub fn avx512vbmi2(self) -> Avx512Vbmi2Token {
        unsafe { Avx512Vbmi2Token::forge_token_dangerously() }
    }

    /// Get an AVX-512BW+VL token
    #[inline(always)]
    pub fn avx512bw_vl(self) -> Avx512bwVlToken {
        unsafe { Avx512bwVlToken::forge_token_dangerously() }
    }
}

// ============================================================================
// x86-64 Microarchitecture Level Tokens (Profiles)
// ============================================================================
//
// These match the x86-64 psABI microarchitecture levels used by multiversed:
// https://gitlab.com/x86-psABIs/x86-64-ABI
//
// | Level | Key Features                           | Hardware              |
// |-------|----------------------------------------|-----------------------|
// | v1    | SSE, SSE2 (baseline x86_64)            | All x86_64            |
// | v2    | + SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT  | Nehalem 2008+         |
// | v3    | + AVX, AVX2, FMA, BMI1, BMI2           | Haswell 2013+, Zen 1+ |
// | v4    | + AVX-512F/BW/CD/DQ/VL                 | Xeon 2017+, Zen 4+    |

/// Proof that SSE4.2 + POPCNT are available (x86-64-v2 level).
///
/// x86-64-v2 implies: SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CX16, SAHF.
/// This is the Nehalem (2008) / Bulldozer (2011) baseline.
#[derive(Clone, Copy, Debug)]
pub struct X64V2Token {
    _private: (),
}

impl SimdToken for X64V2Token {
    const NAME: &'static str = "x86-64-v2";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // Explicitly check all v2 features for robustness against broken emulators
        if crate::is_x86_feature_available!("sse4.2")
            && crate::is_x86_feature_available!("sse4.1")
            && crate::is_x86_feature_available!("popcnt")
        {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl X64V2Token {
    /// Get an SSE4.2 token
    #[inline(always)]
    pub fn sse42(self) -> Sse42Token {
        unsafe { Sse42Token::forge_token_dangerously() }
    }

    /// Get an SSE4.1 token
    #[inline(always)]
    pub fn sse41(self) -> Sse41Token {
        unsafe { Sse41Token::forge_token_dangerously() }
    }
}

/// Proof that AVX2 + FMA + BMI1/2 are available (x86-64-v3 level).
///
/// x86-64-v3 implies all of v2 plus: AVX, AVX2, FMA, BMI1, BMI2, F16C, LZCNT, MOVBE.
/// This is the Haswell (2013) / Zen 1 (2017) baseline.
///
/// This is the most commonly targeted level for high-performance SIMD code.
#[derive(Clone, Copy, Debug)]
pub struct X64V3Token {
    _private: (),
}

impl SimdToken for X64V3Token {
    const NAME: &'static str = "x86-64-v3";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // Explicitly check all v3 features for robustness against broken emulators
        if crate::is_x86_feature_available!("avx2")
            && crate::is_x86_feature_available!("fma")
            && crate::is_x86_feature_available!("bmi2")
            && crate::is_x86_feature_available!("avx")
            && crate::is_x86_feature_available!("sse4.2")
            && crate::is_x86_feature_available!("sse4.1")
            && crate::is_x86_feature_available!("popcnt")
        {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl X64V3Token {
    /// Get a v2 token (v3 implies v2)
    #[inline(always)]
    pub fn v2(self) -> X64V2Token {
        unsafe { X64V2Token::forge_token_dangerously() }
    }

    /// Get an AVX2+FMA combined token
    #[inline(always)]
    pub fn avx2_fma(self) -> Avx2FmaToken {
        unsafe { Avx2FmaToken::forge_token_dangerously() }
    }

    /// Get an AVX2 token
    #[inline(always)]
    pub fn avx2(self) -> Avx2Token {
        unsafe { Avx2Token::forge_token_dangerously() }
    }

    /// Get an FMA token
    #[inline(always)]
    pub fn fma(self) -> FmaToken {
        unsafe { FmaToken::forge_token_dangerously() }
    }

    /// Get an AVX token
    #[inline(always)]
    pub fn avx(self) -> AvxToken {
        unsafe { AvxToken::forge_token_dangerously() }
    }

    /// Get an SSE4.2 token
    #[inline(always)]
    pub fn sse42(self) -> Sse42Token {
        unsafe { Sse42Token::forge_token_dangerously() }
    }

    /// Get an SSE4.1 token
    #[inline(always)]
    pub fn sse41(self) -> Sse41Token {
        unsafe { Sse41Token::forge_token_dangerously() }
    }
}

// ============================================================================
// SSE4.2 Token (needed for v2)
// ============================================================================

/// Proof that SSE4.2 is available.
///
/// SSE4.2 adds string/text processing and CRC32 instructions.
#[derive(Clone, Copy, Debug)]
pub struct Sse42Token {
    _private: (),
}

impl SimdToken for Sse42Token {
    const NAME: &'static str = "SSE4.2";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("sse4.2") {
            Some(unsafe { Self::forge_token_dangerously() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl Sse42Token {
    /// Get an SSE4.1 token (SSE4.2 implies SSE4.1)
    #[inline(always)]
    pub fn sse41(self) -> Sse41Token {
        unsafe { Sse41Token::forge_token_dangerously() }
    }
}

// ============================================================================
// Tier Marker Trait Implementations
// ============================================================================
//
// Based on LLVM x86-64 microarchitecture levels.
// Only tier traits (HasX64V2, HasX64V4) and width traits are implemented.

use super::{Has128BitSimd, Has256BitSimd, HasX64V2};
#[cfg(feature = "avx512")]
use super::{Has512BitSimd, HasX64V4};

// HasX64V2: v2 and above
impl HasX64V2 for X64V2Token {}
impl HasX64V2 for X64V3Token {}
impl HasX64V2 for Avx2FmaToken {}

// Width traits: 128-bit
impl Has128BitSimd for Sse41Token {}
impl Has128BitSimd for Sse42Token {}
impl Has128BitSimd for AvxToken {}
impl Has128BitSimd for Avx2Token {}
impl Has128BitSimd for Avx2FmaToken {}
impl Has128BitSimd for X64V2Token {}
impl Has128BitSimd for X64V3Token {}

// Width traits: 256-bit
impl Has256BitSimd for AvxToken {}
impl Has256BitSimd for Avx2Token {}
impl Has256BitSimd for Avx2FmaToken {}
impl Has256BitSimd for X64V3Token {}

// ============================================================================
// AVX-512 tier trait implementations (requires "avx512" feature)
// ============================================================================
#[cfg(feature = "avx512")]
mod avx512_tier_impls {
    use super::*;

    // HasX64V2 for AVX-512 tokens (v4 implies v2)
    impl HasX64V2 for X64V4Token {}
    impl HasX64V2 for Avx512ModernToken {}
    impl HasX64V2 for Avx512Fp16Token {}

    // HasX64V4: v4 and above
    impl HasX64V4 for X64V4Token {}
    impl HasX64V4 for Avx512ModernToken {}
    impl HasX64V4 for Avx512Fp16Token {}

    // Width traits for AVX-512 tokens
    impl Has128BitSimd for X64V4Token {}
    impl Has256BitSimd for X64V4Token {}
    impl Has512BitSimd for X64V4Token {}

    impl Has128BitSimd for Avx512ModernToken {}
    impl Has256BitSimd for Avx512ModernToken {}
    impl Has512BitSimd for Avx512ModernToken {}

    impl Has128BitSimd for Avx512Fp16Token {}
    impl Has256BitSimd for Avx512Fp16Token {}
    impl Has512BitSimd for Avx512Fp16Token {}

    // Individual AVX-512 tokens (for backwards compatibility with existing code)
    impl Has128BitSimd for Avx512fToken {}
    impl Has256BitSimd for Avx512fToken {}
    impl Has512BitSimd for Avx512fToken {}

    impl Has128BitSimd for Avx512bwToken {}
    impl Has256BitSimd for Avx512bwToken {}
    impl Has512BitSimd for Avx512bwToken {}

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
}

// ============================================================================
// Friendly Aliases
// ============================================================================

/// The recommended baseline for desktop x86_64 (AVX2 + FMA + BMI2).
///
/// This is an alias for [`X64V3Token`], covering all Intel Haswell (2013+) and
/// AMD Zen 1 (2017+) desktop CPUs. Use this as your starting point for desktop
/// applications.
///
/// # Why Desktop64?
///
/// - **Universal on modern desktops**: Every x86_64 desktop/laptop CPU since 2013
/// - **Best performance/compatibility tradeoff**: AVX2 gives 256-bit vectors, FMA
///   enables fused multiply-add
/// - **Excludes AVX-512**: Intel removed AVX-512 from consumer chips (12th-14th gen)
///   due to hybrid P+E core architecture, making it unreliable for desktop targeting
///
/// # Example
///
/// ```rust,ignore
/// use archmage::{Desktop64, SimdToken, arcane};
///
/// #[arcane]
/// fn process(token: Desktop64, data: &mut [f32; 8]) {
///     // AVX2 + FMA intrinsics safe here
/// }
///
/// if let Some(token) = Desktop64::try_new() {
///     process(token, &mut data);
/// }
/// ```
pub type Desktop64 = X64V3Token;

// ============================================================================
// Assembly verification helpers
// ============================================================================

/// Helper to verify Avx2Token::try_new assembly.
/// Without +avx2: runtime detection. With +avx2: just returns Some.
#[inline(never)]
pub fn verify_avx2_try_new() -> Option<Avx2Token> {
    Avx2Token::try_new()
}

/// Helper to verify Avx2FmaToken::try_new assembly.
#[inline(never)]
pub fn verify_avx2_fma_try_new() -> Option<Avx2FmaToken> {
    Avx2FmaToken::try_new()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod tests {
    use super::*;

    #[test]
    fn test_token_is_zst() {
        // Tokens should be zero-sized
        assert_eq!(core::mem::size_of::<Avx2Token>(), 0);
        assert_eq!(core::mem::size_of::<FmaToken>(), 0);
        // Combined token is also ZST (contains two ZSTs)
        assert_eq!(core::mem::size_of::<Avx2FmaToken>(), 0);
    }

    #[test]
    fn test_token_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<Avx2Token>();
        assert_copy::<FmaToken>();
        assert_copy::<Avx2FmaToken>();
    }

    #[test]
    fn test_runtime_detection() {
        // These may or may not be available depending on CPU
        let _avx2 = Avx2Token::try_new();
        let _fma = FmaToken::try_new();
        let _avx2_fma = Avx2FmaToken::try_new();

        // If AVX2+FMA available, test component access
        if let Some(token) = Avx2FmaToken::try_new() {
            let _avx2 = token.avx2();
            let _fma = token.fma();
            let _avx = token.avx();
            let _sse41 = token.sse41();
        }
    }

    #[test]
    fn test_token_hierarchy() {
        if let Some(avx2) = Avx2Token::try_new() {
            // AVX2 implies AVX, SSE4.1
            let _avx = avx2.avx();
            let _sse41 = avx2.sse41();
        }
    }

    #[test]
    fn test_profile_tokens_zst() {
        // Profile tokens should also be zero-sized
        assert_eq!(core::mem::size_of::<X64V2Token>(), 0);
        assert_eq!(core::mem::size_of::<X64V3Token>(), 0);
        #[cfg(feature = "avx512")]
        assert_eq!(core::mem::size_of::<X64V4Token>(), 0);
        assert_eq!(core::mem::size_of::<Sse42Token>(), 0);
    }

    #[test]
    fn test_v2_token_extraction() {
        if let Some(v2) = X64V2Token::try_new() {
            // v2 can extract SSE4.2, SSE4.1
            let _sse42 = v2.sse42();
            let _sse41 = v2.sse41();
        }
    }

    #[test]
    fn test_v3_token_extraction() {
        if let Some(v3) = X64V3Token::try_new() {
            // v3 can extract v2, AVX2+FMA, AVX2, FMA, AVX, SSE4.2, SSE4.1
            let _v2 = v3.v2();
            let _avx2_fma = v3.avx2_fma();
            let _avx2 = v3.avx2();
            let _fma = v3.fma();
            let _avx = v3.avx();
            let _sse42 = v3.sse42();
            let _sse41 = v3.sse41();
        }
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn test_v4_token_extraction() {
        if let Some(v4) = X64V4Token::try_new() {
            // v4 (alias for Avx512Token) can extract v3, AVX2+FMA, etc.
            let _v3 = v4.v3();
            let _avx2_fma = v4.avx2_fma();
            let _avx2 = v4.avx2();
            let _avx = v4.avx();
            let _sse42 = v4.sse42();
        }
    }

    #[test]
    fn test_profile_hierarchy_consistency() {
        // If v3 is available, v2 should also be available
        if X64V3Token::try_new().is_some() {
            assert!(
                X64V2Token::try_new().is_some(),
                "v3 implies v2 should be available"
            );
        }

        // If v4 is available, both v3 and v2 should be available
        #[cfg(feature = "avx512")]
        if X64V4Token::try_new().is_some() {
            assert!(
                X64V3Token::try_new().is_some(),
                "v4 implies v3 should be available"
            );
            assert!(
                X64V2Token::try_new().is_some(),
                "v4 implies v2 should be available"
            );
        }
    }

    #[test]
    fn test_profile_token_names() {
        assert_eq!(X64V2Token::NAME, "x86-64-v2");
        assert_eq!(X64V3Token::NAME, "x86-64-v3");
        #[cfg(feature = "avx512")]
        {
            // X64V4Token is an alias for Avx512Token, so they have the same NAME
            assert_eq!(X64V4Token::NAME, "AVX-512");
            assert_eq!(Avx512Token::NAME, "AVX-512");
        }
        assert_eq!(Sse42Token::NAME, "SSE4.2");
    }

    // ========================================================================
    // Operation Trait Tests (require composite feature)
    // ========================================================================
    #[cfg(feature = "__composite")]
    mod simd_ops_tests {
        use super::*;
        use crate::composite::simd_ops::{DotProduct, HorizontalOps, Transpose8x8};

        #[test]
        fn test_transpose_trait() {
            if let Some(token) = Avx2Token::try_new() {
                let original: [f32; 64] = core::array::from_fn(|i| i as f32);
                let mut block = original;

                // Use trait method
                token.transpose_8x8(&mut block);

                // Verify transpose
                for row in 0..8 {
                    for col in 0..8 {
                        assert_eq!(original[row * 8 + col], block[col * 8 + row]);
                    }
                }
            }
        }

        #[test]
        fn test_transpose_trait_via_profile() {
            if let Some(token) = X64V3Token::try_new() {
                let original: [f32; 64] = core::array::from_fn(|i| i as f32);
                let mut block = original;

                // Use trait method via profile token
                token.transpose_8x8(&mut block);

                for row in 0..8 {
                    for col in 0..8 {
                        assert_eq!(original[row * 8 + col], block[col * 8 + row]);
                    }
                }
            }
        }

        #[test]
        fn test_dot_product_trait() {
            if let Some(token) = Avx2FmaToken::try_new() {
                let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
                let b: Vec<f32> = vec![1.0; 64];

                // Use trait method
                let result = token.dot_product_f32(&a, &b);
                let expected: f32 = (0..64).map(|i| i as f32).sum();

                assert!((result - expected).abs() < 0.001);
            }
        }

        #[test]
        fn test_horizontal_ops_trait() {
            if let Some(token) = Avx2Token::try_new() {
                let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();

                // Use trait methods
                let sum = token.sum_f32(&data);
                let max = token.max_f32(&data);
                let min = token.min_f32(&data);

                assert!((sum - 5050.0).abs() < 0.001);
                assert!((max - 100.0).abs() < 0.001);
                assert!((min - 1.0).abs() < 0.001);
            }
        }

        #[test]
        fn test_generic_trait_bounds() {
            // This tests that we can write generic code over tokens
            fn process_transpose<T: Transpose8x8>(token: T) {
                let mut block: [f32; 64] = core::array::from_fn(|i| i as f32);
                token.transpose_8x8(&mut block);
            }

            fn process_dot<T: DotProduct>(token: T, a: &[f32], b: &[f32]) -> f32 {
                token.dot_product_f32(a, b)
            }

            fn process_horizontal<T: HorizontalOps>(token: T, data: &[f32]) -> f32 {
                token.sum_f32(data)
            }

            // These compile, proving the traits work
            if let Some(token) = Avx2Token::try_new() {
                process_transpose(token);
                let data = vec![1.0f32; 16];
                let _sum = process_horizontal(token, &data);
                let _dot = process_dot(token, &data, &data);
            }

            if let Some(token) = X64V3Token::try_new() {
                process_transpose(token);
                let data = vec![1.0f32; 16];
                let _sum = process_horizontal(token, &data);
                let _dot = process_dot(token, &data, &data);
            }
        }
    }
}

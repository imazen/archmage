//! AVX-512 capability tokens for x86/x86_64.
//!
//! This module contains all AVX-512 related tokens, from basic AVX-512F
//! to the modern Ice Lake/Zen 4 feature set.

use super::SimdToken;
use super::{Avx2FmaToken, Avx2Token, AvxToken, FmaToken, Sse2Token, Sse41Token, X64V2Token};

// ============================================================================
// AVX-512F Token (Foundation)
// ============================================================================

/// Proof that AVX-512F (Foundation) is available.
///
/// AVX-512F is the base AVX-512 feature set with 512-bit vectors.
#[derive(Clone, Copy, Debug)]
pub struct Avx512fToken {
    _private: (),
}

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

// ============================================================================
// AVX-512BW Token (Byte and Word)
// ============================================================================

/// Proof that AVX-512BW (Byte and Word) is available.
#[derive(Clone, Copy, Debug)]
pub struct Avx512bwToken {
    _private: (),
}

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
#[derive(Clone, Copy, Debug)]
pub struct Avx512fVlToken {
    _private: (),
}

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
#[derive(Clone, Copy, Debug)]
pub struct Avx512bwVlToken {
    _private: (),
}

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

// ============================================================================
// AVX-512 VBMI2 Tokens
// ============================================================================

/// Proof that AVX-512 VBMI2 is available.
///
/// AVX-512 VBMI2 (Vector Byte Manipulation Instructions 2) provides
/// compress/expand operations for byte and word elements.
/// Available on Ice Lake+, Zen 4+.
#[derive(Clone, Copy, Debug)]
pub struct Avx512Vbmi2Token {
    _private: (),
}

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
#[derive(Clone, Copy, Debug)]
pub struct Avx512Vbmi2VlToken {
    _private: (),
}

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
// AVX-512 Modern Token (Ice Lake / Zen 4)
// ============================================================================

// Import traits needed for the macro
use super::{
    sealed::Sealed, Has128BitSimd, Has256BitSimd, Has512BitSimd, HasAvx, HasAvx2, HasAvx512bw,
    HasAvx512cd, HasAvx512dq, HasAvx512f, HasAvx512vbmi2, HasAvx512vl, HasFma, HasSse, HasSse2,
    HasSse41, HasSse42,
};

/// Proof that modern AVX-512 features are available (Ice Lake / Zen 4 level).
///
/// This is the "modern" AVX-512 baseline covering Ice Lake (2019+) and Zen 4+ (2022+):
/// - **AVX-512F/CD/VL/DQ/BW**: Foundation + x86-64-v4 baseline
/// - **AVX-512VPOPCNTDQ**: Vector population count
/// - **AVX-512IFMA**: Integer fused multiply-add
/// - **AVX-512VBMI/VBMI2**: Vector byte manipulation
/// - **AVX-512BITALG**: Bit algorithms
/// - **AVX-512VNNI**: Vector neural network instructions
/// - **AVX-512BF16**: BFloat16 support
/// - **VPCLMULQDQ**: Carry-less multiplication
/// - **GFNI**: Galois field instructions
/// - **VAES**: Vector AES
///
/// NOT available on Skylake-X (lacks VBMI2, VNNI, BF16, etc.).
///
/// Available on:
/// - Intel Ice Lake (2019+), Tiger Lake, Alder Lake S (different from P/E hybrid)
/// - Intel Sapphire Rapids (2023+), Emerald Rapids
/// - AMD Zen 4+ (2022+)
///
/// # Feature-Trait Correspondence
///
/// The following traits are implemented because their features are checked:
/// - `HasAvx512f` ← `"avx512f"`
/// - `HasAvx512cd` ← `"avx512cd"`
/// - `HasAvx512vl` ← `"avx512vl"`
/// - `HasAvx512dq` ← `"avx512dq"`
/// - `HasAvx512bw` ← `"avx512bw"`
/// - `HasAvx512vbmi2` ← `"avx512vbmi2"`
///
/// Implied traits (from hierarchy): `HasAvx2`, `HasAvx`, `HasSse*`, `HasFma`, `Has*BitSimd`
#[derive(Clone, Copy, Debug)]
pub struct Avx512ModernToken {
    _private: (),
}

impl SimdToken for Avx512ModernToken {
    const NAME: &'static str = "AVX-512Modern";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("avx512f")
            && crate::is_x86_feature_available!("avx512cd")
            && crate::is_x86_feature_available!("avx512vl")
            && crate::is_x86_feature_available!("avx512dq")
            && crate::is_x86_feature_available!("avx512bw")
            && crate::is_x86_feature_available!("avx512vpopcntdq")
            && crate::is_x86_feature_available!("avx512ifma")
            && crate::is_x86_feature_available!("avx512vbmi")
            && crate::is_x86_feature_available!("avx512vbmi2")
            && crate::is_x86_feature_available!("avx512bitalg")
            && crate::is_x86_feature_available!("avx512vnni")
            && crate::is_x86_feature_available!("avx512bf16")
            && crate::is_x86_feature_available!("vpclmulqdq")
            && crate::is_x86_feature_available!("gfni")
            && crate::is_x86_feature_available!("vaes")
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

// Trait implementations - generated alongside token definition for consistency
impl Sealed for Avx512ModernToken {}
impl HasAvx512f for Avx512ModernToken {}
impl HasAvx512cd for Avx512ModernToken {}
impl HasAvx512vl for Avx512ModernToken {}
impl HasAvx512dq for Avx512ModernToken {}
impl HasAvx512bw for Avx512ModernToken {}
impl HasAvx512vbmi2 for Avx512ModernToken {}
impl HasFma for Avx512ModernToken {}
impl HasAvx2 for Avx512ModernToken {}
impl HasAvx for Avx512ModernToken {}
impl HasSse42 for Avx512ModernToken {}
impl HasSse41 for Avx512ModernToken {}
impl HasSse2 for Avx512ModernToken {}
impl HasSse for Avx512ModernToken {}
impl Has512BitSimd for Avx512ModernToken {}
impl Has256BitSimd for Avx512ModernToken {}
impl Has128BitSimd for Avx512ModernToken {}

impl Avx512ModernToken {
    /// Get an AVX-512F token
    #[inline(always)]
    pub fn avx512f(self) -> Avx512fToken {
        unsafe { Avx512fToken::forge_token_dangerously() }
    }

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

    /// Get an AVX-512BW+VL token
    #[inline(always)]
    pub fn avx512bw_vl(self) -> Avx512bwVlToken {
        unsafe { Avx512bwVlToken::forge_token_dangerously() }
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
}

// ============================================================================
// AVX-512 FP16 Token (Sapphire Rapids+)
// ============================================================================

/// Proof that AVX-512 FP16 (half-precision) is available.
///
/// AVX-512 FP16 provides native 16-bit floating-point arithmetic in 512-bit
/// vectors, enabling efficient ML inference and scientific computing workloads.
///
/// Available on:
/// - Intel Sapphire Rapids (2023+), Emerald Rapids
/// - NOT available on Skylake-X, Ice Lake, AMD Zen 4
///
/// Note: This is distinct from F16C which only provides conversions.
/// AVX-512 FP16 provides full arithmetic operations.
#[derive(Clone, Copy, Debug)]
pub struct Avx512Fp16Token {
    _private: (),
}

impl SimdToken for Avx512Fp16Token {
    const NAME: &'static str = "AVX-512FP16";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        if crate::is_x86_feature_available!("avx512fp16") {
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

impl Avx512Fp16Token {
    /// Get an Avx512ModernToken (FP16 implies modern AVX-512)
    #[inline(always)]
    pub fn avx512_modern(self) -> Avx512ModernToken {
        unsafe { Avx512ModernToken::forge_token_dangerously() }
    }

    /// Get an AVX-512F token
    #[inline(always)]
    pub fn avx512f(self) -> Avx512fToken {
        unsafe { Avx512fToken::forge_token_dangerously() }
    }

    /// Get an AVX2+FMA combined token
    #[inline(always)]
    pub fn avx2_fma(self) -> Avx2FmaToken {
        unsafe { Avx2FmaToken::forge_token_dangerously() }
    }
}

// ============================================================================
// x86-64-v4 Token (AVX-512 psABI level)
// ============================================================================

/// Proof that AVX-512 (F/BW/CD/DQ/VL) is available (x86-64-v4 level).
///
/// x86-64-v4 implies all of v3 plus: AVX-512F, AVX-512BW, AVX-512CD, AVX-512DQ, AVX-512VL.
/// This is the Xeon Skylake-SP (2017) / Zen 4 (2022) baseline.
///
/// Note: Intel consumer CPUs (12th-14th gen) do NOT have AVX-512 due to E-core limitations.
/// Only Xeon server, i9-X workstation, and AMD Zen 4+ have AVX-512.
#[derive(Clone, Copy, Debug)]
pub struct X64V4Token {
    _private: (),
}

impl SimdToken for X64V4Token {
    const NAME: &'static str = "x86-64-v4";

    #[inline(always)]
    fn try_new() -> Option<Self> {
        // v4 requires all AVX-512 subsets used in the psABI level
        if crate::is_x86_feature_available!("avx512f")
            && crate::is_x86_feature_available!("avx512bw")
            && crate::is_x86_feature_available!("avx512cd")
            && crate::is_x86_feature_available!("avx512dq")
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

impl X64V4Token {
    /// Get a v3 token (v4 implies v3)
    #[inline(always)]
    pub fn v3(self) -> super::X64V3Token {
        unsafe { super::X64V3Token::forge_token_dangerously() }
    }

    /// Get a v2 token (v4 implies v2)
    #[inline(always)]
    pub fn v2(self) -> X64V2Token {
        unsafe { X64V2Token::forge_token_dangerously() }
    }

    /// Get an AVX-512F token
    #[inline(always)]
    pub fn avx512f(self) -> Avx512fToken {
        unsafe { Avx512fToken::forge_token_dangerously() }
    }

    /// Get an AVX-512BW token
    #[inline(always)]
    pub fn avx512bw(self) -> Avx512bwToken {
        unsafe { Avx512bwToken::forge_token_dangerously() }
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

    /// Get an SSE4.1 token
    #[inline(always)]
    pub fn sse41(self) -> Sse41Token {
        unsafe { Sse41Token::forge_token_dangerously() }
    }

    /// Get an SSE2 token
    #[inline(always)]
    pub fn sse2(self) -> Sse2Token {
        unsafe { Sse2Token::forge_token_dangerously() }
    }
}

/// Server/workstation baseline with AVX-512 (x86-64-v4).
///
/// This is an alias for [`X64V4Token`], covering Xeon servers (Skylake-SP 2017+),
/// Intel HEDT workstations, and AMD Zen 4+ CPUs. Use this for server workloads
/// or when you know AVX-512 is available.
///
/// # When to use Server64
///
/// - **Cloud servers**: AWS (Xeon, Graviton doesn't apply here), GCP, Azure Xeon instances
/// - **Workstations**: Intel i9-X series, AMD Threadripper Zen 4+
/// - **AMD Zen 4+**: Ryzen 7000+ series desktop CPUs do have AVX-512
///
/// # When NOT to use Server64
///
/// - **Intel consumer laptops/desktops**: 12th-14th gen Core chips lack AVX-512
/// - **Unknown hardware**: Fall back to [`Desktop64`](super::Desktop64) for broader compatibility
///
/// # Example
///
/// ```rust,ignore
/// use archmage::{Server64, Desktop64, SimdToken, arcane};
///
/// // Try server features first, fall back to desktop
/// if let Some(token) = Server64::summon() {
///     process_avx512(token, &mut data);
/// } else if let Some(token) = Desktop64::summon() {
///     process_avx2(token, &mut data);
/// }
/// ```
pub type Server64 = X64V4Token;

// ============================================================================
// Sealed Trait Implementations
// ============================================================================

impl Sealed for Avx512fToken {}
impl Sealed for Avx512bwToken {}
impl Sealed for Avx512fVlToken {}
impl Sealed for Avx512bwVlToken {}
impl Sealed for Avx512Vbmi2Token {}
impl Sealed for Avx512Vbmi2VlToken {}
// Avx512ModernToken Sealed impl is with its definition above
impl Sealed for Avx512Fp16Token {}
impl Sealed for X64V4Token {}

//! Cross-width raising and lowering for the f32 generic SIMD chain.
//!
//! Provides token-gated `from_halves` (combine two narrower halves into a
//! wider vector) and `low` / `high` / `split` (extract narrower halves)
//! for the two adjacent width pairs of the f32 family:
//!
//! - `f32x4` ↔ `f32x8` (W128 ↔ W256), via [`F32x8FromHalves`]
//! - `f32x8` ↔ `f32x16` (W256 ↔ W512), via [`F32x16FromHalves`]
//!
//! ## Why the token is load-bearing
//!
//! On x86, raising one tier (e.g. two `f32x4` into one `f32x8`) emits an
//! AVX instruction (`vinsertf128`) whose execution requires the AVX
//! target feature. The narrower vectors (`f32x4`) only require SSE; the
//! AVX bit is supplied by the *wider* tier's token. Passing that token
//! to `from_halves` is the proof that emitting AVX is sound. The token
//! shape is uniform across tiers — even when no wider feature is needed
//! (NEON / Wasm128 / Scalar reprs are already arrays of two narrower
//! reprs, so combining is array-construction) — so callers don't need
//! per-platform code.
//!
//! ## Deductive safety proof, per backend
//!
//! Each `from_halves` impl is sound when:
//!
//! 1. **Token guarantees runtime feature presence.** Constructing `T`
//!    requires `T::summon()` (which performs runtime CPU feature
//!    detection) or one of the upgrade extractors from a stronger token
//!    that is itself proven by the same chain. The forge bypass
//!    (`forge_token_dangerously`) is `unsafe` and explicitly off the
//!    soundness chain.
//!
//! 2. **The repr-construction operation is sound under those features.**
//!    For native intrinsic paths, the chosen instruction is documented
//!    by the platform vendor as requiring exactly the features `T` proves
//!    (X64V3Token ⇒ AVX, X64V4Token ⇒ AVX-512F+VL, NeonToken ⇒ AdvSIMD,
//!    Wasm128Token ⇒ simd128, ScalarToken ⇒ no platform features).
//!    For polyfilled paths (e.g. `[__m256; 2]` for `f32x16<X64V3Token>`),
//!    the operation is array construction, which is sound on any platform.
//!
//! 3. **Repr produced is layout-compatible with the target type.** Both
//!    `f32x{4,8,16}<T>` are `#[repr(C)]` over `(T::Repr, T)` where `T`
//!    is a 1-ZST, so wrapping a freshly-constructed `Repr` via
//!    `from_repr_unchecked(token, repr)` is sound — the token is
//!    already proven by the caller's wider input type.
//!
//! Per-impl notes on the polyfill ↔ native distinction live next to
//! each impl below.
//!
//! ## Native-when-possible, polyfill-when-not
//!
//! | Pair | Token | Wider Repr | from_halves emits |
//! |---|---|---|---|
//! | f32x4→f32x8 | ScalarToken | `[f32; 8]` | array concat (8 stores) |
//! | f32x4→f32x8 | NeonToken | `[float32x4_t; 2]` | `[lo, hi]` (zero-cost) |
//! | f32x4→f32x8 | Wasm128Token | `[v128; 2]` | `[lo, hi]` (zero-cost) |
//! | f32x4→f32x8 | X64V3Token | `__m256` (AVX) | `vinsertf128` |
//! | f32x4→f32x8 | X64V4Token | `__m256` (delegated) | `vinsertf128` |
//! | f32x8→f32x16 | ScalarToken | `[f32; 16]` | array concat (16 stores) |
//! | f32x8→f32x16 | NeonToken | `[float32x4_t; 4]` | `[lo[0], lo[1], hi[0], hi[1]]` |
//! | f32x8→f32x16 | Wasm128Token | `[v128; 4]` | `[lo[0], lo[1], hi[0], hi[1]]` |
//! | f32x8→f32x16 | X64V3Token | `[__m256; 2]` | `[lo, hi]` (polyfilled) |
//! | f32x8→f32x16 | X64V4Token | `__m512` (AVX-512) | `vinsertf32x8` (native) |
//!
//! V4xToken / Avx512Fp16Token inherit X64V4Token's behavior at every
//! width via the same backend delegation in `impls/x86_v4_f32_delegated.rs`.

use archmage::SimdToken;

#[cfg(feature = "w512")]
use crate::simd::backends::F32x16Backend;
use crate::simd::backends::sealed::Sealed;
use crate::simd::backends::{F32x4Backend, F32x8Backend};
#[cfg(feature = "w512")]
use crate::simd::generic::f32x16;
use crate::simd::generic::{f32x4, f32x8};

// ============================================================================
// W128 ↔ W256 — `f32x4` ↔ `f32x8`
// ============================================================================

/// Raise/lower between `f32x4<T>` and `f32x8<T>` for the same token `T`.
///
/// Implemented for every token that supplies both [`F32x4Backend`] and
/// [`F32x8Backend`].
pub trait F32x8FromHalves:
    F32x8Backend + F32x4Backend + SimdToken + Sealed + Copy + 'static
{
    /// Combine two 128-bit halves into a 256-bit representation.
    ///
    /// The generic exposure on [`f32x8`] takes a token; this trait
    /// method takes `self` (the token) so soundness flows through every
    /// call — each impl can only be entered via a proven token.
    fn from_halves(
        self,
        lo: <Self as F32x4Backend>::Repr,
        hi: <Self as F32x4Backend>::Repr,
    ) -> <Self as F32x8Backend>::Repr;

    /// Extract the low 128-bit half of a 256-bit representation.
    fn low(self, wide: <Self as F32x8Backend>::Repr) -> <Self as F32x4Backend>::Repr;

    /// Extract the high 128-bit half of a 256-bit representation.
    fn high(self, wide: <Self as F32x8Backend>::Repr) -> <Self as F32x4Backend>::Repr;
}

// X64V3Token (AVX) — load-bearing token, native intrinsics
//
// Soundness: X64V3Token's feature set includes AVX. `_mm256_insertf128_ps`
// requires AVX (Intel SDM Vol. 2A, INSERTF128 — encoded as VEX.256), which
// the token guarantees. `_mm256_castps128_ps256` is a no-op cast.
// `_mm256_extractf128_ps` requires AVX (EXTRACTF128). All operate on stack
// register state with no aliasing concerns.
#[cfg(target_arch = "x86_64")]
impl F32x8FromHalves for archmage::X64V3Token {
    #[inline(always)]
    fn from_halves(
        self,
        lo: core::arch::x86_64::__m128,
        hi: core::arch::x86_64::__m128,
    ) -> core::arch::x86_64::__m256 {
        let _ = self;
        // SAFETY: X64V3Token proves AVX is available; both args are valid
        // initialized __m128 values; intrinsic is pure.
        unsafe {
            core::arch::x86_64::_mm256_insertf128_ps(
                core::arch::x86_64::_mm256_castps128_ps256(lo),
                hi,
                1,
            )
        }
    }
    #[inline(always)]
    fn low(self, wide: core::arch::x86_64::__m256) -> core::arch::x86_64::__m128 {
        let _ = self;
        // SAFETY: X64V3Token proves AVX. _mm256_castps256_ps128 is a free
        // truncation (no instruction in well-optimized output).
        unsafe { core::arch::x86_64::_mm256_castps256_ps128(wide) }
    }
    #[inline(always)]
    fn high(self, wide: core::arch::x86_64::__m256) -> core::arch::x86_64::__m128 {
        let _ = self;
        // SAFETY: X64V3Token proves AVX. _mm256_extractf128_ps with imm=1
        // returns the high lane; intrinsic is pure.
        unsafe { core::arch::x86_64::_mm256_extractf128_ps(wide, 1) }
    }
}

// X64V4Token / X64V4xToken / Avx512Fp16Token — same Repr as V3 (delegation
// installed in impls/x86_v4_f32_delegated.rs). Soundness inherited from V3
// by the V3 ⊊ V4 subset relation (proved in that file's module doc).
// Each impl downcasts via the guaranteed `.v3()` extractor and forwards.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl F32x8FromHalves for archmage::X64V4Token {
    #[inline(always)]
    fn from_halves(
        self,
        lo: core::arch::x86_64::__m128,
        hi: core::arch::x86_64::__m128,
    ) -> core::arch::x86_64::__m256 {
        <archmage::X64V3Token as F32x8FromHalves>::from_halves(self.v3(), lo, hi)
    }
    #[inline(always)]
    fn low(self, wide: core::arch::x86_64::__m256) -> core::arch::x86_64::__m128 {
        <archmage::X64V3Token as F32x8FromHalves>::low(self.v3(), wide)
    }
    #[inline(always)]
    fn high(self, wide: core::arch::x86_64::__m256) -> core::arch::x86_64::__m128 {
        <archmage::X64V3Token as F32x8FromHalves>::high(self.v3(), wide)
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl F32x8FromHalves for archmage::X64V4xToken {
    #[inline(always)]
    fn from_halves(
        self,
        lo: core::arch::x86_64::__m128,
        hi: core::arch::x86_64::__m128,
    ) -> core::arch::x86_64::__m256 {
        <archmage::X64V3Token as F32x8FromHalves>::from_halves(self.v3(), lo, hi)
    }
    #[inline(always)]
    fn low(self, wide: core::arch::x86_64::__m256) -> core::arch::x86_64::__m128 {
        <archmage::X64V3Token as F32x8FromHalves>::low(self.v3(), wide)
    }
    #[inline(always)]
    fn high(self, wide: core::arch::x86_64::__m256) -> core::arch::x86_64::__m128 {
        <archmage::X64V3Token as F32x8FromHalves>::high(self.v3(), wide)
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl F32x8FromHalves for archmage::Avx512Fp16Token {
    #[inline(always)]
    fn from_halves(
        self,
        lo: core::arch::x86_64::__m128,
        hi: core::arch::x86_64::__m128,
    ) -> core::arch::x86_64::__m256 {
        <archmage::X64V3Token as F32x8FromHalves>::from_halves(self.v3(), lo, hi)
    }
    #[inline(always)]
    fn low(self, wide: core::arch::x86_64::__m256) -> core::arch::x86_64::__m128 {
        <archmage::X64V3Token as F32x8FromHalves>::low(self.v3(), wide)
    }
    #[inline(always)]
    fn high(self, wide: core::arch::x86_64::__m256) -> core::arch::x86_64::__m128 {
        <archmage::X64V3Token as F32x8FromHalves>::high(self.v3(), wide)
    }
}

// NeonToken — polyfilled wider Repr. Soundness: array construction; no
// platform feature beyond what NeonToken already proves (AdvSIMD).
#[cfg(target_arch = "aarch64")]
impl F32x8FromHalves for archmage::NeonToken {
    #[inline(always)]
    fn from_halves(
        self,
        lo: core::arch::aarch64::float32x4_t,
        hi: core::arch::aarch64::float32x4_t,
    ) -> [core::arch::aarch64::float32x4_t; 2] {
        let _ = self;
        [lo, hi]
    }
    #[inline(always)]
    fn low(self, wide: [core::arch::aarch64::float32x4_t; 2]) -> core::arch::aarch64::float32x4_t {
        let _ = self;
        wide[0]
    }
    #[inline(always)]
    fn high(self, wide: [core::arch::aarch64::float32x4_t; 2]) -> core::arch::aarch64::float32x4_t {
        let _ = self;
        wide[1]
    }
}

// Wasm128Token — same shape as NEON: polyfill via `[v128; 2]`.
#[cfg(target_arch = "wasm32")]
impl F32x8FromHalves for archmage::Wasm128Token {
    #[inline(always)]
    fn from_halves(
        self,
        lo: core::arch::wasm32::v128,
        hi: core::arch::wasm32::v128,
    ) -> [core::arch::wasm32::v128; 2] {
        let _ = self;
        [lo, hi]
    }
    #[inline(always)]
    fn low(self, wide: [core::arch::wasm32::v128; 2]) -> core::arch::wasm32::v128 {
        let _ = self;
        wide[0]
    }
    #[inline(always)]
    fn high(self, wide: [core::arch::wasm32::v128; 2]) -> core::arch::wasm32::v128 {
        let _ = self;
        wide[1]
    }
}

// ScalarToken — pure array math. Sound on every platform.
impl F32x8FromHalves for archmage::ScalarToken {
    #[inline(always)]
    fn from_halves(self, lo: [f32; 4], hi: [f32; 4]) -> [f32; 8] {
        let _ = self;
        [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
    }
    #[inline(always)]
    fn low(self, wide: [f32; 8]) -> [f32; 4] {
        let _ = self;
        [wide[0], wide[1], wide[2], wide[3]]
    }
    #[inline(always)]
    fn high(self, wide: [f32; 8]) -> [f32; 4] {
        let _ = self;
        [wide[4], wide[5], wide[6], wide[7]]
    }
}

impl<T: F32x8FromHalves> f32x8<T> {
    /// Combine two `f32x4<T>` halves into one `f32x8<T>`.
    ///
    /// The token is load-bearing on x86 (proves AVX); a no-op on other
    /// tiers but kept in the signature for uniform use.
    #[inline(always)]
    pub fn from_halves(token: T, lo: f32x4<T>, hi: f32x4<T>) -> Self {
        Self::from_repr_unchecked(
            token,
            <T as F32x8FromHalves>::from_halves(token, lo.into_repr(), hi.into_repr()),
        )
    }
    /// Extract the low 128-bit half.
    #[inline(always)]
    pub fn low(self) -> f32x4<T> {
        f32x4::from_repr_unchecked(
            self.1,
            <T as F32x8FromHalves>::low(self.1, self.into_repr()),
        )
    }
    /// Extract the high 128-bit half.
    #[inline(always)]
    pub fn high(self) -> f32x4<T> {
        f32x4::from_repr_unchecked(
            self.1,
            <T as F32x8FromHalves>::high(self.1, self.into_repr()),
        )
    }
    /// Split into `(low, high)` halves.
    #[inline(always)]
    pub fn split(self) -> (f32x4<T>, f32x4<T>) {
        (self.low(), self.high())
    }
}

// ============================================================================
// W256 ↔ W512 — `f32x8` ↔ `f32x16`
// ============================================================================

/// Raise/lower between `f32x8<T>` and `f32x16<T>` for the same token `T`.
///
/// Same shape as [`F32x8FromHalves`], one tier wider. Implemented for
/// every token that supplies both [`F32x8Backend`] and [`F32x16Backend`].
///
/// Gated behind magetypes' `w512` feature (default-on) since
/// `F32x16Backend` itself is `w512`-gated.
#[cfg(feature = "w512")]
pub trait F32x16FromHalves:
    F32x16Backend + F32x8Backend + SimdToken + Sealed + Copy + 'static
{
    /// Combine two 256-bit halves into a 512-bit representation.
    fn from_halves(
        self,
        lo: <Self as F32x8Backend>::Repr,
        hi: <Self as F32x8Backend>::Repr,
    ) -> <Self as F32x16Backend>::Repr;

    /// Extract the low 256-bit half of a 512-bit representation.
    fn low(self, wide: <Self as F32x16Backend>::Repr) -> <Self as F32x8Backend>::Repr;

    /// Extract the high 256-bit half of a 512-bit representation.
    fn high(self, wide: <Self as F32x16Backend>::Repr) -> <Self as F32x8Backend>::Repr;
}

// X64V3Token — f32x16 polyfills to `[__m256; 2]`. Sound: array construction;
// no AVX-512 needed (the polyfill repr stays in V3 territory).
#[cfg(all(target_arch = "x86_64", feature = "w512"))]
impl F32x16FromHalves for archmage::X64V3Token {
    #[inline(always)]
    fn from_halves(
        self,
        lo: core::arch::x86_64::__m256,
        hi: core::arch::x86_64::__m256,
    ) -> [core::arch::x86_64::__m256; 2] {
        let _ = self;
        [lo, hi]
    }
    #[inline(always)]
    fn low(self, wide: [core::arch::x86_64::__m256; 2]) -> core::arch::x86_64::__m256 {
        let _ = self;
        wide[0]
    }
    #[inline(always)]
    fn high(self, wide: [core::arch::x86_64::__m256; 2]) -> core::arch::x86_64::__m256 {
        let _ = self;
        wide[1]
    }
}

// X64V4Token — native AVX-512. f32x16<V4>::Repr is `__m512`; f32x8<V4>::Repr
// is V3-delegated `__m256`.
//
// Soundness: V4 token proves AVX-512F + AVX-512DQ + AVX-512VL.
// `_mm512_insertf32x8` (Intel SDM, VINSERTF32X8 — EVEX.512) requires
// AVX-512DQ for the variant taking a __m256, which V4 includes.
// `_mm512_castps256_ps512` is a no-op cast (the upper bits are immediately
// overwritten by the insert). For low/high: `_mm512_castps512_ps256`
// truncates; `_mm512_extractf32x8_ps` (AVX-512DQ) returns the high 256-bit
// lane.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl F32x16FromHalves for archmage::X64V4Token {
    #[inline(always)]
    fn from_halves(
        self,
        lo: core::arch::x86_64::__m256,
        hi: core::arch::x86_64::__m256,
    ) -> core::arch::x86_64::__m512 {
        let _ = self;
        // SAFETY: V4 proves AVX-512F + AVX-512DQ + AVX-512VL.
        unsafe {
            core::arch::x86_64::_mm512_insertf32x8(
                core::arch::x86_64::_mm512_castps256_ps512(lo),
                hi,
                1,
            )
        }
    }
    #[inline(always)]
    fn low(self, wide: core::arch::x86_64::__m512) -> core::arch::x86_64::__m256 {
        let _ = self;
        // SAFETY: V4 proves AVX-512. _mm512_castps512_ps256 is a free truncation.
        unsafe { core::arch::x86_64::_mm512_castps512_ps256(wide) }
    }
    #[inline(always)]
    fn high(self, wide: core::arch::x86_64::__m512) -> core::arch::x86_64::__m256 {
        let _ = self;
        // SAFETY: V4 proves AVX-512DQ. _mm512_extractf32x8_ps requires DQ.
        unsafe { core::arch::x86_64::_mm512_extractf32x8_ps(wide, 1) }
    }
}

// X64V4xToken — same as V4 (DQ available, additional extensions don't
// change f32x16's representation). Forwarded via `.v4()` extractor.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl F32x16FromHalves for archmage::X64V4xToken {
    #[inline(always)]
    fn from_halves(
        self,
        lo: core::arch::x86_64::__m256,
        hi: core::arch::x86_64::__m256,
    ) -> core::arch::x86_64::__m512 {
        <archmage::X64V4Token as F32x16FromHalves>::from_halves(self.v4(), lo, hi)
    }
    #[inline(always)]
    fn low(self, wide: core::arch::x86_64::__m512) -> core::arch::x86_64::__m256 {
        <archmage::X64V4Token as F32x16FromHalves>::low(self.v4(), wide)
    }
    #[inline(always)]
    fn high(self, wide: core::arch::x86_64::__m512) -> core::arch::x86_64::__m256 {
        <archmage::X64V4Token as F32x16FromHalves>::high(self.v4(), wide)
    }
}
// Avx512Fp16Token does NOT implement F32x16Backend on origin/main, so it
// cannot implement F32x16FromHalves. Users with an Avx512Fp16Token who need
// f32x16 must extract to X64V4Token first via the standard token-extractor
// chain. (Avx512Fp16Token does have F32x4 / F32x8 via the delegation in
// `impls/x86_v4_f32_delegated.rs`, so the W128↔W256 pair above works for it.)

// NeonToken — cascade polyfill. f32x8<NEON> is `[float32x4_t; 2]`,
// f32x16<NEON> is `[float32x4_t; 4]`. from_halves flattens.
#[cfg(all(target_arch = "aarch64", feature = "w512"))]
impl F32x16FromHalves for archmage::NeonToken {
    #[inline(always)]
    fn from_halves(
        self,
        lo: [core::arch::aarch64::float32x4_t; 2],
        hi: [core::arch::aarch64::float32x4_t; 2],
    ) -> [core::arch::aarch64::float32x4_t; 4] {
        let _ = self;
        [lo[0], lo[1], hi[0], hi[1]]
    }
    #[inline(always)]
    fn low(
        self,
        wide: [core::arch::aarch64::float32x4_t; 4],
    ) -> [core::arch::aarch64::float32x4_t; 2] {
        let _ = self;
        [wide[0], wide[1]]
    }
    #[inline(always)]
    fn high(
        self,
        wide: [core::arch::aarch64::float32x4_t; 4],
    ) -> [core::arch::aarch64::float32x4_t; 2] {
        let _ = self;
        [wide[2], wide[3]]
    }
}

// Wasm128Token — cascade polyfill, same shape as NEON.
#[cfg(all(target_arch = "wasm32", feature = "w512"))]
impl F32x16FromHalves for archmage::Wasm128Token {
    #[inline(always)]
    fn from_halves(
        self,
        lo: [core::arch::wasm32::v128; 2],
        hi: [core::arch::wasm32::v128; 2],
    ) -> [core::arch::wasm32::v128; 4] {
        let _ = self;
        [lo[0], lo[1], hi[0], hi[1]]
    }
    #[inline(always)]
    fn low(self, wide: [core::arch::wasm32::v128; 4]) -> [core::arch::wasm32::v128; 2] {
        let _ = self;
        [wide[0], wide[1]]
    }
    #[inline(always)]
    fn high(self, wide: [core::arch::wasm32::v128; 4]) -> [core::arch::wasm32::v128; 2] {
        let _ = self;
        [wide[2], wide[3]]
    }
}

// ScalarToken — pure array math.
#[cfg(feature = "w512")]
impl F32x16FromHalves for archmage::ScalarToken {
    #[inline(always)]
    fn from_halves(self, lo: [f32; 8], hi: [f32; 8]) -> [f32; 16] {
        let _ = self;
        let mut out = [0.0f32; 16];
        out[..8].copy_from_slice(&lo);
        out[8..].copy_from_slice(&hi);
        out
    }
    #[inline(always)]
    fn low(self, wide: [f32; 16]) -> [f32; 8] {
        let _ = self;
        let mut out = [0.0f32; 8];
        out.copy_from_slice(&wide[..8]);
        out
    }
    #[inline(always)]
    fn high(self, wide: [f32; 16]) -> [f32; 8] {
        let _ = self;
        let mut out = [0.0f32; 8];
        out.copy_from_slice(&wide[8..]);
        out
    }
}

#[cfg(feature = "w512")]
impl<T: F32x16FromHalves> f32x16<T> {
    /// Combine two `f32x8<T>` halves into one `f32x16<T>`.
    #[inline(always)]
    pub fn from_halves(token: T, lo: f32x8<T>, hi: f32x8<T>) -> Self {
        Self::from_repr_unchecked(
            token,
            <T as F32x16FromHalves>::from_halves(token, lo.into_repr(), hi.into_repr()),
        )
    }
    /// Extract the low 256-bit half.
    #[inline(always)]
    pub fn low(self) -> f32x8<T> {
        f32x8::from_repr_unchecked(
            self.1,
            <T as F32x16FromHalves>::low(self.1, self.into_repr()),
        )
    }
    /// Extract the high 256-bit half.
    #[inline(always)]
    pub fn high(self) -> f32x8<T> {
        f32x8::from_repr_unchecked(
            self.1,
            <T as F32x16FromHalves>::high(self.1, self.into_repr()),
        )
    }
    /// Split into `(low, high)` halves.
    #[inline(always)]
    pub fn split(self) -> (f32x8<T>, f32x8<T>) {
        (self.low(), self.high())
    }
}

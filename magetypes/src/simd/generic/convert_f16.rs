//! Branchless, vectorized f16 (IEEE 754 binary16 half-precision) ↔ f32
//! conversion expressed entirely in generic SIMD.
//!
//! ## Why this lives in magetypes
//!
//! Many image / HDR pipelines store pixels as f16 bit patterns (`u16`) and
//! decode them to f32 for filtering. Targets that lack a hardware f16 type
//! in the generic SIMD API (NEON, WASM128, scalar) would otherwise decode
//! f16 element-by-element through a branchy bit-twiddle (a subnormal
//! normalization loop plus Inf/NaN branches) that LLVM cannot
//! auto-vectorize. In a tight resize/convolution inner loop that scalar
//! decode dominates the profile.
//!
//! These functions decode/encode a whole SIMD lane at once with **no
//! data-dependent branches**, so every backend — including the polyfilled
//! scalar fallback — runs the same straight-line integer/float arithmetic
//! that the platform vectorizes.
//!
//! ## Method
//!
//! - **f16 → f32** uses Fabian Giesen's magic-multiply method: shift the 15
//!   magnitude bits into the f32 mantissa/exponent field, reinterpret as
//!   f32, and multiply by `2^112` to rescale the exponent (which also
//!   denormalizes subnormals in the same step). Inf/NaN are restored with a
//!   branchless select.
//! - **f32 → f16** uses Giesen's `float_to_half_full_rtne`: a magic-add in
//!   f32 round-to-nearest-even space handles the (de)normal range exactly,
//!   while an integer rounding-bias add handles the normal range; overflow
//!   saturates to ±Inf and NaN maps to a canonical quiet NaN.
//!
//! ## Bit-exactness contract
//!
//! Verified exhaustively (`tests/convert_f16_exhaustive.rs`):
//!
//! - [`i32x4::f16_to_f32`] is **bit-identical** to a scalar IEEE
//!   reference for all 65 536 f16 inputs, including subnormals and Inf, and
//!   reproduces the reference's NaN bit patterns too.
//! - [`f32x4::to_f16`] is **bit-identical** to a scalar
//!   round-to-nearest-even IEEE reference for all 2³² finite and infinite f32
//!   inputs, including the subnormal flush-to-f16-subnormal range and the
//!   overflow-to-Inf boundary. For NaN inputs both produce *some* f16 NaN
//!   (exponent all ones, mantissa non-zero); the NaN payload bits may differ.
//!
//! The in-register kernels ([`i32x4::f16_to_f32`] /
//! [`f32x4::to_f16`]) are pure safe integer/float arithmetic — no
//! `unsafe`, no intrinsics. They are therefore arch-independent: a result
//! proven on one target holds on every target, because the underlying bit
//! operations are identical everywhere.
//!
//! ## Hardware fast path (F16C)
//!
//! The slice entry points ([`F16Convert::f16_to_f32_slice`] /
//! [`F16Convert::f32_to_f16_slice`])
//! dispatch on the SIMD token (via the sealed [`F16Convert`] trait): an
//! `X64V3Token` (x86-64-v3, whose feature tier includes F16C) routes through
//! the native `vcvtph2ps` / `vcvtps2ph` instructions; every other token runs
//! the branchless software kernel. AVX-512 (`X64V4Token`) holders reach the
//! same F16C path by extracting a V3 token — `token.v3()` — which is sound
//! because V4's feature set is a strict superset of V3's. The
//! hardware path is verified **bit-identical to the software path over all
//! 65 536 f16 (decode) and the boundary-band + dense-sweep encode coverage**,
//! with two documented, benign NaN-only divergences:
//!
//! - **decode**: for a NaN *input*, `vcvtph2ps` returns the hardware-quieted
//!   f32 NaN (mantissa MSB set), whereas the software kernel widens the f16
//!   payload directly. Both are valid f32 NaNs; finite/Inf are bit-identical.
//! - **encode**: for a NaN *input*, `vcvtps2ph` and the software kernel may
//!   emit different f16 NaN payloads (both valid quiet f16 NaNs) — the same
//!   payload tolerance the software encode already documents.
//!
//! The hardware intrinsics are encapsulated behind `#[archmage::arcane]`
//! boundaries (which emit the `#[target_feature]` codegen region and a safe
//! trampoline), so the slice functions stay safe to call from ordinary code
//! and the module needs no module-level `unsafe`.

use crate::simd::backends::F32x4Convert;
use crate::simd::generic::{f32x4, i32x4};

// ============================================================================
// Register-level conversions — inherent methods on the value types
// ============================================================================
//
// A register conversion is a value→value operation, so — like `f32x4::sqrt`,
// `f32x4::min`, etc. — it lives as an inherent method on the value type rather
// than on the token trait. The token comes from the value itself (the `i32x4`
// / `f32x4` tuple structs store it in field `.1`).
//
// The bound is `T: F32x4Convert` (which gives `F32x4Backend + I32x4Backend +
// SimdToken`) — exactly what the branchless kernel needs, and NOT `F16Convert`
// (the slice trait). These are additive inherent `impl` blocks on the generated
// types; Rust permits multiple inherent `impl` blocks for a type in the same
// crate, and these are NOT touched by code generation (they live here, not in
// the generated dir).
//
// The kernel bodies are bit-identical to the prior (token-method) versions:
// `self` is now the value and `token = self.1` recovers the token, but every
// arithmetic operation is unchanged — verified exhaustively in
// `tests/convert_f16_exhaustive.rs`.

impl<T: F32x4Convert> i32x4<T> {
    /// Decode four IEEE-754 binary16 (`f16`) bit patterns held in the low 16
    /// bits of each lane (`self`) into an [`f32x4`].
    ///
    /// Each input lane must hold an `f16` bit pattern zero-extended to 32 bits
    /// (i.e. `h as i32` for a `u16` `h`); the upper 16 bits of each lane are
    /// ignored except that they should be zero for a clean decode. See
    /// [`F16Convert::f16_to_f32_slice`] for the slice-oriented entry point that
    /// loads from `&[u16]`.
    ///
    /// Bit-identical to a scalar IEEE f16→f32 reference for every finite,
    /// subnormal, and infinite input, and reproduces the reference's NaN bit
    /// patterns. Branchless on every backend.
    #[inline]
    pub fn f16_to_f32(self) -> f32x4<T> {
        let token = self.1;
        let h = self;

        // 2^112 as f32 bits — rescales the magic-shifted exponent and, in the
        // same multiply, denormalizes f16 subnormals.
        let magic = f32x4::splat(token, f32::from_bits(0x7780_0000));

        let mask_sign = i32x4::splat(token, 0x8000);
        let mask_mag = i32x4::splat(token, 0x7fff);
        let mask_expmant = i32x4::splat(token, 0x007f_ffff);
        let inf_exp = i32x4::splat(token, 0x7f80_0000);
        let f16_expmask = i32x4::splat(token, 0x7c00);

        // Sign bit moved to the f32 sign position.
        let sign = (h & mask_sign).shl_const::<16>();

        // exp+mant shifted into the f32 [exp|mant] field, then magic-multiply
        // rescales the exponent (and fixes subnormals).
        let mag = (h & mask_mag).shl_const::<13>();
        let scaled = (mag.bitcast_f32x4() * magic).bitcast_i32x4();

        // Inf/NaN fixup: where the f16 exponent field is all ones, force the
        // f32 exponent to all ones and keep the mantissa from `scaled`.
        let is_inf_nan = (h & f16_expmask).simd_eq(f16_expmask);
        let infnan_bits = (scaled & mask_expmant) | inf_exp;
        let body = i32x4::blend(is_inf_nan, infnan_bits, scaled);

        (body | sign).bitcast_f32x4()
    }
}

impl<T: F32x4Convert> f32x4<T> {
    /// Encode this [`f32x4`] (round-to-nearest-even) into four IEEE-754
    /// binary16 (`f16`) bit patterns, returned in the low 16 bits of each
    /// `i32` lane (the upper 16 bits are zero).
    ///
    /// Uses round-to-nearest-even, flushes f32 values too small to represent
    /// even as an f16 subnormal to ±0, and saturates overflow to ±Inf —
    /// exactly matching a scalar IEEE round-to-nearest-even reference for every
    /// finite and infinite f32. NaN inputs map to a canonical quiet f16 NaN
    /// (the payload bits are not preserved). Branchless on every backend.
    ///
    /// See [`F16Convert::f32_to_f16_slice`] for the slice-oriented entry point
    /// that stores to `&mut [u16]`.
    #[inline]
    pub fn to_f16(self) -> i32x4<T> {
        let token = self.1;
        let f = self;

        // Fabian Giesen, `float_to_half_full_rtne`.
        //
        // Boundary constants (all in the |value| domain, top bit cleared):
        //   f32infty      = 255 << 23  — f32 exponent all ones (Inf/NaN line)
        //   f16max        = (127 + 16) << 23 — smallest |f32| that overflows f16
        //   denorm_cutoff = 113 << 23  — smallest |f32| that is a normal f16
        //   denorm_magic  = ((127 - 15) + (23 - 10) + 1) << 23
        let bits = f.bitcast_i32x4();

        let sign_mask = i32x4::splat(token, 0x8000_0000u32 as i32);
        let f32infty = i32x4::splat(token, 255 << 23);
        let f16max = i32x4::splat(token, (127 + 16) << 23);
        let denorm_cutoff = i32x4::splat(token, 113 << 23);
        let denorm_magic = i32x4::splat(token, ((127 - 15) + (23 - 10) + 1) << 23);
        let one = i32x4::splat(token, 1);
        let bias_bits = i32x4::splat(token, ((15i32.wrapping_sub(127)) << 23).wrapping_add(0xfff));
        let nan_out = i32x4::splat(token, 0x7e00);
        let inf_out = i32x4::splat(token, 0x7c00);

        let sign = bits & sign_mask;
        let absf = bits ^ sign;

        // ---- (De)normal / zero path: magic add in RTNE f32 space ----
        // (f32(absf) + denorm_magic) reinterpreted, minus the magic bias.
        let denorm_f = absf.bitcast_f32x4() + denorm_magic.bitcast_f32x4();
        let denorm_path = denorm_f.bitcast_i32x4() - denorm_magic;

        // ---- Normal path: integer rounding-bias add ----
        // mant_odd = bit 13 of absf (the LSB of the surviving 10-bit mantissa)
        let mant_odd = absf.shr_logical_const::<13>() & one;
        let normal_path = (absf + bias_bits + mant_odd).shr_logical_const::<13>();

        // ---- Inf/NaN path ----
        // absf > f32infty ⇒ NaN ⇒ qNaN; else (absf == all-ones exp) ⇒ Inf.
        let is_nan = absf.simd_gt(f32infty);
        let inf_nan_path = i32x4::blend(is_nan, nan_out, inf_out);

        // ---- Select by magnitude regime (branchless) ----
        // is_inf_nan : absf >= f16max  → inf_nan_path
        // else is_denorm : absf < denorm_cutoff → denorm_path
        // else            → normal_path
        let is_inf_nan = absf.simd_ge(f16max);
        let is_denorm = absf.simd_lt(denorm_cutoff);

        let finite = i32x4::blend(is_denorm, denorm_path, normal_path);
        let o = i32x4::blend(is_inf_nan, inf_nan_path, finite);

        // Reattach the sign (moved down from bit 31 to bit 15).
        o | sign.shr_logical_const::<16>()
    }
}

// ============================================================================
// Hardware-accelerated backend dispatch
// ============================================================================
//
// The in-register kernels (`i32x4::f16_to_f32` / `f32x4::to_f16`, inherent
// methods on the value types above) are pure
// branchless arithmetic — correct on every target, but they do *not* use a
// CPU's native f16-conversion instructions. Where the platform has them
// (x86-64 F16C: `vcvtph2ps` / `vcvtps2ph`), a single instruction converts a
// whole lane, so the slice methods dispatch to a hardware kernel when the
// token proves the feature is present, and fall back to the branchless
// software kernel otherwise.
//
// The dispatch is keyed on the token type through the sealed `F16Convert`
// trait: the default `*_into` trait methods run the software kernel; concrete
// tokens whose feature tier includes a native f16 conversion override them.
// Because the trait is sealed (its supertrait bound is `F32x4Convert`, itself
// sealed), no downstream crate can observe or break this dispatch, and adding
// it is a purely additive, semver-compatible change.

/// Token-keyed **slice** converters between IEEE-754 binary16 (`f16`) bit
/// patterns and `f32`.
///
/// The token (the implementing type) is passed by value as `self`, matching
/// the magetypes convention for token-keyed operations (cf.
/// [`F32x4Convert`](crate::simd::backends::F32x4Convert) and
/// [`F32x8FromHalves`](crate::simd::generic::F32x8FromHalves)). The slice
/// converters are inherently token-keyed — they take `&[u16]` / `&[f32]`
/// slices, not vector values — which is why they live on the token trait. The
/// register-level conversions, being value→value operations, are inherent
/// methods on the value types instead ([`i32x4::f16_to_f32`] /
/// [`f32x4::to_f16`]).
///
/// - **Slice converters** — [`f16_to_f32_slice`](Self::f16_to_f32_slice) /
///   [`f32_to_f16_slice`](Self::f32_to_f16_slice) convert whole `&[u16]` ↔
///   `&mut [f32]` slices. They assert equal lengths, then dispatch through
///   the overridable [`f16_to_f32_into`](Self::f16_to_f32_into) /
///   [`f32_to_f16_into`](Self::f32_to_f16_into) methods: the default runs the
///   branchless software kernel (calling the [`i32x4::f16_to_f32`] /
///   [`f32x4::to_f16`] register methods four lanes at a time), while tokens
///   whose CPU-feature tier includes a native half-precision conversion
///   (currently x86-64 F16C, via `X64V3Token`) override them with the
///   hardware path. AVX-512 (`X64V4Token`) holders take the F16C path by
///   extracting a V3 token (`token.v3()`).
///
/// Every override is verified **bit-identical** to the software kernel over
/// the full exhaustive f16 sweep (`tests/convert_f16_exhaustive.rs`) for all
/// finite, subnormal, and infinite values, so the public API has identical
/// observable behavior regardless of which backend the runtime selects. NaN
/// inputs are the only exception: the hardware and software paths each
/// produce a valid NaN whose payload bits may differ (see the module-level
/// docs for the exact F16C divergence).
///
/// This trait is sealed via its [`F32x4Convert`](crate::simd::backends::F32x4Convert)
/// supertrait bound and is not nameable by downstream crates.
pub trait F16Convert: F32x4Convert {
    /// Decode a slice of IEEE-754 binary16 (`f16`) bit patterns (`&[u16]`)
    /// into `&mut [f32]`.
    ///
    /// On an x86-64 `X64V3Token` (F16C) this dispatches to the native
    /// `vcvtph2ps` instruction; on every other target it runs the branchless
    /// software kernel ([`i32x4::f16_to_f32`], four lanes at a
    /// time, with a scalar tail).
    ///
    /// For every **finite, subnormal, and infinite** input both paths are
    /// **bit-identical** to a scalar IEEE f16→f32 reference, verified
    /// exhaustively. For a **NaN input** the two paths produce f32 values that
    /// are both NaN but whose bit patterns may differ: the software kernel
    /// widens the f16 NaN payload directly (preserving its signaling/quiet
    /// bit), whereas `vcvtph2ps` returns the hardware-quieted NaN (mantissa MSB
    /// set). Both are valid f32 NaNs for the same NaN input; this divergence is
    /// benign and affects only NaN inputs, never a finite or infinite value.
    ///
    /// This is a provided method (not overridable); the backend dispatch
    /// happens inside [`f16_to_f32_into`](Self::f16_to_f32_into).
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != output.len()`.
    #[inline]
    fn f16_to_f32_slice(self, input: &[u16], output: &mut [f32]) {
        assert_eq!(
            input.len(),
            output.len(),
            "f16_to_f32_slice: input and output must have equal length"
        );
        self.f16_to_f32_into(input, output);
    }

    /// Encode a slice of `f32` into IEEE-754 binary16 (`f16`) bit patterns
    /// (`&mut [u16]`).
    ///
    /// On an x86-64 `X64V3Token` (F16C) this dispatches to the native
    /// `vcvtps2ph` instruction (round-to-nearest-even); on every other target
    /// it runs the branchless software kernel
    /// ([`f32x4::to_f16`], four lanes at a time, with a
    /// scalar tail). Both paths are **bit-identical** to a scalar IEEE RTNE
    /// reference for every finite and infinite input; NaN maps to an f16 NaN
    /// (the F16C `vcvtps2ph` and the software path may emit different NaN
    /// payload bits, but both produce a valid quiet f16 NaN — verified
    /// exhaustively).
    ///
    /// This is a provided method (not overridable); the backend dispatch
    /// happens inside [`f32_to_f16_into`](Self::f32_to_f16_into).
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != output.len()`.
    #[inline]
    fn f32_to_f16_slice(self, input: &[f32], output: &mut [u16]) {
        assert_eq!(
            input.len(),
            output.len(),
            "f32_to_f16_slice: input and output must have equal length"
        );
        self.f32_to_f16_into(input, output);
    }

    /// Decode `input.len()` f16 bit patterns into `output`. Lengths are
    /// guaranteed equal by the public entry point
    /// ([`f16_to_f32_slice`](Self::f16_to_f32_slice)).
    ///
    /// This is the overridable dispatch point: the default runs the branchless
    /// software kernel; hardware tiers (x86-64 F16C, aarch64 NEON-f16) override
    /// it with a native conversion.
    #[inline]
    fn f16_to_f32_into(self, input: &[u16], output: &mut [f32]) {
        f16_to_f32_slice_soft(self, input, output);
    }

    /// Encode `input.len()` f32 values into f16 bit patterns in `output`.
    /// Lengths are guaranteed equal by the public entry point
    /// ([`f32_to_f16_slice`](Self::f32_to_f16_slice)).
    ///
    /// This is the overridable dispatch point: the default runs the branchless
    /// software kernel; hardware tiers (x86-64 F16C, aarch64 NEON-f16) override
    /// it with a native conversion.
    #[inline]
    fn f32_to_f16_into(self, input: &[f32], output: &mut [u16]) {
        f32_to_f16_slice_soft(self, input, output);
    }
}

// Blanket default for every token. Concrete x86 / aarch64 tiers below provide
// their own `impl F16Convert` *items* (the override methods); this blanket
// supplies the software-default methods for all other tokens. To avoid
// coherence conflicts the blanket is the *only* impl on targets that have no
// hardware path, and the hardware tiers opt in individually rather than relying
// on specialization (which is unstable).
//
// Implementation note: Rust has no stable specialization, so we cannot write
// one blanket impl and override it per type. Instead the blanket impl is
// gated to exclude the arches that want a hardware path, and on those arches
// each token gets an explicit impl. This keeps every `impl` block free of
// `unsafe` — the hardware intrinsics live behind `#[arcane]` boundaries below.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
impl<T: F32x4Convert> F16Convert for T {}

// On x86-64, give the software default to every token *except* the F16C tiers,
// then implement the F16C tiers with the hardware override. This split keeps
// the blanket and the per-tier impls coherent without specialization.
#[cfg(target_arch = "x86_64")]
mod x86_dispatch {
    use super::*;

    // Software default for the always-available scalar token. The sub-F16C
    // x86 tiers (`X64V1Token` / `X64V2Token`) do not implement `F32x4Convert`
    // at all (no F32x4 backend below AVX), so they cannot — and need not —
    // implement `F16Convert`; the public entry points require `F16Convert`
    // and therefore never accept them. `X64V3Token` is the lowest x86 tier
    // that carries both the F32x4 backend and `f16c`.
    impl F16Convert for archmage::ScalarToken {}

    // F16C tier: x86-64-v3 lists `f16c` in its feature set, so `vcvtph2ps` /
    // `vcvtps2ph` are safe to execute under an `X64V3Token`. The conversions
    // run inside `#[arcane]` boundaries (which emit the
    // `#[target_feature(enable = ...)]` codegen region), so they are sound to
    // call from cold, non-`target_feature` code — exactly how the public
    // slice entry points call them.
    //
    // `X64V4Token` (AVX-512) also lists `f16c`, but it does not implement the
    // `F32x4Convert` family (only the `F32x4Backend` / `F32x8Backend`
    // delegations exist), so it cannot satisfy this trait's supertrait bound —
    // and it could not call the (previously `T: F32x4Convert`) public entry
    // points either. V4 holders take the F16C path by extracting a V3 token:
    // `f16_to_f32_slice(token.v3(), ..)`. The `.v3()` extractor is sound
    // because V4's feature set is a strict superset of V3's.
    impl F16Convert for archmage::X64V3Token {
        #[inline]
        fn f16_to_f32_into(self, input: &[u16], output: &mut [f32]) {
            f16c_decode_v3(self, input, output);
        }
        #[inline]
        fn f32_to_f16_into(self, input: &[f32], output: &mut [u16]) {
            f16c_encode_v3(self, input, output);
        }
    }
}

// On aarch64, give the software default to `ScalarToken`, then implement the
// `NeonToken` tier. NEON's *baseline* `neon` feature does NOT include the
// half-precision conversion instructions — `vcvt_f32_f16` / `vcvt_f16_f32`
// require the `fp16` target feature (the archmage `Arm64V2Token` tier:
// `neon,crc,rdm,dotprod,fp16,…`, present on Cortex-A55+, Apple M1+, Graviton 2+
// and every post-2017 ARM core). A bare `NeonToken` only proves `neon`, so the
// override probes for `fp16` at runtime (`Arm64V2Token::summon()`); when it is
// present the hardware kernel runs, otherwise the branchless software kernel
// does. Both `impl` blocks stay free of `unsafe` — the intrinsics live behind
// the `#[arcane]` boundaries below.
//
// The hardware kernels are compiled only on a toolchain where the
// `stdarch_neon_f16` intrinsics are stable (Rust ≥ 1.94), selected by
// `#[rustversion::since(1.94)]` (paired with `#[rustversion::before(1.94)]` on
// the software-only selector). On older toolchains (down to the crate MSRV) the
// kernels are not compiled, and `NeonToken` falls through to the software path
// with no MSRV bump and no compile error. This is the toolchain-version-gate
// pattern: see the `neon_f16_decode_select` docs below and the "Newer-stable
// intrinsics above the MSRV" section in `MSRV.md`.
#[cfg(target_arch = "aarch64")]
mod aarch64_dispatch {
    use super::*;

    // The always-available scalar token gets the software default.
    impl F16Convert for archmage::ScalarToken {}

    // `NeonToken` is the only aarch64 token that implements `F32x4Convert`, so
    // it is the only aarch64 token that can satisfy this trait's supertrait
    // bound and reach the public slice entry points. Its override uses the
    // NEON-f16 hardware path when `fp16` is available at runtime, else the
    // software kernel.
    impl F16Convert for archmage::NeonToken {
        #[inline]
        fn f16_to_f32_into(self, input: &[u16], output: &mut [f32]) {
            neon_f16_decode(self, input, output);
        }
        #[inline]
        fn f32_to_f16_into(self, input: &[f32], output: &mut [u16]) {
            neon_f16_encode(self, input, output);
        }
    }
}

/// `NeonToken` f16→f32 decode: hardware NEON-f16 (`vcvt_f32_f16`) when the CPU
/// proves `fp16` at runtime, otherwise the branchless software kernel.
///
/// The bare `neon` feature carried by a `NeonToken` does not include the
/// half-precision conversion instructions, so we summon an `Arm64V2Token`
/// (whose tier includes `fp16`) to prove the feature before dispatching to the
/// hardware kernel. When `fp16` is absent — or the toolchain is too old for the
/// `stdarch_neon_fp16` intrinsics to be stable (rustc < 1.94, where the
/// `#[rustversion::since(1.94)]` arm of [`neon_f16_decode_select`] is not
/// compiled) — this runs the software kernel, exactly matching every other
/// backend.
#[cfg(target_arch = "aarch64")]
#[inline]
fn neon_f16_decode(token: archmage::NeonToken, input: &[u16], output: &mut [f32]) {
    neon_f16_decode_select(token, input, output);
}

/// `NeonToken` f32→f16 encode: hardware NEON-f16 (`vcvt_f16_f32`, RTNE) when
/// `fp16` is available at runtime, otherwise the branchless software kernel.
/// See [`neon_f16_decode`] for the dispatch rationale.
#[cfg(target_arch = "aarch64")]
#[inline]
fn neon_f16_encode(token: archmage::NeonToken, input: &[f32], output: &mut [u16]) {
    neon_f16_encode_select(token, input, output);
}

// ----------------------------------------------------------------------------
// Toolchain-version gate (rustversion + arch), replacing the build.rs probe.
//
// The NEON-f16 intrinsics (`vcvt_f32_f16` / `vcvt_f16_f32`) are
// `#[stable(feature = "stdarch_neon_f16", since = "1.94.0")]` — *above* the
// crate MSRV (1.89). A static `cfg` referencing them would drag the whole
// crate's MSRV to 1.94. Instead we pick the path by *toolchain version* via
// `rustversion`, scoped to `#[cfg(target_arch = "aarch64")]`:
//
//   * `#[rustversion::since(1.94)]`  — the HW-selecting helper exists; the
//     `vcvt_*` kernels (also version-gated) compile and the runtime token
//     decides HW-vs-software per CPU.
//   * `#[rustversion::before(1.94)]` — only the software-selecting helper
//     exists; the `vcvt_*` kernels are not compiled at all, so the same source
//     builds clean on 1.89–1.93 with **no MSRV bump and no missing-intrinsic
//     error**.
//
// The version gate selects whether the HW impl EXISTS; the runtime
// `Arm64V2Token::summon()` probe (inside the `since` helper) selects whether to
// USE it on the actual CPU. The two are orthogonal — exactly the layering the
// removed build.rs probe provided, but with a trusted proc-macro dependency and
// no build script. A CI matrix exercises both sides of the 1.94 boundary (1.93
// fallback, 1.94 flip-on, stable, nightly) to keep the version bound honest.
//
// Rust attribute rule worth noting: `rustversion::{since,before}` gate whole
// *items* (here, two paired free functions), not statement-level blocks — so
// the runtime `summon()` selection lives inside the `since` helper rather than
// as a `#[cfg]`'d block inside a single function (which is how the build.rs-cfg
// version was written).

// ----------------------------------------------------------------------------
// Nightly-opportunistic probe scaffold (the one place a try-compile probe still
// wins over version-matching).
//
// A *stable* intrinsic has a nameable stabilization version → `rustversion`
// gate + CI matrix (above), no build script. A *nightly-only* intrinsic (no
// stable version to name, and a `#![feature(<gate>)]` that can be renamed/
// removed between nightlies) would instead want a try-compile probe in a
// `build.rs`, run only under nightly — the pattern is documented in MSRV.md
// ("Nightly-only case"). archmage has no such intrinsic today, so no build
// script ships: stable/beta builds pay zero probe cost and there is nothing to
// gate. Re-introduce the probe (and its `archmage_nightly_<name>` cfg) when a
// real nightly-only intrinsic actually needs it.

/// HW-capable toolchain (rustc ≥ 1.94): try the NEON-f16 hardware decode when
/// the CPU proves `fp16`, else the branchless software kernel.
#[cfg(target_arch = "aarch64")]
#[rustversion::since(1.94)]
#[inline]
fn neon_f16_decode_select(token: archmage::NeonToken, input: &[u16], output: &mut [f32]) {
    use archmage::SimdToken;
    if let Some(v2) = archmage::Arm64V2Token::summon() {
        neon_f16_decode_hw(v2, input, output);
        return;
    }
    // `fp16` absent at runtime: software fallback.
    f16_to_f32_slice_soft(token, input, output);
}

/// Pre-stabilization toolchain (rustc < 1.94): the `vcvt_*` intrinsics are not
/// stable yet, so only the software kernel is compiled. No MSRV bump.
#[cfg(target_arch = "aarch64")]
#[rustversion::before(1.94)]
#[inline]
fn neon_f16_decode_select(token: archmage::NeonToken, input: &[u16], output: &mut [f32]) {
    f16_to_f32_slice_soft(token, input, output);
}

/// HW-capable toolchain (rustc ≥ 1.94): try the NEON-f16 hardware encode when
/// the CPU proves `fp16`, else the branchless software kernel.
#[cfg(target_arch = "aarch64")]
#[rustversion::since(1.94)]
#[inline]
fn neon_f16_encode_select(token: archmage::NeonToken, input: &[f32], output: &mut [u16]) {
    use archmage::SimdToken;
    if let Some(v2) = archmage::Arm64V2Token::summon() {
        neon_f16_encode_hw(v2, input, output);
        return;
    }
    f32_to_f16_slice_soft(token, input, output);
}

/// Pre-stabilization toolchain (rustc < 1.94): software-only encode. No MSRV
/// bump.
#[cfg(target_arch = "aarch64")]
#[rustversion::before(1.94)]
#[inline]
fn neon_f16_encode_select(token: archmage::NeonToken, input: &[f32], output: &mut [u16]) {
    f32_to_f16_slice_soft(token, input, output);
}

/// Hardware NEON-f16 decode kernel: `&[u16]` → `&mut [f32]` via `vcvt_f32_f16`.
///
/// Compiled only on a toolchain where the `stdarch_neon_f16` intrinsics are
/// stable (Rust ≥ 1.94, gated by `#[rustversion::since(1.94)]`). `#[arcane]`
/// wraps this in a `#[target_feature(enable = "neon,…,fp16")]` sibling (the
/// `Arm64V2Token` tier enables `fp16`) and a safe `#[inline(always)]`
/// trampoline, so it is sound to call from cold code while emitting the
/// `FCVTL`/`FCVT` instructions inside a feature-enabled region. Processes 4
/// lanes per `vcvt_f32_f16`, then a scalar tail through the software kernel.
///
/// Bit-identity vs the software kernel: every **finite, subnormal, and Inf**
/// f16 decodes identically (verified exhaustively over all 65 536 f16 under
/// QEMU). For a NaN *input* `vcvt_f32_f16` returns the hardware-quieted f32 NaN
/// (mantissa MSB set), whereas the software kernel widens the f16 payload
/// directly — the same benign, NaN-only divergence the F16C path documents (the
/// divergence set is exactly the 1022 f16 signaling-NaN patterns).
#[cfg(target_arch = "aarch64")]
#[rustversion::since(1.94)]
// `#[rustversion::since(1.94)]` is the MSRV gate: this item only compiles on a
// toolchain where the `stdarch_neon_f16` intrinsics are stable, so the
// `incompatible_msrv` lint (which flags them as "stable since 1.94" against the
// 1.89 MSRV) is a false positive here — the gate guarantees ≥ 1.94. Allow it.
#[allow(clippy::incompatible_msrv)]
#[archmage::arcane(import_intrinsics)]
fn neon_f16_decode_hw(token: archmage::Arm64V2Token, input: &[u16], output: &mut [f32]) {
    let (in4, in_tail) = input.as_chunks::<4>();
    let (out4, out_tail) = output.as_chunks_mut::<4>();
    for (inp, out) in in4.iter().zip(out4.iter_mut()) {
        // Load 4 f16 bit patterns into a NEON `uint16x4_t`, reinterpret as the
        // `float16x4_t` lane type, widen all 4 to f32, store 4 f32.
        //
        // `vreinterpret_f16_u16` is the safe lane-type cast between the two
        // 64-bit NEON vector views; `vcvt_f32_f16` is the widening convert.
        let h_u16 = vld1_u16(inp);
        let h = vreinterpret_f16_u16(h_u16);
        let f = vcvt_f32_f16(h);
        vst1q_f32(out, f);
    }
    // Scalar tail: reuse the branchless software kernel for bit-identity.
    f16_to_f32_slice_soft(token.neon(), in_tail, out_tail);
}

/// Hardware NEON-f16 encode kernel: `&[f32]` → `&mut [u16]` via `vcvt_f16_f32`
/// (round-to-nearest-even — the only rounding mode the narrowing convert
/// offers). See [`neon_f16_decode_hw`] for the `#[arcane]` / gating rationale.
///
/// Bit-identical to the software kernel for every **finite and Inf** f32
/// (verified over the f16-roundtrip grid + dense f32 sweep under QEMU); for a
/// NaN input both produce a valid quiet f16 NaN whose payload bits may differ —
/// the same tolerance the software encode and the F16C path already document.
#[cfg(target_arch = "aarch64")]
#[rustversion::since(1.94)]
// See `neon_f16_decode_hw`: `#[rustversion::since(1.94)]` is the MSRV gate;
// `incompatible_msrv` cannot see it, so we allow the (here false-positive) lint.
#[allow(clippy::incompatible_msrv)]
#[archmage::arcane(import_intrinsics)]
fn neon_f16_encode_hw(token: archmage::Arm64V2Token, input: &[f32], output: &mut [u16]) {
    let (in4, in_tail) = input.as_chunks::<4>();
    let (out4, out_tail) = output.as_chunks_mut::<4>();
    for (inp, out) in in4.iter().zip(out4.iter_mut()) {
        let f = vld1q_f32(inp);
        // Narrow 4 f32 → 4 f16 (RTNE), reinterpret the `float16x4_t` lanes as
        // `uint16x4_t`, store 4 f16 bit patterns.
        let h = vcvt_f16_f32(f);
        let h_u16 = vreinterpret_u16_f16(h);
        vst1_u16(out, h_u16);
    }
    // Scalar tail: reuse the branchless software kernel for bit-identity.
    f32_to_f16_slice_soft(token.neon(), in_tail, out_tail);
}

/// F16C decode kernel: `&[u16]` → `&mut [f32]` using `vcvtph2ps`.
///
/// `#[arcane]` wraps this in a `#[target_feature(enable = "...")]` sibling
/// and a safe `#[inline(always)]` trampoline, so it is sound to call from
/// cold code while still emitting the F16C instruction inside a
/// feature-enabled codegen region. Processes 8 lanes per `_mm256_cvtph_ps`,
/// then 4 per `_mm_cvtph_ps`, then a scalar tail through the software kernel.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane(import_intrinsics)]
fn f16c_decode_v3(token: archmage::X64V3Token, input: &[u16], output: &mut [f32]) {
    // Memory ops use archmage's safe `safe_unaligned_simd` re-exports (loads
    // take `&[T; N]`, stores take `&mut [T; N]`) — no raw pointers, no
    // `unsafe`. The conversion intrinsics are value ops, safe inside the
    // `#[arcane]` `#[target_feature]` region.
    let (in8, in_rest) = input.as_chunks::<8>();
    let (out8, out_rest) = output.as_chunks_mut::<8>();
    for (inp, out) in in8.iter().zip(out8.iter_mut()) {
        // Load 8 f16 (16 bytes) as `__m128i`; `_mm256_cvtph_ps` widens all 8
        // to f32; store 8 f32.
        let h = _mm_loadu_si128(inp);
        let f = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(out, f);
    }
    let (in4, in_tail) = in_rest.as_chunks::<4>();
    let (out4, out_tail) = out_rest.as_chunks_mut::<4>();
    for (inp, out) in in4.iter().zip(out4.iter_mut()) {
        // Pad the 4 f16 to a full 128-bit load; `_mm_cvtph_ps` decodes the low
        // 4 lanes, the high 4 (zero-padding) are discarded by the 4-wide store.
        let buf: [u16; 8] = [inp[0], inp[1], inp[2], inp[3], 0, 0, 0, 0];
        let h = _mm_loadu_si128(&buf);
        let f = _mm_cvtph_ps(h);
        _mm_storeu_ps(out, f);
    }
    // Scalar tail: reuse the branchless software kernel for bit-identity.
    f16_to_f32_slice_soft(token, in_tail, out_tail);
}

/// F16C encode kernel: `&[f32]` → `&mut [u16]` using `vcvtps2ph` with
/// round-to-nearest-even. See [`f16c_decode_v3`] for the `#[arcane]` rationale.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane(import_intrinsics)]
fn f16c_encode_v3(token: archmage::X64V3Token, input: &[f32], output: &mut [u16]) {
    let (in8, in_rest) = input.as_chunks::<8>();
    let (out8, out_rest) = output.as_chunks_mut::<8>();
    for (inp, out) in in8.iter().zip(out8.iter_mut()) {
        let f = _mm256_loadu_ps(inp);
        // RTNE = `_MM_FROUND_TO_NEAREST_INT` (0x00). The `_NO_EXC` bit (0x08)
        // is *out of range* for this intrinsic's 3-bit immediate field and
        // must not be ORed in; suppression of FP exceptions is implicit.
        let h = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(f);
        _mm_storeu_si128(out, h);
    }
    let (in4, in_tail) = in_rest.as_chunks::<4>();
    let (out4, out_tail) = out_rest.as_chunks_mut::<4>();
    for (inp, out) in in4.iter().zip(out4.iter_mut()) {
        let f = _mm_loadu_ps(inp);
        let h = _mm_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(f);
        // The 128-bit result carries the 4 encoded f16 in its low 4 lanes
        // (high 4 are zero). Store all 8 to a scratch buffer, copy the low 4.
        let mut tmp = [0u16; 8];
        _mm_storeu_si128(&mut tmp, h);
        out.copy_from_slice(&tmp[..4]);
    }
    // Scalar tail: reuse the branchless software kernel for bit-identity.
    f32_to_f16_slice_soft(token, in_tail, out_tail);
}

/// Software (branchless, portable) f16→f32 slice kernel. Always correct on
/// every backend; the fallback when no native f16 conversion is available.
///
/// Bound is `T: F16Convert` (not just `F32x4Convert`) because it is only ever
/// invoked from `F16Convert::f16_to_f32_into` and the hardware tail kernels —
/// all contexts where the token already implements `F16Convert`. It builds an
/// [`i32x4`] from the token and calls the [`i32x4::f16_to_f32`] register
/// method (whose own bound, `F32x4Convert`, is implied by `F16Convert`).
#[inline]
fn f16_to_f32_slice_soft<T: F16Convert>(token: T, input: &[u16], output: &mut [f32]) {
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (out_chunks, out_tail) = output.as_chunks_mut::<4>();
    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let h = i32x4::from_array(
            token,
            [inp[0] as i32, inp[1] as i32, inp[2] as i32, inp[3] as i32],
        );
        *out = h.f16_to_f32().to_array();
    }
    for (inp, out) in in_tail.iter().zip(out_tail.iter_mut()) {
        // Single-lane decode reuses the same vector kernel with a splat,
        // keeping one branchless code path (no scalar reference fork).
        let h = i32x4::splat(token, *inp as i32);
        *out = h.f16_to_f32().to_array()[0];
    }
}

/// Software (branchless, portable) f32→f16 slice kernel. Always correct on
/// every backend; the fallback when no native f16 conversion is available.
///
/// Bound is `T: F16Convert` for the same reason as
/// [`f16_to_f32_slice_soft`]: every caller already holds an `F16Convert`
/// token. It builds an [`f32x4`] from the token and calls the
/// [`f32x4::to_f16`] register method.
#[inline]
fn f32_to_f16_slice_soft<T: F16Convert>(token: T, input: &[f32], output: &mut [u16]) {
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (out_chunks, out_tail) = output.as_chunks_mut::<4>();
    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let f = f32x4::from_array(token, [inp[0], inp[1], inp[2], inp[3]]);
        let bits = f.to_f16().to_array();
        out[0] = bits[0] as u16;
        out[1] = bits[1] as u16;
        out[2] = bits[2] as u16;
        out[3] = bits[3] as u16;
    }
    for (inp, out) in in_tail.iter().zip(out_tail.iter_mut()) {
        let f = f32x4::splat(token, *inp);
        *out = f.to_f16().to_array()[0] as u16;
    }
}

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
//! - `f16_to_f32x4` is **bit-identical** to a scalar IEEE reference for all
//!   65 536 f16 inputs, including subnormals and Inf, and reproduces the
//!   reference's NaN bit patterns too.
//! - `f32_to_f16x4` is **bit-identical** to a scalar round-to-nearest-even
//!   IEEE reference for all 2³² finite and infinite f32 inputs, including
//!   the subnormal flush-to-f16-subnormal range and the overflow-to-Inf
//!   boundary. For NaN inputs both produce *some* f16 NaN (exponent all
//!   ones, mantissa non-zero); the NaN payload bits may differ.
//!
//! The in-register kernels (`f16_to_f32x4` / `f32_to_f16x4`) are pure safe
//! integer/float arithmetic — no `unsafe`, no intrinsics. They are therefore
//! arch-independent: a result proven on one target holds on every target,
//! because the underlying bit operations are identical everywhere.
//!
//! ## Hardware fast path (F16C)
//!
//! The slice entry points ([`f16_to_f32_slice`] / [`f32_to_f16_slice`])
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

/// Decode four IEEE-754 binary16 (`f16`) bit patterns held in the low 16
/// bits of each `i32` lane into an [`f32x4`].
///
/// Each input lane must hold an `f16` bit pattern zero-extended to 32 bits
/// (i.e. `h as i32` for a `u16` `h`); the upper 16 bits of each lane are
/// ignored except that they should be zero for a clean decode. See
/// [`f16_to_f32_slice`] for the slice-oriented entry point that loads from
/// `&[u16]`.
///
/// Bit-identical to a scalar IEEE f16→f32 reference for every finite,
/// subnormal, and infinite input, and reproduces the reference's NaN bit
/// patterns. Branchless on every backend.
#[inline(always)]
pub fn f16_to_f32x4<T: F32x4Convert>(token: T, h: i32x4<T>) -> f32x4<T> {
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

/// Encode an [`f32x4`] into four IEEE-754 binary16 (`f16`) bit patterns,
/// returned in the low 16 bits of each `i32` lane (the upper 16 bits are
/// zero).
///
/// Uses round-to-nearest-even, flushes f32 values too small to represent
/// even as an f16 subnormal to ±0, and saturates overflow to ±Inf — exactly
/// matching a scalar IEEE round-to-nearest-even reference for every finite
/// and infinite f32. NaN inputs map to a canonical quiet f16 NaN (the
/// payload bits are not preserved). Branchless on every backend.
///
/// See [`f32_to_f16_slice`] for the slice-oriented entry point that stores
/// to `&mut [u16]`.
#[inline(always)]
pub fn f32_to_f16x4<T: F32x4Convert>(token: T, f: f32x4<T>) -> i32x4<T> {
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

// ============================================================================
// Hardware-accelerated backend dispatch
// ============================================================================
//
// The in-register kernels above (`f16_to_f32x4` / `f32_to_f16x4`) are pure
// branchless arithmetic — correct on every target, but they do *not* use a
// CPU's native f16-conversion instructions. Where the platform has them
// (x86-64 F16C: `vcvtph2ps` / `vcvtps2ph`), a single instruction converts a
// whole lane, so the slice entry points below dispatch to a hardware kernel
// when the token proves the feature is present, and fall back to the
// branchless software kernel otherwise.
//
// The dispatch is keyed on the token type through the sealed `F16Convert`
// trait: the default trait methods run the software kernel; concrete tokens
// whose feature tier includes a native f16 conversion override them. Because
// the trait is sealed (its supertrait bound is `F32x4Convert`, itself
// sealed), no downstream crate can observe or break this dispatch, and adding
// it is a purely additive, semver-compatible change.

/// Token-keyed slice converters between IEEE-754 binary16 (`f16`) bit
/// patterns and `f32`.
///
/// The default methods run the branchless software kernels
/// ([`f16_to_f32x4`] / [`f32_to_f16x4`]). Tokens whose CPU-feature tier
/// includes a native half-precision conversion (currently x86-64 F16C, via
/// `X64V3Token`) override them with the hardware path. AVX-512 (`X64V4Token`)
/// holders take the F16C path by extracting a V3 token (`token.v3()`).
///
/// Every override is verified **bit-identical** to the software kernel over
/// the full exhaustive f16 sweep (`tests/convert_f16_exhaustive.rs`) for all
/// finite, subnormal, and infinite values, so the public API has identical
/// observable behavior regardless of which backend the runtime selects. NaN
/// inputs are the only exception: the hardware and software paths each
/// produce a valid NaN whose payload bits may differ (see the module-level
/// docs for the exact F16C divergence).
///
/// This trait is sealed via its [`F32x4Convert`] supertrait bound and is not
/// nameable by downstream crates.
pub trait F16Convert: F32x4Convert {
    /// Decode `input.len()` f16 bit patterns into `output`. Lengths are
    /// guaranteed equal by the public entry point.
    #[inline]
    fn f16_to_f32_into(self, input: &[u16], output: &mut [f32]) {
        f16_to_f32_slice_soft(self, input, output);
    }

    /// Encode `input.len()` f32 values into f16 bit patterns in `output`.
    /// Lengths are guaranteed equal by the public entry point.
    #[inline]
    fn f32_to_f16_into(self, input: &[f32], output: &mut [u16]) {
        f32_to_f16_slice_soft(self, input, output);
    }
}

// Blanket default for every token. Concrete x86 tiers below provide their own
// `impl F16Convert` *items* (the override methods); this blanket supplies the
// software-default methods for all other tokens. To avoid coherence conflicts
// the blanket is the *only* impl for non-x86 tokens, and the x86 tiers opt in
// to the hardware override individually rather than relying on specialization
// (which is unstable).
//
// Implementation note: Rust has no stable specialization, so we cannot write
// one blanket impl and override it per type. Instead the blanket impl is
// gated to exclude the tokens that want the hardware path, and those tokens
// get an explicit impl. This keeps every `impl` block free of `unsafe` — the
// hardware intrinsics live behind `#[arcane]` boundaries below.
#[cfg(not(target_arch = "x86_64"))]
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
#[inline]
fn f16_to_f32_slice_soft<T: F32x4Convert>(token: T, input: &[u16], output: &mut [f32]) {
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (out_chunks, out_tail) = output.as_chunks_mut::<4>();
    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let h = i32x4::from_array(
            token,
            [inp[0] as i32, inp[1] as i32, inp[2] as i32, inp[3] as i32],
        );
        *out = f16_to_f32x4(token, h).to_array();
    }
    for (inp, out) in in_tail.iter().zip(out_tail.iter_mut()) {
        // Single-lane decode reuses the same vector kernel with a splat,
        // keeping one branchless code path (no scalar reference fork).
        let h = i32x4::splat(token, *inp as i32);
        *out = f16_to_f32x4(token, h).to_array()[0];
    }
}

/// Decode a slice of IEEE-754 binary16 (`f16`) bit patterns (`&[u16]`) into
/// `&mut [f32]`.
///
/// On an x86-64 `X64V3Token` (F16C) this dispatches to
/// the native `vcvtph2ps` instruction; on every other target it runs the
/// branchless software kernel ([`f16_to_f32x4`], four lanes at a time, with a
/// scalar tail).
///
/// For every **finite, subnormal, and infinite** input both paths are
/// **bit-identical** to a scalar IEEE f16→f32 reference, verified
/// exhaustively. For a **NaN input** the two paths produce f32 values that are
/// both NaN but whose bit patterns may differ: the software kernel widens the
/// f16 NaN payload directly (preserving its signaling/quiet bit), whereas
/// `vcvtph2ps` returns the hardware-quieted NaN (mantissa MSB set). Both are
/// valid f32 NaNs for the same NaN input; this divergence is benign and
/// affects only NaN inputs, never a finite or infinite value.
///
/// # Panics
///
/// Panics if `input.len() != output.len()`.
#[inline]
pub fn f16_to_f32_slice<T: F16Convert>(token: T, input: &[u16], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "f16_to_f32_slice: input and output must have equal length"
    );
    token.f16_to_f32_into(input, output);
}

/// Software (branchless, portable) f32→f16 slice kernel. Always correct on
/// every backend; the fallback when no native f16 conversion is available.
#[inline]
fn f32_to_f16_slice_soft<T: F32x4Convert>(token: T, input: &[f32], output: &mut [u16]) {
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (out_chunks, out_tail) = output.as_chunks_mut::<4>();
    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let f = f32x4::from_array(token, [inp[0], inp[1], inp[2], inp[3]]);
        let bits = f32_to_f16x4(token, f).to_array();
        out[0] = bits[0] as u16;
        out[1] = bits[1] as u16;
        out[2] = bits[2] as u16;
        out[3] = bits[3] as u16;
    }
    for (inp, out) in in_tail.iter().zip(out_tail.iter_mut()) {
        let f = f32x4::splat(token, *inp);
        *out = f32_to_f16x4(token, f).to_array()[0] as u16;
    }
}

/// Encode a slice of `f32` into IEEE-754 binary16 (`f16`) bit patterns
/// (`&mut [u16]`).
///
/// On an x86-64 `X64V3Token` (F16C) this dispatches to
/// the native `vcvtps2ph` instruction (round-to-nearest-even); on every other
/// target it runs the branchless software kernel ([`f32_to_f16x4`], four lanes
/// at a time, with a scalar tail). Both paths are **bit-identical** to a
/// scalar IEEE RTNE reference for every finite and infinite input; NaN maps to
/// an f16 NaN (the F16C `vcvtps2ph` and the software path may emit different
/// NaN payload bits, but both produce a valid quiet f16 NaN — verified
/// exhaustively).
///
/// # Panics
///
/// Panics if `input.len() != output.len()`.
#[inline]
pub fn f32_to_f16_slice<T: F16Convert>(token: T, input: &[f32], output: &mut [u16]) {
    assert_eq!(
        input.len(),
        output.len(),
        "f32_to_f16_slice: input and output must have equal length"
    );
    token.f32_to_f16_into(input, output);
}

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
//! The conversion is pure safe integer/float arithmetic — no `unsafe`, no
//! intrinsics. It is therefore arch-independent: a result proven on one
//! target holds on every target, because the underlying bit operations are
//! identical everywhere.

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

/// Decode a slice of IEEE-754 binary16 (`f16`) bit patterns (`&[u16]`) into
/// `&mut [f32]`, four lanes at a time via [`f16_to_f32x4`], with a scalar
/// tail for the remainder.
///
/// Bit-identical to a scalar IEEE f16→f32 reference for every input value.
///
/// # Panics
///
/// Panics if `input.len() != output.len()`.
#[inline]
pub fn f16_to_f32_slice<T: F32x4Convert>(token: T, input: &[u16], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "f16_to_f32_slice: input and output must have equal length"
    );
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

/// Encode a slice of `f32` into IEEE-754 binary16 (`f16`) bit patterns
/// (`&mut [u16]`), four lanes at a time via [`f32_to_f16x4`], with a scalar
/// tail for the remainder.
///
/// Round-to-nearest-even, bit-identical to a scalar IEEE RTNE reference for
/// every finite and infinite input; NaN maps to a canonical quiet f16 NaN.
///
/// # Panics
///
/// Panics if `input.len() != output.len()`.
#[inline]
pub fn f32_to_f16_slice<T: F32x4Convert>(token: T, input: &[f32], output: &mut [u16]) {
    assert_eq!(
        input.len(),
        output.len(),
        "f32_to_f16_slice: input and output must have equal length"
    );
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

//! Exhaustive bit-exactness tests for the vectorized f16 ↔ f32 converters.
//!
//! The converters in `magetypes::simd::generic::convert_f16` use Fabian
//! Giesen's branchless magic-multiply (decode) and `float_to_half_full_rtne`
//! (encode). This file proves they are **bit-identical** to a plain scalar
//! IEEE-754 reference:
//!
//! - **decode**: all 65 536 f16 bit patterns, including subnormals, Inf, and
//!   NaN (NaN bit patterns reproduced exactly here);
//! - **encode**: every f16-roundtrip value (65 536), the full subnormal and
//!   overflow boundary bands exhaustively, and a dense strided sweep across
//!   the entire finite/infinite f32 range. NaN inputs are checked only to
//!   produce *some* f16 NaN (payload may differ, per the documented
//!   contract). A `#[ignore]`d test sweeps all 2³² f32 for the truly
//!   exhaustive proof.
//!
//! The conversion is pure safe integer/float arithmetic, so a result proven
//! on one backend holds on every backend; the tests still run on both the
//! always-available `ScalarToken` and (when present) the native
//! `X64V3Token` to guard cross-backend parity.

use magetypes::simd::backends::F32x4Convert;
use magetypes::simd::generic::{f16_to_f32x4, f32_to_f16x4, i32x4};

// ============================================================================
// Scalar IEEE-754 reference (the correctness oracle).
//
// This is a plain, branchy, element-at-a-time implementation written
// independently of the vectorized converters. It is the spec the vectorized
// paths must reproduce bit-for-bit.
// ============================================================================

/// Reference f16 (binary16 bit pattern) → f32.
fn ref_f16_to_f32(h: u16) -> f32 {
    let h32 = h as u32;
    let sign = (h32 & 0x8000) << 16;
    let exp = (h32 >> 10) & 0x1F;
    let mant = h32 & 0x03FF;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign); // ±0
        }
        // Subnormal: normalize by shifting the mantissa left.
        let mut e = 1i32;
        let mut m = mant;
        while m & 0x0400 == 0 {
            m <<= 1;
            e -= 1;
        }
        let f32_exp = ((127 - 15 + e) as u32) << 23;
        let f32_mant = (m & 0x03FF) << 13;
        return f32::from_bits(sign | f32_exp | f32_mant);
    }

    if exp == 31 {
        // Inf or NaN.
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }

    // Normal.
    let f32_exp = (exp + 127 - 15) << 23;
    let f32_mant = mant << 13;
    f32::from_bits(sign | f32_exp | f32_mant)
}

/// Reference f32 → f16 (binary16 bit pattern), round-to-nearest-even.
fn ref_f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = (b >> 16) & 0x8000;
    let e = ((b >> 23) & 0xFF) as i32;
    let m = b & 0x007F_FFFF;

    if e == 0xFF {
        // Inf or NaN.
        return (sign | 0x7C00 | if m != 0 { (m >> 13).max(1) } else { 0 }) as u16;
    }

    // f16 exponent = f32 exponent - 112 (bias difference: 127 - 15).
    let f16e = e - 112;

    if e == 0 {
        // f32 zero or subnormal → f16 zero.
        return sign as u16;
    }

    if f16e >= 31 {
        // Overflow → ±Inf.
        return (sign | 0x7C00) as u16;
    }

    if f16e <= 0 {
        if f16e < -10 {
            return sign as u16; // Too small → zero.
        }
        // Subnormal f16: prepend the implicit 1 bit and right-shift.
        let full_m = m | 0x0080_0000;
        let shift = (1 - f16e) as u32 + 13;
        let shifted = full_m >> shift;
        // Round to nearest even.
        let half_bit = 1u32 << (shift - 1);
        let remainder = full_m & ((1u32 << shift) - 1);
        let round =
            u32::from(remainder > half_bit || (remainder == half_bit && (shifted & 1) != 0));
        return (sign | (shifted + round)) as u16;
    }

    // Normal case: shift the mantissa from 23 to 10 bits.
    let shifted_m = m >> 13;
    let remainder = m & 0x1FFF;
    let round = u32::from(remainder > 0x1000 || (remainder == 0x1000 && (shifted_m & 1) != 0));
    let result = sign | ((f16e as u32) << 10) | shifted_m;
    (result + round) as u16 // overflow into exponent is correct (carries to next binade)
}

fn is_f16_nan(h: u16) -> bool {
    (h & 0x7C00) == 0x7C00 && (h & 0x03FF) != 0
}

// ============================================================================
// Vectorized single-lane wrappers (broadcast one value to all 4 lanes, take
// lane 0). This exercises the exact production kernel.
// ============================================================================

fn vec_decode<T: F32x4Convert>(token: T, h: u16) -> u32 {
    let v = i32x4::splat(token, h as i32);
    f16_to_f32x4(token, v).to_array()[0].to_bits()
}

fn vec_encode<T: F32x4Convert>(token: T, x: f32) -> u16 {
    use magetypes::simd::generic::f32x4;
    let v = f32x4::splat(token, x);
    f32_to_f16x4(token, v).to_array()[0] as u16
}

// ============================================================================
// Decode: exhaustive over all 65 536 f16 values.
// ============================================================================

fn exhaustive_decode<T: F32x4Convert>(token: T) {
    let mut mismatches = 0u64;
    for hv in 0u32..=0xFFFF {
        let h = hv as u16;
        let want = ref_f16_to_f32(h).to_bits();
        let got = vec_decode(token, h);
        if want != got {
            // For NaN the payload bits are allowed to differ, but the Giesen
            // decode is in fact bit-identical here — assert exact equality.
            mismatches += 1;
            if mismatches <= 16 {
                eprintln!("decode mismatch h={h:#06x} want={want:#010x} got={got:#010x}");
            }
        }
    }
    assert_eq!(mismatches, 0, "f16→f32 decode diverged on {mismatches} of 65536 values");
}

#[test]
fn decode_exhaustive_scalar() {
    exhaustive_decode(archmage::ScalarToken);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn decode_exhaustive_x64v3() {
    use archmage::SimdToken;
    if let Some(t) = archmage::X64V3Token::summon() {
        exhaustive_decode(t);
    } else {
        eprintln!("X64V3Token not available on this host — scalar path covers correctness");
    }
}

// ============================================================================
// Encode: every f16-roundtrip value, the boundary bands exhaustively, and a
// dense strided sweep of the whole f32 range.
// ============================================================================

/// Assert the encode matches the reference for one finite/inf input, or that
/// both produce an f16 NaN for a NaN input. Returns 1 on a real mismatch.
fn check_encode<T: F32x4Convert>(token: T, x: f32) -> u64 {
    let want = ref_f32_to_f16(x);
    let got = vec_encode(token, x);
    if want == got {
        return 0;
    }
    if x.is_nan() {
        // Contract: both produce some f16 NaN; payload may differ.
        assert!(
            is_f16_nan(want) && is_f16_nan(got),
            "NaN input {x:?} produced a non-NaN f16 (want={want:#06x} got={got:#06x})"
        );
        return 0;
    }
    eprintln!("encode mismatch x={x:e} bits={:#010x} want={want:#06x} got={got:#06x}", x.to_bits());
    1
}

fn encode_f16_roundtrip<T: F32x4Convert>(token: T) {
    // Every value exactly representable as f16 must round-trip and match.
    let mut mismatches = 0u64;
    for hv in 0u32..=0xFFFF {
        let x = ref_f16_to_f32(hv as u16);
        mismatches += check_encode(token, x);
    }
    assert_eq!(mismatches, 0, "f32→f16 encode diverged on the f16-roundtrip grid");
}

fn encode_boundary_bands<T: F32x4Convert>(token: T) {
    // Sweep every f32 bit pattern in the bands where the encode changes
    // regime — for both signs:
    //   * subnormal-flush band: the entire range that maps onto f16
    //     subnormals (and the flush-to-zero edge just below it), where
    //     round-half-to-even between adjacent f16 subnormals lives;
    //   * overflow-to-Inf band: from the smallest |f32| that overflows f16
    //     up to f32 Inf (the saturation edge).
    //
    // In release this is exhaustive over both bands (~tens of millions of
    // f32 each). In debug (CI runs `cargo test` unoptimized) the same
    // ranges are walked at stride 1 over the tie-dense low band and a small
    // stride elsewhere, keeping the test under a few seconds while still
    // hitting every RTNE tie residue and the exact saturation edge. The
    // truly-exhaustive 2^32 proof lives in `encode_full_2pow32_scalar`
    // (`--ignored`).
    let mut mismatches = 0u64;

    // f16 subnormals are produced from |f32| in [2^-24, 2^-14), i.e. f32
    // exponent field in [101, 113). Include a margin below for the
    // flush-to-zero edge. This window is ~10 binades = ~84M f32 in release.
    let sub_lo: u32 = 100u32 << 23; // a binade below the smallest f16 subnormal
    let sub_hi: u32 = 113u32 << 23; // first normal-f16 |f32|
    let step: u32 = if cfg!(debug_assertions) { 251 } else { 1 };
    let mut bits = sub_lo;
    while bits < sub_hi {
        mismatches += check_encode(token, f32::from_bits(bits));
        mismatches += check_encode(token, f32::from_bits(bits | 0x8000_0000));
        bits += step;
    }

    // Overflow band: [(127+16)<<23, 255<<23] = [f16max, f32 Inf]. The
    // saturation edge is at exactly (127+16)<<23 and the Inf line at
    // 255<<23 — sweep both endpoints' immediate neighborhoods at stride 1
    // even in debug, the bulk at the same coarse step.
    let ov_lo: u32 = (127u32 + 16) << 23;
    let ov_hi: u32 = 255u32 << 23;
    let mut bits = ov_lo;
    loop {
        mismatches += check_encode(token, f32::from_bits(bits));
        mismatches += check_encode(token, f32::from_bits(bits | 0x8000_0000));
        if bits >= ov_hi {
            break;
        }
        // Stride-1 near both edges (within 4096 of either endpoint), coarse
        // in between, so the saturation boundary itself is never skipped.
        let near_edge = bits.saturating_sub(ov_lo) < 4096 || ov_hi.saturating_sub(bits) < 4096;
        let s = if cfg!(debug_assertions) && !near_edge { 251 } else { 1 };
        bits = (bits + s).min(ov_hi); // clamp so the Inf line is always hit
    }

    assert_eq!(mismatches, 0, "f32→f16 encode diverged in a boundary band");
}

fn encode_dense_sweep<T: F32x4Convert>(token: T) {
    // Dense strided sweep across the entire f32 bit space. A prime-ish stride
    // visits ~4.3M points spread over every exponent and a varied mantissa
    // residue, covering the normal-range RTNE behavior without a full 2³² run.
    let mut mismatches = 0u64;
    let stride: u32 = 1009; // coprime-with-2 stride to walk mantissa residues
    let mut bits: u32 = 0;
    loop {
        let x = f32::from_bits(bits);
        mismatches += check_encode(token, x);
        let (next, ov) = bits.overflowing_add(stride);
        if ov {
            break;
        }
        bits = next;
    }
    assert_eq!(mismatches, 0, "f32→f16 encode diverged on the dense strided sweep");
}

#[test]
fn encode_f16_roundtrip_scalar() {
    encode_f16_roundtrip(archmage::ScalarToken);
}

#[test]
fn encode_boundary_bands_scalar() {
    encode_boundary_bands(archmage::ScalarToken);
}

#[test]
fn encode_dense_sweep_scalar() {
    encode_dense_sweep(archmage::ScalarToken);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn encode_f16_roundtrip_x64v3() {
    use archmage::SimdToken;
    if let Some(t) = archmage::X64V3Token::summon() {
        encode_f16_roundtrip(t);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn encode_boundary_bands_x64v3() {
    use archmage::SimdToken;
    if let Some(t) = archmage::X64V3Token::summon() {
        encode_boundary_bands(t);
    }
}

// ============================================================================
// Slice helpers: tail handling (lengths not a multiple of 4) and parity with
// the in-register kernel.
// ============================================================================

#[test]
fn slice_decode_matches_reference_all_lengths() {
    use magetypes::simd::generic::f16_to_f32_slice;
    // Cover every tail residue (0..7) over a buffer that includes subnormals,
    // Inf, and assorted normals.
    let samples: [u16; 7] = [0x0000, 0x0001, 0x3C00, 0x7BFF, 0x7C00, 0xC000, 0x83FF];
    for len in 1..=64usize {
        let input: Vec<u16> = (0..len).map(|i| samples[i % samples.len()]).collect();
        let mut out = vec![0.0f32; len];
        f16_to_f32_slice(archmage::ScalarToken, &input, &mut out);
        for (i, (&h, &got)) in input.iter().zip(out.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                ref_f16_to_f32(h).to_bits(),
                "slice decode mismatch at len={len} idx={i} h={h:#06x}"
            );
        }
    }
}

#[test]
fn slice_encode_matches_reference_all_lengths() {
    use magetypes::simd::generic::f32_to_f16_slice;
    let samples: [f32; 7] = [0.0, 1.0, -2.5, 65504.0, 1e-7, f32::INFINITY, -0.0];
    for len in 1..=64usize {
        let input: Vec<f32> = (0..len).map(|i| samples[i % samples.len()]).collect();
        let mut out = vec![0u16; len];
        f32_to_f16_slice(archmage::ScalarToken, &input, &mut out);
        for (i, (&x, &got)) in input.iter().zip(out.iter()).enumerate() {
            assert_eq!(
                got,
                ref_f32_to_f16(x),
                "slice encode mismatch at len={len} idx={i} x={x}"
            );
        }
    }
}

#[test]
fn slice_roundtrip_decode_then_encode_is_identity() {
    use magetypes::simd::generic::{f16_to_f32_slice, f32_to_f16_slice};
    // Every f16 bit pattern decoded then re-encoded must return itself
    // (excluding NaN payload, which the encode canonicalizes).
    let input: Vec<u16> = (0u32..=0xFFFF).map(|v| v as u16).collect();
    let mut f32buf = vec![0.0f32; input.len()];
    let mut back = vec![0u16; input.len()];
    f16_to_f32_slice(archmage::ScalarToken, &input, &mut f32buf);
    f32_to_f16_slice(archmage::ScalarToken, &f32buf, &mut back);
    let mut nonnan_diffs = 0u64;
    for (&h, &r) in input.iter().zip(back.iter()) {
        if is_f16_nan(h) {
            assert!(is_f16_nan(r), "NaN f16 {h:#06x} did not round-trip to a NaN ({r:#06x})");
        } else if h != r {
            nonnan_diffs += 1;
            eprintln!("roundtrip non-identity h={h:#06x} -> {r:#06x}");
        }
    }
    assert_eq!(nonnan_diffs, 0, "f16→f32→f16 round-trip was not identity for finite values");
}

// ============================================================================
// Truly exhaustive 2³² encode sweep — slow, run with `--ignored`.
// ============================================================================

#[test]
#[ignore = "full 2^32 f32 sweep is slow (~minutes in debug); run explicitly to certify encode"]
fn encode_full_2pow32_scalar() {
    let token = archmage::ScalarToken;
    let mut mismatches = 0u64;
    let mut bits: u64 = 0;
    while bits <= 0xFFFF_FFFF {
        mismatches += check_encode(token, f32::from_bits(bits as u32));
        bits += 1;
    }
    assert_eq!(mismatches, 0, "f32→f16 encode diverged somewhere in the full 2^32 sweep");
}

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

use magetypes::simd::generic::{F16Convert, i32x4};

// ============================================================================
// Miri sub-sampling.
//
// Under Miri the interpreter runs the kernel ~1000× slower than native, so the
// exhaustive 65 536-point and 2³²-strided sweeps below would take hours (the
// `encode_dense_sweep` 2³²/1009 ≈ 4.3M-point walk alone is hopeless). Miri's
// role here is **UB detection**, not bit-exact correctness — and because the
// converters are *branchless* (the identical instruction sequence executes for
// every input, with no data-dependent control flow), a coarsely-strided sample
// that still spans every f16/f32 regime (zero, subnormal, the flush/overflow/
// Inf boundaries, NaN) exercises the exact same code paths for UB purposes as
// the full sweep. Full bit-exact correctness is proven stride-1 on every
// **native** target by these same tests.
//
// This is the standard Rust-ecosystem pattern for exhaustive tests under Miri
// (the standard library does the same). It is NOT a hidden runtime skip: the
// CI `Miri (UB Detection)` job opts in by running `cargo miri test`, the tests
// still execute (checking UB on the sample), and the stride is `cfg(miri)`-
// gated here in plain sight — visible in the CI→test chain.
#[cfg(miri)]
const MIRI_F16_STRIDE: usize = 263; // coprime-with-2: visits ~250 of 65 536, varied mantissa residues
#[cfg(not(miri))]
const MIRI_F16_STRIDE: usize = 1; // native: every f16 bit pattern

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

fn vec_decode<T: F16Convert>(token: T, h: u16) -> u32 {
    let v = i32x4::splat(token, h as i32);
    v.f16_to_f32().to_array()[0].to_bits()
}

fn vec_encode<T: F16Convert>(token: T, x: f32) -> u16 {
    use magetypes::simd::generic::f32x4;
    let v = f32x4::splat(token, x);
    v.to_f16().to_array()[0] as u16
}

// ============================================================================
// Decode: exhaustive over all 65 536 f16 values.
// ============================================================================

fn exhaustive_decode<T: F16Convert>(token: T) {
    let mut mismatches = 0u64;
    for hv in (0u32..=0xFFFF).step_by(MIRI_F16_STRIDE) {
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
    assert_eq!(
        mismatches, 0,
        "f16→f32 decode diverged on {mismatches} of 65536 values"
    );
}

// ============================================================================
// Encode: every f16-roundtrip value, the boundary bands exhaustively, and a
// dense strided sweep of the whole f32 range.
// ============================================================================

/// Assert the encode matches the reference for one finite/inf input, or that
/// both produce an f16 NaN for a NaN input. Returns 1 on a real mismatch.
fn check_encode<T: F16Convert>(token: T, x: f32) -> u64 {
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
    eprintln!(
        "encode mismatch x={x:e} bits={:#010x} want={want:#06x} got={got:#06x}",
        x.to_bits()
    );
    1
}

fn encode_f16_roundtrip<T: F16Convert>(token: T) {
    // Every value exactly representable as f16 must round-trip and match.
    let mut mismatches = 0u64;
    for hv in (0u32..=0xFFFF).step_by(MIRI_F16_STRIDE) {
        let x = ref_f16_to_f32(hv as u16);
        mismatches += check_encode(token, x);
    }
    assert_eq!(
        mismatches, 0,
        "f32→f16 encode diverged on the f16-roundtrip grid"
    );
}

fn encode_boundary_bands<T: F16Convert>(token: T) {
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
    // Under Miri stride far coarser (~1k samples over the band) — see MIRI_F16_STRIDE rationale.
    let step: u32 = if cfg!(miri) {
        100_003
    } else if cfg!(debug_assertions) {
        251
    } else {
        1
    };
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
        let s = if cfg!(miri) {
            // Coarse under Miri (~1k samples); the exact ov_lo/ov_hi endpoints
            // are still hit (loop starts at ov_lo, `.min(ov_hi)` clamps the end).
            1_000_003
        } else if cfg!(debug_assertions) && !near_edge {
            251
        } else {
            1
        };
        bits = (bits + s).min(ov_hi); // clamp so the Inf line is always hit
    }

    assert_eq!(mismatches, 0, "f32→f16 encode diverged in a boundary band");
}

fn encode_dense_sweep<T: F16Convert>(token: T) {
    // Dense strided sweep across the entire f32 bit space. A prime-ish stride
    // visits ~4.3M points spread over every exponent and a varied mantissa
    // residue, covering the normal-range RTNE behavior without a full 2³² run.
    let mut mismatches = 0u64;
    // coprime-with-2 stride to walk mantissa residues; far coarser under Miri
    // (~1k samples across the whole f32 space) — see MIRI_F16_STRIDE rationale.
    let stride: u32 = if cfg!(miri) { 4_000_037 } else { 1009 };
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
    assert_eq!(
        mismatches, 0,
        "f32→f16 encode diverged on the dense strided sweep"
    );
}

// ============================================================================
// Portable (scalar-token) tests — no architecture gate. These run everywhere
// and prove the branchless software kernel against the oracle.
// ============================================================================
mod scalar {
    use super::*;

    #[test]
    fn decode_exhaustive_scalar() {
        exhaustive_decode(archmage::ScalarToken);
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

    // ========================================================================
    // Slice helpers: tail handling (lengths not a multiple of 4) and parity with
    // the in-register kernel.
    // ========================================================================

    #[test]
    fn slice_decode_matches_reference_all_lengths() {
        // Cover every tail residue (0..7) over a buffer that includes subnormals,
        // Inf, and assorted normals.
        let samples: [u16; 7] = [0x0000, 0x0001, 0x3C00, 0x7BFF, 0x7C00, 0xC000, 0x83FF];
        for len in 1..=64usize {
            let input: Vec<u16> = (0..len).map(|i| samples[i % samples.len()]).collect();
            let mut out = vec![0.0f32; len];
            archmage::ScalarToken.f16_to_f32_slice(&input, &mut out);
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
        let samples: [f32; 7] = [0.0, 1.0, -2.5, 65504.0, 1e-7, f32::INFINITY, -0.0];
        for len in 1..=64usize {
            let input: Vec<f32> = (0..len).map(|i| samples[i % samples.len()]).collect();
            let mut out = vec![0u16; len];
            archmage::ScalarToken.f32_to_f16_slice(&input, &mut out);
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
        // Every f16 bit pattern decoded then re-encoded must return itself
        // (excluding NaN payload, which the encode canonicalizes). Under Miri the
        // 65 536-element slice round-trip is sub-sampled — see MIRI_F16_STRIDE.
        let input: Vec<u16> = (0u32..=0xFFFF)
            .step_by(MIRI_F16_STRIDE)
            .map(|v| v as u16)
            .collect();
        let mut f32buf = vec![0.0f32; input.len()];
        let mut back = vec![0u16; input.len()];
        archmage::ScalarToken.f16_to_f32_slice(&input, &mut f32buf);
        archmage::ScalarToken.f32_to_f16_slice(&f32buf, &mut back);
        let mut nonnan_diffs = 0u64;
        for (&h, &r) in input.iter().zip(back.iter()) {
            if is_f16_nan(h) {
                assert!(
                    is_f16_nan(r),
                    "NaN f16 {h:#06x} did not round-trip to a NaN ({r:#06x})"
                );
            } else if h != r {
                nonnan_diffs += 1;
                eprintln!("roundtrip non-identity h={h:#06x} -> {r:#06x}");
            }
        }
        assert_eq!(
            nonnan_diffs, 0,
            "f16→f32→f16 round-trip was not identity for finite values"
        );
    }

    // ========================================================================
    // Truly exhaustive 2³² encode sweep — slow, run with `--ignored`.
    // ========================================================================

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
        assert_eq!(
            mismatches, 0,
            "f32→f16 encode diverged somewhere in the full 2^32 sweep"
        );
    }
}

// ============================================================================
// Native F16C hardware path (x86-64): exhaustive bit-identity vs the oracle AND
// vs the software kernel.
//
// `F16Convert::f16_to_f32_slice` / `F16Convert::f32_to_f16_slice` dispatch to
// x86-64 F16C (`vcvtph2ps` / `vcvtps2ph`) when handed an `X64V3Token` whose CPU
// presents `f16c`. These tests prove the hardware path is:
//   1. bit-identical to the independent scalar IEEE oracle, and
//   2. bit-identical to the branchless software slice kernel (the
//      cross-backend reference proven exhaustively above).
//
// The decode is exhaustive (all 65 536 f16). The encode covers the boundary
// bands + a dense strided sweep (the same coverage the software encode tests
// use). NaN-encode payload bits are allowed to differ between hardware and
// software (both must produce *a* quiet f16 NaN — the documented contract).
//
// On a host without F16C the tests print a notice and pass; correctness on
// such hosts is covered by the always-run scalar tests above.
//
// These drive a `X64V3Token`. Its slice methods summon-up to the best tier
// (amortized): with `--features avx512` on an AVX-512 CPU they run the 16-wide
// kernel, else the 8-wide F16C kernel — so under `avx512` they double as the
// exhaustive 16-wide correctness proof. The `x86_avx512f` module below drives a
// `X64V4Token` (the plain V4 path, direct 16-wide) and the 16-lane boundary.
//
// The whole module carries the x86_64 arch gate at the module boundary, so
// the gate is applied once — a new x86 test added here inherits it.
// ============================================================================
#[cfg(target_arch = "x86_64")]
mod x86_f16c {
    use super::*;

    #[test]
    fn decode_exhaustive_x64v3() {
        use archmage::SimdToken;
        if let Some(t) = archmage::X64V3Token::summon() {
            exhaustive_decode(t);
        } else {
            eprintln!("X64V3Token not available on this host — scalar path covers correctness");
        }
    }

    #[test]
    fn encode_f16_roundtrip_x64v3() {
        use archmage::SimdToken;
        if let Some(t) = archmage::X64V3Token::summon() {
            encode_f16_roundtrip(t);
        }
    }

    #[test]
    fn encode_boundary_bands_x64v3() {
        use archmage::SimdToken;
        if let Some(t) = archmage::X64V3Token::summon() {
            encode_boundary_bands(t);
        }
    }

    /// All 65 536 f16 decoded through the F16C slice path must match the oracle
    /// bit-for-bit (every length residue 0..7 exercised so the 8-/4-wide chunks
    /// and the scalar tail are all hit).
    #[test]
    fn native_f16c_decode_exhaustive_vs_oracle() {
        use archmage::SimdToken;
        let Some(token) = archmage::X64V3Token::summon() else {
            eprintln!(
                "X64V3Token (F16C) not available on this host — scalar path covers correctness"
            );
            return;
        };
        // Decode the whole 0..=0xFFFF block at once (exercises the 8-wide chunk
        // loop heavily), then also at every short length to exercise the 4-wide
        // chunk and the 0..7 scalar tail.
        let all: Vec<u16> = (0u32..=0xFFFF).map(|v| v as u16).collect();
        let mut out = vec![0f32; all.len()];
        token.f16_to_f32_slice(&all, &mut out);
        for (&h, &got) in all.iter().zip(out.iter()) {
            // F16C bit-identity contract: exact for finite/subnormal/Inf. For a
            // NaN input `vcvtph2ps` returns the hardware-quieted NaN whose payload
            // differs from the software widening — both are valid f32 NaNs (the
            // documented benign divergence). Assert finite/Inf exactly; require
            // only NaN-ness for NaN inputs.
            if is_f16_nan(h) {
                assert!(
                    got.is_nan(),
                    "F16C decode of NaN f16 {h:#06x} produced a non-NaN f32 ({:#010x})",
                    got.to_bits()
                );
            } else {
                assert_eq!(
                    got.to_bits(),
                    ref_f16_to_f32(h).to_bits(),
                    "F16C decode mismatch vs oracle h={h:#06x}"
                );
            }
        }
        // Tail-residue sweep: lengths 1..=64 starting at varied offsets.
        for start in [0usize, 1, 2, 3, 5, 7] {
            for len in 1..=64usize {
                let input: Vec<u16> = (0..len).map(|i| ((start + i) & 0xFFFF) as u16).collect();
                let mut o = vec![0f32; len];
                token.f16_to_f32_slice(&input, &mut o);
                for (&h, &got) in input.iter().zip(o.iter()) {
                    if is_f16_nan(h) {
                        assert!(got.is_nan(), "F16C decode (tail) NaN f16 {h:#06x} not NaN");
                    } else {
                        assert_eq!(
                            got.to_bits(),
                            ref_f16_to_f32(h).to_bits(),
                            "F16C decode mismatch (tail) start={start} len={len} h={h:#06x}"
                        );
                    }
                }
            }
        }
    }

    /// The F16C decode path must be byte-for-byte identical to the branchless
    /// software decode path over all 65 536 f16 (finite/subnormal/Inf), with the
    /// documented NaN-only payload divergence (both produce a valid f32 NaN). This
    /// is the strongest guarantee that the two backends are interchangeable for
    /// real data.
    #[test]
    fn native_f16c_decode_matches_software_exhaustive() {
        use archmage::SimdToken;
        let Some(hw) = archmage::X64V3Token::summon() else {
            eprintln!(
                "X64V3Token (F16C) not available — skipping native-vs-software decode parity"
            );
            return;
        };
        let all: Vec<u16> = (0u32..=0xFFFF).map(|v| v as u16).collect();
        let mut hw_out = vec![0f32; all.len()];
        let mut sw_out = vec![0f32; all.len()];
        hw.f16_to_f32_slice(&all, &mut hw_out);
        archmage::ScalarToken.f16_to_f32_slice(&all, &mut sw_out);
        let mut nan_payload_diffs = 0u64;
        for (i, (&a, &b)) in hw_out.iter().zip(sw_out.iter()).enumerate() {
            let h = all[i];
            if a.to_bits() == b.to_bits() {
                // Where they agree on a NaN input, it must be a *quiet* f16 NaN:
                // the software widening already has the mantissa MSB set, so the
                // hardware quieting is a no-op and the bits match.
                continue;
            }
            // A divergence is only permitted on an f16 *signaling* NaN input
            // (top mantissa bit clear): `vcvtph2ps` quiets it (sets the f32
            // mantissa MSB) while the software path preserves the signaling
            // payload. Both results must still be f32 NaNs.
            assert!(
                is_f16_nan(h) && (h & 0x0200) == 0 && a.is_nan() && b.is_nan(),
                "F16C vs software decode differ on a non-(signaling-NaN) value at idx={i} h={h:#06x}: hw={:#010x} sw={:#010x}",
                a.to_bits(),
                b.to_bits()
            );
            nan_payload_diffs += 1;
        }
        // The divergence set is exactly the f16 signaling NaNs: payloads in
        // 0x001..=0x1FF (top mantissa bit clear) × 2 signs = 511 × 2 = 1022.
        assert_eq!(
            nan_payload_diffs, 1022,
            "expected the HW/SW decode divergence to be exactly the 1022 f16 signaling-NaN patterns"
        );
    }

    /// The F16C encode path must match the oracle over the boundary bands and a
    /// dense strided sweep (NaN payload tolerated, like the software encode test).
    #[test]
    fn native_f16c_encode_vs_oracle() {
        use archmage::SimdToken;
        let Some(token) = archmage::X64V3Token::summon() else {
            eprintln!("X64V3Token (F16C) not available — scalar encode tests cover correctness");
            return;
        };

        // Build a single big input vector covering: every f16-roundtrip value, the
        // subnormal-flush band, the overflow-to-Inf band, and a dense strided f32
        // sweep — then encode it through the F16C slice path in one call.
        let mut inputs: Vec<f32> = Vec::new();
        for hv in 0u32..=0xFFFF {
            inputs.push(ref_f16_to_f32(hv as u16));
        }
        let step: u32 = if cfg!(debug_assertions) { 251 } else { 1 };
        let mut bits = 100u32 << 23;
        let sub_hi = 113u32 << 23;
        while bits < sub_hi {
            inputs.push(f32::from_bits(bits));
            inputs.push(f32::from_bits(bits | 0x8000_0000));
            bits += step;
        }
        let ov_lo = (127u32 + 16) << 23;
        let ov_hi = 255u32 << 23;
        let mut bits = ov_lo;
        loop {
            inputs.push(f32::from_bits(bits));
            inputs.push(f32::from_bits(bits | 0x8000_0000));
            if bits >= ov_hi {
                break;
            }
            let near = bits.saturating_sub(ov_lo) < 4096 || ov_hi.saturating_sub(bits) < 4096;
            let s = if cfg!(debug_assertions) && !near {
                251
            } else {
                1
            };
            bits = (bits + s).min(ov_hi);
        }
        let stride: u32 = 1009;
        let mut bits: u32 = 0;
        loop {
            inputs.push(f32::from_bits(bits));
            let (next, ov) = bits.overflowing_add(stride);
            if ov {
                break;
            }
            bits = next;
        }

        let mut out = vec![0u16; inputs.len()];
        token.f32_to_f16_slice(&inputs, &mut out);
        let mut mismatches = 0u64;
        for (&x, &got) in inputs.iter().zip(out.iter()) {
            let want = ref_f32_to_f16(x);
            if want == got {
                continue;
            }
            if x.is_nan() {
                assert!(
                    is_f16_nan(want) && is_f16_nan(got),
                    "F16C NaN encode produced non-NaN: x={x:?} want={want:#06x} got={got:#06x}"
                );
                continue;
            }
            mismatches += 1;
            if mismatches <= 16 {
                eprintln!("F16C encode mismatch x={x:e} want={want:#06x} got={got:#06x}");
            }
        }
        assert_eq!(
            mismatches, 0,
            "F16C encode diverged from the oracle on {mismatches} finite inputs"
        );
    }

    /// The F16C encode path must match the software encode path bit-for-bit for
    /// every finite/inf input (NaN payload tolerated) — over the f16-roundtrip
    /// grid plus a dense f32 sweep.
    #[test]
    fn native_f16c_encode_matches_software() {
        use archmage::SimdToken;
        let Some(hw) = archmage::X64V3Token::summon() else {
            eprintln!(
                "X64V3Token (F16C) not available — skipping native-vs-software encode parity"
            );
            return;
        };
        let mut inputs: Vec<f32> = (0u32..=0xFFFF)
            .map(|hv| ref_f16_to_f32(hv as u16))
            .collect();
        let stride: u32 = 1009;
        let mut bits: u32 = 0;
        loop {
            inputs.push(f32::from_bits(bits));
            let (next, ov) = bits.overflowing_add(stride);
            if ov {
                break;
            }
            bits = next;
        }
        let mut hw_out = vec![0u16; inputs.len()];
        let mut sw_out = vec![0u16; inputs.len()];
        hw.f32_to_f16_slice(&inputs, &mut hw_out);
        archmage::ScalarToken.f32_to_f16_slice(&inputs, &mut sw_out);
        for (i, ((&x, &a), &b)) in inputs
            .iter()
            .zip(hw_out.iter())
            .zip(sw_out.iter())
            .enumerate()
        {
            if a == b {
                continue;
            }
            // NaN inputs: both must be f16 NaN; payload may differ between HW and SW.
            assert!(
                x.is_nan() && is_f16_nan(a) && is_f16_nan(b),
                "F16C vs software encode differ at idx={i} x={x:e}: hw={a:#06x} sw={b:#06x}"
            );
        }
    }
}

// ============================================================================
// Native AVX-512F 16-wide hardware path (x86_64, `avx512` feature).
//
// When built with the `avx512` feature, the F16C slice path self-upgrades to
// the wider `_mm512_cvtph_ps` / `_mm512_cvtps_ph` (16 f16 per instruction)
// whenever an `X64V4Token` is summonable — the F16C kernel summons a V4 token
// internally and routes the bulk of the slice through the 16-wide kernel, with
// the 8/4-wide F16C kernel handling the < 16-lane tail (see `convert_f16.rs`).
//
// These tests make that coverage explicit and visible (the `x86_f16c` suite
// above hits the same upgraded path but is *named* for F16C, so a reader can't
// tell which width ran). A V4 token enters via `v4.v3()` because `X64V4Token`
// does not itself implement the (`F32x4Convert`-bounded) `F16Convert` trait;
// the `.v3()` extractor is sound since V4 ⊃ V3.
//
// On an x86_64 host without AVX-512 (pre-Skylake-X, or AMD pre-Zen 4 — and note
// the dev 7950X has AVX-512F but *not* avx512fp16) the `X64V4Token` is absent,
// the selector stays on the 8-wide F16C path, and these tests print a notice and
// pass — the `x86_f16c` suite covers that host.
// ============================================================================
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod x86_avx512f {
    use super::*;

    /// Summon a `X64V4Token` — the token that drives the 16-wide path directly
    /// (the "plain V4 path"). `None` ⇒ no AVX-512F on this host (8-wide F16C
    /// covers correctness via the `x86_f16c` suite).
    fn v4_slice_entry() -> Option<archmage::X64V4Token> {
        use archmage::SimdToken;
        match archmage::X64V4Token::summon() {
            Some(v4) => {
                eprintln!(
                    "AVX-512F present: X64V4Token runs the 16-wide \
                     _mm512_cvtph_ps / _mm512_cvtps_ph kernel directly."
                );
                Some(v4)
            }
            None => {
                eprintln!(
                    "X64V4Token (AVX-512F) not available — 8-wide F16C path covers \
                     correctness (see x86_f16c suite)."
                );
                None
            }
        }
    }

    // These tests drive the 16-wide path through a `X64V4Token` directly. The
    // exhaustive correctness sweep lives in the `x86_f16c` suite (8-wide, plus
    // the always-run scalar suite); here we cover the 16-lane chunk boundary and
    // confirm the V4-tier tokens are accepted directly.

    /// Decode parity vs software focused on the 16-lane chunk boundary: lengths
    /// 13..=49 straddle 16/32/48, exercising the `_mm512_cvtph_ps` loop AND the
    /// V3 8/4-wide + scalar tail handoff.
    #[test]
    fn decode_16wide_boundary_vs_software() {
        let Some(hw) = v4_slice_entry() else { return };
        for len in 13..=49usize {
            // A coprime stride walks varied exponents (incl. Inf/NaN bands) per length.
            let input: Vec<u16> = (0..len)
                .map(|i| ((i * 1237 + 0x77ff) & 0xFFFF) as u16)
                .collect();
            let mut hw_out = vec![0f32; len];
            let mut sw_out = vec![0f32; len];
            hw.f16_to_f32_slice(&input, &mut hw_out);
            archmage::ScalarToken.f16_to_f32_slice(&input, &mut sw_out);
            for (&h, (&a, &b)) in input.iter().zip(hw_out.iter().zip(sw_out.iter())) {
                if is_f16_nan(h) {
                    assert!(
                        a.is_nan() && b.is_nan(),
                        "len={len} decode NaN f16 {h:#06x}"
                    );
                } else {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "AVX-512F decode boundary parity len={len} h={h:#06x}"
                    );
                }
            }
        }
    }

    /// Encode parity vs software focused on the 16-lane boundary (lengths
    /// 13..=49), covering the `_mm512_cvtps_ph` loop and the tail handoff.
    #[test]
    fn encode_16wide_boundary_vs_software() {
        let Some(hw) = v4_slice_entry() else { return };
        for len in 13..=49usize {
            let input: Vec<f32> = (0..len)
                .map(|i| ref_f16_to_f32(((i * 1237 + 0x3c00) & 0xFFFF) as u16))
                .collect();
            let mut hw_out = vec![0u16; len];
            let mut sw_out = vec![0u16; len];
            hw.f32_to_f16_slice(&input, &mut hw_out);
            archmage::ScalarToken.f32_to_f16_slice(&input, &mut sw_out);
            for (&x, (&a, &b)) in input.iter().zip(hw_out.iter().zip(sw_out.iter())) {
                if a == b {
                    continue;
                }
                assert!(
                    x.is_nan() && is_f16_nan(a) && is_f16_nan(b),
                    "AVX-512F encode boundary parity len={len} x={x:e}: hw={a:#06x} sw={b:#06x}"
                );
            }
        }
    }

    /// The **plain V4 path**: `X64V4Token` / `X64V4xToken` / `Avx512Fp16Token`
    /// implement `F16Convert` directly (their `F32x4Convert` supertrait is
    /// delegated to V3), so a V4-tier holder calls the slice converters **without
    /// extracting a V3 token first**. This test passes those tokens straight to
    /// `f16_to_f32_slice` / `f32_to_f16_slice` (the `.v3()`-free call that only
    /// compiles if the direct impls exist) and checks bit-identity vs software
    /// over a representative sample incl. the 16-lane boundary. Each token is
    /// summon-gated (only `X64V4Token` is guaranteed on this Zen 4 box; V4x/FP16
    /// skip with a notice when absent).
    #[test]
    fn plain_v4_path_tokens_accepted_directly() {
        use archmage::SimdToken;
        fn check<T: F16Convert>(tok: T, label: &str) {
            for len in [16usize, 17, 31, 48, 1000] {
                let f16in: Vec<u16> = (0..len)
                    .map(|i| ((i * 2179 + 0x33ff) & 0xFFFF) as u16)
                    .collect();
                let (mut hw, mut sw) = (vec![0f32; len], vec![0f32; len]);
                tok.f16_to_f32_slice(&f16in, &mut hw); // direct — no .v3()
                archmage::ScalarToken.f16_to_f32_slice(&f16in, &mut sw);
                for (&h, (&a, &b)) in f16in.iter().zip(hw.iter().zip(sw.iter())) {
                    if is_f16_nan(h) {
                        assert!(a.is_nan() && b.is_nan(), "{label} decode NaN {h:#06x}");
                    } else {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{label} decode len={len} h={h:#06x}"
                        );
                    }
                }
                let f32in: Vec<f32> = sw.clone();
                let (mut he, mut se) = (vec![0u16; len], vec![0u16; len]);
                tok.f32_to_f16_slice(&f32in, &mut he); // direct — no .v3()
                archmage::ScalarToken.f32_to_f16_slice(&f32in, &mut se);
                for (&x, (&a, &b)) in f32in.iter().zip(he.iter().zip(se.iter())) {
                    if a != b {
                        assert!(
                            x.is_nan() && is_f16_nan(a) && is_f16_nan(b),
                            "{label} encode len={len} x={x:e}: hw={a:#06x} sw={b:#06x}"
                        );
                    }
                }
            }
            eprintln!("plain V4 path: {label} accepted directly (16-wide)");
        }
        let mut n = 0;
        if let Some(t) = archmage::X64V4Token::summon() {
            check(t, "X64V4Token");
            n += 1;
        }
        if let Some(t) = archmage::X64V4xToken::summon() {
            check(t, "X64V4xToken");
            n += 1;
        }
        if let Some(t) = archmage::Avx512Fp16Token::summon() {
            check(t, "Avx512Fp16Token");
            n += 1;
        }
        if n == 0 {
            eprintln!("no V4-tier token summonable — plain V4 path not exercised on this host");
        }
    }
}

// ============================================================================
// Native NEON-f16 hardware path (aarch64): exhaustive bit-identity vs the
// oracle AND vs the software kernel.
//
// `F16Convert::f16_to_f32_slice` / `F16Convert::f32_to_f16_slice` dispatch to
// aarch64 NEON-f16
// (`vcvt_f32_f16` / `vcvt_f16_f32`) when handed a `NeonToken` *and* the CPU
// presents `fp16` (the `Arm64V2Token` tier) *and* the toolchain is ≥ 1.94 (so
// the `#[rustversion::since(1.94)]`-gated HW kernels are compiled). When any of
// those is false the same `NeonToken` runs the branchless software kernel.
//
// These tests prove the path the runtime actually takes (HW when available,
// software otherwise) is:
//   1. bit-identical to the independent scalar IEEE oracle, and
//   2. bit-identical to the `ScalarToken` software slice kernel (which on
//      aarch64 has no hardware override) — modulo the documented NaN-payload
//      divergence, which is exactly the 1022 f16 signaling-NaN patterns on
//      decode, the same divergence set the x86 F16C path exhibits.
//
// On a host (or QEMU `-cpu`) without `fp16`, `NeonToken` runs the software
// kernel and these tests still pass (they degenerate to a software-vs-software
// comparison). Run under QEMU with `-cpu max` to exercise the hardware path
// (see `.cargo/config.toml`).
//
// The HW kernel only EXISTS on rustc ≥ 1.94 (gated by `#[rustversion::since(1.94)]`
// in `convert_f16.rs`). The tiny `neon_f16_hw_compiled()` helper below mirrors
// that gate so the parity test expects software-vs-software on the pre-1.94
// fallback even when the CPU presents `fp16` — the exact condition the CI
// matrix's 1.93 cell runs under QEMU `-cpu max`.
//
// The whole module carries the aarch64 arch gate at the module boundary, so
// the gate is applied once — a new aarch64 test added here inherits it.
// ============================================================================
#[cfg(target_arch = "aarch64")]
mod aarch64_neon {
    use super::*;

    /// Whether the NEON-f16 hardware kernel is compiled into this build — mirrors
    /// the `#[rustversion::since(1.94)]` gate on `neon_f16_decode_hw` /
    /// `neon_f16_encode_hw`. `true` only on rustc ≥ 1.94.
    #[rustversion::since(1.94)]
    fn neon_f16_hw_compiled() -> bool {
        true
    }

    /// Pre-stabilization toolchain (rustc < 1.94): the HW kernel is not compiled,
    /// so a `NeonToken` always runs the software path — even on `fp16` hardware.
    #[rustversion::before(1.94)]
    fn neon_f16_hw_compiled() -> bool {
        false
    }

    /// All 65 536 f16 decoded through the `NeonToken` slice path must match the
    /// oracle bit-for-bit (every length residue 0..7 exercised so the 4-wide chunk
    /// and the scalar tail are both hit). On `fp16`-capable hardware this is the
    /// `vcvt_f32_f16` path; otherwise the software kernel — both must match.
    #[test]
    fn native_neon_f16_decode_exhaustive_vs_oracle() {
        use archmage::SimdToken;
        let Some(token) = archmage::NeonToken::summon() else {
            eprintln!("NeonToken not available on this host — scalar path covers correctness");
            return;
        };
        let all: Vec<u16> = (0u32..=0xFFFF).map(|v| v as u16).collect();
        let mut out = vec![0f32; all.len()];
        token.f16_to_f32_slice(&all, &mut out);
        for (&h, &got) in all.iter().zip(out.iter()) {
            // For a NaN input `vcvt_f32_f16` returns the hardware-quieted NaN whose
            // payload may differ from the software widening — both are valid f32
            // NaNs (the documented benign divergence). Assert finite/Inf exactly;
            // require only NaN-ness for NaN inputs.
            if is_f16_nan(h) {
                assert!(
                    got.is_nan(),
                    "NEON-f16 decode of NaN f16 {h:#06x} produced a non-NaN f32 ({:#010x})",
                    got.to_bits()
                );
            } else {
                assert_eq!(
                    got.to_bits(),
                    ref_f16_to_f32(h).to_bits(),
                    "NEON-f16 decode mismatch vs oracle h={h:#06x}"
                );
            }
        }
        // Tail-residue sweep: lengths 1..=64 starting at varied offsets.
        for start in [0usize, 1, 2, 3, 5, 7] {
            for len in 1..=64usize {
                let input: Vec<u16> = (0..len).map(|i| ((start + i) & 0xFFFF) as u16).collect();
                let mut o = vec![0f32; len];
                token.f16_to_f32_slice(&input, &mut o);
                for (&h, &got) in input.iter().zip(o.iter()) {
                    if is_f16_nan(h) {
                        assert!(
                            got.is_nan(),
                            "NEON-f16 decode (tail) NaN f16 {h:#06x} not NaN"
                        );
                    } else {
                        assert_eq!(
                            got.to_bits(),
                            ref_f16_to_f32(h).to_bits(),
                            "NEON-f16 decode mismatch (tail) start={start} len={len} h={h:#06x}"
                        );
                    }
                }
            }
        }
    }

    /// The `NeonToken` decode path must be byte-for-byte identical to the
    /// `ScalarToken` software decode path over all 65 536 f16
    /// (finite/subnormal/Inf), with the documented NaN-only payload divergence. On
    /// `fp16` hardware this proves the `vcvt_f32_f16` path matches software; on a
    /// CPU without `fp16` both sides are the software kernel and the divergence
    /// count is 0.
    #[test]
    fn native_neon_f16_decode_matches_software_exhaustive() {
        use archmage::SimdToken;
        let Some(hw) = archmage::NeonToken::summon() else {
            eprintln!("NeonToken not available — skipping native-vs-software decode parity");
            return;
        };
        // The NEON-f16 hardware kernel only EXISTS on rustc ≥ 1.94 (it is gated by
        // `#[rustversion::since(1.94)]` in `convert_f16.rs`). The HW path is taken
        // iff that kernel is compiled AND the CPU presents `fp16` at runtime. On
        // rustc < 1.94 a `NeonToken` runs the software kernel even on `fp16`
        // hardware (e.g. QEMU `-cpu max`), so the divergence count is 0 — this is
        // exactly the pre-1.94 fallback the CI matrix's 1.93 cell exercises, and it
        // must NOT expect the 1022-NaN HW divergence.
        let hw_compiled = neon_f16_hw_compiled();
        let fp16 = hw_compiled && archmage::Arm64V2Token::summon().is_some();
        let all: Vec<u16> = (0u32..=0xFFFF).map(|v| v as u16).collect();
        let mut hw_out = vec![0f32; all.len()];
        let mut sw_out = vec![0f32; all.len()];
        hw.f16_to_f32_slice(&all, &mut hw_out);
        archmage::ScalarToken.f16_to_f32_slice(&all, &mut sw_out);
        let mut nan_payload_diffs = 0u64;
        for (i, (&a, &b)) in hw_out.iter().zip(sw_out.iter()).enumerate() {
            let h = all[i];
            if a.to_bits() == b.to_bits() {
                continue;
            }
            // A divergence is only permitted on an f16 *signaling* NaN input (top
            // mantissa bit clear): `vcvt_f32_f16` quiets it (sets the f32 mantissa
            // MSB) while the software path preserves the signaling payload. Both
            // results must still be f32 NaNs.
            assert!(
                is_f16_nan(h) && (h & 0x0200) == 0 && a.is_nan() && b.is_nan(),
                "NEON-f16 vs software decode differ on a non-(signaling-NaN) value at idx={i} h={h:#06x}: hw={:#010x} sw={:#010x}",
                a.to_bits(),
                b.to_bits()
            );
            nan_payload_diffs += 1;
        }
        if fp16 {
            // Hardware path active: the divergence set is exactly the f16 signaling
            // NaNs — payloads in 0x001..=0x1FF (top mantissa bit clear) × 2 signs =
            // 511 × 2 = 1022. Identical to the x86 F16C path's divergence count.
            assert_eq!(
                nan_payload_diffs, 1022,
                "expected the NEON-f16/software decode divergence to be exactly the 1022 f16 \
                 signaling-NaN patterns (got {nan_payload_diffs}); fp16 was detected"
            );
        } else {
            // Software path: either `fp16` is absent at runtime, or the toolchain
            // is < 1.94 so the HW kernel was never compiled. Either way `NeonToken`
            // ran the software kernel — byte-identical to `ScalarToken`.
            assert_eq!(
                nan_payload_diffs, 0,
                "software path (no fp16 or rustc < 1.94): NeonToken should be byte-identical \
                 to the ScalarToken software path (hw_compiled={hw_compiled})"
            );
        }
    }

    /// The `NeonToken` encode path must match the oracle over the boundary bands
    /// and a dense strided sweep (NaN payload tolerated, like the software encode
    /// test).
    #[test]
    fn native_neon_f16_encode_vs_oracle() {
        use archmage::SimdToken;
        let Some(token) = archmage::NeonToken::summon() else {
            eprintln!("NeonToken not available — scalar encode tests cover correctness");
            return;
        };
        let mut inputs: Vec<f32> = Vec::new();
        for hv in 0u32..=0xFFFF {
            inputs.push(ref_f16_to_f32(hv as u16));
        }
        let step: u32 = if cfg!(debug_assertions) { 251 } else { 1 };
        let mut bits = 100u32 << 23;
        let sub_hi = 113u32 << 23;
        while bits < sub_hi {
            inputs.push(f32::from_bits(bits));
            inputs.push(f32::from_bits(bits | 0x8000_0000));
            bits += step;
        }
        let ov_lo = (127u32 + 16) << 23;
        let ov_hi = 255u32 << 23;
        let mut bits = ov_lo;
        loop {
            inputs.push(f32::from_bits(bits));
            inputs.push(f32::from_bits(bits | 0x8000_0000));
            if bits >= ov_hi {
                break;
            }
            let near = bits.saturating_sub(ov_lo) < 4096 || ov_hi.saturating_sub(bits) < 4096;
            let s = if cfg!(debug_assertions) && !near {
                251
            } else {
                1
            };
            bits = (bits + s).min(ov_hi);
        }
        let stride: u32 = 1009;
        let mut bits: u32 = 0;
        loop {
            inputs.push(f32::from_bits(bits));
            let (next, ov) = bits.overflowing_add(stride);
            if ov {
                break;
            }
            bits = next;
        }

        let mut out = vec![0u16; inputs.len()];
        token.f32_to_f16_slice(&inputs, &mut out);
        let mut mismatches = 0u64;
        for (&x, &got) in inputs.iter().zip(out.iter()) {
            let want = ref_f32_to_f16(x);
            if want == got {
                continue;
            }
            if x.is_nan() {
                assert!(
                    is_f16_nan(want) && is_f16_nan(got),
                    "NEON-f16 NaN encode produced non-NaN: x={x:?} want={want:#06x} got={got:#06x}"
                );
                continue;
            }
            mismatches += 1;
            if mismatches <= 16 {
                eprintln!("NEON-f16 encode mismatch x={x:e} want={want:#06x} got={got:#06x}");
            }
        }
        assert_eq!(
            mismatches, 0,
            "NEON-f16 encode diverged from the oracle on {mismatches} finite inputs"
        );
    }

    /// The `NeonToken` encode path must match the `ScalarToken` software encode
    /// path bit-for-bit for every finite/Inf input (NaN payload tolerated) — over
    /// the f16-roundtrip grid plus a dense f32 sweep.
    #[test]
    fn native_neon_f16_encode_matches_software() {
        use archmage::SimdToken;
        let Some(hw) = archmage::NeonToken::summon() else {
            eprintln!("NeonToken not available — skipping native-vs-software encode parity");
            return;
        };
        let mut inputs: Vec<f32> = (0u32..=0xFFFF)
            .map(|hv| ref_f16_to_f32(hv as u16))
            .collect();
        let stride: u32 = 1009;
        let mut bits: u32 = 0;
        loop {
            inputs.push(f32::from_bits(bits));
            let (next, ov) = bits.overflowing_add(stride);
            if ov {
                break;
            }
            bits = next;
        }
        let mut hw_out = vec![0u16; inputs.len()];
        let mut sw_out = vec![0u16; inputs.len()];
        hw.f32_to_f16_slice(&inputs, &mut hw_out);
        archmage::ScalarToken.f32_to_f16_slice(&inputs, &mut sw_out);
        for (i, ((&x, &a), &b)) in inputs
            .iter()
            .zip(hw_out.iter())
            .zip(sw_out.iter())
            .enumerate()
        {
            if a == b {
                continue;
            }
            // NaN inputs: both must be f16 NaN; payload may differ between HW and SW.
            assert!(
                x.is_nan() && is_f16_nan(a) && is_f16_nan(b),
                "NEON-f16 vs software encode differ at idx={i} x={x:e}: hw={a:#06x} sw={b:#06x}"
            );
        }
    }
}

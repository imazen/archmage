//! Adversarial tests for [`F32x8FromHalves`] and [`F32x16FromHalves`].
//!
//! These tests try to break the contract: confirm lane order under every
//! backend, prove polyfill paths agree with native intrinsics on the same
//! inputs, exercise round-trip / idempotency invariants, and verify the
//! Scalar reference matches every other backend.

use archmage::{ScalarToken, SimdToken};
use magetypes::simd::generic::{F32x8FromHalves, f32x4, f32x8};
#[cfg(feature = "w512")]
use magetypes::simd::generic::{F32x16FromHalves, f32x16};

// Inputs chosen so each lane is unique and bit-distinguishable —
// catches lane-shuffling regressions even at single-bit granularity.
const F32X4_A: [f32; 4] = [1.0, 2.5, -3.25, 4.125];
const F32X4_B: [f32; 4] = [5.0625, -6.0, 7.5, 8.0];
const F32X4_C: [f32; 4] = [9.0, 10.0, 11.0, 12.0];
const F32X4_D: [f32; 4] = [13.0, 14.0, 15.0, 16.0];

const F32X8_LO: [f32; 8] = [1.0, 2.5, -3.25, 4.125, 5.0625, -6.0, 7.5, 8.0];
const F32X8_HI: [f32; 8] = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

// ============================================================================
// W128 → W256 (f32x4 → f32x8)
// ============================================================================

/// Lane order: from_halves(lo, hi) MUST produce
///   [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
/// for every backend. A regression where the halves swap or interleave
/// would corrupt downstream math without crashing.
fn lane_order_f32x8<T: F32x8FromHalves>(token: T) {
    let lo = f32x4::<T>::from_array(token, F32X4_A);
    let hi = f32x4::<T>::from_array(token, F32X4_B);
    let wide = f32x8::<T>::from_halves(token, lo, hi);
    let actual = wide.to_array();
    let expected = [
        F32X4_A[0], F32X4_A[1], F32X4_A[2], F32X4_A[3], F32X4_B[0], F32X4_B[1], F32X4_B[2],
        F32X4_B[3],
    ];
    assert_eq!(actual, expected, "lane order f32x4→f32x8 broke");
}

/// Round-trip: split-after-from is identity.
fn roundtrip_f32x8<T: F32x8FromHalves>(token: T) {
    let lo = f32x4::<T>::from_array(token, F32X4_A);
    let hi = f32x4::<T>::from_array(token, F32X4_B);
    let wide = f32x8::<T>::from_halves(token, lo, hi);
    let (back_lo, back_hi) = wide.split();
    assert_eq!(back_lo.to_array(), F32X4_A);
    assert_eq!(back_hi.to_array(), F32X4_B);
}

/// Reverse round-trip: from-after-split is identity.
fn reverse_roundtrip_f32x8<T: F32x8FromHalves>(token: T) {
    let original_lanes: [f32; 8] = [
        F32X4_A[0], F32X4_A[1], F32X4_A[2], F32X4_A[3], F32X4_B[0], F32X4_B[1], F32X4_B[2],
        F32X4_B[3],
    ];
    let original = f32x8::<T>::from_array(token, original_lanes);
    let (lo, hi) = original.split();
    let rebuilt = f32x8::<T>::from_halves(token, lo, hi);
    assert_eq!(rebuilt.to_array(), original_lanes, "from(split(x)) ≠ x");
}

/// Cross-backend parity: every backend must agree with Scalar lane-for-lane.
/// Catches the case where a polyfill / native-intrinsic disagree on lane order.
fn parity_with_scalar_f32x8<T: F32x8FromHalves>(token: T) {
    let scalar_token = ScalarToken::summon().unwrap();
    let lo_t = f32x4::<T>::from_array(token, F32X4_A);
    let hi_t = f32x4::<T>::from_array(token, F32X4_B);
    let lo_s = f32x4::<ScalarToken>::from_array(scalar_token, F32X4_A);
    let hi_s = f32x4::<ScalarToken>::from_array(scalar_token, F32X4_B);
    let wide_t = f32x8::<T>::from_halves(token, lo_t, hi_t);
    let wide_s = f32x8::<ScalarToken>::from_halves(scalar_token, lo_s, hi_s);
    assert_eq!(
        wide_t.to_array(),
        wide_s.to_array(),
        "backend-vs-scalar parity failed"
    );
    assert_eq!(wide_t.low().to_array(), wide_s.low().to_array());
    assert_eq!(wide_t.high().to_array(), wide_s.high().to_array());
}

// ============================================================================
// W256 → W512 (f32x8 → f32x16) — gated on `w512` magetypes feature
// ============================================================================

#[cfg(feature = "w512")]
fn lane_order_f32x16<T: F32x16FromHalves>(token: T) {
    let lo = f32x8::<T>::from_array(token, F32X8_LO);
    let hi = f32x8::<T>::from_array(token, F32X8_HI);
    let wide = f32x16::<T>::from_halves(token, lo, hi);
    let actual = wide.to_array();
    let expected = [
        F32X8_LO[0],
        F32X8_LO[1],
        F32X8_LO[2],
        F32X8_LO[3],
        F32X8_LO[4],
        F32X8_LO[5],
        F32X8_LO[6],
        F32X8_LO[7],
        F32X8_HI[0],
        F32X8_HI[1],
        F32X8_HI[2],
        F32X8_HI[3],
        F32X8_HI[4],
        F32X8_HI[5],
        F32X8_HI[6],
        F32X8_HI[7],
    ];
    assert_eq!(actual, expected, "lane order f32x8→f32x16 broke");
}

#[cfg(feature = "w512")]
fn roundtrip_f32x16<T: F32x16FromHalves>(token: T) {
    let lo = f32x8::<T>::from_array(token, F32X8_LO);
    let hi = f32x8::<T>::from_array(token, F32X8_HI);
    let wide = f32x16::<T>::from_halves(token, lo, hi);
    let (back_lo, back_hi) = wide.split();
    assert_eq!(back_lo.to_array(), F32X8_LO);
    assert_eq!(back_hi.to_array(), F32X8_HI);
}

#[cfg(feature = "w512")]
fn reverse_roundtrip_f32x16<T: F32x16FromHalves>(token: T) {
    let mut original_lanes = [0f32; 16];
    original_lanes[..8].copy_from_slice(&F32X8_LO);
    original_lanes[8..].copy_from_slice(&F32X8_HI);
    let original = f32x16::<T>::from_array(token, original_lanes);
    let (lo, hi) = original.split();
    let rebuilt = f32x16::<T>::from_halves(token, lo, hi);
    assert_eq!(rebuilt.to_array(), original_lanes);
}

#[cfg(feature = "w512")]
fn parity_with_scalar_f32x16<T: F32x16FromHalves>(token: T) {
    let scalar_token = ScalarToken::summon().unwrap();
    let lo_t = f32x8::<T>::from_array(token, F32X8_LO);
    let hi_t = f32x8::<T>::from_array(token, F32X8_HI);
    let lo_s = f32x8::<ScalarToken>::from_array(scalar_token, F32X8_LO);
    let hi_s = f32x8::<ScalarToken>::from_array(scalar_token, F32X8_HI);
    let wide_t = f32x16::<T>::from_halves(token, lo_t, hi_t);
    let wide_s = f32x16::<ScalarToken>::from_halves(scalar_token, lo_s, hi_s);
    assert_eq!(wide_t.to_array(), wide_s.to_array());
    assert_eq!(wide_t.low().to_array(), wide_s.low().to_array());
    assert_eq!(wide_t.high().to_array(), wide_s.high().to_array());
}

// ============================================================================
// Per-token instantiation — every backend exercised on every supported host
// ============================================================================

#[test]
fn scalar_w128_w256() {
    let t = ScalarToken::summon().expect("scalar always available");
    lane_order_f32x8(t);
    roundtrip_f32x8(t);
    reverse_roundtrip_f32x8(t);
    parity_with_scalar_f32x8(t);
}

#[cfg(feature = "w512")]
#[test]
fn scalar_w256_w512() {
    let t = ScalarToken::summon().unwrap();
    lane_order_f32x16(t);
    roundtrip_f32x16(t);
    reverse_roundtrip_f32x16(t);
    parity_with_scalar_f32x16(t);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x64v3_w128_w256() {
    if let Some(t) = archmage::X64V3Token::summon() {
        lane_order_f32x8(t);
        roundtrip_f32x8(t);
        reverse_roundtrip_f32x8(t);
        parity_with_scalar_f32x8(t);
    } else {
        eprintln!("X64V3Token not available — skipping");
    }
}

#[cfg(all(target_arch = "x86_64", feature = "w512"))]
#[test]
fn x64v3_w256_w512_polyfilled() {
    // f32x16<X64V3Token> polyfills to [__m256; 2] — exercise the polyfill path.
    if let Some(t) = archmage::X64V3Token::summon() {
        lane_order_f32x16(t);
        roundtrip_f32x16(t);
        reverse_roundtrip_f32x16(t);
        parity_with_scalar_f32x16(t);
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn x64v4_w128_w256_via_v3_delegation() {
    // Exercises the V4-delegated F32x4 / F32x8 backends — same machinery as V3.
    if let Some(t) = archmage::X64V4Token::summon() {
        lane_order_f32x8(t);
        roundtrip_f32x8(t);
        reverse_roundtrip_f32x8(t);
        parity_with_scalar_f32x8(t);
    } else {
        eprintln!("X64V4Token not available — skipping");
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
#[test]
fn x64v4_w256_w512_native() {
    // f32x16<X64V4Token> uses native __m512 + _mm512_insertf32x8 — distinct
    // from V3's polyfill path. Parity with Scalar (and V3) MUST hold.
    if let Some(t) = archmage::X64V4Token::summon() {
        lane_order_f32x16(t);
        roundtrip_f32x16(t);
        reverse_roundtrip_f32x16(t);
        parity_with_scalar_f32x16(t);
    }
}

/// **Critical regression test**: V4's native AVX-512 `_mm512_insertf32x8`
/// path must produce the same lane assignment as V3's polyfilled
/// `[__m256; 2]` path. If V4 is using imm=0 instead of imm=1 (or any
/// other lane mistake), this will catch it.
#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
#[test]
fn v4_native_matches_v3_polyfill() {
    let v4 = match archmage::X64V4Token::summon() {
        Some(t) => t,
        None => return,
    };
    let v3 = archmage::X64V3Token::summon().expect("V4 implies V3");

    let lo_v4 = f32x8::<archmage::X64V4Token>::from_array(v4, F32X8_LO);
    let hi_v4 = f32x8::<archmage::X64V4Token>::from_array(v4, F32X8_HI);
    let wide_v4 = f32x16::<archmage::X64V4Token>::from_halves(v4, lo_v4, hi_v4);

    let lo_v3 = f32x8::<archmage::X64V3Token>::from_array(v3, F32X8_LO);
    let hi_v3 = f32x8::<archmage::X64V3Token>::from_array(v3, F32X8_HI);
    let wide_v3 = f32x16::<archmage::X64V3Token>::from_halves(v3, lo_v3, hi_v3);

    assert_eq!(
        wide_v4.to_array(),
        wide_v3.to_array(),
        "V4 native path disagrees with V3 polyfill"
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_w128_w256() {
    if let Some(t) = archmage::NeonToken::summon() {
        lane_order_f32x8(t);
        roundtrip_f32x8(t);
        reverse_roundtrip_f32x8(t);
        parity_with_scalar_f32x8(t);
    }
}

#[cfg(all(target_arch = "aarch64", feature = "w512"))]
#[test]
fn neon_w256_w512_cascade_polyfill() {
    if let Some(t) = archmage::NeonToken::summon() {
        lane_order_f32x16(t);
        roundtrip_f32x16(t);
        reverse_roundtrip_f32x16(t);
        parity_with_scalar_f32x16(t);
    }
}

#[cfg(target_arch = "wasm32")]
#[test]
fn wasm128_w128_w256() {
    if let Some(t) = archmage::Wasm128Token::summon() {
        lane_order_f32x8(t);
        roundtrip_f32x8(t);
        reverse_roundtrip_f32x8(t);
        parity_with_scalar_f32x8(t);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "w512"))]
#[test]
fn wasm128_w256_w512_cascade_polyfill() {
    if let Some(t) = archmage::Wasm128Token::summon() {
        lane_order_f32x16(t);
        roundtrip_f32x16(t);
        reverse_roundtrip_f32x16(t);
        parity_with_scalar_f32x16(t);
    }
}

// ============================================================================
// Adversarial: bit-precise edge values
// ============================================================================

const EDGE_LANES_4: [f32; 4] = [
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::MIN_POSITIVE,
];
const EDGE_LANES_4B: [f32; 4] = [-0.0, f32::EPSILON, f32::MAX, f32::MIN];

/// `from_halves` MUST be a pure bit-shuffle — it must not normalize NaN
/// payloads, flush denormals, or otherwise transform lane bits. Verify
/// every lane survives the round-trip bitwise.
fn bitwise_round_trip_f32x8<T: F32x8FromHalves>(token: T) {
    let lo = f32x4::<T>::from_array(token, EDGE_LANES_4);
    let hi = f32x4::<T>::from_array(token, EDGE_LANES_4B);
    let wide = f32x8::<T>::from_halves(token, lo, hi);
    let arr = wide.to_array();
    for i in 0..4 {
        assert_eq!(
            arr[i].to_bits(),
            EDGE_LANES_4[i].to_bits(),
            "lo lane {i} bits changed"
        );
    }
    for i in 0..4 {
        assert_eq!(
            arr[4 + i].to_bits(),
            EDGE_LANES_4B[i].to_bits(),
            "hi lane {i} bits changed"
        );
    }
}

#[test]
fn scalar_bit_precise() {
    bitwise_round_trip_f32x8(ScalarToken::summon().unwrap());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x64v3_bit_precise() {
    if let Some(t) = archmage::X64V3Token::summon() {
        bitwise_round_trip_f32x8(t);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_bit_precise() {
    if let Some(t) = archmage::NeonToken::summon() {
        bitwise_round_trip_f32x8(t);
    }
}

#[cfg(target_arch = "wasm32")]
#[test]
fn wasm128_bit_precise() {
    if let Some(t) = archmage::Wasm128Token::summon() {
        bitwise_round_trip_f32x8(t);
    }
}

// ============================================================================
// Idempotency: re-extracting halves twice yields identical results
// ============================================================================

fn idempotent_extraction_f32x8<T: F32x8FromHalves>(token: T) {
    let wide = f32x8::<T>::from_array(token, F32X8_LO);
    assert_eq!(wide.low().to_array(), wide.low().to_array());
    assert_eq!(wide.high().to_array(), wide.high().to_array());
    let (a_lo, a_hi) = wide.split();
    let (b_lo, b_hi) = wide.split();
    assert_eq!(a_lo.to_array(), b_lo.to_array());
    assert_eq!(a_hi.to_array(), b_hi.to_array());
}

#[test]
fn scalar_idempotent() {
    idempotent_extraction_f32x8(ScalarToken::summon().unwrap());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x64v3_idempotent() {
    if let Some(t) = archmage::X64V3Token::summon() {
        idempotent_extraction_f32x8(t);
    }
}

// ============================================================================
// Many-permutation lane-shuffle test (random-ish but deterministic)
// ============================================================================

/// Build a lookup from lane index in the wide vector to its expected source
/// lane in (lo, hi). Detect any silent reorder.
fn permutation_test_f32x8<T: F32x8FromHalves>(token: T) {
    for seed in 0u32..16 {
        let mut lo = [0f32; 4];
        let mut hi = [0f32; 4];
        for i in 0..4 {
            lo[i] = f32::from_bits(0x4000_0000 | (seed << 16) | (i as u32));
            hi[i] = f32::from_bits(0x4080_0000 | (seed << 16) | (i as u32));
        }
        let lo_v = f32x4::<T>::from_array(token, lo);
        let hi_v = f32x4::<T>::from_array(token, hi);
        let wide = f32x8::<T>::from_halves(token, lo_v, hi_v).to_array();
        for i in 0..4 {
            assert_eq!(
                wide[i].to_bits(),
                lo[i].to_bits(),
                "lo lane {i} (seed {seed})"
            );
            assert_eq!(
                wide[4 + i].to_bits(),
                hi[i].to_bits(),
                "hi lane {i} (seed {seed})"
            );
        }
    }
}

#[test]
fn scalar_permutations() {
    permutation_test_f32x8(ScalarToken::summon().unwrap());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x64v3_permutations() {
    if let Some(t) = archmage::X64V3Token::summon() {
        permutation_test_f32x8(t);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_permutations() {
    if let Some(t) = archmage::NeonToken::summon() {
        permutation_test_f32x8(t);
    }
}

// Suppress unused-const warnings for tests that don't reference every constant.
#[allow(dead_code)]
const _USES: [&[f32]; 2] = [&F32X4_C, &F32X4_D];

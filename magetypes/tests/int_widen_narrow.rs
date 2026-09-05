//! Differential coverage for integer widening (`widen_low` / `widen_high`) and
//! saturating narrowing (`narrow_saturating_*`), adjacent dot products, and
//! full-range absolute differences / widening byte sums.
//!
//! Every backend reachable from the host arch is compared against a scalar
//! reference computed here in the test — not borrowed from the scalar backend,
//! so a generator bug in *that* backend is caught too — at every width the
//! types come in.
//!
//! What this file is deliberately strict about:
//!
//! 1. **Lane order.** [`widen_lane_order`] and [`narrow_lane_order`] use
//!    operands whose value is distinct in every lane and compare element by
//!    element, so a permuted result fails; a checksum or a `reduce_add` would
//!    not catch it. This is the assertion that pins the AVX2 narrowing fixup:
//!    `_mm256_packs/packus_*` emit `[a.lo, b.lo, a.hi, b.hi]` in 64-bit groups
//!    (the `IDXS` constant in each `pack*` body in `core_arch/x86/avx2.rs`), so
//!    without the `_mm256_permute4x64_epi64::<0xD8>` the 256-bit result is a
//!    permutation of the right bytes — which every other kind of check passes.
//!
//! 2. **Saturation direction.** The narrow patterns include exactly the values
//!    where the signed-destination and unsigned-destination flavours disagree:
//!    `i8::MIN - 1`, `u8::MAX`, `u8::MAX + 1`, `i16::MIN`, `i16::MAX`. A
//!    backend that reached for the wrong `pack*`/`vqmov*n*` variant differs
//!    there and nowhere else.
//!
//! 3. **Exhaustive where the domain allows.** Widening from 8-bit sweeps all
//!    256 source values in all 16 lane positions. Widening from 16-bit and
//!    narrowing from 16-bit sweep all 65 536 source values. For a lane-wise
//!    op with a fixed lane mapping that is the whole function.
//!
//! 4. **The arm actually ran.** Each backend test asserts the token summons
//!    whenever the host reports the features for it, and every runner returns
//!    the number of comparisons it made so the caller can assert a floor. A
//!    macro that silently expanded to nothing, or a `summon()` that quietly
//!    returned `None` on a capable CPU, fails instead of passing green.
//!
//! Native and emulated CI runners exercise x86, NEON, WASM and scalar.
//! Integer primitive cases include every ordered u8 pair, wrapping MIN*MIN
//! products, and distinct per-lane references calculated in i64.

use archmage::{ScalarToken, SimdToken};
use magetypes::simd::generic::{
    i8x16, i8x32, i16x8, i16x16, i32x4, i32x8, u8x16, u8x32, u16x8, u16x16,
};
#[cfg(feature = "w512")]
use magetypes::simd::generic::{i8x64, i16x32, i32x16, u8x64, u16x32};

// ============================================================================
// Operand patterns
// ============================================================================

/// Sign- and magnitude-edge pattern for widening sources. `seed` rotates the
/// pattern through the lanes so every value visits every lane position.
macro_rules! widen_pattern {
    ($elem:ty, $n:expr, $seed:expr) => {{
        let v: [$elem; $n] = core::array::from_fn(|i| match (i + $seed) % 8 {
            0 => 0,
            1 => 1,
            2 => <$elem>::MAX,
            3 => <$elem>::MIN,
            4 => <$elem>::MAX - 1,
            5 => <$elem>::MIN + 1,
            // High bit set on unsigned, mid-range on signed — the lane that
            // separates zero-extension from sign-extension.
            6 => <$elem>::MAX / 2 + 1,
            _ => 42 as $elem,
        });
        v
    }};
}

/// Narrowing-source pattern. Every entry that matters is a boundary of one of
/// the two destination ranges, so the signed and unsigned flavours are forced
/// apart. `$sd` is the signed destination element, `$ud` the unsigned one.
macro_rules! narrow_pattern {
    ($elem:ty, $sd:ty, $ud:ty, $n:expr, $seed:expr) => {{
        let v: [$elem; $n] = core::array::from_fn(|i| match (i + $seed) % 16 {
            0 => 0,
            1 => 1,
            2 => -1,
            3 => <$sd>::MAX as $elem,
            4 => <$sd>::MAX as $elem + 1,
            5 => <$sd>::MIN as $elem,
            6 => <$sd>::MIN as $elem - 1,
            7 => <$ud>::MAX as $elem,
            8 => <$ud>::MAX as $elem + 1,
            9 => -(<$ud>::MAX as $elem),
            10 => <$elem>::MAX,
            11 => <$elem>::MIN,
            12 => <$elem>::MAX - 1,
            13 => <$elem>::MIN + 1,
            14 => 100,
            _ => -100,
        });
        v
    }};
}

// ============================================================================
// Reference implementations (written here, not borrowed from any backend)
// ============================================================================

/// `widen_low` / `widen_high` reference: a plain per-lane cast of one half.
macro_rules! want_widen {
    ($a:expr, $elem:ty, $delem:ty, $half:expr, $offset:expr) => {{
        let src = $a;
        let v: [$delem; $half] = core::array::from_fn(|i| src[i + $offset] as $delem);
        v
    }};
}

/// Saturating-narrow reference: clamp into the destination range, `a` first.
macro_rules! want_narrow {
    ($a:expr, $b:expr, $elem:ty, $dst:ty, $n:expr, $two_n:expr, $lo:expr, $hi:expr) => {{
        let (a, b) = ($a, $b);
        let v: [$dst; $two_n] = core::array::from_fn(|i| {
            let x: $elem = if i < $n { a[i] } else { b[i - $n] };
            let lo: $elem = $lo;
            let hi: $elem = $hi;
            (if x < lo {
                lo
            } else if x > hi {
                hi
            } else {
                x
            }) as $dst
        });
        v
    }};
}

// ============================================================================
// Per-type checks
// ============================================================================

/// One widening family, over eight rotations of the boundary pattern.
macro_rules! check_widen {
    ($Tok:ty, $t:expr, $ty:ident, $elem:ty, $delem:ty, $n:expr, $half:expr) => {{
        let t = $t;
        let mut cases = 0usize;
        for seed in 0..8usize {
            let a: [$elem; $n] = widen_pattern!($elem, $n, seed);
            let va = $ty::<$Tok>::from_array(t, a);
            assert_eq!(
                va.widen_low().to_array(),
                want_widen!(a, $elem, $delem, $half, 0),
                "{}::widen_low (seed {seed})",
                stringify!($ty)
            );
            assert_eq!(
                va.widen_high().to_array(),
                want_widen!(a, $elem, $delem, $half, $half),
                "{}::widen_high (seed {seed})",
                stringify!($ty)
            );
            cases += 2;
        }
        cases += widen_lane_order!($Tok, t, $ty, $elem, $delem, $n, $half);
        cases
    }};
}

/// Lane-order pin for widening: every source lane holds a distinct value, so a
/// swapped low/high half, a reversed polyfill array, or an `unpack`-shaped
/// lowering that interleaves rather than extends all fail element-wise.
macro_rules! widen_lane_order {
    ($Tok:ty, $t:expr, $ty:ident, $elem:ty, $delem:ty, $n:expr, $half:expr) => {{
        let t = $t;
        // 1, 2, 3, ... wrapped into the element range: distinct in every lane
        // for every width generated here (n <= 64 <= u8::MAX).
        let a: [$elem; $n] = core::array::from_fn(|i| (i as $elem).wrapping_add(1));
        let va = $ty::<$Tok>::from_array(t, a);
        let lo = va.widen_low().to_array();
        let hi = va.widen_high().to_array();
        for i in 0..$half {
            assert_eq!(
                lo[i],
                a[i] as $delem,
                "{} lane order: widen_low lane {i}",
                stringify!($ty)
            );
            assert_eq!(
                hi[i],
                a[i + $half] as $delem,
                "{} lane order: widen_high lane {i}",
                stringify!($ty)
            );
        }
        2usize * $half
    }};
}

/// One saturating-narrowing family, over sixteen rotations of the boundary
/// pattern, in both destination signednesses.
macro_rules! check_narrow {
    ($Tok:ty, $t:expr, $ty:ident, $elem:ty,
     $sm:ident, $sd:ty, $um:ident, $ud:ty, $n:expr, $two_n:expr) => {{
        let t = $t;
        let mut cases = 0usize;
        for seed in 0..16usize {
            let a: [$elem; $n] = narrow_pattern!($elem, $sd, $ud, $n, seed);
            let b: [$elem; $n] = narrow_pattern!($elem, $sd, $ud, $n, seed + 5);
            let va = $ty::<$Tok>::from_array(t, a);
            let vb = $ty::<$Tok>::from_array(t, b);
            assert_eq!(
                va.$sm(vb).to_array(),
                want_narrow!(
                    a,
                    b,
                    $elem,
                    $sd,
                    $n,
                    $two_n,
                    <$sd>::MIN as $elem,
                    <$sd>::MAX as $elem
                ),
                "{}::{} (seed {seed})",
                stringify!($ty),
                stringify!($sm)
            );
            assert_eq!(
                va.$um(vb).to_array(),
                want_narrow!(a, b, $elem, $ud, $n, $two_n, 0, <$ud>::MAX as $elem),
                "{}::{} (seed {seed})",
                stringify!($ty),
                stringify!($um)
            );
            cases += 2;
        }
        cases += narrow_lane_order!($Tok, t, $ty, $elem, $sm, $sd, $um, $ud, $n, $two_n);
        cases
    }};
}

/// Lane-order pin for narrowing — the assertion that catches a missing AVX2
/// `permute4x64` fixup.
///
/// Both operands are in range, so saturation is the identity and the expected
/// result is literally `a` followed by `b`. Every lane is distinct, so the
/// `[a.lo, b.lo, a.hi, b.hi]` grouping AVX2's `pack*` produces is a different
/// array in every lane but the first four.
macro_rules! narrow_lane_order {
    ($Tok:ty, $t:expr, $ty:ident, $elem:ty,
     $sm:ident, $sd:ty, $um:ident, $ud:ty, $n:expr, $two_n:expr) => {{
        let t = $t;
        // 1..=n and n+1..=2n, both inside `i8`/`i16` range so no clamping
        // happens and the comparison is against the concatenation itself.
        let a: [$elem; $n] = core::array::from_fn(|i| (i as $elem) + 1);
        let b: [$elem; $n] = core::array::from_fn(|i| (i as $elem) + 1 + $n as $elem);
        let va = $ty::<$Tok>::from_array(t, a);
        let vb = $ty::<$Tok>::from_array(t, b);

        let got_s = va.$sm(vb).to_array();
        let got_u = va.$um(vb).to_array();
        for i in 0..$two_n {
            let want: $elem = if i < $n { a[i] } else { b[i - $n] };
            assert_eq!(
                got_s[i],
                want as $sd,
                "{} lane order: {} lane {i} — result is a permutation, not [a, b]",
                stringify!($ty),
                stringify!($sm)
            );
            assert_eq!(
                got_u[i],
                want as $ud,
                "{} lane order: {} lane {i} — result is a permutation, not [a, b]",
                stringify!($ty),
                stringify!($um)
            );
        }
        2usize * $two_n
    }};
}

// ============================================================================
// Exhaustive sweeps
// ============================================================================

/// Every one of the 256 source values, in every one of the 16 lane positions,
/// through both halves of an 8-bit widening. 8-bit is small enough that
/// "sampled" is not an excuse.
macro_rules! exhaustive_widen_8 {
    ($Tok:ty, $t:expr, $ty:ident, $elem:ty, $delem:ty) => {{
        let t = $t;
        let mut cases = 0usize;
        for base in 0..16i32 {
            for rot in 0..16usize {
                let a: [$elem; 16] = core::array::from_fn(|i| {
                    (base * 16 + ((i + rot) % 16) as i32 + <$elem>::MIN as i32) as $elem
                });
                let va = $ty::<$Tok>::from_array(t, a);
                assert_eq!(
                    va.widen_low().to_array(),
                    want_widen!(a, $elem, $delem, 8, 0),
                    "{} exhaustive widen_low",
                    stringify!($ty)
                );
                assert_eq!(
                    va.widen_high().to_array(),
                    want_widen!(a, $elem, $delem, 8, 8),
                    "{} exhaustive widen_high",
                    stringify!($ty)
                );
                cases += 2;
            }
        }
        cases
    }};
}

/// Every one of the 65 536 source values through both halves of a 16-bit
/// widening (each value exactly once; the lane mapping itself is pinned by
/// [`widen_lane_order`]).
macro_rules! exhaustive_widen_16 {
    ($Tok:ty, $t:expr, $ty:ident, $elem:ty, $delem:ty) => {{
        let t = $t;
        let mut cases = 0usize;
        for base in 0..8192i32 {
            let a: [$elem; 8] =
                core::array::from_fn(|i| (base * 8 + i as i32 + <$elem>::MIN as i32) as $elem);
            let va = $ty::<$Tok>::from_array(t, a);
            assert_eq!(
                va.widen_low().to_array(),
                want_widen!(a, $elem, $delem, 4, 0),
                "{} exhaustive widen_low",
                stringify!($ty)
            );
            assert_eq!(
                va.widen_high().to_array(),
                want_widen!(a, $elem, $delem, 4, 4),
                "{} exhaustive widen_high",
                stringify!($ty)
            );
            cases += 2;
        }
        cases
    }};
}

/// Every one of the 65 536 `i16` source values through both flavours of the
/// 16 -> 8 narrowing, in both operand positions.
macro_rules! exhaustive_narrow_i16 {
    ($Tok:ty, $t:expr) => {{
        let t = $t;
        let mut cases = 0usize;
        for base in 0..8192i32 {
            let a: [i16; 8] = core::array::from_fn(|i| (base * 8 + i as i32 - 32768) as i16);
            // `b` walks the domain in the opposite direction, so every value
            // is exercised in the second operand as well as the first.
            let b: [i16; 8] = core::array::from_fn(|i| {
                (32767 - (base * 8 + i as i32)).clamp(-32768, 32767) as i16
            });
            let va = i16x8::<$Tok>::from_array(t, a);
            let vb = i16x8::<$Tok>::from_array(t, b);
            assert_eq!(
                va.narrow_saturating_i8(vb).to_array(),
                want_narrow!(a, b, i16, i8, 8, 16, i8::MIN as i16, i8::MAX as i16),
                "i16x8 exhaustive narrow_saturating_i8"
            );
            assert_eq!(
                va.narrow_saturating_u8(vb).to_array(),
                want_narrow!(a, b, i16, u8, 8, 16, 0, u8::MAX as i16),
                "i16x8 exhaustive narrow_saturating_u8"
            );
            cases += 2;
        }
        cases
    }};
}

// ============================================================================
// Per-backend runners
// ============================================================================

/// The 128-bit and 2x-polyfill widths.
// Scalar references deliberately compute products and differences in i64;
// truncating only the final result pins modulo-2^32 semantics independently.
macro_rules! check_integer_ops {
    ($Tok:ty, $t:expr, $I:ident, $Acc:ident, $U:ident, $ni:expr, $nu:expr) => {{
        let t = $t;
        let mut comparisons = 0usize;
        let edges = [i16::MIN, i16::MAX, -1, 0, 1, -16384, 16384, 12345];
        for seed in 0..512usize {
            let a: [i16; $ni] = core::array::from_fn(|i| {
                if seed == 0 {
                    i16::MIN
                } else if seed < 65 {
                    edges[(i + seed) % edges.len()]
                } else {
                    (seed.wrapping_mul(19873).wrapping_add(i * 971)) as i16
                }
            });
            let b: [i16; $ni] = core::array::from_fn(|i| {
                if seed == 0 {
                    i16::MIN
                } else if seed < 65 {
                    edges[(i * 3 + seed / 8) % edges.len()]
                } else {
                    (seed.wrapping_mul(32143).wrapping_add(i * 11213)) as i16
                }
            });
            let accum: [i32; $ni / 2] = core::array::from_fn(|i| match (i + seed) % 3 {
                0 => i32::MIN,
                1 => i32::MAX,
                _ => (seed * 19873 + i) as i32,
            });
            let va = $I::<$Tok>::from_array(t, a);
            let vb = $I::<$Tok>::from_array(t, b);
            let dot = va.madd_adjacent(vb).to_array();
            let sub = va
                .msub_adjacent(vb, $Acc::<$Tok>::from_array(t, accum))
                .to_array();
            let diff = va.abs_diff(vb).to_array();
            for k in 0..$ni / 2 {
                let exact =
                    a[2 * k] as i64 * b[2 * k] as i64 + a[2 * k + 1] as i64 * b[2 * k + 1] as i64;
                assert_eq!(dot[k], exact as i32, "dot lane {k}, seed {seed}");
                assert_eq!(
                    sub[k],
                    (accum[k] as i64 - exact) as i32,
                    "msub lane {k}, seed {seed}"
                );
                comparisons += 2;
            }
            for k in 0..$ni {
                assert_eq!(
                    diff[k],
                    (a[k] as i64 - b[k] as i64).unsigned_abs() as u16,
                    "abs_diff lane {k}, seed {seed}"
                );
                comparisons += 1;
            }
        }
        // Every ordered pair of byte values, lane-distinct rhs to catch shuffles.
        for a in 0..=255u16 {
            for start in (0..256usize).step_by($nu) {
                let lhs = [a as u8; $nu];
                let rhs: [u8; $nu] = core::array::from_fn(|i| (start + i) as u8);
                let value = $U::<$Tok>::from_array(t, lhs).abs_diff($U::<$Tok>::from_array(t, rhs));
                let result = value.to_array();
                let mut sum = 0u32;
                for k in 0..$nu {
                    let expected = (lhs[k] as i32 - rhs[k] as i32).unsigned_abs();
                    assert_eq!(
                        result[k] as u32, expected,
                        "byte pair ({a}, {}) at lane {k}",
                        rhs[k]
                    );
                    sum += expected;
                    comparisons += 1;
                }
                assert_eq!(value.reduce_add_u32(), sum);
                assert_eq!(
                    $U::<$Tok>::from_array(t, lhs).sum_abs_diff($U::<$Tok>::from_array(t, rhs)),
                    sum
                );
            }
        }
        assert_eq!($U::<$Tok>::splat(t, 255).reduce_add_u32(), 255 * $nu as u32);
        assert_eq!(
            $I::<$Tok>::splat(t, i16::MIN)
                .abs_diff($I::<$Tok>::splat(t, i16::MAX))
                .to_array(),
            [u16::MAX; $ni]
        );
        assert!(
            comparisons >= 65_536 + 1024 * $ni,
            "integer primitive cases were skipped"
        );
        comparisons
    }};
}

macro_rules! run_all {
    ($Tok:ty, $t:expr) => {{
        let t = $t;
        let mut n = 0usize;
        // Widening, 128-bit.
        n += check_widen!($Tok, t, u8x16, u8, u16, 16, 8);
        n += check_widen!($Tok, t, i8x16, i8, i16, 16, 8);
        n += check_widen!($Tok, t, u16x8, u16, u32, 8, 4);
        n += check_widen!($Tok, t, i16x8, i16, i32, 8, 4);
        // Widening, 256-bit.
        n += check_widen!($Tok, t, u8x32, u8, u16, 32, 16);
        n += check_widen!($Tok, t, i8x32, i8, i16, 32, 16);
        n += check_widen!($Tok, t, u16x16, u16, u32, 16, 8);
        n += check_widen!($Tok, t, i16x16, i16, i32, 16, 8);
        // Narrowing, 128-bit.
        n += check_narrow!(
            $Tok,
            t,
            i16x8,
            i16,
            narrow_saturating_i8,
            i8,
            narrow_saturating_u8,
            u8,
            8,
            16
        );
        n += check_narrow!(
            $Tok,
            t,
            i32x4,
            i32,
            narrow_saturating_i16,
            i16,
            narrow_saturating_u16,
            u16,
            4,
            8
        );
        // Narrowing, 256-bit — the width where AVX2's pack lane order bites.
        n += check_narrow!(
            $Tok,
            t,
            i16x16,
            i16,
            narrow_saturating_i8,
            i8,
            narrow_saturating_u8,
            u8,
            16,
            32
        );
        n += check_narrow!(
            $Tok,
            t,
            i32x8,
            i32,
            narrow_saturating_i16,
            i16,
            narrow_saturating_u16,
            u16,
            8,
            16
        );
        // Exhaustive value sweeps.
        n += exhaustive_widen_8!($Tok, t, u8x16, u8, u16);
        n += exhaustive_widen_8!($Tok, t, i8x16, i8, i16);
        n += exhaustive_widen_16!($Tok, t, u16x8, u16, u32);
        n += exhaustive_widen_16!($Tok, t, i16x8, i16, i32);
        n += exhaustive_narrow_i16!($Tok, t);
        n += check_integer_ops!($Tok, t, i16x8, i32x4, u8x16, 8, 16);
        n += check_integer_ops!($Tok, t, i16x16, i32x8, u8x32, 16, 32);
        n
    }};
}

/// The 512-bit widths: a 4x/2x polyfill on most backends, native AVX-512 on
/// `X64V4Token`/`X64V4xToken`.
#[cfg(feature = "w512")]
macro_rules! run_all_512 {
    ($Tok:ty, $t:expr) => {{
        let t = $t;
        let mut n = 0usize;
        n += check_widen!($Tok, t, u8x64, u8, u16, 64, 32);
        n += check_widen!($Tok, t, i8x64, i8, i16, 64, 32);
        n += check_widen!($Tok, t, u16x32, u16, u32, 32, 16);
        n += check_widen!($Tok, t, i16x32, i16, i32, 32, 16);
        n += check_narrow!(
            $Tok,
            t,
            i16x32,
            i16,
            narrow_saturating_i8,
            i8,
            narrow_saturating_u8,
            u8,
            32,
            64
        );
        n += check_narrow!(
            $Tok,
            t,
            i32x16,
            i32,
            narrow_saturating_i16,
            i16,
            narrow_saturating_u16,
            u16,
            16,
            32
        );
        n += check_integer_ops!($Tok, t, i16x32, i32x16, u8x64, 32, 64);
        n
    }};
}

#[cfg(not(feature = "w512"))]
macro_rules! run_all_512 {
    ($Tok:ty, $t:expr) => {{
        let _ = $t;
        0usize
    }};
}

/// Floors for the "did the arm actually run" assertions. Deliberately close to
/// the real counts so a macro that expands to fewer cases trips them.
const MIN_CASES_W128: usize = 50_000;
#[cfg(feature = "w512")]
const MIN_CASES_W512: usize = 500;
#[cfg(not(feature = "w512"))]
const MIN_CASES_W512: usize = 0;

#[test]
fn scalar_backend() {
    let t = ScalarToken::summon().expect("ScalarToken must always summon");
    let n = run_all!(ScalarToken, t) + run_all_512!(ScalarToken, t);
    assert!(
        n >= MIN_CASES_W128 + MIN_CASES_W512,
        "scalar arm ran only {n} comparisons"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x64v3_backend() {
    // X64V3Token is the x86-64-v3 level: AVX2 is the discriminating feature.
    #[cfg(feature = "std")]
    let host_has_v3 = std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("sse4.2");
    match archmage::X64V3Token::summon() {
        Some(t) => {
            let n = run_all!(archmage::X64V3Token, t) + run_all_512!(archmage::X64V3Token, t);
            assert!(n >= MIN_CASES_W128, "v3 arm ran only {n} comparisons");
        }
        None => {
            #[cfg(feature = "std")]
            assert!(
                !host_has_v3,
                "the host reports AVX2+FMA+SSE4.2 but X64V3Token did not summon — \
             the v3 arm would have been silently skipped"
            );
        }
    }
}

/// `X64V4Token` natively backs only the 512-bit widths; 128/256-bit work on a
/// V4 machine goes through a `.v3()` downcast, so only the 512 matrix runs.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn x64v4_backend() {
    #[cfg(feature = "std")]
    let host_has_v4 = std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512cd");
    match archmage::X64V4Token::summon() {
        Some(t) => {
            let n = run_all_512!(archmage::X64V4Token, t);
            assert!(n >= MIN_CASES_W512, "v4 arm ran only {n} comparisons");
        }
        None => {
            #[cfg(feature = "std")]
            assert!(
                !host_has_v4,
                "the host reports the AVX-512 F/BW/VL/DQ/CD set but X64V4Token did \
             not summon — the v4 arm would have been silently skipped"
            );
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn x64v4x_backend() {
    match archmage::X64V4xToken::summon() {
        Some(t) => {
            let n = run_all_512!(archmage::X64V4xToken, t);
            assert!(n >= MIN_CASES_W512, "v4x arm ran only {n} comparisons");
        }
        None => {
            // v4x adds VBMI/VBMI2/VNNI/BITALG/IFMA/VPOPCNTDQ/GFNI/VAES on top
            // of v4; absence is ordinary, so there is nothing to assert beyond
            // "the v4 arm above covers the shared 512-bit paths".
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_backend() {
    // AdvSIMD is baseline on aarch64: a None here is a defect, not a skip.
    let t = archmage::NeonToken::summon().expect("NEON is baseline on aarch64");
    let n = run_all!(archmage::NeonToken, t) + run_all_512!(archmage::NeonToken, t);
    assert!(
        n >= MIN_CASES_W128 + MIN_CASES_W512,
        "NEON arm ran only {n} comparisons"
    );
}

#[cfg(target_arch = "wasm32")]
#[test]
fn wasm128_backend() {
    let built_with_simd128 = cfg!(target_feature = "simd128");
    match archmage::Wasm128Token::summon() {
        Some(t) => {
            let n = run_all!(archmage::Wasm128Token, t) + run_all_512!(archmage::Wasm128Token, t);
            assert!(n >= MIN_CASES_W128, "wasm128 arm ran only {n} comparisons");
        }
        None => {
            #[cfg(feature = "std")]
            assert!(
                !built_with_simd128,
                "built with target-feature=+simd128 but Wasm128Token did not summon \
             — the wasm arm would have been silently skipped"
            );
        }
    }
}

// The motivating use case: one token-specialized body combines checked loads,
// adjacent products and widening SAD, without per-ISA user implementations.
#[archmage::magetypes(define(i16x8, i32x4, u8x16), v3, neon, wasm128, scalar)]
fn integer_kernel(token: Token, a: &[i16; 8], b: &[i16; 8], bytes: &[u8; 16]) -> ([i32; 4], u32) {
    let a = i16x8::load(token, a);
    let b = i16x8::load(token, b);
    let result = a.msub_adjacent(b, i32x4::splat(token, 10));
    let sad = u8x16::load(token, bytes).sum_abs_diff(u8x16::splat(token, 255));
    (result.to_array(), sad)
}

#[test]
fn integer_primitives_in_one_magetypes_body() {
    let a = [1, 2, 3, 4, 5, 6, 7, 8];
    let b = [2; 8];
    let bytes = [1; 16];
    let expected = ([4, -4, -12, -20], 16 * 254);
    assert_eq!(integer_kernel_scalar(ScalarToken, &a, &b, &bytes), expected);
    assert_eq!(
        archmage::incant!(integer_kernel(&a, &b, &bytes), [v3, neon, wasm128, scalar]),
        expected
    );
}

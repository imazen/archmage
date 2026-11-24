//! Lossless `const` `bf16` (`1 sign + 8 exp + 7 mantissa`) packing and expansion
//! for custom option range/clamp bounds inside `const CFG: u128`.

/// Largest finite `f32` magnitude representable in `bf16` (`0x7f7f_0000`, `~3.3895314e38`).
pub const MAX_BF16: f32 = f32::from_bits(0x7f7f_0000);

/// Largest finite `bf16` magnitude strictly below `f32::MAX / 3.0` (`0x7eaa_0000`, `~1.129842e38`,
/// below `f32::MAX / 3.0 = 0x7eaa_aaaa`), used as the upper magnitude limit for custom
/// [`crate::cbrtf::CbrtfOptions::input_range`] bounds.
pub const MAX_CBRT_BF16: f32 = f32::from_bits(0x7eaa_0000);

const _: () = {
    assert!(MAX_CBRT_BF16 < f32::MAX / 3.0);
    assert!(expand_bf16(floor_bf16(MAX_CBRT_BF16) + 1) > f32::MAX / 3.0);
};

/// Expand a `bf16` bit pattern (`u16`) to an exact `f32` in `const` context (`1` bit-shift).
#[inline(always)]
pub const fn expand_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Largest `bf16` value `<= x` (canonicalizing `±0.0` to `+0.0 = 0x0000`).
///
/// Caller must ensure `!x.is_nan()` and `|x| <= MAX_BF16`.
#[inline(always)]
pub const fn floor_bf16(x: f32) -> u16 {
    if x == 0.0 {
        return 0;
    }
    let b = x.to_bits();
    if (b & 0x8000_0000) == 0 || (b & 0xffff) == 0 {
        (b >> 16) as u16
    } else {
        // Negative sign-magnitude: rounding toward -inf increments the magnitude.
        // Because `x >= -MAX_BF16` (`0xff7f_0000`) and `(b & 0xffff) != 0`, the top
        // 16 bits are at most `0xff7e`, so `+ 1` never overflows into `-inf` (`0xff80`).
        ((b >> 16) as u16) + 1
    }
}

/// Smallest `bf16` value `>= x` (canonicalizing `±0.0` to `+0.0 = 0x0000`).
///
/// Caller must ensure `!x.is_nan()` and `|x| <= MAX_BF16`.
#[inline(always)]
pub const fn ceil_bf16(x: f32) -> u16 {
    let neg = floor_bf16(-x);
    if neg == 0 { 0 } else { neg ^ 0x8000 }
}

/// Decode a `u16` `bf16` payload from a `u128` option word, rejecting encoded
/// `-0.0` (`0x8000`) and non-finite `±inf`/`NaN` (`exp == 0xff`).
#[inline(always)]
pub(crate) const fn decode_bf16_finite(bits: u16) -> Option<f32> {
    if bits == 0x8000 || (bits & 0x7f80) == 0x7f80 {
        return None;
    }
    Some(expand_bf16(bits))
}

/// Decode a bounded finite `[lo, hi]` bf16 pair with `min_lo <= lo <= hi <= max_hi`.
#[inline(always)]
pub(crate) const fn decode_bf16_range(
    lo_u16: u16,
    hi_u16: u16,
    min_lo: f32,
    max_hi: f32,
) -> Option<(u16, u16)> {
    let Some(lo) = decode_bf16_finite(lo_u16) else {
        return None;
    };
    let Some(hi) = decode_bf16_finite(hi_u16) else {
        return None;
    };
    if lo >= min_lo && lo <= hi && hi <= max_hi {
        Some((lo_u16, hi_u16))
    } else {
        None
    }
}

/// Decode an optional bounded bf16 range from a `u128` option word, rejecting
/// nonzero payload bits when `present` is `false`.
#[inline(always)]
pub(crate) const fn decode_optional_bf16_range(
    present: bool,
    lo_u16: u16,
    hi_u16: u16,
    min_lo: f32,
    max_hi: f32,
) -> Option<Option<(u16, u16)>> {
    if present {
        match decode_bf16_range(lo_u16, hi_u16, min_lo, max_hi) {
            Some(pair) => Some(Some(pair)),
            None => None,
        }
    } else if lo_u16 == 0 && hi_u16 == 0 {
        Some(None)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_boundaries_round_trip_losslessly() {
        for exact in [
            0.0_f32,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1.0 / 65_536.0,
            0.25,
            0.5,
            107.0 / 128.0, // PQ c1 = 0.8359375
            1.0,
            -1.0,
            2.0,
            4.0,
            8.0,
            80.0,
            88.0,
            -87.0,
            -126.0,
            127.0,
            127.5,
            65_536.0,
            MAX_CBRT_BF16,
            -MAX_CBRT_BF16,
            MAX_BF16,
            -MAX_BF16,
        ] {
            let lo = floor_bf16(exact);
            let hi = ceil_bf16(exact);
            assert_eq!(lo, hi, "exact bf16 {exact} diverged");
            assert_eq!(expand_bf16(lo).to_bits(), exact.to_bits());
            assert_eq!(decode_bf16_finite(lo), Some(exact));
        }
        assert_eq!(floor_bf16(-0.0), 0);
        assert_eq!(ceil_bf16(-0.0), 0);
        assert_eq!(decode_bf16_finite(0x8000), None);
        assert_eq!(decode_bf16_finite(0x7f80), None);
        assert_eq!(decode_bf16_finite(0xff80), None);
        assert_eq!(decode_bf16_finite(0x7fc0), None);
        assert!(core::hint::black_box(MAX_CBRT_BF16) < f32::MAX / 3.0);
        assert!(expand_bf16(floor_bf16(MAX_CBRT_BF16) + 1) > f32::MAX / 3.0);
    }

    #[test]
    fn outward_rounding_brackets_inexact_inputs_and_carries_across_exponents() {
        // A negative encoding walks toward smaller values as its magnitude
        // grows. Counting from zero gives a single ordered run of finite codes.
        let rank = |bits: u16| {
            let magnitude = i32::from(bits & 0x7fff);
            if bits & 0x8000 != 0 {
                -magnitude
            } else {
                magnitude
            }
        };
        for x in [
            0.01_f32,
            -0.01,
            0.346_573_56,
            -0.346_573_56,
            0.99,
            -0.999_999_94, // 0xbf7f_ffff -> floor carries into -1.0 (0xbf80)
            2.2,
            2.4,
            1.0 / 2.4,
            127.49,
            f32::from_bits(1),
            -f32::from_bits(1),
        ] {
            let lo_bits = floor_bf16(x);
            let hi_bits = ceil_bf16(x);
            let lo = expand_bf16(lo_bits);
            let hi = expand_bf16(hi_bits);
            assert!(lo < x && x < hi, "{x} is outside ({lo}, {hi})");
            assert_eq!(rank(hi_bits) - rank(lo_bits), 1, "bounds of {x}");
            assert_ne!(lo_bits, 0x8000, "floor of {x} encodes negative zero");
            assert_ne!(hi_bits, 0x8000, "ceil of {x} encodes negative zero");
        }
        assert_eq!(expand_bf16(floor_bf16(-0.999_999_94)), -1.0);
        assert_eq!(expand_bf16(ceil_bf16(127.49)), 127.5);
    }
}

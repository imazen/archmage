use crate::numerics::{
    ExtremaDomain, ExtremaOptions, HalfConvert, I32_MAX, I32_MIN, IntConvert, NumericsError,
    NumericsErrorKind, RecipOptions, RecipPrecision, U32_MAX,
};

macro_rules! to_int {
    ($name:ident, $checked:ident, $output:ty, $lo:expr, $hi:expr, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Configure with [`IntConvert`](crate::numerics::IntConvert).
        #[inline(always)]
        pub fn $name<const CFG: u32>(x: f32) -> $output {
            let opts = const { IntConvert::decode(CFG) };
            let rounded = if opts.rounds() {
                if const { $lo == 0.0 && $hi <= 65535.0 } {
                    crate::nostd_math::round_to_small_uint(x, $hi)
                } else {
                    super::round_f32(x)
                }
            } else {
                x
            };
            rounded as $output
        }

        /// Check the input bounds and convert one lane using the encoded integer policy.
        #[inline(always)]
        pub fn $checked<const CFG: u32>(x: f32) -> Result<$output, NumericsError> {
            let opts = const { IntConvert::decode(CFG) };
            opts.check(|| [x], $lo, $hi)?;
            Ok($name::<CFG>(x))
        }
    };
}

to_int!(
    to_u8_with,
    try_to_u8_with,
    u8,
    0.0,
    u8::MAX as f32,
    "Convert one lane to u8 with the vector conversion policy."
);
to_int!(
    to_u16_with,
    try_to_u16_with,
    u16,
    0.0,
    u16::MAX as f32,
    "Convert one lane to u16 with the vector conversion policy."
);
to_int!(
    to_i32_with,
    try_to_i32_with,
    i32,
    I32_MIN,
    I32_MAX,
    "Convert one lane to i32 with the vector conversion policy."
);
to_int!(
    to_u32_with,
    try_to_u32_with,
    u32,
    0.0,
    U32_MAX,
    "Convert one lane to u32 with the vector conversion policy."
);

/// Widen an unsigned 32-bit integer, rounding ties to even.
#[inline(always)]
pub fn from_u32(x: u32) -> f32 {
    x as f32
}

/// Decode an IEEE binary16 bit pattern with the given half policy.
///
/// Configure with [`HalfConvert`](crate::numerics::HalfConvert).
#[inline]
pub fn from_f16_with<const CFG: u32>(bits: u16) -> f32 {
    let opts = const { HalfConvert::decode(CFG) };
    let sign = u32::from(bits & 0x8000) << 16;
    let mag = u32::from(bits & 0x7fff);
    let normal = if mag < 0x400 {
        if opts.flushes() || mag == 0 {
            0
        } else {
            let shift = mag.leading_zeros() - 21;
            ((113 - shift) << 23) | (((mag << shift) & 0x3ff) << 13)
        }
    } else if !opts.is_finite() && mag >= 0x7c00 {
        0x7f80_0000 | ((mag & 0x3ff) << 13)
    } else {
        (mag << 13) + (112 << 23)
    };
    f32::from_bits(sign | normal)
}

/// Encode binary16 with the given half policy.
///
/// Configure with [`HalfConvert`](crate::numerics::HalfConvert).
#[inline]
pub fn to_f16_bits_with<const CFG: u32>(x: f32) -> u16 {
    let opts = const { HalfConvert::decode(CFG) };
    let bits = x.to_bits();
    let sign = (bits >> 16) as u16 & 0x8000;
    let magnitude = bits & 0x7fff_ffff;
    let exponent = (magnitude >> 23) as i32 - 127;
    if !opts.is_finite() && magnitude >= 0x7f80_0000 {
        return sign | if x.is_nan() { 0x7e00 } else { 0x7c00 };
    }

    let round = |word: u32, shift: u32| {
        let bias = if opts.rounds() {
            ((1 << (shift - 1)) - 1) + ((word >> shift) & 1)
        } else {
            0
        };
        (word + bias) >> shift
    };
    let mantissa = magnitude & 0x007f_ffff;
    let half = if exponent > 15 {
        if opts.rounds() { 0x7c00 } else { 0x7bff }
    } else if exponent >= -14 {
        (((exponent + 15) as u32) << 10) + round(mantissa, 13)
    } else if exponent >= -25 {
        let rounded = round(0x0080_0000 | mantissa, (-exponent - 1) as u32);
        if opts.flushes() && rounded < 0x400 {
            0
        } else {
            rounded
        }
    } else {
        0
    };
    sign | half as u16
}

/// Check the input domain and decode one binary16 value.
#[inline(always)]
pub fn try_from_f16_with<const CFG: u32>(bits: u16) -> Result<f32, NumericsError> {
    let opts = const { HalfConvert::decode(CFG) };
    opts.check_f16(&[bits])?;
    Ok(from_f16_with::<CFG>(bits))
}

macro_rules! checked_half {
    ($name:ident, $encode:ident, $doc:literal) => {
        #[doc = $doc]
        #[inline(always)]
        pub fn $name<const CFG: u32>(x: f32) -> Result<u16, NumericsError> {
            let opts = const { HalfConvert::decode(CFG) };
            opts.check_float(|| [x])?;
            Ok($encode::<CFG>(x))
        }
    };
}

checked_half!(
    try_to_f16_bits_with,
    to_f16_bits_with,
    "Check the input domain and encode one binary16 value."
);
checked_half!(
    try_to_bf16_with,
    to_bf16_with,
    "Check the input domain and encode one bfloat16 value."
);

/// Decode bfloat16 by widening the bits in place.
#[inline(always)]
pub fn from_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Encode bfloat16 with the given half policy.
///
/// Configure with [`HalfConvert`](crate::numerics::HalfConvert).
#[inline(always)]
pub fn to_bf16_with<const CFG: u32>(x: f32) -> u16 {
    let opts = const { HalfConvert::decode(CFG) };
    let bits = x.to_bits();
    let rounded = if !opts.is_finite() && x.is_nan() {
        bits | 0x0040_0000
    } else if opts.rounds() {
        bits.wrapping_add(0x7fff + ((bits >> 16) & 1))
    } else {
        bits
    };
    let b = (rounded >> 16) as u16;
    if opts.flushes() && b & 0x7f80 == 0 {
        b & 0x8000
    } else {
        b
    }
}

macro_rules! scalar_float {
    ($f:ty, $min:ident, $max:ident, $clamp:ident,
     $min_ieee:ident, $max_ieee:ident, $recip:ident, $rsqrt:ident, $try_recip:ident, $try_rsqrt:ident, $sqrt:ident,
     $trunc:ident, $trunc_impl:ident,
     $nmadd_fused:ident, $madd_fused:ident, $nmadd_relaxed:ident, $madd_relaxed:ident, $backend:ident) => {
        /// Bit-preserving truncation toward zero, including the sign of zero.
        #[inline(always)]
        pub fn $trunc(x: $f) -> $f {
            crate::scalar_round::$trunc_impl(x)
        }

        /// Compute `c - a * b` with one rounding.
        #[inline(always)]
        pub fn $nmadd_fused(a: $f, b: $f, c: $f) -> $f {
            super::$madd_fused(-a, b, c)
        }

        /// Compute `c - a * b` with the token's multiply-add rounding.
        #[inline(always)]
        pub fn $nmadd_relaxed<T: super::$backend>(token: T, a: $f, b: $f, c: $f) -> $f {
            super::$madd_relaxed(token, -a, b, c)
        }

        /// Scalar reciprocal with the selected policy. Scalar has an exact seed.
        ///
        /// Configure with [`RecipOptions`](crate::numerics::RecipOptions).
        #[inline(always)]
        pub fn $recip<const CFG: u32>(x: $f) -> $f {
            let opts = const { RecipOptions::decode(CFG) };
            let r = 1.0 / x;
            if matches!(opts.selected_precision(), RecipPrecision::Nr1Ceil)
                && r.is_finite()
                && r != 0.0
            {
                r.next_up()
            } else {
                r
            }
        }

        /// Scalar reciprocal square root. Scalar has an exact seed.
        ///
        /// Configure with [`RecipOptions`](crate::numerics::RecipOptions).
        #[inline(always)]
        pub fn $rsqrt<const CFG: u32>(x: $f) -> $f {
            const {
                let _ = RecipOptions::decode(CFG);
            }
            1.0 / super::$sqrt(x)
        }

        /// Evaluate a scalar reciprocal with the same domain check as the vector operation.
        #[inline(always)]
        pub fn $try_recip<const CFG: u32>(x: $f) -> Result<$f, NumericsError> {
            if const { RecipOptions::decode(CFG).checks_normal_domain() }
                && !(<$f>::MIN_POSITIVE..1.0 / <$f>::MIN_POSITIVE).contains(&x.abs())
            {
                return Err(NumericsError {
                    kind: NumericsErrorKind::RecipDomain,
                    lane_mask: 1,
                });
            }
            Ok($recip::<CFG>(x))
        }

        /// Evaluate a scalar reciprocal square root with the same domain check as the vector operation.
        #[inline(always)]
        pub fn $try_rsqrt<const CFG: u32>(x: $f) -> Result<$f, NumericsError> {
            if const { RecipOptions::decode(CFG).checks_normal_domain() }
                && !(<$f>::MIN_POSITIVE..=<$f>::MAX).contains(&x)
            {
                return Err(NumericsError {
                    kind: NumericsErrorKind::RecipDomain,
                    lane_mask: 1,
                });
            }
            Ok($rsqrt::<CFG>(x))
        }

        /// Scalar minimum with the selected NaN policy.
        ///
        /// Configure with [`ExtremaOptions`](crate::numerics::ExtremaOptions).
        #[inline(always)]
        pub fn $min<const CFG: u32>(a: $f, b: $f) -> $f {
            match const { ExtremaOptions::decode(CFG) } {
                ExtremaDomain::PropagateNan => crate::scalar_extrema::$min_ieee(a, b),
                ExtremaDomain::DropNan => a.min(b),
                _ => {
                    if a < b {
                        a
                    } else {
                        b
                    }
                }
            }
        }

        /// Scalar maximum with the selected NaN policy.
        ///
        /// Configure with [`ExtremaOptions`](crate::numerics::ExtremaOptions).
        #[inline(always)]
        pub fn $max<const CFG: u32>(a: $f, b: $f) -> $f {
            match const { ExtremaOptions::decode(CFG) } {
                ExtremaDomain::PropagateNan => crate::scalar_extrema::$max_ieee(a, b),
                ExtremaDomain::DropNan => a.max(b),
                _ => {
                    if a > b {
                        a
                    } else {
                        b
                    }
                }
            }
        }

        /// Clamp a scalar with the selected NaN policy.
        ///
        /// Configure with [`ExtremaOptions`](crate::numerics::ExtremaOptions).
        #[inline(always)]
        pub fn $clamp<const CFG: u32>(x: $f, lo: $f, hi: $f) -> $f {
            $min::<CFG>($max::<CFG>(x, lo), hi)
        }
    };
}

scalar_float!(
    f32,
    min_f32_with,
    max_f32_with,
    clamp_f32_with,
    minimumf,
    maximumf,
    recip_f32_with,
    rsqrt_f32_with,
    try_recip_f32_with,
    try_rsqrt_f32_with,
    sqrt_f32,
    trunc_f32,
    truncf,
    mul_neg_add_fused_f32,
    mul_add_fused_f32,
    mul_neg_add_relaxed_f32,
    mul_add_relaxed_f32,
    F32ScalarBackend
);
scalar_float!(
    f64,
    min_f64_with,
    max_f64_with,
    clamp_f64_with,
    minimum,
    maximum,
    recip_f64_with,
    rsqrt_f64_with,
    try_recip_f64_with,
    try_rsqrt_f64_with,
    sqrt_f64,
    trunc_f64,
    trunc,
    mul_neg_add_fused_f64,
    mul_add_fused_f64,
    mul_neg_add_relaxed_f64,
    mul_add_relaxed_f64,
    F64ScalarBackend
);

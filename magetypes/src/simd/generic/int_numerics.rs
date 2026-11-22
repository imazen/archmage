use crate::numerics::{I32_MAX, I32_MIN, IntConvert, NumericsError, U32_MAX};

macro_rules! int_numerics {
    ($float:ident, $int:ident, $unsigned:ident, $convert:ident, $float_backend:ident, $unsigned_backend:ident) => {
        /// ### Signed integer conversion
        impl<T: crate::simd::backends::$convert> super::$float<T> {
            /// Truncate toward zero with the native conversion instruction.
            ///
            /// Exact when the truncated input fits in `i32`. Out-of-range and
            /// NaN inputs return an unspecified integer. The scalar remainder
            /// equivalent is [`crate::scalar::to_i32_fast`].
            #[inline(always)]
            pub fn to_i32_fast(self) -> super::$int<T> {
                self.to_i32_with::<{ IntConvert::new().trunc().in_range().bits() }>()
            }

            /// Convert to signed integers with the encoded rounding and bounds.
            ///
            /// Configure with [`IntConvert`](crate::numerics::IntConvert).
            #[inline(always)]
            pub fn to_i32_with<const CFG: u32>(self) -> super::$int<T> {
                let opts = const { IntConvert::decode(CFG) };
                if opts.saturates_all() {
                    let rounded = if opts.rounds() { self.round() } else { self };
                    rounded.to_i32_saturating()
                } else if opts.rounds() {
                    self.to_i32_round()
                } else {
                    let repr = T::convert_f32_to_i32_fast(self.1, self.0);
                    super::$int::from_repr_unchecked(self.1, repr)
                }
            }

            /// Check every lane against the encoded integer bounds before converting to i32.
            #[inline(always)]
            pub fn try_to_i32_with<const CFG: u32>(self) -> Result<super::$int<T>, NumericsError> {
                let opts = const { IntConvert::decode(CFG) };
                opts.check(|| self.to_array(), I32_MIN, I32_MAX)?;
                Ok(self.to_i32_with::<CFG>())
            }
        }

        /// ### Unsigned integer conversion
        impl<T: crate::simd::backends::$float_backend + crate::simd::backends::$unsigned_backend>
            super::$float<T>
        {
            /// Round finite inputs to nearest-even unsigned integers.
            /// Inputs must be in `[-2^31, 2^31)`; negative numbers saturate to zero.
            #[inline(always)]
            pub fn to_u32(self) -> super::$unsigned<T> {
                self.to_u32_with::<{ IntConvert::new().bits() }>()
            }

            /// Convert to unsigned integers with the encoded rounding and bounds.
            ///
            /// Configure with [`IntConvert`](crate::numerics::IntConvert).
            #[inline(always)]
            pub fn to_u32_with<const CFG: u32>(self) -> super::$unsigned<T> {
                const {
                    let _ = IntConvert::decode(CFG);
                }
                let repr = T::to_u32_lanes::<CFG>(self.1, self.0);
                super::$unsigned::from_repr_unchecked(self.1, crate::simd_storage::cast(repr))
            }

            /// Check every lane against the encoded integer bounds before converting to u32.
            #[inline(always)]
            pub fn try_to_u32_with<const CFG: u32>(
                self,
            ) -> Result<super::$unsigned<T>, NumericsError> {
                let opts = const { IntConvert::decode(CFG) };
                opts.check(|| self.to_array(), 0.0, U32_MAX)?;
                Ok(self.to_u32_with::<CFG>())
            }

            /// Widen unsigned integers, rounding ties to even.
            #[inline(always)]
            pub fn from_u32_t(token: T, lanes: super::$unsigned<T>) -> Self {
                Self(
                    T::from_u32_lanes(token, crate::simd_storage::cast(lanes.into_repr())),
                    token,
                )
            }
        }
    };
}

int_numerics!(
    f32x4,
    i32x4,
    u32x4,
    F32x4Convert,
    F32x4Backend,
    U32x4Backend
);
int_numerics!(
    f32x8,
    i32x8,
    u32x8,
    F32x8Convert,
    F32x8Backend,
    U32x8Backend
);
#[cfg(feature = "w512")]
int_numerics!(
    f32x16,
    i32x16,
    u32x16,
    F32x16Convert,
    F32x16Backend,
    U32x16Backend
);

use crate::IntConvert;

macro_rules! int_numerics {
    ($float:ident, $int:ident, $unsigned:ident, $convert:ident, $float_backend:ident, $unsigned_backend:ident) => {
        impl<T: crate::simd::backends::$convert> super::$float<T> {
            /// Convert to signed integers with the encoded rounding and bounds.
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
        }

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
            #[inline(always)]
            pub fn to_u32_with<const CFG: u32>(self) -> super::$unsigned<T> {
                const {
                    let _ = IntConvert::decode(CFG);
                }
                let repr = T::to_u32_lanes::<CFG>(self.1, self.0);
                super::$unsigned::from_repr_unchecked(self.1, crate::simd_storage::cast(repr))
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

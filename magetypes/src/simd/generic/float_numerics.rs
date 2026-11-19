use crate::{ExtremaDomain, ExtremaOptions, RecipOptions, RecipPrecision};

macro_rules! float_numerics {
    ($vector:ident, $backend:ident, $elem:ty, $min:ident, $max:ident) => {
        impl<T: crate::simd::backends::$backend> super::$vector<T> {
            /// Truncate each lane toward zero, retaining the sign of zero.
            #[must_use]
            #[inline(always)]
            pub fn trunc(self) -> Self {
                Self(T::trunc(self.1, self.0), self.1)
            }

            /// Compute `c - self * b` with native or separate rounding.
            #[must_use]
            #[inline(always)]
            pub fn mul_neg_add_relaxed(self, b: Self, c: Self) -> Self {
                Self(T::mul_neg_add_relaxed(self.1, self.0, b.0, c.0), self.1)
            }

            /// Compute `c - self * b` with one rounding.
            #[must_use]
            #[inline(always)]
            pub fn mul_neg_add_fused(self, b: Self, c: Self) -> Self {
                Self(T::mul_neg_add_fused(self.1, self.0, b.0, c.0), self.1)
            }

            /// Lane-wise reciprocal with a compile-time domain and precision.
            #[must_use]
            #[inline(always)]
            pub fn recip_with<const CFG: u32>(self) -> Self {
                let opts = const { RecipOptions::decode(CFG) };
                let precision = opts.selected_precision();
                let one = Self::splat_t(self.1, 1.0);
                if matches!(precision, RecipPrecision::Exact) {
                    return one / self;
                }

                let seed = self.rcp_approx();
                let mut result = if matches!(precision, RecipPrecision::Approx) {
                    seed
                } else {
                    let residual = self.mul_neg_add_relaxed(seed, one);
                    seed.mul_add_relaxed(residual, seed)
                };
                if opts.admits_all() {
                    // Estimates accept normal inputs with normal reciprocal
                    // intermediates. Use division on IEEE rails and tiny outputs.
                    let valid = self
                        .abs()
                        .simd_ge(Self::splat_t(self.1, <$elem>::MIN_POSITIVE))
                        & self
                            .abs()
                            .simd_lt(Self::splat_t(self.1, 1.0 / <$elem>::MIN_POSITIVE));
                    result = Self::blend(valid, result, one / self);
                }
                if matches!(precision, RecipPrecision::Nr1Ceil) {
                    let bits = result.to_array().map(|x| {
                        if x.is_finite() && x != 0.0 {
                            x.next_up()
                        } else {
                            x
                        }
                    });
                    result = Self::from_array_t(self.1, bits);
                }
                result
            }

            /// Lane-wise reciprocal square root with a compile-time policy.
            #[must_use]
            #[inline(always)]
            pub fn rsqrt_with<const CFG: u32>(self) -> Self {
                let opts = const { RecipOptions::decode(CFG) };
                let precision = opts.selected_precision();
                let one = Self::splat_t(self.1, 1.0);
                if matches!(precision, RecipPrecision::Exact) {
                    return one / self.sqrt();
                }

                let seed = self.rsqrt_approx();
                let result = if matches!(precision, RecipPrecision::Approx) {
                    seed
                } else {
                    let half = Self::splat_t(self.1, 0.5);
                    let half_seed = half * seed;
                    let residual =
                        (self * seed).mul_neg_add_relaxed(seed, Self::splat_t(self.1, 1.0));
                    half_seed.mul_add_relaxed(residual, seed)
                };
                if opts.admits_all() {
                    let valid = self.simd_ge(Self::splat_t(self.1, <$elem>::MIN_POSITIVE))
                        & self.simd_le(Self::splat_t(self.1, <$elem>::MAX));
                    Self::blend(valid, result, one / self.sqrt())
                } else {
                    result
                }
            }

            /// Lane-wise minimum under the selected NaN and signed-zero contract.
            #[must_use]
            #[inline(always)]
            pub fn min_with<const CFG: u32>(self, rhs: Self) -> Self {
                match const { ExtremaOptions::decode(CFG) } {
                    ExtremaDomain::DropNan => self.min(rhs),
                    ExtremaDomain::PropagateNan => self.minimum(rhs),
                    #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
                    ExtremaDomain::NonNanRhs => self.min(rhs),
                    _ => self.min_fast(rhs),
                }
            }

            /// Lane-wise maximum under the selected NaN and signed-zero contract.
            #[must_use]
            #[inline(always)]
            pub fn max_with<const CFG: u32>(self, rhs: Self) -> Self {
                match const { ExtremaOptions::decode(CFG) } {
                    ExtremaDomain::DropNan => self.max(rhs),
                    ExtremaDomain::PropagateNan => self.maximum(rhs),
                    #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
                    ExtremaDomain::NonNanRhs => self.max(rhs),
                    _ => self.max_fast(rhs),
                }
            }

            /// Clamp each lane with the selected extrema policy.
            #[must_use]
            #[inline(always)]
            pub fn clamp_with<const CFG: u32>(self, lo: Self, hi: Self) -> Self {
                self.max_with::<CFG>(lo).min_with::<CFG>(hi)
            }

            /// Reduce lanes to a minimum with the selected NaN policy.
            /// `NonNanRhs` requires every lane to be non-NaN for a reduction.
            #[inline(always)]
            pub fn reduce_min_with<const CFG: u32>(self) -> $elem {
                match const { ExtremaOptions::decode(CFG) } {
                    ExtremaDomain::PropagateNan => self
                        .to_array()
                        .into_iter()
                        .fold(<$elem>::INFINITY, crate::scalar::$min),
                    ExtremaDomain::DropNan => {
                        let nan = self.simd_ne(self);
                        let identity = <$elem>::INFINITY;
                        let number =
                            Self::blend(nan, Self::splat_t(self.1, identity), self).reduce_min();
                        if number == identity && self.to_array().into_iter().all(<$elem>::is_nan) {
                            <$elem>::NAN
                        } else {
                            number
                        }
                    }
                    _ => self.reduce_min(),
                }
            }

            /// Reduce lanes to a maximum with the selected NaN policy.
            /// `NonNanRhs` requires every lane to be non-NaN for a reduction.
            #[inline(always)]
            pub fn reduce_max_with<const CFG: u32>(self) -> $elem {
                match const { ExtremaOptions::decode(CFG) } {
                    ExtremaDomain::PropagateNan => self
                        .to_array()
                        .into_iter()
                        .fold(<$elem>::NEG_INFINITY, crate::scalar::$max),
                    ExtremaDomain::DropNan => {
                        let nan = self.simd_ne(self);
                        let identity = <$elem>::NEG_INFINITY;
                        let number =
                            Self::blend(nan, Self::splat_t(self.1, identity), self).reduce_max();
                        if number == identity && self.to_array().into_iter().all(<$elem>::is_nan) {
                            <$elem>::NAN
                        } else {
                            number
                        }
                    }
                    _ => self.reduce_max(),
                }
            }
        }
    };
}

float_numerics!(f32x4, F32x4Backend, f32, minimum_f32, maximum_f32);
float_numerics!(f32x8, F32x8Backend, f32, minimum_f32, maximum_f32);
#[cfg(feature = "w512")]
float_numerics!(f32x16, F32x16Backend, f32, minimum_f32, maximum_f32);
float_numerics!(f64x2, F64x2Backend, f64, minimum_f64, maximum_f64);
float_numerics!(f64x4, F64x4Backend, f64, minimum_f64, maximum_f64);
#[cfg(feature = "w512")]
float_numerics!(f64x8, F64x8Backend, f64, minimum_f64, maximum_f64);

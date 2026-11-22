use crate::numerics::{ExtremaDomain, ExtremaOptions, RecipOptions, RecipPrecision};

macro_rules! float_numerics {
    ($vector:ident, $backend:ident, $elem:ty, $min:ident, $max:ident) => {
        /// ### Floating-point numerics
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
            #[doc(alias = "fnmadd")]
            pub fn mul_neg_add_relaxed(self, b: Self, c: Self) -> Self {
                Self(T::mul_neg_add_relaxed(self.1, self.0, b.0, c.0), self.1)
            }

            /// Compute `c - self * b` with one rounding.
            #[must_use]
            #[inline(always)]
            #[doc(alias = "fnmadd")]
            pub fn mul_neg_add_fused(self, b: Self, c: Self) -> Self {
                Self(T::mul_neg_add_fused(self.1, self.0, b.0, c.0), self.1)
            }

            /// Lane-wise reciprocal with a compile-time domain and precision.
            ///
            /// Configure with [`RecipOptions`](crate::numerics::RecipOptions).
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
                let result = if matches!(precision, RecipPrecision::Approx) {
                    seed
                } else {
                    let residual = self.mul_neg_add_relaxed(seed, one);
                    let refined = seed.mul_add_relaxed(residual, seed);
                    if core::mem::size_of::<$elem>() == 8 {
                        let residual2 = self.mul_neg_add_relaxed(refined, one);
                        refined.mul_add_relaxed(residual2, refined)
                    } else {
                        refined
                    }
                };
                let bias = |r: Self| {
                    r.abs()
                        .mul_add_relaxed(Self::splat_t(self.1, 2.0 * <$elem>::EPSILON), r)
                };
                if opts.admits_all() {
                    // Estimates accept normal inputs below 1 / MIN_POSITIVE.
                    // Use division on IEEE rails, subnormals, and ±(1 / MIN_POSITIVE).
                    let magnitude = self.abs();
                    let valid = magnitude.simd_ge(Self::splat_t(self.1, <$elem>::MIN_POSITIVE))
                        & magnitude.simd_lt(Self::splat_t(self.1, 1.0 / <$elem>::MIN_POSITIVE));
                    let exact = one / self;
                    if matches!(precision, RecipPrecision::Nr1Ceil) {
                        // Relative bias can round to zero for a subnormal reciprocal.
                        // Once division has rounded, one representable step suffices.
                        let exact = Self::from_array_t(
                            self.1,
                            exact.to_array().map(|lane| {
                                if lane.is_finite() && lane != 0.0 {
                                    lane.next_up()
                                } else {
                                    lane
                                }
                            }),
                        );
                        let finite_nz = result.abs().simd_gt(Self::splat_t(self.1, 0.0))
                            & result
                                .abs()
                                .simd_lt(Self::splat_t(self.1, <$elem>::INFINITY));
                        let estimate = Self::blend(finite_nz, bias(result), result);
                        return Self::blend(valid, estimate, exact);
                    }
                    return Self::blend(valid, result, exact);
                }
                if matches!(precision, RecipPrecision::Nr1Ceil) {
                    bias(result)
                } else {
                    result
                }
            }

            /// Lane-wise reciprocal square root with a compile-time policy.
            ///
            /// Configure with [`RecipOptions`](crate::numerics::RecipOptions).
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
                    let residual = (self * seed).mul_neg_add_relaxed(seed, one);
                    let refined = half_seed.mul_add_relaxed(residual, seed);
                    if core::mem::size_of::<$elem>() == 8 {
                        let half_ref = half * refined;
                        let residual2 = (self * refined).mul_neg_add_relaxed(refined, one);
                        half_ref.mul_add_relaxed(residual2, refined)
                    } else {
                        refined
                    }
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
            ///
            /// Configure with [`ExtremaOptions`](crate::numerics::ExtremaOptions).
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
            ///
            /// Configure with [`ExtremaOptions`](crate::numerics::ExtremaOptions).
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
            ///
            /// Configure with [`ExtremaOptions`](crate::numerics::ExtremaOptions).
            #[must_use]
            #[inline(always)]
            pub fn clamp_with<const CFG: u32>(self, lo: Self, hi: Self) -> Self {
                self.max_with::<CFG>(lo).min_with::<CFG>(hi)
            }

            /// Reduce lanes to a minimum with the selected NaN policy.
            /// `NonNanRhs` requires every lane to be non-NaN for a reduction.
            ///
            /// Configure with [`ExtremaOptions`](crate::numerics::ExtremaOptions).
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
            ///
            /// Configure with [`ExtremaOptions`](crate::numerics::ExtremaOptions).
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

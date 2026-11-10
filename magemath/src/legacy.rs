//! Legacy Magetypes transcendental contracts, without inherent-method calls.
//! Backend multiply-add rounding and exceptional-value behavior are preserved.
//! This is distinct from portable/relaxed, whose established special handling
//! remains unchanged. Width16 requires w512; avx512 implies w512.
pub mod kernels {
    use magetypes::simd::backends::F32x8Convert;
    use magetypes::simd::generic::f32x8;
    /// Legacy `log2_lowp` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn log2_lowp<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::log2_lowp(t, x)
    }
    /// Legacy `log2_lowp_unchecked` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn log2_lowp_unchecked<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::log2_lowp_unchecked(t, x)
    }
    /// Legacy `exp2_lowp` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn exp2_lowp<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::exp2_lowp(t, x)
    }
    /// Legacy `exp2_lowp_unchecked` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn exp2_lowp_unchecked<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::exp2_lowp_unchecked(t, x)
    }
    /// Legacy `ln_lowp` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn ln_lowp<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::ln_lowp(t, x)
    }
    /// Legacy `ln_lowp_unchecked` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn ln_lowp_unchecked<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::ln_lowp_unchecked(t, x)
    }
    /// Legacy `exp_lowp` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn exp_lowp<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::exp_lowp(t, x)
    }
    /// Legacy `exp_lowp_unchecked` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn exp_lowp_unchecked<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::exp_lowp_unchecked(t, x)
    }
    /// Legacy `log10_lowp` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn log10_lowp<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::log10_lowp(t, x)
    }
    /// Legacy `log10_lowp_unchecked` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn log10_lowp_unchecked<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::log10_lowp_unchecked(t, x)
    }
    /// Legacy `pow_lowp` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn pow_lowp<T: F32x8Convert>(t: T, x: f32x8<T>, n: f32) -> f32x8<T> {
        crate::legacy_impl::width8::pow_lowp(t, x, n)
    }
    /// Legacy `pow_lowp_unchecked` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn pow_lowp_unchecked<T: F32x8Convert>(t: T, x: f32x8<T>, n: f32) -> f32x8<T> {
        crate::legacy_impl::width8::pow_lowp_unchecked(t, x, n)
    }
    /// Legacy `log2_midp_unchecked` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn log2_midp_unchecked<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::log2_midp_unchecked(t, x)
    }
    /// Legacy `log2_midp` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn log2_midp<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::log2_midp(t, x)
    }
    /// Legacy `log2_midp_precise` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn log2_midp_precise<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::log2_midp_precise(t, x)
    }
    /// Legacy `exp2_midp_unchecked` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn exp2_midp_unchecked<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::exp2_midp_unchecked(t, x)
    }
    /// Legacy `exp2_midp` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn exp2_midp<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::exp2_midp(t, x)
    }
    /// Legacy `exp2_midp_precise` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn exp2_midp_precise<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::exp2_midp_precise(t, x)
    }
    /// Legacy `ln_midp` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn ln_midp<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::ln_midp(t, x)
    }
    /// Legacy `ln_midp_unchecked` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn ln_midp_unchecked<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::ln_midp_unchecked(t, x)
    }
    /// Legacy `ln_midp_precise` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn ln_midp_precise<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::ln_midp_precise(t, x)
    }
    /// Legacy `exp_midp` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn exp_midp<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::exp_midp(t, x)
    }
    /// Legacy `exp_midp_unchecked` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn exp_midp_unchecked<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::exp_midp_unchecked(t, x)
    }
    /// Legacy `exp_midp_precise` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn exp_midp_precise<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::exp_midp_precise(t, x)
    }
    /// Legacy `sigmoid_midp` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn sigmoid_midp<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::sigmoid_midp(t, x)
    }
    /// Legacy `silu_midp` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn silu_midp<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::silu_midp(t, x)
    }
    /// Legacy `log10_midp` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn log10_midp<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::log10_midp(t, x)
    }
    /// Legacy `log10_midp_unchecked` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn log10_midp_unchecked<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::log10_midp_unchecked(t, x)
    }
    /// Legacy `log10_midp_precise` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn log10_midp_precise<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::log10_midp_precise(t, x)
    }
    /// Legacy `pow_midp` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn pow_midp<T: F32x8Convert>(t: T, x: f32x8<T>, n: f32) -> f32x8<T> {
        crate::legacy_impl::width8::pow_midp(t, x, n)
    }
    /// Legacy `pow_midp_unchecked` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn pow_midp_unchecked<T: F32x8Convert>(t: T, x: f32x8<T>, n: f32) -> f32x8<T> {
        crate::legacy_impl::width8::pow_midp_unchecked(t, x, n)
    }
    /// Legacy `pow_midp_precise` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn pow_midp_precise<T: F32x8Convert>(t: T, x: f32x8<T>, n: f32) -> f32x8<T> {
        crate::legacy_impl::width8::pow_midp_precise(t, x, n)
    }
    /// Legacy `cbrt_lowp` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn cbrt_lowp<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::cbrt_lowp(t, x)
    }
    /// Legacy `cbrt_midp` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn cbrt_midp<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::cbrt_midp(t, x)
    }
    /// Legacy `cbrt_midp_precise` at width8; original backend rounding and specials.
    #[inline(always)]
    pub fn cbrt_midp_precise<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
        crate::legacy_impl::width8::cbrt_midp_precise(t, x)
    }
    pub mod f32x4 {
        use magetypes::simd::backends::F32x4Convert;
        use magetypes::simd::generic::f32x4;
        /// Legacy `log2_lowp` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn log2_lowp<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::log2_lowp(t, x)
        }
        /// Legacy `log2_lowp_unchecked` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn log2_lowp_unchecked<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::log2_lowp_unchecked(t, x)
        }
        /// Legacy `exp2_lowp` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn exp2_lowp<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::exp2_lowp(t, x)
        }
        /// Legacy `exp2_lowp_unchecked` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn exp2_lowp_unchecked<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::exp2_lowp_unchecked(t, x)
        }
        /// Legacy `ln_lowp` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn ln_lowp<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::ln_lowp(t, x)
        }
        /// Legacy `ln_lowp_unchecked` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn ln_lowp_unchecked<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::ln_lowp_unchecked(t, x)
        }
        /// Legacy `exp_lowp` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn exp_lowp<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::exp_lowp(t, x)
        }
        /// Legacy `exp_lowp_unchecked` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn exp_lowp_unchecked<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::exp_lowp_unchecked(t, x)
        }
        /// Legacy `log10_lowp` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn log10_lowp<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::log10_lowp(t, x)
        }
        /// Legacy `log10_lowp_unchecked` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn log10_lowp_unchecked<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::log10_lowp_unchecked(t, x)
        }
        /// Legacy `pow_lowp` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn pow_lowp<T: F32x4Convert>(t: T, x: f32x4<T>, n: f32) -> f32x4<T> {
            crate::legacy_impl::width4::pow_lowp(t, x, n)
        }
        /// Legacy `pow_lowp_unchecked` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn pow_lowp_unchecked<T: F32x4Convert>(t: T, x: f32x4<T>, n: f32) -> f32x4<T> {
            crate::legacy_impl::width4::pow_lowp_unchecked(t, x, n)
        }
        /// Legacy `log2_midp_unchecked` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn log2_midp_unchecked<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::log2_midp_unchecked(t, x)
        }
        /// Legacy `log2_midp` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn log2_midp<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::log2_midp(t, x)
        }
        /// Legacy `log2_midp_precise` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn log2_midp_precise<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::log2_midp_precise(t, x)
        }
        /// Legacy `exp2_midp_unchecked` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn exp2_midp_unchecked<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::exp2_midp_unchecked(t, x)
        }
        /// Legacy `exp2_midp` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn exp2_midp<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::exp2_midp(t, x)
        }
        /// Legacy `exp2_midp_precise` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn exp2_midp_precise<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::exp2_midp_precise(t, x)
        }
        /// Legacy `ln_midp` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn ln_midp<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::ln_midp(t, x)
        }
        /// Legacy `ln_midp_unchecked` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn ln_midp_unchecked<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::ln_midp_unchecked(t, x)
        }
        /// Legacy `ln_midp_precise` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn ln_midp_precise<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::ln_midp_precise(t, x)
        }
        /// Legacy `exp_midp` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn exp_midp<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::exp_midp(t, x)
        }
        /// Legacy `exp_midp_unchecked` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn exp_midp_unchecked<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::exp_midp_unchecked(t, x)
        }
        /// Legacy `exp_midp_precise` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn exp_midp_precise<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::exp_midp_precise(t, x)
        }
        /// Legacy `sigmoid_midp` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn sigmoid_midp<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::sigmoid_midp(t, x)
        }
        /// Legacy `silu_midp` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn silu_midp<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::silu_midp(t, x)
        }
        /// Legacy `log10_midp` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn log10_midp<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::log10_midp(t, x)
        }
        /// Legacy `log10_midp_unchecked` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn log10_midp_unchecked<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::log10_midp_unchecked(t, x)
        }
        /// Legacy `log10_midp_precise` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn log10_midp_precise<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::log10_midp_precise(t, x)
        }
        /// Legacy `pow_midp` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn pow_midp<T: F32x4Convert>(t: T, x: f32x4<T>, n: f32) -> f32x4<T> {
            crate::legacy_impl::width4::pow_midp(t, x, n)
        }
        /// Legacy `pow_midp_unchecked` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn pow_midp_unchecked<T: F32x4Convert>(t: T, x: f32x4<T>, n: f32) -> f32x4<T> {
            crate::legacy_impl::width4::pow_midp_unchecked(t, x, n)
        }
        /// Legacy `pow_midp_precise` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn pow_midp_precise<T: F32x4Convert>(t: T, x: f32x4<T>, n: f32) -> f32x4<T> {
            crate::legacy_impl::width4::pow_midp_precise(t, x, n)
        }
        /// Legacy `cbrt_lowp` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn cbrt_lowp<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::cbrt_lowp(t, x)
        }
        /// Legacy `cbrt_midp` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn cbrt_midp<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::cbrt_midp(t, x)
        }
        /// Legacy `cbrt_midp_precise` at width4; original backend rounding and specials.
        #[inline(always)]
        pub fn cbrt_midp_precise<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            crate::legacy_impl::width4::cbrt_midp_precise(t, x)
        }
    }
    #[cfg(feature = "w512")]
    pub mod f32x16 {
        use magetypes::simd::backends::F32x16Convert;
        use magetypes::simd::generic::f32x16;
        /// Legacy `log2_lowp` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn log2_lowp<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::log2_lowp(t, x)
        }
        /// Legacy `log2_lowp_unchecked` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn log2_lowp_unchecked<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::log2_lowp_unchecked(t, x)
        }
        /// Legacy `exp2_lowp` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn exp2_lowp<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::exp2_lowp(t, x)
        }
        /// Legacy `exp2_lowp_unchecked` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn exp2_lowp_unchecked<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::exp2_lowp_unchecked(t, x)
        }
        /// Legacy `ln_lowp` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn ln_lowp<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::ln_lowp(t, x)
        }
        /// Legacy `ln_lowp_unchecked` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn ln_lowp_unchecked<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::ln_lowp_unchecked(t, x)
        }
        /// Legacy `exp_lowp` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn exp_lowp<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::exp_lowp(t, x)
        }
        /// Legacy `exp_lowp_unchecked` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn exp_lowp_unchecked<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::exp_lowp_unchecked(t, x)
        }
        /// Legacy `log10_lowp` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn log10_lowp<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::log10_lowp(t, x)
        }
        /// Legacy `log10_lowp_unchecked` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn log10_lowp_unchecked<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::log10_lowp_unchecked(t, x)
        }
        /// Legacy `pow_lowp` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn pow_lowp<T: F32x16Convert>(t: T, x: f32x16<T>, n: f32) -> f32x16<T> {
            crate::legacy_impl::width16::pow_lowp(t, x, n)
        }
        /// Legacy `pow_lowp_unchecked` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn pow_lowp_unchecked<T: F32x16Convert>(t: T, x: f32x16<T>, n: f32) -> f32x16<T> {
            crate::legacy_impl::width16::pow_lowp_unchecked(t, x, n)
        }
        /// Legacy `log2_midp_unchecked` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn log2_midp_unchecked<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::log2_midp_unchecked(t, x)
        }
        /// Legacy `log2_midp` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn log2_midp<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::log2_midp(t, x)
        }
        /// Legacy `log2_midp_precise` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn log2_midp_precise<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::log2_midp_precise(t, x)
        }
        /// Legacy `exp2_midp_unchecked` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn exp2_midp_unchecked<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::exp2_midp_unchecked(t, x)
        }
        /// Legacy `exp2_midp` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn exp2_midp<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::exp2_midp(t, x)
        }
        /// Legacy `exp2_midp_precise` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn exp2_midp_precise<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::exp2_midp_precise(t, x)
        }
        /// Legacy `ln_midp` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn ln_midp<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::ln_midp(t, x)
        }
        /// Legacy `ln_midp_unchecked` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn ln_midp_unchecked<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::ln_midp_unchecked(t, x)
        }
        /// Legacy `ln_midp_precise` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn ln_midp_precise<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::ln_midp_precise(t, x)
        }
        /// Legacy `exp_midp` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn exp_midp<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::exp_midp(t, x)
        }
        /// Legacy `exp_midp_unchecked` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn exp_midp_unchecked<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::exp_midp_unchecked(t, x)
        }
        /// Legacy `exp_midp_precise` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn exp_midp_precise<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::exp_midp_precise(t, x)
        }
        /// Legacy `sigmoid_midp` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn sigmoid_midp<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::sigmoid_midp(t, x)
        }
        /// Legacy `silu_midp` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn silu_midp<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::silu_midp(t, x)
        }
        /// Legacy `log10_midp` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn log10_midp<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::log10_midp(t, x)
        }
        /// Legacy `log10_midp_unchecked` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn log10_midp_unchecked<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::log10_midp_unchecked(t, x)
        }
        /// Legacy `log10_midp_precise` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn log10_midp_precise<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::log10_midp_precise(t, x)
        }
        /// Legacy `pow_midp` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn pow_midp<T: F32x16Convert>(t: T, x: f32x16<T>, n: f32) -> f32x16<T> {
            crate::legacy_impl::width16::pow_midp(t, x, n)
        }
        /// Legacy `pow_midp_unchecked` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn pow_midp_unchecked<T: F32x16Convert>(t: T, x: f32x16<T>, n: f32) -> f32x16<T> {
            crate::legacy_impl::width16::pow_midp_unchecked(t, x, n)
        }
        /// Legacy `pow_midp_precise` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn pow_midp_precise<T: F32x16Convert>(t: T, x: f32x16<T>, n: f32) -> f32x16<T> {
            crate::legacy_impl::width16::pow_midp_precise(t, x, n)
        }
        /// Legacy `cbrt_lowp` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn cbrt_lowp<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::cbrt_lowp(t, x)
        }
        /// Legacy `cbrt_midp` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn cbrt_midp<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::cbrt_midp(t, x)
        }
        /// Legacy `cbrt_midp_precise` at width16; original backend rounding and specials.
        #[inline(always)]
        pub fn cbrt_midp_precise<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            crate::legacy_impl::width16::cbrt_midp_precise(t, x)
        }
    }
}
pub mod array {
    /// Legacy `log2_lowp` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log2_lowp_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::log2_lowp(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `log2_lowp_unchecked` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log2_lowp_unchecked_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::log2_lowp_unchecked(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp2_lowp` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp2_lowp_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::exp2_lowp(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp2_lowp_unchecked` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp2_lowp_unchecked_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::exp2_lowp_unchecked(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `ln_lowp` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn ln_lowp_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::ln_lowp(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `ln_lowp_unchecked` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn ln_lowp_unchecked_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::ln_lowp_unchecked(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp_lowp` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp_lowp_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::exp_lowp(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp_lowp_unchecked` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp_lowp_unchecked_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::exp_lowp_unchecked(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `log10_lowp` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log10_lowp_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::log10_lowp(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `log10_lowp_unchecked` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log10_lowp_unchecked_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::log10_lowp_unchecked(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `pow_lowp` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn pow_lowp_f32x4(a: [f32; 4], n: f32) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::pow_lowp(t, f32x4::from_array_t(t, a), n).to_array()
    }
    /// Legacy `pow_lowp_unchecked` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn pow_lowp_unchecked_f32x4(a: [f32; 4], n: f32) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::pow_lowp_unchecked(t, f32x4::from_array_t(t, a), n).to_array()
    }
    /// Legacy `log2_midp_unchecked` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log2_midp_unchecked_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::log2_midp_unchecked(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `log2_midp` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log2_midp_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::log2_midp(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `log2_midp_precise` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log2_midp_precise_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::log2_midp_precise(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp2_midp_unchecked` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp2_midp_unchecked_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::exp2_midp_unchecked(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp2_midp` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp2_midp_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::exp2_midp(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp2_midp_precise` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp2_midp_precise_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::exp2_midp_precise(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `ln_midp` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn ln_midp_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::ln_midp(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `ln_midp_unchecked` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn ln_midp_unchecked_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::ln_midp_unchecked(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `ln_midp_precise` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn ln_midp_precise_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::ln_midp_precise(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp_midp` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp_midp_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::exp_midp(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp_midp_unchecked` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp_midp_unchecked_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::exp_midp_unchecked(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp_midp_precise` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp_midp_precise_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::exp_midp_precise(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `sigmoid_midp` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn sigmoid_midp_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::sigmoid_midp(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `silu_midp` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn silu_midp_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::silu_midp(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `log10_midp` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log10_midp_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::log10_midp(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `log10_midp_unchecked` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log10_midp_unchecked_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::log10_midp_unchecked(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `log10_midp_precise` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log10_midp_precise_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::log10_midp_precise(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `pow_midp` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn pow_midp_f32x4(a: [f32; 4], n: f32) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::pow_midp(t, f32x4::from_array_t(t, a), n).to_array()
    }
    /// Legacy `pow_midp_unchecked` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn pow_midp_unchecked_f32x4(a: [f32; 4], n: f32) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::pow_midp_unchecked(t, f32x4::from_array_t(t, a), n).to_array()
    }
    /// Legacy `pow_midp_precise` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn pow_midp_precise_f32x4(a: [f32; 4], n: f32) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::pow_midp_precise(t, f32x4::from_array_t(t, a), n).to_array()
    }
    /// Legacy `cbrt_lowp` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn cbrt_lowp_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::cbrt_lowp(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `cbrt_midp` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn cbrt_midp_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::cbrt_midp(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `cbrt_midp_precise` over an array, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn cbrt_midp_precise_f32x4(a: [f32; 4]) -> [f32; 4] {
        let t = Token::from_context();
        super::kernels::f32x4::cbrt_midp_precise(t, f32x4::from_array_t(t, a)).to_array()
    }
    /// Legacy `log2_lowp` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log2_lowp_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::log2_lowp(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `log2_lowp_unchecked` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log2_lowp_unchecked_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::log2_lowp_unchecked(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp2_lowp` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp2_lowp_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::exp2_lowp(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp2_lowp_unchecked` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp2_lowp_unchecked_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::exp2_lowp_unchecked(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `ln_lowp` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn ln_lowp_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::ln_lowp(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `ln_lowp_unchecked` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn ln_lowp_unchecked_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::ln_lowp_unchecked(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp_lowp` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp_lowp_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::exp_lowp(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp_lowp_unchecked` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp_lowp_unchecked_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::exp_lowp_unchecked(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `log10_lowp` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log10_lowp_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::log10_lowp(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `log10_lowp_unchecked` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log10_lowp_unchecked_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::log10_lowp_unchecked(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `pow_lowp` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn pow_lowp_f32x8(a: [f32; 8], n: f32) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::pow_lowp(t, f32x8::from_array_t(t, a), n).to_array()
    }
    /// Legacy `pow_lowp_unchecked` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn pow_lowp_unchecked_f32x8(a: [f32; 8], n: f32) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::pow_lowp_unchecked(t, f32x8::from_array_t(t, a), n).to_array()
    }
    /// Legacy `log2_midp_unchecked` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log2_midp_unchecked_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::log2_midp_unchecked(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `log2_midp` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log2_midp_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::log2_midp(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `log2_midp_precise` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log2_midp_precise_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::log2_midp_precise(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp2_midp_unchecked` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp2_midp_unchecked_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::exp2_midp_unchecked(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp2_midp` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp2_midp_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::exp2_midp(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp2_midp_precise` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp2_midp_precise_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::exp2_midp_precise(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `ln_midp` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn ln_midp_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::ln_midp(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `ln_midp_unchecked` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn ln_midp_unchecked_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::ln_midp_unchecked(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `ln_midp_precise` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn ln_midp_precise_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::ln_midp_precise(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp_midp` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp_midp_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::exp_midp(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp_midp_unchecked` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp_midp_unchecked_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::exp_midp_unchecked(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `exp_midp_precise` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp_midp_precise_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::exp_midp_precise(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `sigmoid_midp` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn sigmoid_midp_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::sigmoid_midp(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `silu_midp` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn silu_midp_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::silu_midp(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `log10_midp` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log10_midp_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::log10_midp(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `log10_midp_unchecked` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log10_midp_unchecked_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::log10_midp_unchecked(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `log10_midp_precise` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log10_midp_precise_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::log10_midp_precise(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `pow_midp` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn pow_midp_f32x8(a: [f32; 8], n: f32) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::pow_midp(t, f32x8::from_array_t(t, a), n).to_array()
    }
    /// Legacy `pow_midp_unchecked` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn pow_midp_unchecked_f32x8(a: [f32; 8], n: f32) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::pow_midp_unchecked(t, f32x8::from_array_t(t, a), n).to_array()
    }
    /// Legacy `pow_midp_precise` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn pow_midp_precise_f32x8(a: [f32; 8], n: f32) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::pow_midp_precise(t, f32x8::from_array_t(t, a), n).to_array()
    }
    /// Legacy `cbrt_lowp` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn cbrt_lowp_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::cbrt_lowp(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `cbrt_midp` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn cbrt_midp_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::cbrt_midp(t, f32x8::from_array_t(t, a)).to_array()
    }
    /// Legacy `cbrt_midp_precise` over an array, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn cbrt_midp_precise_f32x8(a: [f32; 8]) -> [f32; 8] {
        let t = Token::from_context();
        super::kernels::cbrt_midp_precise(t, f32x8::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `log2_lowp` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log2_lowp_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::log2_lowp(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `log2_lowp_unchecked` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log2_lowp_unchecked_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::log2_lowp_unchecked(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp2_lowp` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp2_lowp_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::exp2_lowp(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp2_lowp_unchecked` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp2_lowp_unchecked_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::exp2_lowp_unchecked(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `ln_lowp` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn ln_lowp_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::ln_lowp(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `ln_lowp_unchecked` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn ln_lowp_unchecked_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::ln_lowp_unchecked(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp_lowp` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp_lowp_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::exp_lowp(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp_lowp_unchecked` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp_lowp_unchecked_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::exp_lowp_unchecked(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `log10_lowp` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log10_lowp_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::log10_lowp(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `log10_lowp_unchecked` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log10_lowp_unchecked_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::log10_lowp_unchecked(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `pow_lowp` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn pow_lowp_f32x16(a: [f32; 16], n: f32) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::pow_lowp(t, f32x16::from_array_t(t, a), n).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `pow_lowp_unchecked` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn pow_lowp_unchecked_f32x16(a: [f32; 16], n: f32) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::pow_lowp_unchecked(t, f32x16::from_array_t(t, a), n).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `log2_midp_unchecked` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log2_midp_unchecked_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::log2_midp_unchecked(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `log2_midp` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log2_midp_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::log2_midp(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `log2_midp_precise` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log2_midp_precise_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::log2_midp_precise(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp2_midp_unchecked` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp2_midp_unchecked_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::exp2_midp_unchecked(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp2_midp` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp2_midp_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::exp2_midp(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp2_midp_precise` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp2_midp_precise_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::exp2_midp_precise(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `ln_midp` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn ln_midp_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::ln_midp(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `ln_midp_unchecked` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn ln_midp_unchecked_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::ln_midp_unchecked(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `ln_midp_precise` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn ln_midp_precise_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::ln_midp_precise(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp_midp` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp_midp_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::exp_midp(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp_midp_unchecked` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp_midp_unchecked_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::exp_midp_unchecked(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp_midp_precise` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp_midp_precise_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::exp_midp_precise(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `sigmoid_midp` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn sigmoid_midp_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::sigmoid_midp(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `silu_midp` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn silu_midp_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::silu_midp(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `log10_midp` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log10_midp_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::log10_midp(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `log10_midp_unchecked` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log10_midp_unchecked_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::log10_midp_unchecked(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `log10_midp_precise` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log10_midp_precise_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::log10_midp_precise(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `pow_midp` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn pow_midp_f32x16(a: [f32; 16], n: f32) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::pow_midp(t, f32x16::from_array_t(t, a), n).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `pow_midp_unchecked` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn pow_midp_unchecked_f32x16(a: [f32; 16], n: f32) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::pow_midp_unchecked(t, f32x16::from_array_t(t, a), n).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `pow_midp_precise` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn pow_midp_precise_f32x16(a: [f32; 16], n: f32) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::pow_midp_precise(t, f32x16::from_array_t(t, a), n).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `cbrt_lowp` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn cbrt_lowp_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::cbrt_lowp(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `cbrt_midp` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn cbrt_midp_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::cbrt_midp(t, f32x16::from_array_t(t, a)).to_array()
    }
    #[cfg(feature = "w512")]
    /// Legacy `cbrt_midp_precise` over an array, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn cbrt_midp_precise_f32x16(a: [f32; 16]) -> [f32; 16] {
        let t = Token::from_context();
        super::kernels::f32x16::cbrt_midp_precise(t, f32x16::from_array_t(t, a)).to_array()
    }
}
pub mod slice {
    /// Legacy `log2_lowp` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log2_lowp_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::log2_lowp(t, f32x4::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::log2_lowp(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log2_lowp_unchecked` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log2_lowp_unchecked_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::log2_lowp_unchecked(t, f32x4::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::log2_lowp_unchecked(t, f32x4::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp2_lowp` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp2_lowp_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::exp2_lowp(t, f32x4::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::exp2_lowp(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp2_lowp_unchecked` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp2_lowp_unchecked_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::exp2_lowp_unchecked(t, f32x4::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::exp2_lowp_unchecked(t, f32x4::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `ln_lowp` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn ln_lowp_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::ln_lowp(t, f32x4::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::ln_lowp(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `ln_lowp_unchecked` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn ln_lowp_unchecked_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::ln_lowp_unchecked(t, f32x4::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::f32x4::ln_lowp_unchecked(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp_lowp` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp_lowp_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::exp_lowp(t, f32x4::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::exp_lowp(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp_lowp_unchecked` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp_lowp_unchecked_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::exp_lowp_unchecked(t, f32x4::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::exp_lowp_unchecked(t, f32x4::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log10_lowp` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log10_lowp_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x4::log10_lowp(t, f32x4::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::log10_lowp(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log10_lowp_unchecked` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log10_lowp_unchecked_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::log10_lowp_unchecked(t, f32x4::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::log10_lowp_unchecked(t, f32x4::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `pow_lowp` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn pow_lowp_f32x4(a: &mut [f32], n: f32) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x4::pow_lowp(t, f32x4::from_array_t(t, *chunk), n).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::pow_lowp(t, f32x4::from_array_t(t, pad), n).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `pow_lowp_unchecked` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn pow_lowp_unchecked_f32x4(a: &mut [f32], n: f32) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x4::pow_lowp_unchecked(t, f32x4::from_array_t(t, *chunk), n)
                    .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::pow_lowp_unchecked(t, f32x4::from_array_t(t, pad), n)
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log2_midp_unchecked` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log2_midp_unchecked_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::log2_midp_unchecked(t, f32x4::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::log2_midp_unchecked(t, f32x4::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log2_midp` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log2_midp_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::log2_midp(t, f32x4::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::log2_midp(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log2_midp_precise` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log2_midp_precise_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::log2_midp_precise(t, f32x4::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::f32x4::log2_midp_precise(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp2_midp_unchecked` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp2_midp_unchecked_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::exp2_midp_unchecked(t, f32x4::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::exp2_midp_unchecked(t, f32x4::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp2_midp` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp2_midp_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::exp2_midp(t, f32x4::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::exp2_midp(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp2_midp_precise` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp2_midp_precise_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::exp2_midp_precise(t, f32x4::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::f32x4::exp2_midp_precise(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `ln_midp` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn ln_midp_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::ln_midp(t, f32x4::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::ln_midp(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `ln_midp_unchecked` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn ln_midp_unchecked_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::ln_midp_unchecked(t, f32x4::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::f32x4::ln_midp_unchecked(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `ln_midp_precise` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn ln_midp_precise_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::ln_midp_precise(t, f32x4::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::f32x4::ln_midp_precise(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp_midp` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp_midp_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::exp_midp(t, f32x4::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::exp_midp(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp_midp_unchecked` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp_midp_unchecked_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::exp_midp_unchecked(t, f32x4::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::exp_midp_unchecked(t, f32x4::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp_midp_precise` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn exp_midp_precise_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::exp_midp_precise(t, f32x4::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::f32x4::exp_midp_precise(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `sigmoid_midp` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn sigmoid_midp_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x4::sigmoid_midp(t, f32x4::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::f32x4::sigmoid_midp(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `silu_midp` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn silu_midp_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::silu_midp(t, f32x4::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::silu_midp(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log10_midp` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log10_midp_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x4::log10_midp(t, f32x4::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::log10_midp(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log10_midp_unchecked` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log10_midp_unchecked_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::log10_midp_unchecked(t, f32x4::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::log10_midp_unchecked(t, f32x4::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log10_midp_precise` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn log10_midp_precise_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::log10_midp_precise(t, f32x4::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::log10_midp_precise(t, f32x4::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `pow_midp` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn pow_midp_f32x4(a: &mut [f32], n: f32) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x4::pow_midp(t, f32x4::from_array_t(t, *chunk), n).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::pow_midp(t, f32x4::from_array_t(t, pad), n).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `pow_midp_unchecked` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn pow_midp_unchecked_f32x4(a: &mut [f32], n: f32) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x4::pow_midp_unchecked(t, f32x4::from_array_t(t, *chunk), n)
                    .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::pow_midp_unchecked(t, f32x4::from_array_t(t, pad), n)
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `pow_midp_precise` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn pow_midp_precise_f32x4(a: &mut [f32], n: f32) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::pow_midp_precise(t, f32x4::from_array_t(t, *chunk), n)
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::pow_midp_precise(t, f32x4::from_array_t(t, pad), n)
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `cbrt_lowp` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn cbrt_lowp_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::cbrt_lowp(t, f32x4::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::cbrt_lowp(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `cbrt_midp` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn cbrt_midp_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::cbrt_midp(t, f32x4::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x4::cbrt_midp(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `cbrt_midp_precise` over a slice, using width4 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x4))]
    pub fn cbrt_midp_precise_f32x4(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<4>();
        for chunk in chunks {
            *chunk = super::kernels::f32x4::cbrt_midp_precise(t, f32x4::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 4];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::f32x4::cbrt_midp_precise(t, f32x4::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log2_lowp` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log2_lowp_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::log2_lowp(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::log2_lowp(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log2_lowp_unchecked` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log2_lowp_unchecked_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::log2_lowp_unchecked(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::log2_lowp_unchecked(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp2_lowp` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp2_lowp_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::exp2_lowp(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::exp2_lowp(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp2_lowp_unchecked` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp2_lowp_unchecked_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::exp2_lowp_unchecked(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::exp2_lowp_unchecked(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `ln_lowp` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn ln_lowp_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::ln_lowp(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::ln_lowp(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `ln_lowp_unchecked` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn ln_lowp_unchecked_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::ln_lowp_unchecked(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::ln_lowp_unchecked(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp_lowp` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp_lowp_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::exp_lowp(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::exp_lowp(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp_lowp_unchecked` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp_lowp_unchecked_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::exp_lowp_unchecked(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::exp_lowp_unchecked(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log10_lowp` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log10_lowp_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::log10_lowp(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::log10_lowp(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log10_lowp_unchecked` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log10_lowp_unchecked_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::log10_lowp_unchecked(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::log10_lowp_unchecked(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `pow_lowp` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn pow_lowp_f32x8(a: &mut [f32], n: f32) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::pow_lowp(t, f32x8::from_array_t(t, *chunk), n).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::pow_lowp(t, f32x8::from_array_t(t, pad), n).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `pow_lowp_unchecked` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn pow_lowp_unchecked_f32x8(a: &mut [f32], n: f32) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::pow_lowp_unchecked(t, f32x8::from_array_t(t, *chunk), n).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::pow_lowp_unchecked(t, f32x8::from_array_t(t, pad), n).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log2_midp_unchecked` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log2_midp_unchecked_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::log2_midp_unchecked(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::log2_midp_unchecked(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log2_midp` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log2_midp_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::log2_midp(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::log2_midp(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log2_midp_precise` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log2_midp_precise_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::log2_midp_precise(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::log2_midp_precise(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp2_midp_unchecked` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp2_midp_unchecked_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::exp2_midp_unchecked(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::exp2_midp_unchecked(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp2_midp` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp2_midp_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::exp2_midp(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::exp2_midp(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp2_midp_precise` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp2_midp_precise_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::exp2_midp_precise(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::exp2_midp_precise(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `ln_midp` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn ln_midp_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::ln_midp(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::ln_midp(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `ln_midp_unchecked` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn ln_midp_unchecked_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::ln_midp_unchecked(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::ln_midp_unchecked(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `ln_midp_precise` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn ln_midp_precise_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::ln_midp_precise(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::ln_midp_precise(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp_midp` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp_midp_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::exp_midp(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::exp_midp(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp_midp_unchecked` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp_midp_unchecked_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::exp_midp_unchecked(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::exp_midp_unchecked(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `exp_midp_precise` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn exp_midp_precise_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::exp_midp_precise(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::exp_midp_precise(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `sigmoid_midp` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn sigmoid_midp_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::sigmoid_midp(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::sigmoid_midp(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `silu_midp` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn silu_midp_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::silu_midp(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::silu_midp(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log10_midp` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log10_midp_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::log10_midp(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::log10_midp(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log10_midp_unchecked` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log10_midp_unchecked_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::log10_midp_unchecked(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::log10_midp_unchecked(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `log10_midp_precise` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn log10_midp_precise_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::log10_midp_precise(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::log10_midp_precise(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `pow_midp` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn pow_midp_f32x8(a: &mut [f32], n: f32) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::pow_midp(t, f32x8::from_array_t(t, *chunk), n).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::pow_midp(t, f32x8::from_array_t(t, pad), n).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `pow_midp_unchecked` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn pow_midp_unchecked_f32x8(a: &mut [f32], n: f32) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::pow_midp_unchecked(t, f32x8::from_array_t(t, *chunk), n).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::pow_midp_unchecked(t, f32x8::from_array_t(t, pad), n).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `pow_midp_precise` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn pow_midp_precise_f32x8(a: &mut [f32], n: f32) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::pow_midp_precise(t, f32x8::from_array_t(t, *chunk), n).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::pow_midp_precise(t, f32x8::from_array_t(t, pad), n).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `cbrt_lowp` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn cbrt_lowp_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::cbrt_lowp(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::cbrt_lowp(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `cbrt_midp` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn cbrt_midp_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk = super::kernels::cbrt_midp(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::cbrt_midp(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    /// Legacy `cbrt_midp_precise` over a slice, using width8 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x8))]
    pub fn cbrt_midp_precise_f32x8(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<8>();
        for chunk in chunks {
            *chunk =
                super::kernels::cbrt_midp_precise(t, f32x8::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::cbrt_midp_precise(t, f32x8::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `log2_lowp` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log2_lowp_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::log2_lowp(t, f32x16::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::log2_lowp(t, f32x16::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `log2_lowp_unchecked` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log2_lowp_unchecked_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::log2_lowp_unchecked(t, f32x16::from_array_t(t, *chunk))
                    .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::log2_lowp_unchecked(t, f32x16::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp2_lowp` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp2_lowp_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::exp2_lowp(t, f32x16::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::exp2_lowp(t, f32x16::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp2_lowp_unchecked` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp2_lowp_unchecked_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::exp2_lowp_unchecked(t, f32x16::from_array_t(t, *chunk))
                    .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::exp2_lowp_unchecked(t, f32x16::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `ln_lowp` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn ln_lowp_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk = super::kernels::f32x16::ln_lowp(t, f32x16::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::ln_lowp(t, f32x16::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `ln_lowp_unchecked` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn ln_lowp_unchecked_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk = super::kernels::f32x16::ln_lowp_unchecked(t, f32x16::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::ln_lowp_unchecked(t, f32x16::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp_lowp` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp_lowp_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::exp_lowp(t, f32x16::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::exp_lowp(t, f32x16::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp_lowp_unchecked` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp_lowp_unchecked_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk = super::kernels::f32x16::exp_lowp_unchecked(t, f32x16::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::exp_lowp_unchecked(t, f32x16::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `log10_lowp` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log10_lowp_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::log10_lowp(t, f32x16::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::f32x16::log10_lowp(t, f32x16::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `log10_lowp_unchecked` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log10_lowp_unchecked_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::log10_lowp_unchecked(t, f32x16::from_array_t(t, *chunk))
                    .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::log10_lowp_unchecked(t, f32x16::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `pow_lowp` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn pow_lowp_f32x16(a: &mut [f32], n: f32) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::pow_lowp(t, f32x16::from_array_t(t, *chunk), n).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::f32x16::pow_lowp(t, f32x16::from_array_t(t, pad), n).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `pow_lowp_unchecked` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn pow_lowp_unchecked_f32x16(a: &mut [f32], n: f32) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::pow_lowp_unchecked(t, f32x16::from_array_t(t, *chunk), n)
                    .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::f32x16::pow_lowp_unchecked(t, f32x16::from_array_t(t, pad), n)
                    .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `log2_midp_unchecked` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log2_midp_unchecked_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::log2_midp_unchecked(t, f32x16::from_array_t(t, *chunk))
                    .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::log2_midp_unchecked(t, f32x16::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `log2_midp` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log2_midp_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::log2_midp(t, f32x16::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::log2_midp(t, f32x16::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `log2_midp_precise` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log2_midp_precise_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk = super::kernels::f32x16::log2_midp_precise(t, f32x16::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::log2_midp_precise(t, f32x16::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp2_midp_unchecked` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp2_midp_unchecked_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::exp2_midp_unchecked(t, f32x16::from_array_t(t, *chunk))
                    .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::exp2_midp_unchecked(t, f32x16::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp2_midp` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp2_midp_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::exp2_midp(t, f32x16::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::exp2_midp(t, f32x16::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp2_midp_precise` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp2_midp_precise_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk = super::kernels::f32x16::exp2_midp_precise(t, f32x16::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::exp2_midp_precise(t, f32x16::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `ln_midp` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn ln_midp_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk = super::kernels::f32x16::ln_midp(t, f32x16::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::ln_midp(t, f32x16::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `ln_midp_unchecked` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn ln_midp_unchecked_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk = super::kernels::f32x16::ln_midp_unchecked(t, f32x16::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::ln_midp_unchecked(t, f32x16::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `ln_midp_precise` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn ln_midp_precise_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk = super::kernels::f32x16::ln_midp_precise(t, f32x16::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::f32x16::ln_midp_precise(t, f32x16::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp_midp` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp_midp_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::exp_midp(t, f32x16::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::exp_midp(t, f32x16::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp_midp_unchecked` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp_midp_unchecked_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk = super::kernels::f32x16::exp_midp_unchecked(t, f32x16::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::exp_midp_unchecked(t, f32x16::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `exp_midp_precise` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn exp_midp_precise_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk = super::kernels::f32x16::exp_midp_precise(t, f32x16::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::exp_midp_precise(t, f32x16::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `sigmoid_midp` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn sigmoid_midp_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::sigmoid_midp(t, f32x16::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::f32x16::sigmoid_midp(t, f32x16::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `silu_midp` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn silu_midp_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::silu_midp(t, f32x16::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::silu_midp(t, f32x16::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `log10_midp` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log10_midp_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::log10_midp(t, f32x16::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::f32x16::log10_midp(t, f32x16::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `log10_midp_unchecked` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log10_midp_unchecked_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::log10_midp_unchecked(t, f32x16::from_array_t(t, *chunk))
                    .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::log10_midp_unchecked(t, f32x16::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `log10_midp_precise` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn log10_midp_precise_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk = super::kernels::f32x16::log10_midp_precise(t, f32x16::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::log10_midp_precise(t, f32x16::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `pow_midp` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn pow_midp_f32x16(a: &mut [f32], n: f32) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::pow_midp(t, f32x16::from_array_t(t, *chunk), n).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::f32x16::pow_midp(t, f32x16::from_array_t(t, pad), n).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `pow_midp_unchecked` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn pow_midp_unchecked_f32x16(a: &mut [f32], n: f32) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::pow_midp_unchecked(t, f32x16::from_array_t(t, *chunk), n)
                    .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out =
                super::kernels::f32x16::pow_midp_unchecked(t, f32x16::from_array_t(t, pad), n)
                    .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `pow_midp_precise` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn pow_midp_precise_f32x16(a: &mut [f32], n: f32) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::pow_midp_precise(t, f32x16::from_array_t(t, *chunk), n)
                    .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::pow_midp_precise(t, f32x16::from_array_t(t, pad), n)
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `cbrt_lowp` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn cbrt_lowp_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::cbrt_lowp(t, f32x16::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::cbrt_lowp(t, f32x16::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `cbrt_midp` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn cbrt_midp_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk =
                super::kernels::f32x16::cbrt_midp(t, f32x16::from_array_t(t, *chunk)).to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::cbrt_midp(t, f32x16::from_array_t(t, pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    #[cfg(feature = "w512")]
    /// Legacy `cbrt_midp_precise` over a slice, using width16 vector operations.
    #[archmage::attune(_*,_*_t,dispatch(inline(hint)),+v4x(cfg(avx512)),define(f32x16))]
    pub fn cbrt_midp_precise_f32x16(a: &mut [f32]) {
        let t = Token::from_context();
        let (chunks, tail) = a.as_chunks_mut::<16>();
        for chunk in chunks {
            *chunk = super::kernels::f32x16::cbrt_midp_precise(t, f32x16::from_array_t(t, *chunk))
                .to_array();
        }
        if !tail.is_empty() {
            let mut pad = [1.0; 16];
            pad[..tail.len()].copy_from_slice(tail);
            let out = super::kernels::f32x16::cbrt_midp_precise(t, f32x16::from_array_t(t, pad))
                .to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
}

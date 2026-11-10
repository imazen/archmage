//! Scalar sqrt selection without hidden feature detection or new proof APIs.
//! Keep the public integer implementation as the no_std fallback and independent
//! regression control. Std supplies a correctly rounded hardware lowering.
#![forbid(unsafe_code)]

#[inline(always)]
pub(crate) fn sqrtf(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        crate::nostd_math::sqrtf(x)
    }
}

#[inline(always)]
pub(crate) fn sqrt(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        crate::nostd_math::sqrt(x)
    }
}

//! Native scalar approximation contexts. No coefficient or rounding-order change.
use crate::core_math::PowMath;
use archmage::prelude::*;
trait FloatMath: PowMath {
    fn fmaf(self, a: f32, b: f32, c: f32) -> f32;
    fn floorf(self, x: f32) -> f32;
    fn roundevenf(self, x: f32) -> f32;
}
#[cfg(target_arch = "x86_64")]
impl FloatMath for X64V3Token {
    #[inline(always)]
    fn fmaf(self, a: f32, b: f32, c: f32) -> f32 {
        super::scalar::fmaf_v3_t(self, a, b, c)
    }
    #[inline(always)]
    fn floorf(self, x: f32) -> f32 {
        super::scalar::floorf_v2_t(self.v2(), x)
    }
    #[inline(always)]
    fn roundevenf(self, x: f32) -> f32 {
        super::scalar::roundevenf_v2_t(self.v2(), x)
    }
}
#[cfg(target_arch = "aarch64")]
impl FloatMath for NeonToken {
    #[inline(always)]
    fn fmaf(self, a: f32, b: f32, c: f32) -> f32 {
        super::scalar::fmaf_neon_t(self, a, b, c)
    }
    #[inline(always)]
    fn floorf(self, x: f32) -> f32 {
        super::scalar::floorf_neon_t(self, x)
    }
    #[inline(always)]
    fn roundevenf(self, x: f32) -> f32 {
        super::scalar::roundevenf_neon_t(self, x)
    }
}
#[inline(always)]
fn evaluate<const P: u64>(math: impl FloatMath, x: f32, y: f32) -> f32 {
    let madd = |a, b, c| {
        if P & (1 << 2) == 0 {
            math.fmaf(a, b, c)
        } else {
            a * b + c
        }
    };
    let exponent = |x, y| {
        let log = if P & 3 == 1 {
            crate::reference::log2_midp_evaluate(x, madd)
        } else {
            crate::reference::log2_lowp_evaluate(x, madd)
        };
        log * y
    };
    let exponential = |z| {
        if P & 3 == 1 {
            crate::reference::exp2_midp_evaluate(z, madd, |x| math.roundevenf(x))
        } else {
            crate::reference::exp2_lowp_evaluate(z, madd, |x| math.floorf(x))
        }
    };
    let formula = |x, y| {
        if P & 3 == 2 && x == 0.0 {
            0.0
        } else {
            exponential(exponent(x, y))
        }
    };
    crate::__pow_policy::evaluate::<P>(
        x,
        y,
        |x, y| {
            crate::gamma::try_gamma(x, y)
                .unwrap_or_else(|| crate::core_math::powf_with_math(math, x, y))
        },
        exponent,
        exponential,
        formula,
    )
}
#[archmage::attune(_v3, _v3_t, _neon, _neon_t)]
fn native<const P: u64>(x: f32, y: f32) -> f32 {
    evaluate::<P>(Token::from_context(), x, y)
}
#[inline(always)]
pub(crate) fn select<const P: u64>(x: f32, y: f32) -> f32 {
    // Correct policies share ordinary correct-core selection.
    if P & 3 == 0 {
        return crate::__pow_policy::evaluate_with_core::<P>(x, y, crate::scalar::selected_powf);
    }
    // On x86 this already inlines without software FMA, and the native
    // boundary had no clear benefit. NEON instead benefits from native floor.
    #[cfg(target_arch = "x86_64")]
    if P & 7 == 6 && (P >> 9) & 3 == 2 {
        return crate::reference::powf_with::<P>(x, y);
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = super::available::<X64V3Token>() {
        return native_v3_t::<P>(t, x, y);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = super::available::<NeonToken>() {
        return native_neon_t::<P>(t, x, y);
    }
    crate::__pow_policy::evaluate_with_core::<P>(x, y, crate::scalar::selected_powf)
}

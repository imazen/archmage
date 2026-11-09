use super::{PowfError, PowfExecution, PowfOptions, graph};
use archmage::{ScalarToken, SimdToken};

#[derive(Clone, Copy, Debug)]
pub(super) enum Plan {
    #[cfg(target_arch = "x86_64")]
    V3(archmage::X64V3Token),
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    V4(archmage::X64V4Token),
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    V4x(archmage::X64V4xToken),
    #[cfg(target_arch = "aarch64")]
    Neon(archmage::NeonToken),
    #[cfg(target_arch = "wasm32")]
    Wasm(archmage::Wasm128Token, bool),
    Software,
    #[cfg(all(feature = "std", not(miri)))]
    Std,
}

#[inline(always)]
fn select_impl<T: SimdToken, const EXACT: bool, const DOUBLE: bool>(
    token: T,
    execution: PowfExecution,
) -> Plan {
    let _ = token;
    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    ))]
    let any: &dyn core::any::Any = &token;
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = any.downcast_ref::<archmage::X64V3Token>() {
            return Plan::V3(*t);
        }
        #[cfg(feature = "avx512")]
        {
            if let Some(t) = any.downcast_ref::<archmage::X64V4Token>() {
                return Plan::V4(*t);
            }
            if let Some(t) = any.downcast_ref::<archmage::X64V4xToken>() {
                return Plan::V4x(*t);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = any.downcast_ref::<archmage::NeonToken>() {
        return Plan::Neon(*t);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = any.downcast_ref::<archmage::Wasm128Token>() {
        return Plan::Wasm(
            *t,
            !EXACT
                || if DOUBLE {
                    crate::wasm_fma::powf_f64_fuses()
                } else {
                    crate::wasm_fma::powf_f32_fuses()
                },
        );
    }
    if execution == PowfExecution::Caller {
        return Plan::Software;
    }
    let plan = if DOUBLE {
        archmage::reattune!(upgrade_highp(), [_v3_t, _neon_t, _wasm128_t, _scalar_t])
    } else if EXACT {
        archmage::reattune!(upgrade(), [_v3_t, _neon_t, _wasm128_t, _scalar_t])
    } else {
        archmage::reattune!(upgrade_relaxed(), [_v3_t, _neon_t, _wasm128_t, _scalar_t])
    };
    if !matches!(plan, Plan::Software) {
        return plan;
    }
    #[cfg(all(feature = "std", not(miri)))]
    if EXACT && !DOUBLE && super::std_probe::available() {
        return Plan::Std;
    }
    Plan::Software
}

// Use the supported attunement boundary for autonomous operation selection.
// The selected proof is reused by the complete polynomial, never per FMA.
#[archmage::attune(_*_t)]
fn upgrade(token: Token) -> Plan {
    select(token, PowfExecution::Caller)
}

#[inline(always)]
pub(super) fn select<T: SimdToken>(token: T, execution: PowfExecution) -> Plan {
    select_impl::<T, true, false>(token, execution)
}

#[inline(always)]
pub(super) fn select_relaxed<T: SimdToken>(token: T, execution: PowfExecution) -> Plan {
    select_impl::<T, false, false>(token, execution)
}

#[archmage::attune(_*_t)]
fn upgrade_relaxed(token: Token) -> Plan {
    select_impl::<_, false, false>(token, PowfExecution::Caller)
}

macro_rules! run_const {
    ($name:ident, $v3:ident, $v4:ident, $v4x:ident, $neon:ident, $wasm:ident, $scalar:ident, $n:literal) => {
        #[inline(always)]
        pub(super) fn $name<const OPTIONS: u32>(
            plan: Plan,
            bases: [f32; $n],
            exponent: f32,
        ) -> Result<[f32; $n], PowfError> {
            match plan {
                #[cfg(target_arch = "x86_64")]
                Plan::V3(t) => graph::$v3::<{ graph::NATIVE }, OPTIONS>(t, bases, exponent),
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                Plan::V4(t) => graph::$v4::<{ graph::NATIVE }, OPTIONS>(t, bases, exponent),
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                Plan::V4x(t) => graph::$v4x::<{ graph::NATIVE }, OPTIONS>(t, bases, exponent),
                #[cfg(target_arch = "aarch64")]
                Plan::Neon(t) => graph::$neon::<{ graph::NATIVE }, OPTIONS>(t, bases, exponent),
                #[cfg(target_arch = "wasm32")]
                Plan::Wasm(t, true) => {
                    graph::$wasm::<{ graph::NATIVE }, OPTIONS>(t, bases, exponent)
                }
                #[cfg(target_arch = "wasm32")]
                Plan::Wasm(t, false) => {
                    graph::$wasm::<{ graph::WASM_SOFTWARE }, OPTIONS>(t, bases, exponent)
                }
                Plan::Software => {
                    if OPTIONS & PowfOptions::BACKEND != 0 {
                        graph::$scalar::<{ graph::RELAXED }, OPTIONS>(ScalarToken, bases, exponent)
                    } else {
                        graph::$scalar::<{ graph::SOFTWARE }, OPTIONS>(ScalarToken, bases, exponent)
                    }
                }
                #[cfg(all(feature = "std", not(miri)))]
                Plan::Std => {
                    graph::$scalar::<{ graph::STD }, OPTIONS>(ScalarToken, bases, exponent)
                }
            }
        }
    };
}
run_const!(
    run4_const,
    kernel4_with_v3,
    kernel4_with_v4,
    kernel4_with_v4x,
    kernel4_with_neon,
    kernel4_with_wasm128,
    kernel4_with_scalar,
    4
);
run_const!(
    run8_const,
    kernel8_with_v3,
    kernel8_with_v4,
    kernel8_with_v4x,
    kernel8_with_neon,
    kernel8_with_wasm128,
    kernel8_with_scalar,
    8
);
#[cfg(feature = "w512")]
run_const!(
    run16_const,
    kernel16_with_v3,
    kernel16_with_v4,
    kernel16_with_v4x,
    kernel16_with_neon,
    kernel16_with_wasm128,
    kernel16_with_scalar,
    16
);

#[archmage::attune(_*_t)]
fn upgrade_highp(token: Token) -> Plan {
    select_impl::<_, true, true>(token, PowfExecution::Caller)
}
#[inline(always)]
pub(super) fn select_highp<T: SimdToken>(token: T, execution: PowfExecution) -> Plan {
    select_impl::<T, true, true>(token, execution)
}

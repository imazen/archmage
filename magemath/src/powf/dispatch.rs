use super::{PowfError, graph};
use crate::Execution;
use archmage::{IntoConcreteToken, ScalarToken};

#[derive(Clone, Copy, Debug)]
pub(super) enum Plan {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    V3(archmage::X64V3Token),
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
    V4(archmage::X64V4Token),
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
    V4x(archmage::X64V4xToken),
    #[cfg(target_arch = "aarch64")]
    Neon(archmage::NeonToken),
    #[cfg(target_arch = "wasm32")]
    Wasm(archmage::Wasm128Token, bool),
    Software,
    #[cfg(all(feature = "std", not(miri)))]
    Std,
}

impl Plan {
    #[inline(always)]
    pub(super) fn midp_slice_width(self) -> usize {
        match self {
            Self::Software => 1,
            #[cfg(all(feature = "std", not(miri)))]
            Self::Std => 1,
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
            Self::V4(_) | Self::V4x(_) => 16,
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
            Self::V3(_) => 8,
            #[cfg(target_arch = "aarch64")]
            Self::Neon(_) => 8,
            #[cfg(target_arch = "wasm32")]
            Self::Wasm(_, native) => {
                if native {
                    4
                } else {
                    1
                }
            }
        }
    }

    #[inline(always)]
    pub(super) fn same_tier(self, other: Self) -> bool {
        core::mem::discriminant(&self) == core::mem::discriminant(&other)
    }
}

#[inline(always)]
fn reattune_plan<const EXACT: bool, const DOUBLE: bool>() -> Plan {
    if DOUBLE {
        archmage::reattune!(
            upgrade_highp(),
            [
                _v4x_t(cfg(avx512)),
                _v4_t(cfg(avx512)),
                _v3_t,
                _neon_t,
                _wasm128_t,
                _scalar_t
            ]
        )
    } else if EXACT {
        archmage::reattune!(
            upgrade(),
            [
                _v4x_t(cfg(avx512)),
                _v4_t(cfg(avx512)),
                _v3_t,
                _neon_t,
                _wasm128_t,
                _scalar_t
            ]
        )
    } else {
        archmage::reattune!(
            upgrade_relaxed(),
            [
                _v4x_t(cfg(avx512)),
                _v4_t(cfg(avx512)),
                _v3_t,
                _neon_t,
                _wasm128_t,
                _scalar_t
            ]
        )
    }
}

#[inline(always)]
fn select_impl<T: IntoConcreteToken, const EXACT: bool, const DOUBLE: bool>(
    token: T,
    execution: Execution,
    widen_v3: bool,
) -> Plan {
    let _ = (token, widen_v3);
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        #[cfg(feature = "avx512")]
        {
            if let Some(t) = token.try_v4x() {
                return Plan::V4x(t);
            }
            if let Some(t) = token.try_v4() {
                return Plan::V4(t);
            }
        }
        if let Some(v3) = token.try_v3() {
            #[cfg(feature = "avx512")]
            if widen_v3 && execution == Execution::Upgrade {
                let upgraded = reattune_plan::<EXACT, DOUBLE>();
                if matches!(upgraded, Plan::V4(_) | Plan::V4x(_)) {
                    return upgraded;
                }
            }
            return Plan::V3(v3);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = token.try_neon() {
        return Plan::Neon(t);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = token.try_wasm128() {
        return Plan::Wasm(
            t,
            !EXACT
                || if DOUBLE {
                    crate::wasm_fma::powf_f64_fuses()
                } else {
                    crate::wasm_fma::powf_f32_fuses()
                },
        );
    }
    if execution == Execution::Caller {
        return Plan::Software;
    }
    let plan = reattune_plan::<EXACT, DOUBLE>();
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
#[archmage::attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
fn upgrade(token: Token) -> Plan {
    select(token, Execution::Caller, false)
}

#[inline(always)]
pub(super) fn select<T: IntoConcreteToken>(token: T, execution: Execution, widen_v3: bool) -> Plan {
    select_impl::<T, true, false>(token, execution, widen_v3)
}

#[inline(always)]
pub(super) fn select_relaxed<T: IntoConcreteToken>(
    token: T,
    execution: Execution,
    widen_v3: bool,
) -> Plan {
    select_impl::<T, false, false>(token, execution, widen_v3)
}

#[archmage::attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
fn upgrade_relaxed(token: Token) -> Plan {
    select_impl::<_, false, false>(token, Execution::Caller, false)
}

macro_rules! run_cfg {
    ($name:ident, $v3:ident, $v4:ident, $v4x:ident, $neon:ident, $wasm:ident, $scalar:ident, $n:literal) => {
        #[cfg_attr(debug_assertions, inline(never))]
        #[cfg_attr(not(debug_assertions), inline(always))]
        pub(super) fn $name(
            plan: Plan,
            backend: bool,
            bases: [f32; $n],
            exponent: f32,
            cfg: graph::KernelCfg,
        ) -> Result<[f32; $n], PowfError> {
            match plan {
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                Plan::V3(t) => graph::$v3::<{ graph::NATIVE }>(t, bases, exponent, cfg),
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
                Plan::V4(t) => graph::$v4::<{ graph::NATIVE }>(t, bases, exponent, cfg),
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
                Plan::V4x(t) => graph::$v4x::<{ graph::NATIVE }>(t, bases, exponent, cfg),
                #[cfg(target_arch = "aarch64")]
                Plan::Neon(t) => graph::$neon::<{ graph::NATIVE }>(t, bases, exponent, cfg),
                #[cfg(target_arch = "wasm32")]
                Plan::Wasm(t, true) => graph::$wasm::<{ graph::NATIVE }>(t, bases, exponent, cfg),
                #[cfg(target_arch = "wasm32")]
                Plan::Wasm(t, false) => {
                    graph::$wasm::<{ graph::WASM_SOFTWARE }>(t, bases, exponent, cfg)
                }
                Plan::Software => {
                    if backend {
                        graph::$scalar::<{ graph::RELAXED }>(ScalarToken, bases, exponent, cfg)
                    } else {
                        graph::$scalar::<{ graph::SOFTWARE }>(ScalarToken, bases, exponent, cfg)
                    }
                }
                #[cfg(all(feature = "std", not(miri)))]
                Plan::Std => graph::$scalar::<{ graph::STD }>(ScalarToken, bases, exponent, cfg),
            }
        }
    };
}
run_cfg!(
    run4_cfg,
    kernel4_cfg_v3,
    kernel4_cfg_v4,
    kernel4_cfg_v4x,
    kernel4_cfg_neon,
    kernel4_cfg_wasm128,
    kernel4_cfg_scalar,
    4
);
run_cfg!(
    run8_cfg,
    kernel8_cfg_v3,
    kernel8_cfg_v4,
    kernel8_cfg_v4x,
    kernel8_cfg_neon,
    kernel8_cfg_wasm128,
    kernel8_cfg_scalar,
    8
);
#[cfg(feature = "w512")]
run_cfg!(
    run16_cfg,
    kernel16_cfg_v3,
    kernel16_cfg_v4,
    kernel16_cfg_v4x,
    kernel16_cfg_neon,
    kernel16_cfg_wasm128,
    kernel16_cfg_scalar,
    16
);

macro_rules! run_const {
    ($name:ident, $cfg_fn:ident, $v3:ident, $v4:ident, $v4x:ident, $neon:ident, $wasm:ident, $scalar:ident, $n:literal) => {
        #[inline(always)]
        pub(super) fn $name<K: graph::Kernel, const BACKEND: bool>(
            plan: Plan,
            bases: [f32; $n],
            exponent: f32,
        ) -> Result<[f32; $n], PowfError> {
            #[cfg(debug_assertions)]
            {
                return $cfg_fn(plan, BACKEND, bases, exponent, graph::cfg_of::<K>());
            }
            #[cfg(not(debug_assertions))]
            match plan {
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                Plan::V3(t) => graph::$v3::<K, { graph::NATIVE }>(t, bases, exponent),
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
                Plan::V4(t) => graph::$v4::<K, { graph::NATIVE }>(t, bases, exponent),
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
                Plan::V4x(t) => graph::$v4x::<K, { graph::NATIVE }>(t, bases, exponent),
                #[cfg(target_arch = "aarch64")]
                Plan::Neon(t) => graph::$neon::<K, { graph::NATIVE }>(t, bases, exponent),
                #[cfg(target_arch = "wasm32")]
                Plan::Wasm(t, true) => graph::$wasm::<K, { graph::NATIVE }>(t, bases, exponent),
                #[cfg(target_arch = "wasm32")]
                Plan::Wasm(t, false) => {
                    graph::$wasm::<K, { graph::WASM_SOFTWARE }>(t, bases, exponent)
                }
                Plan::Software => {
                    if BACKEND {
                        graph::$scalar::<K, { graph::RELAXED }>(ScalarToken, bases, exponent)
                    } else {
                        graph::$scalar::<K, { graph::SOFTWARE }>(ScalarToken, bases, exponent)
                    }
                }
                #[cfg(all(feature = "std", not(miri)))]
                Plan::Std => graph::$scalar::<K, { graph::STD }>(ScalarToken, bases, exponent),
            }
        }
    };
}
run_const!(
    run4_const,
    run4_cfg,
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
    run8_cfg,
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
    run16_cfg,
    kernel16_with_v3,
    kernel16_with_v4,
    kernel16_with_v4x,
    kernel16_with_neon,
    kernel16_with_wasm128,
    kernel16_with_scalar,
    16
);

#[archmage::attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
fn upgrade_highp(token: Token) -> Plan {
    select_impl::<_, true, true>(token, Execution::Caller, false)
}

#[inline(always)]
pub(super) fn select_highp<T: IntoConcreteToken>(
    token: T,
    execution: Execution,
    widen_v3: bool,
) -> Plan {
    select_impl::<T, true, true>(token, execution, widen_v3)
}

#[inline(always)]
pub(super) fn run1_cfg(
    plan: Plan,
    input: f32,
    exponent: f32,
    backend: bool,
    cfg: graph::KernelCfg,
) -> Result<f32, PowfError> {
    if matches!(plan, Plan::Software) {
        return if backend {
            graph::kernel1_cfg::<{ graph::RELAXED }>(input, exponent, cfg)
        } else {
            graph::kernel1_cfg::<{ graph::SOFTWARE }>(input, exponent, cfg)
        };
    }
    #[cfg(all(feature = "std", not(miri)))]
    if matches!(plan, Plan::Std) {
        return graph::kernel1_cfg::<{ graph::STD }>(input, exponent, cfg);
    }
    run4_cfg(plan, backend, [input; 4], exponent, cfg)
        .map(|result| result[0])
        .map_err(|error| PowfError {
            lane_mask: 1,
            ..error
        })
}

#[inline(always)]
pub(super) fn run1_const<K: graph::Kernel, const BACKEND: bool>(
    plan: Plan,
    input: f32,
    exponent: f32,
) -> Result<f32, PowfError> {
    if matches!(plan, Plan::Software) {
        return if BACKEND {
            graph::kernel1_with::<K, { graph::RELAXED }>(input, exponent)
        } else {
            graph::kernel1_with::<K, { graph::SOFTWARE }>(input, exponent)
        };
    }
    #[cfg(all(feature = "std", not(miri)))]
    if matches!(plan, Plan::Std) {
        return graph::kernel1_with::<K, { graph::STD }>(input, exponent);
    }
    run4_const::<K, BACKEND>(plan, [input; 4], exponent)
        .map(|result| result[0])
        .map_err(|error| PowfError {
            lane_mask: 1,
            ..error
        })
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn run_slice_cfg(
    plan: Plan,
    backend: bool,
    src: &[f32],
    dst: &mut [f32],
    exponent: f32,
    cfg: graph::KernelCfg,
) -> Result<(), (usize, PowfError)> {
    match plan {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        Plan::V3(t) => graph::slice8_cfg_v3_t::<{ graph::NATIVE }>(t, src, dst, exponent, cfg),
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4(t) => graph::slice16_cfg_v4::<{ graph::NATIVE }>(t, src, dst, exponent, cfg),
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4x(t) => graph::slice16_cfg_v4x::<{ graph::NATIVE }>(t, src, dst, exponent, cfg),
        #[cfg(target_arch = "aarch64")]
        Plan::Neon(t) => graph::slice8_cfg_neon_t::<{ graph::NATIVE }>(t, src, dst, exponent, cfg),
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(t, true) => {
            graph::slice4_cfg_wasm128_t::<{ graph::NATIVE }>(t, src, dst, exponent, cfg)
        }
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(_, false) => {
            graph::slice1_cfg_scalar::<{ graph::SOFTWARE }>(src, dst, exponent, cfg)
        }
        Plan::Software => {
            if backend {
                graph::slice1_cfg_scalar::<{ graph::RELAXED }>(src, dst, exponent, cfg)
            } else {
                graph::slice1_cfg_scalar::<{ graph::SOFTWARE }>(src, dst, exponent, cfg)
            }
        }
        #[cfg(all(feature = "std", not(miri)))]
        Plan::Std => graph::slice1_cfg_scalar::<{ graph::STD }>(src, dst, exponent, cfg),
    }
}

#[inline(always)]
pub(super) fn run_slice<K: graph::Kernel, const BACKEND: bool>(
    plan: Plan,
    src: &[f32],
    dst: &mut [f32],
    exponent: f32,
) -> Result<(), (usize, PowfError)> {
    #[cfg(debug_assertions)]
    {
        run_slice_cfg(plan, BACKEND, src, dst, exponent, graph::cfg_of::<K>())
    }
    #[cfg(not(debug_assertions))]
    match plan {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        Plan::V3(t) => graph::slice8_with_v3_t::<K, { graph::NATIVE }>(t, src, dst, exponent),
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4(t) => graph::slice16_with_v4::<K, { graph::NATIVE }>(t, src, dst, exponent),
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4x(t) => graph::slice16_with_v4x::<K, { graph::NATIVE }>(t, src, dst, exponent),
        #[cfg(target_arch = "aarch64")]
        Plan::Neon(t) => graph::slice8_with_neon_t::<K, { graph::NATIVE }>(t, src, dst, exponent),
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(t, true) => {
            graph::slice4_with_wasm128_t::<K, { graph::NATIVE }>(t, src, dst, exponent)
        }
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(_, false) => {
            graph::slice1_with_scalar::<K, { graph::SOFTWARE }>(src, dst, exponent)
        }
        Plan::Software => {
            if BACKEND {
                graph::slice1_with_scalar::<K, { graph::RELAXED }>(src, dst, exponent)
            } else {
                graph::slice1_with_scalar::<K, { graph::SOFTWARE }>(src, dst, exponent)
            }
        }
        #[cfg(all(feature = "std", not(miri)))]
        Plan::Std => graph::slice1_with_scalar::<K, { graph::STD }>(src, dst, exponent),
    }
}

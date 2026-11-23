use super::{PowfError, graph};
use crate::Execution;
pub(super) use crate::plan::Plan;
use archmage::{IntoConcreteToken, ScalarToken};

#[inline(always)]
fn std_available() -> bool {
    #[cfg(all(feature = "std", not(miri)))]
    {
        super::std_probe::available()
    }
    #[cfg(not(all(feature = "std", not(miri))))]
    {
        false
    }
}

#[inline(always)]
pub(super) fn select<T: IntoConcreteToken>(token: T, execution: Execution, widen_v3: bool) -> Plan {
    crate::plan::select_powf::<T, true, false>(token, execution, widen_v3, std_available)
}

#[inline(always)]
pub(super) fn select_relaxed<T: IntoConcreteToken>(
    token: T,
    execution: Execution,
    widen_v3: bool,
) -> Plan {
    crate::plan::select_powf::<T, false, false>(token, execution, widen_v3, std_available)
}

#[inline(always)]
pub(super) fn select_highp<T: IntoConcreteToken>(
    token: T,
    execution: Execution,
    widen_v3: bool,
) -> Plan {
    crate::plan::select_powf::<T, true, true>(token, execution, widen_v3, std_available)
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
                        graph::$scalar::<{ graph::NATIVE }>(ScalarToken, bases, exponent, cfg)
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
    kernel4_cfg_v3_t,
    kernel4_cfg_v4_t,
    kernel4_cfg_v4x_t,
    kernel4_cfg_neon_t,
    kernel4_cfg_wasm128_t,
    kernel4_cfg_scalar_t,
    4
);
run_cfg!(
    run8_cfg,
    kernel8_cfg_v3_t,
    kernel8_cfg_v4_t,
    kernel8_cfg_v4x_t,
    kernel8_cfg_neon_t,
    kernel8_cfg_wasm128_t,
    kernel8_cfg_scalar_t,
    8
);
#[cfg(feature = "w512")]
run_cfg!(
    run16_cfg,
    kernel16_cfg_v3_t,
    kernel16_cfg_v4_t,
    kernel16_cfg_v4x_t,
    kernel16_cfg_neon_t,
    kernel16_cfg_wasm128_t,
    kernel16_cfg_scalar_t,
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
                        graph::$scalar::<K, { graph::NATIVE }>(ScalarToken, bases, exponent)
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
    kernel4_with_v3_t,
    kernel4_with_v4_t,
    kernel4_with_v4x_t,
    kernel4_with_neon_t,
    kernel4_with_wasm128_t,
    kernel4_with_scalar_t,
    4
);
run_const!(
    run8_const,
    run8_cfg,
    kernel8_with_v3_t,
    kernel8_with_v4_t,
    kernel8_with_v4x_t,
    kernel8_with_neon_t,
    kernel8_with_wasm128_t,
    kernel8_with_scalar_t,
    8
);
#[cfg(feature = "w512")]
run_const!(
    run16_const,
    run16_cfg,
    kernel16_with_v3_t,
    kernel16_with_v4_t,
    kernel16_with_v4x_t,
    kernel16_with_neon_t,
    kernel16_with_wasm128_t,
    kernel16_with_scalar_t,
    16
);

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
            graph::kernel1_cfg::<{ graph::NATIVE }>(input, exponent, cfg)
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
            graph::kernel1_with::<K, { graph::NATIVE }>(input, exponent)
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
        Plan::V4(t) => graph::slice16_cfg_v4_t::<{ graph::NATIVE }>(t, src, dst, exponent, cfg),
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4x(t) => graph::slice16_cfg_v4x_t::<{ graph::NATIVE }>(t, src, dst, exponent, cfg),
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
                graph::slice1_cfg_scalar::<{ graph::NATIVE }>(src, dst, exponent, cfg)
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
        Plan::V4(t) => graph::slice16_with_v4_t::<K, { graph::NATIVE }>(t, src, dst, exponent),
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4x(t) => graph::slice16_with_v4x_t::<K, { graph::NATIVE }>(t, src, dst, exponent),
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
                graph::slice1_with_scalar::<K, { graph::NATIVE }>(src, dst, exponent)
            } else {
                graph::slice1_with_scalar::<K, { graph::SOFTWARE }>(src, dst, exponent)
            }
        }
        #[cfg(all(feature = "std", not(miri)))]
        Plan::Std => graph::slice1_with_scalar::<K, { graph::STD }>(src, dst, exponent),
    }
}

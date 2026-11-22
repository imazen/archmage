//! One-time operation selection. The chosen proof is reused for the whole
//! polynomial or slice, never per operation.
use super::{LogfError, graph};
use crate::Execution;
pub(crate) use crate::plan::Plan;
use archmage::{IntoConcreteToken, ScalarToken};

#[inline(always)]
pub(crate) fn select<T: IntoConcreteToken>(
    token: T,
    execution: Execution,
    backend: bool,
    widen_v3: bool,
) -> Plan {
    crate::plan::select_f32(token, execution, backend, widen_v3)
}

macro_rules! run_cfg {
    ($name:ident, $v3:ident, $v4:ident, $v4x:ident, $neon:ident, $wasm:ident, $scalar:ident, $n:literal) => {
        #[cfg_attr(debug_assertions, inline(never))]
        #[cfg_attr(not(debug_assertions), inline(always))]
        pub(crate) fn $name(
            plan: Plan,
            input: [f32; $n],
            op: u8,
            backend: bool,
            cfg: graph::KernelCfg,
        ) -> Result<[f32; $n], LogfError> {
            match plan {
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                Plan::V3(t) => graph::$v3::<{ graph::NATIVE }>(t, input, op, cfg),
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
                Plan::V4(t) => graph::$v4::<{ graph::NATIVE }>(t, input, op, cfg),
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
                Plan::V4x(t) => graph::$v4x::<{ graph::NATIVE }>(t, input, op, cfg),
                #[cfg(target_arch = "aarch64")]
                Plan::Neon(t) => graph::$neon::<{ graph::NATIVE }>(t, input, op, cfg),
                #[cfg(target_arch = "wasm32")]
                Plan::Wasm(t, true) => graph::$wasm::<{ graph::NATIVE }>(t, input, op, cfg),
                #[cfg(target_arch = "wasm32")]
                Plan::Wasm(t, false) => graph::$wasm::<{ graph::WASM_SOFTWARE }>(t, input, op, cfg),
                Plan::Software => {
                    if backend {
                        graph::$scalar::<{ graph::NATIVE }>(ScalarToken, input, op, cfg)
                    } else {
                        graph::$scalar::<{ graph::SOFTWARE }>(ScalarToken, input, op, cfg)
                    }
                }
                #[cfg(all(feature = "std", not(miri)))]
                Plan::Std => {
                    if backend {
                        graph::$scalar::<{ graph::NATIVE }>(ScalarToken, input, op, cfg)
                    } else {
                        graph::$scalar::<{ graph::SOFTWARE }>(ScalarToken, input, op, cfg)
                    }
                }
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
        pub(crate) fn $name<K: graph::Kernel, const OP: u8, const BACKEND: bool>(
            plan: Plan,
            input: [f32; $n],
        ) -> Result<[f32; $n], LogfError> {
            #[cfg(debug_assertions)]
            {
                return $cfg_fn(plan, input, OP, BACKEND, graph::cfg_of::<K>());
            }
            #[cfg(not(debug_assertions))]
            match plan {
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                Plan::V3(t) => graph::$v3::<K, OP, { graph::NATIVE }>(t, input),
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
                Plan::V4(t) => graph::$v4::<K, OP, { graph::NATIVE }>(t, input),
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
                Plan::V4x(t) => graph::$v4x::<K, OP, { graph::NATIVE }>(t, input),
                #[cfg(target_arch = "aarch64")]
                Plan::Neon(t) => graph::$neon::<K, OP, { graph::NATIVE }>(t, input),
                #[cfg(target_arch = "wasm32")]
                Plan::Wasm(t, true) => graph::$wasm::<K, OP, { graph::NATIVE }>(t, input),
                #[cfg(target_arch = "wasm32")]
                Plan::Wasm(t, false) => graph::$wasm::<K, OP, { graph::WASM_SOFTWARE }>(t, input),
                Plan::Software => {
                    if BACKEND {
                        graph::$scalar::<K, OP, { graph::NATIVE }>(ScalarToken, input)
                    } else {
                        graph::$scalar::<K, OP, { graph::SOFTWARE }>(ScalarToken, input)
                    }
                }
                #[cfg(all(feature = "std", not(miri)))]
                Plan::Std => {
                    if BACKEND {
                        graph::$scalar::<K, OP, { graph::NATIVE }>(ScalarToken, input)
                    } else {
                        graph::$scalar::<K, OP, { graph::SOFTWARE }>(ScalarToken, input)
                    }
                }
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
pub(crate) fn run1_cfg(
    plan: Plan,
    input: f32,
    op: u8,
    backend: bool,
    cfg: graph::KernelCfg,
) -> Result<f32, LogfError> {
    if matches!(plan, Plan::Software) {
        return if backend {
            graph::kernel1_cfg::<{ graph::NATIVE }>(input, op, cfg)
        } else {
            graph::kernel1_cfg::<{ graph::SOFTWARE }>(input, op, cfg)
        };
    }
    run4_cfg(plan, [input; 4], op, backend, cfg)
        .map(|result| result[0])
        .map_err(|error| LogfError {
            lane_mask: 1,
            ..error
        })
}

#[inline(always)]
pub(crate) fn run1_const<K: graph::Kernel, const OP: u8, const BACKEND: bool>(
    plan: Plan,
    input: f32,
) -> Result<f32, LogfError> {
    if matches!(plan, Plan::Software) {
        return if BACKEND {
            graph::kernel1_with::<K, OP, { graph::NATIVE }>(input)
        } else {
            graph::kernel1_with::<K, OP, { graph::SOFTWARE }>(input)
        };
    }
    run4_const::<K, OP, BACKEND>(plan, [input; 4])
        .map(|result| result[0])
        .map_err(|error| LogfError {
            lane_mask: 1,
            ..error
        })
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) fn run_slice_cfg(
    plan: Plan,
    src: &[f32],
    dst: &mut [f32],
    op: u8,
    backend: bool,
    cfg: graph::KernelCfg,
) -> Result<(), (usize, LogfError)> {
    match plan {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        Plan::V3(t) => graph::slice8_cfg_v3_t::<{ graph::NATIVE }>(t, src, dst, op, cfg),
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4(t) => graph::slice16_cfg_v4_t::<{ graph::NATIVE }>(t, src, dst, op, cfg),
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4x(t) => graph::slice16_cfg_v4x_t::<{ graph::NATIVE }>(t, src, dst, op, cfg),
        #[cfg(target_arch = "aarch64")]
        Plan::Neon(t) => graph::slice8_cfg_neon_t::<{ graph::NATIVE }>(t, src, dst, op, cfg),
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(t, true) => {
            graph::slice4_cfg_wasm128_t::<{ graph::NATIVE }>(t, src, dst, op, cfg)
        }
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(_, false) => graph::slice1_cfg_scalar::<{ graph::SOFTWARE }>(src, dst, op, cfg),
        Plan::Software => {
            if backend {
                graph::slice1_cfg_scalar::<{ graph::NATIVE }>(src, dst, op, cfg)
            } else {
                graph::slice1_cfg_scalar::<{ graph::SOFTWARE }>(src, dst, op, cfg)
            }
        }
        #[cfg(all(feature = "std", not(miri)))]
        Plan::Std => {
            if backend {
                graph::slice1_cfg_scalar::<{ graph::NATIVE }>(src, dst, op, cfg)
            } else {
                graph::slice1_cfg_scalar::<{ graph::SOFTWARE }>(src, dst, op, cfg)
            }
        }
    }
}

#[inline(always)]
pub(crate) fn run_slice_const<K: graph::Kernel, const OP: u8, const BACKEND: bool>(
    plan: Plan,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, LogfError)> {
    #[cfg(debug_assertions)]
    {
        run_slice_cfg(plan, src, dst, OP, BACKEND, graph::cfg_of::<K>())
    }
    #[cfg(not(debug_assertions))]
    match plan {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        Plan::V3(t) => graph::slice8_with_v3_t::<K, OP, { graph::NATIVE }>(t, src, dst),
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4(t) => graph::slice16_with_v4_t::<K, OP, { graph::NATIVE }>(t, src, dst),
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4x(t) => graph::slice16_with_v4x_t::<K, OP, { graph::NATIVE }>(t, src, dst),
        #[cfg(target_arch = "aarch64")]
        Plan::Neon(t) => graph::slice8_with_neon_t::<K, OP, { graph::NATIVE }>(t, src, dst),
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(t, true) => {
            graph::slice4_with_wasm128_t::<K, OP, { graph::NATIVE }>(t, src, dst)
        }
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(_, false) => graph::slice1_with_scalar::<K, OP, { graph::SOFTWARE }>(src, dst),
        Plan::Software => {
            if BACKEND {
                graph::slice1_with_scalar::<K, OP, { graph::NATIVE }>(src, dst)
            } else {
                graph::slice1_with_scalar::<K, OP, { graph::SOFTWARE }>(src, dst)
            }
        }
        #[cfg(all(feature = "std", not(miri)))]
        Plan::Std => {
            if BACKEND {
                graph::slice1_with_scalar::<K, OP, { graph::NATIVE }>(src, dst)
            } else {
                graph::slice1_with_scalar::<K, OP, { graph::SOFTWARE }>(src, dst)
            }
        }
    }
}

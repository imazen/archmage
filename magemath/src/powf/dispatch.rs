use super::{PowfError, PowfExecution, graph};
use archmage::{IntoConcreteToken, ScalarToken};

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
fn select_impl<T: IntoConcreteToken, const EXACT: bool, const DOUBLE: bool>(
    token: T,
    execution: PowfExecution,
) -> Plan {
    let _ = token;
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = token.as_x64v3() {
            return Plan::V3(t);
        }
        if let Some(t) = token.as_x64v3_crypto() {
            return Plan::V3(t.v3());
        }
        if let Some(t) = token.as_x64v3_gfni_crypto() {
            return Plan::V3(t.v3());
        }
        #[cfg(feature = "avx512")]
        {
            if let Some(t) = token.as_x64v4() {
                return Plan::V4(t);
            }
            if let Some(t) = token.as_x64v4x() {
                return Plan::V4x(t);
            }
            if let Some(t) = token.as_avx512_fp16() {
                return Plan::V4(t.v4());
            }
        }
        #[cfg(not(feature = "avx512"))]
        {
            if let Some(t) = token.as_x64v4() {
                return Plan::V3(t.v3());
            }
            if let Some(t) = token.as_x64v4x() {
                return Plan::V3(t.v3());
            }
            if let Some(t) = token.as_avx512_fp16() {
                return Plan::V3(t.v3());
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if let Some(t) = token.as_neon() {
            return Plan::Neon(t);
        }
        if let Some(t) = token.as_neon_aes() {
            return Plan::Neon(t.neon());
        }
        if let Some(t) = token.as_neon_sha3() {
            return Plan::Neon(t.neon());
        }
        if let Some(t) = token.as_neon_crc() {
            return Plan::Neon(t.neon());
        }
        if let Some(t) = token.as_arm_v2() {
            return Plan::Neon(t.neon());
        }
        if let Some(t) = token.as_arm_v3() {
            return Plan::Neon(t.neon());
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(t) = token
            .as_wasm128()
            .or_else(|| token.as_wasm128_relaxed().map(|r| r.wasm128()))
        {
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
pub(super) fn select<T: IntoConcreteToken>(token: T, execution: PowfExecution) -> Plan {
    select_impl::<T, true, false>(token, execution)
}

#[inline(always)]
pub(super) fn select_relaxed<T: IntoConcreteToken>(token: T, execution: PowfExecution) -> Plan {
    select_impl::<T, false, false>(token, execution)
}

#[archmage::attune(_*_t)]
fn upgrade_relaxed(token: Token) -> Plan {
    select_impl::<_, false, false>(token, PowfExecution::Caller)
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
                #[cfg(target_arch = "x86_64")]
                Plan::V3(t) => graph::$v3::<{ graph::NATIVE }>(t, bases, exponent, cfg),
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                Plan::V4(t) => graph::$v4::<{ graph::NATIVE }>(t, bases, exponent, cfg),
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
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
                #[cfg(target_arch = "x86_64")]
                Plan::V3(t) => graph::$v3::<K, { graph::NATIVE }>(t, bases, exponent),
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                Plan::V4(t) => graph::$v4::<K, { graph::NATIVE }>(t, bases, exponent),
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
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

#[archmage::attune(_*_t)]
fn upgrade_highp(token: Token) -> Plan {
    select_impl::<_, true, true>(token, PowfExecution::Caller)
}

#[inline(always)]
pub(super) fn select_highp<T: IntoConcreteToken>(token: T, execution: PowfExecution) -> Plan {
    select_impl::<T, true, true>(token, execution)
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
        let cfg = graph::cfg_of::<K>();
        match plan {
            #[cfg(target_arch = "x86_64")]
            Plan::V3(t) => graph::slice8_cfg_v3::<{ graph::NATIVE }>(t, src, dst, exponent, cfg),
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            Plan::V4(t) => graph::slice8_cfg_v4::<{ graph::NATIVE }>(t, src, dst, exponent, cfg),
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            Plan::V4x(t) => graph::slice8_cfg_v4x::<{ graph::NATIVE }>(t, src, dst, exponent, cfg),
            #[cfg(target_arch = "aarch64")]
            Plan::Neon(t) => {
                graph::slice8_cfg_neon::<{ graph::NATIVE }>(t, src, dst, exponent, cfg)
            }
            #[cfg(target_arch = "wasm32")]
            Plan::Wasm(t, true) => {
                graph::slice8_cfg_wasm128::<{ graph::NATIVE }>(t, src, dst, exponent, cfg)
            }
            #[cfg(target_arch = "wasm32")]
            Plan::Wasm(t, false) => {
                graph::slice8_cfg_wasm128::<{ graph::WASM_SOFTWARE }>(t, src, dst, exponent, cfg)
            }
            Plan::Software => {
                if BACKEND {
                    graph::slice8_cfg_scalar::<{ graph::RELAXED }>(
                        ScalarToken,
                        src,
                        dst,
                        exponent,
                        cfg,
                    )
                } else {
                    graph::slice8_cfg_scalar::<{ graph::SOFTWARE }>(
                        ScalarToken,
                        src,
                        dst,
                        exponent,
                        cfg,
                    )
                }
            }
            #[cfg(all(feature = "std", not(miri)))]
            Plan::Std => {
                graph::slice8_cfg_scalar::<{ graph::STD }>(ScalarToken, src, dst, exponent, cfg)
            }
        }
    }
    #[cfg(not(debug_assertions))]
    match plan {
        #[cfg(target_arch = "x86_64")]
        Plan::V3(t) => graph::slice8_with_v3::<K, { graph::NATIVE }>(t, src, dst, exponent),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Plan::V4(t) => graph::slice8_with_v4::<K, { graph::NATIVE }>(t, src, dst, exponent),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Plan::V4x(t) => graph::slice8_with_v4x::<K, { graph::NATIVE }>(t, src, dst, exponent),
        #[cfg(target_arch = "aarch64")]
        Plan::Neon(t) => graph::slice8_with_neon::<K, { graph::NATIVE }>(t, src, dst, exponent),
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(t, true) => {
            graph::slice8_with_wasm128::<K, { graph::NATIVE }>(t, src, dst, exponent)
        }
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(t, false) => {
            graph::slice8_with_wasm128::<K, { graph::WASM_SOFTWARE }>(t, src, dst, exponent)
        }
        Plan::Software => {
            if BACKEND {
                graph::slice8_with_scalar::<K, { graph::RELAXED }>(ScalarToken, src, dst, exponent)
            } else {
                graph::slice8_with_scalar::<K, { graph::SOFTWARE }>(ScalarToken, src, dst, exponent)
            }
        }
        #[cfg(all(feature = "std", not(miri)))]
        Plan::Std => {
            graph::slice8_with_scalar::<K, { graph::STD }>(ScalarToken, src, dst, exponent)
        }
    }
}

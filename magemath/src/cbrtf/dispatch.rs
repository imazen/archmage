//! One-time operation selection for `cbrtf`.
use super::{CbrtfError, graph};
use crate::Execution;
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
    Wasm(archmage::Wasm128Token),
    Software,
}

impl Plan {
    #[inline(always)]
    pub(super) fn same_tier(self, other: Self) -> bool {
        core::mem::discriminant(&self) == core::mem::discriminant(&other)
    }
}

#[inline(always)]
fn reattune_plan() -> Plan {
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
}

#[inline(always)]
pub(super) fn select<T: IntoConcreteToken>(token: T, execution: Execution, widen_v3: bool) -> Plan {
    let _ = (token, widen_v3);
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if let Some(t) = token.as_x64v4x() {
                return Plan::V4x(t);
            }
            if let Some(t) = token.as_x64v4() {
                return Plan::V4(t);
            }
            if let Some(t) = token.as_avx512_fp16() {
                return Plan::V4(t.v4());
            }
        }
        #[cfg(not(feature = "avx512"))]
        {
            if let Some(t) = token.as_x64v4x() {
                return Plan::V3(t.v3());
            }
            if let Some(t) = token.as_x64v4() {
                return Plan::V3(t.v3());
            }
            if let Some(t) = token.as_avx512_fp16() {
                return Plan::V3(t.v3());
            }
        }
        if let Some(v3) = token
            .as_x64v3()
            .or_else(|| token.as_x64v3_crypto().map(|t| t.v3()))
            .or_else(|| token.as_x64v3_gfni_crypto().map(|t| t.v3()))
        {
            #[cfg(feature = "avx512")]
            if widen_v3 && execution == Execution::Upgrade {
                let upgraded = reattune_plan();
                if matches!(upgraded, Plan::V4(_) | Plan::V4x(_)) {
                    return upgraded;
                }
            }
            return Plan::V3(v3);
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
            return Plan::Wasm(t);
        }
    }
    if execution == Execution::Caller {
        return Plan::Software;
    }
    reattune_plan()
}

#[archmage::attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
fn upgrade(token: Token) -> Plan {
    select(token, Execution::Caller, false)
}

macro_rules! run_cfg {
    ($name:ident, $v3:ident, $v4:ident, $v4x:ident, $neon:ident, $wasm:ident, $scalar:ident, $n:literal) => {
        #[cfg_attr(debug_assertions, inline(never))]
        #[cfg_attr(not(debug_assertions), inline(always))]
        pub(super) fn $name(
            plan: Plan,
            input: [f32; $n],
            backend: bool,
            cfg: graph::KernelCfg,
        ) -> Result<[f32; $n], CbrtfError> {
            match plan {
                #[cfg(target_arch = "x86_64")]
                Plan::V3(t) => graph::$v3(t, input, backend, cfg),
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                Plan::V4(t) => graph::$v4(t, input, backend, cfg),
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                Plan::V4x(t) => graph::$v4x(t, input, backend, cfg),
                #[cfg(target_arch = "aarch64")]
                Plan::Neon(t) => graph::$neon(t, input, backend, cfg),
                #[cfg(target_arch = "wasm32")]
                Plan::Wasm(t) => graph::$wasm(t, input, backend, cfg),
                Plan::Software => graph::$scalar(ScalarToken, input, backend, cfg),
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
            input: [f32; $n],
        ) -> Result<[f32; $n], CbrtfError> {
            #[cfg(debug_assertions)]
            {
                return $cfg_fn(plan, input, BACKEND, graph::cfg_of::<K>());
            }
            #[cfg(not(debug_assertions))]
            match plan {
                #[cfg(target_arch = "x86_64")]
                Plan::V3(t) => graph::$v3::<K, BACKEND>(t, input),
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                Plan::V4(t) => graph::$v4::<K, BACKEND>(t, input),
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                Plan::V4x(t) => graph::$v4x::<K, BACKEND>(t, input),
                #[cfg(target_arch = "aarch64")]
                Plan::Neon(t) => graph::$neon::<K, BACKEND>(t, input),
                #[cfg(target_arch = "wasm32")]
                Plan::Wasm(t) => graph::$wasm::<K, BACKEND>(t, input),
                Plan::Software => graph::$scalar::<K, BACKEND>(ScalarToken, input),
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

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn run_slice_cfg(
    plan: Plan,
    src: &[f32],
    dst: &mut [f32],
    backend: bool,
    cfg: graph::KernelCfg,
) -> Result<(), (usize, CbrtfError)> {
    match plan {
        #[cfg(target_arch = "x86_64")]
        Plan::V3(t) => graph::slice8_cfg_v3(t, src, dst, backend, cfg),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Plan::V4(t) => graph::slice16_cfg_v4(t, src, dst, backend, cfg),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Plan::V4x(t) => graph::slice16_cfg_v4x(t, src, dst, backend, cfg),
        #[cfg(target_arch = "aarch64")]
        Plan::Neon(t) => graph::slice8_cfg_neon(t, src, dst, backend, cfg),
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(t) => graph::slice8_cfg_wasm128(t, src, dst, backend, cfg),
        Plan::Software => graph::slice8_cfg_scalar(ScalarToken, src, dst, backend, cfg),
    }
}

#[inline(always)]
pub(super) fn run_slice_const<K: graph::Kernel, const BACKEND: bool>(
    plan: Plan,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, CbrtfError)> {
    #[cfg(debug_assertions)]
    {
        run_slice_cfg(plan, src, dst, BACKEND, graph::cfg_of::<K>())
    }
    #[cfg(not(debug_assertions))]
    match plan {
        #[cfg(target_arch = "x86_64")]
        Plan::V3(t) => graph::slice8_with_v3::<K, BACKEND>(t, src, dst),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Plan::V4(t) => graph::slice16_with_v4::<K, BACKEND>(t, src, dst),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Plan::V4x(t) => graph::slice16_with_v4x::<K, BACKEND>(t, src, dst),
        #[cfg(target_arch = "aarch64")]
        Plan::Neon(t) => graph::slice8_with_neon::<K, BACKEND>(t, src, dst),
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(t) => graph::slice8_with_wasm128::<K, BACKEND>(t, src, dst),
        Plan::Software => graph::slice8_with_scalar::<K, BACKEND>(ScalarToken, src, dst),
    }
}

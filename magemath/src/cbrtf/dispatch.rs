//! One-time operation selection for `cbrtf`.
use super::{CbrtfError, graph};
use crate::Execution;
pub(crate) use crate::plan::Plan;
use archmage::{IntoConcreteToken, ScalarToken};

#[inline(always)]
pub(crate) fn select<T: IntoConcreteToken>(token: T, execution: Execution, widen_v3: bool) -> Plan {
    crate::plan::select_tier(token, execution, widen_v3)
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
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                Plan::V3(t) => graph::$v3(t, input, backend, cfg),
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
                Plan::V4(t) => graph::$v4(t, input, backend, cfg),
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
                Plan::V4x(t) => graph::$v4x(t, input, backend, cfg),
                #[cfg(target_arch = "aarch64")]
                Plan::Neon(t) => graph::$neon(t, input, backend, cfg),
                #[cfg(target_arch = "wasm32")]
                Plan::Wasm(t, _) => graph::$wasm(t, input, backend, cfg),
                Plan::Software => graph::$scalar(ScalarToken, input, backend, cfg),
                #[cfg(all(feature = "std", not(miri)))]
                Plan::Std => graph::$scalar(ScalarToken, input, backend, cfg),
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
            input: [f32; $n],
        ) -> Result<[f32; $n], CbrtfError> {
            #[cfg(debug_assertions)]
            {
                return $cfg_fn(plan, input, BACKEND, graph::cfg_of::<K>());
            }
            #[cfg(not(debug_assertions))]
            match plan {
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                Plan::V3(t) => graph::$v3::<K, BACKEND>(t, input),
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
                Plan::V4(t) => graph::$v4::<K, BACKEND>(t, input),
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
                Plan::V4x(t) => graph::$v4x::<K, BACKEND>(t, input),
                #[cfg(target_arch = "aarch64")]
                Plan::Neon(t) => graph::$neon::<K, BACKEND>(t, input),
                #[cfg(target_arch = "wasm32")]
                Plan::Wasm(t, _) => graph::$wasm::<K, BACKEND>(t, input),
                Plan::Software => graph::$scalar::<K, BACKEND>(ScalarToken, input),
                #[cfg(all(feature = "std", not(miri)))]
                Plan::Std => graph::$scalar::<K, BACKEND>(ScalarToken, input),
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
    backend: bool,
    cfg: graph::KernelCfg,
) -> Result<f32, CbrtfError> {
    if matches!(plan, Plan::Software) {
        return graph::kernel1_cfg(input, backend, cfg);
    }
    run4_cfg(plan, [input; 4], backend, cfg)
        .map(|result| result[0])
        .map_err(|error| CbrtfError {
            lane_mask: 1,
            ..error
        })
}

#[inline(always)]
pub(super) fn run1_const<K: graph::Kernel, const BACKEND: bool>(
    plan: Plan,
    input: f32,
) -> Result<f32, CbrtfError> {
    if matches!(plan, Plan::Software) {
        return graph::kernel1_with::<K, BACKEND>(input);
    }
    run4_const::<K, BACKEND>(plan, [input; 4])
        .map(|result| result[0])
        .map_err(|error| CbrtfError {
            lane_mask: 1,
            ..error
        })
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
enum WasmSlice {
    Vector {
        token: archmage::Wasm128Token,
        relaxed: bool,
    },
    Scalar,
}

#[cfg(target_arch = "wasm32")]
impl WasmSlice {
    #[inline(always)]
    fn select(token: archmage::Wasm128Token, backend: bool) -> Self {
        use archmage::{SimdToken, Wasm128RelaxedToken};

        if backend || (Wasm128RelaxedToken::summon().is_some() && crate::wasm_fma::powf_f32_fuses())
        {
            Self::Vector {
                token,
                // In testable builds the relaxed operation rechecks the token
                // at every multiply-add. Use fused arithmetic for BitExact
                // so a later toggle can only change its implementation.
                relaxed: backend || Wasm128RelaxedToken::compiled_with() == Some(true),
            }
        } else {
            Self::Scalar
        }
    }

    #[inline(always)]
    fn run_cfg(
        self,
        src: &[f32],
        dst: &mut [f32],
        cfg: graph::KernelCfg,
    ) -> Result<(), (usize, CbrtfError)> {
        match self {
            Self::Vector { token, relaxed } => {
                graph::slice4_cfg_wasm128_t(token, src, dst, relaxed, cfg)
            }
            Self::Scalar => graph::slice1_cfg_scalar(src, dst, false, cfg),
        }
    }

    #[cfg(not(debug_assertions))]
    #[inline(always)]
    fn run_const<K: graph::Kernel>(
        self,
        src: &[f32],
        dst: &mut [f32],
    ) -> Result<(), (usize, CbrtfError)> {
        match self {
            Self::Vector {
                token,
                relaxed: true,
            } => graph::slice4_with_wasm128_t::<K, true>(token, src, dst),
            Self::Vector {
                token,
                relaxed: false,
            } => graph::slice4_with_wasm128_t::<K, false>(token, src, dst),
            Self::Scalar => graph::slice1_with_scalar::<K, false>(src, dst),
        }
    }
}

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
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        Plan::V3(t) => graph::slice8_cfg_v3_t(t, src, dst, backend, cfg),
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4(t) => graph::slice16_cfg_v4_t(t, src, dst, backend, cfg),
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4x(t) => graph::slice16_cfg_v4x_t(t, src, dst, backend, cfg),
        #[cfg(target_arch = "aarch64")]
        Plan::Neon(t) => graph::slice8_cfg_neon_t(t, src, dst, backend, cfg),
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(t, _) => WasmSlice::select(t, backend).run_cfg(src, dst, cfg),
        Plan::Software => graph::slice1_cfg_scalar(src, dst, backend, cfg),
        #[cfg(all(feature = "std", not(miri)))]
        Plan::Std => graph::slice1_cfg_scalar(src, dst, backend, cfg),
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
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        Plan::V3(t) => graph::slice8_with_v3_t::<K, BACKEND>(t, src, dst),
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4(t) => graph::slice16_with_v4_t::<K, BACKEND>(t, src, dst),
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        Plan::V4x(t) => graph::slice16_with_v4x_t::<K, BACKEND>(t, src, dst),
        #[cfg(target_arch = "aarch64")]
        Plan::Neon(t) => graph::slice8_with_neon_t::<K, BACKEND>(t, src, dst),
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(t, _) => WasmSlice::select(t, BACKEND).run_const::<K>(src, dst),
        Plan::Software => graph::slice1_with_scalar::<K, BACKEND>(src, dst),
        #[cfg(all(feature = "std", not(miri)))]
        Plan::Std => graph::slice1_with_scalar::<K, BACKEND>(src, dst),
    }
}

#[cfg(all(test, target_arch = "wasm32", target_feature = "relaxed-simd"))]
mod tests {
    use super::{WasmSlice, graph};
    use crate::cbrtf::CbrtfOptions;
    use archmage::{SimdToken, Wasm128RelaxedToken, Wasm128Token};

    #[test]
    #[ignore = "requires archmage/testable_dispatch and a fusing relaxed SIMD engine"]
    fn bit_exact_slice_stays_fused_after_admission() {
        const OPTS: CbrtfOptions = CbrtfOptions::new().caller();
        let token = Wasm128Token::summon().expect("SIMD128 is enabled");
        let src = [f32::from_bits(0x3fa0_0497); 17];
        let mut runtime = [0.0f32; 17];
        #[cfg(not(debug_assertions))]
        let mut const_out = [0.0f32; 17];

        for len in [2, 4, 17] {
            let admission = WasmSlice::select(token, false);
            assert!(matches!(admission, WasmSlice::Vector { .. }));
            Wasm128RelaxedToken::dangerously_disable_token_process_wide(true).unwrap();
            assert!(Wasm128RelaxedToken::summon().is_none());
            assert!(Wasm128Token::summon().is_some());
            assert!(matches!(WasmSlice::select(token, false), WasmSlice::Scalar));

            let cfg = graph::KernelCfg::from_options(OPTS);
            admission
                .run_cfg(&src[..len], &mut runtime[..len], cfg)
                .unwrap();
            #[cfg(not(debug_assertions))]
            admission
                .run_const::<crate::cbrtf::const_api::Cfg<{ OPTS.bits() }>>(
                    &src[..len],
                    &mut const_out[..len],
                )
                .unwrap();
            Wasm128RelaxedToken::dangerously_disable_token_process_wide(false).unwrap();

            for &value in &runtime[..len] {
                assert_eq!(value.to_bits(), 0x3f89_e393, "runtime len={len}");
            }
            #[cfg(not(debug_assertions))]
            for &value in &const_out[..len] {
                assert_eq!(value.to_bits(), 0x3f89_e393, "const len={len}");
            }
        }
    }
}

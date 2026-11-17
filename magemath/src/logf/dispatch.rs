//! One-time operation selection. The chosen proof is reused for the whole
//! polynomial, never per operation.
use super::{LogfError, LogfExecution, graph};
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

#[inline(always)]
fn select_impl<T: IntoConcreteToken>(token: T, execution: LogfExecution) -> Plan {
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
            return Plan::Wasm(t);
        }
    }
    if execution == LogfExecution::Caller {
        return Plan::Software;
    }
    archmage::reattune!(upgrade(), [_v3_t, _neon_t, _wasm128_t, _scalar_t])
}

// Use the supported attunement boundary for autonomous operation selection.
#[archmage::attune(_*_t)]
fn upgrade(token: Token) -> Plan {
    select(token, LogfExecution::Caller)
}

#[inline(always)]
pub(super) fn select<T: IntoConcreteToken>(token: T, execution: LogfExecution) -> Plan {
    select_impl(token, execution)
}

macro_rules! run_const {
    ($name:ident, $v3:ident, $v4:ident, $v4x:ident, $neon:ident, $wasm:ident, $scalar:ident, $n:literal) => {
        #[inline(always)]
        pub(super) fn $name<const OP: u8, const OPTIONS: u32>(
            plan: Plan,
            input: [f32; $n],
        ) -> Result<[f32; $n], LogfError> {
            match plan {
                #[cfg(target_arch = "x86_64")]
                Plan::V3(t) => graph::$v3::<OP, OPTIONS>(t, input),
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                Plan::V4(t) => graph::$v4::<OP, OPTIONS>(t, input),
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                Plan::V4x(t) => graph::$v4x::<OP, OPTIONS>(t, input),
                #[cfg(target_arch = "aarch64")]
                Plan::Neon(t) => graph::$neon::<OP, OPTIONS>(t, input),
                #[cfg(target_arch = "wasm32")]
                Plan::Wasm(t) => graph::$wasm::<OP, OPTIONS>(t, input),
                Plan::Software => graph::$scalar::<OP, OPTIONS>(ScalarToken, input),
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

#[inline(always)]
pub(super) fn run_slice_const<const OP: u8, const OPTIONS: u32>(
    plan: Plan,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), (usize, LogfError)> {
    match plan {
        #[cfg(target_arch = "x86_64")]
        Plan::V3(t) => graph::slice8_with_v3::<OP, OPTIONS>(t, src, dst),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Plan::V4(t) => graph::slice8_with_v4::<OP, OPTIONS>(t, src, dst),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Plan::V4x(t) => graph::slice8_with_v4x::<OP, OPTIONS>(t, src, dst),
        #[cfg(target_arch = "aarch64")]
        Plan::Neon(t) => graph::slice8_with_neon::<OP, OPTIONS>(t, src, dst),
        #[cfg(target_arch = "wasm32")]
        Plan::Wasm(t) => graph::slice8_with_wasm128::<OP, OPTIONS>(t, src, dst),
        Plan::Software => graph::slice8_with_scalar::<OP, OPTIONS>(ScalarToken, src, dst),
    }
}

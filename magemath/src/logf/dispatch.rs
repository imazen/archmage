//! One-time operation selection. The chosen proof is reused for the whole
//! polynomial, never per operation.
use super::{LogfError, LogfExecution, graph};
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
    Wasm(archmage::Wasm128Token),
    Software,
}

#[inline(always)]
fn select_impl<T: SimdToken>(token: T, execution: LogfExecution) -> Plan {
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
        return Plan::Wasm(*t);
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
pub(super) fn select<T: SimdToken>(token: T, execution: LogfExecution) -> Plan {
    select_impl(token, execution)
}

macro_rules! run_const {
    ($name:ident, $v3:ident, $v4:ident, $v4x:ident, $neon:ident, $wasm:ident, $scalar:ident, $n:literal) => {
        #[inline(always)]
        pub(super) fn $name<const OPTIONS: u32>(
            plan: Plan,
            input: [f32; $n],
        ) -> Result<[f32; $n], LogfError> {
            match plan {
                #[cfg(target_arch = "x86_64")]
                Plan::V3(t) => graph::$v3::<OPTIONS>(t, input),
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                Plan::V4(t) => graph::$v4::<OPTIONS>(t, input),
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                Plan::V4x(t) => graph::$v4x::<OPTIONS>(t, input),
                #[cfg(target_arch = "aarch64")]
                Plan::Neon(t) => graph::$neon::<OPTIONS>(t, input),
                #[cfg(target_arch = "wasm32")]
                Plan::Wasm(t) => graph::$wasm::<OPTIONS>(t, input),
                Plan::Software => graph::$scalar::<OPTIONS>(ScalarToken, input),
            }
        }
    };
}
run_const!(
    run4_const,
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
    kernel16_with_v3_t,
    kernel16_with_v4_t,
    kernel16_with_v4x_t,
    kernel16_with_neon_t,
    kernel16_with_wasm128_t,
    kernel16_with_scalar_t,
    16
);

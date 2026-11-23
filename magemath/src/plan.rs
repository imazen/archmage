//! One-time SIMD tier selection shared across all `magemath` operation families.
//!
//! Each operation resolves a [`Plan`] once at the entry boundary and reuses the
//! chosen capability proof across the whole vector or slice — never probing per
//! FMA or per chunk.

use crate::Execution;
use archmage::IntoConcreteToken;

/// Selected execution tier for a `magemath` operation.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Plan {
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
    pub(crate) fn same_tier(self, other: Self) -> bool {
        #[cfg(target_arch = "wasm32")]
        if let (Self::Wasm(_, a), Self::Wasm(_, b)) = (self, other) {
            return a == b;
        }
        core::mem::discriminant(&self) == core::mem::discriminant(&other)
    }

    #[inline(always)]
    pub(crate) fn midp_slice_width(self) -> usize {
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
}

#[inline(always)]
fn reattune_tier() -> Plan {
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

#[archmage::attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
fn upgrade(token: Token) -> Plan {
    select_tier(token, Execution::Caller, false)
}

/// Select the hardware tier without probing WASM FMA fusion or `std::f32::mul_add`.
///
/// On `wasm32`, `Plan::Wasm(token, false)` carries the raw `Wasm128Token`;
/// callers that distinguish native vs. software WASM FMA use [`select_f32`] or
/// [`select_powf`].
#[inline(always)]
pub(crate) fn select_tier<T: IntoConcreteToken>(
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
                let upgraded = reattune_tier();
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
        return Plan::Wasm(t, false);
    }
    if execution == Execution::Caller {
        return Plan::Software;
    }
    reattune_tier()
}

/// Select a tier for `f32` polynomial families (`expf`, `logf`, `log1pf`).
#[inline(always)]
pub(crate) fn select_f32<T: IntoConcreteToken>(
    token: T,
    execution: Execution,
    backend: bool,
    widen_v3: bool,
) -> Plan {
    let _ = backend;
    let plan = select_tier(token, execution, widen_v3);
    #[cfg(target_arch = "wasm32")]
    if let Plan::Wasm(t, _) = plan {
        return Plan::Wasm(t, backend || crate::wasm_fma::powf_f32_fuses());
    }
    plan
}

/// Select a tier for `powf` (`Midp`/`Lowp` `f32` and `Highp` `f64`), optionally
/// upgrading scalar software to `Plan::Std` when `std` hardware FMA is available.
#[inline(always)]
pub(crate) fn select_powf<T: IntoConcreteToken, const EXACT: bool, const DOUBLE: bool>(
    token: T,
    execution: Execution,
    widen_v3: bool,
    std_available: impl FnOnce() -> bool,
) -> Plan {
    let _ = &std_available;
    let plan = select_tier(token, execution, widen_v3);
    #[cfg(target_arch = "wasm32")]
    if let Plan::Wasm(t, _) = plan {
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
    #[cfg(all(feature = "std", not(miri)))]
    if EXACT
        && !DOUBLE
        && execution == Execution::Upgrade
        && matches!(plan, Plan::Software)
        && std_available()
    {
        return Plan::Std;
    }
    plan
}

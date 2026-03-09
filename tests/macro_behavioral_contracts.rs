//! Behavioral contract tests for `#[arcane]`, `#[rite]`, and `incant!`.
//!
//! These tests establish the observable contracts of the macro system.
//! They serve as a baseline: any macro refactoring (lightfn, sibling expansion, etc.)
//! must preserve the behavior tested here.

#![allow(unused, clippy::needless_return, clippy::unnecessary_wraps)]

// =============================================================================
// A. #[arcane] contracts (x86_64)
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod arcane_contracts_x86 {
    use archmage::{Desktop64, HasNeon, HasX64V2, SimdToken, X64V2Token, X64V3Token, arcane};

    // --- Concrete token: compiles and executes ---

    #[arcane]
    fn double_values(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        for i in 0..8 {
            out[i] = data[i] * 2.0;
        }
        out
    }

    #[test]
    fn concrete_token_executes() {
        if let Some(token) = X64V3Token::summon() {
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let result = double_values(token, &input);
            assert_eq!(result, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    // --- Desktop64 alias ---

    #[arcane]
    fn negate_desktop(token: Desktop64, data: &[f32; 4]) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            out[i] = -data[i];
        }
        out
    }

    #[test]
    fn desktop64_alias_works() {
        if let Some(token) = Desktop64::summon() {
            let result = negate_desktop(token, &[1.0, -2.0, 3.0, -4.0]);
            assert_eq!(result, [-1.0, 2.0, -3.0, 4.0]);
        }
    }

    // --- Wildcard token ---

    #[arcane]
    fn sum_wildcard(_: X64V3Token, data: &[f32; 4]) -> f32 {
        data.iter().sum()
    }

    #[test]
    fn wildcard_token_works() {
        if let Some(token) = X64V3Token::summon() {
            assert_eq!(sum_wildcard(token, &[1.0, 2.0, 3.0, 4.0]), 10.0);
        }
    }

    // --- Multiple params, various return types ---

    #[arcane]
    fn add_arrays(token: X64V3Token, a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            out[i] = a[i] + b[i];
        }
        out
    }

    #[test]
    fn multiple_params_work() {
        if let Some(token) = X64V3Token::summon() {
            let result = add_arrays(token, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]);
            assert_eq!(result, [11.0, 22.0, 33.0, 44.0]);
        }
    }

    #[arcane]
    fn scalar_return(token: X64V3Token, val: f32) -> f32 {
        val * val
    }

    #[test]
    fn scalar_return_type() {
        if let Some(token) = X64V3Token::summon() {
            assert_eq!(scalar_return(token, 5.0), 25.0);
        }
    }

    #[arcane]
    fn bool_return(token: X64V3Token, val: f32) -> bool {
        val > 0.0
    }

    #[test]
    fn bool_return_type() {
        if let Some(token) = X64V3Token::summon() {
            assert!(bool_return(token, 1.0));
            assert!(!bool_return(token, -1.0));
        }
    }

    // --- _self = Type with &self, &mut self, self ---

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct SimdVec8 {
        data: [f32; 8],
    }

    impl SimdVec8 {
        fn new(data: [f32; 8]) -> Self {
            Self { data }
        }

        #[arcane(_self = SimdVec8)]
        fn sum_ref(&self, token: X64V3Token) -> f32 {
            _self.data.iter().sum()
        }

        #[arcane(_self = SimdVec8)]
        fn scale_mut(&mut self, token: X64V3Token, factor: f32) {
            for v in _self.data.iter_mut() {
                *v *= factor;
            }
        }

        #[arcane(_self = SimdVec8)]
        fn into_sum(self, token: X64V3Token) -> f32 {
            _self.data.iter().sum()
        }

        // Self in return type
        #[arcane(_self = SimdVec8)]
        fn doubled(&self, token: X64V3Token) -> Self {
            let mut data = [0.0f32; 8];
            for i in 0..8 {
                data[i] = _self.data[i] * 2.0;
            }
            SimdVec8 { data }
        }
    }

    #[test]
    fn self_ref_method() {
        if let Some(token) = X64V3Token::summon() {
            let v = SimdVec8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            assert_eq!(v.sum_ref(token), 36.0);
        }
    }

    #[test]
    fn self_mut_method() {
        if let Some(token) = X64V3Token::summon() {
            let mut v = SimdVec8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            v.scale_mut(token, 2.0);
            assert_eq!(v.data, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    #[test]
    fn self_owned_method() {
        if let Some(token) = X64V3Token::summon() {
            let v = SimdVec8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            assert_eq!(v.into_sum(token), 36.0);
        }
    }

    #[test]
    fn self_return_type() {
        if let Some(token) = X64V3Token::summon() {
            let v = SimdVec8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            let d = v.doubled(token);
            assert_eq!(d.data, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        }
    }

    // --- Generic / trait bounds ---

    #[arcane]
    fn generic_impl_trait(token: impl HasX64V2, val: f32) -> f32 {
        val + 1.0
    }

    #[test]
    fn impl_trait_bound() {
        if let Some(token) = X64V2Token::summon() {
            assert_eq!(generic_impl_trait(token, 5.0), 6.0);
        }
    }

    #[arcane]
    fn generic_type_param<T: HasX64V2>(token: T, val: f32) -> f32 {
        val + 2.0
    }

    #[test]
    fn generic_type_param_bound() {
        if let Some(token) = X64V3Token::summon() {
            // X64V3Token: HasX64V2 (superset)
            assert_eq!(generic_type_param(token, 5.0), 7.0);
        }
    }

    #[arcane]
    fn generic_where_clause<T>(token: T, val: f32) -> f32
    where
        T: HasX64V2,
    {
        val + 3.0
    }

    #[test]
    fn where_clause_bound() {
        if let Some(token) = X64V2Token::summon() {
            assert_eq!(generic_where_clause(token, 5.0), 8.0);
        }
    }
}

// =============================================================================
// B. #[rite] contracts (x86_64)
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod rite_contracts_x86 {
    use archmage::{SimdToken, X64V3Token, arcane, rite};

    #[rite]
    fn helper_add(_: X64V3Token, a: f32, b: f32) -> f32 {
        a + b
    }

    #[rite]
    fn helper_mul(token: X64V3Token, a: f32, b: f32) -> f32 {
        a * b
    }

    // #[rite] called from #[arcane] context
    #[arcane]
    fn combined(token: X64V3Token, a: f32, b: f32) -> f32 {
        let sum = helper_add(token, a, b);
        helper_mul(token, sum, 2.0)
    }

    #[test]
    fn rite_from_arcane_context() {
        if let Some(token) = X64V3Token::summon() {
            // (3 + 4) * 2 = 14
            assert_eq!(combined(token, 3.0, 4.0), 14.0);
        }
    }

    #[test]
    fn rite_direct_call_unsafe() {
        if let Some(token) = X64V3Token::summon() {
            let result = unsafe { helper_add(token, 10.0, 20.0) };
            assert_eq!(result, 30.0);
        }
    }
}

// =============================================================================
// C. Cross-arch behavior: cfg-out default, stub opt-in
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod cross_arch_cfgout_x86 {
    use archmage::{NeonToken, SimdToken, arcane};

    // Default: ARM function cfg'd out on x86 — doesn't exist
    #[arcane]
    fn arm_cfgout(_token: NeonToken, data: &[f32]) -> f32 {
        data.iter().sum()
    }

    // With stub: ARM function exists as unreachable stub on x86
    #[arcane(stub)]
    fn arm_with_stub(_token: NeonToken, data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[test]
    fn arm_token_not_available_on_x86() {
        assert!(NeonToken::summon().is_none());
    }

    #[test]
    fn stub_function_exists() {
        // arm_with_stub exists as a stub — can take a reference
        let _fn_ref: fn(NeonToken, &[f32]) -> f32 = arm_with_stub;
    }
}

#[cfg(target_arch = "aarch64")]
mod cross_arch_cfgout_arm {
    use archmage::{SimdToken, X64V3Token, arcane};

    #[arcane]
    fn x86_cfgout(_token: X64V3Token, data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[arcane(stub)]
    fn x86_with_stub(_token: X64V3Token, data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[test]
    fn x86_token_not_available_on_arm() {
        assert!(X64V3Token::summon().is_none());
    }
}

// =============================================================================
// D. incant! integration
// =============================================================================

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
mod incant_contracts {
    use archmage::{ScalarToken, SimdToken, incant};

    // --- Dispatch target functions ---

    fn compute_scalar(_token: ScalarToken, data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[cfg(target_arch = "x86_64")]
    #[archmage::arcane]
    fn compute_v3(_token: archmage::X64V3Token, data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[archmage::arcane]
    fn compute_v4(_token: archmage::X64V4Token, data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[cfg(target_arch = "aarch64")]
    #[archmage::arcane]
    fn compute_neon(_token: archmage::NeonToken, data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[cfg(target_arch = "wasm32")]
    #[archmage::arcane]
    fn compute_wasm128(_token: archmage::Wasm128Token, data: &[f32]) -> f32 {
        data.iter().sum()
    }

    // Entry mode
    pub fn compute_entry(data: &[f32]) -> f32 {
        incant!(compute(data))
    }

    #[test]
    fn incant_entry_dispatches() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(compute_entry(&data), 15.0);
    }

    // Passthrough mode
    fn compute_passthrough<T: archmage::IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
        incant!(compute(data) with token)
    }

    #[test]
    fn incant_passthrough_dispatches() {
        let data = [1.0f32, 2.0, 3.0];
        let scalar = ScalarToken::summon().unwrap();
        assert_eq!(compute_passthrough(scalar, &data), 6.0);
    }

    // Explicit tiers
    fn compute_explicit(data: &[f32]) -> f32 {
        incant!(compute(data), [v3, neon, scalar])
    }

    #[test]
    fn incant_explicit_tiers() {
        let data = [10.0f32, 20.0];
        assert_eq!(compute_explicit(&data), 30.0);
    }
}

// =============================================================================
// E. ScalarToken
// =============================================================================

mod scalar_token_contracts {
    use archmage::{ScalarToken, SimdToken};

    // ScalarToken can't be used with #[arcane] (no features to enable),
    // but can be used as a plain function parameter for dispatch

    fn scalar_fn(_token: ScalarToken, val: f32) -> f32 {
        val * 3.0
    }

    #[test]
    fn scalar_always_available() {
        assert!(ScalarToken::summon().is_some());
    }

    #[test]
    fn scalar_fn_executes() {
        let token = ScalarToken::summon().unwrap();
        assert_eq!(scalar_fn(token, 5.0), 15.0);
    }

    #[test]
    fn scalar_compiled_with() {
        assert_eq!(ScalarToken::compiled_with(), Some(true));
    }
}

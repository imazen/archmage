//! Tests for the `stub` parameter on `#[arcane]` and `#[rite]`.
//!
//! Default behavior (no `stub`): functions are cfg'd out on wrong architectures.
//! With `stub`: generates an `unreachable!()` stub that compiles but panics at runtime.

#![allow(unused)]

// =============================================================================
// #[arcane(stub)] on wrong architecture
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod arcane_stub_x86 {
    use archmage::{NeonToken, SimdToken, arcane};

    // ARM function with stub on x86: function exists but token returns None
    #[arcane(stub)]
    fn neon_with_stub(_token: NeonToken, data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[test]
    fn arcane_stub_function_exists() {
        // Function compiles (it's a stub)
        assert!(NeonToken::summon().is_none());
        // We could reference neon_with_stub as a function pointer, etc.
        let _fn_ref: fn(NeonToken, &[f32]) -> f32 = neon_with_stub;
    }
}

// =============================================================================
// #[rite(stub)] on wrong architecture
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod rite_stub_x86 {
    use archmage::{NeonToken, SimdToken, rite};

    #[rite(stub)]
    fn neon_rite_stub(_token: NeonToken, val: f32) -> f32 {
        val * 2.0
    }

    #[test]
    fn rite_stub_function_exists() {
        assert!(NeonToken::summon().is_none());
        // The stub exists — function is referenceable
        let _fn_ref: fn(NeonToken, f32) -> f32 = neon_rite_stub;
    }
}

// =============================================================================
// #[arcane(stub, _self = Type)] — combined (implies nested)
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod arcane_stub_with_self {
    use archmage::{NeonToken, SimdToken, arcane};

    struct Processor {
        factor: f32,
    }

    impl Processor {
        #[arcane(stub, _self = Processor)]
        fn process(&self, _token: NeonToken, val: f32) -> f32 {
            _self.factor * val
        }
    }

    #[test]
    fn stub_with_self_compiles() {
        assert!(NeonToken::summon().is_none());
    }
}

// =============================================================================
// #[arcane(nested)] explicit — old behavior
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod arcane_nested_explicit {
    use archmage::{SimdToken, X64V3Token, arcane};

    // Explicit nested mode (free function)
    #[arcane(nested)]
    fn nested_fn(token: X64V3Token, val: f32) -> f32 {
        val * 3.0
    }

    #[test]
    fn explicit_nested_works() {
        if let Some(token) = X64V3Token::summon() {
            assert_eq!(nested_fn(token, 5.0), 15.0);
        }
    }

    // Nested with _self (implied by _self = Type)
    struct Holder {
        val: f32,
    }

    impl Holder {
        #[arcane(_self = Holder)]
        fn get(&self, token: X64V3Token) -> f32 {
            _self.val
        }

        // Explicit nested + _self
        #[arcane(nested, _self = Holder)]
        fn doubled(&self, token: X64V3Token) -> f32 {
            _self.val * 2.0
        }
    }

    #[test]
    fn nested_with_self_works() {
        if let Some(token) = X64V3Token::summon() {
            let h = Holder { val: 42.0 };
            assert_eq!(h.get(token), 42.0);
            assert_eq!(h.doubled(token), 84.0);
        }
    }
}

// =============================================================================
// Default (no stub): functions don't exist on wrong arch
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod cfgout_default_x86 {
    use archmage::{NeonToken, SimdToken, arcane, rite};

    // These ARM functions are cfg'd out on x86 — they don't exist at all
    #[arcane]
    fn neon_cfgout(_token: NeonToken, val: f32) -> f32 {
        val
    }

    #[rite]
    fn neon_rite_cfgout(_token: NeonToken, val: f32) -> f32 {
        val
    }

    #[test]
    fn cfgout_token_returns_none() {
        assert!(NeonToken::summon().is_none());
        // neon_cfgout and neon_rite_cfgout don't exist as symbols on x86.
        // We can't reference them, which is the point of cfg-out.
    }
}

// =============================================================================
// Correct-arch with and without stub: both work identically
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod correct_arch_both_modes {
    use archmage::{SimdToken, X64V3Token, arcane, rite};

    // Default (sibling, no stub)
    #[arcane]
    fn sibling_default(token: X64V3Token, val: f32) -> f32 {
        val + 1.0
    }

    // With stub (sibling + stub on wrong arch)
    #[arcane(stub)]
    fn sibling_stub(token: X64V3Token, val: f32) -> f32 {
        val + 2.0
    }

    // Nested explicit
    #[arcane(nested)]
    fn nested_default(token: X64V3Token, val: f32) -> f32 {
        val + 3.0
    }

    // Nested with stub
    #[arcane(nested, stub)]
    fn nested_stub(token: X64V3Token, val: f32) -> f32 {
        val + 4.0
    }

    // Rite default
    #[rite]
    fn rite_default(token: X64V3Token, val: f32) -> f32 {
        val + 5.0
    }

    // Rite with stub
    #[rite(stub)]
    fn rite_stub(token: X64V3Token, val: f32) -> f32 {
        val + 6.0
    }

    #[test]
    fn all_modes_work_on_correct_arch() {
        if let Some(token) = X64V3Token::summon() {
            assert_eq!(sibling_default(token, 10.0), 11.0);
            assert_eq!(sibling_stub(token, 10.0), 12.0);
            assert_eq!(nested_default(token, 10.0), 13.0);
            assert_eq!(nested_stub(token, 10.0), 14.0);
            assert_eq!(unsafe { rite_default(token, 10.0) }, 15.0);
            assert_eq!(unsafe { rite_stub(token, 10.0) }, 16.0);
        }
    }
}

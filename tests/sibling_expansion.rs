//! Tests for sibling function expansion (default #[arcane] behavior).
//!
//! Sibling expansion generates two functions at the same scope:
//! - `__arcane_fn`: #[target_feature] unsafe sibling (doc-hidden)
//! - `fn original`: safe wrapper that calls the sibling
//!
//! This avoids the nested inner function approach where Self/self
//! don't work naturally.

#![allow(unused)]

#[cfg(target_arch = "x86_64")]
mod sibling_tests {
    use archmage::{Desktop64, SimdToken, X64V3Token, arcane};

    // --- Free function: sibling works ---

    #[arcane]
    fn free_fn_double(token: X64V3Token, data: &[f32; 4]) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            out[i] = data[i] * 2.0;
        }
        out
    }

    #[test]
    fn free_function_works() {
        if let Some(token) = X64V3Token::summon() {
            let result = free_fn_double(token, &[1.0, 2.0, 3.0, 4.0]);
            assert_eq!(result, [2.0, 4.0, 6.0, 8.0]);
        }
    }

    // --- Method with &self: self.data accessible in body ---

    #[derive(Clone, Debug)]
    struct DataHolder {
        data: [f32; 4],
    }

    impl DataHolder {
        fn new(data: [f32; 4]) -> Self {
            Self { data }
        }

        #[arcane]
        fn sum(&self, token: X64V3Token) -> f32 {
            self.data.iter().sum()
        }

        #[arcane]
        fn scale(&mut self, token: X64V3Token, factor: f32) {
            for v in self.data.iter_mut() {
                *v *= factor;
            }
        }

        #[arcane]
        fn into_sum(self, token: X64V3Token) -> f32 {
            self.data.iter().sum()
        }

        // Self in return type works naturally in sibling mode
        #[arcane]
        fn doubled(&self, token: X64V3Token) -> Self {
            Self {
                data: [
                    self.data[0] * 2.0,
                    self.data[1] * 2.0,
                    self.data[2] * 2.0,
                    self.data[3] * 2.0,
                ],
            }
        }

        // Associated constant reference
        #[arcane]
        fn with_offset(&self, token: X64V3Token) -> [f32; 4] {
            let mut out = self.data;
            for v in out.iter_mut() {
                *v += Self::OFFSET;
            }
            out
        }

        const OFFSET: f32 = 10.0;
    }

    #[test]
    fn method_ref_self() {
        if let Some(token) = X64V3Token::summon() {
            let h = DataHolder::new([1.0, 2.0, 3.0, 4.0]);
            assert_eq!(h.sum(token), 10.0);
        }
    }

    #[test]
    fn method_mut_self() {
        if let Some(token) = X64V3Token::summon() {
            let mut h = DataHolder::new([1.0, 2.0, 3.0, 4.0]);
            h.scale(token, 3.0);
            assert_eq!(h.data, [3.0, 6.0, 9.0, 12.0]);
        }
    }

    #[test]
    fn method_owned_self() {
        if let Some(token) = X64V3Token::summon() {
            let h = DataHolder::new([1.0, 2.0, 3.0, 4.0]);
            assert_eq!(h.into_sum(token), 10.0);
        }
    }

    #[test]
    fn method_returns_self() {
        if let Some(token) = X64V3Token::summon() {
            let h = DataHolder::new([1.0, 2.0, 3.0, 4.0]);
            let d = h.doubled(token);
            assert_eq!(d.data, [2.0, 4.0, 6.0, 8.0]);
        }
    }

    #[test]
    fn method_uses_self_constant() {
        if let Some(token) = X64V3Token::summon() {
            let h = DataHolder::new([1.0, 2.0, 3.0, 4.0]);
            let result = h.with_offset(token);
            assert_eq!(result, [11.0, 12.0, 13.0, 14.0]);
        }
    }

    // --- Wildcard token with sibling ---

    #[arcane]
    fn wildcard_sibling(_: X64V3Token, val: f32) -> f32 {
        val * val
    }

    #[test]
    fn wildcard_token_sibling() {
        if let Some(token) = X64V3Token::summon() {
            assert_eq!(wildcard_sibling(token, 5.0), 25.0);
        }
    }

    // --- Desktop64 alias with sibling ---

    #[arcane]
    fn alias_sibling(token: Desktop64, val: f32) -> f32 {
        val + 1.0
    }

    #[test]
    fn alias_works_with_sibling() {
        if let Some(token) = Desktop64::summon() {
            assert_eq!(alias_sibling(token, 5.0), 6.0);
        }
    }

    // --- Trait impls require nested mode ---
    // Sibling expansion generates __arcane_fn which isn't in the trait definition.
    // Trait impls must use nested (or _self = Type which implies nested).

    trait SimdOps {
        fn compute(&self, token: X64V3Token) -> f32;
        fn transform(&self, token: X64V3Token) -> Self;
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Point {
        x: f32,
        y: f32,
    }

    impl SimdOps for Point {
        #[arcane(_self = Point)]
        fn compute(&self, token: X64V3Token) -> f32 {
            _self.x * _self.x + _self.y * _self.y
        }

        #[arcane(_self = Point)]
        fn transform(&self, token: X64V3Token) -> Self {
            Point {
                x: _self.x * 2.0,
                y: _self.y * 2.0,
            }
        }
    }

    #[test]
    fn trait_impl_with_nested() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point { x: 3.0, y: 4.0 };
            assert_eq!(p.compute(token), 25.0);
            let t = p.transform(token);
            assert_eq!(t, Point { x: 6.0, y: 8.0 });
        }
    }

    // --- Lint attributes propagated to sibling (GH-17) ---
    // #[allow], #[expect], etc. must be propagated to the generated __arcane_ sibling
    // function, otherwise clippy fires on the sibling even when the user suppressed it.

    #[allow(clippy::too_many_arguments)]
    #[arcane]
    fn many_args_fn(
        token: X64V3Token,
        a: f32,
        b: f32,
        c: f32,
        d: f32,
        e: f32,
        f: f32,
        g: f32,
        h: f32,
        i: f32,
    ) -> f32 {
        a + b + c + d + e + f + g + h + i
    }

    #[test]
    fn lint_attrs_propagated_to_sibling() {
        if let Some(token) = X64V3Token::summon() {
            let result = many_args_fn(token, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
            assert_eq!(result, 45.0);
        }
    }

    // Verify #[allow] works on methods too (sibling has self receiver + args)
    impl DataHolder {
        #[allow(clippy::too_many_arguments)]
        #[arcane]
        fn many_args_method(
            &self,
            token: X64V3Token,
            a: f32,
            b: f32,
            c: f32,
            d: f32,
            e: f32,
            f: f32,
            g: f32,
        ) -> f32 {
            self.data[0] + a + b + c + d + e + f + g
        }
    }

    #[test]
    fn lint_attrs_propagated_to_method_sibling() {
        if let Some(token) = X64V3Token::summon() {
            let h = DataHolder::new([100.0, 0.0, 0.0, 0.0]);
            let result = h.many_args_method(token, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
            assert_eq!(result, 128.0);
        }
    }

    // --- User #[inline(always)] stripped to avoid duplicate attribute warning ---

    #[inline(always)]
    #[arcane]
    fn user_inline_fn(token: X64V3Token, data: &[f32; 4]) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            out[i] = data[i] + 1.0;
        }
        out
    }

    #[test]
    fn user_inline_always_no_duplicate() {
        if let Some(token) = X64V3Token::summon() {
            let result = user_inline_fn(token, &[1.0, 2.0, 3.0, 4.0]);
            assert_eq!(result, [2.0, 3.0, 4.0, 5.0]);
        }
    }
}

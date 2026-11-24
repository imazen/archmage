//! Process-wide token permutations run in their own test binary.
#![cfg(all(feature = "std", target_arch = "x86_64", target_feature = "sse2"))]

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use archmage::{SimdToken, X64V3Token};
use magemath::powf::{Powf, PowfOptions};

#[test]
fn automatic_scalar_fallback_preserves_exact_powers() {
    let mut checked = false;
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |_| {
        if checked || X64V3Token::summon().is_some() {
            return;
        }
        let runtime = PowfOptions::new().runtime();
        for (base, exponent, expected) in [(0.25, 0.5, 0.5_f32), (4.0, -1.0, 0.25), (2.0, 3.0, 8.0)]
        {
            assert_eq!(
                <Powf>::scalar(base, exponent).unwrap().to_bits(),
                expected.to_bits()
            );
            assert_eq!(
                runtime.scalar(base, exponent).unwrap().to_bits(),
                expected.to_bits()
            );
        }
        checked = true;
    });
    assert!(
        checked
            || report
                .warnings
                .iter()
                .any(|warning| warning.starts_with(X64V3Token::NAME)),
        "expected to test scalar dispatch when V3 can be disabled: {report}"
    );
}

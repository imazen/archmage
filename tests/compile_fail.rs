//! Compile-fail tests to verify that unsafe intrinsics require unsafe blocks
//!
//! These tests ensure that our safety guarantees are enforced by the compiler.
//! Pointer-based SIMD operations (load, store, gather, masked ops) must always
//! require unsafe blocks, even inside #[target_feature] functions.
//!
//! Note: These tests are skipped in CI because trybuild's exact stderr matching
//! is fragile across Rust versions and platforms. Run locally to verify.

// These tests only apply to x86_64 (the UI tests use x86_64 intrinsics)
#![cfg(target_arch = "x86_64")]

fn is_ci() -> bool {
    std::env::var("CI").is_ok() || std::env::var("GITHUB_ACTIONS").is_ok()
}

#[test]
fn ui_tests() {
    if is_ci() {
        eprintln!("Skipping trybuild tests in CI (stderr output varies by platform/version)");
        return;
    }

    let t = trybuild::TestCases::new();

    // These tests verify that pointer-based intrinsics fail without unsafe
    t.compile_fail("tests/ui/unsafe_load_requires_unsafe.rs");
    t.compile_fail("tests/ui/unsafe_store_requires_unsafe.rs");
    t.compile_fail("tests/ui/unsafe_gather_requires_unsafe.rs");
    t.compile_fail("tests/ui/unsafe_maskload_requires_unsafe.rs");

    // Token type safety tests
    t.compile_fail("tests/compile_fail/wrong_token.rs");

    // Macro rejects unknown trait bounds (e.g., removed HasAvx2, HasFma)
    t.compile_fail("tests/compile_fail/unknown_trait_bound.rs");
    t.compile_fail("tests/compile_fail/unknown_generic_bound.rs");
}

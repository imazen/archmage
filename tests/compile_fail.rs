//! Compile-fail tests to verify that unsafe intrinsics require unsafe blocks
//!
//! These tests ensure that our safety guarantees are enforced by the compiler.
//! Pointer-based SIMD operations (load, store, gather, masked ops) must always
//! require unsafe blocks, even inside #[target_feature] functions.
//!
//! These DO run in CI, on every platform. trybuild compares rustc's rendered
//! stderr byte-for-byte, so only put a case here when its diagnostic is
//! platform- and version-stable. In particular, anything whose message names
//! target features is not: rustc appends a note listing the features enabled in
//! the *build configuration*, and that set differs per target (Linux x86-64
//! says "the sse and sse2"; macOS-Intel says "the cmpxchg16b, sse, sse2, sse3,
//! sse4.1, and ssse3"). Those cases belong in `tests/soundness_exploits.rs`,
//! which asserts an error code plus message fragments instead.
//!
//! `cargo xtask validate` enforces this mechanically: it rejects a committed
//! `.stderr` that names target features, quotes the build configuration,
//! embeds an absolute or toolchain path, carries a rustc version, or depends
//! on pointer width.

// These tests only apply to x86_64 (the UI tests use x86_64 intrinsics)
#![cfg(target_arch = "x86_64")]

#[test]
fn ui_tests() {
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

    // Macro rejects featureless traits (SimdToken, IntoConcreteToken)
    t.compile_fail("tests/compile_fail/featureless_simdtoken.rs");

    // incant! always emits fn_scalar — missing _scalar function = compile error
    t.compile_fail("tests/compile_fail/missing_scalar.rs");

    // incant! with explicit tiers requires `scalar` in the list
    // (gated behind REQUIRE_EXPLICIT_SCALAR, currently false — re-enable in v1.0)
    // t.compile_fail("tests/compile_fail/scalar_not_in_tier_list.rs");

    // scalar + default are mutually exclusive
    t.compile_fail("tests/compile_fail/scalar_default_mutual_exclusion.rs");

    // #[autoversion] rejects concrete tokens
    t.compile_fail("tests/compile_fail/autoversion_concrete_token.rs");

    // Token shadowing: local struct with same name as archmage token must fail
    t.compile_fail("tests/compile_fail/token_shadowing.rs");

    // Token aliasing: renaming a lower-tier token to a higher-tier name must fail
    t.compile_fail("tests/compile_fail/token_aliasing.rs");

    // NOTE: from_context()'s rejection cases deliberately live in
    // tests/soundness_exploits.rs, not here. Their rustc output names the
    // target features enabled in the build configuration, and that set differs
    // per platform (Linux x86-64 lists "sse and sse2"; macOS-Intel lists
    // "cmpxchg16b, sse, sse2, sse3, sse4.1, and ssse3"), so a committed
    // .stderr snapshot cannot pass on every runner. The exploit harness
    // asserts an error code plus message fragments instead.
}

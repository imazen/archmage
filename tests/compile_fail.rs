//! Compile-fail tests to verify that unsafe intrinsics require unsafe blocks
//!
//! These tests ensure that our safety guarantees are enforced by the compiler.
//! Pointer-based SIMD operations (load, store, gather, masked ops) must always
//! require unsafe blocks, even inside #[target_feature] functions.

#[test]
fn ui_tests() {
    let t = trybuild::TestCases::new();

    // These tests verify that pointer-based intrinsics fail without unsafe
    t.compile_fail("tests/ui/unsafe_load_requires_unsafe.rs");
    t.compile_fail("tests/ui/unsafe_store_requires_unsafe.rs");
    t.compile_fail("tests/ui/unsafe_gather_requires_unsafe.rs");
    t.compile_fail("tests/ui/unsafe_maskload_requires_unsafe.rs");
}

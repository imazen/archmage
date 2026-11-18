//! Baseline WASM (`-Ctarget-feature=-simd128,-relaxed-simd` at `opt-level=0`)
//! and SIMD-enabled WASM runtime verification for `from_context()`, untaken
//! `summon()` branches, `#[attune(dispatch)]`, and `unsafe fn` forwarding.
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "wasm32")]
use archmage::SimdToken;
use archmage::{attune, attuned};

#[attune(dispatch, _*, _*_t)]
fn add_ten(x: u32) -> u32 {
    x + 10
}

#[attune(dispatch)]
fn call_contextual(x: u32) -> u32 {
    attuned!(add_ten(x))
}

/// # Safety
/// `ptr` must be valid for reading one `u32`.
#[attune(dispatch, _*_t)]
unsafe fn read_u32_family(ptr: *const u32) -> u32 {
    // SAFETY: caller guarantees `ptr` is valid for reading one `u32`.
    unsafe { *ptr }
}

/// # Safety
/// `ptr` must be valid for reading one `u32`.
#[attune]
unsafe fn read_u32_inferred_scalar_t(_t: archmage::ScalarToken, ptr: *const u32) -> u32 {
    // SAFETY: caller guarantees `ptr` is valid for reading one `u32`.
    unsafe { *ptr }
}

#[cfg(target_arch = "wasm32")]
/// # Safety
/// `ptr` must be valid for reading one `u32`.
#[attune]
unsafe fn read_u32_inferred_wasm128_t(_t: archmage::Wasm128Token, ptr: *const u32) -> u32 {
    // SAFETY: caller guarantees `ptr` is valid for reading one `u32`.
    unsafe { *ptr }
}

#[cfg(target_arch = "wasm32")]
fn dead_branch_strict(x: u32) -> u32 {
    if let Some(_t) = archmage::Wasm128Token::summon() {
        let _ctx = archmage::Wasm128Token::from_context();
        x + 100
    } else {
        x + 1
    }
}

#[cfg(target_arch = "wasm32")]
fn dead_branch_relaxed(x: u32) -> u32 {
    if let Some(_t) = archmage::Wasm128RelaxedToken::summon() {
        let _ctx = archmage::Wasm128RelaxedToken::from_context();
        x + 200
    } else {
        x + 2
    }
}

#[test]
fn wasm_baseline_and_unsafe_forwarding() {
    let mode = std::env::var("ARCHMAGE_WASM_BASELINE_MODE").unwrap_or_default();
    #[cfg(target_arch = "wasm32")]
    match mode.as_str() {
        "panic_strict" => {
            let _ = std::hint::black_box(archmage::Wasm128Token::from_context());
            panic!("expected Wasm128Token::from_context() to panic before reaching here");
        }
        "panic_relaxed" => {
            let _ = std::hint::black_box(archmage::Wasm128RelaxedToken::from_context());
            panic!("expected Wasm128RelaxedToken::from_context() to panic before reaching here");
        }
        "expect_no_simd" => {
            assert!(archmage::Wasm128Token::summon().is_none());
            assert!(archmage::Wasm128RelaxedToken::summon().is_none());
            assert_eq!(dead_branch_strict(41), 42);
            assert_eq!(dead_branch_relaxed(41), 43);
        }
        "expect_simd128" => {
            let tok = archmage::Wasm128Token::summon().expect("simd128 enabled");
            assert_eq!(dead_branch_strict(41), 141);
            let val = 88u32;
            // SAFETY: `&val` is a valid aligned `u32`.
            assert_eq!(unsafe { read_u32_family_wasm128_t(tok, &val) }, 88);
            // SAFETY: `&val` is a valid aligned `u32`.
            assert_eq!(unsafe { read_u32_inferred_wasm128_t(tok, &val) }, 88);
        }
        _ => {}
    }
    let _ = mode;

    assert_eq!(add_ten(32), 42);
    assert_eq!(call_contextual(32), 42);
    let val = 77u32;
    // SAFETY: `&val` is a valid aligned `u32`.
    assert_eq!(unsafe { read_u32_family(&val) }, 77);
    // SAFETY: `&val` is a valid aligned `u32`.
    assert_eq!(
        unsafe { read_u32_inferred_scalar_t(archmage::ScalarToken, &val) },
        77
    );
}

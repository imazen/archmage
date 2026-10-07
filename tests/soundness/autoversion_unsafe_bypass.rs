// REGRESSION CASE: the `#[autoversion]` variants of an `unsafe fn` are
// `#[arcane]` expansions, so each had a safe `__arcane_<fn>_<tier>` sibling
// reachable without `unsafe` from a matching feature context (same hole as
// sibling_unsafe_bypass.rs). tests/soundness_exploits.rs expects E0133
// naming `__arcane_sum_v3`.
use archmage::{X64V3Token, arcane, autoversion};

/// # Safety
/// `ptr` must be valid for `len` reads.
#[autoversion(v3, scalar)]
pub unsafe fn sum(ptr: *const f32, len: usize) -> f32 {
    let mut s = 0.0;
    for i in 0..len {
        s += unsafe { *ptr.add(i) };
    }
    s
}

#[arcane]
fn bypass(t: X64V3Token, ptr: *const f32) -> f32 {
    __arcane_sum_v3(t, ptr, 1)
}

fn main() {
    if let Some(token) = archmage::SimdToken::summon() {
        let _ = bypass(token, core::ptr::null());
    }
}

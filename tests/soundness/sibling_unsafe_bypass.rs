// REGRESSION CASE: `#[arcane]` on an `unsafe fn` emitted a *safe* sibling.
// The sibling is callable without `unsafe` from any function with matching
// target features (another `#[arcane]`/`#[rite]` body in the module), which
// discarded the preconditions the user declared on the wrapper. The sibling
// now keeps the input's `unsafe`: tests/soundness_exploits.rs expects E0133
// naming `__arcane_deref`. Found by an external review before 0.9.30
// (2026-10-07).
use archmage::{X64V3Token, arcane};

/// # Safety
/// `ptr` must be valid for reads.
#[arcane]
pub unsafe fn deref(_t: X64V3Token, ptr: *const f32) -> f32 {
    unsafe { *ptr }
}

#[arcane]
fn bypass(t: X64V3Token, ptr: *const f32) -> f32 {
    __arcane_deref(t, ptr)
}

fn main() {
    if let Some(token) = archmage::SimdToken::summon() {
        let _ = bypass(token, core::ptr::null());
    }
}

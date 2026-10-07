#![allow(dead_code)]
use archmage::{arcane, X64V3Token};
// Safety: ptr must point to a readable u32.
#[arcane]
unsafe fn read_pointer(token: X64V3Token, ptr: *const u32) -> u32 {
    unsafe { *ptr }
}
// Entirely safe caller bypasses the original function's pointer precondition.
#[arcane]
fn safe_bypass(token: X64V3Token) -> u32 {
    __arcane_read_pointer(token, core::ptr::null())
}
fn main() {} // Compile only: do not execute undefined behavior.

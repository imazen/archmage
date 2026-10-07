#![forbid(unsafe_code)]
#![allow(dead_code)]
use archmage::arcane;
struct X64V3Token;
impl X64V3Token {
    pub const __ARCHMAGE_ASSERT_TIER_F38B284B: () = ();
}
#[arcane]
fn features_without_proof(_: X64V3Token) -> i32 {
    let a = core::arch::x86_64::_mm256_set1_epi32(1);
    core::arch::x86_64::_mm256_extract_epi32::<0>(a)
}
fn safe_without_detection() -> i32 { features_without_proof(X64V3Token) }
fn main() {} // Compile only: do not run on an unsupported CPU.

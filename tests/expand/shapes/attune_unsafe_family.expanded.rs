#![deny(unsafe_op_in_unsafe_fn)]
use archmage::attune;
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
unsafe fn __attune_load_word_v3(ptr: *const u32) -> u32 {
    unsafe { *ptr }
}
#[inline(always)]
#[allow(clippy::inline_always, dead_code)]
pub unsafe fn load_word_v3_t(
    __attune_token: ::archmage::X64V3Token,
    ptr: *const u32,
) -> u32 {
    unsafe { __attune_load_word_v3(ptr) }
}
#[inline]
unsafe fn __attune_load_word_scalar(ptr: *const u32) -> u32 {
    unsafe { *ptr }
}
#[inline(always)]
#[allow(clippy::inline_always, dead_code)]
pub unsafe fn load_word_scalar_t(
    __attune_token: ::archmage::ScalarToken,
    ptr: *const u32,
) -> u32 {
    unsafe { __attune_load_word_scalar(ptr) }
}
pub unsafe fn load_word(ptr: *const u32) -> u32 {
    '__attune_dispatch: {
        {
            if let Some(__attune_proof) = <::archmage::X64V3Token as ::archmage::SimdToken>::summon() {
                break '__attune_dispatch unsafe { load_word_v3_t(__attune_proof, ptr) };
            }
        }
        {
            break '__attune_dispatch unsafe {
                load_word_scalar_t(::archmage::ScalarToken, ptr)
            };
        }
    }
}
fn main() {}

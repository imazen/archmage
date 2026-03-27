use archmage::{arcane, X64V3Token, NeonToken};
#[doc(hidden)]
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn __arcane_process_x86(token: X64V3Token, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
#[inline(always)]
fn process_x86(token: X64V3Token, data: &[f32; 4]) -> f32 {
    unsafe { __arcane_process_x86(token, data) }
}
fn process_arm(token: NeonToken, data: &[f32; 4]) -> f32 {
    let _ = (token, data);
    {
        ::core::panicking::panic_fmt(
            format_args!(
                "internal error: entered unreachable code: {0}",
                format_args!("BUG: {0}() was called but requires {1} (target_arch = \"{2}\"). {3}::summon() returns None on this architecture, so this function is unreachable in safe code. If you used forge_token_dangerously(), that is the bug.",
                "process_arm", "NeonToken", "aarch64", "NeonToken",),
            ),
        );
    }
}
fn main() {}

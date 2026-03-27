use archmage::{arcane, X64V3Token, NeonToken};
#[doc(hidden)]
#[target_feature(enable = "sse")]
#[target_feature(enable = "sse2")]
#[target_feature(enable = "sse3")]
#[target_feature(enable = "ssse3")]
#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "sse4.2")]
#[target_feature(enable = "popcnt")]
#[target_feature(enable = "cmpxchg16b")]
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "bmi1")]
#[target_feature(enable = "bmi2")]
#[target_feature(enable = "f16c")]
#[target_feature(enable = "lzcnt")]
#[target_feature(enable = "movbe")]
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

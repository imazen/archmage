use archmage::rite;
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
fn compute_stub_v3(data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
fn compute_stub_neon(data: &[f32; 4]) -> f32 {
    {
        ::core::panicking::panic_fmt(
            format_args!(
                "internal error: entered unreachable code: {0}",
                format_args!("This function requires aarch64 architecture"),
            ),
        );
    }
}
fn main() {}

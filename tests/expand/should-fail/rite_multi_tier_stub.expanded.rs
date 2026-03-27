use archmage::rite;
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
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

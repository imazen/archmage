#![forbid(unsafe_code)]
mod arm_data;
mod arm_powf;

#[inline(never)]
pub fn separate(x: f32, n: f32) -> f32 {
    arm_powf::pow::<false, true, false>(x, n)
}
#[inline(never)]
pub fn fused_std(x: f32, n: f32) -> f32 {
    arm_powf::pow::<true, true, false>(x, n)
}
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(never)]
pub fn fused_v3(_t: archmage::X64V3Token, x: f32, n: f32) -> f32 {
    arm_powf::pow::<true, true, false>(x, n)
}
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
#[inline(never)]
pub fn fused_v3_warm(_t: archmage::X64V3Token, x: f32, n: f32) -> f32 {
    arm_powf::pow::<true, false, false>(x, n)
}
fn main() {
    use archmage::SimdToken;
    let x = std::hint::black_box(0.75f32);
    let n = std::hint::black_box(2.4f32);
    std::hint::black_box(separate(x, n));
    std::hint::black_box(fused_std(x, n));
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = archmage::X64V3Token::summon() {
        std::hint::black_box(fused_v3(t, x, n));
        std::hint::black_box(fused_v3_warm(t, x, n));
    }
}

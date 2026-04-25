//! `#[cpu_tier]` with unambiguous feature — arch inferred as x86_64 from
//! `avx2`. Expected expansion: function gets `#[cfg(target_arch = "x86_64")]`
//! + `#[target_feature(enable = "avx2,fma")]` + `#[inline]`; sibling hidden
//! const `__ARTISAN_CPU_TIER_FEATS_dot_v3` with the normalized feature string.

use artisan_macros::cpu_tier;

#[cpu_tier(enable = "avx2,fma")]
pub fn dot_v3(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

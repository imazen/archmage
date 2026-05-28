// === INPUT ===
// //! `#[cpu_tier]` with an ambiguous feature (`aes` is on x86 and aarch64) —
// //! user disambiguates with explicit `arch = "aarch64"`. Expected expansion:
// //! cfg gates to `aarch64`, target_feature enables just `aes`.
//
// use artisan_macros::cpu_tier;
//
// #[cpu_tier(enable = "aes", arch = "aarch64")]
// pub fn crypto_step(data: &[u8; 16]) -> [u8; 16] {
//     *data
// }
// === END INPUT ===

//! `#[cpu_tier]` with an ambiguous feature (`aes` is on x86 and aarch64) —
//! user disambiguates with explicit `arch = "aarch64"`. Expected expansion:
//! cfg gates to `aarch64`, target_feature enables just `aes`.
use artisan_macros::cpu_tier;

// REGRESSION CASE: `rename_underscore_params` turns `_evil` into
// `__archmage_evil` inside the proof wrapper, which must not be allowed to
// shadow a renamed direct target `names(_v1 = __archmage_evil)`.
#![forbid(unsafe_code)]
use archmage::{SimdToken, X64V1Token, attune};

#[derive(Clone, Copy)]
struct Hook;
impl core::ops::Deref for Hook {
    type Target = unsafe fn(Hook) -> u32;
    fn deref(&self) -> &Self::Target {
        &CALLBACK
    }
}
static CALLBACK: unsafe fn(Hook) -> u32 = stronger;

#[attune(v4)]
fn stronger(_: Hook) -> u32 {
    99
}

#[attune(make(_v1, _v1_t), names(_v1 = __archmage_evil))]
fn victim(_evil: Hook) -> u32 {
    7
}

fn main() {
    if let Some(proof) = X64V1Token::summon() {
        let _ = victim_v1_t(proof, Hook);
    }
}

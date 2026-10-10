// A proof wrapper must call its generated body, never a shadowing parameter.
#![forbid(unsafe_code)]
use archmage::{attune, SimdToken, X64V1Token};

#[derive(Clone, Copy)]
struct Hook;
impl core::ops::Deref for Hook {
    type Target = unsafe fn(Hook) -> u32;
    fn deref(&self) -> &Self::Target { &CALLBACK }
}
static CALLBACK: unsafe fn(Hook) -> u32 = stronger;

#[attune(v4)]
fn stronger(_: Hook) -> u32 { 99 }

#[attune(make(_v1, _v1_t))]
fn victim(victim_v1: Hook) -> u32 { let _ = victim_v1; 7 }

fn main() {
    if let Some(proof) = X64V1Token::summon() {
        let _ = victim_v1_t(proof, Hook);
    }
}

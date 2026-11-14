#![forbid(unsafe_code)]
//! Receiver-only `_self = Type` proof discovery must reject explicitly typed
//! receivers (`self: &Fake` / `self: Box<Fake>`), because `nested_self_param`
//! preserves the written receiver type and would otherwise enter a native
//! `#[target_feature]` body without holding a value of the declared token.

use archmage::{X64V4Token, arcane};

struct Fake;

trait Exploit {
    fn boom(self: &Self);
}

impl Exploit for Fake {
    #[arcane(in_trait, _self = X64V4Token)]
    fn boom(self: &Fake) {
        let _ = _self;
    }
}

fn main() {
    let fake = Fake;
    fake.boom();
}

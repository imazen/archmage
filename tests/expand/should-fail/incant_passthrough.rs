// incant! passthrough mode — uses existing token via IntoConcreteToken
use archmage::{arcane, incant, IntoConcreteToken, X64V3Token, NeonToken, ScalarToken};

#[arcane]
fn inner_v3(_token: X64V3Token, x: f32) -> f32 {
    x * 2.0
}

#[arcane]
fn inner_neon(_token: NeonToken, x: f32) -> f32 {
    x * 2.0
}

fn inner_scalar(_token: ScalarToken, x: f32) -> f32 {
    x * 2.0
}

fn pass_through<T: IntoConcreteToken>(token: T, x: f32) -> f32 {
    incant!(inner(x) with token)
}

fn main() {}

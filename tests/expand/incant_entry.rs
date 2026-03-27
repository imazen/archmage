// incant! entry mode — summons tokens and dispatches
use archmage::{arcane, incant, X64V3Token, NeonToken, ScalarToken};

#[arcane]
fn process_v3(_token: X64V3Token, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

#[arcane]
fn process_neon(_token: NeonToken, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

fn process_scalar(_token: ScalarToken, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

fn dispatch(data: &[f32; 4]) -> f32 {
    incant!(process(data))
}

fn main() {}

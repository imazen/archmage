use archmage::{X64V3Token, SimdToken};
fn uses_token(token: X64V3Token, data: &[f32; 4]) -> f32 {
    let _ = token;
    data.iter().sum()
}
fn dispatches() -> Option<f32> {
    let token = X64V3Token::summon()?;
    Some(uses_token(token, &[1.0, 2.0, 3.0, 4.0]))
}
fn main() {}

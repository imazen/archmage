#![forbid(unsafe_code)]
use archmage::{HasX64V2, SimdToken, X64V2Token, X64V3Token, arcane};

pub trait OtherTrait {
    type Output;
    fn finish(self, sum: f32) -> Self::Output;
}
impl OtherTrait for X64V2Token {
    type Output = u32;
    fn finish(self, sum: f32) -> u32 {
        sum as u32 + 2
    }
}
impl OtherTrait for X64V3Token {
    type Output = u64;
    fn finish(self, sum: f32) -> u64 {
        sum as u64 + 3
    }
}

#[arcane]
pub fn work<T: HasX64V2 + OtherTrait>(token: T, data: &[f32]) -> T::Output {
    let _context_proof = X64V2Token::from_context();
    token.finish(data.iter().sum())
}

fn main() {
    let v2 = X64V2Token::summon().expect("probe requires V2");
    let v3 = X64V3Token::summon().expect("probe requires V3");
    let result_v2: u32 = work(v2, &[1.0, 2.0]);
    let result_v3: u64 = work(v3, &[1.0, 2.0]);
    assert_eq!(result_v2, 5);
    assert_eq!(result_v3, 6);
    println!("V2 -> u32 {result_v2}; V3 -> u64 {result_v3}");
}

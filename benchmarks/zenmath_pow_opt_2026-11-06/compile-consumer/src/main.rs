#![forbid(unsafe_code)]
use std::hint::black_box;
use zenmath::{PowBehavior as B, PowConsistency as C, PowRange as R};
#[inline(never)]
fn calculate<const P: u64>(x: f32, y: f32) -> f32 {
    zenmath::scalar::powf_with::<P>(x, y)
}
fn main() {
    const MID: u64 = B::MIDP.to_u64();
    const LOW: u64 = B::LOWP.to_u64();
    const MS: u64 = B::MIDP.consistency(C::PortableSeparate).to_u64();
    const LS: u64 = B::LOWP.consistency(C::PortableSeparate).to_u64();
    const MP: u64 = B::MIDP.range(R::PositiveNormalPrecondition).to_u64();
    const LP: u64 = B::LOWP.range(R::PositiveNormalPrecondition).to_u64();
    const MPS: u64 = B::MIDP
        .consistency(C::PortableSeparate)
        .range(R::PositiveNormalPrecondition)
        .to_u64();
    const LPS: u64 = B::LOWP
        .consistency(C::PortableSeparate)
        .range(R::PositiveNormalPrecondition)
        .to_u64();
    let mut results = Vec::new();
    for x in black_box([0.0030412826, 0.1, 0.5, 0.99]) {
        for y in black_box([2.4, (1.0_f64 / 2.4) as f32]) {
            let out = [
                calculate::<0>(x, y),
                calculate::<MID>(x, y),
                calculate::<LOW>(x, y),
                calculate::<MS>(x, y),
                calculate::<LS>(x, y),
                calculate::<MP>(x, y),
                calculate::<LP>(x, y),
                calculate::<MPS>(x, y),
                calculate::<LPS>(x, y),
            ];
            results.extend(out.map(f32::to_bits));
        }
    }
    println!("{:x?}", black_box(results));
}

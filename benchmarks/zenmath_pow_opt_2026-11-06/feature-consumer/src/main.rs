#![forbid(unsafe_code)]
use std::hint::black_box;
fn bits(x: f32) -> u32 {
    if x.is_nan() { 0x7fc00000 } else { x.to_bits() }
}
fn main() {
    let input = black_box([
        0.0,
        -0.0,
        0.0030412826,
        0.1,
        0.5,
        0.99,
        f32::INFINITY,
        f32::NAN,
    ]);
    println!(
        "primitive={:?}",
        input.map(|x| bits(zenmath::scalar::floorf(x)))
    );
    #[cfg(any(feature = "correct-rounding", feature = "normal"))]
    {
        #[cfg(feature = "normal")]
        use zenmath::normal as math;
        #[cfg(not(feature = "normal"))]
        use zenmath::scalar as math;
        let mut out = Vec::new();
        for x in input {
            out.extend([
                bits(math::expf(x)),
                bits(math::logf(x)),
                bits(math::powf(x, black_box(2.4))),
            ]);
        }
        println!("precise={out:?}");
        #[cfg(feature = "vector")]
        {
            #[cfg(feature = "normal")]
            use magemath::normal::array;
            #[cfg(not(feature = "normal"))]
            use magemath::portable::array;
            println!(
                "vector-precise={:?}",
                array::pow_f32x8(input, [2.4; 8]).map(bits)
            );
        }
    }
    #[cfg(feature = "configurable")]
    {
        use zenmath::{PowBehavior as B, PowRange as R};
        const P: u64 = B::MIDP.range(R::PositiveNormalPrecondition).to_u64();
        let x = black_box([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 2.0]);
        println!(
            "configurable={:?}",
            x.map(|v| bits(zenmath::scalar::powf_with::<P>(v, 2.4)))
        );
        const MID: u64 = B::MIDP.to_u64();
        const LOW: u64 = B::LOWP.to_u64();
        const MS: u64 = B::MIDP
            .consistency(zenmath::PowConsistency::PortableSeparate)
            .to_u64();
        const LS: u64 = B::LOWP
            .consistency(zenmath::PowConsistency::PortableSeparate)
            .to_u64();
        const LP: u64 = B::LOWP.range(R::PositiveNormalPrecondition).to_u64();
        const MPS: u64 = B::MIDP
            .consistency(zenmath::PowConsistency::PortableSeparate)
            .range(R::PositiveNormalPrecondition)
            .to_u64();
        const LPS: u64 = B::LOWP
            .consistency(zenmath::PowConsistency::PortableSeparate)
            .range(R::PositiveNormalPrecondition)
            .to_u64();
        let mut policies = Vec::new();
        for x in black_box([0.0030412826, 0.1, 0.5, 0.99]) {
            for y in black_box([2.4, (1.0_f64 / 2.4) as f32]) {
                policies.extend(
                    [
                        zenmath::scalar::powf_with::<0>(x, y),
                        zenmath::scalar::powf_with::<MID>(x, y),
                        zenmath::scalar::powf_with::<LOW>(x, y),
                        zenmath::scalar::powf_with::<MS>(x, y),
                        zenmath::scalar::powf_with::<LS>(x, y),
                        zenmath::scalar::powf_with::<P>(x, y),
                        zenmath::scalar::powf_with::<LP>(x, y),
                        zenmath::scalar::powf_with::<MPS>(x, y),
                        zenmath::scalar::powf_with::<LPS>(x, y),
                    ]
                    .map(bits),
                );
            }
        }
        println!("policies={policies:?}");
        #[cfg(feature = "vector")]
        println!(
            "vector-configurable={:?}",
            magemath::portable::array::pow_f32x8_with::<P>(x, [2.4; 8]).map(bits)
        );
    }
}

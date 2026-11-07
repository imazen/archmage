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
        #[cfg(feature = "vector")]
        println!(
            "vector-configurable={:?}",
            magemath::portable::array::pow_f32x8_with::<P>(x, [2.4; 8]).map(bits)
        );
    }
}

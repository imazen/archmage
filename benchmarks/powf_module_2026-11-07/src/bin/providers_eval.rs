//! Runtime std/Rust-libm outputs and mixed-lane width consistency.
#![forbid(unsafe_code)]
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use archmage::SimdToken;
use std::io::{self, BufRead};
fn power<const LIBM: bool>(x: f32, n: f32) -> f32 {
    if LIBM { libm::powf(x, n) } else { x.powf(n) }
}
#[archmage::magetypes(v3, neon, wasm128, scalar)]
fn lanes<const N: usize, const LIBM: bool>(token: Token, input: [f32; N], n: f32) -> [f32; N] {
    let _ = token;
    input.map(|x| power::<LIBM>(x, n))
}
fn check<const N: usize, const LIBM: bool>(context: &str, x: f32, n: f32) -> f32 {
    let input = std::hint::black_box(core::array::from_fn(|i| if i % 2 == 0 { x } else { 1. }));
    let result = match context {
        "scalar" => lanes_scalar::<N, LIBM>(archmage::ScalarToken, input, n),
        #[cfg(target_arch = "x86_64")]
        "native" => lanes_v3::<N, LIBM>(
            archmage::X64V3Token::summon().expect("required V3"),
            input,
            n,
        ),
        #[cfg(target_arch = "aarch64")]
        "native" => lanes_neon::<N, LIBM>(
            archmage::NeonToken::summon().expect("required NEON"),
            input,
            n,
        ),
        #[cfg(target_arch = "wasm32")]
        "native" => lanes_wasm128::<N, LIBM>(
            archmage::Wasm128Token::summon().expect("required SIMD128"),
            input,
            n,
        ),
        _ => panic!("unsupported explicit provider context"),
    };
    for i in 0..N {
        let expected = power::<LIBM>(input[i], n);
        assert!(
            expected.to_bits() == result[i].to_bits() || expected.is_nan() && result[i].is_nan(),
            "provider width {N} lane {i}: x={:08x} n={:08x}",
            input[i].to_bits(),
            n.to_bits()
        );
    }
    result[0]
}
fn main() {
    let context = std::env::args()
        .nth(1)
        .expect("explicit scalar/native context required");
    assert!(matches!(context.as_str(), "scalar" | "native"));
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 3);
        let x = std::hint::black_box(f32::from_bits(u32::from_str_radix(fields[0], 16).unwrap()));
        let n = std::hint::black_box(f32::from_bits(u32::from_str_radix(fields[1], 16).unwrap()));
        let std = check::<1, false>(&context, x, n);
        let libm = check::<1, true>(&context, x, n);
        macro_rules! width {
            ($n:literal) => {{
                let a = check::<$n, false>(&context, x, n);
                let b = check::<$n, true>(&context, x, n);
                assert!(std.to_bits() == a.to_bits() || std.is_nan() && a.is_nan());
                assert!(libm.to_bits() == b.to_bits() || libm.is_nan() && b.is_nan());
            }};
        }
        width!(4);
        width!(8);
        width!(16);
        println!(
            "{} {} {} {:08x} {:08x}",
            fields[0],
            fields[1],
            fields[2],
            std.to_bits(),
            libm.to_bits()
        );
    }
}

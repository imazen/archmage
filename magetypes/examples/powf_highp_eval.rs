//! Stdin numerical evaluator. Explicit contexts never silently fall back.
#![forbid(unsafe_code)]
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use archmage::SimdToken;
use magetypes::powf::{PowfExecution, PowfOptions};
use std::io::{BufRead, Write};
const FLAGS: u32 = PowfOptions::PORTABLE_HIGHP
    .with_execution(PowfExecution::Caller)
    .bits();
#[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn values<const N: usize>(token: Token, x: [f32; N], n: f32) -> [f32; N] {
    use magetypes::simd::generic::{f32x4, f32x8};
    let mut out = [0.; N];
    if N == 1 {
        out[0] = f32x4::splat_t(token, x[0])
            .try_powf_with::<FLAGS>(n)
            .unwrap()
            .to_array()[0];
    } else if N == 4 {
        let mut a = [0.; 4];
        a.copy_from_slice(&x);
        out.copy_from_slice(
            &f32x4::from_array_t(token, a)
                .try_powf_with::<FLAGS>(n)
                .unwrap()
                .to_array(),
        );
    } else if N == 8 {
        let mut a = [0.; 8];
        a.copy_from_slice(&x);
        out.copy_from_slice(
            &f32x8::from_array_t(token, a)
                .try_powf_with::<FLAGS>(n)
                .unwrap()
                .to_array(),
        );
    } else {
        #[cfg(feature = "w512")]
        {
            assert_eq!(N, 16);
            let mut a = [0.; 16];
            a.copy_from_slice(&x);
            out.copy_from_slice(
                &magetypes::simd::generic::f32x16::from_array_t(token, a)
                    .try_powf_with::<FLAGS>(n)
                    .unwrap()
                    .to_array(),
            );
        }
        #[cfg(not(feature = "w512"))]
        panic!("unsupported width");
    }
    out
}
fn evaluate<const N: usize>(ctx: &str, x: [f32; N], n: f32) -> [f32; N] {
    match ctx {
        "scalar" => values_scalar(archmage::ScalarToken, x, n),
        "automatic" => {
            magetypes::powf::try_powf_array_with::<{ PowfOptions::PORTABLE_HIGHP.bits() }, N>(x, n)
                .unwrap()
        }
        #[cfg(target_arch = "x86_64")]
        "native" => values_v3(archmage::X64V3Token::summon().expect("required V3"), x, n),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "native-v4" => values_v4(archmage::X64V4Token::summon().expect("required V4"), x, n),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "native-v4x" => values_v4x(archmage::X64V4xToken::summon().expect("required V4x"), x, n),
        #[cfg(target_arch = "aarch64")]
        "native" => values_neon(archmage::NeonToken::summon().expect("required NEON"), x, n),
        #[cfg(target_arch = "wasm32")]
        "native" => values_wasm128(
            archmage::Wasm128Token::summon().expect("required SIMD128 module"),
            x,
            n,
        ),
        _ => panic!("unsupported context"),
    }
}
fn request<const N: usize>(ctx: &str, f: &[&str]) -> [u32; N] {
    assert_eq!(f.len(), 3 + 2 * N);
    let x =
        core::array::from_fn(|i| f32::from_bits(u32::from_str_radix(f[3 + 2 * i], 16).unwrap()));
    let ns: [f32; N] =
        core::array::from_fn(|i| f32::from_bits(u32::from_str_radix(f[4 + 2 * i], 16).unwrap()));
    if ns.iter().all(|n| n.to_bits() == ns[0].to_bits()) {
        return evaluate(ctx, x, ns[0]).map(f32::to_bits);
    }
    // Public powf has a uniform exponent. For a mixed-exponent fixture,
    // execute its complete mixed-base vector for each exponent, then select
    // the corresponding lane. Thus native packed paths still see mixed bases.
    core::array::from_fn(|i| evaluate(ctx, x, ns[i])[i].to_bits())
}
fn main() {
    let ctx = std::env::args()
        .nth(1)
        .expect("scalar/native/automatic context");
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let f: Vec<_> = line.split_whitespace().collect();
        assert_eq!(f[0], "E");
        let a = match f[2] {
            "1" => request::<1>(&ctx, &f).to_vec(),
            "4" => request::<4>(&ctx, &f).to_vec(),
            "8" => request::<8>(&ctx, &f).to_vec(),
            "16" => request::<16>(&ctx, &f).to_vec(),
            _ => panic!("unsupported width"),
        };
        for b in a {
            write!(out, "{b:08x} ").unwrap();
        }
        writeln!(out).unwrap();
    }
}

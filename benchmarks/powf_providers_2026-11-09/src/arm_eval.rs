//! Portable stdin evaluator for the actual scalar graph on non-Unix targets.
#![forbid(unsafe_code)]
mod arm_data;
mod arm_powf;
use std::io::{BufRead, Write};

#[inline(always)]
fn value(method: &str, x: f32, n: f32) -> f32 {
    match method {
        "std" => x.powf(n),
        "Arm_Rust_fused_cold" | "Arm_Rust_fused_unrolled" => {
            arm_powf::pow::<true, true, false>(x, n)
        }
        "Arm_Rust_separate_cold" => arm_powf::pow::<false, true, false>(x, n),
        "Arm_Rust_fused_warm" => arm_powf::pow::<true, false, false>(x, n),
        "Arm_Rust_fused_assume_Gamma" | "Arm_Rust_assume_Gamma_unrolled" => {
            arm_powf::pow::<true, true, true>(x, n)
        }
        _ => panic!("unsupported graph"),
    }
}
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn native(_t: archmage::X64V3Token, m: &str, x: f32, n: f32) -> f32 {
    value(m, x, n)
}
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn native_neon(_t: archmage::NeonToken, m: &str, x: f32, n: f32) -> f32 {
    value(m, x, n)
}
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn native_wasm(_t: archmage::Wasm128Token, m: &str, x: f32, n: f32) -> f32 {
    value(m, x, n)
}
fn main() {
    let context = std::env::args()
        .nth(1)
        .expect("explicit scalar/native context");
    assert!(matches!(context.as_str(), "scalar" | "native"));
    #[cfg(target_arch = "x86_64")]
    let token = {
        use archmage::SimdToken;
        if context == "native" {
            Some(archmage::X64V3Token::summon().expect("required V3"))
        } else {
            None
        }
    };
    #[cfg(target_arch = "aarch64")]
    let token = {
        use archmage::SimdToken;
        if context == "native" {
            Some(archmage::NeonToken::summon().expect("required NEON"))
        } else {
            None
        }
    };
    #[cfg(target_arch = "wasm32")]
    let token = {
        use archmage::SimdToken;
        if context == "native" {
            Some(archmage::Wasm128Token::summon().expect("required SIMD128 module"))
        } else {
            None
        }
    };
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    assert_eq!(
        context, "scalar",
        "native context is unavailable on this target"
    );
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields[0], "E");
        let width: usize = fields[2].parse().unwrap();
        assert!([1, 4, 8, 16].contains(&width));
        assert_eq!(fields.len(), 3 + width * 2);
        for i in 0..width {
            let x = f32::from_bits(u32::from_str_radix(fields[3 + 2 * i], 16).unwrap());
            let n = f32::from_bits(u32::from_str_radix(fields[4 + 2 * i], 16).unwrap());
            #[cfg(target_arch = "x86_64")]
            let y = if let Some(t) = token {
                native(t, fields[1], x, n)
            } else {
                value(fields[1], x, n)
            };
            #[cfg(target_arch = "aarch64")]
            let y = if let Some(t) = token {
                native_neon(t, fields[1], x, n)
            } else {
                value(fields[1], x, n)
            };
            #[cfg(target_arch = "wasm32")]
            let y = if let Some(t) = token {
                native_wasm(t, fields[1], x, n)
            } else {
                value(fields[1], x, n)
            };
            #[cfg(not(any(
                target_arch = "x86_64",
                target_arch = "aarch64",
                target_arch = "wasm32"
            )))]
            let y = value(fields[1], x, n);
            write!(out, "{:08x} ", y.to_bits()).unwrap();
        }
        writeln!(out).unwrap();
    }
}

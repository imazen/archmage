//! Independent MPFR record validation for the correctly rounded vector kernels.
use archmage::SimdToken;
use magemath::portable::array as a;
use std::{env, fs};
fn same(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}
type Op = fn([f32; 8], [f32; 8]) -> [f32; 8];
fn check(label: &str, op: Op, records: &[[f32; 3]]) {
    for (n, chunk) in records.chunks(8).enumerate() {
        let x = core::array::from_fn(|i| chunk.get(i).map_or(1.0, |r| r[0]));
        let y = core::array::from_fn(|i| chunk.get(i).map_or(1.0, |r| r[1]));
        let out = op(x, y);
        for (i, r) in chunk.iter().enumerate() {
            assert!(
                same(out[i], r[2]),
                "{label} record {}: {:08x}, {:08x}: got {:08x}, expected {:08x}",
                n * 8 + i,
                r[0].to_bits(),
                r[1].to_bits(),
                out[i].to_bits(),
                r[2].to_bits()
            );
        }
    }
    println!("{label}: {} MPFR records passed", records.len());
}
fn main() {
    let args: Vec<_> = env::args().collect();
    assert_eq!(args.len(), 3, "FUNCTION RECORDS");
    let data = fs::read(&args[2]).expect("MPFR records");
    assert!(!data.is_empty() && data.len() % 12 == 0);
    let records: Vec<[f32; 3]> = data
        .as_chunks::<12>()
        .0
        .iter()
        .map(|r| {
            core::array::from_fn(|i| {
                f32::from_bits(u32::from_le_bytes(r[i * 4..i * 4 + 4].try_into().unwrap()))
            })
        })
        .collect();
    let name = args[1].as_str();
    {
        let op: Op = match name {
            "expf" => |x, _y| a::exp_f32x8(x),
            "logf" => |x, _y| a::ln_f32x8(x),
            "log2f" => |x, _y| a::log2_f32x8(x),
            "powf" => |x, y| a::pow_f32x8(x, y),
            _ => panic!("unknown function"),
        };
        check("dispatch", op, &records);
    }
    {
        if archmage::ScalarToken::summon().is_some() {
            let op: Op = match name {
                "expf" => {
                    |x, _y| a::exp_f32x8_scalar_t(archmage::ScalarToken::summon().unwrap(), x)
                }
                "logf" => |x, _y| a::ln_f32x8_scalar_t(archmage::ScalarToken::summon().unwrap(), x),
                "log2f" => {
                    |x, _y| a::log2_f32x8_scalar_t(archmage::ScalarToken::summon().unwrap(), x)
                }
                "powf" => {
                    |x, y| a::pow_f32x8_scalar_t(archmage::ScalarToken::summon().unwrap(), x, y)
                }
                _ => panic!("unknown function"),
            };
            check("scalar", op, &records);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if archmage::X64V3Token::summon().is_some() {
            let op: Op = match name {
                "expf" => |x, _y| a::exp_f32x8_v3_t(archmage::X64V3Token::summon().unwrap(), x),
                "logf" => |x, _y| a::ln_f32x8_v3_t(archmage::X64V3Token::summon().unwrap(), x),
                "log2f" => |x, _y| a::log2_f32x8_v3_t(archmage::X64V3Token::summon().unwrap(), x),
                "powf" => |x, y| a::pow_f32x8_v3_t(archmage::X64V3Token::summon().unwrap(), x, y),
                _ => panic!("unknown function"),
            };
            check("v3", op, &records);
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if archmage::X64V4xToken::summon().is_some() {
            let op: Op = match name {
                "expf" => |x, _y| a::exp_f32x8_v4x_t(archmage::X64V4xToken::summon().unwrap(), x),
                "logf" => |x, _y| a::ln_f32x8_v4x_t(archmage::X64V4xToken::summon().unwrap(), x),
                "log2f" => |x, _y| a::log2_f32x8_v4x_t(archmage::X64V4xToken::summon().unwrap(), x),
                "powf" => |x, y| a::pow_f32x8_v4x_t(archmage::X64V4xToken::summon().unwrap(), x, y),
                _ => panic!("unknown function"),
            };
            check("v4x", op, &records);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if archmage::NeonToken::summon().is_some() {
            let op: Op = match name {
                "expf" => |x, _y| a::exp_f32x8_neon_t(archmage::NeonToken::summon().unwrap(), x),
                "logf" => |x, _y| a::ln_f32x8_neon_t(archmage::NeonToken::summon().unwrap(), x),
                "log2f" => |x, _y| a::log2_f32x8_neon_t(archmage::NeonToken::summon().unwrap(), x),
                "powf" => |x, y| a::pow_f32x8_neon_t(archmage::NeonToken::summon().unwrap(), x, y),
                _ => panic!("unknown function"),
            };
            check("neon", op, &records);
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        if archmage::Wasm128Token::summon().is_some() {
            let op: Op = match name {
                "expf" => {
                    |x, _y| a::exp_f32x8_wasm128_t(archmage::Wasm128Token::summon().unwrap(), x)
                }
                "logf" => {
                    |x, _y| a::ln_f32x8_wasm128_t(archmage::Wasm128Token::summon().unwrap(), x)
                }
                "log2f" => {
                    |x, _y| a::log2_f32x8_wasm128_t(archmage::Wasm128Token::summon().unwrap(), x)
                }
                "powf" => {
                    |x, y| a::pow_f32x8_wasm128_t(archmage::Wasm128Token::summon().unwrap(), x, y)
                }
                _ => panic!("unknown function"),
            };
            check("wasm128", op, &records);
        }
    }
}

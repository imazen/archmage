//! Hex protocol for independent rational-graph and MPFR checks.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use archmage::SimdToken;
use magetypes::{expf::*, logf::*};
use std::io::Read;

#[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn token_value<const O: u32>(token: Token, x: f32, log: bool) -> f32 {
    if log {
        try_log2_with_token::<O, _>(token, x).unwrap()
    } else {
        try_exp2_with_token::<O, _>(token, x).unwrap()
    }
}

fn native<const O: u32>(x: f32, log: bool) -> f32 {
    if std::env::var_os("AUDIT_SOFTWARE_TEST").is_some() {
        return token_value_scalar::<O>(archmage::ScalarToken, x, log);
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if std::env::var_os("AUDIT_V4X").is_some() {
        return token_value_v4x::<O>(
            archmage::X64V4xToken::summon().expect("V4x required"),
            x,
            log,
        );
    }
    #[cfg(target_arch = "x86_64")]
    return token_value_v3::<O>(archmage::X64V3Token::summon().expect("V3 required"), x, log);
    #[cfg(target_arch = "aarch64")]
    return token_value_neon::<O>(
        archmage::NeonToken::summon().expect("NEON required"),
        x,
        log,
    );
    #[cfg(target_arch = "wasm32")]
    return token_value_wasm128::<O>(
        archmage::Wasm128Token::summon().expect("SIMD required"),
        x,
        log,
    );
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    return token_value_scalar::<O>(archmage::ScalarToken, x, log);
}

fn array<const O: u32, const N: usize>(x: [f32; N], log: bool) -> [f32; N] {
    if log {
        try_log2_array_with::<O, N>(x).unwrap()
    } else {
        try_exp2_array_with::<O, N>(x).unwrap()
    }
}

fn policy<const O: u32, const CALLER: u32>(xs: &[f32; 16], log: bool) {
    let a16 = array::<O, 16>(*xs, log);
    for (i, &x) in xs.iter().enumerate() {
        let a8 = array::<O, 8>(xs[(i / 8) * 8..(i / 8 + 1) * 8].try_into().unwrap(), log);
        let a4 = array::<O, 4>(xs[(i / 4) * 4..(i / 4 + 1) * 4].try_into().unwrap(), log);
        let scalar = token_value_scalar::<CALLER>(archmage::ScalarToken, x, log);
        println!(
            "{}\t{:08x}\t{O}\t{:08x}\t{:08x}\t{:08x}\t{:08x}\t{:08x}\t{:08x}",
            if log { "L" } else { "E" },
            x.to_bits(),
            scalar.to_bits(),
            array::<O, 1>([x], log)[0].to_bits(),
            a4[i % 4].to_bits(),
            a8[i % 8].to_bits(),
            a16[i].to_bits(),
            native::<O>(x, log).to_bits()
        );
    }
}

fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let rows: Vec<_> = text
        .lines()
        .map(|line| {
            let mut p = line.split_whitespace();
            (
                p.next().unwrap() == "L",
                f32::from_bits(u32::from_str_radix(p.next().unwrap(), 16).unwrap()),
            )
        })
        .collect();
    assert_eq!(rows.len() % 16, 0);
    for chunk in rows.chunks_exact(16) {
        let log = chunk[0].0;
        assert!(chunk.iter().all(|r| r.0 == log));
        let xs: [f32; 16] = core::array::from_fn(|i| chunk[i].1);
        policy::<0, 4>(&xs, log);
        policy::<8, 12>(&xs, log);
        policy::<16, 20>(&xs, log);
        policy::<24, 28>(&xs, log);
        if std::env::var_os("AUDIT_NO_LEGACY").is_none() {
            policy::<64, 68>(&xs, log);
            policy::<72, 76>(&xs, log);
            policy::<80, 84>(&xs, log);
            policy::<88, 92>(&xs, log);
        }
    }
}

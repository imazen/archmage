//! Persistent, untimed provider process. Timing belongs to zenbench's parent.
#![feature(portable_simd)]
#![forbid(unsafe_code)]
use std::io::{BufRead, Write};
use std::simd::Simd;
mod arm_data;
mod arm_powf;

#[inline(always)]
fn calculate<const N: usize, const M: u8>(x: [f32; N], n: [f32; N]) -> [f32; N] {
    match M {
        0 => core::array::from_fn(|i| x[i].powf(n[i])),
        1 => core::array::from_fn(|i| libm::powf(x[i], n[i])),
        2 => core::array::from_fn(|i| sleef::f32::pow_u10(x[i], n[i])),
        3 => core::array::from_fn(|i| sleef::f32::pow_fast(x[i], n[i])),
        4 => sleef::f32x::pow_u10(Simd::from_array(x), Simd::from_array(n)).to_array(),
        5 => sleef::f32x::pow_fast(Simd::from_array(x), Simd::from_array(n)).to_array(),
        6 => core::array::from_fn(|i| arm_powf::pow::<true, true, false>(x[i], n[i])),
        7 => core::array::from_fn(|i| arm_powf::pow::<false, true, false>(x[i], n[i])),
        8 => core::array::from_fn(|i| arm_powf::pow::<true, false, false>(x[i], n[i])),
        9 => core::array::from_fn(|i| arm_powf::pow::<true, true, true>(x[i], n[i])),
        _ => panic!("unknown provider index"),
    }
}
#[inline(always)]
fn execute_method<const N: usize, const M: u8>(f: &[&str]) -> Vec<u32> {
    if f[0] == "E" {
        assert_eq!(f.len(), 3 + 2 * N);
        let x: [f32; N] = core::array::from_fn(|i| {
            f32::from_bits(u32::from_str_radix(f[3 + 2 * i], 16).unwrap())
        });
        let n: [f32; N] = core::array::from_fn(|i| {
            f32::from_bits(u32::from_str_radix(f[4 + 2 * i], 16).unwrap())
        });
        return calculate::<N, M>(x, n).map(f32::to_bits).to_vec();
    }
    assert!(["B", "T"].contains(&f[0]));
    assert_eq!(f.len(), 7);
    let size: usize = f[3].parse().unwrap();
    let chain = match f[4] {
        "0" => false,
        "1" => true,
        _ => panic!("bad mode"),
    };
    let n = f32::from_bits(u32::from_str_radix(f[5], 16).unwrap());
    let repeats: usize = f[6].parse().unwrap();
    assert!(
        [64, 4096].contains(&size)
            && size.is_multiple_of(N)
            && if f[0] == "T" {
                repeats == 1
            } else {
                repeats * size == 1_048_576
            }
    );
    let input: Vec<[f32; N]> = (0..size / N)
        .map(|i| core::array::from_fn(|j| (((i * N + j) % 255) + 1) as f32 / 255.))
        .collect();
    let mut output = vec![[0.; N]; input.len()];
    for _ in 0..repeats {
        let mut acc = input[0];
        for (x, out) in std::hint::black_box(&input).iter().zip(&mut output) {
            let y = calculate::<N, M>(if chain { acc } else { *x }, [std::hint::black_box(n); N]);
            acc = if chain {
                y.map(|v| f32::from_bits(0x3f000000 | (v.to_bits() & 0x007fffff)))
            } else {
                y
            };
            *out = acc;
        }
        std::hint::black_box(&output);
    }
    if f[0] == "T" {
        output.iter().flatten().map(|v| v.to_bits()).collect()
    } else {
        output.last().unwrap().map(f32::to_bits).to_vec()
    }
}
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn arm_v3<const N: usize, const M: u8>(_t: archmage::X64V3Token, f: &[&str]) -> Vec<u32> {
    execute_method::<N, M>(f)
}
fn execute_arm<const N: usize, const M: u8>(f: &[&str]) -> Vec<u32> {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::X64V3Token::summon() {
            return arm_v3::<N, M>(t, f);
        }
    }
    execute_method::<N, M>(f)
}
fn execute<const N: usize>(f: &[&str]) -> Vec<u32> {
    match f[1] {
        "std" => execute_method::<N, 0>(f),
        "Rust_libm_0_2_16" => execute_method::<N, 1>(f),
        "SLEEF_0_3_3_scalar_u10" => execute_method::<N, 2>(f),
        "SLEEF_0_3_3_scalar_fast" => execute_method::<N, 3>(f),
        "SLEEF_0_3_3_vector_u10" => execute_method::<N, 4>(f),
        "SLEEF_0_3_3_vector_fast" => execute_method::<N, 5>(f),
        "Arm_Rust_fused_cold" => execute_arm::<N, 6>(f),
        "Arm_Rust_separate_cold" => execute_arm::<N, 7>(f),
        "Arm_Rust_fused_warm" => execute_arm::<N, 8>(f),
        "Arm_Rust_fused_assume_Gamma" => execute_arm::<N, 9>(f),
        _ => panic!("unknown named provider"),
    }
}
fn serve(input: impl BufRead, mut out: impl Write) {
    for line in input.lines() {
        let line = line.unwrap();
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields == ["EMPTY"] {
            writeln!(out, "00000000").unwrap();
            out.flush().unwrap();
            continue;
        }
        let width: usize = fields[2].parse().unwrap();
        let bits = match width {
            1 => execute::<1>(&fields),
            4 => execute::<4>(&fields),
            8 => execute::<8>(&fields),
            16 => execute::<16>(&fields),
            _ => panic!("unsupported width"),
        };
        for value in bits {
            write!(out, "{value:08x} ").unwrap();
        }
        writeln!(out).unwrap();
        out.flush().unwrap();
    }
}
fn main() {
    if let Some(path) = std::env::args().nth(1) {
        let listener = std::os::unix::net::UnixListener::bind(path).unwrap();
        for connection in listener.incoming() {
            let stream = connection.unwrap();
            std::thread::spawn(move || {
                serve(std::io::BufReader::new(stream.try_clone().unwrap()), stream)
            });
        }
    } else {
        serve(
            std::io::stdin().lock(),
            std::io::BufWriter::new(std::io::stdout().lock()),
        );
    }
}

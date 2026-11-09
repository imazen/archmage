//! Persistent, untimed provider process. Timing belongs to zenbench's parent.
#![feature(portable_simd)]
#![forbid(unsafe_code)]
use std::io::{BufRead, Write};
use std::simd::Simd;
mod arm_data;
mod arm_powf;
mod library_powf;
use library_powf::LibraryToken;

#[inline(always)]
fn arm_array<const N: usize, const FUSED: bool, const COLD: bool, const ASSUME: bool>(
    x: [f32; N],
    n: [f32; N],
) -> [f32; N] {
    let mut output = [0.; N];
    let mut i = 0;
    while i < N {
        output[i] = arm_powf::pow::<FUSED, COLD, ASSUME>(x[i], n[i]);
        i += 1;
    }
    output
}

#[inline(always)]
fn arm_unrolled<const N: usize, const ASSUME: bool>(x: [f32; N], n: [f32; N]) -> [f32; N] {
    const {
        assert!(N <= 16);
    }
    let mut out = [0.; N];
    macro_rules! lane {
        ($i:literal) => {
            if $i < N {
                out[$i] = arm_powf::pow::<true, true, ASSUME>(x[$i], n[$i]);
            }
        };
    }
    lane!(0);
    lane!(1);
    lane!(2);
    lane!(3);
    lane!(4);
    lane!(5);
    lane!(6);
    lane!(7);
    lane!(8);
    lane!(9);
    lane!(10);
    lane!(11);
    lane!(12);
    lane!(13);
    lane!(14);
    lane!(15);
    out
}

#[inline(always)]
fn calculate<T: LibraryToken, const N: usize, const M: u8>(
    t: T,
    x: [f32; N],
    n: [f32; N],
) -> [f32; N] {
    match M {
        0 => core::array::from_fn(|i| x[i].powf(n[i])),
        1 => core::array::from_fn(|i| libm::powf(x[i], n[i])),
        2 => core::array::from_fn(|i| sleef::f32::pow_u10(x[i], n[i])),
        3 => core::array::from_fn(|i| sleef::f32::pow_fast(x[i], n[i])),
        4 => sleef::f32x::pow_u10(Simd::from_array(x), Simd::from_array(n)).to_array(),
        5 => sleef::f32x::pow_fast(Simd::from_array(x), Simd::from_array(n)).to_array(),
        6 => arm_array::<N, true, true, false>(x, n),
        7 => arm_array::<N, false, true, false>(x, n),
        8 => arm_array::<N, true, false, false>(x, n),
        9 => arm_array::<N, true, true, true>(x, n),
        10 => arm_unrolled::<N, false>(x, n),
        11 => arm_unrolled::<N, true>(x, n),
        12..=16 => library_powf::calculate::<T, N, M>(t, x, n),
        _ => panic!("unknown provider index"),
    }
}
#[inline(always)]
fn execute_method<T: LibraryToken, const N: usize, const M: u8>(t: T, f: &[&str]) -> Vec<u32> {
    if f[0] == "E" {
        assert_eq!(f.len(), 3 + 2 * N);
        let x: [f32; N] = core::array::from_fn(|i| {
            f32::from_bits(u32::from_str_radix(f[3 + 2 * i], 16).unwrap())
        });
        let n: [f32; N] = core::array::from_fn(|i| {
            f32::from_bits(u32::from_str_radix(f[4 + 2 * i], 16).unwrap())
        });
        return calculate::<T, N, M>(t, x, n).map(f32::to_bits).to_vec();
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
            let y = calculate::<T, N, M>(
                t,
                if chain { acc } else { *x },
                [std::hint::black_box(n); N],
            );
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
fn arm_v3<const N: usize, const M: u8>(t: archmage::X64V3Token, f: &[&str]) -> Vec<u32> {
    execute_method::<_, N, M>(t, f)
}
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn arm_neon<const N: usize, const M: u8>(t: archmage::NeonToken, f: &[&str]) -> Vec<u32> {
    execute_method::<_, N, M>(t, f)
}
fn execute_arm<const N: usize, const M: u8>(f: &[&str]) -> Vec<u32> {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::X64V3Token::summon() {
            return arm_v3::<N, M>(t, f);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        return arm_neon::<N, M>(archmage::NeonToken::summon().expect("required NEON"), f);
    }
    #[cfg(not(target_arch = "aarch64"))]
    execute_method::<_, N, M>(archmage::ScalarToken, f)
}
fn execute<const N: usize>(f: &[&str]) -> Vec<u32> {
    match f[1] {
        "std" => execute_method::<_, N, 0>(archmage::ScalarToken, f),
        "Rust_libm_0_2_16" => execute_method::<_, N, 1>(archmage::ScalarToken, f),
        "SLEEF_0_3_3_scalar_u10" => execute_method::<_, N, 2>(archmage::ScalarToken, f),
        "SLEEF_0_3_3_scalar_fast" => execute_method::<_, N, 3>(archmage::ScalarToken, f),
        "SLEEF_0_3_3_vector_u10" => execute_method::<_, N, 4>(archmage::ScalarToken, f),
        "SLEEF_0_3_3_vector_fast" => execute_method::<_, N, 5>(archmage::ScalarToken, f),
        "Arm_Rust_fused_cold" => execute_arm::<N, 6>(f),
        "Arm_Rust_separate_cold" => execute_arm::<N, 7>(f),
        "Arm_Rust_fused_warm" => execute_arm::<N, 8>(f),
        "Arm_Rust_fused_assume_Gamma" => execute_arm::<N, 9>(f),
        "Arm_Rust_fused_unrolled" => execute_arm::<N, 10>(f),
        "Arm_Rust_assume_Gamma_unrolled" => execute_arm::<N, 11>(f),
        "Magetypes_HIGHP_Full_packed" => execute_arm::<N, 12>(f),
        "Magetypes_HIGHP_Gamma_checked_packed" => execute_arm::<N, 13>(f),
        "Magetypes_HIGHP_Gamma_assumed_packed" => execute_arm::<N, 14>(f),
        "Magetypes_legacy_midp" => execute_arm::<N, 15>(f),
        "Magetypes_legacy_lowp" => execute_arm::<N, 16>(f),
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

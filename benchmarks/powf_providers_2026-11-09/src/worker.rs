//! Persistent, untimed provider process. Timing belongs to zenbench's parent.
#![feature(portable_simd)]
#![forbid(unsafe_code)]
use std::io::{BufRead, Write};
use std::simd::Simd;

fn calculate<const N: usize>(method: &str, x: [f32; N], n: [f32; N]) -> [f32; N] {
    match method {
        "std" => core::array::from_fn(|i| x[i].powf(n[i])),
        "Rust_libm_0_2_16" => core::array::from_fn(|i| libm::powf(x[i], n[i])),
        "SLEEF_0_3_3_scalar_u10" => core::array::from_fn(|i| sleef::f32::pow_u10(x[i], n[i])),
        "SLEEF_0_3_3_scalar_fast" => core::array::from_fn(|i| sleef::f32::pow_fast(x[i], n[i])),
        "SLEEF_0_3_3_vector_u10" => sleef::f32x::pow_u10(Simd::from_array(x), Simd::from_array(n)).to_array(),
        "SLEEF_0_3_3_vector_fast" => sleef::f32x::pow_fast(Simd::from_array(x), Simd::from_array(n)).to_array(),
        _ => panic!("unknown named provider {method}"),
    }
}
fn execute<const N: usize>(f: &[&str]) -> Vec<u32> {
    if f[0] == "E" {
        assert_eq!(f.len(), 3 + 2*N);
        let x = core::array::from_fn(|i| f32::from_bits(u32::from_str_radix(f[3+2*i],16).unwrap()));
        let n = core::array::from_fn(|i| f32::from_bits(u32::from_str_radix(f[4+2*i],16).unwrap()));
        return calculate(f[1], x, n).map(f32::to_bits).to_vec();
    }
    assert_eq!(f[0], "B"); assert_eq!(f.len(), 7);
    let size: usize = f[3].parse().unwrap();
    let chain = match f[4] { "0" => false, "1" => true, _=>panic!("bad mode") };
    let n = f32::from_bits(u32::from_str_radix(f[5],16).unwrap());
    let repeats: usize = f[6].parse().unwrap();
    assert!([64,4096].contains(&size) && size%N==0 && repeats*size==1_048_576);
    let input: Vec<[f32;N]> = (0..size/N).map(|i|core::array::from_fn(|j|(((i*N+j)%255)+1) as f32/255.)).collect();
    let mut output = vec![[0.;N];input.len()];
    for _ in 0..repeats {
        let mut acc = input[0];
        for (x,out) in std::hint::black_box(&input).iter().zip(&mut output) {
            let y = calculate(f[1], if chain { acc } else { *x }, [std::hint::black_box(n);N]);
            acc = if chain {y.map(|v|f32::from_bits(0x3f000000|(v.to_bits()&0x007fffff)))} else {y};
            *out=acc;
        }
        std::hint::black_box(&output);
    }
    output.last().unwrap().map(f32::to_bits).to_vec()
}
fn serve(input:impl BufRead, mut out:impl Write) {
    for line in input.lines() {
        let line=line.unwrap(); let fields:Vec<_>=line.split_whitespace().collect();
        if fields == ["EMPTY"] { writeln!(out,"00000000").unwrap(); out.flush().unwrap(); continue; }
        let width:usize=fields[2].parse().unwrap();
        let bits=match width {1=>execute::<1>(&fields),4=>execute::<4>(&fields),8=>execute::<8>(&fields),16=>execute::<16>(&fields),_=>panic!("unsupported width")};
        for value in bits { write!(out,"{value:08x} ").unwrap(); }
        writeln!(out).unwrap(); out.flush().unwrap();
    }
}
fn main() {
    if let Some(path)=std::env::args().nth(1) {
        let listener=std::os::unix::net::UnixListener::bind(path).unwrap();
        for connection in listener.incoming() {
            let stream=connection.unwrap();
            std::thread::spawn(move||serve(std::io::BufReader::new(stream.try_clone().unwrap()),stream));
        }
    } else { serve(std::io::stdin().lock(),std::io::BufWriter::new(std::io::stdout().lock())); }
}

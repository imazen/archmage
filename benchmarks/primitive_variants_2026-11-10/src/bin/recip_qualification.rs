//! Stream a C-generated reciprocal oracle; fixed vectors and a 64 KiB reader.
#![forbid(unsafe_code)]
#[path = "../seed_graphs.rs"]
mod parallel;
#[path = "../rcp14.rs"]
mod rcp14;
use archmage::{ScalarToken, SimdToken};
use std::io::{BufReader, Read, Seek, SeekFrom};
#[derive(Debug)]
struct Summary {
    lanes: u64,
    worst_parallel: f64,
    worst_rcp14: f64,
}
fn relative(y: f32, x: f32) -> f64 {
    let exact = 1.0 / f64::from(x);
    ((f64::from(y) - exact) / exact).abs()
}
fn scaled(bits: u32, input_exp: u32, negative: bool) -> u32 {
    let b = (i64::from(bits) - ((i64::from(input_exp) - 127) << 23)) as u32;
    b | if negative { 0x80000000 } else { 0 }
}
macro_rules! check {
    ($name:ident,$v:ident,$n:literal,$par:ident,$em:ident,$native:ident) => {
        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        fn $name(t: Token, path: &str, stride: usize, hardware: bool) -> Summary {
            use magetypes::simd::generic::$v as V;
            let file = std::fs::File::open(path).expect("explicit reference file required");
            assert_eq!(
                file.metadata().unwrap().len(),
                8388608 * 8,
                "reference size"
            );
            let mut reader = BufReader::with_capacity(65536, file);
            let mut buffer = [0_u8; 64];
            let mut summary = Summary {
                lanes: 0,
                worst_parallel: 0.0,
                worst_rcp14: 0.0,
            };
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            let native = if hardware {
                Some(archmage::X64V4Token::summon().expect("requested RCP14 proof"))
            } else {
                None
            };
            #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
            assert!(!hardware, "hardware RCP14 requires its compiled ISA");
            let mut base = 0_u32;
            let mut verify = |input: [f32; $n], p: [u32; $n], r: [u32; $n], tail: bool| {
                let x = V::from_array_t(t, input);
                let a = parallel::$par(t, x).to_array();
                let b = rcp14::$em(t, x).to_array();
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                if let Some(proof) = native {
                    assert_eq!(
                        rcp14::$native(proof, input).map(f32::to_bits),
                        r,
                        "native RCP14"
                    );
                }
                for i in 0..$n {
                    assert_eq!(
                        a[i].to_bits(),
                        p[i],
                        "parallel x={:08x}",
                        input[i].to_bits()
                    );
                    assert_eq!(b[i].to_bits(), r[i], "RCP14 x={:08x}", input[i].to_bits());
                    assert_eq!(rcp14::scalar(input[i]).to_bits(), r[i], "scalar RCP14");
                    let ep = relative(a[i], input[i]);
                    let er = relative(b[i], input[i]);
                    assert!(
                        ep <= 1e-4 && er < 2_f64.powi(-14),
                        "numerical bounds x={:08x}",
                        input[i].to_bits()
                    );
                    summary.worst_parallel = summary.worst_parallel.max(ep);
                    summary.worst_rcp14 = summary.worst_rcp14.max(er);
                    summary.lanes += 1;
                }
                if tail {
                    assert_eq!(
                        parallel::$par(t, V::splat_t(t, input[0])).to_array()[0].to_bits(),
                        p[0]
                    );
                    assert_eq!(
                        rcp14::$em(t, V::splat_t(t, input[0])).to_array()[0].to_bits(),
                        r[0]
                    );
                }
            };
            while base < 8388608 {
                reader.read_exact(&mut buffer[..$n / 2 * 8]).unwrap();
                let input = core::array::from_fn(|i| {
                    f32::from_bits(
                        (0x3f800000 + base + (i / 2) as u32)
                            | if i % 2 == 1 { 0x80000000 } else { 0 },
                    )
                });
                let word = |i: usize, col: usize| {
                    u32::from_le_bytes(
                        buffer[i / 2 * 8 + col * 4..i / 2 * 8 + col * 4 + 4]
                            .try_into()
                            .unwrap(),
                    ) | if i % 2 == 1 { 0x80000000 } else { 0 }
                };
                verify(
                    input,
                    core::array::from_fn(|i| word(i, 0)),
                    core::array::from_fn(|i| word(i, 1)),
                    base % 4096 == 0,
                );
                base += stride.max($n / 2) as u32;
                if stride > $n / 2 {
                    reader.seek(SeekFrom::Start(u64::from(base) * 8)).unwrap();
                }
            }
            // Verify power-of-two homogeneity at every accepted exponent, using
            // mixed lanes and both signs. No materialized input-space array.
            for exp in 2..=252_u32 {
                for m in [
                    0_u32, 1, 127, 128, 0x1ffff, 0x20000, 0x3fffff, 0x400000, 0x7ffffe, 0x7fffff,
                ] {
                    if exp == 252 && m != 0 {
                        continue;
                    }
                    reader.seek(SeekFrom::Start(u64::from(m) * 8)).unwrap();
                    reader.read_exact(&mut buffer[..8]).unwrap();
                    let p = u32::from_le_bytes(buffer[..4].try_into().unwrap());
                    let r = u32::from_le_bytes(buffer[4..8].try_into().unwrap());
                    let input = core::array::from_fn(|i| {
                        f32::from_bits((exp << 23) | m | if i % 2 == 1 { 0x80000000 } else { 0 })
                    });
                    verify(
                        input,
                        core::array::from_fn(|i| scaled(p, exp, i % 2 == 1)),
                        core::array::from_fn(|i| scaled(r, exp, i % 2 == 1)),
                        true,
                    );
                }
            }
            summary
        }
    };
}
check!(check4, f32x4, 4, recip4, emulate4, native4);
check!(check8, f32x8, 8, recip8, emulate8, native8);
#[cfg(feature = "w512")]
check!(check16, f32x16, 16, recip16, emulate16, native16);
macro_rules! run {
    ($token:expr,$f4:ident,$f8:ident,$f16:ident,$path:ident,$dense:ident,$hardware:expr) => {{
        let t = $token;
        println!(
            "x4 {:?}",
            $f4(t, &$path, if $dense { 2 } else { 4096 }, $hardware)
        );
        println!("x8 {:?}", $f8(t, &$path, 4096, $hardware));
        #[cfg(feature = "w512")]
        println!("x16 {:?}", $f16(t, &$path, 4096, $hardware));
    }};
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let value = |key: &str| {
        args.iter()
            .position(|v| v == key)
            .and_then(|i| args.get(i + 1))
            .expect("--tier and --reference are required")
            .clone()
    };
    let tier = value("--tier");
    let path = value("--reference");
    let dense = args.iter().any(|v| v == "--dense");
    println!(
        "tier={tier}, dense signed x4={dense}; x8/x16 sampled and every edge exponent; explicit C oracle {path}"
    );
    match tier.as_str() {
        "scalar" => run!(
            ScalarToken,
            check4_scalar,
            check8_scalar,
            check16_scalar,
            path,
            dense,
            false
        ),
        #[cfg(target_arch = "x86_64")]
        "v3" => run!(
            archmage::X64V3Token::summon().expect("V3 required"),
            check4_v3,
            check8_v3,
            check16_v3,
            path,
            dense,
            false
        ),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4" => run!(
            archmage::X64V4Token::summon().expect("V4 required"),
            check4_v4,
            check8_v4,
            check16_v4,
            path,
            dense,
            true
        ),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4x" => run!(
            archmage::X64V4xToken::summon().expect("V4x required"),
            check4_v4x,
            check8_v4x,
            check16_v4x,
            path,
            dense,
            true
        ),
        #[cfg(target_arch = "aarch64")]
        "neon" => run!(
            archmage::NeonToken::summon().expect("NEON required"),
            check4_neon,
            check8_neon,
            check16_neon,
            path,
            dense,
            false
        ),
        #[cfg(target_arch = "wasm32")]
        "wasm" => run!(
            archmage::Wasm128Token::summon().expect("SIMD128 required"),
            check4_wasm128,
            check8_wasm128,
            check16_wasm128,
            path,
            dense,
            false
        ),
        _ => panic!("requested tier is not compiled into this tool"),
    }
}

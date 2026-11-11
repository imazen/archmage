//! Stream independent C fmaf/integer-division references; fixed memory.
#![forbid(unsafe_code)]
#[path = "../recip_research.rs"]
mod research;
use archmage::{ScalarToken, SimdToken};
use std::io::{BufReader, Read, Seek, SeekFrom};
#[derive(Debug, Default)]
struct Summary {
    lanes: u64,
    midp_error: f64,
    rn14_error: f64,
    rn8_error: f64,
}
fn scale(bits: u32, e: u32, negative: bool) -> u32 {
    (i64::from(bits) - ((i64::from(e) - 127) << 23)) as u32 | if negative { 0x80000000 } else { 0 }
}
macro_rules! check {
    ($name:ident,$v:ident,$n:literal,$bounded:ident,$canonical:ident) => {
        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        fn $name(t: Token, path: &str, dense: bool) -> Summary {
            use magetypes::simd::generic::$v as V;
            let file = std::fs::File::open(path).expect("explicit C reference required");
            assert_eq!(file.metadata().unwrap().len(), 8388608 * 16);
            let mut r = BufReader::with_capacity(65536, file);
            let mut buffer = [0u8; 128];
            let mut summary = Summary::default();
            let mut verify = |input: [f32; $n], expected: [[u32; $n]; 4], tails: bool| {
                let x = V::from_array_t(t, input);
                let bm = research::$bounded::<2, _>(t, x).to_array();
                let bl = research::$bounded::<1, _>(t, x).to_array();
                let a = research::$canonical::<14, 0, _>(t, x).to_array();
                let b = research::$canonical::<14, 1, _>(t, x).to_array();
                let c = research::$canonical::<14, 2, _>(t, x).to_array();
                let cold_native = research::$canonical::<14, 3, _>(t, x).to_array();
                let cold_division = research::$canonical::<14, 4, _>(t, x).to_array();
                let cold_native8 = research::$canonical::<8, 3, _>(t, x).to_array();
                let cold_division8 = research::$canonical::<8, 4, _>(t, x).to_array();
                let d = research::$canonical::<8, 0, _>(t, x).to_array();
                let e = research::$canonical::<8, 1, _>(t, x).to_array();
                let f = research::$canonical::<8, 2, _>(t, x).to_array();
                for i in 0..$n {
                    assert_eq!(
                        bm[i].to_bits(),
                        expected[0][i],
                        "bounded MIDP x={:08x}",
                        input[i].to_bits()
                    );
                    assert_eq!(
                        bl[i].to_bits(),
                        expected[1][i],
                        "bounded LOWP x={:08x}",
                        input[i].to_bits()
                    );
                    for (label, y) in [
                        ("native-certified14", a[i]),
                        ("native-guarded14", b[i]),
                        ("division-guarded14", c[i]),
                        ("cold-native14", cold_native[i]),
                        ("cold-division14", cold_division[i]),
                    ] {
                        assert_eq!(
                            y.to_bits(),
                            expected[2][i],
                            "{label} x={:08x}",
                            input[i].to_bits()
                        );
                    }
                    for (label, y) in [
                        ("native-certified8", d[i]),
                        ("native-guarded8", e[i]),
                        ("division-guarded8", f[i]),
                        ("cold-native8", cold_native8[i]),
                        ("cold-division8", cold_division8[i]),
                    ] {
                        assert_eq!(
                            y.to_bits(),
                            expected[3][i],
                            "{label} x={:08x}",
                            input[i].to_bits()
                        );
                    }
                    let error = |y: f32| (f64::from(y) * f64::from(input[i]) - 1.0).abs();
                    let (em, e14, e8) = (error(bm[i]), error(a[i]), error(d[i]));
                    assert!(em <= 1e-4 && e14 <= 2_f64.powi(-14) && e8 <= 2_f64.powi(-8));
                    summary.midp_error = summary.midp_error.max(em);
                    summary.rn14_error = summary.rn14_error.max(e14);
                    summary.rn8_error = summary.rn8_error.max(e8);
                    summary.lanes += 1;
                }
                if tails {
                    for i in 0..2 {
                        let x = V::splat_t(t, input[i]);
                        assert_eq!(
                            research::$bounded::<2, _>(t, x).to_array()[0].to_bits(),
                            expected[0][i]
                        );
                        assert_eq!(
                            research::$canonical::<14, 0, _>(t, x).to_array()[0].to_bits(),
                            expected[2][i]
                        );
                        assert_eq!(
                            research::$canonical::<14, 1, _>(t, x).to_array()[0].to_bits(),
                            expected[2][i]
                        );
                        assert_eq!(
                            research::$canonical::<14, 2, _>(t, x).to_array()[0].to_bits(),
                            expected[2][i]
                        );
                        assert_eq!(
                            research::$canonical::<14, 3, _>(t, x).to_array()[0].to_bits(),
                            expected[2][i]
                        );
                        assert_eq!(
                            research::$canonical::<14, 4, _>(t, x).to_array()[0].to_bits(),
                            expected[2][i]
                        );
                    }
                }
            };
            let mut base = 0u32;
            while base < 8388608 {
                r.read_exact(&mut buffer[..$n / 2 * 16]).unwrap();
                let input = core::array::from_fn(|i| {
                    f32::from_bits(
                        (0x3f800000 + base + (i / 2) as u32)
                            | if i % 2 == 1 { 0x80000000 } else { 0 },
                    )
                });
                let expected = core::array::from_fn(|col| {
                    core::array::from_fn(|i| {
                        u32::from_le_bytes(
                            buffer[i / 2 * 16 + col * 4..i / 2 * 16 + col * 4 + 4]
                                .try_into()
                                .unwrap(),
                        ) | if i % 2 == 1 { 0x80000000 } else { 0 }
                    })
                });
                verify(input, expected, base % 4096 == 0);
                base += if dense { $n / 2 } else { 4096 };
                if !dense {
                    r.seek(SeekFrom::Start(u64::from(base) * 16)).unwrap();
                }
            }
            for exp in 2..=252 {
                for m in [
                    0u32, 1, 127, 128, 0x1ffff, 0x20000, 0x3fffff, 0x400000, 0x7ffffe, 0x7fffff,
                ] {
                    if exp == 252 && m != 0 {
                        continue;
                    }
                    r.seek(SeekFrom::Start(u64::from(m) * 16)).unwrap();
                    r.read_exact(&mut buffer[..16]).unwrap();
                    let input = core::array::from_fn(|i| {
                        f32::from_bits((exp << 23) | m | if i % 2 == 1 { 0x80000000 } else { 0 })
                    });
                    let expected = core::array::from_fn(|col| {
                        core::array::from_fn(|i| {
                            scale(
                                u32::from_le_bytes(
                                    buffer[col * 4..col * 4 + 4].try_into().unwrap(),
                                ),
                                exp,
                                i % 2 == 1,
                            )
                        })
                    });
                    verify(input, expected, true);
                }
            }
            summary
        }
    };
}
check!(check4, f32x4, 4, bounded4, canonical4);
check!(check8, f32x8, 8, bounded8, canonical8);
#[cfg(feature = "w512")]
check!(check16, f32x16, 16, bounded16, canonical16);
macro_rules! run {
    ($t:expr,$a:ident,$b:ident,$c:ident,$path:ident,$dense:ident) => {{
        let t = $t;
        println!("x4 {:?}", $a(t, &$path, $dense));
        println!("x8 {:?}", $b(t, &$path, $dense));
        #[cfg(feature = "w512")]
        println!("x16 {:?}", $c(t, &$path, $dense));
    }};
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let value = |key: &str| {
        args.iter()
            .position(|x| x == key)
            .and_then(|i| args.get(i + 1))
            .expect("--reference and --tier required")
            .clone()
    };
    let path = value("--reference");
    let tier = value("--tier");
    let dense = args.iter().any(|x| x == "--dense");
    println!("tier={tier}; dense all widths={dense}; signed C/integer oracle; both-sign tails");
    match tier.as_str() {
        "scalar" => run!(
            ScalarToken,
            check4_scalar,
            check8_scalar,
            check16_scalar,
            path,
            dense
        ),
        #[cfg(target_arch = "x86_64")]
        "v3" => run!(
            archmage::X64V3Token::summon().expect("V3 required"),
            check4_v3,
            check8_v3,
            check16_v3,
            path,
            dense
        ),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4" => run!(
            archmage::X64V4Token::summon().expect("V4 required"),
            check4_v4,
            check8_v4,
            check16_v4,
            path,
            dense
        ),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4x" => run!(
            archmage::X64V4xToken::summon().expect("V4x required"),
            check4_v4x,
            check8_v4x,
            check16_v4x,
            path,
            dense
        ),
        #[cfg(target_arch = "aarch64")]
        "neon" => run!(
            archmage::NeonToken::summon().expect("NEON required"),
            check4_neon,
            check8_neon,
            check16_neon,
            path,
            dense
        ),
        #[cfg(target_arch = "wasm32")]
        "wasm" => run!(
            archmage::Wasm128Token::summon().expect("SIMD128 required"),
            check4_wasm128,
            check8_wasm128,
            check16_wasm128,
            path,
            dense
        ),
        _ => panic!("requested tier unavailable"),
    }
}

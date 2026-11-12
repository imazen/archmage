//! Finite prototype accuracy and raw-estimate witnesses; no performance timers.
#![forbid(unsafe_code)]
#[path = "../seed_graphs.rs"]
mod graphs;
use archmage::{ScalarToken, SimdToken};
#[archmage::magetypes(v3, neon, scalar)]
fn scan(t: Token, stride: usize) -> [(f64, u32); 2] {
    use magetypes::simd::generic::f32x4 as V;
    let mut worst = [(0.0, 0); 2];
    let mut check = |input: [f32; 4]| {
        let x = V::from_array_t(t, input);
        let output = [
            graphs::recip4(t, x).to_array(),
            graphs::rsqrt4(t, x).to_array(),
        ];
        for j in 0..2 {
            for (i, &x) in input.iter().enumerate() {
                let seed = f32::from_bits(if j == 0 {
                    0x7ef127ea_u32.wrapping_sub(x.to_bits())
                } else {
                    0x5f3759df_u32.wrapping_sub(x.to_bits() >> 1)
                });
                let expected = if j == 0 {
                    let e = (-x).mul_add(seed, 1.0);
                    seed.mul_add(e, seed) * e.mul_add(e, 1.0)
                } else {
                    let e = (-x).mul_add(seed * seed, 1.0);
                    seed * 0.375_f32
                        .mul_add(e, f32::from_bits(0x3f004b44))
                        .mul_add(e, 1.0)
                };
                assert_eq!(
                    output[j][i].to_bits(),
                    expected.to_bits(),
                    "graph {j} bits {:08x}",
                    x.to_bits()
                );
                let exact = if j == 0 {
                    1.0 / x as f64
                } else {
                    1.0 / (x as f64).sqrt()
                };
                let err = ((output[j][i] as f64 - exact) / exact).abs();
                assert!(
                    err <= 1e-4,
                    "graph {j} bits {:08x} error {err}",
                    x.to_bits()
                );
                if err > worst[j].0 {
                    worst[j] = (err, x.to_bits());
                }
            }
        }
    };
    for start in (0x3f800000..0x40800000_u32).step_by(stride) {
        check(core::array::from_fn(|i| f32::from_bits(start + i as u32)));
    }
    for e in [
        2, 3, 62, 63, 64, 125, 126, 127, 128, 190, 191, 250, 251, 252,
    ] {
        for m in [
            0, 1, 0x1fffff, 0x3fffff, 0x400000, 0x600001, 0x7ffffe, 0x7fffff,
        ] {
            let bits = e << 23 | m;
            if bits <= 252 << 23 {
                check([f32::from_bits(bits); 4]);
            }
        }
    }
    worst
}
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn raw<T: magetypes::simd::backends::F32x4Backend>(t: T, x: f32) -> [u32; 2] {
    use magetypes::simd::backends::F32x4Backend as B;
    let v = B::from_array(t, [x; 4]);
    [
        B::to_array(t, B::rcp_approx(t, v))[0].to_bits(),
        B::to_array(t, B::rsqrt_approx(t, v))[0].to_bits(),
    ]
}
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn raw_v3(t: archmage::X64V3Token, x: f32) -> [u32; 2] {
    raw(t, x)
}
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn raw_neon(_t: archmage::NeonToken, x: f32) -> [u32; 2] {
    use core::arch::aarch64 as n;
    let v = n::vdupq_n_f32(x);
    [
        n::vgetq_lane_f32::<0>(n::vrecpeq_f32(v)).to_bits(),
        n::vgetq_lane_f32::<0>(n::vrsqrteq_f32(v)).to_bits(),
    ]
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn raw14(t: archmage::X64V4Token, x: f32) -> [u32; 2] {
    use magetypes::simd::backends::F32x16Backend as B;
    let v = B::from_array(t, [x; 16]);
    [
        B::to_array(t, B::rcp_approx(t, v))[0].to_bits(),
        B::to_array(t, B::rsqrt_approx(t, v))[0].to_bits(),
    ]
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn masks() {
    let a = archmage::X64V3Token::summon().expect("V3 required");
    let b = archmage::X64V4Token::summon().expect("V4 required");
    for keep in [0_u32, 4, 8, 12] {
        let drop = 23 - keep;
        let mask = u32::MAX << drop;
        let round = |bits: u32| (bits + ((1 << (drop - 1)) - 1) + ((bits >> drop) & 1)) & mask;
        let mut witness = None;
        'outer: for cell in 0..1_u32 << keep {
            let midpoint = 1.0 + (cell as f64 + 0.5) / (1_u32 << keep) as f64;
            let center = (1.0 / midpoint) as f32;
            for delta in -8_i64..=8 {
                let bits = (i64::from(center.to_bits()) + delta) as u32;
                let x = f32::from_bits(bits);
                let v3 = raw_v3(a, x)[0];
                let v4 = raw14(b, x)[0];
                if round(v3) != round(v4) {
                    witness = Some((bits, v3, v4, round(v3), round(v4)));
                    break 'outer;
                }
            }
        }
        match witness {
            Some((x, a, b, c, d)) => println!(
                "rounded keep={keep} x={x:08x} rawV3={a:08x} rawV4={b:08x} quantV3={c:08x} quantV4={d:08x}"
            ),
            None => println!(
                "rounded keep={keep}: no witness in this finite boundary sample; no guarantee follows"
            ),
        }
    }
    let a = raw_v3(a, 1.0)[0];
    let b = raw14(b, 1.0)[0];
    println!(
        "truncate keep=0 x=3f800000 rawV3={a:08x} rawV4={b:08x} quantV3={:08x} quantV4={:08x}",
        a & 0xff800000,
        b & 0xff800000
    );
}
fn main() {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if std::env::args().any(|v| v == "--masks") {
        masks();
        return;
    }
    let seeds = std::env::args().any(|v| v == "--seeds");
    if seeds {
        for bits in [
            0x3f000000, 0x3f7fffff, 0x3f800000, 0x3f800001, 0x3f804000, 0x3f808000, 0x3f810000,
            0x3f98f7ff, 0x3fc00000, 0x3fffffff, 0x40000000, 0x40000001, 0x40400000,
        ] {
            let x = f32::from_bits(bits);
            #[cfg(target_arch = "x86_64")]
            {
                let a = raw_v3(archmage::X64V3Token::summon().expect("V3 required"), x);
                println!("V3 {bits:08x} {:08x} {:08x}", a[0], a[1]);
                #[cfg(feature = "avx512")]
                {
                    let b = raw14(archmage::X64V4Token::summon().expect("V4 required"), x);
                    println!("V4x16 {bits:08x} {:08x} {:08x}", b[0], b[1]);
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                let a = raw_neon(archmage::NeonToken::summon().expect("NEON required"), x);
                println!("NEON {bits:08x} {:08x} {:08x}", a[0], a[1]);
            }
        }
        return;
    }
    let stride = if std::env::args().any(|v| v == "--dense") {
        4
    } else {
        4096
    };
    println!(
        "pointwise prototypes [1,4), stride {stride}, directed edge exponents; MIDP bound 1e-4"
    );
    println!("Scalar {:?}", scan_scalar(ScalarToken, stride));
    #[cfg(target_arch = "x86_64")]
    println!(
        "V3 {:?}",
        scan_v3(archmage::X64V3Token::summon().expect("V3 required"), stride)
    );
    #[cfg(target_arch = "aarch64")]
    println!(
        "NEON {:?}",
        scan_neon(
            archmage::NeonToken::summon().expect("NEON required"),
            stride
        )
    );
}

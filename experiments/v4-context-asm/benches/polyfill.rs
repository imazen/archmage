//! Times the polyfill kernels (`group_p`) at 4, 8 and 16 lanes on the tier
//! this host has: NEON on AArch64, AVX2 (`v3`) on x86-64.
//!
//!     cargo bench --bench polyfill -- --format=md
//!
//! Each call goes through the same `#[inline(never)]` entry whose assembly
//! `dump_polyfill.sh` dumps. Lengths are multiples of 16, so no kernel runs a
//! scalar tail. The gain factor is 1.0, so the buffer keeps its values.
use archmage::prelude::*;
use std::hint::black_box;
use v4_context_asm::*;
use zenbench::prelude::*;

/// Floats per call: 256 B, 4 KiB, 32 KiB and 4 MiB of data.
const SIZES: [usize; 4] = [64, 1024, 8192, 1 << 20];

fn data(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i.wrapping_mul(2_654_435_761) >> 8) & 0xffff) as f32 / 65536.0 + 0.5)
        .collect()
}

macro_rules! tier_benches {
    ($token:ty, $g4:ident, $g8:ident, $g16:ident, $s4:ident, $s8:ident, $s16:ident) => {
        fn benches(suite: &mut Suite) {
            let token = <$token>::summon().expect("this host lacks the tier");
            for n in SIZES {
                suite.group(format!("gain_{n}"), |g| {
                    g.throughput(Throughput::Elements(n as u64));
                    for (name, kernel) in [
                        ("f32x4", $g4 as fn($token, &mut [f32], f32)),
                        ("f32x8", $g8),
                        ("f32x16", $g16),
                    ] {
                        let mut buf = data(n);
                        g.bench(name, move |b| {
                            b.iter(|| kernel(token, black_box(&mut buf[..]), black_box(1.0)))
                        });
                    }
                });
                suite.group(format!("sum_{n}"), |g| {
                    g.throughput(Throughput::Elements(n as u64));
                    for (name, kernel) in [
                        ("f32x4", $s4 as fn($token, &[f32]) -> f32),
                        ("f32x8", $s8),
                        ("f32x16", $s16),
                    ] {
                        let buf = data(n);
                        g.bench(name, move |b| b.iter(|| kernel(token, black_box(&buf[..]))));
                    }
                });
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
tier_benches!(NeonToken, p4_gain_neon, p8_gain_neon, p16_gain_neon, p4_sum_neon, p8_sum_neon, p16_sum_neon);
#[cfg(target_arch = "x86_64")]
tier_benches!(X64V3Token, p4_gain_v3, p8_gain_v3, p16_gain_v3, p4_sum_v3, p8_sum_v3, p16_sum_v3);

zenbench::main!(benches);

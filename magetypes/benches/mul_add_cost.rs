//! `mul_add` against `a * b + c`, per backend.
//!
//! Since 0.9.30 `mul_add` rounds once on every backend: hardware FMA on x86
//! v3/v4 and NEON, software fusion on the scalar backend and on strict WASM.
//! `a * b + c` rounds twice; it is what the scalar and WASM backends computed
//! for `mul_add` in 0.9.29. zenbench does not build for wasm32, so the WASM
//! numbers come from the wasmtime probe in
//! `benchmarks/mul_add_wasm_wasmtime_zen5-9950x3d_2026-10-05.md`.
//!
//! Two shapes per vector type, each inside one `#[arcane]` region:
//!
//! - **stream**: `out[i] = a[i].mul_add(b[i], c[i])` over 1,024 L1-resident
//!   vectors, where throughput, loads and stores decide.
//! - **chain**: `acc = acc.mul_add(x, c)` 1,024 times in one dependency chain,
//!   where the latency of one fused operation against a multiply followed by an
//!   add decides.
//!
//! Before timing, every fused kernel must match `f32::mul_add`/`f64::mul_add`
//! lane for lane and every unfused kernel must match `a * b + c`.
//!
//! Run (no `-Ctarget-cpu=native`; bench what users get):
//!   cargo bench -p magetypes --bench mul_add_cost --features avx512   # x86_64
//!   cargo bench -p magetypes --bench mul_add_cost                     # aarch64
//! Redraw code layout once (Zen 5 placement effects) and compare:
//!   RUSTFLAGS="--cfg mul_add_cost_redraw" cargo bench -p magetypes --bench mul_add_cost

/// Vectors per stream pass and steps per chain.
const N: usize = 1024;

/// Shifts the layout of everything after it when built with
/// `--cfg mul_add_cost_redraw`, so a second run draws different code placement.
#[cfg(mul_add_cost_redraw)]
#[inline(never)]
fn layout_pad(x: u64) -> u64 {
    let mut h = x;
    for i in 0..64u64 {
        h = h.rotate_left(7) ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    h
}

/// Defines `stream_fused`, `stream_unfused`, `chain_fused` and `chain_unfused`
/// for one vector type on one token.
macro_rules! kernels {
    ($module:ident, $vec:ident, $token:ident, $elem:ident, $lanes:literal) => {
        pub mod $module {
            use archmage::{arcane, $token};
            use magetypes::simd::generic::$vec;

            pub type Lanes = [$elem; $lanes];

            #[arcane]
            pub fn stream_fused(
                t: $token,
                a: &[Lanes],
                b: &[Lanes],
                c: &[Lanes],
                out: &mut [Lanes],
            ) {
                for (((a, b), c), o) in a.iter().zip(b).zip(c).zip(out.iter_mut()) {
                    let x = $vec::<$token>::load_t(t, a);
                    let y = $vec::<$token>::load_t(t, b);
                    *o = x.mul_add(y, $vec::<$token>::load_t(t, c)).to_array();
                }
            }

            #[arcane]
            pub fn stream_unfused(
                t: $token,
                a: &[Lanes],
                b: &[Lanes],
                c: &[Lanes],
                out: &mut [Lanes],
            ) {
                for (((a, b), c), o) in a.iter().zip(b).zip(c).zip(out.iter_mut()) {
                    let x = $vec::<$token>::load_t(t, a);
                    let y = $vec::<$token>::load_t(t, b);
                    *o = (x * y + $vec::<$token>::load_t(t, c)).to_array();
                }
            }

            #[arcane]
            pub fn chain_fused(t: $token, x: &Lanes, c: &Lanes) -> Lanes {
                let x = $vec::<$token>::load_t(t, x);
                let c = $vec::<$token>::load_t(t, c);
                let mut acc = c;
                for _ in 0..super::super::N {
                    acc = acc.mul_add(x, c);
                }
                acc.to_array()
            }

            #[arcane]
            pub fn chain_unfused(t: $token, x: &Lanes, c: &Lanes) -> Lanes {
                let x = $vec::<$token>::load_t(t, x);
                let c = $vec::<$token>::load_t(t, c);
                let mut acc = c;
                for _ in 0..super::super::N {
                    acc = acc * x + c;
                }
                acc.to_array()
            }
        }
    };
}

mod kernels {
    #[cfg(target_arch = "x86_64")]
    kernels!(v3_f32x8, f32x8, X64V3Token, f32, 8);
    #[cfg(target_arch = "x86_64")]
    kernels!(v3_f64x4, f64x4, X64V3Token, f64, 4);
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    kernels!(v4_f32x16, f32x16, X64V4Token, f32, 16);
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    kernels!(v4_f64x8, f64x8, X64V4Token, f64, 8);
    #[cfg(target_arch = "x86_64")]
    kernels!(scalar_f32x8, f32x8, ScalarToken, f32, 8);
    #[cfg(target_arch = "x86_64")]
    kernels!(scalar_f64x4, f64x4, ScalarToken, f64, 4);

    #[cfg(target_arch = "aarch64")]
    kernels!(neon_f32x4, f32x4, NeonToken, f32, 4);
    #[cfg(target_arch = "aarch64")]
    kernels!(neon_f64x2, f64x2, NeonToken, f64, 2);
    #[cfg(target_arch = "aarch64")]
    kernels!(scalar_f32x4, f32x4, ScalarToken, f32, 4);
    #[cfg(target_arch = "aarch64")]
    kernels!(scalar_f64x2, f64x2, ScalarToken, f64, 2);
}

/// Checks one kernel set against std, then registers its stream and chain groups.
macro_rules! groups {
    ($suite:expr, $label:expr, $module:ident, $token:expr, $elem:ident, $lanes:literal) => {{
        use kernels::$module as k;
        use zenbench::prelude::*;
        let t = $token;

        let lanes = |base: f64, step: f64, i: usize| -> k::Lanes {
            core::array::from_fn(|j| (base + ((i * $lanes + j) % 97) as f64 * step) as $elem)
        };
        let a: Vec<k::Lanes> = (0..N).map(|i| lanes(1.0, 0.01, i)).collect();
        let b: Vec<k::Lanes> = (0..N).map(|i| lanes(0.5, 0.0103, i)).collect();
        let c: Vec<k::Lanes> = (0..N).map(|i| lanes(-0.25, 0.0051, i)).collect();
        let x: k::Lanes = core::array::from_fn(|j| (0.999 - j as f64 * 1e-4) as $elem);
        let addend: k::Lanes = core::array::from_fn(|j| (1e-3 + j as f64 * 1e-5) as $elem);

        // The fused kernels must round once and the unfused ones twice.
        let mut out = vec![[0 as $elem; $lanes]; N];
        k::stream_fused(t, &a, &b, &c, &mut out);
        for i in 0..N {
            for j in 0..$lanes {
                assert_eq!(
                    out[i][j].to_bits(),
                    a[i][j].mul_add(b[i][j], c[i][j]).to_bits(),
                    "{} stream fused",
                    $label
                );
            }
        }
        k::stream_unfused(t, &a, &b, &c, &mut out);
        for i in 0..N {
            for j in 0..$lanes {
                assert_eq!(
                    out[i][j].to_bits(),
                    (a[i][j] * b[i][j] + c[i][j]).to_bits(),
                    "{} stream unfused",
                    $label
                );
            }
        }
        let (fused, unfused) = (
            k::chain_fused(t, &x, &addend),
            k::chain_unfused(t, &x, &addend),
        );
        for j in 0..$lanes {
            let (mut f, mut u) = (addend[j], addend[j]);
            for _ in 0..N {
                f = f.mul_add(x[j], addend[j]);
                u = u * x[j] + addend[j];
            }
            assert_eq!(fused[j].to_bits(), f.to_bits(), "{} chain fused", $label);
            assert_eq!(
                unfused[j].to_bits(),
                u.to_bits(),
                "{} chain unfused",
                $label
            );
        }

        $suite.group(format!("{} stream", $label), |g| {
            g.throughput(Throughput::Elements((N * $lanes) as u64));
            for (name, kernel) in [
                (
                    "a * b + c",
                    k::stream_unfused
                        as fn(_, &[k::Lanes], &[k::Lanes], &[k::Lanes], &mut [k::Lanes]),
                ),
                ("mul_add", k::stream_fused),
            ] {
                let (a, b, c) = (a.clone(), b.clone(), c.clone());
                let mut out = vec![[0 as $elem; $lanes]; N];
                g.bench(name, move |bench| {
                    bench.iter(|| {
                        kernel(t, black_box(&a), black_box(&b), black_box(&c), &mut out);
                        black_box(out[0])
                    })
                });
            }
            g.baseline("a * b + c");
        });
        $suite.group(format!("{} chain", $label), |g| {
            g.throughput(Throughput::Elements(N as u64));
            for (name, kernel) in [
                (
                    "a * b + c",
                    k::chain_unfused as fn(_, &k::Lanes, &k::Lanes) -> k::Lanes,
                ),
                ("mul_add", k::chain_fused),
            ] {
                g.bench(name, move |bench| {
                    bench.iter(|| kernel(t, black_box(&x), black_box(&addend)))
                });
            }
            g.baseline("a * b + c");
        });
    }};
}

fn bench_mul_add(suite: &mut zenbench::prelude::Suite) {
    #[cfg(mul_add_cost_redraw)]
    std::hint::black_box(layout_pad(std::hint::black_box(N as u64)));

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::{ScalarToken, SimdToken, X64V3Token};
        match X64V3Token::summon() {
            Some(v3) => {
                groups!(suite, "f32x8, AVX2 (V3)", v3_f32x8, v3, f32, 8);
                groups!(suite, "f64x4, AVX2 (V3)", v3_f64x4, v3, f64, 4);
            }
            None => eprintln!("skipped AVX2 groups: this CPU lacks x86-64-v3"),
        }
        #[cfg(feature = "avx512")]
        match archmage::X64V4Token::summon() {
            Some(v4) => {
                groups!(suite, "f32x16, AVX-512 (V4)", v4_f32x16, v4, f32, 16);
                groups!(suite, "f64x8, AVX-512 (V4)", v4_f64x8, v4, f64, 8);
            }
            None => eprintln!("skipped AVX-512 groups: this CPU lacks x86-64-v4"),
        }
        groups!(
            suite,
            "f32x8, scalar backend",
            scalar_f32x8,
            ScalarToken,
            f32,
            8
        );
        groups!(
            suite,
            "f64x4, scalar backend",
            scalar_f64x4,
            ScalarToken,
            f64,
            4
        );
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::{NeonToken, ScalarToken, SimdToken};
        match NeonToken::summon() {
            Some(neon) => {
                groups!(suite, "f32x4, NEON", neon_f32x4, neon, f32, 4);
                groups!(suite, "f64x2, NEON", neon_f64x2, neon, f64, 2);
            }
            None => eprintln!("skipped NEON groups: NEON not detected"),
        }
        groups!(
            suite,
            "f32x4, scalar backend",
            scalar_f32x4,
            ScalarToken,
            f32,
            4
        );
        groups!(
            suite,
            "f64x2, scalar backend",
            scalar_f64x2,
            ScalarToken,
            f64,
            2
        );
    }
}

zenbench::main!(bench_mul_add);

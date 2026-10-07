//! `mul_add` and `mul_add_portable` against `a * b + c`, per backend.
//!
//! `mul_add` fuses where the hardware does (x86 v3/v4, NEON) and is a multiply
//! then an add on the scalar backend and strict WASM. `mul_add_portable` rounds
//! once everywhere: hardware FMA where it exists, software fusion elsewhere.
//! zenbench does not build for wasm32, so WASM numbers come from the wasmtime
//! probes in `benchmarks/`.
//!
//! Two shapes per vector type, each inside one `#[arcane]` region:
//!
//! - **stream**: `out[i] = a[i].mul_add(b[i], c[i])` over 1,024 L1-resident
//!   vectors, where throughput, loads and stores decide.
//! - **chain**: `acc = acc.mul_add(x, c)` 1,024 times in one dependency chain,
//!   where the latency of one fused operation against a multiply followed by an
//!   add decides.
//!
//! Before timing, every kernel must match its rounding contract lane for lane:
//! `a * b + c` two roundings, `mul_add_portable` one (`f32::mul_add`), and
//! `mul_add` one on FMA backends and two on the scalar backend.
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

/// Defines a stream and a chain kernel for one multiply-add form.
macro_rules! form {
    ($stream:ident, $chain:ident, $vec:ident, $token:ident, |$x:ident, $y:ident, $z:ident| $op:expr) => {
        #[arcane]
        pub fn $stream(t: $token, a: &[Lanes], b: &[Lanes], c: &[Lanes], out: &mut [Lanes]) {
            for (((a, b), c), o) in a.iter().zip(b).zip(c).zip(out.iter_mut()) {
                let $x = $vec::<$token>::load_t(t, a);
                let $y = $vec::<$token>::load_t(t, b);
                let $z = $vec::<$token>::load_t(t, c);
                *o = ($op).to_array();
            }
        }

        #[arcane]
        pub fn $chain(t: $token, x: &Lanes, c: &Lanes) -> Lanes {
            let $y = $vec::<$token>::load_t(t, x);
            let $z = $vec::<$token>::load_t(t, c);
            let mut $x = $z;
            for _ in 0..super::super::N {
                $x = $op;
            }
            $x.to_array()
        }
    };
}

/// Defines the three forms for one vector type on one token.
macro_rules! kernels {
    ($module:ident, $vec:ident, $token:ident, $elem:ident, $lanes:literal) => {
        pub mod $module {
            use archmage::{arcane, $token};
            use magetypes::simd::generic::$vec;

            pub type Lanes = [$elem; $lanes];

            form!(stream_unfused, chain_unfused, $vec, $token, |x, y, z| x * y
                + z);
            form!(stream_fast, chain_fast, $vec, $token, |x, y, z| x
                .mul_add(y, z));
            form!(stream_portable, chain_portable, $vec, $token, |x, y, z| x
                .mul_add_portable(y, z));
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

/// A stream kernel: token, the three inputs, the output.
type Stream<T, L> = fn(T, &[L], &[L], &[L], &mut [L]);
/// A chain kernel: token, multiplier, addend; returns the final accumulator.
type Chain<T, L> = fn(T, &L, &L) -> L;

/// Coerces a stream kernel to a function pointer so the three forms share a type.
fn as_stream<T, L>(f: Stream<T, L>) -> Stream<T, L> {
    f
}

/// Coerces a chain kernel to a function pointer so the three forms share a type.
fn as_chain<T, L>(f: Chain<T, L>) -> Chain<T, L> {
    f
}

/// Checks one kernel set against its contracts, then registers its groups.
/// `$fma` says whether `mul_add` fuses on this token.
macro_rules! groups {
    ($suite:expr, $label:expr, $module:ident, $token:expr, $fma:expr, $elem:ident, $lanes:literal) => {{
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

        let fused = |p: $elem, q: $elem, r: $elem| p.mul_add(q, r);
        let unfused = |p: $elem, q: $elem, r: $elem| p * q + r;
        let forms = [
            (
                "a * b + c",
                as_stream(k::stream_unfused),
                as_chain(k::chain_unfused),
                false,
            ),
            (
                "mul_add",
                as_stream(k::stream_fast),
                as_chain(k::chain_fast),
                $fma,
            ),
            (
                "mul_add_portable",
                as_stream(k::stream_portable),
                as_chain(k::chain_portable),
                true,
            ),
        ];
        for (name, stream, chain, one_rounding) in forms {
            let want: &dyn Fn($elem, $elem, $elem) -> $elem =
                if one_rounding { &fused } else { &unfused };
            let mut out = vec![[0 as $elem; $lanes]; N];
            stream(t, &a, &b, &c, &mut out);
            for i in 0..N {
                for j in 0..$lanes {
                    assert_eq!(
                        out[i][j].to_bits(),
                        want(a[i][j], b[i][j], c[i][j]).to_bits(),
                        "{} stream {name}",
                        $label
                    );
                }
            }
            let got = chain(t, &x, &addend);
            for j in 0..$lanes {
                let mut acc = addend[j];
                for _ in 0..N {
                    acc = want(acc, x[j], addend[j]);
                }
                assert_eq!(got[j].to_bits(), acc.to_bits(), "{} chain {name}", $label);
            }
        }

        $suite.group(format!("{} stream", $label), |g| {
            g.throughput(Throughput::Elements((N * $lanes) as u64));
            for (name, stream, _, _) in forms {
                let (a, b, c) = (a.clone(), b.clone(), c.clone());
                let mut out = vec![[0 as $elem; $lanes]; N];
                g.bench(name, move |bench| {
                    bench.iter(|| {
                        stream(t, black_box(&a), black_box(&b), black_box(&c), &mut out);
                        black_box(out[0])
                    })
                });
            }
            g.baseline("a * b + c");
        });
        $suite.group(format!("{} chain", $label), |g| {
            g.throughput(Throughput::Elements(N as u64));
            for (name, _, chain, _) in forms {
                g.bench(name, move |bench| {
                    bench.iter(|| chain(t, black_box(&x), black_box(&addend)))
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
                groups!(suite, "f32x8, AVX2 (V3)", v3_f32x8, v3, true, f32, 8);
                groups!(suite, "f64x4, AVX2 (V3)", v3_f64x4, v3, true, f64, 4);
            }
            None => eprintln!("skipped AVX2 groups: this CPU lacks x86-64-v3"),
        }
        #[cfg(feature = "avx512")]
        match archmage::X64V4Token::summon() {
            Some(v4) => {
                groups!(suite, "f32x16, AVX-512 (V4)", v4_f32x16, v4, true, f32, 16);
                groups!(suite, "f64x8, AVX-512 (V4)", v4_f64x8, v4, true, f64, 8);
            }
            None => eprintln!("skipped AVX-512 groups: this CPU lacks x86-64-v4"),
        }
        groups!(
            suite,
            "f32x8, scalar backend",
            scalar_f32x8,
            ScalarToken,
            false,
            f32,
            8
        );
        groups!(
            suite,
            "f64x4, scalar backend",
            scalar_f64x4,
            ScalarToken,
            false,
            f64,
            4
        );
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::{NeonToken, ScalarToken, SimdToken};
        match NeonToken::summon() {
            Some(neon) => {
                groups!(suite, "f32x4, NEON", neon_f32x4, neon, true, f32, 4);
                groups!(suite, "f64x2, NEON", neon_f64x2, neon, true, f64, 2);
            }
            None => eprintln!("skipped NEON groups: NEON not detected"),
        }
        groups!(
            suite,
            "f32x4, scalar backend",
            scalar_f32x4,
            ScalarToken,
            false,
            f32,
            4
        );
        groups!(
            suite,
            "f64x2, scalar backend",
            scalar_f64x2,
            ScalarToken,
            false,
            f64,
            2
        );
    }
}

zenbench::main!(bench_mul_add);

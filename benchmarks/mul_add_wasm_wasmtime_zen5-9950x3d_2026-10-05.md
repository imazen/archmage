# `mul_add` on WASM SIMD128: fused vs unfused (wasmtime, Zen 5, 2026-10-05)

0.9.30 makes `mul_add`/`mul_sub` round once on the scalar and strict-WASM backends
(#116; 11a35a8b). In 0.9.29 the WASM backend computed `self * a + b` with two
roundings, as its docs said, so the `a*b+c` rows below are the 0.9.29 cost.

Host: AMD Ryzen 9 9950X3D. rustc 1.99.0 (b940084d7 2026-09-28), wasm32-wasip1,
release with `codegen-units = 1`. Engine: wasmtime 40.0.1 (Cranelift), default
settings. archmage and magetypes from main at 1a6250a3.

## Method

The probe below streams 1,024 L1-resident vectors through each form inside one
`#[arcane]` region, 16 passes per sample. 200 rounds run the four variants in a
rotating order, and the table gives the median per variant. Two builds, each run
twice: `-C target-feature=+simd128` (strict) and `+simd128,+relaxed-simd`.

## Results

ns per pass over 1,024 vectors (two runs).

| Build | Variant | Median | Ratio to `a*b+c` |
|---|---|---:|---:|
| strict | `f32x4::mul_add` | 5,627 / 5,650 | 8.7× |
| strict | `f32x4` `a*b+c` | 649 / 647 | — |
| strict | `f64x2::mul_add` | 16,798 / 16,875 | 24.5× / 24.8× |
| strict | `f64x2` `a*b+c` | 685 / 681 | — |
| relaxed | `f32x4::mul_add` | 614 / 615 | 0.97× / 0.98× |
| relaxed | `f32x4` `a*b+c` | 631 / 631 | — |
| relaxed | `f64x2::mul_add` | 612 / 616 | 0.97× |
| relaxed | `f64x2` `a*b+c` | 631 / 636 | — |

## Reading it

On strict SIMD128, `f32x4::mul_add` widens each half to f64 and runs TwoSum with
round-to-odd (`magetypes/src/wasm_fma.rs`); `f64x2::mul_add` extracts both lanes
and calls `libm::fma`. A kernel built on `mul_add` pays that per call. Relaxed
SIMD builds emit the native relaxed madd and keep their speed, and its rounding
follows the engine.

Only wasmtime on one x86 host was measured. V8, SpiderMonkey and JavaScriptCore
compile WASM differently, and this loop does no other work, so a real kernel's
slowdown depends on how much of it is `mul_add`.

## Probe source

`Cargo.toml` (path dependencies on this repository):

```toml
[package]
name = "wasm-fma-probe"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
archmage = { path = "/home/lilith/work/archmage" }
magetypes = { path = "/home/lilith/work/archmage/magetypes" }

[profile.release]
codegen-units = 1

[workspace]
```

`src/main.rs`:

```rust
//! Times magetypes `mul_add` against `a * b + c` (the 0.9.29 lowering on WASM)
//! for f32x4 and f64x2 over 1,024 L1-resident vectors, interleaved over 200 rounds.
use archmage::{SimdToken, Wasm128Token, arcane};
use magetypes::simd::generic::{f32x4, f64x2};
use std::hint::black_box;
use std::time::Instant;

const N: usize = 1024;

#[arcane]
fn f32_fused(t: Wasm128Token, a: &[[f32; 4]], b: &[[f32; 4]], c: &[[f32; 4]], out: &mut [[f32; 4]]) {
    for i in 0..a.len() {
        let x = f32x4::load_t(t, &a[i]);
        out[i] = x.mul_add(f32x4::load_t(t, &b[i]), f32x4::load_t(t, &c[i])).to_array();
    }
}
#[arcane]
fn f32_unfused(t: Wasm128Token, a: &[[f32; 4]], b: &[[f32; 4]], c: &[[f32; 4]], out: &mut [[f32; 4]]) {
    for i in 0..a.len() {
        let x = f32x4::load_t(t, &a[i]);
        out[i] = (x * f32x4::load_t(t, &b[i]) + f32x4::load_t(t, &c[i])).to_array();
    }
}
#[arcane]
fn f64_fused(t: Wasm128Token, a: &[[f64; 2]], b: &[[f64; 2]], c: &[[f64; 2]], out: &mut [[f64; 2]]) {
    for i in 0..a.len() {
        let x = f64x2::load_t(t, &a[i]);
        out[i] = x.mul_add(f64x2::load_t(t, &b[i]), f64x2::load_t(t, &c[i])).to_array();
    }
}
#[arcane]
fn f64_unfused(t: Wasm128Token, a: &[[f64; 2]], b: &[[f64; 2]], c: &[[f64; 2]], out: &mut [[f64; 2]]) {
    for i in 0..a.len() {
        let x = f64x2::load_t(t, &a[i]);
        out[i] = (x * f64x2::load_t(t, &b[i]) + f64x2::load_t(t, &c[i])).to_array();
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let t = Wasm128Token::summon().expect("simd128");
    let a4: Vec<[f32; 4]> = (0..N).map(|i| core::array::from_fn(|j| 1.0 + (i * 4 + j) as f32 * 1e-3)).collect();
    let b4: Vec<[f32; 4]> = (0..N).map(|i| core::array::from_fn(|j| 0.5 + (i + j) as f32 * 1e-4)).collect();
    let c4: Vec<[f32; 4]> = (0..N).map(|i| core::array::from_fn(|j| -0.25 + (i ^ j) as f32 * 1e-5)).collect();
    let a2: Vec<[f64; 2]> = (0..N).map(|i| core::array::from_fn(|j| 1.0 + (i * 2 + j) as f64 * 1e-3)).collect();
    let b2: Vec<[f64; 2]> = (0..N).map(|i| core::array::from_fn(|j| 0.5 + (i + j) as f64 * 1e-4)).collect();
    let c2: Vec<[f64; 2]> = (0..N).map(|i| core::array::from_fn(|j| -0.25 + (i ^ j) as f64 * 1e-5)).collect();
    let mut o4 = vec![[0f32; 4]; N];
    let mut o2 = vec![[0f64; 2]; N];
    let names = ["f32x4 mul_add", "f32x4 a*b+c", "f64x2 mul_add", "f64x2 a*b+c"];
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); 4];
    for round in 0..200 {
        for k in 0..4 {
            let v = (k + round) % 4;
            let start = Instant::now();
            for _ in 0..16 {
                match v {
                    0 => f32_fused(t, black_box(&a4), black_box(&b4), black_box(&c4), black_box(&mut o4)),
                    1 => f32_unfused(t, black_box(&a4), black_box(&b4), black_box(&c4), black_box(&mut o4)),
                    2 => f64_fused(t, black_box(&a2), black_box(&b2), black_box(&c2), black_box(&mut o2)),
                    _ => f64_unfused(t, black_box(&a2), black_box(&b2), black_box(&c2), black_box(&mut o2)),
                }
            }
            times[v].push(start.elapsed().as_nanos() as f64 / 16.0);
        }
    }
    black_box((&o4, &o2));
    let relaxed = cfg!(target_feature = "relaxed-simd");
    println!("build: {}", if relaxed { "simd128 + relaxed-simd" } else { "simd128 (strict)" });
    let med: Vec<f64> = times.into_iter().map(median).collect();
    for (i, n) in names.iter().enumerate() {
        println!("{n:14} median {:8.0} ns per 1,024 vectors", med[i]);
    }
    println!("f32x4 mul_add / (a*b+c) = {:.2}x", med[0] / med[1]);
    println!("f64x2 mul_add / (a*b+c) = {:.2}x", med[2] / med[3]);
}
```

Commands, from the probe directory:

```sh
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-wasip1
wasmtime run target/wasm32-wasip1/release/wasm-fma-probe.wasm
RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" cargo build --release --target wasm32-wasip1
wasmtime run target/wasm32-wasip1/release/wasm-fma-probe.wasm
```

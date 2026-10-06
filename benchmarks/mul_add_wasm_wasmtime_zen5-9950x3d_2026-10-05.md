# `mul_add` and `mul_add_portable` on WASM SIMD128 (wasmtime, Zen 5, 2026-10-05)

Two rounds of measurement on the same day. The first timed the interim design
(#116; 11a35a8b), in which `mul_add` itself fused in software on strict WASM.
Before release, `mul_add` returned to the 0.9.29 behavior and the software path
moved to `mul_add_portable` (66986145); the second round times that.

Host: AMD Ryzen 9 9950X3D. rustc 1.99.0 (b940084d7 2026-09-28), wasm32-wasip1,
release with `codegen-units = 1`. Engine: wasmtime 40.0.1 (Cranelift), default
settings.

## Current forms (main at 2f856d2c)

The probe below streams 1,024 L1-resident vectors through `a * b + c`, `mul_add`
and `mul_add_portable` for `f32x4` and `f64x2`, inside one `#[arcane]` region,
16 passes per sample. 200 rounds run the six variants in a rotating order, and
the table gives the median per variant. Before timing, the probe checks each form
against its rounding contract bit for bit: `a * b + c` and strict `mul_add` round
twice, `mul_add_portable` once (std `mul_add`). It reports, without asserting,
how relaxed `mul_add` rounds. Two builds, each run twice:
`-C target-feature=+simd128` (strict) and `+simd128,+relaxed-simd`.

ns per pass over 1,024 vectors (two runs), and the ratio to `a * b + c` from the
same run.

| Build | Variant | Median | Ratio to `a*b+c` |
|---|---|---:|---:|
| strict | `f32x4` `a*b+c` | 891 / 688 | — |
| strict | `f32x4::mul_add` | 891 / 688 | 1.00× |
| strict | `f32x4::mul_add_portable` | 7,272 / 5,636 | 8.17× / 8.19× |
| strict | `f64x2` `a*b+c` | 947 / 738 | — |
| strict | `f64x2::mul_add` | 947 / 741 | 1.00× |
| strict | `f64x2::mul_add_portable` | 21,540 / 16,819 | 22.8× |
| relaxed | `f32x4` `a*b+c` | 693 / 696 | — |
| relaxed | `f32x4::mul_add` | 735 / 737 | 1.06× |
| relaxed | `f32x4::mul_add_portable` | 5,674 / 5,683 | 8.19× / 8.17× |
| relaxed | `f64x2` `a*b+c` | 694 / 703 | — |
| relaxed | `f64x2::mul_add` | 668 / 740 | 0.96× / 1.05× |
| relaxed | `f64x2::mul_add_portable` | 16,938 / 16,960 | 24.4× / 24.1× |

The first strict run was slower across the board; its ratios match the second's.
On this host wasmtime fused every relaxed `mul_add` lane (4,096 of 4,096 `f32x4`
and 2,048 of 2,048 `f64x2` results matched one rounding).

Strict `mul_add` is the same code as `a * b + c`, so it costs nothing. Relaxed
`mul_add` emits the engine's madd: 4% faster to 6% slower than `a * b + c` here,
where the first round measured it 2–3% faster with a smaller probe; both are
within code-placement noise. `mul_add_portable` fuses in software in both builds,
because relaxed madd may round twice: 8.2× (`f32x4`) and 23–24× (`f64x2`) the
time of `a * b + c`.

### Probe source (current forms)

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
//! Times magetypes `mul_add` and `mul_add_portable` against `a * b + c` for
//! f32x4 and f64x2 over 1,024 L1-resident vectors, interleaved over 200 rounds.
//! Before timing, checks each form against its rounding contract: `a * b + c`
//! and strict `mul_add` round twice, `mul_add_portable` once (std `mul_add`).
use archmage::{SimdToken, Wasm128Token, arcane};
use magetypes::simd::generic::{f32x4, f64x2};
use std::hint::black_box;
use std::time::Instant;

const N: usize = 1024;
type A4 = [[f32; 4]];
type A2 = [[f64; 2]];

macro_rules! kernel {
    ($name:ident, $vec:ident, $arr:ident, |$x:ident, $y:ident, $z:ident| $op:expr) => {
        #[arcane]
        fn $name(t: Wasm128Token, a: &$arr, b: &$arr, c: &$arr, out: &mut $arr) {
            for i in 0..a.len() {
                let $x = $vec::load_t(t, &a[i]);
                let $y = $vec::load_t(t, &b[i]);
                let $z = $vec::load_t(t, &c[i]);
                out[i] = ($op).to_array();
            }
        }
    };
}
kernel!(f32_unfused, f32x4, A4, |x, y, z| x * y + z);
kernel!(f32_fast, f32x4, A4, |x, y, z| x.mul_add(y, z));
kernel!(f32_portable, f32x4, A4, |x, y, z| x.mul_add_portable(y, z));
kernel!(f64_unfused, f64x2, A2, |x, y, z| x * y + z);
kernel!(f64_fast, f64x2, A2, |x, y, z| x.mul_add(y, z));
kernel!(f64_portable, f64x2, A2, |x, y, z| x.mul_add_portable(y, z));

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let t = Wasm128Token::summon().expect("simd128");
    let relaxed = cfg!(target_feature = "relaxed-simd");
    let a4: Vec<[f32; 4]> = (0..N).map(|i| core::array::from_fn(|j| 1.0 + (i * 4 + j) as f32 * 1e-3)).collect();
    let b4: Vec<[f32; 4]> = (0..N).map(|i| core::array::from_fn(|j| 0.5 + (i + j) as f32 * 1e-4)).collect();
    let c4: Vec<[f32; 4]> = (0..N).map(|i| core::array::from_fn(|j| -0.25 + (i ^ j) as f32 * 1e-5)).collect();
    let a2: Vec<[f64; 2]> = (0..N).map(|i| core::array::from_fn(|j| 1.0 + (i * 2 + j) as f64 * 1e-3)).collect();
    let b2: Vec<[f64; 2]> = (0..N).map(|i| core::array::from_fn(|j| 0.5 + (i + j) as f64 * 1e-4)).collect();
    let c2: Vec<[f64; 2]> = (0..N).map(|i| core::array::from_fn(|j| -0.25 + (i ^ j) as f64 * 1e-5)).collect();
    let mut o4 = vec![[0f32; 4]; N];
    let mut o2 = vec![[0f64; 2]; N];

    // Contract checks. Relaxed `mul_add` may round either way, so it is only
    // reported, not asserted.
    let f32s: [(&str, fn(Wasm128Token, &A4, &A4, &A4, &mut A4)); 3] =
        [("f32x4 a*b+c", f32_unfused), ("f32x4 mul_add", f32_fast), ("f32x4 mul_add_portable", f32_portable)];
    for (name, f) in f32s {
        f(t, &a4, &b4, &c4, &mut o4);
        let (mut fused, mut unfused) = (0usize, 0usize);
        for i in 0..N {
            for j in 0..4 {
                let (p, q, r) = (a4[i][j], b4[i][j], c4[i][j]);
                fused += (o4[i][j].to_bits() == p.mul_add(q, r).to_bits()) as usize;
                unfused += (o4[i][j].to_bits() == (p * q + r).to_bits()) as usize;
            }
        }
        println!("{name:24} matches fused {fused}/{} unfused {unfused}/{}", N * 4, N * 4);
        let want_fused = name.ends_with("portable");
        let want_unfused = name.ends_with("a*b+c") || (name.ends_with("mul_add") && !relaxed);
        assert!(!want_fused || fused == N * 4, "{name} must round once");
        assert!(!want_unfused || unfused == N * 4, "{name} must round twice");
    }
    let f64s: [(&str, fn(Wasm128Token, &A2, &A2, &A2, &mut A2)); 3] =
        [("f64x2 a*b+c", f64_unfused), ("f64x2 mul_add", f64_fast), ("f64x2 mul_add_portable", f64_portable)];
    for (name, f) in f64s {
        f(t, &a2, &b2, &c2, &mut o2);
        let (mut fused, mut unfused) = (0usize, 0usize);
        for i in 0..N {
            for j in 0..2 {
                let (p, q, r) = (a2[i][j], b2[i][j], c2[i][j]);
                fused += (o2[i][j].to_bits() == p.mul_add(q, r).to_bits()) as usize;
                unfused += (o2[i][j].to_bits() == (p * q + r).to_bits()) as usize;
            }
        }
        println!("{name:24} matches fused {fused}/{} unfused {unfused}/{}", N * 2, N * 2);
        let want_fused = name.ends_with("portable");
        let want_unfused = name.ends_with("a*b+c") || (name.ends_with("mul_add") && !relaxed);
        assert!(!want_fused || fused == N * 2, "{name} must round once");
        assert!(!want_unfused || unfused == N * 2, "{name} must round twice");
    }

    let names = [
        "f32x4 a*b+c", "f32x4 mul_add", "f32x4 mul_add_portable",
        "f64x2 a*b+c", "f64x2 mul_add", "f64x2 mul_add_portable",
    ];
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); 6];
    for round in 0..200 {
        for k in 0..6 {
            let v = (k + round) % 6;
            let start = Instant::now();
            for _ in 0..16 {
                match v {
                    0 => f32_unfused(t, black_box(&a4), black_box(&b4), black_box(&c4), black_box(&mut o4)),
                    1 => f32_fast(t, black_box(&a4), black_box(&b4), black_box(&c4), black_box(&mut o4)),
                    2 => f32_portable(t, black_box(&a4), black_box(&b4), black_box(&c4), black_box(&mut o4)),
                    3 => f64_unfused(t, black_box(&a2), black_box(&b2), black_box(&c2), black_box(&mut o2)),
                    4 => f64_fast(t, black_box(&a2), black_box(&b2), black_box(&c2), black_box(&mut o2)),
                    _ => f64_portable(t, black_box(&a2), black_box(&b2), black_box(&c2), black_box(&mut o2)),
                }
            }
            times[v].push(start.elapsed().as_nanos() as f64 / 16.0);
        }
    }
    black_box((&o4, &o2));
    println!("build: {}", if relaxed { "simd128 + relaxed-simd" } else { "simd128 (strict)" });
    let med: Vec<f64> = times.into_iter().map(median).collect();
    for (i, n) in names.iter().enumerate() {
        let base = if i < 3 { med[0] } else { med[3] };
        println!("{n:24} median {:8.0} ns per 1,024 vectors  {:5.2}x a*b+c", med[i], med[i] / base);
    }
}
```

Commands, from the probe directory:

```sh
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-wasip1
wasmtime run target/wasm32-wasip1/release/wasm-fma-probe.wasm
RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" cargo build --release --target wasm32-wasip1
wasmtime run target/wasm32-wasip1/release/wasm-fma-probe.wasm
```

## Interim design (main at 1a6250a3)

Here `mul_add` was the software-fused form, so its rows are what
`mul_add_portable` costs now, and the `a*b+c` rows are the 0.9.29 cost of
`mul_add`.

### Method

The probe below streams 1,024 L1-resident vectors through each form inside one
`#[arcane]` region, 16 passes per sample. 200 rounds run the four variants in a
rotating order, and the table gives the median per variant. Two builds, each run
twice: `-C target-feature=+simd128` (strict) and `+simd128,+relaxed-simd`.

### Results

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

### Reading it

On strict SIMD128, `f32x4::mul_add` widens each half to f64 and runs TwoSum with
round-to-odd (`magetypes/src/wasm_fma.rs`); `f64x2::mul_add` extracts both lanes
and calls `libm::fma`. A kernel built on `mul_add` pays that per call. Relaxed
SIMD builds emit the native relaxed madd and keep their speed, and its rounding
follows the engine.

Only wasmtime on one x86 host was measured. V8, SpiderMonkey and JavaScriptCore
compile WASM differently, and this loop does no other work, so a real kernel's
slowdown depends on how much of it is `mul_add`.

### Probe source (interim)

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

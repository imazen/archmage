# Generic transcendentals on the scalar backend and strict WASM: 0.9.29 vs main (2026-10-05)

The generic transcendentals (`exp2_midp`, `log2_midp`, `pow_midp` and the rest)
evaluate their polynomials with `mul_add`: 17 call sites in
`xtask/src/simd_types/generic_gen/transcendentals.rs`, unchanged since 0.9.29.
In 0.9.29 the scalar and WASM backends computed `mul_add` as a multiply and an
add. On main they fuse in software (#116; 11a35a8b), so every polynomial step
pays that cost. x86 v3/v4 and NEON used hardware FMA in 0.9.29 already, so their
transcendentals did not change.

## Method

The probe below was built twice, against `git archive v0.9.29` and against main
at 93ca6a3, using only API both versions have. It times `f32x4<Wasm128Token>` on
wasm32 and `f32x4<ScalarToken>` on native targets: 1,024 L1-resident vectors,
16 passes per sample, 200 rounds rotating the four functions, median per
function. Each build ran twice, alternating 0.9.29 and main.

- Strict WASM: `-C target-feature=+simd128`, wasm32-wasip1, wasmtime 40.0.1, on
  an AMD Ryzen 9 9950X3D. rustc 1.99.0.
- x86 scalar backend: native x86_64 on the same machine, no `-Ctarget-cpu`.
- aarch64 scalar backend: native on an Apple M4 Pro, rustc 1.99.0.

## Results

ns per pass over 1,024 vectors, round 1 / round 2.

Strict WASM SIMD128 (wasmtime):

| Function | 0.9.29 | main | main / 0.9.29 |
|---|---:|---:|---:|
| `exp2_midp` | 6,666 / 5,342 | 77,439 / 77,914 | 11.6–14.6× |
| `log2_midp` | 2,444 / 1,879 | 42,940 / 43,212 | 17.6–23.0× |
| `exp_midp` | 7,084 / 5,445 | 77,743 / 78,206 | 11.0–14.4× |
| `pow_midp(2.4)` | 14,714 / 11,320 | 133,347 / 134,169 | 9.1–11.9× |

The 0.9.29 build's first round ran 25–30% slower than its second; main's did
not move.

x86 scalar backend (Ryzen 9 9950X3D):

| Function | 0.9.29 | main | main / 0.9.29 |
|---|---:|---:|---:|
| `exp2_midp` | 6,683 / 6,895 | 32,348 / 32,332 | 4.7–4.8× |
| `log2_midp` | 2,618 / 2,697 | 18,822 / 18,813 | 7.0–7.2× |
| `exp_midp` | 6,809 / 7,022 | 32,614 / 32,620 | 4.6–4.8× |
| `pow_midp(2.4)` | 9,700 / 9,997 | 55,130 / 55,102 | 5.5–5.7× |

aarch64 scalar backend (Apple M4 Pro):

| Function | 0.9.29 | main | main / 0.9.29 |
|---|---:|---:|---:|
| `exp2_midp` | 21,122 / 19,357 | 32,346 / 32,578 | 1.5–1.7× |
| `log2_midp` | 17,253 / 15,172 | 39,698 / 40,424 | 2.3–2.7× |
| `exp_midp` | 21,328 / 20,763 | 33,096 / 33,406 | 1.6× |
| `pow_midp(2.4)` | 38,948 / 35,742 | 109,219 / 108,432 | 2.8–3.0× |

## Reading it

Code that calls transcendentals on the scalar backend or a strict-WASM build got
1.5–23× slower between 0.9.29 and main, without calling `mul_add` itself.
0.9.29 shipped these functions with two roundings on scalar and WASM, and its
accuracy tests passed on every backend. Not measured: browser engines, other
functions and widths, and real kernels.

## Probe source

`Cargo.toml` (one copy per side; the path points at the archmage tree under test):

```toml
[package]
name = "transcendental-probe"
version = "0.0.0"
edition = "2024"
publish = false

[[bin]]
name = "transcendental-probe"
path = "../src/main.rs"

[dependencies]
archmage = { path = "<archmage tree>" }
magetypes = { path = "<archmage tree>/magetypes" }

[profile.release]
codegen-units = 1

[workspace]
```

`src/main.rs`:

```rust
//! Times magetypes f32x4 transcendentals on the WASM backend (wasm32) or the
//! scalar backend (native), 1,024 L1-resident vectors, 200 interleaved rounds.
//! Built once against 0.9.29 and once against main; uses only API both have.
#![allow(deprecated)]
use magetypes::simd::generic::f32x4;
use std::hint::black_box;
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
type Tok = archmage::Wasm128Token;
#[cfg(not(target_arch = "wasm32"))]
type Tok = archmage::ScalarToken;

const N: usize = 1024;
type V = [[f32; 4]];

#[inline(never)]
fn exp2(t: Tok, s: &V, o: &mut V) {
    for (s, o) in s.iter().zip(o.iter_mut()) {
        *o = f32x4::<Tok>::load(t, s).exp2_midp().to_array();
    }
}
#[inline(never)]
fn log2(t: Tok, s: &V, o: &mut V) {
    for (s, o) in s.iter().zip(o.iter_mut()) {
        *o = f32x4::<Tok>::load(t, s).log2_midp().to_array();
    }
}
#[inline(never)]
fn exp(t: Tok, s: &V, o: &mut V) {
    for (s, o) in s.iter().zip(o.iter_mut()) {
        *o = f32x4::<Tok>::load(t, s).exp_midp().to_array();
    }
}
#[inline(never)]
fn pow(t: Tok, s: &V, o: &mut V) {
    for (s, o) in s.iter().zip(o.iter_mut()) {
        *o = f32x4::<Tok>::load(t, s).pow_midp(2.4).to_array();
    }
}

fn main() {
    use archmage::SimdToken;
    let t = Tok::summon().expect("token");
    let small: Vec<[f32; 4]> = (0..N).map(|i| core::array::from_fn(|j| -8.0 + ((i * 4 + j) % 1600) as f32 * 0.01)).collect();
    let positive: Vec<[f32; 4]> = (0..N).map(|i| core::array::from_fn(|j| 0.01 + ((i * 4 + j) % 997) as f32 * 0.1)).collect();
    let mut out = vec![[0f32; 4]; N];
    let kernels: [(&str, fn(Tok, &V, &mut V), &Vec<[f32; 4]>); 4] =
        [("exp2_midp", exp2, &small), ("log2_midp", log2, &positive), ("exp_midp", exp, &small), ("pow_midp(2.4)", pow, &positive)];
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); 4];
    for round in 0..200 {
        for k in 0..4 {
            let v = (k + round) % 4;
            let (_, f, src) = kernels[v];
            let start = Instant::now();
            for _ in 0..16 {
                f(t, black_box(src), black_box(&mut out));
            }
            times[v].push(start.elapsed().as_nanos() as f64 / 16.0);
        }
    }
    black_box(&out);
    for (v, (name, _, _)) in kernels.iter().enumerate() {
        let mut s = times[v].clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{name:14} median {:9.0} ns per 1,024 vectors", s[s.len() / 2]);
    }
}
```

Commands, in each side's directory:

```sh
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-wasip1
wasmtime run target/wasm32-wasip1/release/transcendental-probe.wasm
cargo build --release && ./target/release/transcendental-probe
```

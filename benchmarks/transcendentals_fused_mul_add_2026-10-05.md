# Generic transcendentals on the scalar backend and strict WASM: 0.9.29 vs main (2026-10-05)

Two rounds of measurement on the same day. The first (below, "Interim design")
timed main while `mul_add` fused in software on the scalar backend and strict
WASM (#116; 11a35a8b), and found the transcendentals 1.5–23× slower than in
0.9.29. Before release, `mul_add` returned to the 0.9.29 behavior and the
software path moved to `mul_add_portable` (66986145). The same probe, rebuilt
against main at 2f856d2c, now matches 0.9.29.

## After the split (main at 2f856d2c)

Same probe, hosts, toolchain and method as below; the 0.9.29 binaries were
reused. ns per pass over 1,024 vectors.

x86 scalar backend (Ryzen 9 9950X3D), four rounds:

| Function | 0.9.29 | main | main / 0.9.29, rounds 2 and 4 |
|---|---:|---:|---:|
| `exp2_midp` | 8,607 / 6,862 / 8,562 / 6,662 | 6,639 / 6,890 / 6,657 / 6,687 | 1.004 / 1.004 |
| `log2_midp` | 3,372 / 2,689 / 3,359 / 2,612 | 2,600 / 2,699 / 2,609 / 2,619 | 1.004 / 1.003 |
| `exp_midp` | 8,771 / 6,996 / 8,729 / 6,789 | 6,766 / 7,020 / 6,783 / 6,814 | 1.003 / 1.004 |
| `pow_midp(2.4)` | 12,489 / 9,966 / 12,435 / 9,669 | 9,636 / 10,000 / 9,666 / 9,716 | 1.003 / 1.005 |

Rounds 1 and 3 each began a new shell loop, and the first binary run in each
(0.9.29 both times) ran about 29% slow; rounds 2 and 4 are the settled
comparison.

Strict WASM SIMD128 (wasmtime 40.0.1, same machine), two rounds:

| Function | 0.9.29 | main | main / 0.9.29 |
|---|---:|---:|---:|
| `exp2_midp` | 5,284 / 5,380 | 5,306 / 5,383 | 1.00 |
| `log2_midp` | 1,872 / 1,888 | 1,887 / 1,889 | 1.00–1.01 |
| `exp_midp` | 5,430 / 5,472 | 5,467 / 5,471 | 1.00–1.01 |
| `pow_midp(2.4)` | 11,277 / 11,365 | 11,359 / 11,363 | 1.00–1.01 |

aarch64 scalar backend (Apple M4 Pro), three rounds:

| Function | 0.9.29 | main | main / 0.9.29 |
|---|---:|---:|---:|
| `exp2_midp` | 20,875 / 21,450 / 22,211 | 21,435 / 21,443 / 21,464 | 0.97–1.03 |
| `log2_midp` | 17,253 / 17,253 / 17,253 | 16,703 / 17,253 / 17,253 | 0.97–1.00 |
| `exp_midp` | 21,089 / 21,904 / 22,300 | 20,688 / 22,292 / 22,940 | 0.98–1.03 |
| `pow_midp(2.4)` | 38,943 / 38,945 / 38,948 | 38,815 / 38,932 / 38,948 | 1.00 |

The scalar backend and strict WASM run the same code as 0.9.29 again, so the
transcendentals' speed and results match it. Exhaustive precision for every
backend class: `transcendental_precision_2026-10-05.md`.

## Interim design (main at 93ca6a3)

The generic transcendentals (`exp2_midp`, `log2_midp`, `pow_midp` and the rest)
evaluate their polynomials with `mul_add`: 17 call sites in
`xtask/src/simd_types/generic_gen/transcendentals.rs`, unchanged since 0.9.29.
In 0.9.29 the scalar and WASM backends computed `mul_add` as a multiply and an
add. On main they fuse in software (#116; 11a35a8b), so every polynomial step
pays that cost. x86 v3/v4 and NEON used hardware FMA in 0.9.29 already, so their
transcendentals did not change.

### Method

The probe below was built twice, against `git archive v0.9.29` and against main
at 93ca6a3, using only API both versions have. It times `f32x4<Wasm128Token>` on
wasm32 and `f32x4<ScalarToken>` on native targets: 1,024 L1-resident vectors,
16 passes per sample, 200 rounds rotating the four functions, median per
function. Each build ran twice, alternating 0.9.29 and main.

- Strict WASM: `-C target-feature=+simd128`, wasm32-wasip1, wasmtime 40.0.1, on
  an AMD Ryzen 9 9950X3D. rustc 1.99.0.
- x86 scalar backend: native x86_64 on the same machine, no `-Ctarget-cpu`.
- aarch64 scalar backend: native on an Apple M4 Pro, rustc 1.99.0.

### Results

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

### Reading it

Code that calls transcendentals on the scalar backend or a strict-WASM build got
1.5–23× slower between 0.9.29 and main, without calling `mul_add` itself.
0.9.29 shipped these functions with two roundings on scalar and WASM, and its
accuracy tests passed on every backend. Not measured: browser engines, other
functions and widths, and real kernels.

### Probe source

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

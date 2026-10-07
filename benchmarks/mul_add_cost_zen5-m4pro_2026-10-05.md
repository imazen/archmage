# `mul_add` vs `a * b + c` per backend (Zen 5 and Apple M4 Pro, 2026-10-05)

> **Interim design.** These runs measured main between 11a35a8b and 66986145,
> when `mul_add` fused in software on the scalar backend and strict WASM (#116).
> Before release, `mul_add` returned to the 0.9.29 behavior and the software
> path moved to `mul_add_portable` (66986145). The scalar-backend rows below are
> what `mul_add_portable` costs now. The current forms are measured in
> `mul_add_portable_zen5-m4pro_2026-10-05.md`, which did not reproduce the
> `f32x8` AVX2 stream slowdown reported here: it was code placement.

Bench: `magetypes/benches/mul_add_cost.rs` (zenbench, interleaved). Since 0.9.30
`mul_add` rounds once everywhere: hardware FMA on x86 v3/v4 and NEON, software
fusion on the scalar backend and on strict WASM. zenbench does not build for
wasm32; the WASM numbers are in `mul_add_wasm_wasmtime_zen5-9950x3d_2026-10-05.md`.
Raw zenbench output is in the two `.raw.txt` files with this date.

Hosts and commands (no `-Ctarget-cpu=native`):

- AMD Ryzen 9 9950X3D, Linux 7.0.0-34-generic, rustc 1.99.0 (b940084d7 2026-09-28).
  Run 1: `~/work/zen/scripts/run-heavy --mem 16G -- cargo bench -p magetypes --bench mul_add_cost --features avx512`.
  Run 2: the same with `RUSTFLAGS="--cfg mul_add_cost_redraw"` and a separate target
  directory, which inserts a padding function and redraws code placement.
- Apple M4 Pro, macOS, rustc 1.99.0. Two runs of
  `nice -n 10 cargo bench -p magetypes --bench mul_add_cost`.

magetypes sources were main at 93ca6a3 plus this bench.

## Method

Two shapes per vector type, each inside one `#[arcane]` region:

- **stream**: `out[i] = a[i].mul_add(b[i], c[i])` over 1,024 L1-resident vectors,
  where throughput, loads and stores decide.
- **chain**: `acc = acc.mul_add(x, c)` 1,024 times in one dependency chain, where
  the latency of one fused operation against a multiply and then an add decides.

Before timing, the setup checks that every fused kernel matches
`f32::mul_add`/`f64::mul_add` and every unfused kernel matches `a * b + c`, lane
for lane and bit for bit. Each group's baseline is `a * b + c`; the `mul_add`
column is zenbench's 95% interval for the paired change against it. Negative
means `mul_add` takes less time.

## Results: hardware FMA

| Group | `a * b + c`, run 1 / run 2 | `mul_add`, run 1 | `mul_add`, run 2 |
|---|---|---|---|
| `f32x8` AVX2 stream (Zen 5) | 690 / 672 ns | +5.8% to +9.3% | +4.7% to +7.7% |
| `f32x8` AVX2 chain | 1.1 / 1.1 µs | −34.6% to −27.5% | −37.9% to −33.8% |
| `f64x4` AVX2 stream | 624 / 677 ns | −6.5% to +0.3% | −7.1% to −2.0% |
| `f64x4` AVX2 chain | 1.1 / 1.1 µs | −31.3% to −26.4% | −35.7% to −35.5% |
| `f32x16` AVX-512 stream | 1.2 / 1.2 µs | −4.9% to −0.5% | −0.8% to +0.6% |
| `f32x16` AVX-512 chain | 1.1 / 1.2 µs | −35.7% to −35.5% | −30.2% to −23.4% |
| `f64x8` AVX-512 stream | 1.2 / 1.2 µs | −1.5% to +2.3% | −5.8% to −1.1% |
| `f64x8` AVX-512 chain | 1.1 / 1.2 µs | −35.7% to −35.5% | −36.4% to −29.0% |
| `f32x4` NEON stream (M4 Pro) | 1.2 / 1.1 µs | −1.7% to +1.1% | −0.3% to +3.1% |
| `f32x4` NEON chain | 3.5 / 3.5 µs | −29.6% to −25.8% | −28.4% to −23.4% |
| `f64x2` NEON stream | 1.1 / 1.1 µs | −2.5% to +0.3% | −2.9% to +0.5% |
| `f64x2` NEON chain | 3.7 / 3.8 µs | −28.2% to −25.8% | −27.6% to −25.8% |

## Results: scalar backend (software fusion)

Expressed as `mul_add` time over `a * b + c` time, from the same intervals.

| Group | `a * b + c`, run 1 / run 2 | `mul_add`, run 1 | `mul_add`, run 2 |
|---|---|---|---|
| `f32x8` stream (Zen 5) | 2.3 / 2.9 µs | 3.46–3.47× | 3.08–3.49× |
| `f32x8` chain | 1.3 / 1.4 µs | 5.80–6.62× | 5.87–6.68× |
| `f64x4` stream | 1.4 / 1.6 µs | 12.1–12.8× | 11.3–13.1× |
| `f64x4` chain | 1.4 / 1.1 µs | 10.2–10.8× | 12.7–12.8× |
| `f32x4` stream (M4 Pro) | 1.2 / 1.4 µs | 10.3–11.0× | 10.0–11.8× |
| `f32x4` chain | 4.4 / 4.7 µs | 2.57–2.84× | 2.67–2.99× |
| `f64x2` stream | 1.3 / 1.4 µs | 25.6–28.2× | 26.6–29.0× |
| `f64x2` chain | 5.1 / 3.9 µs | 7.89–8.37× | 8.06–8.63× |

zenbench flagged several M4 Pro scalar rows for a coefficient of variation of
20–29%; their ratios still sit far from 1.

## Reading it

- With hardware FMA, `mul_add` is the better choice for dependency chains, such
  as Horner polynomials: 23–38% less time on both machines. In streaming loops it
  stays within a few percent of `a * b + c`, with one exception: `f32x8` on AVX2
  took 5–9% longer on Zen 5 in both code layouts. Its loop has the same
  instruction count as the unfused one (two loads, an FMA with the third load
  folded, a store, against one load, a multiply and an add with folded loads, a
  store); the cause was not identified.
- The scalar backend fuses in software on every ISA: `fmaf` widens to f64 and
  runs TwoSum with round-to-odd, and f64 calls `libm::fma`. That is 2.6–29×
  slower than `a * b + c` here. On aarch64 the FMA instruction is always
  present, so the scalar backend's software path there is avoidable; it was not
  changed.
- Strict WASM: 8.7× (`f32x4`) and 25× (`f64x2`) slower under wasmtime; relaxed
  SIMD builds 2–3% faster. See the wasmtime record.

Limits: one machine per ISA, L1-resident microbenchmarks. A real kernel's change
depends on how much of its time is in `mul_add`.

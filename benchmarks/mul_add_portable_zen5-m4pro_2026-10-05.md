# `mul_add` and `mul_add_portable` vs `a * b + c` (Zen 5 and Apple M4 Pro, 2026-10-05)

Bench: `magetypes/benches/mul_add_cost.rs` (zenbench, interleaved). It times the
three multiply-add forms 0.9.30 offers:

- `a * b + c`: two roundings on every backend.
- `mul_add`: one rounding where the hardware fuses (x86 v3/v4, NEON). On the
  scalar backend and strict WASM it is `a * b + c`, as in 0.9.29.
- `mul_add_portable`: one rounding on every backend. Hardware FMA on x86 v3/v4 and
  NEON, the same instruction as `mul_add`; software fusion on the scalar backend
  and WASM.

zenbench does not build for wasm32; the WASM numbers are in
`mul_add_wasm_wasmtime_zen5-9950x3d_2026-10-05.md`. Raw zenbench output is in the
two `mul_add_portable_*_2026-10-05.raw.txt` files.

Hosts and commands (no `-Ctarget-cpu=native`), both with rustc 1.99.0
(b940084d7 2026-09-28):

- AMD Ryzen 9 9950X3D, Linux 7.0.0-34-generic.
  Run 1: `~/work/zen/scripts/run-heavy --mem 16G -- cargo bench -p magetypes --bench mul_add_cost --features avx512`.
  Run 2: the same with `RUSTFLAGS="--cfg mul_add_cost_redraw"` and a separate
  target directory, which inserts a padding function and redraws code placement.
- Apple M4 Pro, macOS 27.0. Two runs of
  `nice -n 10 cargo bench -p magetypes --bench mul_add_cost`.

Sources: main at 2f856d2c (`mul_add_portable` added in 66986145) plus this
version of the bench.

## Method

Two shapes per vector type, each inside one `#[arcane]` region:

- **stream**: `out[i] = a[i].mul_add(b[i], c[i])` over 1,024 L1-resident vectors,
  where throughput, loads and stores decide.
- **chain**: `acc = acc.mul_add(x, c)` 1,024 times in one dependency chain, where
  the latency of one fused operation against a multiply and then an add decides.

Before timing, the setup checks every kernel against its rounding contract, lane
for lane and bit for bit: `a * b + c` against two roundings,
`mul_add_portable` against `f32::mul_add`/`f64::mul_add`, and `mul_add` against
one rounding on FMA backends and two on the scalar backend. Each group's baseline
is `a * b + c`; the other columns are zenbench's 95% intervals for the paired
change against it. Negative means less time.

## Results: hardware FMA

`mul_add` and `mul_add_portable` compile to the same instruction here.

| Group | `a * b + c`, run 1 / 2 | `mul_add`, run 1 | `mul_add`, run 2 | `mul_add_portable`, run 1 | `mul_add_portable`, run 2 |
|---|---|---|---|---|---|
| `f32x8` AVX2 stream (Zen 5) | 656 / 665 ns | +0.7% to +1.8% | −2.5% to −0.4% | +0.5% to +1.5% | −3.3% to −1.3% |
| `f32x8` AVX2 chain | 1.1 / 1.0 µs | −35.7% to −35.6% | −35.7% to −35.5% | −35.7% to −35.5% | −35.9% to −35.5% |
| `f64x4` AVX2 stream | 675 / 646 ns | +3.6% to +6.2% | −8.2% to −7.5% | −5.2% to −4.0% | −10.1% to −9.1% |
| `f64x4` AVX2 chain | 1.1 / 1.1 µs | −36.8% to −30.6% | −35.2% to −32.6% | −38.1% to −35.3% | −35.9% to −35.2% |
| `f32x16` AVX-512 stream | 1.2 / 1.1 µs | −1.3% to +1.5% | −1.7% to −0.1% | −2.0% to +0.9% | +2.7% to +3.5% |
| `f32x16` AVX-512 chain | 1.1 / 1.1 µs | −34.5% to −29.1% | −34.9% to −31.8% | −37.9% to −35.4% | −35.7% to −35.6% |
| `f64x8` AVX-512 stream | 1.2 / 1.2 µs | −3.6% to −1.3% | −4.3% to −2.8% | −3.4% to −2.5% | −4.2% to −3.1% |
| `f64x8` AVX-512 chain | 1.1 / 1.1 µs | −34.4% to −30.0% | −35.2% to −33.3% | −35.5% to −34.1% | −35.6% to −35.5% |
| `f32x4` NEON stream (M4 Pro) | 1.0 / 1.0 µs | −5.0% to +1.1% | −3.2% to +2.2% | −4.3% to +1.2% | −2.5% to +2.3% |
| `f32x4` NEON chain | 3.6 / 3.6 µs | −30.3% to −25.0% | −29.6% to −24.4% | −30.2% to −25.1% | −28.6% to −23.8% |
| `f64x2` NEON stream | 1.0 / 1.0 µs | +1.9% to +8.0% | +2.2% to +8.4% | −2.8% to +2.3% | −2.0% to +3.2% |
| `f64x2` NEON chain | 3.8 / 3.7 µs | −30.2% to −25.6% | −30.2% to −25.6% | −29.7% to −25.4% | −30.8% to −26.1% |

## Results: scalar backend

`mul_add` is `a * b + c` here; `mul_add_portable` fuses in software. Its column is
expressed as time over `a * b + c` time, from the same intervals.

| Group | `a * b + c`, run 1 / 2 | `mul_add`, run 1 | `mul_add`, run 2 | `mul_add_portable`, run 1 | `mul_add_portable`, run 2 |
|---|---|---|---|---|---|
| `f32x8` stream (Zen 5) | 2.5 / 2.4 µs | +3.1% to +3.3% | +2.3% to +2.6% | 3.50–3.51× | 3.50–3.51× |
| `f32x8` chain | 1.1 / 1.2 µs | −0.1% to +0.1% | −0.1% to +1.8% | 6.38–6.40× | 6.42–6.43× |
| `f64x4` stream | 1.3 / 1.3 µs | −1.0% to −0.7% | −2.1% to −1.7% | 12.4–12.5× | 12.4× |
| `f64x4` chain | 1.1 / 1.2 µs | −0.1% to +0.1% | 0.0% to +0.2% | 12.2× | 12.3–12.4× |
| `f32x4` stream (M4 Pro) | 360 / 360 ns | −4.9% to −1.5% | −4.8% to −1.0% | 14.9–15.2× | 15.2–15.5× |
| `f32x4` chain | 3.3 / 3.3 µs | −1.5% to +2.4% | −3.6% to +0.2% | 2.71–2.79× | 2.71–2.80× |
| `f64x2` stream | 291 / 294 ns | +0.1% to +2.1% | −3.9% to +0.9% | 29.3–29.5× | 29.2–29.4× |
| `f64x2` chain | 2.3 / 2.3 µs | −0.7% to +0.9% | −2.7% to +0.8% | 9.26–9.44× | 9.12–9.32× |

zenbench flagged several M4 Pro rows for a coefficient of variation of 20–29%.

## Reading it

- With hardware FMA, both fused forms take 24–38% less time than `a * b + c` in
  dependency chains such as Horner polynomials. In streams they land between
  10% faster and 8% slower, and that spread is code placement, not the
  operation: `mul_add` and `mul_add_portable` are the same instruction, yet on
  `f64x4` AVX2 (run 1) one measured +3.6% to +6.2% and the other −5.2% to
  −4.0%, and on `f64x2` NEON +1.9% to +8.4% against −2.8% to +3.2% in both runs.
- The earlier record (`mul_add_cost_zen5-m4pro_2026-10-05.md`) found `f32x8`
  AVX2 streams 5–9% slower with `mul_add` in both layouts. This bench, with the
  same kernels plus the third form, measured −2.5% to +1.8%, so that result was
  placement as well.
- On the scalar backend, `mul_add` is the same code as `a * b + c`; the measured
  −4.9% to +3.3% is placement again. `mul_add_portable` costs 2.7–29.5× the time
  of `a * b + c`: f32 lanes widen to f64 and run TwoSum with round-to-odd, f64
  lanes call `libm::fma`. On aarch64 the scalar backend fuses in software though
  the FMA instruction is always present.
- The M4 Pro's scalar `a * b + c` streams ran at 360 ns (f32x4) and 291 ns
  (f64x2) here, against 1.2–1.4 µs in the earlier record's bench with the same
  kernel bodies; the cause was not identified. The `mul_add_portable` ratios on
  that machine are therefore larger than the earlier record's `mul_add` ratios
  for the same software path (15× against 10–11× for f32x4 streams).

Limits: one machine per ISA, L1-resident microbenchmarks. A real kernel's change
depends on how much of its time is in multiply-add.

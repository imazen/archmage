# Do polyfilled vectors run faster or slower than native-width ones?

Timings, 2026-10-06, of the two kernels whose assembly is in
[polyfill_asm_2026-10-06.md](polyfill_asm_2026-10-06.md): an in-place gain
(multiply) and a sum, each written with `f32x4`, `f32x8` and `f32x16`.

## Result

- **Reductions run faster with the wider type, on every machine.** Each part
  of a polyfilled vector keeps its own accumulator, so the add chain is
  shorter. On a Neoverse-N1, summing 8,192 floats took 1.45 µs with `f32x4`,
  734 ns with `f32x8` and 488 ns with `f32x16`.
- **An elementwise multiply gained nothing on the Neoverse-N1.** The three
  widths ran within 1% of each other from 1,024 floats up. The wider loops
  execute fewer instructions per float, and that did not show up as time.
- **On AVX2 (Zen 5), the `f32x16` multiply took 2 to 9% longer than
  `f32x8`** from 1,024 floats up. The `f32x16` sum ran 1.4 to 1.9 times faster.
- **Small inputs favor the native width.** At 64 floats the `f32x16` sum took
  16% longer than `f32x4` on the Neoverse-N1. The `f32x16` gain took 20% longer
  than `f32x8` on Zen 5.
- **The Apple M4 Pro run is not settled.** Its mean times are 1.9 to 4.8 times
  its minimum times, so something disturbed it. Both columns favor the wider
  types for both kernels. They disagree on how much for the gain kernel: 1.1
  to 1.2 times in the means, up to 2 times in the minima.

## Neoverse-N1 (NEON; native width is `f32x4`)

Mean time per call, and speed relative to `f32x4`. Mean times are within 19%
of the minimum times.

| Kernel | Floats | `f32x4` | `f32x8` | `f32x16` |
|---|---|---|---|---|
| gain | 64 | 11.7 ns | 11.8 ns (0.99×) | 11.3 ns (1.04×) |
| sum | 64 | 7.7 ns | 7.7 ns (1.00×) | 8.9 ns (0.87×) |
| gain | 1,024 | 105.1 ns | 104.9 ns (1.00×) | 104.5 ns (1.01×) |
| sum | 1,024 | 172.6 ns | 88.5 ns (1.95×) | 58.9 ns (2.93×) |
| gain | 8,192 | 806.4 ns | 804.3 ns (1.00×) | 799.7 ns (1.01×) |
| sum | 8,192 | 1.45 µs | 734.3 ns (1.97×) | 488.4 ns (2.97×) |
| gain | 1,048,576 | 193.42 µs | 193.60 µs (1.00×) | 193.79 µs (1.00×) |
| sum | 1,048,576 | 198.16 µs | 139.89 µs (1.42×) | 137.68 µs (1.44×) |

## Zen 5 (AVX2 tier; native width is `f32x8`)

Mean time per call, and speed relative to `f32x8`. Mean times are within 37%
of the minimum times.

| Kernel | Floats | `f32x4` | `f32x8` | `f32x16` |
|---|---|---|---|---|
| gain | 64 | 3.8 ns (0.53×) | 2.0 ns | 2.4 ns (0.83×) |
| sum | 64 | 2.4 ns (0.67×) | 1.6 ns | 1.8 ns (0.89×) |
| gain | 1,024 | 29.6 ns (0.64×) | 18.8 ns | 20.4 ns (0.92×) |
| sum | 1,024 | 73.7 ns (0.38×) | 27.9 ns | 20.2 ns (1.38×) |
| gain | 8,192 | 215.2 ns (0.65×) | 140.8 ns | 146.1 ns (0.96×) |
| sum | 8,192 | 733.0 ns (0.51×) | 371.4 ns | 191.2 ns (1.94×) |
| gain | 1,048,576 | 43.60 µs (0.90×) | 39.18 µs | 39.81 µs (0.98×) |
| sum | 1,048,576 | 96.48 µs (0.51×) | 48.90 µs | 33.04 µs (1.48×) |

## Apple M4 Pro (NEON; native width is `f32x4`)

Mean time per call, and speed relative to `f32x4`:

| Kernel | Floats | `f32x4` | `f32x8` | `f32x16` |
|---|---|---|---|---|
| gain | 64 | 24.1 ns | 21.0 ns (1.15×) | 20.8 ns (1.16×) |
| sum | 64 | 15.7 ns | 12.8 ns (1.23×) | 11.9 ns (1.32×) |
| gain | 1,024 | 245.0 ns | 207.2 ns (1.18×) | 201.3 ns (1.22×) |
| sum | 1,024 | 428.4 ns | 208.0 ns (2.06×) | 121.3 ns (3.53×) |
| gain | 8,192 | 1.73 µs | 1.57 µs (1.10×) | 1.55 µs (1.12×) |
| sum | 8,192 | 4.20 µs | 1.70 µs (2.47×) | 996.5 ns (4.21×) |
| gain | 1,048,576 | 165.64 µs | 153.50 µs (1.08×) | 150.97 µs (1.10×) |
| sum | 1,048,576 | 364.64 µs | 220.36 µs (1.65×) | 205.67 µs (1.77×) |

Minimum time per call from the same run:

| Kernel | Floats | `f32x4` | `f32x8` | `f32x16` |
|---|---|---|---|---|
| gain | 64 | 6.7 ns | 6.9 ns (0.97×) | 6.8 ns (0.99×) |
| sum | 64 | 6.4 ns | 3.0 ns (2.13×) | 3.5 ns (1.83×) |
| gain | 1,024 | 90.0 ns | 49.6 ns (1.81×) | 45.3 ns (1.99×) |
| sum | 1,024 | 130.8 ns | 64.8 ns (2.02×) | 39.6 ns (3.30×) |
| gain | 8,192 | 635.5 ns | 346.2 ns (1.84×) | 324.5 ns (1.96×) |
| sum | 8,192 | 1.39 µs | 660.0 ns (2.11×) | 396.8 ns (3.50×) |
| gain | 1,048,576 | 60.62 µs | 54.73 µs (1.11×) | 54.90 µs (1.10×) |
| sum | 1,048,576 | 197.51 µs | 88.02 µs (2.24×) | 52.57 µs (3.76×) |

A clock-calibration loop started the same way took 0.9 ns per iteration,
against 2.9 ns under `taskpolicy -b`. So the process was not confined to
efficiency cores. Re-running two groups in the foreground
gave the same means. Why the means sit so far above the minima is not
determined.

## Setup

- Kernels and benchmark: `experiments/v4-context-asm/benches/polyfill.rs` at
  `4adad6a9`, branch `draft/v4-context-asm-2026-10-06` (not on `main`).
  Command: `cargo bench --bench polyfill -- --format=md`. The raw output is in
  [polyfill_timing_2026-10-06/](polyfill_timing_2026-10-06/), with the paired
  95% confidence intervals.
- zenbench 0.1.10: every round runs the three widths interleaved, and the
  comparison is paired within rounds.
- Each call goes through the `#[inline(never)]` entry whose assembly was
  dumped. Lengths are multiples of 16, so no kernel runs a scalar tail. The
  gain factor is 1.0, so the buffer keeps its values. Buffers are 256 B, 4 KiB,
  32 KiB and 4 MiB.
- rustc 1.99.0 (b940084d7 2026-09-28), LLVM 23.1.1, release profile, no
  `-C target-cpu`, on all three machines.
- Neoverse-N1: Hetzner CAX31 (8 shared vCPUs), `aarch64-unknown-linux-gnu`,
  `nice -n 19`, load average 0.6 to 1.0 during the run.
- Apple M4 Pro: `aarch64-apple-darwin`, default priority, load average 1.1 to
  1.8 with other sessions active. The build was niced; the timing run was not.
- Zen 5: Ryzen 9 9950X3D, `x86_64-unknown-linux-gnu`, the `v3` tier, niced,
  with other sessions active. This run used the benchmark source just before
  it was committed.

## Not covered

- Kernels with many live vectors, where a wider type's extra registers matter.
- Lengths with a scalar tail.
- Operations that move data between the parts, conversions and transcendentals.

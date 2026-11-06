# Four correctly rounded vector functions

Draft, unpublished. Implemented 4 of 4 selected f32 functions: `exp`, `ln`,
`log2`, `pow`. This does not vectorize the other 29 scalar CORE-MATH ports.
The public interfaces are eight-element arrays and arbitrary-length slices in
`magemath::portable`; pow accepts independent exponents. No consumer migration
or release is included.

Implementation: `42fa8553`; selected layout: `5c9335d4` (arithmetically the same
as `5fc55985`). Built on [CORE-MATH](https://gitlab.inria.fr/core-math/core-math)
`660067dfd3e751c8f07acc31c899ff410aa7206e`, through the safe Rust scalar ports
at `52120bdf`. MIT notices are retained beside each table.

## Algorithm and dispatch

Four f32 inputs use four f64 intermediate lanes. V3 uses packed 256-bit double
arithmetic; NEON and WASM use two 128-bit halves. Array calls evaluate two
blocks; slices dispatch once and handle their final partial block with padding.
The original operation order and rounding interval tests are preserved.
Exp/log polynomials use separate multiply/add; pow contains two explicit fused
steps. Exceptional or ambiguous lanes call the complete scalar implementation.
This is a packed common path with individual scalar repair, not a fully packed
accurate fallback. There are no raw intrinsics or unsafe Rust in magemath.

Nearest-even rounding and gradual underflow are required. NaN classification
is portable; payload/sign, errno and floating-point exception flags are not
part of the contract. Compatibility with a particular libc stays separate.

## Measurements

`results.csv` contains paired zenbench means and 95% confidence intervals for
the paired time difference. Lower vector time is better. Each call processes
1024 inputs including the same input copy on both sides. Normal inputs are a
permuted grid: exp in [-8,8), logs and pow bases in [0.01,4.01); pow alternates
exponents +2.4/-2.4. Mixed inputs replace every eighth base with NaN. These are
throughput workloads, not latency or whole-application measurements.

Core Ultra 7 265K baseline binary, normal inputs (ns/value):

| Operation | Scalar | Vector | Scalar / vector |
|---|---:|---:|---:|
| exp | 2.0649 | 0.9337 | 2.212 |
| ln | 1.9936 | 1.2033 | 1.657 |
| log2 | 1.6179 | 1.1756 | 1.376 |
| pow | 20.6844 | 8.3218 | 2.486 |

The global V3 control gives scalar hardware FMA as well: pow still improves
from 17.2674 to 8.1932 ns/value (2.108x). Thus the benefit is not solely the
baseline scalar implementation's software FMA cost.

Neoverse-N1, normal inputs: exp 8.5955→6.5864, ln 7.8150→5.4313,
log2 5.9404→5.1841, pow 29.5511→17.1085 ns/value. The benchmark ran on an idle
machine; correctness/builds used a separate machine with background fuzzing.
Mixed inputs expose fallback cost: ARM log2 takes 6.7824 vs scalar 5.3698
ns/value, exp 6.6762 vs 6.3376; ln is effectively tied. Intel mixed log2 also
loses in the baseline run (1.5317 vs 1.4300). Do not advertise a universal win.

## How close to optimal?

These are measured improvements, not a proof of optimality. Assembly confirms
packed f64 multiply/add and two packed fused operations in pow; there are no
hardware exp/log/pow instructions used. Table indexing still uses scalar loads
and packs, and fallback branches introduce spills. Those remain optimization
opportunities. The AVX-512 tier still uses logical f64x4, not eight f64 lanes;
a wider backend is not implied by the CPU's capability.

Three layout trials were measured, preserving the bit contract:

- Moving unary repair into one cold call cut Intel normal exp from 1.788 to
  0.957 ns/value. ln/log2 also improved. Applying the same layout to pow made
  mixed pow worse (8.283→9.018), so pow retains per-lane repair.
- Batching eight inputs per slice iteration regressed the selected unary
  layout; the final loop processes four.
- Inlining IEEE rail handling passed full-domain parity and MPFR samples but
  regressed the logarithms. That experiment is preserved at `73f27a43`; the
  selected layout was restored in a forward commit.

These layout trials are separate runs; use their paired scalar controls rather
than treating cross-run point estimates as confidence intervals. AVX-512,
V3, baseline, and ARM results are retained in the CSV. Zen 5, Windows, macOS,
other WASM engines, latency chains, application workloads and native eight-f64
kernels have not been measured here.

## Correctness

- Checked-in directed-MPFR regressions, special values, mixed lanes and slice
  lengths 0 through 40.
- 200,000 directed-MPFR records per function on x86 scalar/V3/V4x, real ARM
  scalar/NEON, strict WASM SIMD, relaxed WASM SIMD, and the forced non-fusing
  wasmtime configuration, plus emulated Nehalem without FMA. Exact bits except
  NaN classification.
- All 460,544 upstream pow hard-case records on x86 and ARM.
- Every binary32 input pattern for exp, ln, log2, and pow(x,2.4) matches the
  scalar implementation on the x86 dispatched path. This is exhaustive
  vector/scalar parity, not an independent proof; pow's two-dimensional input
  space is not exhaustive.
- Existing magemath tests, no_std build, clippy, rustdoc, Rust 1.89 regression
  tests and i686 regression tests. Unsupported V4x tests on the non-AVX-512
  runner were explicitly excluded with the existing caller environment option;
  V4x MPFR checks ran on Ryzen 7900X.

## Reproduction and artifacts

Run under the repository resource wrapper on an eligible host:

```sh
just magemath-core-check expf /path/to/expf.bin
just magemath-core-check logf /path/to/logf.bin
just magemath-core-check log2f /path/to/log2f.bin
just magemath-core-check powf /path/to/powf.bin
just magemath-core-exhaustive
just magemath-core-cost cost.json
just magemath-core-summary cost.json
```

The independent MPFR generator and hard-case import are documented in
[scalar coverage](../../zenmath/docs/CORE_MATH.md). Raw JSON, full logs, assembly,
and reference binaries are outside git at
`/mnt/v/output/archmage/core-math-vector-2026-11-06/`, mirrored under
`/mnt/tower/output/archmage/core-math-vector-2026-11-06/`.
The raw logs record compiler commands and resource-wrapper completion lines.

Ryzen 7900X normal-input vector ns/value, separately built configurations:

| Operation | Default V3 | Opt-in V4x |
|---|---:|---:|
| exp | 1.3141 | 1.8809 |
| ln | 1.4376 | 2.5895 |
| log2 | 1.5421 | 2.4599 |
| pow | 8.4803 | 9.3654 |

Keep the default feature set for these kernels. The optional `avx512` feature
is not a performance recommendation: its higher-tier lowering did not improve
this workload. V4x log2 was significantly slower than its paired scalar control;
its ln/exp differences against scalar had intervals spanning zero. The two
vector configurations were separate runs, not a paired V3/V4x experiment.

`perf record -g` of the selected Intel benchmark captured 15,515 samples with
zero lost samples. The core-cycle report identifies the packed slice symbols
and the logarithm repair helper (`profile-report.txt`); annotated pow/log2
assembly is retained. This profile combines all benchmark groups and the
harness, so its percentages are not per-operation costs. It supports inspecting
table loads, packing and repair paths, not a claim that one instruction is the
universal bottleneck.

Resource-wrapper summaries (the first line includes build/check time):

```text
Intel final: rc=0 56s | peak-RSS 0.40GiB | min-avail 22158MiB | peak-load 2.14
Intel profile: rc=0 18s | peak-RSS 0.02GiB | min-avail 22909MiB | peak-load 0.60
ARM timing: rc=0 17s | peak-RSS 0.01GiB | min-avail 14890MiB | peak-load 0.23
AMD V4x: rc=0 20s | peak-RSS 0.04GiB | min-avail 18554MiB | peak-load 0.61
```

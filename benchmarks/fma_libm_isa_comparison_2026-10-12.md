# Portable FMA: published 0.9.30 versus direct libm, 2026-10-12

The unconditional direct-libm scalar path regresses native throughput and some
x86 dependency chains substantially. Matching direct libm assembly is not a
sufficient performance criterion: the published implementation exposed FMA to
LLVM in ways that libm's architecture implementations do not. The recommendation
from this comparison is to preserve native intrinsic lowering before releasing
this change. ARM and WASM also have measured wins, so a blanket statement that
either implementation is faster would be wrong.

This report measures runtime, not compile time or downstream codec throughput.
No FMA implementation was changed during this benchmark pass.

The unmeasured cases are physical x86 CPUs without FMA, browser WASM engines,
32-bit x86, no_std runtime profiles, exception-heavy input workloads, and
every possible token/width polyfill combination. V4x and higher
ARM tokens are not separately enumerated. These results cover the native FMA
widths below, all scalar float widths, and explicit scalar-in-V3 contexts.

## Sources and coverage

Both source trees are unmodified release/main snapshots, except for adding the
same standalone benchmark package:

- Published `v0.9.30`: `e2dbab66ef5aa08f8e23ed05248e7d1217f58475`.
- Current main: `a3372cea0c5596d25e42ac16c7dc3f64417257f1`.
- Direct-libm implementation: `9e462e61c65053be71a55b32197c8c709b4813d3`.
- Compiler for every artifact: Rust 1.99.0 (`b940084d7`, LLVM 23.1.1).
- Both standalone packages enable default std features plus `avx512` (and
  consequently W512), with release `codegen-units=1`.
- Both dependency locks resolve libm 0.2.16 and zenbench 0.1.10. Only the three
  Archmage workspace package versions differ between the locks.

This is a release-to-main comparison, including macro changes and the W512
explicit-lane generator change. Native controls and disassembly identify the
scalar FMA differences; the measured total is not attributed to one source line.

| Profile | Physical CPU / engine | Global target flags | Selected native controls |
|---|---|---|---|
| `intel-full` | Intel Core Ultra 7 265K | baseline x86-64 | AVX2 |
| `intel-v3-full` | Intel Core Ultra 7 265K | `-Ctarget-cpu=x86-64-v3` | AVX2 |
| `zen5` | AMD Ryzen 9 9950X3D (Zen 5) | baseline x86-64 | AVX2, AVX-512 |
| `arm-n1` | ARM Neoverse-N1 | baseline AArch64 Linux | NEON |
| `mac-m4` | Apple M4 Pro | baseline AArch64 Darwin | NEON |
| `wasm-strict-full` | Intel Core Ultra 7 265K / Wasmtime 49.0.2 | `+simd128` | SIMD128 |
| `wasm-relaxed-full` | same CPU / engine | `+simd128,+relaxed-simd` | SIMD128 |

Scalar types: f32x1/x4/x8/x16 and f64x1/x2/x4/x8. Native widths:
AVX2 f32x4/x8 and f64x2/x4; AVX-512 f32x16 and f64x8; NEON and WASM
f32x4 and f64x2. Each x86 profile also runs scalar f32x1/x8 and f64x1/x4
inside a V3 function. Linux kernels are CPU-pinned. Mac runs are not pinned.
No build uses `target-cpu=native`; ARM Linux is cross-compiled and executed
on the native CPU, not emulated.

## Results

Ratios are **current elapsed time / published elapsed time**: above 1 is slower.
Each table entry is the median of three independent process medians. Times are
per complete kernel invocation, not per lane or operation. Normal streams
process 1,024 vectors; chains perform 1,024 dependent vector operations.

| Case | Published stream µs | Current stream µs | Ratio | Published chain µs | Current chain µs | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Intel, baseline, scalar f32x8 | 6.368 | 6.896 | 1.08× | 2.932 | 6.068 | 2.07× |
| Intel, V3 function, scalar f32x8 | 4.024 | 6.936 | 1.72× | 2.149 | 6.066 | 2.82× |
| Intel, global V3, scalar f32x8 | 0.753 | 6.735 | 8.95× | 0.706 | 6.231 | 8.83× |
| Zen 5, baseline, scalar f32x8 | 7.707 | 8.431 | 1.09× | 7.049 | 7.574 | 1.07× |
| Zen 5, V3 function, scalar f32x8 | 4.970 | 8.415 | 1.69× | 2.456 | 7.583 | 3.09× |
| Neoverse-N1, scalar f32x4 | 0.684 | 2.089 | 3.05× | 2.404 | 1.387 | 0.58× |
| Apple M4 Pro, scalar f32x4 | 0.248 | 0.511 | 2.06× | 0.993 | 0.733 | 0.74× |
| WASM strict, scalar f32x4 | 7.600 | 4.242 | 0.56× | 4.821 | 5.857 | 1.21× |
| WASM relaxed, scalar f32x4 | 7.599 | 4.241 | 0.56× | 4.820 | 5.860 | 1.22× |
| Zen 5, native AVX2 f32x8 | 0.689 | 0.690 | 1.00× | 0.708 | 0.704 | 1.00× |
| Zen 5, native AVX-512 f32x16 | 1.274 | 1.309 | 1.03× | 0.709 | 0.704 | 0.99× |
| Neoverse-N1, native NEON f32x4 | 0.719 | 0.713 | 0.99× | 2.404 | 2.399 | 1.00× |
| WASM strict, SIMD128 f32x4 | 6.141 | 6.141 | 1.00× | 9.845 | 9.877 | 1.00× |

The native controls are useful context, not proof of zero change everywhere.
For example, the Zen 5 AVX2 f32x4 portable stream measured 1.13× while the
AVX2 f32x8 control above measured 1.00×. Small differences can reflect layout
and scheduling. The substantial scalar regressions have distinct assembly
changes and recur across all three process pairs.

Double precision also varies by context. W512 includes the explicit-lane
generator improvement, so its baseline gain must not be generalized to other
widths or feature contexts.

| Normal-input scalar case | Portable add stream | Portable add chain | Portable subtract stream | Portable subtract chain |
|---|---:|---:|---:|---:|
| Intel baseline f64x8 | 0.72× | 0.60× | 0.74× | 0.62× |
| Intel global V3 f64x4 | 4.79× | 3.62× | 5.12× | 2.91× |
| Neoverse-N1 f64x2 | 1.41× | 0.57× | 1.58× | 0.57× |
| Apple M4 Pro f64x2 | 1.19× | 0.71× | 1.50× | 0.71× |
| WASM strict f64x2 | 1.02× | 0.99× | 1.02× | 0.99× |

Short streams process 16 vectors; the chain length remains 1,024. The subnormal
scenario scales stream multiplicands and addends, and chain addends, by
`MIN_POSITIVE`. Subnormal arithmetic can dominate other overhead on Intel,
while the packed-versus-scalar distinction remains costly in the global V3
build. These scenarios are separate measurements, not pooled with normal data.

| Portable-add case | Short stream | Short chain | Subnormal stream | Subnormal chain |
|---|---:|---:|---:|---:|
| Intel V3 function, scalar f32x8 | 1.71× | 2.79× | 1.03× | 1.02× |
| Intel global V3, scalar f32x8 | 12.87× | 9.06× | 7.83× | 8.07× |
| Zen 5 V3 function, scalar f32x8 | 1.78× | 3.10× | 1.70× | 3.10× |
| Neoverse-N1 scalar f32x4 | 2.36× | 0.58× | 3.00× | 0.58× |
| Apple M4 Pro scalar f32x4 | 2.11× | 0.79× | 2.07× | 0.74× |
| WASM strict scalar f32x4 | 0.58× | 1.22× | 0.65× | 1.21× |

Subtraction shows the same broad issues: normal scalar f32x4 streams take
3.48× as long on N1 and 2.03× on M4 Pro; the Intel global-V3 scalar f32x8
stream takes 8.83×. Neither version exposes `f64x1::mul_sub_portable`, so that
case is explicitly named `mul_add_portable(-c)` in the raw data.

Cross-process variation is retained in the CSV. For the Intel global-V3 scalar
f32x8 normal stream, the published process medians span 0.725–0.780 µs and
current medians 6.699–6.827 µs. For the N1 scalar f32x4 normal stream, the
ranges are 0.675–0.685 µs and 2.087–2.104 µs. Some smaller effects are noisier:
the WASM strict scalar f32x4 current stream medians span 4.236–5.557 µs;
the median is 4.242 µs. No significance threshold or pooled ISA average is used.

## Assembly evidence

The saved disassembly comes from the measured binaries:

- `artifacts/asm/published-global-v3-scalar-f32x8-chain.s` contains packed
  `vfmadd213ps` on YMM registers. The corresponding `current-...` file has
  repeated `call *%r15` per lane, with spills. Libm's x86 `select_once!`
  caches the selected function pointer; warming selection does not eliminate
  the indirect calls, even with global V3 features.
- The baseline-build `published-baseline-v3ctx-f32x8-chain.s` contains scalar
  FMA instructions in the hardware arm, alongside the feature-cache/software
  branch. Its current counterpart again calls libm per lane. This regression
  also affects ordinary baseline binaries with a per-function feature context.
- `published-arm-scalar-f32x4-stream.s` uses vector `fmla ... .4s`;
  its current counterpart uses four scalar `fmadd` instructions. Libm's
  AArch64 implementation uses scalar inline assembly. It prevents the
  vectorization seen in the published intrinsic-based path. The scalar chain
  results can improve despite worse stream throughput; latency and throughput
  are different workloads.

The scalar WASM f32 implementations also differ: the published helper uses
f64 arithmetic with its rounding correction, while current uses libm's
software algorithm. The measured stream wins and chain losses are
input-dependent. The SIMD128 portable f32 implementation is a separate path
and remains close in these measurements, including the relaxed-SIMD build.

A separate no_std probe on Rust 1.99 confirms that core f32/f64 `.mul_add()` is
still unavailable. A future fix needs an actual LLVM-visible lowering; merely
substituting a method name is not an implemented solution. The earlier check
that the helper matched direct libm remains valid, but it did not compare
whole-kernel performance against the shipped implementation.

## Measurement and validation

Every registered stream output and final chain lane is checked bit-for-bit
against its fused or unfused contract before timing. This covers the benchmark
inputs, not an exhaustive floating-point correctness proof. The controls are
`a * b + c` and `mul_add`; portable add and subtract are the operations under
comparison. WASM omits the bare `mul_add` control because relaxed madd may
round once or twice, unlike the portable contract.

For each of seven profiles and three scenarios there are three alternating
published/current process pairs: old/new, new/old, old/new. The validator
requires all **126 process runs**, matching named case counts, 30–100 samples
per case, no resource-unreliable flag, and three samples of process medians on
each side of every comparison. It accepted **2,040 case/scenario comparisons**,
including controls. No incomplete pilot or failed run is included.

Native timing uses zenbench 0.1.10 with Instant, 20 ms warmup, 30–100 rounds,
and a requested three-second measured-group budget. All cases share an
interleaved group to amortize host-resource checks; its group-wide ratios and
throughput are ignored. Only matching case names across versions are compared.
Allocation, process startup, and initial CPU-detection cache population are
outside timing. Working sets vary with vector width and are not all L1-resident.

Zenbench's WASI engine aborts in `std::env::temp_dir` before measurement; the
failed run is preserved. The WASI-only adapter measures the same validated
kernels using Instant, 20 ms warmup per case, batches calibrated to 1 ms,
30–100 rotating rounds, and a three-second group budget. All raw samples are
saved. JIT and process startup are excluded; host resource monitoring is
provided by the outer run wrapper, not a WASI guest resource gate.

Heavy jobs ran serially. Linux commands used `run-heavy`; these are its complete
resource-summary lines for the final measurement commands, not per-kernel
allocation measurements:

```text
intel-full-driver.log: run-heavy: done rc=0 404s | peak-RSS 0.02GiB | min-avail 23806MiB | peak-load 4.92
zen5-driver.log: run-heavy: done rc=0 310s | peak-RSS 0.22GiB | min-avail 37951MiB | peak-load 2.95
arm-n1-driver.log: run-heavy: done rc=0 55s | peak-RSS 0.03GiB | min-avail 29796MiB | peak-load 1.42
arm-n1-edge-driver.log: run-heavy: done rc=0 143s | peak-RSS 0.03GiB | min-avail 29384MiB | peak-load 6.55
wasm-matrix-driver.log: run-heavy: done rc=0 153s | peak-RSS 0.04GiB | min-avail 25234MiB | peak-load 3.56
mac-m4-driver.log: run-heavy: done rc=0 58s | peak-RSS 0.01GiB | min-avail 24212MiB | peak-load 1.35
mac-m4-edge-driver.log: run-heavy: done rc=0 124s | peak-RSS 0.01GiB | min-avail 25989MiB | peak-load 1.04
```

For Mac, that wrapper monitors only the Linux SSH coordinator. The native Mac
command used `nice -n19` and BSD `/usr/bin/time -l`: normal runs reported
22,691,840 bytes maximum RSS; short/subnormal runs reported 22,282,240 bytes.
The Linux cgroup does not impose a remote Mac memory cap.

## Reproduction and retained artifacts

The experimental harness stays outside main, as required by release discipline.
The durable artifact directory is `/mnt/v/output/archmage/fma_isa_2026-10-12/`
on `dev`. It contains the standalone experiment's source/history, both source
snapshots, locked dependencies, binaries, disassembly, complete logs, and
`results/comparison.csv`. `results/SHA256SUMS.json` hashes the raw logs/results
and binaries. The root `ARCHIVE_SHA256SUMS.json` checks the complete archive
excluding itself. Build caches are excluded.

- Archive manifest SHA-256: `6f57589fd864ccbcd12ff22b45c02aed5ceb716a783b7484f5d3e968c84cfdbd`
  (3,082 archived files verified after transfer).
- Comparison CSV SHA-256: `05b3b6062b6e039c377618e2abd5efdbb1426de314b6a5edf4c97e87dc382abf`.
- Results manifest SHA-256: `9b30b2ee1c8f2690006c864e61abc7f1fdfa73ae24370c9e258fc4378ba13b49`.
- Identical native harness SHA-256: `1f51d52d4e7c7f80e0d447522a688a69c1cb04644b60b9a2ad3d2a1320c861dd`.

From a copy of that experiment, `prepare.py` reconstructs the native harness
and standalone manifests; `prepare_wasm.py` derives the WASI harness. Restore
each `locks/*.Cargo.lock` to its corresponding standalone package before
building. `build.py` records compiler commands and reuses existing artifacts;
use a fresh output location or preserve/rename artifacts before rebuilding.
The source snapshots can be recreated with `git archive` at the revisions above.

Relevant commands, run through the host's resource wrapper:

```sh
just matrix x86 arm wasm relaxed v3
just intel-scenarios 4
just scenarios zen5 x86 --isas v3,v4 --cpu 2
just scenarios arm-n1 arm --isas neon --cpu 6
FMA_TOOLCHAIN=+stable just matrix mac
just scenarios mac-m4 mac --isas neon
just wasm-scenarios /absolute/path/to/wasmtime 4
python3 finalize.py
```

Run each profile on the corresponding machine in the coverage table. WASM needs
Wasmtime 49.0.2 and the `wasm32-wasip1` Rust target. The archive's per-run metadata
is authoritative for the actual CPU pin, environment, command, and timestamp.
The finalizer deliberately refuses to overwrite an existing comparison CSV;
preserve the old output before collecting a new experiment.

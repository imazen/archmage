# Scalar Rust CORE-MATH cost

33 safe Rust ports, 2026-11-05/06. This standalone, unpublished workspace keeps
measurement dependencies out of zenmath. `zenbench = 0.1.10` is pinned to the
version used by the existing workspace benchmarks.

Each group streams 1024 values through the Rust port and host std, interleaved
by zenbench. All output elements reach black_box. The same indirect-call shape
is used for both implementations. Inputs are permuted grids: [-8,8) generally,
[0.25,4) for logs/pow/hypot, and [-0.99,0.99) for acos/atanh. Binary operations
use second argument 2.4. These measure common inputs, not hard-case tail latency.
Std is a performance reference with a different accuracy contract, not an oracle.

## Reproduce

On an eligible idle host, with TMPDIR set to a disk-backed scratch directory:

```sh
run-heavy --mem 4G --jobs 1 -- just zenmath-core-bench /path/cost-software.json
RUSTFLAGS='-C target-cpu=x86-64-v3' run-heavy --mem 4G --jobs 1 -- just zenmath-core-bench-hardware /path/cost-v3.json
# ARM64 hardware FMA needs std but no extra target flags:
run-heavy --mem 4G --jobs 1 -- just zenmath-core-bench-hardware /path/cost-arm-hardware.json
python3 benchmarks/core_math_rust_2026-11-05/analyze.py /path/cost-*.json
```

The standalone workspace can also be run directly with
`cargo run --release --manifest-path benchmarks/core_math_rust_2026-11-05/Cargo.toml -- OUTPUT.json`.
Add `--features std` before `--` for hardware configurations. No build used
`target-cpu=native`. x86-64-v3 requires a compatible CPU; the baseline executable
does not. Both implement the same result contract.

## Conditions and interpretation

Core Ultra 7 265K: Rust 1.99.1 / LLVM 23.1.1, glibc 2.43.
Neoverse-N1: Rust 1.97.1 / LLVM 22.1.6, glibc 2.39.
The configurations are separate process runs, not paired software-versus-hardware
measurements. Core-versus-std comparisons within a configuration are interleaved.
The logs contain drift and variance warnings for some groups. Do not interpret
small differences as established wins. `costs.csv` retains each mean, MAD and
sample count; full JSON retains the paired samples and diagnostics.

x86 measurements used draft implementation `0cbf7740` (validation script fix
`283f00f1`); ARM measurements include the exact-power proof guard `306b58f7`.
That guard only changes fallback acceptance, not the common arithmetic path.
Bench source and lockfile are committed beside this report.

## Code generation

Disassembly of the x86 benchmark executables confirms the std/V3 FMA selection.
For the standalone powf/log/sin/pow symbols respectively, baseline contains
0/0/0/0 FMA instructions; V3 contains 2/7/2/25. These are static instruction
counts, not execution counts. Ordinary multiply/add expressions remain separate.
The retained assembly is an artifact, not a claim that LLVM found the fastest
schedule. The earlier [C/ICX study](../core_math_cost_2026-11-05/README.md) used
different inputs and a different harness; do not compare absolute timings across
the two studies as a compiler speed ratio.

## Artifacts

Raw JSON, logs, assembly, MPFR records and hashes are retained under
`/mnt/v/output/archmage/core-math-rust-2026-11-05/`, mirrored under the matching
`/mnt/tower/output/` path. The empty acosf-wc.bin is a retained failed-attempt
artifact, not a validation corpus; no upstream acosf hard-case file exists.

## Measured stream costs

Mean nanoseconds per value, from the complete 33-function runs:

| Function | x86 baseline software FMA | x86 V3 hardware FMA | ARM software FMA | ARM hardware FMA |
|---|---:|---:|---:|---:|
| exp2f | 2.277 | 1.916 | 7.602 | 7.660 |
| powf | 18.202 | 18.181 | 52.321 | 28.794 |
| log | 32.364 | 3.190 | 91.441 | 12.406 |
| sin | 24.106 | 5.157 | 65.522 | 17.064 |
| pow | 113.059 | 18.885 | 337.608 | 38.378 |
| tanh | 18.844 | 4.706 | 54.019 | 22.182 |

The hardware paths substantially reduce several f64 costs. Powf did not show
a corresponding x86 gain on this workload. No universal fastest-implementation
claim follows from this table; runtime CPU dispatch and SIMD ports are not
implemented by this change. The x86 baseline remains executable without FMA.

The first ARM run encountered background load (363 gate waits in the hardware
run). The table uses a rerun of the **same binaries** on an idle Neoverse-N1
host; both runs are retained. Resource wrapper summaries:

- x86 build and two runs: rc=0, 265 s, peak RSS 0.34 GiB, minimum available 22591 MiB, peak load 0.82.
- initial ARM validation/build/two runs: rc=0, 329 s, peak RSS 0.33 GiB, minimum available 27500 MiB, peak load 6.70.
- idle ARM two runs: rc=0, 206 s, peak RSS 0.01 GiB, minimum available 14873 MiB, peak load 0.43.

These RSS figures describe the resource wrapper's entire command, including
compilation where named; they are not estimates of per-function memory use.

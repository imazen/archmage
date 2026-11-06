# PowBehavior validation and costs — 2026-11-06

Implementation source: `125cdea9` (reviewed and integrated in primary draft
`8ad71166`). This directory is an unpublished measurement harness. The
[implemented policy guide](../../magemath/docs/POW_BEHAVIOR.md) owns semantics;
the [preceding candidate study](../portable_f32_2026-11-06/README.md) remains separate.

The default keeps packed CORE-MATH. Nondefault correct-rounding policies prepare
packed magnitude inputs and restore explicit signs afterward; Midp/Lowp use
packed log2 × per-lane exponent → exp2, with scalar exceptional rescue. This
adds no new global approximation bound or ULP1 preset. Direct matching-feature
kernels add no CPU probe; array dispatch selects per call, slice dispatch once
per buffer.

## Correctness and portability

The frozen 791,134-record pow corpus includes independent MPFR references,
CORE-MATH hard-to-round inputs and boundaries. Independently derived
negative-integer cases extend it to **909,487**. The preceding study replayed
16 targeted-search cases separately; that replay was not rerun for policies.
Eighteen selected policies pass scalar/packed parity; default also passes the
independent correctly rounded oracle. These are broad sampled tests, not a new
proof or every possible policy combination.

[Cross-target results](cross-equality.csv): 90 policy/target comparisons, each
909,487 records, have zero non-NaN bit mismatches against baseline x86 guarded
parity. Comparisons cover the original eager-parity native output, real AArch64,
global x86 V3, the **same baseline binary** under QEMU Nehalem, and strict WASM.
Signed zeros count as mismatches. NaN payload/sign differences are counted
separately; the canonical-NaN policy requires exact bits.

Scoped native/ARM/WASM functional suites, no-default-feature checks, bare-metal
`aarch64-unknown-none`, and **actual Rust 1.89.0** tests/checks/doctests passed.
Compile-fail tests reject unsupported encodings and conflicting policies. The
policy guide's actual Rust fence runs as a doctest. Final source-identical native
and WASM all-target clippy, formatting and doctests passed; all 129 tracked
source hashes matched before/after. No Windows/macOS policy execution is claimed.

## Mathematical negative-base optimization

Frozen original branch fallback versus selected guarded packed abs/parity,
paired zenbench 0.1.11 results. Negative percentages mean less elapsed time.
Confidence intervals are paired **time deltas**, ns/value, at 95% confidence.

| Workload | Core Ultra 7 265K change | 95% delta CI | Neoverse-N1 change | 95% delta CI |
|---|---:|---:|---:|---:|
| Positive | -10.587% | [-0.8262, -0.7625] | **+7.547%** | [1.1932, 1.3533] |
| Negative integer | -64.176% | [-15.5236, -15.3636] | -52.331% | [-23.5350, -23.3213] |
| Mixed negative | -27.562% | [-3.5753, -3.4237] | -20.751% | [-5.6889, -5.4525] |
| Exceptional | -36.656% | [-5.0844, -4.9489] | -27.188% | [-8.0693, -7.7751] |

This is a workload tradeoff, including an ARM positive-input regression. The
initial eager-parity experiment cost +2.151% on x86 positive inputs versus the
original. Guarding parity when no lane is negative improves positive throughput
versus eager parity on both measured CPUs, but costs 4.259–15.515% on the other
x86 workloads and 10.171–11.552% on the other ARM workloads versus eager parity.
All paired comparisons, CIs, outliers and source JSON names are retained in
[paired-timing.csv](paired-timing.csv); raw rounds remain in the archive.

## Nondefault packed policies

Same-contract scalar versus packed policies, paired retained means in ns/value.
Each cell is **scalar / packed**. These compare policy implementations, not
approximate precision against correctly rounded precision.

| Policy, CPU | Positive | Negative integer | Mixed | Exceptional |
|---|---:|---:|---:|---:|
| SignedMagnitude, x86 | 19.809 / 8.589 | 19.525 / 8.510 | 19.831 / 8.548 | 16.684 / 8.448 |
| Midp Full, x86 | 34.506 / 4.139 | 34.523 / 4.776 | 25.430 / 4.059 | 28.303 / 4.072 |
| Lowp Full, x86 | 12.164 / 3.362 | 12.626 / 4.433 | 10.747 / 3.569 | 10.834 / 3.575 |
| SignedMagnitude, ARM | 31.763 / 17.983 | 31.847 / 18.003 | 31.829 / 18.018 | 29.922 / 18.367 |
| Midp Full, ARM | 20.273 / 8.978 | 21.715 / 10.392 | 18.724 / 9.864 | 19.931 / 10.110 |
| Lowp Full, ARM | 16.918 / 8.275 | 20.577 / 9.764 | 16.388 / 9.283 | 17.443 / 9.745 |

Timing uses 1,024-value arrays, identical copy/black-box controls, baseline
compilation without target-cpu=native, one process and paired rounds. x86 runs
on CPU0 within an enforced performance-core cpuset; ARM uses CPU0 on a quiet
homogeneous machine. ARM timing binaries were built with Rust 1.97.1 on another
AArch64 machine. x86 used Rust 1.99.1. CIs describe within-run uncertainty;
these are not independent multi-process reproducibility estimates. No reruns
were made solely to chase small noise.

## Codegen, size and build cost

[Measured report](codegen-report.json), reproducible with `run-codegen.sh` and
`analyze_codegen.py`:

- Optimized legacy/default array wrappers contain identical normalized
  instructions: 983 instructions, 4,595-byte bodies. Both bodies must be found
  before the analyzer compares them.
- Release precondition policies with the optional debug flag off/on call one
  shared 409-byte body. Assertion text and the debug-input symbol are absent
  even in an opt-level=0 release build. This demonstrates structural cfg removal.
- One additional SignedMagnitude call/instantiation increased the executable
  from 457,624 to 460,888 bytes (+3,264), `.text` from 274,133 to 277,237
  (+3,104), and `.rodata` from 27,304 to 27,320. Call-site/alignment costs are
  included. This is one probe, not a general per-policy size guarantee.
- Isolated cold builds: base 14.10 s wall / 25.70 s user / 2.11 s system,
  GNU time max RSS 503,944 KB; extra policy 14.15 / 25.66 / 2.19 s,
  503,404 KB. **One cold pair does not establish a build-time overhead bound.**
  Opt0 release took 5.04 s, executable 1,020,984 bytes; no opt0 instruction
  equality between legacy/default wrappers is claimed.

## Execution evidence and reproduction

Run from this directory via the checked-in justfile and runner scripts. Use
`just functional` for scoped tests and separate doctests; benchmark targets
are harness=false, so `cargo test --all-targets` also runs unrelated benches.
The accidentally started debug benchmark in `policy-native-suite.log` was
interrupted by its exact owned PID; **all timing from that log is invalid**.
Other retained failed attempts: initial clippy `manual_clamp`, duplicate table
module clippy, and rustdoc interval strings interpreted as links. Corrected
named checks passed; no established expectation was weakened.

Serialized run-heavy resource summaries (peak RSS; minimum available RAM; peak
load), with full stdout/stderr retained:

| Job/log | Exit, wall | Peak RSS | Min available | Peak load |
|---|---|---:|---:|---:|
| policy-native-scoped.log | 0, 23 s | 0.33 GiB | 22,416 MiB | 1.18 |
| sign-conditional-fixed.log | 0, 6 s | 0.30 GiB | 22,847 MiB | 0.28 |
| policy-arm.log | 0, 106 s | 0.51 GiB | 26,202 MiB | 9.80 |
| policy-x86-cross-fixed.log | 0, 97 s | 0.73 GiB | 22,015 MiB | 1.46 |
| policy-msrv.log | 0, 16 s | 0.34 GiB | 47,109 MiB | 0.97 |
| policy-timing-arm.log | 0, 12 s | 0.01 GiB | 14,896 MiB | 0.15 |
| policy-codegen.log | 0, 40 s | 0.48 GiB | 22,226 MiB | 1.12 |
| policy-final-source.log | 0, 7 s | 0.45 GiB | 22,637 MiB | 0.10 |

[Artifact identities and verified mirror](ARTIFACTS.md) retain raw timing JSON,
all policy outputs, oracle corpus, binaries, codegen/disassembly, compiler and
CPU identities, logs, source archive and SHA manifests. No push or release.

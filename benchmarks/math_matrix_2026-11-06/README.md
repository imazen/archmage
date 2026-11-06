# Math throughput matrix and algebraic probes — 2026-11-06

Unavailable first: Windows/macOS execution, Magetypes f64 vector log2 and other
missing public APIs listed in [MATRIX.md](MATRIX.md). One WASM cell,
metallic general pow/specials, traps. No global approximation error proof,
Zen 5 speed claim, or full instruction-latency matrix is made.

The requested nine f32 operations and f64 log2 have **4,568 of 4,569 runnable throughput
cells**, plus 180 algebraic and 343 representative native feedback cells:
**5,091 total**, all at least 30 rounds, complete/reliable per zenbench. Uniform 2.4
power and separately labeled ties-even rounding are extra controls.
[Exact coverage](results/coverage.csv), [source/API inventory](api-matrix.csv),
[accuracy](ACCURACY.md), [algebraic codegen/results](CODEGEN.md) and
[feedback units](FEEDBACK.md) give the scope and limitations.

This is unpublished draft work based on 77b2545b. No production API, MSRV,
math contract or established expectation changed. No push/PR/release.

## Matching CPU context matters

Intel ordinary inputs, 1,024 values, mean ns/value. Baseline is the default
x86-64 build; V3 is an explicit feature context in the same binary. The scalar
Zenmath primitives use their software algorithms even when std is enabled.

| Operation | std baseline | std V3 loop | Zenmath baseline | Zenmath V3 loop | Packed V3 correct/native |
|---|---:|---:|---:|---:|---:|
| powf | 2.6819 | 2.6932 | 19.4035 | 21.5668 | 7.0049 |
| exp | 1.2177 | 1.2480 | 1.7037 | 1.7032 | 0.8489 |
| ln | 1.4133 | 1.2388 | 1.8782 | 1.6352 | 1.0767 |
| log2 | 1.2311 | 1.2541 | 1.6907 | 1.6419 | 1.1740 |
| log10 | 1.9865 | 1.7021 | 2.1468 | 2.3512 | unavailable |
| sqrt | 0.1656 | 0.0843 | 2.6704 | 2.6679 | 0.0858 |
| floor | 0.8922 | 0.0294 | 0.9463 | 0.5368 | 0.0348 |
| ceil | 0.8499 | 0.0428 | 0.6743 | 0.5472 | 0.0522 |
| round | 0.7381 | 0.0446 | 1.1345 | 0.6112 | unavailable |
| log2_f64 | 2.4024 | 1.7801 | 8.1459 | 7.6442 | unavailable |

The large baseline-to-V3 primitive gain is already present in the std loop.
Explicit vectors do not own that gain. Scalar software sqrt is a distinct
cost; do not hide it by substituting native std. Ties-away round has no existing
Magemath/Magetypes packed counterpart. Correct Magemath slice controls include
an input-buffer copy, disclosed in MATRIX.md; these are API-shape measurements.
Packed sqrt/floor/ceil use Magetypes native methods. Close point estimates are
not ranked without the CSV's paired intervals.

## Scalar alternatives

Same Intel ordinary workload, ns/value, directly invoking each pinned public
function. Missing APIs are never supplied by std adapters.

| Operation | libm 0.2.16 | fpmath 0.2.0 | micromath 2.1.0 | metallic 0.3.1 |
|---|---:|---:|---:|---:|
| powf | 15.7930 | 6.7524 | 7.8627 | 38.5720 |
| exp | 2.1969 | 1.2080 | 3.6720 | 20.3451 |
| ln | 1.8881 | 1.7607 | 0.5399 | 12.1582 |
| log2 | 2.3603 | 1.7783 | 0.6289 | 11.2570 |
| log10 | 2.7457 | 1.7815 | 0.5583 | 11.0728 |
| sqrt | 0.7051 | 10.0178 | 0.1032 | unavailable |
| floor | 0.6988 | 0.9481 | 0.6316 | unavailable |
| ceil | 0.8207 | 0.8231 | 0.5043 | unavailable |
| round | 0.6860 | 0.9321 | 0.4817 | 0.7135 |
| log2_f64 | 3.4121 | 4.6376 | unavailable | 13.8086 |

These alternatives have different precision/special-value contracts; the speed
columns alone do not establish eligibility. The full matrix also measures
Zenmath approximations, Magemath portable/unfused/relaxed midp/lowp, Magetypes
midp/lowp/precise, packed PowBehavior general exponents, and direct-proof versus
once-buffer dispatch. Checked forms are used; unchecked precondition-only APIs
are not separately timed. The libm build uses default features; this study does
not add another force-soft-floats timing build.

## Precision and portability findings

Independent directed MPFR checked 1,024 records in each of the 4,568 accepted math cells.
Correctly rounded Zenmath/Magemath matched all references on the four measured
targets; this is sampled evidence rather than a global proof. Approximation
errors, specials and signed zeros remain recorded even outside stated domains.

[Scalar consistency](results/cross-scalar.csv) finds std log10 differs from
x86 on 258/1024 ordinary ARM values and 105/1024 WASM values, at most one
representable step. Example x=`405523d7`: x86=`3f05c193`, ARM/WASM=`3f05c194`.
The ARM std log10 observed true-error upper bound is 1.446035 ULP in this sample.
Compiler/libc/OS are confounded; this does not isolate ISA causality or create a
uniform std guarantee. Other sampled std scalar cells agreed across targets.
Micromath power also differs on ARM specials. Full correctly rounded candidates
and pinned libm/fpmath agree across the checked scalar target comparisons.

## Algebraic findings

Stable algebraic methods produce only `reassoc nsz arcp contract` flags.
The scalar algebraic V3 reduction and array-map polynomial become packed
instructions. Plain vector Add/Mul stays unflagged. Actual timed thin-LTO
binary disassembly confirms the key V3/NEON patterns, not merely a separate
non-LTO probe: [CODEGEN.md](CODEGEN.md).

Intel V3 automatic algebraic dot measures 0.026155 versus 0.374579 ns/value for
the ordinary ordered loop. Counterexamples matter: NEON array dot costs
0.349417 versus 0.092628 explicit fused; its main loop has ten lane shuffles
and eight two-lane additions per 16 values, no loop spills/calls. Intel array
versus fused is a near tie (paired delta [-0.000320, 0.002096] ns/value).
Ryzen V4x algebraic polynomial is slower than ordinary, with paired delta
[0.002288, 0.032111] ns/value. No uniform algebraic or array-map recommendation.
Cancellation snapshots show ordinary 1, automatic/array algebraic 512 and
explicit vector 0 on the documented repeated input. No result bound is inferred.

## Reproduction and measurement limits

Use the directory justfile, runner scripts and retained per-build manifests.
Native timings use baseline compilation plus explicit proofs, one whole-buffer
feature region, interleaved zenbench 0.1.11, matched inputs and observed outputs.
Intel uses an enforced performance-core cpuset; timing on Neoverse uses a quiet
machine separate from its build machine. Builds have per-job resource budgets;
zenbench admission controls benchmark isolation. No heavy work ran on dev.

Native cases allocate inputs/output outside timed iteration. WASM matrix_kernel
**constructs case vectors, inputs and output per host invocation**, amortized
over 256 repeats of 1,024 values. Algebraic WASM wrappers do likewise. Native and
WASM therefore have different wrapper costs; within-platform comparisons retain
their stated workload, while absolute native-vs-WASM comparisons include that
wrapper difference. Host calls and module setup are not per-vector kernel costs.
The WasmBench engine is locked 49.0.5; strict SIMD is enabled without relaxed SIMD.
Numerical engine-validation details are in ACCURACY.md.

Every CSV retains rounds, MAD and paired 95% time-delta intervals versus its
baseline. Single-process intervals are not between-run reproducibility claims;
some cells show substantial drift/outliers, preserved rather than hidden.
No rerun was made solely to chase a favorable close result. The first 3-second
wall-budget passes had empty cells; those are archived as invalid/incomplete,
not accepted as zero-time measurements. Final runs use 30-second group budgets.

Actual final guide/reductions Rust fences passed native default, AVX-512 and
strict WASM (three actual examples per configuration). All four repository
health commands and scoped native/ARM/WASM lints passed; full logs and tested
doc hashes remain in [ARTIFACTS.md](ARTIFACTS.md).

Windows is unmeasured after a fresh 2026-11-06 probe: the existing Windows
cmd.exe is present, but `timeout 15s /mnt/c/Windows/System32/cmd.exe /c ver`
returns rc 1 with WSL UtilAcceptVsock accept4 failure 110. The exact command,
full output and host observation are retained privately in the archive.
No OS restart or mac connection was attempted.

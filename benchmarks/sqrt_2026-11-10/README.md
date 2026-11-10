# Scalar sqrt performance and portability

Scope: improve the scalar f32/f64 sqrt route while preserving the existing
correctly rounded result contract. Native SIMD sqrt implementations and all
published signatures stay unchanged. No approximate sqrt profile is introduced.

The pre-change software controls are the unchanged public
`magetypes::nostd_math::{sqrtf,sqrt}` at main `bd83f034`. ScalarToken vectors,
single-lane wrappers and their exact rsqrt compositions previously used them
unconditionally, even when the compilation target had hardware sqrt.

Acceptance checks:

- Independent exact-rational squared-midpoint fixtures, hardware/std and the
  retained software controls agree on non-NaN output bits. Signed zeros remain
  bit exact; NaN classification is checked without promising payload/sign bits.
- Check scalar and native vector widths with std/no_std; include subnormals,
  exponent boundaries, large values, infinities and negative inputs. Keep the
  f32 streaming sweep constant-memory; stride one is explicit opt-in.
- Verify no_std/MSRV builds and inspect baseline and native assembly. Retain
  software fallback where the build cannot safely select hardware.
- Use zenbench interleaved 50-round comparisons, requiring every retained round
  to pass admission. Compare named software, scalar x1, scalar packed and std
  paths on identical data; report ns per 64 values or dependent steps. No
  target-cpu=native flags. Preserve raw results and resource-wrapper records.

The numerical contract assumes round-to-nearest and gradual underflow. Changing
FP control registers is outside it. NaN payloads/signs and floating exception
flags are not cross-backend contracts. Correctness observations do not establish
performance, and sampled f64 tests do not constitute an exhaustive proof.

## Result

The std scalar route now calls the standard correctly rounded sqrt. This reaches
ScalarToken vectors (all enabled widths), single-lane wrappers and scalar exact
rsqrt compositions. Native SIMD is unchanged. The public integer software
functions remain unchanged as the no_std fallback and regression controls.
No public signature, dependency of the shipped library, or accuracy bound changes.

The owner approved disclosed internal summons on 2026-11-10. The no_std
selector now uses SSE2, NEON or WASM SIMD128 only where target features prove
availability. Without that proof it keeps integer software. In ordinary builds
these summons are constants; `archmage/testable_dispatch` can disable them and
exercise the software fallback. No runtime CPU upgrade occurs. WASM sqrt does
not require relaxed-SIMD. The std path continues to use primitive sqrt directly.

The public `nostd_math::{sqrtf,sqrt}` functions remain software controls. The
[WASM relaxed support audit](../../docs/WASM-RELAXED.md) records token testing,
existing FMA selection and the proposed operation-specific relaxed contracts.

## Correctness and codegen

- All **4,294,967,296 f32 input bit patterns** pass the streaming comparison of
  retained software, the selected scalar wrapper and hardware/std on Ryzen 9
  9950X3D. Non-NaN bits, including signed zero, are exact; NaNs compare by class.
  This run uses stride one and constant memory, not a materialized input array.
- Independent rational fixtures pass for f32/f64 across scalar and native vector
  widths on x86-64, Apple M6 NEON and strict SIMD WASM, with and without std.
  Default tests retain directed cases and a strided f32 sample; full f32 replay
  is `just sqrt-check-exhaustive`. f64 is directed/sampled, not exhaustive.
- Rust 1.89 checks the std all-target set and no_std library. Strict Clippy,
  generation/soundness checks and focused Miri rational-fixture checks pass.
- Baseline x86-64 scalar probes lower to `sqrtss; ret` and `sqrtsd; ret`.
  LLVM merges the new scalar loops with the direct-std loops; no arithmetic
  policy decoder, software iteration or runtime CPU upgrade remains there.

Production routing was introduced at `42a31313`; fixtures and the initial
harness at `9dbb5a83`; the corrected benchmark dependency at `57b4aedf`.
Later fixture-gate, documentation and explicit benchmark measurement-opt-in edits leave
the timed kernels unchanged; reported timing binaries retain the source
identities above. The final measurement tool is a workspace-excluded package here, leaving the
workspace's existing zenbench dependency and Cargo.lock unchanged. Its own
manifest uses 0.1.11. This avoids changing existing benchmark admission behavior
or adding timing work to `cargo test --all-targets`. `just sqrt-bench-check`
checks its kernels without timing; `just sqrt-bench` explicitly opts into timing.
The relocation leaves timed kernel source unchanged; old binary/source identities
remain attached to the recorded observations.

## Timing observations

These are **single-run exploratory comparisons**, not a whole-matrix promotion.
Every row has 50 interleaved rounds. Admission requires all 50 clean rounds;
only the two bold clean counts meet that cell condition. The complete runs are
flagged unreliable, so no full-suite performance gate is claimed passed.

Times are mean **ns per 64 values** for stream rows; chain rows are **ns per 64
dependent sqrt-plus-0.5 steps**. They are not individual-call latency estimates.
`software` means the unchanged public integer kernel. `scalar x1` means the
new wrapper using std sqrt. `std` is a direct primitive sqrt in the same loop.

| CPU | Workload | Software ns | Scalar x1 ns | Direct std ns | Clean rounds |
|---|---|---:|---:|---:|---:|
| Ryzen 9 9950X3D | f32 stream | 180.35 | 12.62 | 12.60 | 29/50 |
| Ryzen 9 9950X3D | f32 chain | 538.76 | 108.75 | 107.51 | 16/50 |
| Ryzen 9 9950X3D | f64 stream | 247.88 | 42.26 | 42.59 | 40/50 |
| Ryzen 9 9950X3D | f64 chain | 709.04 | 152.30 | 151.44 | **50/50** |
| Ryzen 9 7900X | f32 stream | 232.50 | 14.27 | 14.29 | 35/50 |
| Ryzen 9 7900X | f32 chain | 595.91 | 205.76 | 206.49 | 40/50 |
| Ryzen 9 7900X | f64 stream | 303.18 | 49.06 | 49.26 | **50/50** |
| Ryzen 9 7900X | f64 chain | 785.73 | 276.66 | 276.40 | 44/50 |
| Apple M6 | f32 stream | 81.19 | 6.50 | 6.52 | 0/50 |
| Apple M6 | f32 chain | 602.64 | 52.61 | 52.56 | 0/50 |
| Apple M6 | f64 stream | 108.33 | 13.23 | 13.23 | 0/50 |
| Apple M6 | f64 chain | 746.91 | 71.15 | 71.19 | 0/50 |

The 9950X3D run used an enforced cpuset confined to its 32 MiB L3 class;
7900X reported uniform hardware. Foreign process activity flagged the remaining
Linux cells. M6 ran with the explicit unrestricted exploratory policy because
macOS cannot enforce core-class confinement; all its rounds were also CPU-load
flagged. The strict refusal and every invalid attempt remain preserved.

Example zenbench bars from the all-rounds-clean 9950X3D f64 chain cell:

```text
scalar x1/std sqrt  ███████████                             152.3 ns
integer software   ██████████████████████████████████████████████████ 709.0 ns
```

No clean Intel, Neoverse or WASM timing comparison was collected. Native SIMD
bodies are unchanged; this study does not establish a new application-level
speedup or qualify every scalar workload under the 2% promotion bound.

Resource records (validation or benchmark wrapper totals, not per-lane memory):

```text
x86 std/no_std + generation + MSRV + benchmark build:
run-heavy: done rc=0 45s | peak-RSS 0.55GiB | min-avail 41116MiB | peak-load 1.69
WASM checks + x86 codegen + complete f32 sweep:
run-heavy: done rc=0 12s | peak-RSS 0.34GiB | min-avail 41982MiB | peak-load 0.70
9950X3D cpuset timing attempt:
run-heavy: done rc=101 7s | peak-RSS 0.25GiB | min-avail 41821MiB | peak-load 0.71
7900X timing attempt:
run-heavy: done rc=101 9s | peak-RSS 0.09GiB | min-avail 17059MiB | peak-load 0.50
M6 exploratory timing attempt:
run-heavy: done rc=101 6s | peak-RSS 0.06GiB | min-avail 10061MiB | peak-load 1.91
```

The Mac wrapper uses a monitored process-tree RSS guard and a reclaimable-page
proxy, not Linux cgroups/MemAvailable. Raw JSON, full logs, exact binary/source
identities and assembly live under `/mnt/v/output/archmage/sqrt-study-2026-11-10`,
mirrored beneath `/mnt/tower/output/archmage` with a SHA-256 manifest.


Admission setup correction: initial zenbench 0.1.10 runs counted benchmark CPU
as foreign activity. They remain retained and unqualified. The measurement
harness now uses published zenbench 0.1.11 (source `983eafaa49ce`) with owned
launcher registration and trace evidence. All thresholds, 50 retained rounds
and the all-rounds-clean requirement stay unchanged. Both numerical paths are
interleaved under the corrected harness; production math is unchanged.

## Approved assembly expectation

CI at `4deedd1c` rejects the intentional scalar sqrt assembly changes. An
independent inventory of all 3,211 probes finds exactly six changed x86 probes;
the other 3,205 are identical, including all 794 ARM and all 794 WASM probes.
These are expanded static instruction entries, excluding labels, not dynamic
work estimates (the old wide software kernels contain loops).

| Changed x86 probe | Before | After |
|---|---:|---:|
| Scalar f32x4 sqrt | 274 | 5 |
| Scalar f32x8 sqrt | 88 | 8 |
| Scalar f32x16 sqrt | 89 | 14 |
| Scalar f64x2 sqrt | 177 | 5 |
| Scalar f64x4 sqrt | 116 | 8 |
| Scalar f64x8 sqrt | 117 | 14 |

The new bodies contain only pointer moves, contiguous vector loads/stores,
`sqrtps`/`sqrtpd` and return. No std work is hidden outside these counts.
The owner approved replacing old-software equality with independently generated
direct-std sqrt references for exactly these six probes on 2026-11-10. The gate
now requires exact expanded-body equality to those references, even when the
before/after implementations agree. Reference functions must be present on both
sides and cannot call the SIMD sqrt method or scalar selection helper.

The full gate passes: six reviewed x86 differences and zero ARM/WASM differences;
all other 3,205 primary probes are identical. Adversarial checks reject missing
references, wrong sqrt precision/operations, wrong memory offsets, missing stores,
extra calls and any widening of the six-probe/target scope. Every numerical bound
and all other assembly expectations remain intact. No production arithmetic or
timed kernel changed in this gate update.

```text
run-heavy: done rc=0 20s | peak-RSS 0.41GiB | min-avail 41772MiB | peak-load 0.50
```

The complete timing runs are still flagged unreliable as described above; this
assembly approval does not promote unqualified timing cells.

Check the approved gate and inventory at the PR head:

```sh
just codegen-diff bd83f034 HEAD # approved std-reference checks at the current head
python3 -B benchmarks/sqrt_2026-11-10/codegen_inventory.py RUN_DIRECTORY OUTPUT_JSON
```

The full failing-gate logs, expanded instructions and reference experiment are
preserved in `sqrt-study-2026-11-10-followup` beneath the same local/mirrored
output roots. The first evidence archive remains immutable.

Approved-gate logs, generated references, assembly, source snapshot and SHA-256
index are preserved in `sqrt-study-2026-11-10-approved-gate` beneath the same
local/mirrored output roots. Earlier evidence archives remain immutable.

## no_std and WASM follow-up (2026-11-10)

The disclosed no_std selector passes all 4,294,967,296 f32 inputs against
hardware/std and the retained integer control (7.78 seconds for the streaming
comparison). F64 remains directed/sampled. Native Apple M6 no_std fixture/width
checks and enabled/disabled/re-enabled scalar checks also pass. The fixed WASM
matrix covers no-SIMD, SIMD128 and relaxed-SIMD builds, std/no_std, baseline and
relaxed token permutations, and fusing/nonfusing Wasmtime configurations.

Compile-only no_std probes contain no token load, branch or software call:
x86-64 uses `sqrtss/sqrtsd` and `sqrtps/sqrtpd`; AArch64 uses scalar/vector
`fsqrt`; relaxed-WASM emits scalar `f32.sqrt/f64.sqrt` and vector
`f32x4.sqrt/f64x2.sqrt`. Relaxed FMA selectors retain their existing fusion probe;
the extra disabled-token check is compiled away outside `testable_dispatch`.
No new timings for no_std or the other proposed relaxed families were collected.

Public API snapshots were regenerated for the added WASM disabling methods.
Generation, structural soundness, focused Clippy, Rust 1.89 no_std native/WASM
checks and the full 3,211-probe comparison pass. The latter still has exactly the
six approved x86 differences. No numerical or assembly bounds were relaxed.
A broader optional WASM all-tests Clippy invocation reported two `single_match`
lints in the unchanged `int_widen_narrow` test; focused affected tests are clean.

The native M6 correctness/build wrapper reports:
`run-heavy: done rc=0 7s | peak-RSS 0.37GiB | min-avail 8696MiB | peak-load 2.01`.
Tool/codegen/workflow checks report:
`run-heavy: done rc=0 5s | peak-RSS 0.33GiB | min-avail 39895MiB | peak-load 0.79`.
These are complete wrapper runs, not per-operation memory measurements.

Source snapshots, all attempt logs (including setup failures), assembly and
manifest are retained in `sqrt-study-2026-11-10-no-std-wasm` beneath the same
local and mirrored output roots as the preceding evidence archives.

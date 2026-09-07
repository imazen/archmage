# ARM pairwise widening co-development

Tested on Apple M4 Pro with rustc 1.98.0 / LLVM 22.1.8, without
`target-cpu=native`. Archmage implementation: `bdab9808`, based on
[PR #96](https://github.com/imazen/archmage/pull/96) at `9981d4f9`.
The matching local [zenav1-svt](https://github.com/imazen/zenav1-svt) change is
`03e089b5`, based on `74d92430`. These are development revisions, not crate releases.

The experiment supports one new operation name, `pairwise_widen_add`, for
unsigned byte and halfword vectors. Each output lane sums two adjacent input
lanes after widening; the full unsigned input range is exact. Native and
polyfilled widths preserve the same lane order. Existing widening,
multiplication, absolute difference and addition express the other operations.

The ARM u8 implementation emits UADDLP; adding it to a u16 accumulator emits
UADALP. The u16/u32 form behaves the same way. Tests cover every possible byte
pair at each lane, plus lane-distinct full-range halfword patterns and width
boundaries. The codegen gate exercises both primitive and accumulation forms.

`just integer-codegen` checks 44 new comparisons: 20 x86, 12 ARM and 12 WASM.
42 bodies match their independently written intrinsic references exactly.
Two WASM W512 accumulation bodies commute integer addition operands: the gate
checks exact symbolic load/store dataflow, lane widths and signedness, and
identical opcode counts. Adversarial parser tests reject changed input/address,
operation width, signedness, store destination and incomplete stack effects.
No pre-existing codegen expectation or arithmetic assertion was removed.
[Per-probe results](arm_pairwise_codegen_2026-09-07.json) include instruction counts.

Three SVT NEON kernels consume the method: `variance::sse`,
`me_sad::block_sad_neon`, and `me_sad::block_sum_sse_neon`. The stronger dotprod
SAD arm stays separate. These consumers retain their accumulation schedules,
strides, tails and narrow-block handling.

42 same-binary zenbench comparisons covered three kernels, seven block shapes
from 4x4 to 128x128 (including 17x9 tails), and tight/unequal padded strides.
They show no meaningful throughput improvement over handwritten NEON. The
final SAD and sum/SSE executable bodies have identical instructions and local
branch targets through their returns (185 and 231 instructions respectively).
This is evidence for zero-cost API coverage, not a claimed encoder speedup.
Results apply to the measured M4 Pro; other ARM microarchitectures were not timed.

A second experiment packed two narrow rows into registers without a staging
buffer. Its four 4x4/8x8 tight/padded comparisons showed no useful improvement,
so it remains only in the benchmark reference module. It requires no extra API.

Consumer evidence is retained in zenav1-svt's
`rust/benchmarks/arm_pairwise_2026-09-07.{meta,txt}`,
`arm_pairwise_codegen_2026-09-07.json`, and `arm_rowpack_2026-09-07.txt`.
The metadata records exact baseline/snapshot revisions, methodology, commands,
local-patch requirements and the known C-submodule checkout discrepancy.

The measured consumer does not justify adding widening-multiply, square,
signed-pairwise or MAC/MSUB operations. PR #96's existing `msub_adjacent`,
signed `abs_diff`, `reduce_add_u32` and `sum_abs_diff` still have no calls in
these production SVT kernels; that is a scope argument, not proof that the
operations have no other consumers. Their removal was not part of this change.

SVT correctness passed 2587/2587 workspace tests and 106/106 regression
spot-checks. Strict Clippy is blocked by baseline diagnostics, including an
ARM dotprod/MSRV mismatch (the manifest declares 1.89, Clippy reports the
existing dotprod intrinsics stabilized in 1.98). Each reported source site was
verified unchanged from `74d92430`; the complete DSP diagnostics are retained
with the consumer measurements. These results are not merge certification.

Archmage verification: generation/idempotence, token validation, intrinsic
soundness, API parity, verifier tests, strict Clippy, root tests, no_std tests
and bare-metal builds, formatting, regenerated API snapshots and rustdoc all
passed (`just ci` steps 1–14). Step 15 stops at the repository's documented
Miri limitation: unsupported `llvm.aarch64.neon.fcvtns`, exercised by the
existing RGBA block-ops test. This is not a reported arithmetic mismatch or
undefined-behavior finding; the xtask's generic error label is misleading.
No test was ignored or weakened to bypass it. Full log:
`~/tmp/archmage-arm-pairwise-ci.log`.

The full native magetypes suite with `std avx512` also passed (1061 tests,
11 pre-existing ignored tests). The integer suite executed under Wasmtime
with SIMD128 enabled: all 3 tests passed, including every new byte-pair and
halfword case on the real WASM backend and scalar backend, all three widths.
Log: `~/tmp/archmage-arm-pairwise-native-wasm.log`. x86 instruction references
were compiled and compared; x86 hardware execution was not performed here.

# Linked codegen evidence

Full disassembly and extracted functions live in the external artifact archive
listed in [ARTIFACTS.md](ARTIFACTS.md). `extract-codegen.py` records exact symbol,
address, source disassembly line and SHA-256 per extract. These are linked timing
executables, not a separate non-LTO approximation.

The original native timing executable has SHA-256
`706088a3d594fd676a4e71cb5b44198b1ac952b98ae02657dca11b1e0c22c7b3`.
It was overwritten in Cargo output before preservation; rebuilding the frozen
source/compiler/lock/profile at the original path recovered exactly those bytes.
The recovered executable is retained outside Cargo output, byte-verified locally
and on the NAS. The recovery log and supplemental identity preserve this history.

| Native symbol in `native/codegen/timing.disasm` | Address | Line | Observation |
|---|---|---:|---|
| `generated::exp_f32_stream_context_v3` | `0x1f1560` | 405979 | Full scalar algorithm in loop; calls shown at the end are assertion paths |
| `generated::ln_f32_stream_context_v3` | `0x1edf50` | 402549 | Full scalar algorithm, not baseline forwarding |
| `generated::log2_f32_stream_context_v3` | `0x1f8d30` | 413658 | Full scalar algorithm, not baseline forwarding |
| `generated::pow_f32_stream_context_v3` | `0x1f3b70` | 408370 | Native `vfmadd132sd` at `0x1f3fb7`, `vfmadd213sd` at `0x1f40ef`; accurate fallback calls remain |
| `generated::floor_f32_stream_context_v3` | `0x1fd910` | 418441 | Unrolled scalar `vroundss`, not packed auto-vectorized floor |
| `zenmath::simd::slice::exp_f32_v3` | `0x20ddc0` | 434693 | Packed YMM arithmetic |
| `zenmath::simd::slice::exp_f32_wide_v4x` | `0x2120f0` | 438745 | Actual packed ZMM arithmetic |
| `zenmath::simd::slice::pow_f32_wide_v4x` | `0x212540` | 438988 | ZMM common arithmetic, array helpers and accurate/special fallback remain |

Prefix `generated::` above means `zenmath_dispatch_study::generated::`.
No full-function call-free claim is made. Slice kernels retain scalar rescue,
length assertions, tails and, in some wide power/table handling, helper calls.
Register width alone does not predict speed. The std floor context is an important
semantic-intrinsic/vectorization control; explicit scalar lowering is not always
best for a full loop.

The native facade timing executable has its own manifest under
`native/frozen-facade`. In its disassembly, `fma_f64_feedback_branch` at
`0x1f2890` (line 403573) and `fma_f64_feedback_ordinary` at `0x1fc660`
(line 413774) have the same hot-loop instructions and call the same
`generated::fma_f64_branch` at `0x1e5150`. Their observed feedback difference
was 0.467 ns/step, CI [0.122, 0.830], despite this common arithmetic path.
Layout/sampling differences therefore cannot be treated as evidence of a different
FMA implementation. Assertions and addresses differ; exact-byte identity is not
claimed. The ordinary facade still includes the measured feature branch cost.

The original ARM timing executable SHA-256 is
`4a512da79dc07b8150fd8fc884508734d0494708585bfac838e601b112d8a104`.
In `arm/codegen/timing.disasm`, `generated::pow_f32_stream_context_neon`
at `0x14eb1c`, line 322098, contains native `fmadd` at `0x14eecc` and
`0x14efac`; the accurate fallback calls `is_exact`/`accurate` at `0x14f04c`
and `0x14f058`. `zenmath::simd::slice::pow_f32_neon` at `0x155cd8`,
line 329555, contains packed NEON arithmetic on paired f64 lanes.
Exp/ln/log2 complete scalar context extracts are indexed alongside it.

WASM codegen is retained as the actual timed module plus `wasm/codegen/wasm-kernel.wat`.
It contains strict `f64x2.mul`/`f64x2.add` packed operations and exact software FMA;
no relaxed SIMD rounding is enabled. The guest uses SIMD128, link-dead-code and
export-all with LTO disabled to expose the safe Rust export, following the prior
harness. Native timing uses thin LTO; that difference is explicit provenance.

The original full WASM run used guest SHA-256
`5bee7e91afe13ca6bda6bc2c057d1c5a8b0dc796689f16587f6006b74c6ce9d3`
and guarded host
`5e51ef69cd69883382987087d65356c34572f429a9bde579cfce8a70d71531e8`.
The WAT above describes that exact guest; its WAT SHA-256 is
`bc34203c4a84bbc475503d8d89be8e6b235aef7b77944d7986f9f608f2e1d2ee`.
The final facade uses separately frozen fresh-target v2 modules and tests,
identified in ARTIFACTS; neither is relabeled as the full-run module.

# Algebraic lowering: actual optimized IR and assembly

No production API or arithmetic changes. The probes use stable scalar methods
inside whole-loop Attune contexts. Each adapter asserts matching lengths and
fixed-width divisibility. The polynomial is the cubic Taylor exp arithmetic
`((x/6 + 1/2)*x + 1)*x + 1`; its leading coefficient is nontrivial in binary.
This measures arithmetic and makes no exp approximation accuracy claim.

The complete source for measured binaries is frozen separately per build.
Native x86 uses Rust 1.99.1/LLVM 23.1.1; ARM uses 1.99.0. No target-cpu=native.
For codegen, `CARGO_PROFILE_RELEASE_LTO=false` is essential: thin-LTO emitted
IR is prelink, before final vectorization, and was retained but not used to
claim packed lowering. Actual timings retain the release thin-LTO setting.
Codegen explains transformations but is not byte-identical proof of the
thin-LTO linked timing executable. Full artifact paths and compiler identities
are in [ARTIFACTS.md](ARTIFACTS.md).

## Flags and feature contexts

All observed scalar algebraic arithmetic uses exactly
`reassoc nsz arcp contract`, **without afn/nnan/ninf**. Ordinary Magetypes vector
Add/Mul has none of those flags. V3 auto algebraic dot emits `<8 x float>` and
`llvm.vector.reduce.fadd.v8f32`; V4x f64 emits `<8 x double>` and its reduction.
Ordinary V3 dot retains a scalar ordered addition chain.

The [V3 array polynomial assembly](codegen/v3-array-polynomial.s) contains one
vector load, three packed FMAs and one store per 8 values in its hot loop.
There are no hot-loop calls or stack accesses. Its assertion frame is outside
the loop: do not describe the entire function as stack-free. The
[ordinary vector control](codegen/v3-ordinary-polynomial.s) uses separate
vmulps/vaddps. Explicit fused vector controls use llvm.fma and packed FMA
instructions, isolating contraction from reassociation.

## NEON counterexample: array reduction

The public helper inlines into a closure. The exact mapping, original assembly
filename/line and symbol are recorded in [index.json](codegen/index.json).
The complete extracted function is [arm-array-dot.s](codegen/arm-array-dot.s),
from `math_matrix_study-107ee31fd7b8ef38.s`, symbol
`_RNCNvNtCsffbGKKrTuqY_17math_matrix_study9algebraic9cases_f32sd_0B5_`,
IR identity `cases_f32::{closure#15}`.

Its main loop `.LBB20_7` processes 16 values with four fmul .4s, ten zip/uzp
lane rearrangements and eight fadd .2s accumulators. No hot-loop stack access
or call appears. This is packed arithmetic, not scalarization or spilling.
LLVM interleaved successive outer iterations and transposed the per-lane
accumulators; the resulting main loop does not contract multiply/add.
The tail loop does use FMA, which is why whole-function FMA counts alone
would misdescribe the measured 1,024-value main path.

[Explicit fused dot](codegen/arm-fused-dot.s), closure#19 `.LBB24_5`, uses two
fmla .4s per 8 values and two vector accumulators. [Auto algebraic dot](codegen/arm-auto-dot.s),
closure#2 `.LBB8_7`, uses four fmla .4s per 16 values/four accumulators.
The measured NEON array dot is 0.349417 ns/value versus 0.092628 explicit fused
and 0.090994 automatic algebraic. The array-minus-fused paired 95% interval is
[0.237268, 0.281232] ns/value. The extra lane rearrangement/accumulation is an
assembly-supported explanation, not a causal-profiler attribution.

## Measured limits

[Intel V3 array dot versus fused](results/algebraic-paired.csv) is a near tie:
0.057578 versus 0.056955 ns/value, paired delta interval [-0.000320, 0.002096].
Ryzen V4x ordinary polynomial is 0.050507 versus 0.067319 algebraic; the
algebraic-minus-ordinary interval is [0.002288, 0.032111] ns/value.
No uniform array-map or algebraic speed recommendation follows.

The direct paired comparisons use every retained round and 10,000 seeded
bootstrap resamples, separately labeled from zenbench's own outlier-filtered
baseline comparisons. Single-run intervals do not measure between-run variance.

## Changed reduction results

The source repeats `[1e20_f32, 1, -1e20_f32, 1]` 256 times and multiplies by 1.
On measured V3, ordinary ordered reduction returns 1 (`3f800000`), automatic
and array algebraic return 512 (`44000000`), and explicit vector/fused reduction
return 0. The real sum of these exactly representable inputs is 512. This is an
observed cancellation example, **not an algebraic error bound or determinism
promise**. Mixed-magnitude and f64 snapshots are retained in timing logs.

## Strict WASM and compiler versions

[WASM array polynomial](codegen/wasm-array-polynomial.s) and
[array dot](codegen/wasm-array-dot.s) show packed f32x4 arithmetic with ordinary
strict SIMD multiply/add. Algebraic flags permit contraction, but strict SIMD
has no fused instruction. No relaxed-SIMD instruction is enabled.
`algebraic-basic.rs` using all five f32/f64 methods fails to compile on installed
Rust 1.89.0 and succeeds on 1.98.1. The coordinator owns the MSRV recommendation.

## Actually timed thin-LTO executable check

The important loop patterns above also survive in the actual measured executables:

- ARM binary SHA256 `8c42854b11baa88cda6765831600ed4fab799bf02a0616cb2b3acbdbbbc37563`.
  [Array-dot disassembly](codegen/timed-arm-array-dot.txt), exact demangled
  `math_matrix_study::algebraic::cases_f32::{closure#15}` at `0x2c99c`:
  main loop `0x2ca24`–`0x2ca90` retains four vector multiplies, ten shuffles,
  eight two-lane additions, and no calls/stack access.
  [Fused-dot disassembly](codegen/timed-arm-fused-dot.txt), closure#19 at
  `0x2cfa0`, loop `0x2cff0`–`0x2d00c`, retains two vector FMAs.
- x86 binary SHA256 `8a3ef4d37bc2cc5f158512e87411525649532815f4a7257453539b8692fbe16a`.
  [V3 array-polynomial disassembly](codegen/timed-v3-array-polynomial.txt),
  `math_matrix_study::algebraic::__attune_poly_array_algebraic_f32_v3` at
  `0x7e520`, loop `0x7e5b0`–`0x7e5fb`, has six packed FMAs per 16 values,
  two input loads/two stores and no loop call or stack access.

Full objdump output, nm symbol listings and exact binary hashes are retained
under each platform's `timed-codegen/` artifact directory. Thus the measured
ARM slowdown explanation is backed by the actual linked timing executable,
while broader non-LTO IR observations retain their stated limitation.

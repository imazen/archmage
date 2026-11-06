# Optional Archmage acceleration

Zenmath's default scalar core remains dependency-free, no_std, Rust 1.89 and
forbid unsafe. The optional `archmage` feature adds intrinsic-only packed math.
There is no Magetypes dependency and no Archmage-to-Zenmath dependency.
`runtime-dispatch` explicitly enables native CPU selection and implies std;
`avx512` compiles additional proof entries. These are implementation choices,
not numerical policies: unifying features must not change contracted bits.

`simd::array::{exp_f32x4, ln_f32x4, log2_f32x4, pow_f32x4}` and corresponding
`simd::slice` operations share the previous Magemath CORE-MATH common paths.
Magemath's existing correct array, slice and power-policy APIs delegate to them.
The upstream MIT notices, polynomial order, fused points, ambiguity predicates
and per-lane scalar rescue are retained. Default scalar functions remain the
independently callable oracle; no fallback calls a dispatcher recursively.

Each operation has direct feature bodies (`*_v3`) and explicit proof entries
(`*_v3_t`, and NEON/WASM/scalar equivalents). Use a matching feature body to
compose a loop; the proof bridge enters from an ordinary context. Kernels do
not summon tokens or select an ISA. Four f32 values use four f64 intermediates:
256 bits on V3, two 128-bit registers on NEON/WASM. V4x currently reuses this
same four-lane kernel; its suffix does not imply a 512-bit algorithm.

Selection is outside the packed loop. Without Zenmath's own runtime feature,
`compiled_with() == Some(true)` is required before asking Archmage for a proof.
In that case Archmage documents that proof acquisition is constant-folded.
This explicitly prevents runtime probing when another dependency merely unifies
`archmage/std`. WASM selection relies on module features, never an engine probe.
Independent no_std consumers must check this behavior without test dependencies.

The scalar/WASM f64 fused software common path reuses Magetypes' whole-array
Dekker/TwoSum algorithm (source at b1316a4e). It combines rare-lane checks before
cold exact scalar rescue instead of branching through scalar FMA in each lane.
Hardware V3/NEON use native packed FMA. All paths assume nearest-even and gradual
underflow; NaN payloads/signs are unspecified, signed zero is part of the contract.

This is an implementation record, not a performance recommendation. Dispatch
strategy, actual wider candidates and full-domain cross-ISA hashes are still
being evaluated. No speed claim follows from replacing a dependency or passing
finite fixtures. The scalar auto-dispatch policy will follow measured data;
losing selector candidates stay experimental.

## Scalar instruction entries

`simd::scalar` supplies f32/f64 sqrt, floor, ceil, trunc, ties-even and ties-away
rounding, plus fused multiply-add. x86 sqrt uses baseline V1; rounding uses V2
SSE4.1; FMA uses V3. NEON uses the corresponding instructions. Strict WASM
rounding/sqrt uses SIMD128 instructions and fused FMA keeps exact software.
Ties-away rounding uses truncation plus a fractional comparison on x86/WASM;
it never adds 0.5 before truncating. Scalar direct contexts for the full four
CORE-MATH algorithms are also available; they call the actual scalar algorithm,
not replicated vector lanes. No production legacy scalar call is redirected yet.

Independent exact F32/F64 FMA fixtures and full-bit primitive samples passed on
native x86, including signed zero and subnormal cases. Targeted predecessor/
successor cases around ±0.5/±1.5 supplement random bits. Runtime-tier presence is
asserted by validation caller configuration. The separate compile consumer
unifies `archmage/std` without runtime dispatch: optimized consumer floor/exp
bodies contain no CPU selection, stronger-tier call or detector reference.
Default/accelerated/runtime compile and cold build costs remain to be measured.

## Wider proof candidate

With `avx512`, `array::*_f32x8_v4x(_t)` and `slice::*_f32_wide_v4x(_t)`
evaluate eight f32 inputs with eight binary64 intermediates in actual AVX-512
registers. They reuse the same const-width formulas and tables, not two V3
calls. The original four-input V4x entry still uses four binary64 lanes. No
512-bit automatic selection policy has been chosen from timings yet. Exact
mixed-bit samples and lengths 0 through 17 passed on native V4x, together with
legacy Magemath core/policy regressions, clippy and Rust 1.89 checking.

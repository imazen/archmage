# Optional Archmage acceleration

Zenmath defaults to dependency-free `no_std`, Rust 1.89 and `forbid(unsafe_code)`.
`archmage` adds intrinsic-only proof/context APIs. `runtime-dispatch` adds native
selection and implies `std` plus `archmage/std`. No Zenmath dependency on Magetypes,
or Archmage dependency on Zenmath, is introduced. Feature unification preserves
the numerical contracts.

`simd::array` and `simd::slice` share the existing Magemath accurate f32 exp, ln,
log2 and pow algorithms. Magemath delegates through proof adapters. Polynomial
order, fused points, ambiguity predicates, accurate fallback and upstream MIT
notices are retained. Explicit `*_v3` context bodies compose inside matching
feature functions; `*_v3_t` accepts a held proof. NEON, WASM and scalar equivalents
remain available. Kernels never probe. Tails are bounded; power checks both
lengths before mutation.

Four inputs use four f64 intermediates: 256 bits on V3, two 128-bit registers on
NEON/WASM. Four-input V4x keeps that width. With `avx512`, explicit
`*_f32x8_v4x(_t)` arrays and `*_f32_wide_v4x(_t)` slices use eight actual f64 lanes.
Automatic x86 array/slice selection prefers V3; the measured
[width comparison](../../benchmarks/zenmath_dispatch_2026-11-06/WIDTHS.md) records
per-operation paired intervals and limits. An ISA suffix is not a speed guarantee.

Without Zenmath's own runtime feature, selection requires
`compiled_with() == Some(true)` before proof acquisition. Another dependency
unifying `archmage/std` cannot enable runtime probing. An external consumer
verifies this in optimized IR. WASM uses module-load feature guarantees, never
engine detection.

The scalar/WASM f64 fused software path reuses Magetypes' whole-array
Dekker/TwoSum common path and combined rare-lane rescue. V3/NEON use native FMA.
All paths assume nearest-even rounding and gradual underflow. Signed zero is
preserved; NaN payloads and signs are unspecified.

## Ordinary scalar policy

Ordinary `scalar` names accelerate only with **`runtime-dispatch`**. With default,
`std` alone, or `archmage` alone, the facade retains its reference implementations.
The explicit proof/context APIs remain usable with `archmage` alone.

| Target | Ordinary accelerated names |
|---|---|
| x86-64 | `sqrtf`, `sqrt`, `floorf`, `floor`, `ceil`, `roundf`, `round`, `roundeven`, `fma`, `powf` |
| AArch64 | `sqrtf`, `sqrt`, `floorf`, `floor`, `ceilf`, `ceil`, `roundf`, `round`, `roundevenf`, `powf` |
| WASM | `sqrtf`, `sqrt`, `floorf`, `floor`, `ceilf`, `ceil`, `roundf`, `round`, `roundevenf`, `roundeven`; SIMD128 module guarantee, reference fallback in strict no-SIMD modules |
| Other targets | Reference facade |

Other names retain reference code. In particular, f32 FMA and scalar exp/ln/log2
had mixed results or regressions; x86 ceilf and roundevenf are not redirected.
These are workload-informed choices, not claims that dispatch always wins.
Single-value calls include wrapper/indirect benchmark costs and frequently tie.
WASM FMA and transcendentals retain reference code, including the power rescue
mixture that regressed in the measured context. The simpler cached-feature branch is a design tie-break where measurements do
not distinguish selectors. Safe cached pointers won some x86 FMA comparisons but
had no universal advantage; they remain benchmark-only and carry the real proof,
including its match cost. They do not implement IFUNC.

`scalar::reference` exposes the original non-dispatch algorithms. Internal
polynomial steps, vector rescue and policy fallback call private reference paths,
never the facade. Historical compile-time hardware FMA under `std` is retained
inside that reference. Thus “non-dispatch” does not mean “always software”.

## Explicit primitives and context bodies

`simd::scalar` supplies f32/f64 sqrt, floor, ceil, trunc, roundeven, ties-away
round and FMA. x86 uses V1 sqrt, V2 rounding and V3 FMA; NEON uses native
instructions. Strict WASM has SIMD sqrt/rounding and exact software FMA.
Ties-away rounding avoids the double-rounding error of adding 0.5 first.

The full scalar exp/ln/log2 bodies inline in their feature contexts. Pow's private
arithmetic strategy specializes its original two fused operations and floor;
linked V3/NEON code contains native FMA and no per-polynomial-step CPU probes.
Rare accurate pow rescue retains the original software algorithm. Scalar contexts
do not replicate a scalar value into a vector to disguise its cost.

Scalar intrinsics need not optimize like semantic Rust builtins. The linked V3
floor loop uses scalar `vroundss`; the std context control can vectorize better.
Use the measured controls, not an assumption that explicit instructions are best.

## Evidence and limits

The [dispatch study](../../benchmarks/zenmath_dispatch_2026-11-06/README.md)
records native x86, NEON and strict WASM throughput/feedback controls, direct
paired intervals, linked codegen, cold build costs and artifact identities.
The separate [digest study](../../benchmarks/zenmath_digests_2026-11-06/README.md)
validated 89 full-domain cells against production kernels frozen at `5a9cc685`.
Final facade and selector changes alter routing only; direct final tests cover
those changes. The frozen kernel sweeps are not labeled as sweeps of later facade
source.

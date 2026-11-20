+++
title = "Magetypes Fast Paths & Contract Tiers"
description = "Skipping unneeded saturation, rail repairs, NaN blends, and software FMA directly on Magetypes vectors"
weight = 2
+++

The same principle behind [Magemath's domain and validation knobs](@/magemath/tuning-and-subnormals.md) applies to primitive SIMD operations in `magetypes`. Several operations require multi-instruction compare/blend sequences on x86 or WASM to match Rust's scalar edge-case rules (`NaN`, `±0.0`, `±inf`, integer overflow), while single-instruction hardware forms exist when your caller has already bounded the inputs.

## Summary of contract tiers and measured costs

| Operation family | Strict / portable method | Fast / relaxed method | Repair skipped | Edge-case change | Measured difference |
|---|---|---|---|---|---|
| Float → `i32` | `v.to_i32_saturating()` | `v.to_i32_fast()` / `v.to_i32()` | Overflow compare + `i32::MAX` select + `NaN` zeroing on x86 (`relaxed_trunc` on relaxed WASM) | Overflow & `NaN` return `i32::MIN` (`0x8000_0000`) on x86 instead of `i32::MAX` / `0` | Portable costs **`+84.2%`** on x86-v3 `f32x4`, **`+44.7%`** on x86-v4 `f32x16` (Ryzen 9 7950X); `0%` on M4 Pro NEON |
| Reciprocal `1 / x` | `v.recip_portable()` (`0 ULP` div) | `v.recip()` (`<= 4 ULP` + rails) → `v.rcp_approx()` (~12-bit) | Division → Newton + rail blend → raw `vrcpps` / `vrecpe` | `recip(f32::MAX)` flushes subnormal output to `0.0` on x86 V3; `rcp_approx` leaves `±0`/`±inf`/`NaN` unspecified | `1,800 ns` (`_portable`) vs **`1,400 ns`** (`recip`) vs **`~830 ns`** (`rcp_approx`) per `32K` lanes on Zen 5 V3 (`+31.8%` rail repair on 7950X V3 `f32x4`, `0%` on NEON) |
| Reciprocal sqrt `1 / √x` | `v.rsqrt_portable()` (`0 ULP` div+sqrt) | `v.rsqrt()` (`<= 4 ULP` + rails) → `v.rsqrt_approx()` (~12-bit) | `sqrt` + `div` → Newton + rail blend → raw `vrsqrtps` | Subnormal inputs unspecified on `rsqrt`; all rails unspecified on `rsqrt_approx` | `5,400 ns` (`_portable`) vs **`1,900 ns`** (`rsqrt`) vs **`830 ns`** (`rsqrt_approx`) per `32K` lanes on Zen 5 V3 (`+26.7%` rail repair on 7950X V3 `f32x4`, `0%` on NEON) |
| Min / Max / Clamp | `v.minimum(b)` (IEEE `NaN`-prop) / `v.min(b)` (`f32::min`) | `v.min_fast(b)` / `v.clamp(lo, hi)` | Unordered `NaN` compare + blend on x86 V3 / WASM | `a.min_fast(b)` returns `b` on `NaN` on x86/WASM/Scalar; `v.clamp(lo, hi)` maps quiet `NaN -> lo` | `1` instruction per bound (`vminps`/`vmaxps`, `fminnm`/`fmaxnm`) vs `3` instructions for `min`/`max` on x86 V3 |
| Multiply-add | `a.mul_add_fused(b, c)` (1 rounding everywhere) | `a.mul_add_relaxed(b, c)` or `a * b + acc` | Software `f64` FMA on `ScalarToken` & non-fusing WASM; FMA latency on single-accumulator chains | `mul_add_relaxed` rounds twice where hardware lacks FMA; `a * b + acc` always rounds twice | `mul_add_fused` is `7.9–11x` on strict WASM `f32x4` & `1.7–8.8x` on baseline scalar (7900X); `a * b + acc` is **`0.49–0.80x` time** of FMA on single-acc loops (i265 / 7900X) |
| Flat transcendentals | `v.exp2_midp()`, `v.log2_midp()`, `v.cbrt_midp_precise()` | `v.exp2_midp_unchecked()`, `v.cbrt_midp()`, or [Magemath](@/magemath/_index.md) | Rail clamps/blends and `* 2^24` subnormal/overflow scaling | Out-of-domain inputs produce unspecified floats (never memory UB) | Up to **`2.56x–3.15x` faster** in Magemath when domain bounds also elide reduction stages |
| Proof wrapper check | `#[attune(wrap)]` | `#[attune(wrap, suppress_const_test)]` | `<Token>::__ARCHMAGE_ASSERT_TIER_*` const item lookup | Skips token type-alias guard (keeps `#[target_feature]` and `rustc` intrinsic checks) | Zero runtime instructions either way; trims generated wrapper AST/MIR |

Sources: [`docs/site/content/magetypes/isa-quirks.md`](@/magetypes/isa-quirks.md), [`benchmarks/recip_x86_zen5-9950x3d_2026-09-03.md`](https://github.com/imazen/archmage/blob/main/benchmarks/recip_x86_zen5-9950x3d_2026-09-03.md), and [`benchmarks/fused_vs_separate_2026-10-25.md`](https://github.com/imazen/archmage/blob/main/benchmarks/fused_vs_separate_2026-10-25.md).

---

## 1. Float-to-int conversions: `to_i32_saturating()` vs `to_i32_fast()` / `to_i32()`

Rust's scalar `x as i32` cast truncates toward zero, saturates positive overflow to `i32::MAX` and negative overflow to `i32::MIN`, and maps `NaN` to `0`. AArch64 NEON (`fcvtzs`) and strict WASM (`i32x4.trunc_sat_f32x4_s`) implement that exact contract in one instruction (`0.0%` fixup overhead on Apple M4 Pro).

x86's `vcvttps2dq` does **not**: it returns the "integer indefinite" sentinel `i32::MIN` (`-2_147_483_648`, `0x8000_0000`) for positive overflow (`>= 2_147_483_648.0`), negative overflow, and `NaN` alike. Enforcing Rust `as i32` semantics in `v.to_i32_saturating()` therefore adds four extra vector operations per register on x86 V3 (`+84.2%` kernel time on Ryzen 9 7950X `f32x4`) and two mask comparisons + two masked moves on native AVX-512 (`+44.7%` on `f32x16`).

Whenever your values are already clamped to a finite `i32` sub-range — such as `[0.0, 255.0]`, `[0.0, 65535.0]`, or `[-32768.0, 32767.0]` — `v.to_i32_fast()` (or `v.to_i32()`) emits the single `vcvttps2dq` instruction on x86 (`to_i32_fast()` also uses `i32x4.relaxed_trunc_f32x4_s` under WASM `relaxed-simd`). Pair it with [`magetypes::scalar::to_i32_fast`](https://docs.rs/magetypes/latest/magetypes/scalar/fn.to_i32_fast.html) in scalar tails, or use [`magetypes::numerics::IntConvert`](https://docs.rs/magetypes/latest/magetypes/numerics/struct.IntConvert.html) (`to_i32_with::<CFG>()`, `to_u16_with::<CFG>()`) for nearest-even rounding policies:

```rust
use archmage::prelude::*;
use magetypes::scalar;

#[attune(_*_t, define(f32x8, i32x8))]
fn quantize_u8_levels_tier(token: Token, src: &[f32], dst: &mut [i32]) {
    assert_eq!(src.len(), dst.len());
    let zero = f32x8::zero_t(token);
    let max = f32x8::splat_t(token, 255.0);
    let (sc, st) = f32x8::partition_slice_t(token, src);
    let (dc, dt) = i32x8::partition_slice_mut_t(token, dst);
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        // `clamp(0.0, 255.0)` bounds every lane into [0, 255] (and maps quiet NaN -> 0.0
        // on x86/NEON/strict WASM/Scalar), so `to_i32_fast()` is 1 instruction on x86.
        f32x8::load_t(token, s).clamp(zero, max).to_i32_fast().store(d);
    }
    for (&x, y) in st.iter().zip(dt.iter_mut()) {
        *y = scalar::to_i32_fast(token, scalar::clamp_f32(token, x, 0.0, 255.0));
    }
}

pub fn quantize_u8_levels(src: &[f32], dst: &mut [i32]) {
    attuned!(quantize_u8_levels_tier(src, dst));
}

let src = [-4.0, 0.0, 12.9, 128.4, 255.0, 999.0, f32::INFINITY, 42.1, -1.0, 300.0];
let mut dst = [0_i32; 10];
quantize_u8_levels(&src, &mut dst);
assert_eq!(dst, [0, 0, 12, 128, 255, 255, 255, 42, 0, 255]);
```

---

## 2. Reciprocal and reciprocal square root: `_portable` vs working tier vs `_approx`

`magetypes` provides three reciprocal tiers (timings below are `32,768` L1-resident `f32x8<X64V3Token>` lanes on AMD Ryzen 9 9950X3D without `-C target-cpu`, [`benchmarks/recip_x86_zen5-9950x3d_2026-09-03.md`](https://github.com/imazen/archmage/blob/main/benchmarks/recip_x86_zen5-9950x3d_2026-09-03.md)):

1. **`v.recip_portable()` / `v.rsqrt_portable()` (`1,800 ns` / `5,400 ns` on Zen 5 V3):** `0 ULP` IEEE division (`1.0 / v`) and `1.0 / v.sqrt()`. Bit-identical on every architecture and exact on subnormal inputs and subnormal outputs such as `1.0 / f32::MAX`. (Note: on Apple Silicon M4 Pro, hardware `fdiv` + `fsqrt` is fast enough that exact division is `1.35x` / `1.18x` *faster* than Newton refinement, so NEON uses exact division for `recip_portable` with no throughput penalty.)
2. **`v.recip()` / `v.rsqrt()` (`1,400 ns` / `1,900 ns` on Zen 5 V3):** working tier (`<= 4 ULP` on normal inputs whose outputs are normal) **plus exact IEEE rails** (`±0 -> ±inf`, `+inf -> +0`, `-inf -> -0` / `NaN`, `NaN -> NaN`).
   - On x86 V3, raw Newton refinement turns `±0` and `±inf` into `NaN` via `0 * inf`, so `recip()`/`rsqrt()` blend rail lanes back to the hardware estimate (`+31.8%` / `+26.7%` over un-repaired Newton on Ryzen 9 7950X `f32x4`, or `+17.2%` / `+13.7%` via `VFIXUPIMM` on AVX-512 `f32x16`).
   - On NEON, `FRECPS` and `FRSQRTS` special-case `inf * 0 -> 2.0` / `1.5` in hardware (`0%` repair overhead).
   - *Known V3 limitation:* x86 `vrcpps` flushes subnormal outputs, so `recip(f32::MAX)` returns `0.0` rather than the subnormal `0x0020_0000`. Use `recip_portable()` if magnitudes above `2^126` must produce nonzero subnormal reciprocals.
   - If your inputs are strictly finite normals in `[2^-126, 2^126)` and you want 1 Newton step *without* the x86 rail-rescue blend, use `v.recip_with::<{ RecipOptions::new().nr1().bits() }>()` from [`magetypes::numerics::RecipOptions`](https://docs.rs/magetypes/latest/magetypes/numerics/struct.RecipOptions.html).
3. **`v.rcp_approx()` / `v.rsqrt_approx()` (`~830 ns` / `830 ns` on Zen 5 V3):** single-instruction ~12-bit hardware estimate on x86 (`~2.4e-4` max relative error), no Newton step, no rail rescue. Use for positive normal denominators where ~12 bits suffice or when fusing your own refinement step.

```rust
use archmage::ScalarToken;
use magetypes::{numerics::RecipOptions, simd::generic::f32x4};

const NR1_FINITE_NORMAL: u32 = RecipOptions::new().nr1().bits();

let v = f32x4::from_array_t(ScalarToken, [0.5, 2.0, 4.0, 8.0]);
assert_eq!(v.recip_portable().to_array(), [2.0, 0.5, 0.25, 0.125]);

for (got, want) in v.recip().to_array().into_iter().zip([2.0, 0.5, 0.25, 0.125]) {
    assert!((got - want).abs() <= 4.0 * f32::EPSILON * want);
}
for (got, want) in v.recip_with::<NR1_FINITE_NORMAL>().to_array().into_iter().zip([2.0, 0.5, 0.25, 0.125]) {
    assert!((got - want).abs() <= 4.0 * f32::EPSILON * want);
}
for (got, want) in v.rcp_approx().to_array().into_iter().zip([2.0, 0.5, 0.25, 0.125]) {
    assert!((got - want).abs() <= 5e-4 * want);
}

// Working-tier `recip()` preserves ±0 and ±inf rails on every backend:
let rails = f32x4::from_array_t(ScalarToken, [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY]);
let r = rails.recip().to_array();
assert_eq!(r[0], f32::INFINITY);
assert_eq!(r[1], f32::NEG_INFINITY);
assert_eq!(r[2].to_bits(), 0.0_f32.to_bits());
assert_eq!(r[3].to_bits(), (-0.0_f32).to_bits());
```

---

## 3. Min, Max, and Clamp: `minimum` vs `min` vs `min_fast` and `clamp`

x86 (`vminps`/`vmaxps`), NEON (`vminq`/`fminnm`), and WASM (`f32x4.min`/`pmin`/`relaxed_min`) all disagree on `NaN` and `(-0.0, +0.0)`. `magetypes` exposes all three contracts explicitly:

- **`v.minimum(b)` / `v.maximum(b)`** (scalar tail: `magetypes::scalar::minimum_f32` / `maximum_f32`): IEEE 754-2019 — returns a quiet `NaN` if either operand is `NaN`, and strictly orders `-0.0 < +0.0`. Native 1-instruction `vminq`/`vmaxq` on NEON and `f32x4.min`/`max` on WASM; ordered compare + blend on x86 and scalar.
- **`v.min(b)` / `v.max(b)`** (scalar tail: `f32::min` / `f32::max`): suppresses a single quiet `NaN` (returning the numeric operand); signed-zero tie-breaking is unspecified. Native 1-instruction `vminnmq`/`vmaxnmq` on NEON and `vrangeps` on AVX-512 `f32`; 3 instructions (`vminps` + `vcmpunordps` + `vblendvps`) on x86 V3 and WASM (`pmin` + `bitselect`).
- **`v.min_fast(b)` / `v.max_fast(b)` and `v.clamp(lo, hi)`** (scalar tail: `magetypes::scalar::{min_fast_f32, max_fast_f32, clamp_f32}`): 1 instruction per bound (`vminps`/`vmaxps` on x86, `vminnmq`/`vmaxnmq` on NEON, `pmin`/`pmax` or `relaxed_min`/`max` on WASM).
  - **Operand order matters on `NaN`:** on x86, strict WASM, and `ScalarToken`, `a.min_fast(b)` evaluates `a < b ? a : b` — returning **`b` (the second operand)** whenever either lane is `NaN`. Thus `f32::NAN.min_fast(1.0) == 1.0`, while `(1.0).min_fast(f32::NAN)` is `NaN`.
  - **`v.clamp(lo, hi)`** expands to `v.max_fast(lo).min_fast(hi)` (`2` instructions). Because `lo` and `hi` sit in the second-operand slot, a quiet `NaN` lane in `v` maps cleanly to `lo` on x86, NEON, strict WASM, and `ScalarToken` without panicking (matched by `magetypes::scalar::clamp_f32`, unlike `f32::clamp` which keeps `NaN` and panics if `lo > hi`). Conversely, writing `hi.min_fast(lo.max_fast(v))` puts `v` in the second slot and preserves quiet `NaN` in 2 instructions on x86.

```rust
use archmage::ScalarToken;
use magetypes::{scalar, simd::generic::f32x4};

let a = f32x4::from_array_t(ScalarToken, [f32::NAN, 1.0, -0.0, 2.0]);
let b = f32x4::from_array_t(ScalarToken, [1.0, f32::NAN, 0.0, 0.5]);

// IEEE 754-2019 `minimum`: propagates NaN and orders -0.0 < +0.0
let ieee = a.minimum(b).to_array();
assert!(ieee[0].is_nan() && ieee[1].is_nan());
assert_eq!(ieee[2].to_bits(), (-0.0_f32).to_bits());

// Rust `f32::min`: suppresses a single quiet NaN in either operand slot
let rust_min = a.min(b).to_array();
assert_eq!(rust_min[0], 1.0);
assert_eq!(rust_min[1], 1.0);

// `min_fast` on ScalarToken / x86 / strict WASM returns the 2nd operand on NaN:
let fast = a.min_fast(b).to_array();
assert_eq!(fast[0], 1.0);
assert!(fast[1].is_nan());

// `v.clamp(lo, hi)` matches `scalar::clamp_f32` (mapping quiet NaN -> lo here):
let lo = f32x4::zero_t(ScalarToken);
let hi = f32x4::splat_t(ScalarToken, 1.0);
let clamped = a.clamp(lo, hi).to_array();
assert_eq!(
    clamped,
    [f32::NAN, 1.0, -0.0, 2.0].map(|x| scalar::clamp_f32(ScalarToken, x, 0.0, 1.0)),
);
assert_eq!(clamped[0], 0.0);
```

---

## 4. Multiply-add: `mul_add_fused` vs `mul_add_relaxed` vs `a * b + acc`

Which multiply-add spelling is fastest depends on both the target ISA and which operand carries the loop dependency ([`benchmarks/fused_vs_separate_2026-10-25.md`](https://github.com/imazen/archmage/blob/main/benchmarks/fused_vs_separate_2026-10-25.md), [`docs/site/content/magetypes/isa-quirks.md`](@/magetypes/isa-quirks.md#fused-arithmetic-and-reductions)):

- **`a.mul_add_fused(b, c)`** (scalar tail: `magetypes::scalar::mul_add_fused_f32`): single rounding (`round(a * b + c)`) on every backend. Costs 1 FMA instruction on x86 V3/V4 and NEON, **zero overhead over `mul_add_relaxed`**. On strict WASM `f32x4` (wasmtime 47, Ryzen 9 7900X) software single-rounding takes `7.9–11x` the time of `a * b + c`, and on baseline x86-64 `ScalarToken` `1.7–8.8x` (`0.68–1.45x` on fusing `relaxed-simd` WASM engines on Ryzen 9 9950X3D). Pick `mul_add_fused` when cross-ISA bit identity is required.
- **`a.mul_add_relaxed(b, c)`** (scalar tail: `magetypes::scalar::mul_add_relaxed_f32`): 1 FMA on x86 V3/V4 and NEON (`24–38%` less time than `a * b + c` in Horner chains `x = x.mul_add_relaxed(c, k)` where the dependency runs through the **multiplicand**), plain `a * b + c` on `ScalarToken` and strict WASM, and direct `relaxed_madd` on relaxed WASM.
- **Separate `a * b + acc` in single-accumulator reductions:** when a loop accumulates `acc = a[i] * b[i] + acc`, the loop-carried dependency runs only through the **addend** `acc`. The `vmulps` instructions execute off the critical path in parallel, leaving only `vaddps` latency on the dependency chain — which is shorter than `vfmadd231ps` latency on modern x86 cores. On `f32x8<X64V3Token>`, separate `a * b + acc` takes **`0.49–0.58x` the time of FMA** on Intel Core Ultra 7 265K (Arrow Lake) and **`0.78–0.80x` the time** on AMD Ryzen 9 7900X (Zen 4). Unrolling to **4 independent vector accumulators** hides FMA latency and brings FMA and `a * b + acc` to a tie (`1.02x`).

```rust
use archmage::prelude::*;

#[attune(_*_t, define(f32x8))]
fn dot_product_tier(token: Token, a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let (ac, at) = f32x8::partition_slice_t(token, a);
    let (bc, bt) = f32x8::partition_slice_t(token, b);
    let mut acc = f32x8::zero_t(token);
    for (va, vb) in ac.iter().zip(bc.iter()) {
        // Single-accumulator chain: separate `*` + `+` keeps shorter `vaddps`
        // latency on the loop-carried `acc` dependency (0.49–0.80x FMA time).
        acc += f32x8::load_t(token, va) * f32x8::load_t(token, vb);
    }
    let mut sum = acc.reduce_add();
    for (&xa, &xb) in at.iter().zip(bt.iter()) {
        sum += xa * xb;
    }
    sum
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    attuned!(dot_product_tier(a, b))
}

assert_eq!(dot_product(&[1.0; 19], &[2.0; 19]), 38.0);
```

---

## 5. Flat `magetypes` transcendentals vs `magemath`

`magetypes` keeps all of its flat transcendental methods (`exp2_midp`, `log2_midp`, `pow_midp`, `cbrt_midp`, `cbrt_midp_precise`, `sigmoid_midp`, `silu_midp`, plus `_lowp` and `_unchecked` variants). None are deprecated. Pick between them based on what your kernel needs:

| Requirement | Best fit |
|---|---|
| Inline polynomial inside an existing `magetypes` loop, inputs already clamped, 1–3 ULP cross-ISA FMA drift acceptable, custom scalar tail | Flat `v.exp2_midp_unchecked()`, `v.log2_midp_unchecked()`, `v.pow_midp_unchecked(n)` |
| Neural activations (`1 / (1 + exp(-x))`, `x * sigmoid(x)`) with exact `0.0` / `1.0` / `-0.0` saturation rails | Flat `v.sigmoid_midp()`, `v.silu_midp()` |
| Cube root over arbitrary floats including subnormals and `\|x\| >= f32::MAX / 3` without adding a crate | Flat `v.cbrt_midp_precise()` (note: the other six `*_midp_precise` names are no-op aliases of `*_midp`, queued for removal) |
| Whole-slice buffer transform with automatic 16-lane AVX-512 widening (`2.7x–19.2x` faster than per-chunk calls) | `magemath` slice entries (`Powf::<CFG>::slice`, `Expf::<CFG>::exp2_slice`, ...) |
| Compile-time `.const_exponent(n)` (including `vsqrtps` / `vmulps` algebraic lowering), `SingleOctave` / `AroundZero` reduction elision, or dedicated Kahan `log1p` | `magemath` (`PowfOptions`, `ExpfOptions`, `LogfOptions`, `Log1pfOptions`, `CbrtfOptions`) |
| Bit-for-bit identical output across x86, ARM64, WASM, and `ScalarToken`, matching `*_with_token` scalar tails, `Highp` `powf`, or whole-slice error reporting (`SliceError { kind, index }`) | `magemath` (`Portability::BitExact` default, `SimdMath` + `*_with_token`) |

---

## 6. Skipping token type-alias const checks: `#[attune(wrap, suppress_const_test)]`

When you write a proof wrapper with `#[attune(wrap)]` (or `#[arcane]`) taking a concrete token such as `token: X64V3Token`, the generated safe outer wrapper emits a compile-time constant reference before calling the private `#[target_feature]` inner function:

```text
let _: () = <X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
```

That constant has zero runtime instructions — it exists purely to catch an accidental local `type X64V3Token = X64V2Token;` alias before the wrapper's `unsafe` feature-boundary call (`rustc` still checks every intrinsic inside the `#[target_feature]` body independently).

Generated families (`#[attune(_*_t, ...)]`) and suffix-inferred wrappers (`#[attune] fn work_v3_t(...)`) already omit that constant because the macro synthesized the token type itself. When macro-generated code or internal crates emit many concrete `#[attune(wrap)]` boundaries where the token path cannot be aliased, pass `suppress_const_test` to skip generating the constant lookup as well:

```rust
use archmage::prelude::*;
use magetypes::simd::generic::f32x8;

#[attune(wrap, suppress_const_test)]
fn add_pair_v3(token: X64V3Token, a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
    (f32x8::from_array_t(token, a) + f32x8::from_array_t(token, b)).to_array()
}

#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() {
    assert_eq!(add_pair_v3(token, [1.0; 8], [2.0; 8]), [3.0; 8]);
}
```

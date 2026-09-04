# Cross-ISA divergences: what we fix up, what we type away, what stays natural

magetypes runs one generic body on x86 (SSE→AVX-512), NEON, WASM SIMD128, and
a scalar fallback. Those ISAs disagree on corner cases — silently, one lane at
a time. This is the canonical inventory: every known divergence, and which of
three resolutions it got.

- **Fixed up** — we pay instructions so every backend returns identical
  results. The fixup cost and mechanism are listed; the "natural" column shows
  what the hardware would have done.
- **Typed away** — the API shape makes the divergent case unrepresentable.
- **Documented divergence** — behavior is per-backend by contract; the caller
  chooses an op whose looseness they can afford.

Related deep-dives: [`CROSS-ISA-INT-PRIMITIVES.md`](CROSS-ISA-INT-PRIMITIVES.md)
(the shift/saturate/widen/narrow audit with stdarch citations),
[`PERFORMANCE.md`](PERFORMANCE.md), and the measured records under
[`../benchmarks/`](../benchmarks/).

## 1. Fixed up — uniform semantics, fixup on the divergent backend

| Case | Natural (divergent) behavior | Uniform contract | Fixup & cost |
|---|---|---|---|
| `recip()`/`rsqrt()` at `±0`/`±inf`/NaN (f32 working tier) | Newton refinement computes `a·r = inf·0 = NaN` on rail lanes (x86; pre-fix NEON rsqrt) | ≤4 ULP **and** exact IEEE rails: `recip(±0)=±inf`, `recip(±inf)=±0` signed, `rsqrt(-x)=NaN`, NaN propagates | x86 128/256-bit: FMA-Newton error term is NaN exactly on rail lanes → `cmp_unord`+`blendv` rescue to the raw estimate (rail-exact on x86); +2.6% recip, +20% rsqrt vs unrescued Newton. AVX-512 512-bit: one `VFIXUPIMM` (tables `0x00870622`/`0x03830622`, silicon-verified); +11.7%. NEON: free — FRECPS/FRSQRTS special-case `inf·0` by design; rsqrt just forms the product *inside* the fused op (`y·FRSQRTS(a, y·y)`, never a plain `a·y` mul). WASM/scalar/f64: exact division, nothing to fix. |
| Shift-by-const with out-of-range `N` | x86 clamps/zeroes, WASM masks `N mod bits`, scalar splits compile-error vs wrap, NEON rejects `N==0` but not `N==bits` | `N ∈ 0..=lane_bits-1` or it **fails to compile, identically everywhere** | Front-end `const` asserts; NEON right shifts lowered via `vshlq(a, dup(-N))` so `N==0` exists (#62/#63). Zero runtime cost. |
| Uniform variable shift with `count ≥ lane_bits` | x86 gives 0/sign-fill, NEON's USHL reads the low *signed byte* of the count (256 wraps to a no-op), WASM takes `count mod lane_bits` | 0 for `shl`/`shr_logical`, sign-fill for `shr_arithmetic`, everywhere | Free on x86 (hardware behavior); NEON clamps the count before splatting; WASM ANDs a keep-mask. ≤2 ops, loop-invariant. |
| 8-bit shifts on x86 | No `PSLLB` exists at any x86 tier | Same semantics as every other width | 16-bit shift + byte-mask polyfill (~2 ops), same technique for const and uniform forms. |
| `i64` arithmetic right shift on x86 | `_mm_sra_epi64` requires AVX-512 | Sign-filling shift at v3 too | Polyfilled below AVX-512; native `VPSRAQ` on V4+. |
| 256-bit saturating narrow lane order | `_mm256_packus_epi16` emits `[a.lo, b.lo, a.hi, b.hi]` — silently permuted **on exactly one tier** | `[a…, b…]` element order everywhere | One `permute4x64::<0xD8>` after the pack on AVX2 (~3-cycle lane-crossing shuffle); 128-bit pack and NEON/WASM narrows are naturally ordered. Pinned by per-lane-distinct tests. |
| 512-bit saturating narrow | `_mm512_pack*` interleaves across all four 128-bit lanes — worse than AVX2 | Same element order | Avoided entirely: `VPMOV*` conversions + `inserti64x4` (natural order); unsigned-destination forms clamp at zero first (`max_epi16`) so `VPMOVUSWB`'s unsigned read matches `packus`. |
| `round()` ties | SSE2-era paths and libm `round` disagree on ties | Ties-to-even on every backend (matches the hardware rounding mode and `roundscale`/`FRINTN`/`f32.nearest`) | Fixed in 0.9.16; note this **differs from Rust scalar `f32::round()`**, which rounds ties away from zero — see §4. |
| Scalar-tier `f32→u8` pixel pack | Naïve scalar `as` chains | round-nearest-even, clamp to `[0,255]`, NaN→0 — matching `CVTPS2DQ`+`PACKUSWB` and `FCVTNS`+saturating narrows | Explicit `roundevenf().clamp()` scalar body; native instruction sequences elsewhere. |

## 2. Typed away — the divergent operation is unrepresentable

| Would-be operation | Why it diverges | API shape instead |
|---|---|---|
| `u16 → u8` / `u32 → u16` saturating narrow | x86 `packus` and WASM `narrow` clamp the **source as signed**: `0x8000` narrows to 0 on x86/WASM, 255 on NEON | Narrowing is exposed only from **signed** sources (`i16xN`, `i32xN`); a `u16` source cannot be passed. |
| Truncating (non-saturating) narrow | NEON-only (`vmovn`) | Not exposed. |
| Per-lane variable shifts | 16-bit forms need AVX-512BW+VL; WASM has none at any width; 8-bit is NEON-only | Only the uniform-count form exists (`*_uniform(count: u32)`). |
| 32/64-bit saturating add/sub | No `_mm*_adds_epi32` at any x86 tier; no WASM 32-bit `add_sat` | `saturating_add`/`saturating_sub` exist only on 8/16-bit types. |
| Rounding narrowing shift (`vqrshrn`-family) | Single instruction on NEON, 3-4 ops elsewhere | Not exposed (audit verdict: portability lie without a measured case). |

## 3. Documented divergence — per-backend by contract

| Case | x86 | NEON | WASM | scalar | Guidance |
|---|---|---|---|---|---|
| `mul_add`/`mul_sub` rounding | fused, 1 rounding | fused, 1 rounding | mul+add, 2 roundings | 2 roundings | ≤1 ULP cross-backend; avoid near-zero cancellation. |
| `to_i32` on out-of-range/NaN lanes | `i32::MIN` sentinel (`CVTTPS2DQ`) | saturates, NaN→0 (`FCVTZS`) | saturates, NaN→0 (`trunc_sat`) | saturates, NaN→0 (`as`) | Bare op stays natural (single instruction; `3e9_f32` gives −2³¹ on x86, +2³¹−1 elsewhere — documented on the method). **`to_i32_saturating` is the uniform fixed-up form** ([#80](https://github.com/imazen/archmage/issues/80)): Rust-`as` semantics everywhere, native on NEON/WASM/scalar, a 4-op compare/blend fixup over `cvttps` on x86 (mask-register form at 512-bit). |
| `min`/`max` NaN propagation | returns 2nd operand when 1st is NaN | returns the non-NaN operand | returns the non-NaN operand | Rust semantics | Filter NaN first, or compare+blend. |
| `simd_ne` NaN semantics | unordered (NaN≠x → true) | ordered on some impls | ordered | ordered | Use `simd_eq` + `not` for portable unordered-NE. |
| `neg(0.0)` | `sub(0,x)` → `+0.0` | `vneg` → `−0.0` | `−0.0` | `−0.0` | XOR the sign mask if zero's sign matters. |
| `blend` argument order | `(mask, t, f)` | `(mask, t, f)` | `(self, other, mask)` | — | Avoid in portable generic code. |
| Bitwise ops on integers | operator traits | methods only | methods only | traits | Use `.and()/.or()/.xor()` methods portably. |
| `reduce_add` association | tree | tree | tree | tree | Same shape but still FP-order-sensitive; allow ~1e-6 rel on large sums. |
| `rcp_approx`/`rsqrt_approx` rails | raw estimates happen to be rail-exact | rail-clean after the fused-op reorder | division (exact) | bit-hack: garbage at `±0` | **Unspecified at this tier** — only `recip()`/`rsqrt()` and up contract rails. |
| Subnormal inputs to `recip()`/`rsqrt()` | `rcpps` flushes → NaN/±inf outcomes | deep subnormals (< ~3e-39) saturate | exact | exact | Unspecified at the working tier; `_portable` is exact everywhere including subnormals. |
| `interleave_lo/hi` | f32x4 only | f32x4 only | f32x4 only | — | Don't use on integer types. |

## 4. Divergence from *scalar Rust*, not between backends

These bite people porting scalar code:

- **`round()` is ties-to-even; `f32::round()` is ties-away-from-zero.**
  `round(2.5)` = `2.0` here, `3.0` in scalar Rust. Every backend here agrees
  with each other (and with hardware rounding), not with libm.
- **Don't use `f32::mul_add` as a test oracle on non-FMA targets** — Rust's
  scalar fallback (via compiler-builtins/libm) has a known subnormal
  halfway-case rounding bug on platforms without hardware FMA (i686, pre-AVX2
  x86, some ARM32).
- **Never implement `trunc`/`fract` as `x - ((x as i32) as f32)`** — it
  breaks for `|x| ≥ 2³¹`. Backends use dedicated rounding instructions.

## 5. Open items

- [#81](https://github.com/imazen/archmage/issues/81) — adoption list from the
  fearless_simd review (swizzle_dyn, token materialization, formulation wins).

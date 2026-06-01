# Newer-intrinsic acceleration opportunities — survey 2026-06-01

Read-only survey across **magetypes**, the **jxl-encoder-simd DCT/IDCT
kernels**, and **zenpredict** (MLP picker runtime). Goal: find ops still
using a generic/scalar lowering where a newer-stable (or nightly-gated)
per-arch intrinsic would help, with **verified** stdarch stability so each
candidate gets the right gating mechanism.

Stability was verified against the installed **stable toolchain rustc
1.95.0** stdarch source at
`~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/stdarch/crates/core_arch/src/`
(rust-src present on stable). Every `#[stable(... since ...)]` /
`#[unstable(feature = ...)]` line below was read from that source, not from
docs or memory. magetypes MSRV = **1.89** (workspace `Cargo.toml`).

Lesson reused from PR #51/#52: newer-stable intrinsics →
`rustversion::since(X.Y)` + `#[cfg(target_arch)]` + CI-both-sides; nightly-only
intrinsics → try-compile probe. A *dependency* MSRV is a hard floor
`rustversion` can't lower (it gates code, not Cargo resolution).

---

## 1. Verified stability table (the load-bearing data)

| Intrinsic | Arch | target_feature | Stability (from stdarch source) | Gating mechanism |
|---|---|---|---|---|
| `_mm256_fmadd_ps` / `_mm_fmadd_ps` / `_mm512_fmadd_ps` | x86 | `fma` / `avx512f` | **stable 1.27.0** (`simd_x86`) | already baseline |
| `_mm256_rcp_ps` / `_mm256_rsqrt_ps` | x86 | `avx` | **stable 1.27.0** (`simd_x86`) | already baseline |
| `_mm256_round_ps` | x86 | `avx` | **stable 1.27.0** (`simd_x86`) | already baseline |
| `_mm_dp_ps` / `_mm256_dp_ps` | x86 | `sse4.1` / `avx` | **stable 1.27.0** (`simd_x86`) | already baseline |
| `_mm256_cvtph_ps` / `_mm256_cvtps_ph` (F16C) | x86 | `f16c` | **stable 1.68.0** (`x86_f16c_intrinsics`) | already baseline (PR #51 uses) |
| `_mm512_cvtph_ps` (AVX-512F) | x86 | `avx512f` | **stable 1.89** (`stdarch_x86_avx512`) | already at MSRV |
| `_mm256_dpbusd_epi32` / `_mm256_dpwssd_epi32` (VNNI) | x86 | `avx512vnni,avx512vl` | **stable 1.89** (`stdarch_x86_avx512`) | already at MSRV — but see note ‡ |
| `_mm512_fmadd_ph` (AVX-512-FP16) | x86 | `avx512fp16` | **stable 1.94.0** (`stdarch_x86_avx512fp16`) | `rustversion::since(1.94)` — same gate as PR #52 |
| `vfmaq_f32` | aarch64 | `neon` | **stable 1.59.0** (`neon_intrinsics`) | already baseline |
| `vrecpeq_f32` / `vrsqrteq_f32` | aarch64 | `neon` | **stable 1.59.0** (`neon_intrinsics`) | already baseline |
| `vrndnq_f32` (round-to-nearest-even) | aarch64 | `neon` | **stable 1.59.0** (`neon_intrinsics`) | already baseline |
| `vcvtnq_s32_f32` (round-and-convert) | aarch64 | `neon` | **stable 1.59.0** (`neon_intrinsics`) | already baseline |
| `vtrn1q_f32` / `vzip1q_f32` / `vuzp1q_f32` (transpose) | aarch64 | `neon` | **stable 1.59.0** (`neon_intrinsics`) | already baseline |
| `vaddq_f16` / `vmulq_f16` / `vfmaq_f16` (FEAT_FP16 arith) | aarch64 | `neon,fp16` | **stable 1.94.0** (`stdarch_neon_fp16`) | `rustversion::since(1.94)` — **identical gate to PR #51/#52 f16 conversion** |
| `vdot_s32` / `vdotq_s32` (DotProd) | aarch64 | `neon,dotprod` | **unstable** (`stdarch_neon_dotprod`, issue 117224) | **nightly try-compile probe** |
| `vmmlaq_s32` / `vusmmlaq_s32` (i8mm) | aarch64 | `neon,i8mm` | **unstable** (`stdarch_neon_i8mm`, issue 117223) | **nightly try-compile probe** |
| any `sve` module | aarch64 | — | **absent from stable stdarch** (no `sve` module under `aarch64/`) | nightly-only, not located on stable |
| `f32x4_relaxed_madd` (relaxed-SIMD FMA) | wasm32 | `relaxed-simd` | **stable 1.82.0** (`stdarch_wasm_relaxed_simd`) | already < MSRV; needs `relaxed-simd` target-feature opt-in |
| `i32x4_relaxed_dot_i8x16_i7x16_add_s` (int8 dot+acc) | wasm32 | `relaxed-simd` | **stable 1.82.0** (`stdarch_wasm_relaxed_simd`) | already < MSRV; `relaxed-simd` opt-in |
| `i32x4_dot_i16x8` (i16 widening dot, baseline) | wasm32 | `simd128` | **stable 1.54.0** (`wasm_simd`) | already baseline |

‡ **VNNI note (important):** current stdarch ships VNNI **only** in
`x86/avx512vnni.rs` with `target_feature = "avx512vnni,avx512vl"`. There is
**no `avxvnni.rs`** module — i.e. the standalone AVX-VNNI (`avxvnni`)
encoding available on Alder Lake / Zen 4 *without* AVX-512 is **not exposed
on stable**. So on x86 the stable VNNI path needs AVX-512-VNNI hardware,
which narrows its applicability (it's an `_v4x`/AVX-512-tier-only op, not a
v3/AVX2-tier op).

---

## 2. magetypes op-by-op audit

Read of `magetypes/src/simd/impls/{x86_v3,x86_v4,x86_v4_f32_delegated,arm_neon,wasm128,scalar}.rs`
plus `backends/` and `generic/generated/`.

| Op | x86 v3 (AVX2) | x86 v4 (AVX-512) | neon | wasm128 | scalar | Verdict |
|---|---|---|---|---|---|---|
| `mul_add` | `_mm256_fmadd_ps` (true FMA) | `_mm512_fmadd_ps` | `vfmaq_f32` | **`f32x4_add(f32x4_mul(a,b),c)` — NOT fused** | `f32::mul_add` | **wasm gap** |
| `recip` | `_mm_rcp_ps`/`_mm256_rcp_ps` + Newton | delegates to v3 | `vrecpeq_f32` + 2 Newton steps | `1/x` via `f32x4_div` (no HW rcp on wasm) | f32 recip | OK — wasm correct (no rcp instr exists) |
| `rsqrt` | `_mm_rsqrt_ps`/`_mm256_rsqrt_ps` + Newton | delegates to v3 | `vrsqrteq_f32` + 2 Newton | `1/sqrt(x)` via div+sqrt | f32 | OK — wasm correct |
| `round` | `_mm256_round_ps<NEAREST>` | (via v3 delegate) | `vrndnq_f32` | `f32x4_nearest` | `f32::round_ties_even` | OK everywhere |
| convert f16↔f32 | F16C `_mm256_cvtph_ps` (PR #51) | (F16C/AVX-512) | `vcvt_f32_f16` family | scalar bit-math | scalar | OK (recently landed) |
| `dot` | **not a primitive** — `dot` only exists as a doc-example (`generic/generated/mod.rs:13`, `tests/archmage_doc_examples.rs:434`); callers build it from `mul_add` + horizontal reduce | — | — | — | — | N/A — no dedicated dot |
| int8 dot (`i8x16` ops) | `backends/i8x16.rs` + `generic/generated/block_ops_i8x16.rs` are **load/store/cast only** — no madd / widening-mul / dot at all | — | — | — | — | **gap if int8-dot ever needed** |
| `transpose_8x8` (generic helper) | scalar gather (`core::array::from_fn(|j| r[j][i])` in `block_ops_f32x8.rs:275`) | same | same | same | same | **scalar; intrinsics exist but unused** (note A) |

### Headline gaps

1. **wasm `mul_add` is a separate mul+add (no fusion).** This is the
   zenresize-f16 pathology pattern: x86 and neon get a true single-instruction
   FMA, wasm does `f32x4_mul` then `f32x4_add`. `f32x4_relaxed_madd` (stable
   1.82.0, < MSRV) fuses it into one `f32x4.relaxed_madd`. Caveat: relaxed-madd
   is *non-deterministic across runtimes* (may or may not round the
   intermediate) and requires the `relaxed-simd` target feature — so it can't
   be the unconditional `mul_add` without a determinism opt-in. Best as an
   explicit `mul_add_relaxed` variant or a `relaxed-simd`-gated body.

2. **No int8 dot-product primitive anywhere in magetypes.** The `i8x16`
   backend is pure data movement (cast/load/store). If any consumer wants int8
   GEMM/dot acceleration (zenpredict picker, quantized convolution), the
   widening-MAC primitive doesn't exist yet to build on. (Whether it's worth
   building depends on a consumer actually quantizing activations — see §4.)

(note A) The generic `transpose_8x8` helper is a scalar gather, but **the only
hot consumer (the jxl DCT path) ships its own optimized `transpose_8x8_avx2` /
`transpose_8x8_neon`** in `jxl-encoder-simd/src/transpose.rs`, so the scalar
helper isn't on a measured hot path today. Upgrading the magetypes helper to
use `vtrn1q`/`vzip1q` (neon) and `_mm256_unpacklo_ps`/`vperm2f128` (x86) is a
nice-to-have, not urgent.

---

## 3. jxl-encoder-simd DCT/IDCT audit

Hot SIMD path is the `dct1d_N_batch(token, v: &mut [magetypes::simd::f32x8; N])`
family; the `*_scalar` functions are reference paths. The batch path operates
on `f32x8` via `Mul`/`Add` operator overloads.

### 3a. FMA status — REAL low-effort opportunity

Rust does **not** contract `a * b + c` into an FMA by default (no `-ffast-math`
fast-contract); only an explicit `.mul_add()` emits `vfmadd…`. The kernels are
inconsistent:

| File | `mul_add` count | Status |
|---|---|---|
| `dct8.rs` | 11 (uses `scalarmath::mul_add_f32` + `f32x8::mul_add`) | **good — the model** |
| `dct16.rs` | 9 | mostly good |
| `dct4.rs` | 1 | partial |
| `fused_dct8.rs` | 2 | partial |
| `dct32.rs` | **0** in batch path | gap — `s[0] = sqrt2 * s[0] + s[1]` (dct32.rs:338) is mul+add |
| `dct64.rs` | **0** | gap |
| `idct16.rs` | **0** | gap |
| `idct32.rs` | **0** | gap |
| `idct64.rs` | **0** | gap |

Each "B-transform" step in the batch functions has a `sqrt2 * s[0] + s[1]`
shaped line that should be `s[0].mul_add(sqrt2, s[1])`. These are on the
critical path of the larger transforms (32/64-point and all the inverse
transforms). The win is the same FMA already proven in dct8/dct16:
1 instruction instead of 2 + better rounding. All baseline-stable
(`_mm256_fmadd_ps` 1.27, `vfmaq_f32` 1.59) — **no new gating needed**, pure
source edit. Recommend a `cargo asm` confirm on the dct8 model first to verify
LLVM emits `vfmadd231ps` there and a separate `vmulps`+`vaddps` in dct32.

### 3b. f16 arithmetic for DCT intermediates — gated, precision-risky

DCT intermediates are f32. Native half-precision FMA (`vfmaq_f16` aarch64
FEAT_FP16, stable 1.94.0 / `_mm512_fmadd_ph` AVX-512-FP16, stable 1.94.0) could
double lane throughput **if precision allows** — but a forward+inverse 32/64
DCT accumulates rounding, and f16 has only ~10 mantissa bits. JXL VarDCT
coefficients feed quantization where error is visible. This is **not a free
swap**; it needs an error-budget study against the reference scalar DCT before
any f16 intermediate. Flag as research, not quick-win. If pursued, it reuses
the exact PR #52 `rustversion::since(1.94)` + `neon,fp16` gate.

### 3c. Transpose — already optimized (no action)

`transpose.rs` already dispatches `transpose_8x8_avx2` (unpack/shuffle/permute)
and `transpose_8x8_neon` (`transpose_4x4_neon` building blocks). **However**,
the 32×32 transpose *inside* `dct_32x32_avx2` (dct32.rs ~line 378) is a scalar
double-loop `transposed[c*32+r] = tmp[r*32+c]` that does **not** call the fast
8×8 transpose — it could tile into 8×8 blocks and reuse `transpose_8x8_avx2`.
Medium-effort, baseline-stable. Same likely applies to the 64×64 path in
dct64.rs.

### 3d. Integer/fixed-point DCT — N/A

The DCT path is f32 throughout; there is no integer/fixed-point DCT, so
`vmull`/`vmlal` widening-MAC and `vcvtnq_s32_f32` rounding-convert have no
target here. (Those would matter for an int-DCT JPEG path, not this JXL f32
path.)

---

## 4. zenpredict MLP — int8 dot feasibility

`zenpredict/src/inference.rs` already has three matmul paths selected by weight
dtype: `saxpy_matmul_f32`, `saxpy_matmul_f16`, `saxpy_matmul_i8`.

**The i8 path is NOT an int8 dot-product** — it is a SAXPY that *dequantizes
each i8 weight to f32* and accumulates in f32:

```rust
// saxpy_matmul_i8, inference.rs:197-200
let wf = weight_chunk[k] as f32;      // i8 -> f32 dequant
acc_chunk[k] = fma(s, wf, acc_chunk[k]); // f32 FMA, src `s` is f32
```

The activations (`src`) are **f32**, never quantized. VNNI
(`_mm256_dpbusd_epi32`) / NEON DotProd (`vdotq_s32`) / i8mm (`vmmlaq_s32`) all
require **both** operands as int8 and accumulate in int32. Using them would
require a per-layer activation-quantization scheme (compute per-layer scale,
quantize `src` to int8, int8×int8→int32 accumulate, then dequantize the int32
sums) — a numerically-significant change to the inference contract, not a
storage-format swap.

**Feasibility: real but a bigger change.** The recovery docs note bakes are
f32/f16 today; an int8-activation path also needs the bake side to ship
quantization scales. Flag as a larger workstream, **not** a quick intrinsic
swap. If it ever lands, the x86 stable path is AVX-512-VNNI-only (‡ above) so
it would be an `_v4x`-tier optimization; neon DotProd/i8mm are nightly
(try-compile probe). The wasm `i32x4_relaxed_dot_i8x16_i7x16_add_s` (stable
1.82) is the most accessible int8-dot primitive but is wasm-only and the "i7"
asymmetric-range caveat applies.

Note: the source comment at inference.rs:9-12 already anticipates a
`magetypes::f32x8` FMA loop — the current code relies on LLVM auto-vectorizing
the fixed-`[f32; 8]`-chunk `fma()` into `vfmadd…`. A `cargo asm` check that
this actually vectorizes (vs scalar `vfmadd231ss`) is a worthwhile zero-risk
verification independent of int8.

---

## 5. Prioritized backlog (value × low-effort)

| # | Opportunity | Where it lands | Candidate intrinsic | Verified stability | Gating | Reuses PR #52? | Effort | Expected win |
|---|---|---|---|---|---|---|---|---|
| **1** | **FMA the DCT32/64 + all IDCT batch B-transform `sqrt2*x+y` steps** | jxl-encoder-simd `dct32/64.rs`, `idct16/32/64.rs` | `_mm256_fmadd_ps` / `vfmaq_f32` (via `f32x8::mul_add`) | stable 1.27 / 1.59 | **none — baseline** | no | **low** (mechanical `.mul_add()` edits matching dct8.rs) | removes a `vmulps`+`vaddps` pair per butterfly on the largest/inverse transforms; better rounding |
| **2** | **wasm `mul_add` → relaxed-madd variant** | magetypes `impls/wasm128.rs` | `f32x4_relaxed_madd` | stable 1.82.0 (`stdarch_wasm_relaxed_simd`) | `relaxed-simd` target-feature gate (NOT a rustversion gate; < MSRV already) | partial (cfg pattern) | **low-medium** (add gated `mul_add` body or `mul_add_relaxed`; keep deterministic default) | fuses the only non-FMA `mul_add` in the matrix; closes the wasm-vs-x86/neon gap; helps every wasm DCT/MLP/transcendental consumer |
| **3** | **f16-arith primitives in magetypes (`f16x8` add/mul/fma)** | magetypes (new `f16` SIMD type) | `vfmaq_f16`/`vmulq_f16` (neon), `_mm512_fmadd_ph` (x86) | **stable 1.94.0** both arches | **`rustversion::since(1.94)` + `cfg(target_arch)` + CI-both-sides — IDENTICAL to PR #51/#52** | **YES — the obvious quick win, same 1.94 gate** | **low-medium** (the gating machinery already exists from #52; the f16 *conversion* already landed, this adds *arithmetic*) | unlocks native half-precision FMA for any future f16 kernel; the lowest-friction newer-intrinsic add because the gate scaffold is done |
| **4** | **32×32 / 64×64 DCT transpose → tile + reuse `transpose_8x8_avx2/neon`** | jxl-encoder-simd `dct32.rs`/`dct64.rs` | `_mm256_unpacklo_ps`/`vperm2f128` (x86), `vtrn1q`/`vzip1q` (neon) — already wrapped in `transpose.rs` | stable 1.27 / 1.59 | **none — baseline** | no | **medium** (replace scalar double-loop with 8×8 tiling) | removes a scalar element-by-element transpose between DCT passes on the two largest transform sizes |
| **5** | **`cargo asm` verify zenpredict saxpy auto-vectorizes to `vfmadd…`** | zenpredict `inference.rs` (verification only) | `_mm256_fmadd_ps` (auto-vec) | stable 1.27 | none | no | **trivial** (asm spot-check) | confirms the MLP inner loop isn't silently scalar; gates whether to hand-vectorize with `magetypes::f32x8` |

**Explicitly NOT recommended as quick wins:**
- **int8 VNNI/DotProd/i8mm dot in zenpredict** — needs activation quantization
  (a real numerical change), x86 stable path is AVX-512-only, neon path is
  nightly. Larger workstream; revisit only if a quantized-activation bake ships.
- **f16 DCT intermediates** — precision-risky; needs an error-budget study
  before any swap. Research, not backlog quick-win.
- **AVX-VNNI without AVX-512** — not exposed on stable stdarch (no `avxvnni`
  module), so the Alder-Lake/Zen-4 standalone path is unavailable regardless.
- **SVE** — no `sve` module in stable stdarch; nightly-only.

**Top quick win that reuses PR #52's machinery:** **#3 (f16 arithmetic
primitives)** — identical `rustversion::since(1.94)` + arch-cfg + nightly-probe
scaffold already built for the f16 *conversion*; adding f16 *arithmetic* slots
straight into it. Pair with **#1 (DCT FMA)** which is pure baseline-stable
source edits with zero gating risk.

---

## Provenance

- stdarch source: stable rustc 1.95.0 toolchain, paths cited inline.
- magetypes: `~/work/archmage/magetypes/src/simd/` (read-only; primary tree had
  an unrelated in-progress f16-conversion change from another session — not
  touched; this doc was written in an isolated `jj workspace` checked out from
  clean `main@origin` 4bc2a957).
- jxl-encoder-simd: `~/work/zen/jxl-encoder/jxl-encoder-simd/src/`.
- zenpredict: `~/work/zen/zenanalyze/zenpredict/src/inference.rs`.
- Survey log: `/tmp/intrinsic_opportunity_survey_2026-06-01.log`.
- No perf runs, no code changes, no cloud. Stability verified from source, not
  docs or memory. Symbols not located on stable (any `sve`, `avxvnni`) are
  marked as such rather than guessed.

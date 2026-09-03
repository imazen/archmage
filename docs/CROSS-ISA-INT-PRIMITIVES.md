# Cross-ISA audit: variable shift, saturating add/sub, widening/narrowing

**Question asked:** do these integer primitives exist on NEON *and* on the x86 tiers
(`v3`/`v4`/`v4x`) *and* on `wasm128`, do they behave identically, and can they be
exposed with a semantics that is provable on every tier?

**Answer, in one line:** two of the three are soundly universal at a *reduced* set of
widths, and the third (widening/narrowing) is universal only for a specific
signed-source/saturating shape — the "obvious" shape a caller would reach for is
**not** portable and would silently produce different pixels on AVX2.

Date: 2026-09-03. Toolchain used for every citation below: `rustc 1.98.0`
(`88d9e12ae`), whose vendored `stdarch` lives at
`$(rustc --print sysroot)/lib/rustlib/src/rust/library/stdarch/crates/core_arch/src`
(abbreviated `core_arch` below). Feature attributions were cross-checked against
this repo's own extracted database, `docs/intrinsics/complete_intrinsics.csv`
(11,719 rows, column 3 = required features).

---

## 0. Tier map — one correction before anything else

The magetypes tier names do **not** mean what an outside reader would guess:

| magetypes namespace | archmage token | Actual feature floor | Source |
|---|---|---|---|
| `v3` | `X64V3Token` | SSE…SSE4.2 **+ AVX + AVX2 + FMA + BMI1/2 + F16C** | `token-registry.toml` L68–80 |
| `v4` | `X64V4Token` | v3 + AVX-512 **F/BW/CD/DQ/VL** | `token-registry.toml` L120–133 |
| `v4x` | `X64V4xToken` | v4 + VBMI/VBMI2/VNNI/BITALG/IFMA/VPOPCNTDQ/GFNI/VAES/VPCLMULQDQ | `token-registry.toml` L139–155 |
| `neon` | `NeonToken` | AArch64 AdvSIMD baseline | `token-registry.toml` L180–187 |
| `wasm128` | `Wasm128Token` | `simd128` | `token-registry.toml` L248–254 |
| — | `ScalarToken` | none (array math) | — |

So **`v3` is AVX2, not SSE4.2**, and there is *no* SSE-only SIMD backend in
magetypes at all: `grep 'impl .*Backend for' magetypes/src/simd/impls/` yields
impls for exactly `ScalarToken`, `X64V3Token`, `NeonToken`, `Wasm128Token`
(43 each) plus `X64V4Token` (11) and `X64V4xToken` (19). The AVX-512 tokens
back **only the 512-bit types**; 128/256-bit work on a V4 machine goes through
`token.v3()` (see `magetypes/src/simd/generated/mod.rs`, `pub mod v4`). That
matters for this audit: an op only needs an AVX-512 implementation at the
`*x64`/`*x32` (512-bit) widths, and the SSE-era instruction set only ever
appears *inside* the AVX2 tier, never as a tier of its own.

---

## 1. Saturating integer add / sub

### Availability

| Element width | v3 (AVX2) | v4 / v4x (AVX-512) | neon | wasm128 | scalar |
|---|---|---|---|---|---|
| **8-bit** (`i8`,`u8`) | ✅ `_mm_adds_epi8` / `_mm_subs_epu8` … (`core_arch/x86/sse2.rs`), 256-bit `_mm256_adds_epi8` … (`x86/avx2.rs`) | ✅ `_mm512_adds_epi8` … (`x86/avx512bw.rs`) | ✅ `vqaddq_s8`/`vqsubq_u8` … | ✅ `i8x16_add_sat`, `u8x16_sub_sat` … (`wasm32/simd128.rs`) | ✅ `i8::saturating_add` |
| **16-bit** (`i16`,`u16`) | ✅ `_mm_adds_epi16`/`_mm_adds_epu16`/`_mm_subs_*` (sse2.rs), `_mm256_*` (avx2.rs) | ✅ `_mm512_adds_epi16` … (avx512bw.rs) | ✅ `vqaddq_s16`/`vqsubq_u16` … | ✅ `i16x8_add_sat`, `u16x8_sub_sat` … | ✅ |
| **32-bit** | ❌ **none at any x86 tier** | ❌ **none at any x86 tier** | ✅ `vqaddq_s32`,`vqsubq_u32` | ❌ none | ✅ |
| **64-bit** | ❌ | ❌ | ✅ `vqaddq_s64`,`vqsubq_u64` | ❌ | ✅ |

**How the x86 rows were established (not from memory):** an exhaustive scan of
`core_arch/src/x86` for `_adds_ep*` / `_subs_ep*` returns exactly 8 base
intrinsics per width class — `{adds,subs} × {epi8,epu8,epi16,epu16}` — in
`sse2.rs` (128-bit), `avx2.rs` (256-bit) and `avx512bw.rs` (512-bit, plus
mask/maskz variants). **There is no `_mm*_adds_epi32` / `_mm*_subs_epi32` at any
width or any tier.** Same scan of `wasm32/simd128.rs` for `_add_sat`/`_sub_sat`
returns exactly `{i8x16,u8x16,i16x8,u16x8} × {add,sub}` — again nothing at 32-bit.
NEON is the outlier that *does* have 32- and 64-bit forms
(`vqaddq_s32/u32/s64/u64`, `vqsubq_…`).

### Semantics

Identical on all five: unsigned clamps to `[0, MAX]`, signed clamps to
`[MIN, MAX]`, per lane, no lane crossing, no rounding, no flags. Verified
empirically for NEON on this host (`vqsubq_u16(3, 10) == 0`), and the scalar
reference is `core`'s `saturating_add`/`saturating_sub`, which is defined
identically.

### Verdict

**Soundly universal — at 8-bit and 16-bit only.** The user's hypothesis was
correct: a generic `saturating_add` over `i32xN` would be native on NEON and
emulated on every x86 tier and on wasm. Exposing it at 32/64-bit would mean
shipping an API whose cost differs by ~4× depending on the ISA, disguised as
one primitive.

The honest reduced surface is therefore:
`saturating_add` / `saturating_sub` on `i8xN`, `u8xN`, `i16xN`, `u16xN` — and
**not** on the 32/64-bit types. (`vqsubq_u16(thr, shifted)`, the CDEF clamp the
consumer named, is inside that surface.)

For the record, the 32-bit emulation that is being declined is not exotic —
unsigned is `a - min(a, b)` for `qsub` and `a + min(b, !a)` for `qadd`, i.e.
2 existing backend ops — but at 2–3× the instruction count of NEON's single
`vqsubq_s32` it is a portability lie, not a primitive.

---

## 2. Variable (runtime-count) shift

This is really **two** different primitives, and conflating them is where the
soundness problem lives.

### 2a. Uniform variable shift (one runtime count, same for every lane)

| Tier | left | logical right | arithmetic right | Source |
|---|---|---|---|---|
| v3, 128-bit | `_mm_sll_epi16/32/64` | `_mm_srl_epi16/32/64` | `_mm_sra_epi16/32` (**no `epi64`**) | `x86/sse2.rs`, features `sse2` |
| v3, 256-bit | `_mm256_sll_epi16/32/64` | `_mm256_srl_…` | `_mm256_sra_epi16/32` (**no `epi64`**) | `x86/avx2.rs`, `avx2` |
| v4/v4x, 512-bit | `_mm512_sll_epi16` (`avx512bw`), `_mm512_sll_epi32/64` (`avx512f`) | ditto `srl` | `_mm512_sra_epi16/32/64` | `x86/avx512bw.rs`, `x86/avx512f.rs` |
| **8-bit, any x86 tier** | ❌ no `psllb` exists | ❌ | ❌ | — |
| neon | `vshlq_{s,u}{8,16,32,64}` with a splatted positive count | same, splatted **negative** count | `vshlq_s*` with negative count | `core_arch/aarch64` |
| wasm128 | `i8x16_shl(v, amt: u32)` … `i64x2_shl` | `u8x16_shr(v, amt: u32)` … | `i8x16_shr` … | `wasm32/simd128.rs` |
| scalar | native | native | native | — |

Note the count operand shapes differ but all are *uniform*: x86 takes the count
in the low 64 bits of an XMM register (so `_mm_cvtsi32_si128(count)` supplies
it); NEON has no scalar-count form at all and needs the count splatted into a
vector (`vdupq_n_s16`), which is the same lowering this repo *already* uses for
its const shifts (`magetypes/src/simd/impls/arm_neon.rs`:
`vshlq_u16(a, vdupq_n_s16((-N) as i16))`); wasm takes a plain `u32`.

### 2b. Per-lane variable shift (a different count per lane)

| Tier | 8-bit | 16-bit | 32-bit | 64-bit |
|---|---|---|---|---|
| v3 (AVX2) | ❌ | ❌ | ✅ `_mm_sllv_epi32`/`_mm256_sllv_epi32` (`avx2`) | ✅ `…v_epi64` |
| v4/v4x | ❌ | ✅ `_mm{,256}_sllv_epi16` requires **`avx512bw,avx512vl`**; `_mm512_sllv_epi16` requires `avx512bw` | ✅ | ✅ |
| neon | ✅ `vshlq_u8` | ✅ `vshlq_u16` | ✅ | ✅ |
| wasm128 | ❌ | ❌ | ❌ | ❌ |

**Confirmed: the user's hypothesis is right.** Per-lane 16-bit variable shift is
an AVX-512BW+VL instruction, wasm128 has *no* per-lane variable shift at any
width, and 8-bit per-lane variable shift exists only on NEON. A generic per-lane
`shlv` would be one instruction on NEON and a multi-instruction emulation on
v3 and wasm — i.e. exactly the portability lie described above.

### Semantic divergence at out-of-range counts — the real hazard

The three ISAs disagree, and they disagree *silently*:

| count ≥ lane bits | x86 | NEON | wasm128 |
|---|---|---|---|
| `shl`, `shr_logical` | result **0** | result **0** | count taken **modulo** lane width |
| `shr_arithmetic` | **sign fill** (count clamped to bits−1) | **sign fill** | count taken **modulo** lane width |

Sources.
*x86:* stdarch's own const-shift implementations encode the hardware rule —
`_mm_slli_epi16` is `if IMM8 >= 16 { _mm_setzero_si128() } else { … }` and
`_mm_srli_epi16` likewise (`core_arch/x86/sse2.rs`), while `_mm_srai_epi16` is
`simd_shr(a, splat(IMM8.min(15)))`. The register-count forms
(`psllw`/`psrlw`/`psraw`) carry the identical Intel-documented rule.
*wasm:* the doc comment on `u16x8_shr` states verbatim *"Only the low bits of the
shift amount are used if the shift amount is greater than the lane width."*
(`core_arch/wasm32/simd128.rs`).
*NEON:* measured on this host (aarch64-apple-darwin, rustc 1.98) rather than
inferred — see the probe results below. USHL/SSHL read the **low 8 bits of each
shift-amount lane as a signed byte**, so a count of 256 wraps to a *no-op*, which
is why a clamp is required even though counts 16…127 already behave correctly:

```
n=  0 ushl_right=0x8001 ushl_left=0x8001 sshl_neg=    -2 sshl_pos= 16385
n= 15 ushl_right=0x0001 ushl_left=0x8000 sshl_neg=    -1 sshl_pos=     0
n= 16 ushl_right=0x0000 ushl_left=0x0000 sshl_neg=    -1 sshl_pos=     0
n=127 ushl_right=0x0000 ushl_left=0x0000 sshl_neg=    -1 sshl_pos=     0
   (input 0x8001 unsigned / -2 and 0x4001 signed, uint16x8_t / int16x8_t)
```

Also note `vshlq_*` treats a **negative** count as a right shift — x86 has no
such behaviour — so the sign of the count operand is part of the ABI of any
shared API and must never be exposed raw.

### Convention adopted (and why)

Define the portable op by the **strictest, fully-defined** semantics — count
≥ lane bits yields `0` for `shl`/`shr_logical` and a sign fill for
`shr_arithmetic` — for every `u32` count, with no UB and no "don't do that".

This is the same move `std::simd::swizzle_dyn` makes for out-of-range indices
(<https://shnatsel.github.io/improving-std-simd-swizzle-dyn/>): it guarantees
"the values for out-of-bounds indices will be set to `0`" rather than inheriting
each ISA's wrap-or-zero behaviour. That article also makes the point that the
strict guarantee is not merely a tax — because it often *is* the native
behaviour on some ISA, and can then be exploited to make the emulation cheap.
That holds here exactly: the chosen contract is **free on x86 and free on NEON
for counts ≤ 127** (it is the hardware behaviour), and costs wasm128 one scalar
`min`/mask. The alternative — "counts ≥ lane bits are the caller's problem" —
would make a byte-exact port silently ISA-dependent, which for the consumer
(SVT-AV1 CDEF) is the one unacceptable outcome.

### Verdict

**Uniform variable shift: soundly universal at every tier and every width where
the existing `*_const` shifts exist** (8/16/32/64-bit), with the contract above.
Native on x86 (16/32/64-bit) and NEON; native-with-a-scalar-clamp on NEON/wasm;
on **8-bit x86** it is emulated exactly the way this repo *already* emulates
`shl_const` for `i8x16` (16-bit shift + byte mask — see
`xtask/src/simd_types/backend_gen_remaining_int.rs::generate_x86_int_shifts`),
so it introduces no new emulation class.

**Per-lane variable shift: NOT viable as a portable primitive.** Do not expose
it. CDEF does not need it — `max(damping − msb(strength), 0)` is one scalar per
call, which is the uniform form.

---

## 3. Integer widening / narrowing

### Widening (N lanes of width W → N/2 lanes of width 2W, low and high halves)

| Tier | u8→u16 / i8→i16 | u16→u32 / i16→i32 | Lane order | Source |
|---|---|---|---|---|
| v3, 128→128 | `_mm_cvtepu8_epi16` / `_mm_cvtepi8_epi16` (`sse4.1`) for the low half; high half via a byte-shift then the same | `_mm_cvtepu16_epi32` (`sse4.1`) | natural | `x86/sse41.rs` |
| v3, 256→256 | `_mm256_cvtepu8_epi16(__m128i) -> __m256i` (`avx2`) applied to `castsi256_si128` / `extracti128_si256::<1>` | `_mm256_cvtepu16_epi32` | **natural** | `x86/avx2.rs` |
| v4/v4x, 512 | `_mm512_cvtepu8_epi16` (`avx512bw`), `_mm512_cvtepu16_epi32` (`avx512f`) | ditto | natural | `x86/avx512bw.rs`, `avx512f.rs` |
| neon | `vmovl_u8` / `vmovl_high_u8` (and `_s8`) | `vmovl_u16`/`vmovl_high_u16` | natural | `core_arch/aarch64` |
| wasm128 | `u16x8_extend_low_u8x16` / `…_high_…`, `i16x8_extend_low_i8x16` | `u32x4_extend_low_u16x8`, `i32x4_extend_low_i16x8` | natural | `wasm32/simd128.rs` |
| scalar | `as` casts | `as` casts | natural | — |

**The user's mirrored-hazard hypothesis for widening does not fire.** The reason
is that the correct AVX2 primitive is `vpmovzxbw`, not `punpcklbw`:
`_mm256_cvtepu8_epi16` is implemented in stdarch as
`transmute::<i16x16,_>(simd_cast(a.as_u8x16()))` (`x86/avx2.rs`:884) — a plain
element-wise cast of the 16-byte input, with no lane crossing and no reordering
to undo. The lane-crossing trap is real for `unpacklo`/`unpackhi`, which is the
instruction pair one reaches for by habit; it is not real for `cvtep*`. NEON is
the tier that needs the *explicit* low/high pair (`vmovl_u8` takes a 64-bit
`uint8x8_t`, `vmovl_high_u8` takes the 128-bit register), which is a signature
difference, not a semantic one.

### Narrowing (two N-lane inputs of width W → 2N lanes of width W/2)

This is where portability breaks, in two independent ways.

**(i) Lane order — hypothesis CONFIRMED.** `_mm256_packus_epi16(a, b)` does not
produce `[a…, b…]`. stdarch spells the permutation out (`x86/avx2.rs`:2395):

```rust
const IDXS: [u32; 32] = [
    00, 02, 04, 06, 08, 10, 12, 14, // a-lo i16 to u8 conversions
    32, 34, 36, 38, 40, 42, 44, 46, // b-lo
    16, 18, 20, 22, 24, 26, 28, 30, // a-hi
    48, 50, 52, 54, 56, 58, 60, 62, // b-hi
];
```

i.e. the result is `[a.lo, b.lo, a.hi, b.hi]` in 64-bit groups. The 128-bit form
`_mm_packus_epi16` *is* natural (`IDXS = [0,2,…,14, 16,18,…,30]`,
`x86/sse2.rs`:1558), and AVX-512's `vpmovwb` family (`_mm512_cvtepi16_epi8`,
`_mm512_cvtusepi16_epi8`) is natural too. So **only AVX2** is out of order, and
correcting it costs one `_mm256_permute4x64_epi64::<0xD8>` (`avx2`) — a
lane-crossing shuffle with 3-cycle latency. A generic narrow that skipped that
fixup would be *silently wrong on exactly one tier* — the worst possible outcome
for a byte-exact port, and precisely what the consumer must not ship.

**(ii) Source signedness — an unflagged hazard, and the bigger one.** x86 and
wasm only offer narrowing that reads the source as **signed**:

| Shape | v3 / v4 | neon | wasm128 |
|---|---|---|---|
| i16 → u8, unsigned saturate | ✅ `_mm_packus_epi16` / `_mm256_packus_epi16` / `_mm512_packus_epi16` | ✅ `vqmovun_s16` | ✅ `u8x16_narrow_i16x8` |
| i16 → i8, signed saturate | ✅ `_mm_packs_epi16` … | ✅ `vqmovn_s16` | ✅ `i8x16_narrow_i16x8` |
| **u16 → u8, unsigned saturate** | ❌ no such instruction | ✅ `vqmovn_u16` | ❌ |
| **truncating narrow (no saturation)** | ❌ (mask + `packus`, or AVX-512 `vpmovwb`) | ✅ `vmovn_u16` | ❌ |

The x86 `packus` family clamps the source *as `i16`* — stdarch shows it directly
as `simd_imax(simd_imin(a.as_i16x8(), splat(255)), splat(0))`. So a `u16` lane of
`0x8000` (= `-32768` as `i16`) narrows to **0** on x86/wasm and to **255** on
NEON. A generic `narrow_saturating(u16xN, u16xN) -> u8x2N` would therefore
disagree across ISAs on every input above `0x7FFF`.

### Verdict

- **`widen_low` / `widen_high` (u8→u16, i8→i16, u16→u32, i16→i32): soundly
  universal.** Native at every tier and every width, natural lane order
  everywhere, no divergence. Cost is 1 instruction per half on x86/wasm/AVX-512
  and 1 on NEON (plus, at 256-bit on v3, a free `castsi256_si128` or a 1-cycle
  `extracti128`).
- **Narrowing: reduced surface only.** The exportable shapes are
  **signed-source** `i16 → u8` (unsigned saturate) and `i16 → i8` (signed
  saturate) — plus the same at 32→16. Everything else (`u16 → u8` saturating,
  truncating narrow) is NEON-only and must be emulated, and the AVX2 arm needs a
  documented `permute4x64` fixup that must be pinned by an explicit lane-order
  assertion, not by a checksum.
- **Rounding narrowing shift** (`vrshrn_n_*`, `vqrshrun_n_*` — the AV1
  workhorse): NEON-only as a single instruction. It *is* definable portably
  (add rounding bias, arithmetic-shift, narrow-with-saturation), but on x86 and
  wasm that is 3–4 ops against NEON's 1. It belongs in the "expose only with a
  measured benchmark attached" bucket, not in this pass.

---

## 4. Summary table

| Primitive | Universal? | Exposed surface |
|---|---|---|
| `saturating_add` / `saturating_sub` | ✅ at 8/16-bit; ❌ at 32/64-bit (x86 + wasm lack it entirely) | `i8xN`, `u8xN`, `i16xN`, `u16xN` |
| uniform variable shift (`shl_uniform`, `shr_logical_uniform`, `shr_arithmetic_uniform`) | ✅ every tier, every width, with a strict out-of-range contract | same widths as the existing `*_const` shifts |
| per-lane variable shift | ❌ (16-bit needs AVX-512BW+VL; wasm has none at any width) | not exposed |
| `widen_low` / `widen_high` | ✅ | u8↔u16, i8↔i16, u16↔u32, i16↔i32 |
| narrowing, signed source, saturating | ✅ with an AVX2 `permute4x64` fixup | i16→u8, i16→i8 (and 32→16) |
| narrowing, unsigned source / truncating | ❌ NEON-only natively | not exposed |
| rounding narrowing shift | ⚠️ definable, 3–4× cost off NEON | not exposed in this pass |

## 5. What could not be verified locally

This audit was performed on `aarch64-apple-darwin`. Locally *executed*
verification therefore covers: the scalar backend, the NEON backend (native),
the wasm128 backend (via `wasmtime`), and SSE-level x86 semantics (via Rosetta,
which reports `sse4.1=true, avx=false, avx2=false`). **AVX2 and AVX-512 arms
cannot be executed on this machine** — Rosetta 2 and Docker Desktop's amd64
emulation both report `avx=false`. Those arms are covered by CI
(`.github/workflows/ci.yml`: `test-x64` on x86 runners for AVX2, and the `sde`
job under Intel SDE `-hsw`/`-skx`/`-icl` for AVX2 and AVX-512). Every claim
about them above is sourced from the vendored `stdarch` implementation — which,
for the lane-order questions that matter most, is written in portable
`simd_shuffle!` with explicit indices and is therefore *readable* rather than
opaque.

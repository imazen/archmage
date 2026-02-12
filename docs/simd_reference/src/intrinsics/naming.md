# Naming Conventions

Each platform has its own intrinsic naming scheme. This page maps between them and highlights where the same conceptual operation has different semantics.

## x86 Intrinsics

```
_mm[width]_[op]_[suffix]
```

| Part | Values | Example |
|------|--------|---------|
| `_mm` | Always present | |
| width | (empty)=128, `256`=256, `512`=512 | `_mm256_` |
| op | `add`, `mul`, `cmp`, etc. | `_mm256_add_` |
| suffix | Type indicator | `_mm256_add_ps` |

### Suffixes

| Suffix | Meaning | Element type |
|--------|---------|-------------|
| `_ps` | Packed single | `f32` |
| `_pd` | Packed double | `f64` |
| `_ss` | Scalar single | `f32` (lowest lane only) |
| `_sd` | Scalar double | `f64` (lowest lane only) |
| `_epi8/16/32/64` | Packed signed int | `i8`, `i16`, `i32`, `i64` |
| `_epu8/16/32/64` | Packed unsigned int | `u8`, `u16`, `u32`, `u64` |
| `_si128/256/512` | Bitwise on full register | Any |

## ARM NEON Intrinsics

```
v[op][modifier]q_[type]
```

| Part | Values | Example |
|------|--------|---------|
| `v` | Vector prefix (always) | |
| op | `add`, `mul`, `min`, etc. | `vadd` |
| modifier | Optional | `vaddl` (long/widening) |
| `q` | Quadword (128-bit) | `vaddq_f32` |
| type | `f32`, `s32`, `u8`, etc. | |

### Modifiers

| Modifier | Meaning | Example |
|----------|---------|---------|
| `l` | Long (widen result) | `vaddlq_s16` → i32 result |
| `n` | Narrow (shrink result) | `vaddhn_s32` → i16 result |
| `h` | High half | `vaddhn_high_s32` |
| `p` | Pairwise | `vpaddq_f32` |
| `a` | Accumulate | `vmlaq_f32` (multiply-accumulate) |
| `r` | Rounding | `vrndnq_f32` |

### NEON type suffixes

| Suffix | Meaning |
|--------|---------|
| `f32` | 32-bit float |
| `f64` | 64-bit float |
| `s8/s16/s32/s64` | Signed integer |
| `u8/u16/u32/u64` | Unsigned integer |

## WASM SIMD Intrinsics

```
[type]x[lanes]_[op]
```

| Part | Values | Example |
|------|--------|---------|
| type | `f32`, `i32`, `u8`, etc. | `f32x4_add` |
| lanes | Number of lanes | |
| op | `add`, `mul`, etc. | |

WASM uses a single `v128` runtime type for everything — the function name determines interpretation.

### Examples

| Operation | x86 | NEON | WASM |
|-----------|-----|------|------|
| Add f32×4 | `_mm_add_ps` | `vaddq_f32` | `f32x4_add` |
| Mul i32×4 | `_mm_mullo_epi32` | `vmulq_s32` | `i32x4_mul` |
| Min f32×4 | `_mm_min_ps` | `vminq_f32` | `f32x4_min` |
| Load f32×4 | `_mm_loadu_ps` | `vld1q_f32` | `v128_load` |
| Splat f32 | `_mm_set1_ps` | `vdupq_n_f32` | `f32x4_splat` |
| Zero | `_mm_setzero_ps` | `vdupq_n_f32(0.0)` | `f32x4_splat(0.0)` |

## Semantic differences (traps for the unwary)

These operations have the same *name* across platforms but different *semantics*. See [Behavioral Differences](../cross-platform/differences.md) for full details.

| Operation | Difference |
|-----------|-----------|
| **FMA argument order** | x86: `fmadd(a,b,c) = a*b+c`. NEON: `fma(a,b,c) = a+b*c`. |
| **Blend/select** | x86: `blendv(false, true, mask)`. NEON: `bsl(mask, true, false)`. |
| **Comparison returns** | x86: returns same float type. NEON: returns uint type. |
| **Float bitwise** | x86: direct `_mm_and_ps`. NEON: cast to int, operate, cast back. |
| **Reciprocal precision** | x86: ~12 bits. NEON: ~8 bits (needs more Newton-Raphson). |

Magetypes handles all these differences internally. When using raw intrinsics, you need to account for them yourself.

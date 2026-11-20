# Archmage Safety Model & Idiomatic Usage

Authoritative reference for understanding and teaching archmage. Read before writing docs or examples.

> **`#![forbid(unsafe_code)]` compatible.** Downstream crates can use `#![forbid(unsafe_code)]` when combining archmage tokens with `#[attune]` (`import_intrinsics` or `define(...)`), `attuned!`, and `reattune!`. The `unsafe` lives inside archmage's generated boundary wrappers, not yours.

> **Unified macros and legacy aliases.** [`docs/ATTUNE.md`](ATTUNE.md) documents `#[attune]`, `attuned!`, and `reattune!`. Legacy `#[arcane]`, `#[rite]`, `#[magetypes]`, `#[autoversion]`, and `incant!` (plus descriptive aliases `#[token_target_features_boundary]`, `#[token_target_features]`, and `dispatch_variant!`) remain supported.

> **Source of truth.** All token-to-feature mappings are defined in [`token-registry.toml`](../token-registry.toml). Everything else (generated code, macro registries, docs) is derived from it.

## Terminology

| Term | Definition |
|------|------------|
| **safe** | No `unsafe` keyword required |
| **unsafe** | Requires `unsafe` keyword (Rust can't verify the invariant) |
| **sound** | Cannot cause UB when used as documented |
| **unsound** | CAN cause UB even when used correctly |

These are orthogonal. Archmage's `#[attune(wrap)]` (and `_tier_t` proof wrappers) generate an internal `unsafe` boundary call that is sound because the authenticated token proves the CPU features exist.

## The Core Safety Model

### 1. Tokens Are Proofs

```rust
#[derive(Clone, Copy)]
pub struct X64V3Token {
    _private: (),  // zero-sized; the private field blocks `X64V3Token {}` outside archmage
}

impl SimdToken for X64V3Token {
    fn summon() -> Option<Self> {
        if /* runtime CPUID check */ {
            Some(Self { _private: () })  // Existence = proof features are available
        } else {
            None
        }
    }
}
```

Safe code gets a token three ways: `summon()`, which checks the CPU; an
extraction from a stronger token (`v4.v3()`); or `from_context()` inside a
function whose `#[target_feature]` set covers the token's, where rustc checks
the claim. If you have a token, the features are available.

### 2. Proof Wrappers Generate Safe-to-Call Code

```rust
use archmage::prelude::*;

// What you write:
#[attune(wrap, import_intrinsics)]
fn kernel(_token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = _mm256_loadu_ps(data);  // Safe inside #[target_feature]
    _mm256_cvtss_f32(_mm256_castps256_ps128(v))
}
```

`#[attune(wrap)]` (or `#[attune] fn kernel_v3_t(...)`) generates the `#[target_feature]` boundary and its token-justified internal call.
User code does not implement that boundary; see [`docs/ATTUNE.md`](ATTUNE.md) and the expansion tests for exact output.

The outer function is safe. The `unsafe` is an implementation detail justified by the token.

### 3. Rust 1.86-1.87 Intrinsic Safety

Value-based intrinsics are safe inside `#[target_feature]` functions:

```rust
use archmage::prelude::*;

#[attune(wrap, import_intrinsics)]
fn example(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let a = _mm256_loadu_ps(data);
    let b = _mm256_add_ps(a, a);
    let mut out = [0.0; 8];
    _mm256_storeu_ps(&mut out, b);
    out
}
```

## Two Crates

**Archmage** provides tokens, macros, and detection. Minimal.

**Magetypes** provides SIMD types (`f32x8`, `i32x4`), token-first `*_t` constructors, operators, methods, transcendentals, and cross-platform polyfills.

## Idiomatic Patterns

### Direct `#[attune(v3)]` inside, `#[attune(wrap)]` at the boundary

Direct feature bodies (`#[attune(v3)]`, `#[attune] fn helper_v3(...)`, or multi-tier `#[attune(_v3, _neon)]` / `#[attune(_*)]`) add `#[target_feature]` + `#[inline]` directly, so LLVM inlines them into callers with matching or superset features. Proof wrappers (`#[attune(wrap)]`, `#[attune] fn entry_v3_t(...)`, or `#[attune(_*_t, ...)]`) wrap a private `#[target_feature]` body in a safe outer function for calls from non-SIMD code.

Use direct bodies when no wrapper is needed inside an already-attuned context. Use `#[attune(_*_t, define(f32x8))]` + `attuned!` (or `#[attune(dispatch)]`) when compiling the same kernel across portable architectures.

```rust
use archmage::prelude::*;

pub fn public_api(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = X64V3Token::summon() {
        return process_simd(token, data);
    }
    data.iter().sum()
}

#[attune(wrap, import_intrinsics)]  // Boundary: called from non-SIMD code
fn process_simd(_token: X64V3Token, data: &[f32]) -> f32 {
    let (chunks, remainder) = data.as_chunks::<8>();
    let mut sum = 0.0;
    for chunk in chunks {
        sum += process_chunk(chunk);  // direct V3 call, no token needed
    }
    sum + remainder.iter().copied().sum::<f32>()
}

#[attune(v3, import_intrinsics)]  // Direct V3 body: inlines, no wrapper
fn process_chunk(chunk: &[f32; 8]) -> f32 {
    let v = _mm256_loadu_ps(chunk);
    let hi = _mm256_extractf128_ps::<1>(v);
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(v), hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}
```

Calling a proof wrapper per iteration from non-SIMD code crosses the `#[target_feature]` boundary every iteration (4–6x slower depending on workload; see [benchmark data](PERFORMANCE.md)). Direct `#[attune(v3)]` helpers share the caller's features and inline with zero boundary overhead.

### Enter the proof wrapper once, use direct helpers inside

The cost isn't `summon()` (~1.3 ns cached) — it's the `#[target_feature]` boundary. Each wrapper call from non-SIMD code crosses a feature mismatch that LLVM cannot inline across. Hoisting the token outside the loop doesn't help — put the loop inside the `#[attune]` entry point and keep helpers in matching feature contexts.

### Concrete tokens for hot paths

`#[attune(wrap)]` enables the features its proof parameter names. With a tier-trait bound,
that is the bound's tier, whichever concrete token the caller passes:

```rust
use archmage::prelude::*;

// Compiled with V2 features (SSE4.2), even when called with an X64V4Token
#[attune(wrap, import_intrinsics)]
fn process<T: HasX64V2>(_token: T, data: &[f32]) -> f32 {
    data.iter().sum()
}

// Compiled with V3 features (AVX2, FMA)
#[attune(wrap, import_intrinsics)]
fn process_v3(_token: X64V3Token, data: &[f32]) -> f32 {
    data.iter().sum()
}
```

Generics are not the cost: they monomorphize, and a generic helper marked `#[inline(always)]` inlines into
a caller whose features cover it. The cost is the feature set of the region the
hot loop runs in, and any call from it into a stronger-feature function, which
is a boundary. Passing a stronger token down is free (`v4.v3()`). Dispatch once
at the entry point, with the concrete token for the tier you want.

### Memory operations via `import_intrinsics`

`import_intrinsics` brings safe memory ops into scope — they take references instead of raw pointers:

```rust
use archmage::prelude::*;

#[attune(wrap, import_intrinsics)]
fn load_and_square(_token: X64V3Token, data: &[f32; 8]) -> __m256 {
    let v = _mm256_loadu_ps(data);  // Takes &[f32; 8], not *const f32
    _mm256_mul_ps(v, v)
}
```

For high-level portable code, prefer magetypes (`#[attune(_*_t, define(f32x8))]` + `f32x8::load_t(token, data)`).

## The Soundness Invariant

```text
features_enabled_by_attune(Token) ⊆ features_checked_by_summon(Token)
```

`cargo xtask validate` checks the right side: each `summon()` tests every
feature `token-registry.toml` lists. The macros' feature lists are generated
from the same registry, and `just check-generated` fails if they drift.

## Teaching Checklist

When explaining archmage:

1. Tokens are zero-sized proofs of CPU features
2. `summon()` does the runtime check, returns `Option<Token>`
3. `#[attune(wrap)]` and `_tier_t` generate safe `#[target_feature]` boundary wrappers; `#[attune(v3)]` and `_*` generate direct feature bodies
4. Inside `#[target_feature]`, value-based intrinsics are safe (Rust 1.87+)
5. Enter a proof wrapper once at the boundary, and use direct `#[attune]` bodies (or `#[inline(always)]` generic helpers) inside
6. `#[attune(_*_t, define(f32x8))]` + `attuned!` (or `#[attune(dispatch)]`) generates and dispatches portable multi-tier families (explicitly adding `+v4(cfg(avx512))` when AVX-512 is wanted)
7. Token-first magetypes constructors use `*_t(token, ...)` (`splat_t`, `load_t`, `zero_t`, `from_array_t`, `from_slice_t`); receiver methods (`v.store(out)`, `v.mul_add_relaxed(b, c)`, `v.mul_add_fused(b, c)`) take no token
8. A trait bound on `#[attune(wrap)]` compiles the body with the bound's tier; take a concrete token for the tier you want
9. `import_intrinsics` provides safe memory operations (references, not pointers)
10. magetypes provides high-level portable SIMD types

When showing examples:

1. Show the portable path first (`#[attune(_*_t, define(f32x8))]` + `attuned!` + `*_t` constructors) or `#[attune(wrap)]` for single-arch raw intrinsics
2. Explain what the boundary wrapper generates (for understanding)
3. Memory ops use references (via `import_intrinsics` or magetypes `load_t`/`store`), not raw pointers
4. Include the `summon()` or `attuned!` call in context (with `#[cfg(target_arch)]` on manual single-arch call sites)
5. Show `#[attune(v3)]` for single-tier helpers and `#[attune(_*)]` / `#[attune(_v3, _neon)]` for multi-tier helpers

## Banned from Docs

These historical prelude aliases were removed in 0.9.27 and must not appear in current documentation or examples:

| Alias | Use instead |
|-------|-------------|
| `F32Vec`, `I32Vec`, etc. | `f32x8`, `f32x4`, `i32x8`, `i32x4` |
| `RecommendedToken` | `X64V3Token`, `Arm64`, `Wasm128Token` |
| `LANES` (outside tier namespaces) | Explicit: `8`, `4`, or width in type name |

Fixed-width examples should make the lane count explicit. `#[attune(..., define(f32x8))]` (and legacy `#[magetypes]`) substitutes
`Token`; the old `f32xN` and `LANES` substitutions were removed in `36c8caf`.
Tier namespaces still provide natural-width `f32xN` aliases and `LANES_F32`
constants, imported with `#[attune(..., import_magetypes)]`.

## Cross-Architecture

All tokens exist on all architectures. On unsupported architectures, `summon()` returns `None`. `#[attune]` omits inapplicable target-architecture functions via `#[cfg(target_arch)]`, so use `attuned!` / `#[attune(dispatch)]` or `#[cfg(target_arch)]` guards at manual call sites:

```rust
use archmage::prelude::*;

#[attune(wrap, import_intrinsics)]
fn x86_kernel(_token: X64V3Token, data: &[f32; 8]) -> f32 {
    // On ARM/WASM: this function is omitted via #[cfg(target_arch)]
    _mm256_cvtss_f32(_mm256_castps256_ps128(_mm256_loadu_ps(data)))
}

// Guard manual single-arch call sites, or use attuned! for multi-arch dispatch
#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() {
    let _ = x86_kernel(token, &data);  // Only exists on x86_64
}
```

### Eliminating Runtime Dispatch

| Platform | Compile Flag | Effect |
|----------|--------------|--------|
| x86-64 AVX2 | `-Ctarget-cpu=haswell` | `X64V3Token::summon()` compiles away |
| x86-64 AVX-512 | `-Ctarget-cpu=skylake-avx512` | `Server64::summon()` compiles away |
| AArch64 | (default target) | `Arm64::summon()` always succeeds (NEON is baseline) |
| WASM | `--target wasm32-unknown-unknown -Ctarget-feature=+simd128` | `Wasm128Token::summon()` compiles away |

## Open Design Questions

1. **Implicit token downcasting:** Should `impl From<X64V4Token> for X64V3Token` exist? Not implemented; use the explicit extraction methods (`v4.v3()`).

Vector halves are explicit too: `f32x8::low()`, `high()` and `split()` return
`f32x4` values.

## Missing Methods

| Method | Description | Workaround |
|--------|-------------|------------|
| `signum` | Returns -1, 0, or 1 | Comparison + blend |
| `tanh` / `tanh_lowp` | Hyperbolic tangent | `(exp(2x) - 1) / (exp(2x) + 1)` |
| `sin` / `cos` | Trigonometric | Not implemented |

Float vectors implement `&`, `|` and `^` directly.

Reciprocals come in three tiers. `rcp_approx()` and `rsqrt_approx()` take the
cheapest path on each platform, at least about 12 bits, with unspecified results
at ±0, ±inf and NaN. `recip()` and `rsqrt()` are within 4 ULP and exact at ±0,
±inf and NaN; V3 `recip` flushes results below the normal range to zero.
`recip_portable()` and `rsqrt_portable()` are exact (0 ULP) and give the same
bits on every backend. The method docs give each platform's lowering.

## License

MIT OR Apache-2.0

## Documentation audit and remaining API gaps

The public guides now lead with complete call chains adapted from `zenfilters`,
`zenblend`, `zenwebp`, and `linear-srgb`, with pinned source links and explicit
adaptation notes. Their executable counterparts are in
`magetypes/tests/doc_examples.rs`. Entry points use `#[attune]` (or legacy
`#[magetypes]` / `#[arcane]`); `#[inline(always)]` on a generic helper does not
establish target features.

Authored Markdown and browser examples no longer contain explicit unsafe blocks.
Macro expansion internals are explained in prose. The intrinsic registry retains
upstream documentation as data; the browser omits upstream code fences and does
not offer a callable example when no safe wrapper exists.

| Gap | Current safe approach | Requirement before adding an abstraction |
|---|---|---|
| Portable gather/scatter | Checked slice indexing and ordinary loads/stores. On AVX-512 only, `u32x16`, `i32x16` and `f32x16` on `X64V4Token` have bounds-safe `gather_wrapping`, `gather_or` and `scatter_select`. | Define bounds, masking, scale, and duplicate scatter ordering for other widths and backends, and measure against scalar lookups: a hardware gather is not reliably faster. |
| Non-temporal stores | Ordinary reference-based stores | Encode alignment, writable extent, and completion/fence obligations. |
| Generic prefetch | Leave prefetch out of portable examples | A reference-based interface and evidence of a useful consumer. |
| Floating-point environment changes | Use operations with the required documented semantics | An ambient MXCSR/FTZ/DAZ change affects surrounding code; a local token alone is insufficient. |

These gaps are not reasons to expose a safe function that accepts unchecked raw
pointers. The ISA quirks page separately records numerical portability limits,
including V3 reciprocal underflow for large finite inputs, and distinguishes
measured float fixups from integer fixups whose isolated timings remain unmeasured.

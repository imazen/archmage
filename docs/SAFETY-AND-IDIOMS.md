# Archmage Safety Model & Idiomatic Usage

Authoritative reference for understanding and teaching archmage. Read before writing docs or examples.

> **`#![forbid(unsafe_code)]` compatible.** Downstream crates can use `#![forbid(unsafe_code)]` when combining archmage tokens + `#[arcane(import_intrinsics)]`/`#[rite(import_intrinsics)]` macros. The `unsafe` lives inside archmage's generated code, not yours.

> **Descriptive aliases.** `#[token_target_features_boundary]` = `#[arcane]`, `#[token_target_features]` = `#[rite]`, `dispatch_variant!` = `incant!`. These help AI tools and newcomers infer what the macros do from the name alone.

> **Source of truth.** All token-to-feature mappings are defined in [`token-registry.toml`](../token-registry.toml). Everything else (generated code, macro registries, docs) is derived from it.

## Terminology

| Term | Definition |
|------|------------|
| **safe** | No `unsafe` keyword required |
| **unsafe** | Requires `unsafe` keyword (Rust can't verify the invariant) |
| **sound** | Cannot cause UB when used as documented |
| **unsound** | CAN cause UB even when used correctly |

These are orthogonal. Archmage's `#[arcane]` generates unsafe code that IS sound — the token proves features exist.

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

### 2. `#[arcane]` Generates Safe-to-Call Code

```rust
// What you write:
#[arcane(import_intrinsics)]
fn kernel(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = _mm256_setzero_ps();  // Safe inside #[target_feature]!
    // ...
}
```

The macro generates the target-feature boundary and its justified internal call.
User code does not implement that boundary; see the expansion tests for exact output.

The outer function is safe. The `unsafe` is an implementation detail justified by the token.

### 3. Rust 1.86-1.87 Changed Everything

Value-based intrinsics are safe inside `#[target_feature]` functions:

```rust
use archmage::prelude::*;

#[arcane(import_intrinsics)]
fn example(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let a = _mm256_loadu_ps(data);
    let b = _mm256_add_ps(a, a);
    let mut out = [0.0; 8];
    _mm256_storeu_ps(&mut out, b);
    out
}
```

## Two Crates

**Archmage** provides tokens, macros, and detection. Minimal.

**Magetypes** provides SIMD types (`f32x8`, `i32x4`), operators, methods, transcendentals, and cross-platform polyfills. Maximal.

## Idiomatic Patterns

### `#[rite]` inside, `#[arcane]` at the boundary

`#[rite]` adds `#[target_feature]` + `#[inline]` directly, so LLVM inlines it into callers with matching features. `#[arcane]` generates an inner `#[target_feature]` function called from a safe outer function (needed when transitioning from non-SIMD code — this crossing is the target-feature boundary).

`#[rite]` works three ways: **token-based** (`#[rite]` with a token parameter), **tier-based** (`#[rite(v3)]` with no token), and **multi-tier** (`#[rite(v3, v4, neon)]` generating suffixed variants). Token-based and tier-based produce identical code — the token form can be easier to remember if you already have the token. Multi-tier generates one function per tier (`fn_v3`, `fn_v4`, `fn_neon`), each compiled with different features. Use tier-based when the token is just being threaded through unused. Use multi-tier when you want the same body compiled for multiple architectures.

```rust
pub fn public_api(data: &[f32]) -> f32 {
    if let Some(token) = X64V3Token::summon() {
        process_simd(token, data)
    } else {
        data.iter().sum()
    }
}

#[arcane(import_intrinsics)]  // Boundary: called from non-SIMD code
fn process_simd(token: X64V3Token, data: &[f32]) -> f32 {
    let mut sum = 0.0;
    for chunk in data.chunks_exact(8) {
        sum += process_chunk(chunk.try_into().unwrap());  // no token needed!
    }
    sum
}

#[rite(v3, import_intrinsics)]  // Tier-based: inlines, no token parameter
fn process_chunk(chunk: &[f32; 8]) -> f32 {
    let v = _mm256_loadu_ps(chunk);
    // horizontal sum...
    let sum = _mm256_hadd_ps(v, v);
    let sum = _mm256_hadd_ps(sum, sum);
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}
```

Calling `#[arcane]` from a hot loop crosses the `#[target_feature]` boundary every iteration (4-6x slower depending on workload; see [benchmark data](PERFORMANCE.md)). `#[rite]` inlines into callers with matching features — no boundary.

### Enter `#[arcane]` once, use `#[rite]` inside

The cost isn't `summon()` (~1.3 ns cached) — it's the `#[target_feature]` boundary. Each `#[arcane]` call from non-SIMD code crosses a boundary that LLVM can't inline across. Even hoisting the token outside the loop doesn't help — you need the loop *inside* `#[arcane]` with `#[rite]` helpers.

### Concrete tokens for hot paths

`#[arcane]` enables the features its signature names. With a tier-trait bound,
that is the bound's tier, whichever token the caller passes:

```rust
// Compiled with V2 features (SSE4.2), even when called with an X64V4Token
#[arcane(import_intrinsics)]
fn process<T: HasX64V2>(token: T, data: &[f32]) -> f32 { ... }

// Compiled with V3 features (AVX2, FMA)
#[arcane(import_intrinsics)]
fn process_v3(token: X64V3Token, data: &[f32]) -> f32 { ... }
```

Generics are not the cost: they monomorphize, and a generic helper inlines into
a caller whose features cover it. The cost is the feature set of the region the
hot loop runs in, and any call from it into a stronger-feature function, which
is a boundary. Passing a stronger token down is free (`v4.v3()`). Dispatch once
at the entry point, with the concrete token for the tier you want.

### Memory operations via `import_intrinsics`

`import_intrinsics` brings safe memory ops into scope — they take references instead of raw pointers:

```rust
use archmage::prelude::*;

#[arcane(import_intrinsics)]
fn load_and_square(token: X64V3Token, data: &[f32; 8]) -> __m256 {
    let v = _mm256_loadu_ps(data);  // Takes &[f32; 8], not *const f32
    _mm256_mul_ps(v, v)
}
```

For high-level code, prefer magetypes (which uses safe memory ops internally).

## The Soundness Invariant

```
features_enabled_by_arcane(Token) ⊆ features_checked_by_summon(Token)
```

`cargo xtask validate` checks the right side: each `summon()` tests every
feature `token-registry.toml` lists. The macros' feature lists are generated
from the same registry, and `just check-generated` fails if they drift.

## Teaching Checklist

When explaining archmage:

1. Tokens are zero-sized proofs of CPU features
2. `summon()` does the runtime check, returns `Option<Token>`
3. `#[arcane]` generates `#[target_feature]` code
4. Inside `#[target_feature]`, most intrinsics are safe (Rust 1.87+)
5. `#[arcane]` at the boundary, `#[rite]` for everything else
6. `#[rite]` has three modes: token-based, tier-based (`#[rite(v3)]`), multi-tier (`#[rite(v3, v4, neon)]`)
7. Enter `#[arcane]` once, `#[rite]` for everything inside
8. A trait bound compiles the body with the bound's tier; take a concrete token for the tier you want
9. `import_intrinsics` provides safe memory operations (references, not pointers)
10. magetypes provides high-level SIMD types

When showing examples:

1. Show the simple path first (magetypes + `#[arcane]`)
2. Explain what the macro generates (for understanding)
3. Memory ops use references (via `import_intrinsics`), not raw pointers
4. Include the `summon()` call in context
5. Show `#[rite(v3)]` for internal helpers, `#[rite]` with token for magetypes, `#[rite(v3, v4, neon)]` for multi-tier

## Banned from Docs

These historical prelude aliases were removed in 0.9.27 and must not appear in current documentation or examples:

| Alias | Use instead |
|-------|-------------|
| `F32Vec`, `I32Vec`, etc. | `f32x8`, `f32x4`, `i32x8`, `i32x4` |
| `RecommendedToken` | `X64V3Token`, `Arm64`, `Wasm128Token` |
| `LANES` (outside `#[magetypes]`) | Explicit: `8`, `4`, or width in type name |

Fixed-width examples should make the lane count explicit. `#[magetypes]` substitutes
`Token`; its old `f32xN` and `LANES` substitutions were removed in `36c8caf`.
Tier namespaces still provide natural-width `f32xN` aliases and `LANES_F32`
constants, imported with `rite(import_magetypes)` or `arcane(import_magetypes)`.

## Cross-Architecture

All tokens exist on all architectures. On wrong arch, `summon()` returns `None`. `#[arcane]` and `#[rite]` cfg-out the function on wrong architectures (no code emitted), so use `incant!` or `#[cfg(target_arch)]` guards at call sites:

```rust
#[arcane(import_intrinsics)]
fn x86_kernel(token: X64V3Token, data: &[f32; 8]) -> f32 {
    // On ARM: this function is not emitted (cfg'd out)
    // ...
}

// Use cfg guard at the call site, or use incant! for multi-arch dispatch
#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() {
    x86_kernel(token, &data);  // Only exists on x86
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
`magetypes/tests/doc_examples.rs`. Entry points use `#[magetypes]` or `#[arcane]`;
`#[inline(always)]` on a generic helper does not establish target features.

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

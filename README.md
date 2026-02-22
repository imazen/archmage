# archmage

[![CI](https://github.com/imazen/archmage/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/archmage/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/archmage.svg)](https://crates.io/crates/archmage)
[![docs.rs](https://docs.rs/archmage/badge.svg)](https://docs.rs/archmage)
[![codecov](https://codecov.io/gh/imazen/archmage/graph/badge.svg)](https://codecov.io/gh/imazen/archmage)
[![MSRV](https://img.shields.io/crates/msrv/archmage.svg)](https://crates.io/crates/archmage)
[![License](https://img.shields.io/crates/l/archmage.svg)](https://github.com/imazen/archmage#license)

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU.

**Zero overhead.** Archmage generates identical assembly to hand-written unsafe code. The safety abstractions exist only at compile time—at runtime, you get raw SIMD instructions. Calling an `#[arcane]` function costs exactly the same as calling a bare `#[target_feature]` function directly.

**Zero `unsafe`.** Crates using archmage + magetypes + safe_unaligned_simd are required to use\* `#![forbid(unsafe_code)]`. There is no reason to write `unsafe` in SIMD code anymore.

```toml
[dependencies]
archmage = "0.8"
magetypes = "0.8"
```

## Raw intrinsics with `#[arcane]` <sub>(alias: `#[token_target_features_boundary]`)</sub>

```rust
use archmage::prelude::*;

#[arcane]
fn dot_product(_token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    let mul = _mm256_mul_ps(va, vb);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, mul);
    out.iter().sum()
}

fn main() {
    if let Some(token) = Desktop64::summon() {
        println!("{}", dot_product(token, &[1.0; 8], &[2.0; 8]));
    }
}
```

`summon()` checks CPUID. `#[arcane]` enables `#[target_feature]`, making intrinsics safe (Rust 1.85+). The prelude re-exports `safe_unaligned_simd` functions directly — `_mm256_loadu_ps` takes `&[f32; 8]`, not a raw pointer. Compile with `-C target-cpu=haswell` to elide the runtime check.

## Inner helpers with `#[rite]` <sub>(alias: `#[token_target_features]`)</sub>

**`#[rite]` should be your default.** Use `#[arcane]` only at entry points.

```rust
use archmage::prelude::*;

// Entry point: use #[arcane]
#[arcane]
fn dot_product(token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let products = mul_vectors(token, a, b);  // Calls #[rite] helper
    horizontal_sum(token, products)
}

// Inner helper: use #[rite] (inlines into #[arcane] — features match)
#[rite]
fn mul_vectors(_: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> __m256 {
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    _mm256_mul_ps(va, vb)
}

#[rite]
fn horizontal_sum(_: Desktop64, v: __m256) -> f32 {
    let sum = _mm256_hadd_ps(v, v);
    let sum = _mm256_hadd_ps(sum, sum);
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}
```

Both macros read the token type from your function signature to decide which `#[target_feature]` to emit. `Desktop64` → `avx2,fma,...`. `X64V4Token` → `avx512f,avx512bw,...`. The token type *is* the feature selector.

`#[arcane]` generates a wrapper: an outer function that calls an inner `#[target_feature]` function via `unsafe`. This is how you cross into SIMD code without writing `unsafe` yourself — but the wrapper creates an LLVM optimization boundary. `#[rite]` applies `#[target_feature]` + `#[inline]` directly, with no wrapper and no boundary. Since Rust 1.85+, calling `#[target_feature]` functions from matching contexts is safe — no `unsafe` needed between `#[arcane]` and `#[rite]` functions.

**`#[rite]` should be your default.** Use `#[arcane]` only at the entry point (the first call from non-SIMD code), and `#[rite]` for everything inside. Passing the same token type through your call hierarchy keeps every function compiled with matching features, so LLVM inlines freely.

### The cost of mismatched features

Processing 1000 8-float vector additions ([full benchmark details](docs/PERFORMANCE.md)):

| Pattern | Time | Why |
|---------|------|-----|
| `#[rite]` in `#[arcane]` | 547 ns | Features match — LLVM inlines |
| `#[arcane]` per iteration | 2209 ns (4x) | Target-feature boundary per call |
| Bare `#[target_feature]` (no archmage) | 2222 ns (4x) | Same boundary — archmage adds nothing |

The 4x penalty comes from LLVM's `#[target_feature]` optimization boundary, not from archmage. Bare `#[target_feature]` has the same cost. With real workloads (DCT-8), the boundary costs up to 6.2x.

Use `#[rite]` for helpers called from SIMD code. When the token type matches, `#[rite]` emits the same `#[target_feature]` as the caller, so LLVM inlines freely — no boundary. The token flows through your call tree, keeping features consistent everywhere it goes.

## SIMD types with `magetypes`

```rust
use archmage::{Desktop64, SimdToken};
use magetypes::simd::f32x8;

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if let Some(token) = Desktop64::summon() {
        let mut sum = f32x8::zero(token);
        for (a_chunk, b_chunk) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
            let va = f32x8::load(token, a_chunk.try_into().unwrap());
            let vb = f32x8::load(token, b_chunk.try_into().unwrap());
            sum = va.mul_add(vb, sum);
        }
        sum.reduce_add()
    } else {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}
```

`f32x8` wraps `__m256` with token-gated construction and natural operators.

## Runtime dispatch with `incant!` <sub>(alias: `dispatch_variant!`)</sub>

Write platform-specific variants with concrete types, then dispatch at runtime:

```rust
use archmage::incant;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;

#[cfg(target_arch = "x86_64")]
const LANES: usize = 8;

/// AVX2 path — processes 8 floats at a time.
#[cfg(target_arch = "x86_64")]
fn sum_squares_v3(token: archmage::X64V3Token, data: &[f32]) -> f32 {
    let chunks = data.chunks_exact(LANES);
    let mut acc = f32x8::zero(token);
    for chunk in chunks {
        let v = f32x8::from_array(token, chunk.try_into().unwrap());
        acc = v.mul_add(v, acc);
    }
    acc.reduce_add() + chunks.remainder().iter().map(|x| x * x).sum::<f32>()
}

/// Scalar fallback.
fn sum_squares_scalar(_token: archmage::ScalarToken, data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum()
}

/// Public API — dispatches to the best available at runtime.
fn sum_squares(data: &[f32]) -> f32 {
    incant!(sum_squares(data))
}
```

`incant!` looks for `_v3`, `_v4`, `_neon`, `_wasm128`, and `_scalar` suffixed functions by default, and dispatches to the best one the CPU supports. Each variant uses concrete SIMD types for its platform; the scalar fallback uses plain math.

You can specify explicit tiers to control which variants are dispatched to:

```rust
// Only dispatch to v1, v3, neon, and scalar
fn sum_squares(data: &[f32]) -> f32 {
    incant!(sum_squares(data), [v1, v3, neon])
}
```

Known tiers: `v1`, `v2`, `v3`, `v4`, `v4x`, `arm_v2`, `arm_v3`, `neon`, `neon_aes`, `neon_sha3`, `neon_crc`, `wasm128`, `scalar`. The `scalar` tier is always included implicitly.

### `#[magetypes]` for simple cases

If your function body doesn't use SIMD types (only `Token`), `#[magetypes]` can generate the variants for you by replacing `Token` with the concrete token type for each platform:

```rust
use archmage::magetypes;

#[magetypes]
fn process(token: Token, data: &[f32]) -> f32 {
    // Token is replaced with X64V3Token, NeonToken, ScalarToken, etc.
    // But SIMD types like f32x8 are NOT replaced — use incant! pattern
    // for functions that need different types per platform.
    data.iter().sum()
}
```

Specify explicit tiers to control which variants are generated:

```rust
#[magetypes(v1, v3, neon)]
fn process(token: Token, data: &[f32]) -> f32 {
    // Generates: process_v1, process_v3, process_neon, process_scalar
    data.iter().sum()
}
```

For functions that use platform-specific SIMD types (`f32x8`, `f32x4`, etc.), write the variants manually and use `incant!` as shown above.

## Tokens

| Token | Alias | Features |
|-------|-------|----------|
| `X64V1Token` | `Sse2Token` | SSE, SSE2 (x86_64 baseline — always available) |
| `X64V2Token` | | SSE4.2, POPCNT |
| `X64CryptoToken` | | V2 + PCLMULQDQ, AES-NI (Westmere 2010+) |
| `X64V3Token` | `Desktop64` | AVX2, FMA, BMI2 |
| `X64V3CryptoToken` | | V3 + VPCLMULQDQ, VAES (Zen 3+ 2020, Alder Lake 2021+) |
| `X64V4Token` | `Server64` | AVX-512 (requires `avx512` feature) |
| `NeonToken` | `Arm64` | NEON |
| `Arm64V2Token` | | + CRC, RDM, DotProd, FP16, AES, SHA2 (A55+, M1+) |
| `Arm64V3Token` | | + FHM, FCMA, SHA3, I8MM, BF16 (A510+, M2+, Snapdragon X) |
| `Wasm128Token` | | WASM SIMD |
| `ScalarToken` | | Always available |

All tokens compile on all platforms. `summon()` returns `None` on unsupported architectures. Detection is cached: ~1.3 ns after first call, 0 ns with `-Ctarget-cpu=haswell` (compiles away).

See [`token-registry.toml`](token-registry.toml) for the complete mapping of tokens to CPU features.

## Safety model

Archmage's safety rests on three pillars, all enabled by Rust 1.85+:

1. **Value-based SIMD intrinsics are safe inside `#[target_feature]` functions.** Arithmetic, shuffle, compare, and bitwise operations need no `unsafe`. Only pointer-based memory operations remain unsafe.

2. **Calling a `#[target_feature]` function from another function with matching features is safe.** No `unsafe` needed between `#[arcane]` and `#[rite]` functions — LLVM knows the features match.

3. **`safe_unaligned_simd` makes memory operations safe.** It shadows pointer-based load/store intrinsics with reference-based alternatives (e.g., `_mm256_loadu_ps` takes `&[f32; 8]` instead of `*const f32`).

Together, these mean your crate should use `#![forbid(unsafe_code)]`. The `unsafe` lives inside archmage's generated wrappers, not in your code. If you find yourself writing `unsafe` in a crate that uses archmage, something has gone wrong.

## The prelude

`use archmage::prelude::*` gives you:

- Tokens: `Desktop64`, `Arm64`, `Arm64V2Token`, `Arm64V3Token`, `ScalarToken`, etc.
- Traits: `SimdToken`, `IntoConcreteToken`, `HasX64V2`, etc.
- Macros: `#[arcane]`, `#[rite]`, `#[magetypes]`, `incant!`
- Intrinsics: `core::arch::*` for your platform
- Memory ops: `safe_unaligned_simd` functions (reference-based, no raw pointers)

## Testing SIMD dispatch paths

Every `incant!` dispatch and `if let Some(token) = summon()` branch creates a fallback path. You can test all of them on your native hardware — no cross-compilation needed.

### Exhaustive permutation testing

`for_each_token_permutation` runs your closure once for every unique combination of token tiers, from "all SIMD enabled" down to "scalar only". It handles the disable/re-enable lifecycle, mutex serialization, cascade logic, and deduplication.

```rust
use archmage::testing::{for_each_token_permutation, CompileTimePolicy};

#[test]
fn sum_squares_matches_across_tiers() {
    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let expected: f32 = data.iter().map(|x| x * x).sum();

    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let result = sum_squares(&data);
        assert!(
            (result - expected).abs() < 1e-1,
            "mismatch at tier: {perm}"
        );
    });

    assert!(report.permutations_run >= 2, "expected multiple tiers");
}
```

On an AVX-512 machine, this runs 5–7 permutations (all enabled → AVX-512 only → AVX2+FMA → SSE4.2 → scalar). On a Haswell-era CPU without AVX-512, 3 permutations. Tokens the CPU doesn't have are skipped — they'd produce duplicate states.

Token disabling is process-wide, so run with `--test-threads=1`:

```sh
cargo test -- --test-threads=1
```

### `CompileTimePolicy` and `-Ctarget-cpu`

If you compiled with `-Ctarget-cpu=native`, the compiler bakes feature detection into the binary. `summon()` returns `Some` unconditionally, and tokens can't be disabled at runtime — the runtime check was compiled out.

The `CompileTimePolicy` enum controls what happens when `for_each_token_permutation` encounters these undisableable tokens:

- **`Warn`** — Exclude the token from permutations silently. Warnings are collected in the report.
- **`WarnStderr`** — Same, but also prints each warning to stderr with actionable fix instructions.
- **`Fail`** — Panic with the exact compiler flags needed to fix it.

For full coverage in CI, use the `testable_dispatch` feature. This makes `compiled_with()` return `None` even when features are baked in, so `summon()` uses runtime detection and tokens can be disabled:

```toml
# In your CI test configuration
[dev-dependencies]
archmage = { version = "0.7", features = ["testable_dispatch"] }
```

### Enforcing full coverage via env var

Wire an environment variable to switch between `Warn` in local development and `Fail` in CI:

```rust
use archmage::testing::{for_each_token_permutation, CompileTimePolicy};

fn permutation_policy() -> CompileTimePolicy {
    if std::env::var_os("ARCHMAGE_FULL_PERMUTATIONS").is_some() {
        CompileTimePolicy::Fail
    } else {
        CompileTimePolicy::WarnStderr
    }
}

#[test]
fn my_dispatch_works_at_all_tiers() {
    let report = for_each_token_permutation(permutation_policy(), |perm| {
        let result = my_simd_function(&data);
        assert_eq!(result, expected, "failed at: {perm}");
    });
    eprintln!("{report}");
}
```

Then in CI (with `testable_dispatch` enabled):

```sh
ARCHMAGE_FULL_PERMUTATIONS=1 cargo test -- --test-threads=1
```

If a token is still compile-time guaranteed (you forgot the feature or have stale RUSTFLAGS), `Fail` panics with the exact flags to fix it:

```
x86-64-v3: compile-time guaranteed, excluded from permutations. To include it, either:
  1. Add `testable_dispatch` to archmage features in Cargo.toml
  2. Remove `-Ctarget-cpu` from RUSTFLAGS
  3. Compile with RUSTFLAGS="-Ctarget-feature=-avx2,-fma,-bmi1,-bmi2,-f16c,-lzcnt"
```

### Manual single-token disable

For targeted tests that only need to disable one token:

```rust
use archmage::{X64V3Token, SimdToken};

#[test]
fn scalar_fallback_matches_simd() {
    let data = vec![1.0f32; 1024];
    let simd_result = sum_squares(&data);

    // Disable AVX2+FMA — summon() returns None until re-enabled
    X64V3Token::dangerously_disable_token_process_wide(true).unwrap();
    let scalar_result = sum_squares(&data);
    X64V3Token::dangerously_disable_token_process_wide(false).unwrap();

    assert!((simd_result - scalar_result).abs() < 1e-3);
}
```

Disabling cascades downward: disabling V2 also disables V3/V4/Modern/Fp16; disabling NEON also disables Aes/Sha3/Crc.

### Disabling all SIMD at once

`dangerously_disable_tokens_except_wasm(true)` disables all SIMD tokens in one call:

```rust
use archmage::dangerously_disable_tokens_except_wasm;

// Force scalar-only execution for benchmarking
dangerously_disable_tokens_except_wasm(true).unwrap();
let scalar_result = my_simd_function(&data);
dangerously_disable_tokens_except_wasm(false).unwrap();
```

This disables V2 on x86 (cascading to V3/V4/Modern/Fp16) and NEON on ARM (cascading to Aes/Sha3/Crc). V1 (`Sse2Token`) is not disabled — SSE2 is the x86_64 baseline and can't be meaningfully turned off at runtime. WASM is excluded because `simd128` is always a compile-time decision.

## Feature flags

| Feature | Default | |
|---------|---------|---|
| `std` | yes | Standard library |
| `macros` | yes | `#[arcane]`, `#[magetypes]`, `incant!` |
| `safe_unaligned_simd` | yes | Re-exports via prelude |
| `avx512` | no | AVX-512 tokens |
| `testable_dispatch` | no | Makes token disabling work with `-Ctarget-cpu=native` |

## License

MIT OR Apache-2.0

---

<sub>\* OK, `#![forbid(unsafe_code)]` isn't technically *enforced* by archmage. But with `#[arcane]`/`#[rite]` handling `#[target_feature]`, `safe_unaligned_simd` handling memory ops, and Rust 1.85+ making value intrinsics safe — there's genuinely nothing left that needs `unsafe` in your SIMD code. If your crate uses archmage and still has `unsafe` blocks, that's a code smell, not a necessity.</sub>

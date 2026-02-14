# Testing SIMD Dispatch

Every `incant!` dispatch and `if let Some(token) = summon()` branch creates a fallback path. You can test all of them on your native hardware — no cross-compilation needed.

## Exhaustive Permutation Testing

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

On an AVX-512 machine, this runs 5–7 permutations (all enabled, AVX-512 only, AVX2+FMA, SSE4.2, scalar). On a Haswell-era CPU without AVX-512, 3 permutations. Tokens the CPU doesn't have are skipped — they'd produce duplicate states.

Token disabling is process-wide, so run with `--test-threads=1`:

```sh
cargo test -- --test-threads=1
```

## `CompileTimePolicy` and `-Ctarget-cpu`

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

## Enforcing Full Coverage via Env Var

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

## Manual Single-Token Disable

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

Disabling cascades downward: disabling V2 also disables V3/V4/Modern/Fp16; disabling NEON also disables Aes/Sha3/Crc. `dangerously_disable_tokens_except_wasm(true)` disables everything at once.

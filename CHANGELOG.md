# Changelog

## 0.7.0 — 2026-02-13

New token, explicit dispatch control, and docs refresh.

- **`X64V1Token` / `Sse2Token`** — baseline x86_64 token covering SSE + SSE2. Rust 1.85+ made intrinsics safe inside `#[target_feature]` functions, but that means even `_mm_add_ps` requires a `#[target_feature(enable = "sse2")]` context. Without a token to enter that context, `#![forbid(unsafe_code)]` crates couldn't call baseline SIMD intrinsics at all. `X64V1Token::summon()` succeeds on every x86_64 CPU (SSE2 is mandatory for the architecture), so it compiles down to nothing — but it gives you the `#[target_feature]` gate you need.

- **Explicit tier lists for `incant!`** — control which dispatch tiers are attempted:

  ```rust
  // Only dispatch to V1, V3, and NEON (plus implicit scalar fallback)
  pub fn sum(data: &[f32]) -> f32 {
      incant!(sum(data), [v1, v3, neon])
  }
  ```

  Without a tier list, `incant!` tries all tiers and you need `_v2`, `_v3`, `_v4`, `_neon`, `_wasm128`, and `_scalar` variants. With a tier list, you only write the variants you care about. Scalar is always implicit.

- **Explicit tier lists for `#[magetypes]`** — same idea, applied to code generation:

  ```rust
  #[magetypes([v3, neon])]
  fn process(token: Token, data: &[f32]) -> f32 { ... }
  // Generates: process_v3, process_neon, process_scalar
  ```

- **`testable_dispatch` feature** — renamed from `disable_compile_time_tokens`. The old name described the mechanism; the new name says what it's for. Enable it in dev-dependencies so `for_each_token_permutation()` and `dangerously_disable_token_process_wide()` work even when compiled with `-Ctarget-cpu=native`.

- **Documentation refresh** — updated safety model docs, token reference, and README to cover V1 token, tier lists, and the `dangerously_disable_tokens_except_wasm` API.

## 0.6.1 — 2026-02-12

- **`archmage::testing` module** — `for_each_token_permutation()` runs a closure for every unique combination of SIMD tokens disabled, testing all dispatch fallback tiers on native hardware. Handles cascade hierarchy, mutex serialization, panic-safe re-enable, and deduplication of equivalent effective states. On an AVX-512 machine this produces 5–7 permutations; on Haswell-era, 3.

  ```rust
  use archmage::testing::{for_each_token_permutation, CompileTimePolicy};

  #[test]
  fn dispatch_works_at_all_tiers() {
      let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
          let result = sum_squares(&data);
          assert!((result - expected).abs() < 1e-1, "failed at: {perm}");
      });
      assert!(report.permutations_run >= 2);
  }
  ```

- **`CompileTimePolicy` enum** — `Warn` (silent, collect in report), `WarnStderr` (also prints), `Fail` (panics with exact compiler flags to fix). Wire an env var for CI enforcement.

## 0.6.0 — 2026-02-12

Cross-platform hardening, testability, and CI infrastructure.

- **Test every dispatch path without cross-compilation.** Disable individual tokens or kill all SIMD at once to force your code through scalar and lower-tier fallbacks — on your native hardware, in your existing test suite:

  ```rust
  use archmage::{X64V3Token, SimdToken, dangerously_disable_tokens_except_wasm};

  #[test]
  fn scalar_fallback_produces_same_results() {
      let result_simd = my_function(&data);

      // Kill V3 (AVX2+FMA) — summon() now returns None
      X64V3Token::dangerously_disable_token_process_wide(true).unwrap();
      let result_scalar = my_function(&data);
      X64V3Token::dangerously_disable_token_process_wide(false).unwrap();

      assert_eq!(result_simd, result_scalar);
  }

  #[test]
  fn everything_works_without_simd() {
      dangerously_disable_tokens_except_wasm(true).unwrap();
      // entire test runs through scalar fallbacks
      run_full_integration_test();
      dangerously_disable_tokens_except_wasm(false).unwrap();
  }
  ```

  Disabling V2 cascades to V3/V4/Modern/Fp16. Disabling NEON cascades to Aes/Sha3/Crc. Per-token `manually_disabled()` lets you query the current state.

- **Cross-architecture SIMD API consistency** — final alignment pass across all platforms
- **Coverage tests** — targeted tests for stubs, `forge()`, `disable()`, `detect` helpers, and `IntoConcreteToken` on ARM/WASM stubs
- **Feature combination CI** — 8 feature combos tested (no-default, individual, pairs, all-features) plus aarch64 coverage with codecov flag-based merging
- **Cross-platform CI** — ARM, WASM, and i686 compilation verified; arch guards on platform-specific tests
- **Performance documentation** — DCT-8 and cross-token nesting benchmarks consolidated into `docs/PERFORMANCE.md`
- **SIMD reference mdbook** — searchable docs with ASM-verified load/store patterns
- **Feature flag strings and `AUDITING.md`** — `DISABLE_TARGET_FEATURES` string per token tells auditors exactly which RUSTFLAGS to set
- **Codegen quality** — replaced 330 `writeln!` chains with `formatdoc!` across token_gen.rs and main.rs
- **Miri CI stability** — isolated target dirs, pinned nightly, gated platform-specific tests

## 0.5.0 — 2025-12-20

Macro system overhaul and performance infrastructure.

- **`#[rite]` macro** — inner SIMD helpers with `#[target_feature]` + `#[inline]`. Use this by default; `#[arcane]` only at entry points. Benchmarked: calling `#[arcane]` per iteration costs 4-6x vs `#[rite]` inlining.
- **`incant!` macro** — runtime dispatch across `_v3`, `_v4`, `_neon`, `_wasm128`, `_scalar` suffixed functions
- **`#[magetypes]` macro** — replaces `#[multiwidth]`. Generates platform-specific variants from a generic `Token` parameter.
- **`ScalarToken`** — always-available fallback token for `incant!` dispatch
- **`IntoConcreteToken`** — safe upcasting with `as_x64v3()`, `as_x64v4()`, etc.
- **Atomic `summon()` caching** — 2-6x faster detection after first call (~1.3 ns cached, 0 ns when compiled away)
- **`compiled_with()` rename** — `guaranteed()` → `compiled_with()` for clarity
- **Token disable mechanism** — per-token `.disable(true)` for testing
- **NEON runtime detection fix** — no longer assumes NEON on AArch64 (broken on some Android/Linux kernels)
- **`SimdToken` sealed** — removed unsound `From` impls, added `forge()` for `unsafe` token construction
- **Per-token namespace modules** — `archmage::x64v3::Token`, `archmage::neon::Token`, etc.
- **Prelude module** — `use archmage::prelude::*` for tokens, traits, macros, intrinsics, and memory ops
- **`implementation_name()`** — all magetypes vectors report their backing implementation
- **Cross-architecture stubs** — all tokens compile on all platforms; `summon()` returns `None` on wrong arch
- **Removed `#[multiwidth]`** — replaced by `#[magetypes]`
- **Removed `bytemuck` dependency** — token-gated cast methods instead
- **`safe_unaligned_simd` integration** — re-exported via prelude, reference-based loads/stores

## 0.4.0 — 2025-11-15

Full cross-platform parity.

- **`token-registry.toml`** — single source of truth for all token definitions, feature sets, and trait mappings. Code generation reads this; validation checks against it.
- **API parity: 270 → 0 issues** — every W128 SIMD type has identical methods across x86, ARM, and WASM
- **ARM transcendentals** — full lowp + midp coverage: log2, exp2, ln, exp, log10, pow, cbrt for f32x4; lowp for f64x2
- **WASM transcendentals** — cbrt_midp, f64x2 log10_lowp, complete `_unchecked` and `_precise` suffix variants
- **ARM/WASM block ops** — interleave, deinterleave, transpose for all W128 types
- **x86 byte shift polyfills** — i8x16/u8x16 shl, shr, shr_arithmetic
- **WASM u64x2 ordering comparisons** — simd_lt/le/gt/ge via bias-to-signed polyfill
- **AVX-512 fast-path methods** — `_fast` variants for i64/u64 min/max/abs accepting `X64V4Token`
- **`cargo xtask parity`** — detects API variance between architectures
- **`cargo xtask validate`** — static soundness verification for all magetypes intrinsics
- **Miri boundary tests** — exhaustive load/store verification under Miri
- **Proptest fuzzing** — divergence detection across implementations
- **Codegen refactor** — generated files moved to `generated/` subfolders; all codegen uses `formatdoc!`

## 0.3.0 — 2025-10-28

Architecture cleanup.

- **Microarchitecture-level tokens** — replaced granular per-feature tokens with `X64V2Token`, `X64V3Token`, `X64V4Token` matching LLVM's x86-64 levels
- **Full WASM SIMD128 support** — `Wasm128Token` with optimized codegen
- **Intrinsic safety validation** — complete intrinsic database with automated safety checks
- **Intrinsic reference docs** — auto-generated docs organized by token tier
- **Removed ~2200 lines of dead wrapper code** and 6 unused dependencies

## 0.2.0 — 2025-10-15

Types and cross-platform.

- **`magetypes` crate** — token-gated SIMD types with natural operators (`f32x4`, `f32x8`, `i32x4`, etc.)
- **WASM SIMD support** — `Wasm128Token` and W128 types
- **AArch64 NEON support** — `NeonToken` with polyfilled wide types
- **`#[multiwidth]` macro** — generate multi-width SIMD variants (later replaced by `#[magetypes]`)
- **Token-gated bytemuck replacements** — `cast_slice`, `as_bytes`, `from_bytes` without `unsafe`
- **AVX-512 types** — 512-bit SIMD types behind `avx512` feature
- **`WidthDispatch` trait** — associated-type-based SIMD width dispatch

## 0.1.0 — 2025-10-01

Initial release.

- **Token-based SIMD capability proof** — `X64V3Token::summon()` returns `Some` only if CPU supports AVX2+FMA
- **`#[arcane]` macro** — generates `#[target_feature]` functions with cross-arch stubs
- **Zero overhead** — identical assembly to hand-written `unsafe` code
- **`#[forbid(unsafe_code)]` compatible** — all unsafety is inside the macro expansion
- **`no_std` + `alloc` by default** — `std` opt-in via feature flag

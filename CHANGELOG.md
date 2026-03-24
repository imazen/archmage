# Changelog

## 0.9.7 — 2026-03-24

### Backwards compatibility fix

- **`incant!`/`#[magetypes]` with explicit `[v4, ...]` tier lists** — AVX-512 tiers are now silently skipped when the `avx512` feature is off, even in explicit tier lists. This matches the old behavior where v4 dispatch arms were cfg-gated, but via the correct mechanism (expansion-time check on archmage-macros, not `#[cfg(feature)]` in output). Crates like linear-srgb that use `incant!(foo(x), [v4, v3, neon])` with cfg-gated `_v4` functions now work without changes.

## 0.9.6 — 2026-03-24

### Bug fixes

- **`i16x16`/`u16x16` bitmask correctness** — `bitmask()` was returning incorrect results on x86_64 AVX2: lanes 8-15 were always zero. Root cause: `_mm256_packs_epi16(shifted, shifted)` interleaves within 128-bit lanes, producing wrong lane ordering. Fix: extract 128-bit halves first, then use `_mm_packs_epi16(lo, hi)` for correct order. Fixed in both the raw W256 types and the generic backend implementations. ([#16])

- **`#[arcane]` lint attribute propagation** — `#[allow(clippy::too_many_arguments)]` and similar lint-control attributes (`#[expect]`, `#[deny]`, `#[warn]`, `#[forbid]`) now propagate to the generated dispatch wrapper in both sibling mode (default) and nested mode. Previously, clippy would lint the generated code even when the user explicitly suppressed the warning. ([#17])

- **`#[autoversion]` v4/v4x variants no longer require `avx512` feature** — `#[autoversion]` generates scalar code compiled with `#[target_feature]`, so the `avx512` cargo feature was never needed. Previously, v4/v4x variants were silently eliminated by `#[cfg(feature = "avx512")]` in macro output — which checked the *calling crate's* features (always wrong for downstream crates). Now v4/v4x variants are always generated.

### avx512 feature gating overhaul

The `avx512` cargo feature handling was redesigned. The old approach emitted `#[cfg(feature = "avx512")]` in proc-macro output, which checked the calling crate's features instead of archmage's — always wrong for downstream crates, and triggering `unexpected_cfgs` warnings on modern rustc.

**New behavior:**

- **`avx512` feature propagated to `archmage-macros`** — macros check `cfg!(feature = "avx512")` at expansion time on their own crate, not via `#[cfg]` in output.
- **`#[autoversion]`** — always generates v4/v4x (scalar code, no safe memory ops needed).
- **`incant!`/`#[magetypes]` default tiers** — include v4 only when `avx512` is enabled. Without it, dispatch gracefully skips v4.
- **`#[arcane(import_intrinsics)]`/`#[rite(import_intrinsics)]` with AVX-512 tokens** — clear `compile_error!` when `avx512` is not enabled, telling users exactly what to add to `Cargo.toml`. Works for all token spellings: concrete (`X64V4Token`, `Avx512Token`, `Server64`), trait bounds (`impl HasX64V4`), and generics (`T: HasX64V4`).
- **`#[arcane]`/`#[rite]` without `import_intrinsics`** — always works with any token. Value intrinsics don't need the cargo feature.
- **No `#[cfg(feature = "...")]` ever emitted in macro output** — eliminates `unexpected_cfgs` warnings entirely.

### Testing

- 192 bitmask correctness tests covering all 24 integer SIMD types (W128/W256/W512).
- No-features integration test crate (`archmage-no-features-test`) with `#![deny(warnings)]`.
- 5 standalone avx512-cfg-test crates verifying every feature gating scenario.
- Token infrastructure tests now serialized in CI to prevent races from process-wide state mutations.

[#16]: https://github.com/imazen/archmage/issues/16
[#17]: https://github.com/imazen/archmage/issues/17

## 0.9.5 — 2026-03-09

### Transcendental accuracy improvements

- **`exp2_midp`: floor → round-to-nearest split** — Splitting the input into integer and fractional parts now uses round-to-nearest instead of floor, keeping |frac| ≤ 0.5 instead of [0, 1). This eliminates the accuracy hot spot near integer boundaries where the polynomial was evaluating at frac ≈ 1.0. The integer part is clamped to 127 to prevent the `(n+127)<<23` bit trick from overflowing. Accuracy is now uniform across all input regions (1 ULP for evenly-spaced inputs, 63 ULP worst case overall). Applied on all platforms (x86, ARM, WASM, generic).

- **`exp2_midp` overflow threshold**: changed from `> 128` to `>= 128`. Since 2^128 > f32::MAX, `exp2_midp(128.0)` now correctly returns inf instead of ~3.4e38.

- **`exp2_midp` underflow limit**: tightened from -150 to -126. Inputs in [-150, -126] were producing garbage via the bit trick (which can't construct denormal floats). Now returns 0 for all inputs below -126.

- **`pow_lowp(0, n)` for positive n**: now returns 0 instead of NaN. The zero-mask was being applied before the exp2 computation instead of after.

- **`cbrt` zero handling**: `cbrt(±0)` now returns ±0 (preserving sign) instead of producing small nonzero values from the bit-hack initial guess.

- **Accuracy improvements** (max ULP vs std, measured on x86-64):
  - `exp2_midp`: 132 → 63 ULP
  - `exp_midp`: 256+ → 58 ULP
  - `pow_midp` (n=0.5): 149 → 9 ULP
  - `pow_midp` (n=2): ~130 → 16 ULP
  - `pow_midp` (n=3): ~150 → 55 ULP

- **No performance impact**: round instruction costs the same as floor (same opcode, different rounding mode bit). The added `min(xi, 127)` is one extra SIMD min instruction.

## 0.9.4 — 2026-03-08

Multi-tier `#[rite]`, `#[inline(always)]` wrappers, improved cbrt, docs overhaul.

### `#[rite]` multi-tier support

`#[rite]` now supports three modes:

- **Token-based** (`#[rite]`): reads the token type from the function signature (existing behavior)
- **Tier-based** (`#[rite(v3)]`): specifies features via tier name, no token parameter needed (existing behavior)
- **Multi-tier** (`#[rite(v3, v4, neon)]`): generates a suffixed variant for each tier (`fn_v3`, `fn_v4`, `fn_neon`), each with its own `#[target_feature]` and `#[cfg(target_arch)]`

Multi-tier lets you write one function body and get per-tier compiled variants — like `#[autoversion]` but for internal SIMD functions (no dispatcher generated). Each variant is safe to call from matching `#[arcane]` or `#[rite]` contexts (Rust 1.85+). Single-tier behavior is unchanged — no suffix is added.

```rust
#[rite(v3, v4, neon)]
fn scale(data: &[f32; 4], factor: f32) -> [f32; 4] {
    [data[0] * factor, data[1] * factor, data[2] * factor, data[3] * factor]
}
// Generates: scale_v3(), scale_v4(), scale_neon()
```

### `#[inline(always)]` on `#[arcane]` wrappers

`#[arcane]` now generates `#[inline(always)]` on the safe wrapper function. Previously the wrapper had no inline hint, which could prevent LLVM from inlining the dispatch trampoline. If you had `#[inline(always)]` on the function yourself, the macro strips it to avoid the duplicate-attribute warning (which Rust is phasing into a hard error).

### `incant!` explicit tiers: `scalar` recommended

When using explicit tier lists with `incant!`, always include `scalar`: `incant!(sum(data), [v3, neon, scalar])`. This documents the mandatory fallback path. Currently `scalar` is auto-appended if omitted for backwards compatibility; this will become a compile error in v1.0.

### Improved `cbrt` (cube root)

- `cbrt_midp` now uses 2-iteration Halley refinement (was Newton-Raphson). ~2 ULP max error across the full f32 range, down from ~4 ULP.
- `cbrt_lowp` uses 1-iteration Halley. ~22 ULP max error, faster than midp.
- Added `cbrt_lowp`/`cbrt_midp` with `_unchecked` and `_precise` variants.
- All variants handle negative inputs, zero, NaN, and infinity correctly.
- Added scalar `ScalarToken` implementations for all cbrt variants.

### `macros` feature is now always-on

The `macros` cargo feature is now a no-op — macros (`#[arcane]`, `#[rite]`, `incant!`, etc.) are always available. The feature flag still exists so `features = ["macros"]` doesn't break existing code.

### Documentation overhaul

- **README**: safety model diagram showing Rust's `#[target_feature]` call rules and how archmage makes dispatch sound. Macro selection flowchart (`#[arcane]` vs `#[rite]` vs `#[autoversion]` vs `incant!`). Tier naming conventions table. Both `#[rite]` syntaxes with code examples. Expanded testing section with `testable_dispatch`, `CompileTimePolicy`, and `lock_token_testing()`.
- **All docs**: updated `incant!` examples to include `scalar` in explicit tier lists.

### Other fixes

- Use `f64::clamp()` instead of manual min/max pattern.
- User `#[inline(always)]` on `#[arcane]`/`#[rite]` functions no longer causes duplicate attribute warnings.
- `#[rite]` strips user `#[inline]` attributes to avoid conflicts with its own `#[inline]`.

## 0.9.3 — 2026-03-05

Fixed `no_std` compilation on bare-metal targets, added `no_std` CI enforcement.

- **Fixed `no_std` on aarch64/WASM bare metal** — ARM and WASM `f64x2` transcendentals (`log2_lowp`, `exp2_lowp`, `ln_lowp`, `exp_lowp`, `log10_lowp`, `pow_lowp`) used `f64` inherent methods (`.log2()`, `.exp2()`, etc.) that only exist with `std`. Added scalar polynomial approximations to `nostd_math` using the same coefficients as the x86 SIMD implementations.

- **Mandatory `no_std` CI checks** — CI now auto-installs and compiles against `aarch64-unknown-none` and `thumbv7m-none-eabi` targets. Host-target `--no-default-features` checks don't catch `std` leaks because libstd is always linkable on the host; cross-target checks are required to catch them.

- **`just test-nostd`** — new justfile target runs `no_std` compilation checks and tests for all crates.

- **Bitmask tests handle missing runtime detection** — tests now skip gracefully when `summon()` returns `None` (happens under `no_std` without `-Ctarget-cpu`) instead of panicking.

## 0.9.2 — 2026-03-05

Const generic support for `#[autoversion]` and `#[arcane]`, semver-checks CI.

- **Const generic support** — `#[autoversion]`, `#[arcane]` (sibling mode), and `#[arcane]` (nested mode) now forward const generic parameters via turbofish in generated dispatch/wrapper calls. Previously, const generics that couldn't be inferred from argument types alone (e.g., `<const BPP: usize>` used only in the function body) caused `E0282: type annotations needed`. This matches `multiversion`'s `#[multiversed]` behavior.

  ```rust
  // Now works — CHUNK forwarded via turbofish in dispatcher
  #[autoversion]
  fn fill_row<const BPP: usize>(_token: SimdToken, data: &[u8]) { ... }

  fill_row::<3>(&data); // Dispatcher calls fill_row_v3::<3>(...)
  ```

- **Semver-checks CI** — new `semver-checks.yml` workflow runs `cargo-semver-checks` on every PR for all three crates, catching accidental breaking changes before merge.

- **22 new const generic tests** — covers `#[autoversion]` and `#[arcane]` with: basic const generics, body-only const generics, return-type-only, multiple const generics, mixed type+const generics, lifetimes, self receivers, `_self = Type` nested mode, explicit tiers, and direct variant calls with turbofish.

## 0.9.1 — 2026-03-05

Generic `f32x16<T>` transcendentals, bitmask bug fix, `#[autoversion]` improvements.

- **Fixed i16x16/u16x16 bitmask** — `_mm256_packs_epi16` lane interleaving caused lanes 8-15 to be dropped on x86_64. The fix uses per-lane `_mm256_extract_epi16` + manual bit assembly, matching the pattern used for other element sizes.

- **`#[autoversion]` self receiver fix** — inherent methods with `self`/`&self`/`&mut self` now work without `_self = Type`. The `_self` parameter is only needed for trait impl delegation (nested mode). Previously, `#[autoversion]` incorrectly required `_self` for all self receivers.

- **Generated functions are private** — `#[autoversion]` variants and `#[arcane]` sibling functions no longer inherit the user's visibility. Only the dispatcher (for `#[autoversion]`) or the safe wrapper (for `#[arcane]`) gets the original visibility. This prevents leaking internal implementation functions.

- **`#[autoversion]` reference page** — comprehensive documentation at `docs/site/content/archmage/dispatch/autoversion.md` covering all parameters, tier tables, dispatch flow, and usage patterns.

- **`F32x16Convert` trait** — new backend trait enabling bitcast and numeric conversion between `f32x16` and `i32x16`. Implemented for all backends: X64V3Token (2×256-bit polyfill), X64V4Token/X64V4xToken (native AVX-512), NeonToken (4×128-bit polyfill), Wasm128Token (4×128-bit polyfill), ScalarToken.

- **Generic `f32x16<T>` transcendentals** — `pow_midp`, `log2`, `exp2`, `ln`, `exp`, `log10`, `cbrt` (all with `_lowp`/`_midp`, `_unchecked`, `_precise` variants). Same polynomial approximations as `f32x4<T>` and `f32x8<T>`, works on any backend that implements `F32x16Convert`.

- **`f32x16<T>` ↔ `i32x16<T>` conversion methods** — `bitcast_to_i32`, `from_i32_bitcast`, `to_i32`, `to_i32_round`, `from_i32` on `f32x16<T>`; `bitcast_to_f32`, `to_f32` on `i32x16<T>`.

- **Comprehensive bitmask tests** — correctness tests for all 24 generic SIMD integer types (W128/W256/W512, signed/unsigned, 8/16/32/64-bit) covering individual lanes, cross-boundary patterns, all-set, and all-clear.

- **30 `#[autoversion]` integration tests** — plain self receivers, owned self, explicit tiers, wildcards, tuple/Option returns, in-place mutation, scalar/v3 variant direct calls.

- **34 f32x16 tests** covering transcendentals, conversions, edge cases, roundtrips, cross-backend consistency, and generic function usage.

## 0.9.0 — 2026-03-04

Sibling expansion, cfg-out default, macro options, `import_intrinsics`.

**BREAKING:** `#[arcane]` and `#[rite]` now cfg-out functions on non-matching architectures by default (no unreachable stub). Code referencing these functions on wrong platforms without `#[cfg]` guards will fail to compile. **Migration:** Add `#[arcane(stub)]` / `#[rite(stub)]` to restore old behavior, or use `#[cfg(target_arch)]` guards, or use `incant!` (unaffected — already cfg-gates dispatch calls).

**BREAKING:** The `safe_unaligned_simd` cargo feature is gone — the dependency is now always included. `import_intrinsics` (new in 0.9) generates combined intrinsics modules that re-export `safe_unaligned_simd`'s reference-based memory ops alongside `core::arch`, so the dependency is no longer optional. A no-op `safe_unaligned_simd` feature flag exists for backwards compatibility — `features = ["safe_unaligned_simd"]` won't break, it just does nothing. If you were using `default-features = false` to exclude it, note that the `safe_unaligned_simd` crate is now always pulled in (it's small and `no_std`).

- **Sibling expansion (default)** — `#[arcane]` now generates a sibling `__arcane_fn` with `#[target_feature]` at the same scope, plus a safe wrapper. Both functions share the `impl` block, so `self`, `Self`, and associated constants work naturally in methods. No more `_self` boilerplate for inherent methods.

- **Nested expansion (opt-in)** — `#[arcane(nested)]` uses the old inner-function approach. `#[arcane(_self = Type)]` implies nested. **Required for trait impls** — sibling expansion adds `__arcane_fn` which isn't in the trait definition.

- **Cfg-out default** — On wrong architectures, `#[arcane]` and `#[rite]` emit no code at all. Less dead code. `incant!` is unaffected (already cfg-gates each tier).

- **`stub` param** — `#[arcane(stub)]` and `#[rite(stub)]` generate `unreachable!()` stubs on wrong architectures, restoring previous cross-platform behavior for dispatch patterns that reference functions without `#[cfg]` guards.

- **LightFn (internal)** — Proc macro now parses only the function signature into AST, leaving the body as opaque `TokenStream`. Saves ~2ms per function. Token-level `Self`/`self` replacement instead of syn `Fold`.

- **`import_intrinsics` parameter** — `#[arcane(import_intrinsics)]` and `#[rite(import_intrinsics)]` inject `use archmage::intrinsics::{arch}::*` into the function body. This brings all `core::arch` types and value intrinsics into scope alongside [`safe_unaligned_simd`](https://crates.io/crates/safe_unaligned_simd)'s reference-based memory ops. Rust's name resolution makes explicit re-exports shadow glob imports, so `_mm256_loadu_ps` resolves to the safe version (takes `&[f32; 8]`) automatically. Combined with Rust 1.85+ making value intrinsics safe inside `#[target_feature]`, this means zero `unsafe` in your SIMD code — `#![forbid(unsafe_code)]` compatible.

- **Combined intrinsics modules** — `archmage::intrinsics::{x86_64, x86, aarch64, wasm32}` glob-import `core::arch` and explicitly re-export every `safe_unaligned_simd` function. Auto-generated by `cargo xtask generate`. These modules are what `import_intrinsics` injects.

- **`#[autoversion]`** — new single-attribute macro that generates architecture-specific function variants *and* a runtime dispatcher from one annotated function. Write a plain scalar loop with a `SimdToken` placeholder parameter; `#[autoversion]` clones it per tier (v4, v3, neon, wasm128, scalar by default), replaces `SimdToken` with each concrete token type in the signature, wraps non-scalar variants in `#[arcane]` for `#[target_feature]`, and emits a dispatcher function (same name, `SimdToken` param removed) that calls the best variant at runtime via `Token::summon()`.

  - **Signature-only replacement** — only the `SimdToken` type annotation in the parameter list is swapped. The function body is never reparsed (uses `LightFn`'s opaque body), keeping compile times low. Compare with `#[magetypes]` which does full text substitution including the body.

  - **Explicit tiers** — `#[autoversion(v3, v4, v4x, neon, arm_v2, wasm128)]`. `scalar` is always appended implicitly. Unknown tier names produce a compile error. Tiers are sorted by dispatch priority automatically.

  - **Self receivers** — inherent methods with `self`/`&self`/`&mut self` work naturally (fixed in 0.9.1 — originally required `_self = Type`). For trait impl delegation, use `#[autoversion(_self = Type)]` with `_self` in the body.

  - **Trait method delegation** — trait impl methods can't expand to multiple siblings, so `#[autoversion]` can't be used directly on them. Documented delegation pattern: trait method calls an autoversioned inherent method.

  - **Generated variants are private** — individually callable within the module, `incant!`-compatible, with `#[cfg(target_arch)]` and `#[cfg(feature)]` guards matching each tier. Only the dispatcher inherits the user's visibility.

  - **74 unit tests** — argument parsing, `SimdToken` parameter discovery, tier resolution, AST replacement for all known tiers, dispatcher parameter removal and wildcard renaming, tier descriptor properties, suffix_path.

- **MSRV 1.89** — required for stabilized target features and intrinsics. On x86, Rust 1.89 stabilizes `avx512fp16`, `sm3`, `sm4`, `kl`, and `widekl` target features, plus additional AVX-512 intrinsics and target features. These are needed for archmage's token-to-feature mappings and `#[target_feature]` attributes emitted by `#[arcane]`.

## 0.8.3 — 2026-02-19

Complete `X64CryptoToken` integration with `incant!` dispatch.

- **`x64_crypto` tier in `incant!`** — dispatch to `X64CryptoToken` via explicit tier lists: `incant!(func(data), [v4x, x64_crypto, arm_v2, neon_aes])`. Priority 25 (between v3=30 and v2=20), so crypto is tried before plain v2.

- **`IntoConcreteToken::as_x64_crypto()`** — safe downcasting to `X64CryptoToken` for passthrough dispatch.

- **Prelude** — `X64CryptoToken` now re-exported from `archmage::prelude::*`.

## 0.8.2 — 2026-02-19

New `X64CryptoToken` for PCLMULQDQ + AES-NI.

- **`X64CryptoToken`** — new leaf token off `X64V2Token` providing `pclmulqdq` and `aes` features. PCLMULQDQ and AES-NI are not part of the psABI v2 spec (Nehalem 2008 and some VMs lack them), so they belong in a dedicated token rather than in `X64V2Token`. Available on Westmere (2010)+, Bulldozer+, Silvermont+, all Zen. Use for CRC-32 folding, AES encryption, and GF(2) polynomial arithmetic. Dispatch tier name: `x64_crypto`.

- **Reverts 0.8.1** — removed `pclmulqdq` and `aes` from `X64V2Token` and all higher tokens (V3, V4, V4x, FP16). V2 now matches the psABI spec exactly.

## 0.8.1 — 2026-02-18 [YANKED]

Incorrectly added PCLMULQDQ/AES-NI to V2 baseline. These are not in the psABI v2 spec — Nehalem (2008) and QEMU's x86-64-v2 CPU model lack them. Use 0.8.2's `X64CryptoToken` instead.

## 0.8.0 — 2026-02-18

ARM compute tiers, better macro diagnostics, edition 2024.

- **`Arm64V2Token` and `Arm64V3Token`** — new compute-tier tokens for AArch64, analogous to x86's V2/V3/V4 hierarchy. V2 covers Cortex-A55+, Apple M1+, Graviton 2+ (adds CRC, RDM, DotProd, FP16, AES, SHA2 over baseline NEON). V3 covers Cortex-A510+, Apple M2+, Snapdragon X, Graviton 3+ (adds FHM, FCMA, SHA3, I8MM, BF16). Both tokens work with `#[arcane]`, `#[rite]`, `incant!`, and `#[magetypes]`.

- **`HasArm64V2` and `HasArm64V3` traits** — tier traits for generic bounds over the new ARM compute tiers, matching the pattern of `HasX64V2`/`HasX64V4`.

- **`incant!` tiers `arm_v2` and `arm_v3`** — dispatch to ARM compute tiers in explicit tier lists: `incant!(process(data), [v3, arm_v2, arm_v3, neon])`.

- **`IntoConcreteToken` gains `as_arm_v2()` and `as_arm_v3()`** — safe downcasting to the new ARM tokens.

- **Featureless trait rejection** — `#[arcane]` and `#[rite]` now give a specific error when you use `SimdToken` or `IntoConcreteToken` as a token bound. These traits carry no CPU features, so the macros can't emit `#[target_feature]`. The error message explains why and suggests concrete tokens or feature traits like `HasX64V2`.

- **`detect_features` example** — new example (`cargo run --example detect_features`) that prints all detected SIMD capabilities on the current CPU.

- **Edition 2024** — `archmage-macros` upgraded from Rust edition 2021 to 2024 (requires rustc 1.89+). `archmage` and `magetypes` were already on edition 2024.

## 0.7.1 — 2026-02-14

Docs, warnings, and magetypes 0.7.0.

- **Documentation: token type is the feature selector** — all docs (lib.rs, README, PERFORMANCE.md, spec.md, magetypes README) now explain that `#[arcane]` and `#[rite]` parse the token type from your function signature to determine which `#[target_feature]` to emit. Passing the same token through your call hierarchy keeps features consistent; mismatched types create optimization boundaries.

- **Documentation: `#[arcane]` wrapper vs `#[rite]` direct** — clarified that `#[arcane]` generates a wrapper function to cross the `#[target_feature]` boundary without `unsafe` at the call site, but that wrapper *is* the optimization boundary. `#[rite]` applies `#[target_feature]` + `#[inline]` directly with no wrapper. `#[rite]` should be the default; `#[arcane]` only at entry points.

- **Zero compiler warnings** — fixed all warnings across xtask (14), tests, and examples. Removed unused imports, unnecessary `unsafe` blocks (safe since Rust 1.85), and minor clippy lints.

- **Fixed rustdoc warning** — escaped `#[arcane]` doc link in X64V1Token.

- **`magetypes` 0.7.0** — version aligned with archmage 0.7.0 dependency.

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

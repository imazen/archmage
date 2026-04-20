# Token-by-Self Refactor — Design Notes

Record of the `fix/token-by-self-soundness` branch (PR #40). Every backend trait method now takes `self`; the generic wrapper stores the token inline so the typestate matches the runtime contract.

---

## 1. What changed

### Part A — backend trait receivers

Every method on every `F32xNBackend` / `I*xNBackend` / `U*xNBackend` trait takes `self` as its first parameter. Before the branch, methods were associated functions:

```rust
// before
pub trait F32x8Backend: SimdToken + Sealed + Copy + 'static {
    type Repr: Copy + Clone + Send + Sync;
    fn splat(v: f32) -> Self::Repr;
    fn add(a: Self::Repr, b: Self::Repr) -> Self::Repr;
    // ...
}

// after
pub trait F32x8Backend: SimdToken + Sealed + Copy + 'static {
    type Repr: Copy + Clone + Send + Sync;
    fn splat(self, v: f32) -> Self::Repr;
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
    // ...
}
```

With `self` required, a caller cannot invoke any backend method without first producing a token value. The only paths to a token remain `T::summon()`, extractor methods like `X64V4Token::v3()`, and the explicitly `unsafe` `T::forge_token_dangerously()`.

### Part B — generic wrapper stores the token

```rust
// before
pub struct f32x8<T: F32x8Backend>(T::Repr, PhantomData<T>);

// after
#[repr(C)]
pub struct f32x8<T: F32x8Backend>(T::Repr, pub(crate) T);
```

`T` is ZST for every token type, so `size_of::<f32x8<T>>() == size_of::<T::Repr>()` and the layout is bit-identical to the underlying vector type. Operator impls, block ops, and inherent methods now access `self.1` to re-supply the token when calling backend traits, instead of asking the user to pass one.

`#[repr(transparent)]` won't compile on a two-field struct even if one field is ZST (E0690), so the branch uses `#[repr(C)]` paired with `const _: ()` size/alignment assertions in every generated `*_impl.rs` file. If a token ever gains a non-ZST field, those assertions fail the build.

### Knock-on fixes

- NEON `f32x8` polyfill `recip`/`rsqrt` now use a two-step Newton-Raphson so the polyfill matches single-width NEON precision (QEMU runner had enough noise to expose the gap).
- `_approx` tolerance widened from 1e-3 → 4e-3 to match the ARM ARM bound for `frecpe`/`frsqrte`. The prior value was tighter than the spec permits — it was passing on hardware by luck and failing under QEMU where the emulated estimate sits closer to the edge of the allowed error band.
- CI now runs magetypes integration tests on `aarch64-unknown-linux-gnu` (via `cross`) and `wasm32-wasip1` (via `wasmtime`), closing #39.
- Toolchain-drift cleanups: trybuild stderr snapshots, three clippy lints, `cargo fmt`.

---

## 2. Why

Before this branch, a safe-code caller could invoke `<X64V3Token as F32x8Backend>::splat(7.0)` via UFCS without holding a token. The trait is sealed (only archmage tokens can implement it), but sealing controls implementers, not callers — anyone could still reach the associated function through a qualified-type path. The same gap applied to five inherent conversion methods on the generic wrappers (`f32x4::from_u8`, `f32x4::load_4_rgba_u8`, `f32x8::from_u8`, `f32x8::load_8_rgba_u8`, `f32x8::load_8x8`), which were plain associated functions taking no token.

The runtime consequence of bypassing the check: executing an intrinsic like `vbroadcastss %ymm` on a CPU that doesn't support AVX triggers `SIGILL` and the process dies. That is a crash, not exploitable corruption — but the crate's value proposition is "no unsupported-instruction crashes in safe code," and the type system should enforce what the runtime check enforces. The refactor closes the gap end-to-end.

---

## 3. Construction-surface audit

Every path that constructs, wraps, or operates on a `Repr` now requires a token value. Summary:

| Path | Before | After |
|---|---|---|
| `T::splat(v)` (UFCS on the trait) | `fn splat(v) -> Repr` | `fn splat(self, v) -> Repr` |
| `T::zero()` / `T::load(data)` / `T::from_array(arr)` | no `self` | takes `self` |
| `T::add` / `sub` / `mul` / `div` / `neg` | no `self` | takes `self` |
| `T::min` / `max` / `abs` / `sqrt` / `floor` / `ceil` / `round` / `trunc` | no `self` | takes `self` |
| `T::mul_add` / `mul_sub` | no `self` | takes `self` |
| `T::simd_eq` / `ne` / `lt` / `le` / `gt` / `ge` / `blend` | no `self` | takes `self` |
| `T::reduce_add` / `reduce_min` / `reduce_max` | no `self` | takes `self` |
| `T::store` / `to_array` | no `self` | takes `self` |
| `T::not` / `bitand` / `bitor` / `bitxor` | no `self` | takes `self` |
| `T::shl_const` / `shr_logical_const` / `shr_arithmetic_const` | no `self` | takes `self` |
| `T::all_true` / `any_true` / `bitmask` / `popcount` | no `self` | takes `self` |
| `T::rcp_approx` / `rsqrt_approx` / `recip` / `rsqrt` | no `self` | takes `self` |
| `T::bitcast_*` / `convert_*` (convert traits) | no `self` | takes `self` |
| `T::clamp` (default body) | used `Self::min` / `Self::max` without `self` | default body uses `<Self as Trait>::min(self, ...)` |
| `f32xN::splat(token, v)` (inherent generic API) | took token | unchanged shape (token is `self.1` internally) |
| `f32xN::from_repr(token, repr)` | took token, used `PhantomData` | stores `token` in the struct |
| `f32xN::from_repr_unchecked(repr)` | `pub(super)`, no token | `pub(crate)`, takes `token` |
| `f32xN::from_u8` / `load_4_rgba_u8` / `load_8_rgba_u8` / `load_8x8` | static, no token | takes `token: T` |
| `deinterleave_4ch` / `interleave_4ch` / `transpose_4x4` / `transpose_8x8` | static over `[Self; N]`, no token | pulls token from `arr[0].1` |
| `from_m128` / `from_v128` / `from_float32x4_t` (platform conversions) | took `_: T` — token discarded | takes `token: T`, stores it |

~30 backend traits × ~35 methods = ~1100 trait signatures updated. ~600 generic-exposure call sites regenerated through xtask codegen. Plus the five inherent methods above, which are the only ones `cargo semver-checks` flags (the rest are shielded by `Sealed`).

### Fabrication escapes (unchanged and still documented)

A caller can still construct a `Repr` value by routes outside archmage — none of them produce a token, so the fabricated `Repr` can't feed any backend operation:

- `bytemuck::cast::<[f32; 8], __m256>(arr)` — only with `bytemuck/simd` feature; magetypes' `Cargo.toml` documents this.
- `core::mem::transmute::<[f32; 8], __m256>(arr)` — `unsafe`, caller opted in.
- `core::mem::zeroed::<__m256>()` — `unsafe`, caller opted in.

A fabricated `Repr` that never passed through `summon()` cannot be combined with any backend operation without also producing a `Self` (token) value, because every method now requires `self`.

---

## 4. Deductive proof template

For functions that produce or wrap a `Self::Repr`, the soundness argument follows a fixed shape:

1. **Token premise.** `Self` here is a token type. A value of `Self` exists only if `Self::summon()` returned `Some`, an upgrade extractor produced one, or `forge_token_dangerously()` was called. In every case, every feature in the registry's `features = [...]` list for this token is runtime-available.
2. **Intrinsic precondition.** The intrinsic's vendor docs list the required target_feature (e.g. `_mm256_set1_ps` requires `avx` per Intel SDM Vol. 2C VBROADCASTSS).
3. **Match.** The registry lists the required feature in the token's feature set, so (1) implies the precondition holds.
4. **Conclusion.** The `unsafe { ... }` block is sound at this call site.

For polyfilled paths (`f32x16<X64V3Token>::Repr = [__m256; 2]`), add:

> **Polyfill premise.** The wider Repr is `[NarrowerRepr; N]`. Constructing it as `[lo, hi, ...]` is array literal construction, sound on any platform.

For cross-token delegation (V4 delegating F32x4Backend to V3), add:

> **Subset premise.** `V4 ⊋ V3` per registry. Any intrinsic sound under V3 is sound under V4. Delegating V4's narrower-width methods to V3's impls is sound by inheritance.

Write these as SAFETY comments on the non-trivial `unsafe` blocks in backend impls. Short blocks whose soundness is visible at a glance don't need the full template.

---

## 5. Test patterns

### 5a. Compile-fail doctests

Every UFCS call to a backend method without a token must fail to compile. `magetypes/src/bypass_adversarial.rs` has 20 doctests marked `compile_fail`, one per method category:

```rust
//! ```compile_fail
//! use archmage::X64V3Token;
//! use magetypes::simd::backends::F32x8Backend;
//! let _ = <X64V3Token as F32x8Backend>::splat(7.0);
//! ```
```

If a future patch removes `self` from a method sig, the corresponding doctest starts compiling — CI goes red.

### 5b. Sanctioned counterparts

`magetypes/tests/bypass_closed.rs` has 12 matching runtime tests that build a token, call the method correctly, and assert the result. This catches the inverse failure — a doctest that fails to compile for the wrong reason (e.g. an unrelated API change).

All sanctioned tests use `ScalarToken` so every target runs them; no cfg-gating.

### 5c. Struct-layout assertions

Every `*_impl.rs` file includes:

```rust
const _: () = assert!(
    core::mem::size_of::<f32x8<X64V3Token>>()
        == core::mem::size_of::<<X64V3Token as F32x8Backend>::Repr>()
);
const _: () = assert!(
    core::mem::align_of::<f32x8<X64V3Token>>()
        == core::mem::align_of::<<X64V3Token as F32x8Backend>::Repr>()
);
```

These fire at compile time if a token gains a non-ZST field and breaks the `#[repr(C)]` invariant.

### 5d. Parity tests

For every operation, scalar (the trusted reference) must match every other backend bit-for-bit:

```rust
fn parity<T: F32x8Backend>(token: T) {
    let scalar_token = ScalarToken::summon().unwrap();
    let lhs = [1.0, 2.0, /* incl. NaN / ±0 / denormals */];
    let rhs = [/* ... */];
    let result_t: [f32; 8] = compute_via_token(token, &lhs, &rhs);
    let result_s: [f32; 8] = compute_via_token(scalar_token, &lhs, &rhs);
    for i in 0..8 {
        assert_eq!(result_t[i].to_bits(), result_s[i].to_bits());
    }
}
```

Run across the CI matrix (aarch64 Cobalt 100, V4 AVX-512, M1 macOS, wasm32-wasip1). Bit-exact agreement on NaN lanes is the canary for `Repr` corruption.

### 5e. Cross-feature-flag coverage

Parity must hold under every cargo feature combination: `--no-default-features`, `--features std`, `--features 'std avx512'`, `--features 'std w512'`, `--all-features`. The existing feature-matrix CI jobs cover this.

### 5f. Sealed-trait coverage

A periodic script checks every `pub` token type in `archmage/src/tokens/` has a matching `impl Sealed for archmage::TokenName {}` in `magetypes/src/simd/backends/sealed.rs`. A missing entry is a functional gap (the token can't be used as a backend) — catching it here avoids a later surprise.

---

## 6. Implementation notes

- **formatdoc args list.** `formatdoc!` treats `{var}` as a placeholder. A sed sweep that inserted `{p}_set1_{s}` broke codegen because `p` and `s` weren't in the args list. Check the args list before adding placeholders.

- **Static vs method context.** `self.1` only works where `self` is a value of the generic type. In static methods (`from_repr`, `from_m128`, `blend`, the construction methods themselves) there is no `self` — use the named token parameter instead. An early pass put `self.1` in static contexts and produced 50+ "expected value, found module `self`" errors.

- **Default trait method bodies.** Any default body that called `Self::splat(c)` to materialize a constant broke once `splat` required `self`. Three options:
  - (a) Take `self` in the method that needs the constant, and call `Self::splat(self, c)` — this is what task #5 did for the remaining surface.
  - (b) Use intrinsic-level constants in backend overrides — this is what x86 f32/f64 `recip`/`rsqrt` do, calling `_mm{,256,512}_set1_{ps,pd}` directly.
  - (c) Neuter the default to a passthrough assuming every backend overrides. Only use after verifying every impl overrides.

- **Multi-trait ambiguity.** Inside `impl F32x8Backend for X64V3Token`, calling `Self::rcp_approx(self, a)` is ambiguous because X64V3Token also implements F32x4Backend, F32x16Backend, etc. Use fully-qualified syntax: `<Self as F32x8Backend>::rcp_approx(self, a)`. The ambiguity only surfaces on tokens that implement many width traits — x86 hits it, NEON mostly doesn't. Safer pattern in codegen: always emit `<Self as {trait_name}>::NAME(...)` in impl bodies.

- **Cross-width helpers.** `magetypes/src/simd/generic/cross_width.rs` (PR #38) needed updating — it called `T::method(args)` and `f32x4::from_repr_unchecked(repr)` tokenless. The wider input always carries a token, so `from_halves(token, lo, hi)` threads cleanly.

- **`#[repr(transparent)]` is off the table** for a generic ZST field (E0690). `#[repr(C)]` + `const _: ()` size/alignment assertions is the equivalent pattern.

- **`bytemuck/simd` escape** stays documented as-is. Closing it requires a sealed `Repr` newtype, which is a larger redesign and out of scope.

- **Git diff over commit messages.** Commit messages overstate completeness during a long refactor. Use the diff as the authoritative record.

---

## 7. Branch state

```
fix/token-by-self-soundness  HEAD = 6cd5bb0
  6cd5bb0 docs: changelog entry for token-by-self-soundness
  ed9aa1d chore: fix toolchain drift — trybuild stderr + clippy lints
  c1dfe14 chore: cargo fmt — formatting drift from previous commits
  9d34c81 ci: run magetypes integration tests on cross + WASM targets (fixes #39)
  9c48311 fix(ci-accuracy): fix NEON precision on QEMU + widen _approx tolerance
  6ea5448 fix: NEON f32x8 polyfill Newton-Raphson for recip/rsqrt; fix wasm test strings
  4197620 feat: emit compile-time layout asserts in generic `*_impl.rs` files
  b635ae3 test: adversarial bypass-closure suite (20 compile_fail × 12 sanctioned)
  f2a76fd fix: thread `self` through all ARM/WASM polyfill UFCS trait calls
  bf58cf5 docs: mark task #5 complete — every backend method self-gated
  4447892 fix: complete task #5 — all backend trait methods now take `self`
  00b7c5e docs: update soundness handoff with compile-green + partial task-5 state
  4e31ec9 fix: thread self through recip/rsqrt/rcp_approx/rsqrt_approx; x86 Newton refine
  4d74432 fix: drive token-by-self soundness cascade to green build
  687072b docs: mission-critical soundness handoff for next session
  5b0ecf3 WIP: token-by-self soundness fix — DOES NOT COMPILE
  ce5169b chore: bump version to 0.9.20                           [origin/main]
```

Feature matrix green on default, `--features avx512`, `--no-default-features`, `--features w512`, `--all-features`. 1545 integration tests pass on x86; aarch64 and wasm32 run via cross/wasmtime in CI.

PR #40 tracks the branch. `cargo semver-checks` flags five inherent methods on `f32x4`/`f32x8` as parameter-count breakage — all were themselves part of the bypass and are expected breakage. The other ~1100 trait-signature changes are shielded by `Sealed`.

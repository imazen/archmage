# Token-by-Self Soundness Refactor — Mission-Critical Handoff

**Status as of HEAD (`fix/token-by-self-soundness`, commit `4447922`):** **task #5 complete.** Every method on every backend trait now takes `self`. Fabricated `Repr` values (via `bytemuck/simd` feature or `mem::transmute` — documented escapes) cannot feed any backend operation without also presenting a `Self` token value. magetypes builds clean on default / avx512 / no-default-features / w512 / all-features. 1545 integration tests pass. The bypass-closure PoC (`<X64V3Token as F32x8Backend>::splat(7.0)` — and now any other backend method called UFCS without a token) fails to compile.

**Previous status (commit `4e31ec9`):** construction + neg + reciprocal self-gated; arithmetic/comparison/reduction/memory/bitwise/shift/math/blend still tokenless.

**Previous status (commit `5b0ecf3`):** WIP, ~402 compile errors. Trait-side bypass closed; cascade not converged.

**You** are the next session picking this up. This document is your full briefing. Read it before touching any code.

---

## 1. The promise archmage makes

> **A token value's existence is proof that its CPU features are runtime-available.**

Tokens are zero-sized types (`X64V3Token`, `NeonToken`, `ScalarToken`, etc.) that can only be constructed via:

1. `T::summon() -> Option<T>` — runtime-detects the features, returns `Some` only if all are present
2. Token "extractor" methods on stronger tokens (`X64V4Token::v3() -> X64V3Token`) — sound because the stronger token already proves a superset
3. `T::forge_token_dangerously() -> T` — explicitly `unsafe`, the documented escape hatch

A function taking `T: F32x8Backend` parameter can therefore `unsafe { _mm256_*(...) }` inside its body, and `#[arcane]` / `#[rite]` macros wrap that with `#[target_feature(enable = "avx2,...")]` automatically. **No `unsafe` for the user.**

The promise breaks if any safe-code path can:
- Call a backend trait method (`T::splat`, `T::add`, etc.) without first holding a `T` value
- Construct a `Repr` value (`__m256`, `float32x4_t`, etc.) without going through token-gated machinery, AND then wrap or operate on that `Repr` without a token

The **safety oracle** for any change to magetypes/archmage is:

> *Can a downstream crate, with `#![forbid(unsafe_code)]` and without `bytemuck`'s `simd` feature, cause a `vbroadcastss %ymm` (or any other above-baseline instruction) to execute in their compiled binary, without first calling `summon()` / an extractor / `forge_token_dangerously`?*

If yes: bypass.

---

## 2. The agent-found PoC (closed at trait sig but cascade incomplete)

```rust
#![forbid(unsafe_code)]
use archmage::X64V3Token;
use magetypes::simd::backends::F32x8Backend;
fn main() {
    let _ = <X64V3Token as F32x8Backend>::splat(7.0);  // emits vbroadcastss
    let _ = <X64V4Token as F32x16Backend>::splat(7.0); // emits vbroadcastss %zmm
}
```

Why it worked: trait methods like `fn splat(v: f32) -> Self::Repr` are associated functions — no `self`, no token value parameter. Sealing the trait prevents *implementing* it externally but doesn't prevent *calling* its associated functions UFCS-style.

The branch's commit `5b0ecf3` makes this PoC **fail to compile** at the trait-sig level: `splat` now requires `self`. That's the soundness floor. The cascading work is making the rest of the codebase compile around the new shape.

---

## 3. The fix design (in two parts)

### Part A: Trait method receivers

Every backend trait method that can produce or operate on `Self::Repr` takes `self,` (or `_: Self,`) as its first parameter:

```rust
// Before
pub trait F32x8Backend: SimdToken + Sealed + Copy + 'static {
    type Repr: Copy + Clone + Send + Sync;
    fn splat(v: f32) -> Self::Repr;  // ❌ no self, callable without a token value
    fn add(a: Self::Repr, b: Self::Repr) -> Self::Repr;  // ❌
    // ...
}

// After
pub trait F32x8Backend: SimdToken + Sealed + Copy + 'static {
    type Repr: Copy + Clone + Send + Sync;
    fn splat(self, v: f32) -> Self::Repr;  // ✅ requires `self`
    fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;  // ✅
    // ...
}
```

The current branch only does this for **construction** methods (`splat`, `zero`, `load`, `from_array`). **Full coverage means every trait method**, otherwise downstream code can fabricate Reprs (via bytemuck-with-simd, or via future stable Rust changes) and feed them to the still-tokenless arithmetic methods.

### Part B: Generic struct stores the token

```rust
// Before
pub struct f32x8<T: F32x8Backend>(T::Repr, PhantomData<T>);

// After
pub struct f32x8<T: F32x8Backend>(T::Repr, T);
```

`T` is ZST → `sizeof(f32x8<T>) == sizeof(T::Repr)` unchanged → `#[repr(transparent)]` preserved. The `T` field is the stored token, accessible via `self.1` in any method receiving `self: f32x8<T>`. This lets operator impls and block-ops methods re-supply the token to backend calls without asking the user for it.

`from_repr_unchecked` necessarily takes `(token, repr)` — without a token, you can't construct a `f32x8` because the `T` field needs a value.

---

## 4. Deductive proof template (use this for every impl)

For any function or impl that produces a `Self::Repr` or wraps one, write a soundness comment with this structure:

```rust
// SAFETY (deductive proof):
//
// Premise 1 (token-as-proof): `Self` here is a token type (one of {ScalarToken,
//   NeonToken, X64V3Token, X64V4Token, Wasm128Token}). By the archmage soundness
//   contract, a value of `Self` exists only if `Self::summon()` returned `Some`
//   (or an upgrade extractor / `forge_token_dangerously` produced one — both
//   covered by the chain). Therefore: every feature in the registry's
//   `features = [...]` list for this token is runtime-available.
//
// Premise 2 (intrinsic preconditions): the intrinsic `_mm256_set1_ps` (Intel
//   SDM Vol. 2C, VBROADCASTSS) requires `target_feature = "avx"`. The registry
//   lists `avx` in X64V3Token's feature set; therefore Premise 1 implies AVX
//   is runtime-available at this call site.
//
// Premise 3 (Rust-level `unsafe` precondition): `_mm256_set1_ps` is documented
//   `unsafe` because executing it on a CPU without AVX is UB. By Premises 1-2,
//   AVX is runtime-available. The `unsafe { ... }` block is sound.
//
// Conclusion: this function may be called from any context that has a `Self`
//   value (which Premise 1 says is the only way to obtain `Self` in safe code).
//   The result is a valid `__m256` representing splat(v).
```

**Every** non-trivial `unsafe { ... }` block in backend impls deserves this proof. The pattern is:
1. **Token premise** (what the token guarantees)
2. **Intrinsic precondition** (what the platform vendor documents as required)
3. **Match** (precondition ⊆ token's guarantee → sound)
4. **Conclusion** (what the function output represents)

For polyfilled paths (e.g. `f32x16<X64V3Token>::Repr = [__m256; 2]`), the proof additionally cites the array-construction step as soundness-trivial:

> Premise 4 (polyfill): the wider Repr is `[NarrowerRepr; N]`. Constructing it as `[lo, hi, ...]` is array literal construction, sound on any platform.

For cross-token delegation (e.g. V4 delegating F32x4Backend to V3), the proof includes a **subset-relation** premise:

> Premise 5 (subset): `V4 ⊋ V3` per registry. Any intrinsic sound under V3's feature set is sound under V4's (which is a strict superset). Therefore delegating V4's narrower-width methods to V3's impls is sound by inheritance.

---

## 5. Adversarial testing patterns

The bypass-closure tests must be **adversarial** — they try to break the contract. A passing test isn't proof; a *failing-to-compile* test is.

### Pattern 1: Compile-fail PoC

```rust
//! ```compile_fail
//! use archmage::X64V3Token;
//! use magetypes::simd::backends::F32x8Backend;
//! let _ = <X64V3Token as F32x8Backend>::splat(7.0);  // MUST fail to compile
//! ```
```

Use `///` doctest blocks with `compile_fail` for each trait method. If a future patch makes `splat` callable without a token, this test starts compiling — i.e. the doctest fails — i.e. CI red.

### Pattern 2: Bit-precise round-trip parity

For every operation that produces a `Repr`, verify Scalar (the trusted reference) matches every other backend bit-for-bit:

```rust
fn parity<T: F32x8Backend>(token: T) {
    let scalar_token = ScalarToken::summon().unwrap();
    let lhs = [1.0, 2.0, ...];  // distinguishable, includes NaN/±0/denormals
    let rhs = [...];

    let result_t: [f32; 8] = compute_via_token(token, &lhs, &rhs);
    let result_s: [f32; 8] = compute_via_token(scalar_token, &lhs, &rhs);

    for i in 0..8 {
        assert_eq!(result_t[i].to_bits(), result_s[i].to_bits(),
                   "lane {i}: backend disagrees with scalar");
    }
}
```

Run for every backend on every relevant runner (Cobalt 100 ARM, V4 AVX-512 host, M1 macOS, Wasm32-wasmtime). NaN-bit preservation is the canary for any subtle Repr corruption.

### Pattern 3: Repr-fabrication attempt

Verify that the only way to obtain a `Repr` value goes through token-gated machinery. List every safe-Rust path that constructs `__m256`, `float32x4_t`, `v128`, `[f32; 4]`, etc.:

- `Default::default()` — verify `__m256` doesn't impl Default (it doesn't, but check on every Rust version bump)
- `bytemuck::cast::<[f32; 8], __m256>(arr)` — works **only with `bytemuck/simd` feature**. Document this as the known escape; magetypes' `Cargo.toml` already does.
- `core::mem::zeroed::<__m256>()` — `unsafe`, out of scope
- Any future stable Rust addition (e.g. const construction of `__m256`)

Add a `cargo deny` rule prohibiting `bytemuck/simd` in magetypes's allowed-feature list, OR keep the existing `Cargo.toml` documentation and accept the documented escape.

### Pattern 4: Cross-feature-flag combinatorics

Run the parity tests under every cargo feature combination — `--no-default-features`, `--features std`, `--features 'std avx512'`, `--features 'std w512'`, `--all-features`. The bypass closure must hold under all of them. CI matrix should already have these (see archmage's existing `Features (...)` jobs).

### Pattern 5: ZST struct layout assertion

After the storage refactor (`PhantomData<T>` → `T`), verify in tests:

```rust
const _: () = assert!(core::mem::size_of::<f32x8<X64V3Token>>() == core::mem::size_of::<__m256>());
const _: () = assert!(core::mem::align_of::<f32x8<X64V3Token>>() == core::mem::align_of::<__m256>());
```

If `T` ever stops being ZST (someone adds a non-ZST field to a token by accident), the `repr(transparent)` invariant breaks and these `const _: ()` asserts fail at compile time.

### Pattern 6: Sealed-trait coverage audit

Periodically run a script that grep's for every `pub` token type in `archmage/src/tokens/` and verifies it `impl Sealed for archmage::TokenName {}` exists in `magetypes/src/simd/backends/sealed.rs`. Missing tokens are a functional gap (can't be backend tokens) but also a soundness audit signal — if someone adds a token without sealing it, downstream could implement backends for it.

---

## 6. Construction surface inventory

Every path that produces or wraps a `Repr`, audited for token-gating:

| Path | Pre-fix | Post-fix (target) | Status on branch |
|---|---|---|---|
| `T::splat(v)` (trait UFCS) | ❌ no token | ✅ `T::splat(self, v)` | ✅ done |
| `T::zero()` | ❌ | ✅ `T::zero(self)` | ✅ done |
| `T::load(data)` | ❌ | ✅ `T::load(self, data)` | ✅ done |
| `T::from_array(arr)` | ❌ | ✅ `T::from_array(self, arr)` | ✅ done |
| `T::neg(a)` | ❌ | ✅ `T::neg(self, a)` | ✅ done (commit `4d74432`) |
| `T::rcp_approx(a)` / `rsqrt_approx(a)` | ❌ | ✅ `T::rcp_approx(self, a)` etc. | ✅ done (commit `4e31ec9`) |
| `T::recip(a)` / `rsqrt(a)` | ❌ | ✅ `T::recip(self, a)` etc. + x86 Newton override | ✅ done (commit `4e31ec9`) |
| `T::add(a, b)` / `sub` / `mul` / `div` | ❌ no token, takes Reprs | ✅ `T::add(self, a, b)` | ✅ done (commit `4447922`) |
| `T::min` / `max` / `abs` / `sqrt` / `floor` / `ceil` / `round` / `trunc` | ❌ | ✅ `T::min(self, a, b)` etc. | ✅ done (commit `4447922`) |
| `T::mul_add` / `mul_sub` | ❌ | ✅ `T::mul_add(self, a, b, c)` | ✅ done (commit `4447922`) |
| `T::simd_eq` / `ne` / `lt` / `le` / `gt` / `ge` / `blend` | ❌ | ✅ `T::simd_eq(self, a, b)` etc. | ✅ done (commit `4447922`) |
| `T::reduce_add` / `reduce_min` / `reduce_max` | ❌ | ✅ `T::reduce_add(self, a)` | ✅ done (commit `4447922`) |
| `T::store(repr, &mut [..])` | ❌ | ✅ `T::store(self, repr, ..)` | ✅ done (commit `4447922`) |
| `T::to_array(repr) -> [..]` | ❌ | ✅ `T::to_array(self, repr)` | ✅ done (commit `4447922`) |
| `T::not` / `bitand` / `bitor` / `bitxor` | ❌ | ✅ `T::not(self, a)` etc. | ✅ done (commit `4447922`) |
| `T::shl_const::<N>` / `shr_logical_const` / `shr_arithmetic_const` | ❌ | ✅ `T::shl_const(self, a)` | ✅ done (commit `4447922`) |
| `T::all_true` / `any_true` / `bitmask` / `popcount` / `popcnt` | ❌ | ✅ `T::all_true(self, a)` etc. | ✅ done (commit `4447922`) |
| `T::clamp` (default body) | ❌ | ✅ fully-qualified `<Self as {trait}>::min/max` with `self` | ✅ done (commit `4447922`) |
| `T::bitcast_*` / `convert_*` (convert traits) | ❌ | ✅ `T::bitcast_(self, a)` | ✅ done (commit `4447922`) |
| `f32x8::splat(token, v)` (generic API) | ✅ takes token | ✅ unchanged shape | ✅ done |
| `f32x8::from_repr(token, repr)` | ✅ | ✅ stores token | ✅ done |
| `f32x8::from_repr_unchecked(repr)` | ⚠️ pub(super), no token | ✅ pub(crate), takes token | ✅ done |
| `f32x8::from_u8` / `load_4_rgba_u8` / `load_8_rgba_u8` / `load_8x8` | static, no token | takes `token: T` | ✅ done (commit `4d74432`) |
| `f32x8::interleave_lo/hi/interleave` + other self-receivers | used `self.1` from wrong scope | uses `self.1` correctly | ✅ done (commit `4d74432`) |
| `f32x8::deinterleave_4ch` / `interleave_4ch` / `transpose_4x4` / `transpose_8x8` | static over `[Self; N]`, no token | pulls token from `arr[0].1` | ✅ done (commit `4d74432`) |
| `from_m128`, `from_v128`, `from_float32x4_t` (platform conversions) | takes `_: T` — token discarded | takes `token: T` — store it | ✅ done (commit `4d74432`) |
| Generic struct storage (`PhantomData<T>` → `T` field) | `#[repr(transparent)]` | `#[repr(C)]` with `pub(crate)` trailing ZST `T`; size+align preserved | ✅ done (commit `4d74432`) — `#[repr(transparent)]` cannot be proven for a generic ZST field at struct-def time |

**Reprs that can be fabricated outside any trait path** (these are escapes, document them):
- `bytemuck::cast::<[f32; 8], __m256>(arr)` with `bytemuck/simd` feature → produces `__m256`. Documented escape per magetypes `Cargo.toml`.
- `core::mem::transmute::<[f32; 8], __m256>(arr)` — `unsafe`, user opted in.
- `core::mem::zeroed::<__m256>()` — `unsafe`, user opted in.

A fabricated `Repr` is not a soundness break **as long as the methods that operate on it require a token**. This property now holds across the full trait surface: every `add`/`sub`/`mul`/`div`/`min`/`max`/`store`/`to_array`/`reduce_add`/`bitand`/`shl_const`/`simd_eq`/`blend`/`bitcast_*`/`convert_*`/etc. method requires `self: Self` as its first argument. A bytemuck-fabricated `__m256` that never passed through `summon()` cannot be combined with any backend operation without also producing a `Self` value — and the only paths to `Self` remain `summon()`, extractor methods like `X64V4Token::v3()`, or the explicitly `unsafe` `forge_token_dangerously()`.

---

## 7. Trait method inventory by category

### 7a. Per-element-type backend traits

Backend traits live in `magetypes/src/simd/backends/{f32x4,f32x8,f32x16,f64x2,f64x4,f64x8,i8x16,i8x32,i8x64,i16x8,i16x16,i16x32,i32x4,i32x8,i32x16,i64x2,i64x4,i64x8,u8x16,u8x32,u8x64,u16x8,u16x16,u16x32,u32x4,u32x8,u32x16,u64x2,u64x4,u64x8}.rs`. ~30 traits total.

Each trait has roughly:

- **Construction (4)**: `splat`, `zero`, `load`, `from_array`
- **Memory (2)**: `store`, `to_array`
- **Arithmetic (4-5)**: `add`, `sub`, `mul`, `div` (float), `neg`
- **Math (8-9)**: `min`, `max`, `sqrt` (float), `abs`, `floor`/`ceil`/`round`/`trunc`, `mul_add`, `mul_sub`
- **Comparisons (7)**: `simd_eq`, `simd_ne`, `simd_lt`, `simd_le`, `simd_gt`, `simd_ge`, `blend`
- **Reductions (3)**: `reduce_add`, `reduce_min`, `reduce_max`
- **Approximations (4 floats only)**: `rcp_approx`, `rsqrt_approx`, `recip`, `rsqrt`
- **Bitwise (4)**: `not`, `bitand`, `bitor`, `bitxor`
- **Shifts (3 ints only)**: `shl_const`, `shr_logical_const`, `shr_arithmetic_const`
- **Boolean (4)**: `all_true`, `any_true`, `bitmask`, `popcount`
- **Default helpers**: `clamp` (already self-free since it uses min/max only)

**Roughly 35-40 methods × ~30 traits = ~1100 method signatures.**

### 7b. Convert traits

`F32x4Convert`, `F32x8Convert`, `F32x16Convert`, `U32x4Bitcast`, `U32x8Bitcast`, `I64x2Bitcast`, `I64x4Bitcast` — about 5 methods each, ~7 traits. Cross-type bitcasts and conversions. All need `self`.

### 7c. Extension traits

`PopcntBackend` (V4x), maybe others. Search `pub trait .*Backend\b` in `magetypes/src/simd/backends/*.rs` for the full list.

### 7d. Generic exposure on `f32xN<T>`

For each backend trait method, `magetypes/src/simd/generic/generated/{type}_impl.rs` exposes a corresponding method or operator. After storage refactor, every one needs `T::method(self.1, ..., ...)` instead of `T::method(...)`.

**Roughly 20 generic types × ~30 methods = ~600 generic-exposure call sites.**

### 7e. Operator impls

`Add`, `Sub`, `Mul`, `Div`, `BitAnd`, `BitOr`, `BitXor`, `AddAssign` (etc.), `Neg`, `Index`, `IndexMut`. Generated by `gen_operators` / `gen_assign_operators`. ~10 ops × 20 types × {token-passing on construction calls} = ~200 sites.

### 7f. Block ops (`block_ops_*.rs`)

Memory-layout / interleave / transpose / pixel-channel ops. Heavy callers of `T::from_array(...)` for narrower-width construction inside methods on wider types. Codegen lives in `xtask/src/simd_types/block_ops.rs`, `block_ops_arm.rs`, `block_ops_wasm.rs`. ~150 sites need `T::from_array(self.1, arr)` instead of `T::from_array(arr)`.

---

## 8. Codegen surgery roadmap

### Completed (commits `4d74432`, `4e31ec9`)

1. ✅ Static-method `self.1` regression — fixed. Platform raw-conversion methods (`from_m128`, `from_m256`, `from_m128d`, `from_m256d`, `from_m128i`, `from_m256i`) now take `token: T`. `blend` uses `mask.1`.

2. ✅ Block-ops codegen surgery — `xtask/src/simd_types/generic_gen/block_ops.rs`. All `T::from_array(arr)` → `T::from_array(self.1, arr)` (or `token, arr` for static methods that took a new leading `token: T` param). Array-of-Self static methods (`deinterleave_4ch`, `interleave_4ch`, `transpose_4x4`, `transpose_8x8`) pull token from `arr[0].1`.

3. ✅ V3 W512 polyfill construction delegations — `splat`, `zero`, `load`, `from_array` now pass `self` to the half-trait calls. `neg` (signed) passes `self`; `neg` (unsigned) uses `zero(self)` to build the sub operand.

4. ⚠️ **Partial**: Stage 2 full coverage. `self` is threaded through construction methods (`splat`, `zero`, `load`, `from_array`) + `neg` + reciprocal surface (`rcp_approx`, `rsqrt_approx`, `recip`, `rsqrt`). x86 f32/f64 impls gained explicit Newton-Raphson `recip`/`rsqrt` overrides using value-based splat intrinsics (no token required for the splat; impl block gated on target feature).

5. ⚠️ **Partial**: Generic-exposure threading. All generic-exposure sites for the self-threaded methods are done (including the scalar-broadcast operator bodies, which pass `self.1` to `T::splat`). Arithmetic/comparison/etc. still call `T::method(self.0, ...)` tokenless — they'll be updated when the trait gets `self`.

6. ⚠️ **Partial**: Tests updated for the currently-self-gated surface. `generic_block_ops.rs`, `generic_ergonomics.rs`, and the generated `generated_simd_types.rs` tests all pass the token. Transcendentals test `recip_approx` now passes (was the last failing test).

7. ✅ Convergence to 0 errors — achieved. Full feature matrix builds clean.

### Remaining (follow-up work — none blocks soundness)

The core soundness refactor is complete. The outstanding items below are polish, testing depth, and cross-arch verification — not additional API surgery.

**Cross-arch verification** — `cargo check --target aarch64-unknown-linux-gnu`, `--target wasm32-unknown-unknown`, `--target aarch64-pc-windows-msvc`. Each may surface backend-specific issues (e.g., a NEON impl that forgot a `self,` the perl sweep didn't catch). Fix individually.

**Adversarial test suite** — expand `magetypes/tests/bypass_closed.rs` to exercise every trait method UFCS-style and expect compile-fail without a token. Use `compile_fail` doctests for each method × each trait. This guards against any future patch that removes `self` from a method sig.

**`const _: ()` size-assertion codegen** — add compile-time asserts in `xtask/src/simd_types/generic_gen/type_impl.rs` that `size_of::<{name}<T>>() == size_of::<T::Repr>()` and `align_of::<{name}<T>>() == align_of::<T::Repr>()`. If a token ever stops being a ZST, these fire at compile time.

**Doc examples and README** — the struct layout note already explains `#[repr(C)]` + ZST trailing field. Check that `/docs/site/content/magetypes/` and `/magetypes/README.md` don't contradict the post-refactor API (particularly any surviving `#[repr(transparent)]` claims).

---

## 9. Common pitfalls

- **Sed across formatdoc strings** is dangerous because formatdoc!'s `{var}` placeholders look like Rust expressions. My initial `{p}_set1_{s}` insert broke because `p` and `s` weren't in the formatdoc args list. **Always check the args list before adding placeholders.**

- **Token in scope** — every `self.1` you write needs `self` to be a value of the generic type. In static methods (`from_repr`, `from_m128`, `blend`, the construction methods themselves) there's no `self` — use the named token parameter instead. The sed I ran put `self.1` in static contexts and produced 50+ "expected value, found module `self`" errors.

- **Default trait method bodies** that use `Self::splat(c)` for a constant: after splat takes `self`, the default bodies break. Either (a) make those methods also take `self` and use `Self::splat(self, c)`, (b) replace with intrinsic-level constants in backend overrides, or (c) neuter defaults to identity passthrough on the assumption every backend overrides (verify first!). Option (b) is what's now on the branch for `recip`/`rsqrt` on x86 f32/f64 — Newton-Raphson uses `_mm{,256,512}_set1_{ps,pd}` directly. Option (a) is what task #5 will do for the remaining surface. Option (c) was the `5b0ecf3` stop-gap.

- **Multi-trait ambiguity in impl bodies** — when an impl like `impl F32x8Backend for X64V3Token` calls `Self::rcp_approx(self, a)` inside its own `recip` override, the compiler sees multiple `rcp_approx` because X64V3Token also impls F32x4Backend, F32x16Backend, etc. Use fully-qualified syntax: `<Self as F32x8Backend>::rcp_approx(self, a)`. Codegen should thread the `trait_name` placeholder into body emissions. This is why the NEON `recip` body already uses `Self::mul(...)` without error — NEON tokens (`NeonToken`) implement fewer overlapping trait methods in the same arithmetic surface, so the ambiguity only fires on x86. Safer pattern in all new codegen: always use `<Self as {trait_name}>::NAME(...)` in impl bodies.

- **Cross-width helpers** in `magetypes/src/simd/generic/cross_width.rs` (from PR #38) need updating. They currently call `T::method(args)` and `f32x4::from_repr_unchecked(repr)` — both need token threading. The good news: cross_width helpers always have token access via the wider input (`from_halves(token, lo, hi)`).

- **`#[repr(transparent)]` invariant — lifted.** `#[repr(transparent)]` cannot be proven for a generic ZST field `T` at struct-definition time (Rust rejects with E0690: "transparent struct needs at most one field with non-trivial size or alignment, but has 2"). The branch now uses `#[repr(C)]` with the token as a trailing ZST field; size + alignment match `T::Repr` as long as tokens remain ZSTs. Assert with `const _: () = assert!(core::mem::size_of::<f32xN<T>>() == core::mem::size_of::<T::Repr>());` — if a token ever stops being ZST, this fires at compile time. `const _: ()` blocks don't exist in the codegen yet; adding them to `xtask/src/simd_types/generic_gen/type_impl.rs` is a small follow-up.

- **`bytemuck/simd` feature** is a documented escape. Don't try to close it — magetypes' `Cargo.toml` already says users who enable it opt out of safety. Closing it without breaking bytemuck users requires a sealed `Repr` newtype, which is a much bigger redesign.

- **Don't trust commit messages over `git diff`** — my own commit messages overstate completeness. The diff is authoritative.

---

## 10. Current branch state, concretely

```
fix/token-by-self-soundness  HEAD = 4447922
  4447922 fix: complete task #5 — all backend trait methods now take `self`
  00b7c5e docs: update soundness handoff with compile-green + partial task-5 state
  4e31ec9 fix: thread self through recip/rsqrt/rcp_approx/rsqrt_approx; x86 Newton refine
  4d74432 fix: drive token-by-self soundness cascade to green build
  687072b docs: mission-critical soundness handoff for next session
  5b0ecf3 WIP: token-by-self soundness fix — DOES NOT COMPILE   [previous top]
  ce5169b chore: bump version to 0.9.20                           [origin/main]
```

```
$ cargo build -p magetypes 2>&1 | grep -c '^error'
0
$ cargo test -p magetypes --tests 2>&1 | grep -c 'test result: ok'
27   # all 27 test binaries green
$ cargo test -p magetypes --tests 2>&1 | grep -oE '[0-9]+ passed' | awk '{s+=$1}END{print s}'
1545   # total tests passed; 0 failed
```

Feature matrix all green: default, `--features avx512`, `--no-default-features`, `--features w512`, `--all-features`.

**The soundness refactor is complete.** Every path that produces, wraps, or operates on a `Self::Repr` now requires a `Self` (token) value. The agent-found UFCS-bypass PoC (`<X64V3Token as F32x8Backend>::splat(7.0)`) and every other backend method invoked UFCS-style without a token fail to compile.

Stash for next session:
- `magetypes/tests/bypass_closed.rs` is the smoke test. The module-level compile_fail doctest still asserts the original PoC fails. Expanding this to cover every backend method × every trait is the main remaining task (polish — doesn't affect soundness).
- Cross-arch verification (`cross test` on aarch64-unknown-linux-gnu / wasm32-wasip1 / aarch64-pc-windows-msvc) is recommended before release.
- The full API-break summary is listed in the commit body of `4447922`.

---

## 11. Why this matters

The current archmage main has a **demonstrated soundness bypass**. Any safe-Rust user of magetypes can emit AVX/AVX-512/NEON instructions on incompatible CPUs by calling backend trait methods UFCS-style. This is CVE-class for a crate marketing itself as "safely invoke your intrinsic power."

The trait-sig change in this branch closes the demonstrated PoC. The cascade work makes the rest of magetypes compile around the new shape so users actually have a working crate. **Both halves are required to ship.**

The user's instruction was: do this mission-critically. That means: don't ship a partial fix that's worse than what's there, don't accept "good enough" on a soundness boundary, and document the design so the next session can resume without re-deriving the model.

Resume from `5b0ecf3`. Read this doc. Run `cargo build -p magetypes` and start with the static-method `self.1` regression — that's the smallest first fix that drops the error count fastest and validates you have a working environment.

Good luck.

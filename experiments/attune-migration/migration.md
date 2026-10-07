# Provisional attune migration, pinned baseline

Baseline: [imazen/archmage cf07592212e96294ef9ca5dca9a364fa8d15d8ad](https://github.com/imazen/archmage/tree/cf07592212e96294ef9ca5dca9a364fa8d15d8ad). `attune` and `attuned!` are proposed, not implemented. All “to” blocks below are design sketches, not verified compiling Rust. Ellipses stand for the **unchanged original body**, subject to the token/call adaptations explicitly described. The occurrence index preserves exact “from” spans and points to these destination rules. Apply the method, generics, cfg and visibility qualifications as well as the numbered base rule.

MISSING: finalized trait-wrapper placement, `entry(token)` compatibility grammar, tier/cfg selection grammar, generic `Token` substitution, explicit-inline override policy, and a way to describe callee availability across crates. No measurements of attune itself establish equivalent codegen or numerical results; parent-agent wrapper measurements are recorded in P13. These gaps prevent an automatic compiling rewrite; they do not justify dropping source occurrences.

## Output contract used throughout

| Proposed request | Exposed output names for source `foo` |
|---|---|
| `#[attune(make(_*))]` | Direct tokenless `foo_<tier>` for the default tier set |
| `#[attune(make(_*_t))]` | Explicit-token `foo_<tier>_t(token, ...)` trampolines |
| `#[attune(make(_))]` | Tokenless central dispatcher `foo(...)` |
| `#[attune(make(all))]` | Exact union of those three sets, not every registry tier |
| `#[attune(make(_v3))]` | Only the selected direct `foo_v3` |
| `#[attune(make(_v3_t))]` | Only the selected explicit-token `foo_v3_t` |
| `#[attune(v3)]` | One direct function with the original name |
| Bare `#[attune]` on `foo_v3` | Infer v3; preserve `foo_v3`; one direct function |

Requested outputs have source visibility (`pub`, `pub(crate)`, restricted or private); generated implementation helpers are private. “Exposed” does not imply `pub` or a guaranteed linker symbol: it means a Rust item name at that visibility. A direct function with target features is callable safely only from an adequate feature context; an ordinary caller needs a safe boundary with a proof or runtime dispatch. A token argument in ordinary Rust does not itself establish a target-feature context.

Tentative defaults: direct `#[inline]`, thin explicit-token trampoline `#[inline(always)]`, dispatcher no forced inline. User inline overrides and conflicting attributes need design, not guessed syntax. Generic helpers that stay plain Rust retain their existing inline policy.

The current default tier list is **v4, v3, neon, wasm128, scalar**, not the registry: [tiers.rs:197](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/tiers.rs#L197). Current `incant`/`magetypes` defaults auto-gate v4 on the caller's `avx512` feature; autoversion does not. Whether attune preserves that auto-gating is unresolved. Do not imply that `make(all)` adds v4x, crypto, ARM-v2/v3, or relaxed WASM.

## P0

**Supporting attributes and ordinary functions → retain, then assess attachment.** The index intentionally includes all attributes, a superset of function attributes: module cfgs and macro-generated function templates can affect every occurrence below them. `#[test]`, lints, derives, reprs and non-dispatch attributes are not renamed to attune. Some records belong to structs/modules, not functions. Their presence is coverage evidence, not a proposed function conversion.

From ordinary generic helper with `#[inline(always)]`, e.g. [idiomatic_patterns_all.rs:83](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/idiomatic_patterns_all.rs#L83), to:

```rust
// API: Keep T, its bound, the token parameter and return type; no new items.
// INLINE: Retain this plain Rust helper's existing inline(always).
// BEHAVIOR: Bound supplies backend operations, not target features; caller supplies context.
#[inline(always)]
fn dot_kernel<T: F32x8Backend>(token: T, a: &[f32], b: &[f32]) -> f32 {
    /* original body, including token-first _t constructors */
}
```

Do not promote source comments claiming fixed slowdowns or universal inlining to measured migration guarantees. This archive contains such comments, but this task did not measure them.

## P1

**`#[arcane]` → direct attune body plus deliberately selected boundary surface.** Today arcane emits a safe token-taking boundary and a feature-enabled implementation, except scalar/WASM special cases. It filters all user inline attributes, gives the ordinary wrapper `inline(always)`, and normally gives its implementation `inline`. [arcane.rs:264](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/arcane.rs#L264), [arcane.rs:450](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/arcane.rs#L450).

From `#[arcane] fn process(token: X64V3Token, a: f32, b: f32) -> f32 { a + b }` ([sibling.rs:4](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/sibling.rs#L4)), to a direct-only alternative:

```rust
// API: Preserve process name/visibility; remove X64V3Token parameter. This breaks old calls.
// INLINE: Proposed direct inline replaces current inline(always) boundary + inline body.
// BEHAVIOR: No runtime probe; ordinary safe callers can no longer call process directly.
#[attune(v3)]
fn process(a: f32, b: f32) -> f32 { a + b }
```

For a name already suffixed, e.g. a hand-tuned `scale_plane_impl_v4x(token, plane, factor)`:

```rust
// API: Preserve the suffixed name; remove its proof parameter; visibility stays private.
// INLINE: Direct inline is tentative; old boundary was inline(always).
// BEHAVIOR: Infer v4x only; keep original arch/feature cfg; derive the body's proof locally.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[attune]
fn scale_plane_impl_v4x(plane: &mut [f32], factor: f32) {
    let token = X64V4xToken::from_context();
    /* original body; splat_t(token, factor) remains token-first */
}
```

Original: [idiomatic_patterns_all.rs:131](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/idiomatic_patterns_all.rs#L131). Do not infer a tier from an arbitrary unsuffixed name. Do not generate `foo_v3_v3` by accidentally applying a family request to an already suffixed function.

A boundary replacement is an **alternative API**, not the same signature:

```rust
// API: Source foo yields foo_v3_t(token, x); foo(token, x) disappears unless retained separately.
// INLINE: Proposed thin foo_v3_t is inline(always); private body is inline.
// BEHAVIOR: Safe explicit-proof entry, no runtime selection; only v3 is selected.
#[attune(make(_v3_t))]
pub fn foo(x: f32) -> f32 { x * 2.0 }
```

Keep the old arcane declaration/calls as legacy compatibility coverage when exact tokenful name/signature matters. `entry(token)` might eventually keep that spelling, but its syntax and emitted visibility are unresolved. Do not write an invented `#[attune(entry(token))]` as a compiling solution.

`import_intrinsics`, `import_magetypes`, `cfg(...)`, `nested`, `_self`, `inline_always`, and `suppress_const_test` are not silently forwarded to an unimplemented attune parser. Preserve explicit body imports where possible; intrinsic imports must vary by selected architecture. Tests of these options remain legacy tests until equivalent semantics are specified. `suppress_const_test` is especially not a normal migration pattern: preserve adversarial tests and do not discard identity checks.

## P2

**`#[rite]`, tiered rite and multi-tier rite → direct outputs.** Current rite attaches target features directly, strips user inline attributes (including `never` and `always`), and injects `#[inline]`: [rite.rs:305](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/rite.rs#L305), [rite.rs:456](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/rite.rs#L456), [common.rs:53](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/common.rs#L53).

From `#[rite(v3)] fn f(x: f32) -> f32 { ... }` to:

```rust
// API: Same name, visibility and tokenless signature; one direct item, no trampoline.
// INLINE: Tentative direct inline matches the current generated inline attribute.
// BEHAVIOR: Same selected v3 context and calling requirement; no probe in either form.
#[attune(v3)]
fn f(x: f32) -> f32 { /* original body */ }
```

Tokenful `#[rite] fn f(token: X64V3Token, ...)` uses P1's parameter-removal rule, but the starting function already has direct target-feature calling requirements. Derive `token` with `X64V3Token::from_context()` only if the body needs it. Preserving a generic proof parameter is a separate choice (P12), not an automatic deletion.

From multi-tier `#[rite(v3, neon, wasm128, scalar)] fn inner<const N: usize>(...)` ([tokenless_context.rs:11](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/tokenless_context.rs#L11)) to this **semantic specification**, pending explicit multi-selector grammar:

```text
// API: Generate exactly inner_v3, inner_neon, inner_wasm128, inner_scalar; preserve const N.
// INLINE: Each direct function gets tentative inline, matching current rite's policy.
// BEHAVIOR: No dispatcher or token trampolines; no added v4 tier; keep static helper calls.
attune request: direct outputs {_v3, _neon, _wasm128, _scalar} for inner
```

Do not replace a restricted tier set with default `make(_*)` unless the added/removed variants are intentional. Bare rite has no dispatch-default set; its modifiers operate on its explicit tier list ([rite.rs:49](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/rite.rs#L49)).

`#[rite(default)]` is a portable tokenless direct function with no target features. `#[rite(scalar)]` is also featureless; explicit scalar proof handling differs from a `default` fallback. Multi-rite can emit both scalar and default (the calling-convention matrix does); this does **not** imply both may be selected in one incant dispatch tier list.

## P3

**`#[autoversion]` → `#[attune(make(_))]` by default for a dispatcher-only API.** Current autoversion makes every variant private and forwards user attributes only to the dispatcher: [autoversion.rs:204](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/autoversion.rs#L204), [autoversion.rs:224](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/autoversion.rs#L224). No forced-inline dispatcher default should be described as preserving a user-supplied inline attribute unless the new propagation rule actually preserves it.

```rust
// FROM: #[autoversion] pub fn sum(data: &[f32; 4]) -> f32 { data.iter().sum() }
// API: Keep only public sum(data); generated helpers private, no new public tier names.
// INLINE: Dispatcher has no forced inline; preserve explicit dispatcher attrs by design review.
// BEHAVIOR: Runtime dispatch remains; default tier cfg policy must be settled before equivalence.
#[attune(make(_))]
pub fn sum(data: &[f32; 4]) -> f32 { data.iter().sum() }
```

`make(all)` on this public function would add public direct and `_t` names and new calling contracts; it is not a harmless spelling change. Use it only when that API expansion is intended. On a private function it adds private callable names, still potentially changing name collisions and call resolution.

Current `ScalarToken` input **stays in the dispatcher**, while legacy `SimdToken` is stripped; absent proof is auto-injected internally and stripped. [autoversion.rs:166](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/autoversion.rs#L166). From [scalar_token.rs:4](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/autoversion/scalar_token.rs#L4):

```text
// API: Legacy process(ScalarToken, data) must remain if preserving the existing contract.
// INLINE: Keep legacy dispatcher attrs until a wrapper-placement policy is approved.
// BEHAVIOR: Do not remove proof and silently break incant nesting; retain the old test.
Retain #[autoversion] on this compatibility case.
New tokenless counterpart: separate process_new(data), central-dispatch request make(_).
```

For legacy `SimdToken`, the externally visible dispatcher is already tokenless, but tier bodies use substituted proofs. Reconstruct concrete proofs as required; retain the deprecation test. Unsafe functions stay unsafe; return types, generics and self receivers are not erased.

`autoversion(cfg(simd_opt))` emits a scalar fallback when disabled, not simple function omission. Preserve this distinction from arcane/rite cfg omission. The proposed `tiers(cfg(...))`/whole-function cfg policy is unresolved; keep the fixture until specified. Explicit old tier lists, modifiers and fallback choices must accompany the request as semantic requirements, not silently become defaults.

## P4

**`#[magetypes]` → output selection plus explicit treatment of Token substitution.** Current magetypes clones signatures, preserves source visibility, substitutes identifier `Token` with a concrete type per tier, and optionally injects local vector aliases. Default mode uses arcane boundaries; `rite` flag uses direct feature functions. Scalar/default variants bypass that wrapping. [magetypes.rs:38](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/magetypes.rs#L38).

From `#[magetypes(...)] fn dot_impl(token: Token, a: &[f32], b: &[f32])` to a **semantic specification**:

```text
// API: New direct dot_impl_<tier>(a,b) removes token; new _t names retain explicit proof entry.
// INLINE: Direct inline; _t inline(always). These are tentative, not codegen measurements.
// BEHAVIOR: Per-tier concrete body calls dot_kernel(concrete_token,a,b); keep backend bounds.
attune request: make(_*) for a direct-only internal family,
                make(_*_t) for a cold-callable explicit-proof family,
                or make(all) only if a central dispatcher and both families are wanted.
Keep the original tier selection. Generic Token substitution syntax is UNRESOLVED.
```

For current tokenful `dot_impl_v3(token, a, b)`, the natural new explicit entry is `dot_impl_v3_t(token, a, b)`: this is a rename, not ABI/source equivalence. `magetypes(rite, ...)` is closest to the direct family, but token parameters still need deliberate adaptation. New wildcard outputs must not assume every registered backend satisfies every existing bound.

A concrete specialization can avoid placeholder ambiguity:

```rust
// API: Private direct dot_impl_v3(a,b); preserve return type; remove only concrete proof input.
// INLINE: Direct inline tentative; dot_kernel retains its separate plain-Rust inline policy.
// BEHAVIOR: from_context creates the proven v3 token; generic kernel remains generic Rust.
#[attune(v3)]
fn dot_impl_v3(a: &[f32], b: &[f32]) -> f32 {
    dot_kernel(X64V3Token::from_context(), a, b)
}
```

`define(f32x8)` currently means a local alias of `magetypes::simd::generic::f32x8<Token>`. Do not describe vector types themselves as removed generics. Keep `f32x8::<T>`, ordinary `T`/`N`, where clauses and associated bounds; only the macro placeholder is subject to per-tier substitution. `_t` vector operations still take token **first**, even if the original function accepted its proof last. Default tier has no `Token` replacement in current magetypes; a body that uses that placeholder cannot simply be assigned a default variant without specifying a concrete portable backend.

## P5

**Ordinary `incant!(foo(args), optional_tiers)` → `attuned!(foo(args))`, conditional on outputs and context.** Original call text, including path, turbofish, `Token` slot and tier list, is preserved in the occurrence index. Never lose an explicit tier list while shortening syntax.

```rust
// FROM: incant!(dot_impl(a, b)) in a plain public wrapper
// API: Wrapper signature unchanged; callee must expose _t entries or a central dispatcher.
// INLINE: Call macro supplies no function inline attr; destination policies govern inlining.
// BEHAVIOR: Outside an attune context dispatch remains runtime; do not call bare feature fns.
attuned!(dot_impl(a, b))
```

```rust
// FROM: incant!(inner::<N>(values), [v3, neon, wasm128, scalar]) in a tier body
// API: Preserve path, N and data arguments; remove only proof placeholder if present.
// INLINE: No new call-site attribute; selected direct callee uses tentative inline.
// BEHAVIOR: In attune, select only a covered tier; direct call or proposed public _t path, no probe. Preserve tier restriction.
attuned!(inner::<N>(values)) // Callee family must encode the original restricted tier set.
```

The last form is not a settled spelling for an independent call-site tier restriction. If different calls select different subsets, a declaration-only tier set cannot preserve both; keep legacy calls pending call-site grammar or create separately named families with explicit API review.

Current tokenful arcane/rite-body rewriting can **probe stronger or unrelated tiers** before exact/downgrade/fallback calls; tokenless rite context rewriting only chooses covered tiers. Thus the first kind changes selected implementation under new static composition. [rewrite.rs:119](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/rewrite.rs#L119), [rewrite.rs:279](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/rewrite.rs#L279). Keep `arcane_upgrade`, `autoversion_upgrade`, gated upgrade and cross-branch fixtures as legacy tests. Numerical output might differ across implementations; this inventory proves no parity.

Current `Token` placeholder can request a first/middle/last proof slot. Without it, incant prepends proof. For new direct bodies remove the placeholder from call arguments; for `_t` entries normalize proof first and update signature/callers together. Never remove an ordinary user variable merely because its spelling resembles a token. Preserve argument evaluation order and single evaluation; effects in discarded expressions need dedicated tests before implementation.

## P6

**`incant!(foo(args) without token)` → ordinary `attuned!`, only inside attune context.** Current form directly picks the **exact caller suffix**, with no independent tier list; the new model can pick covered lower tiers. [incant.rs:89](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/incant.rs#L89), [rewrite.rs:125](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/rewrite.rs#L125).

```rust
// FROM: incant!(inner::<3>(values) without token)
// API: No token argument added; preserve generic call and value arguments.
// INLINE: No call attribute; callee's direct inline policy applies.
// BEHAVIOR: Covered-tier static selection replaces exact-suffix selection; no upgrade/probe.
attuned!(inner::<3>(values))
```

If exact suffix selection is part of a test's assertion, retain that legacy test. Moving this call into a cold function is not equivalent: cold attuned is runtime dispatch, while current `without token` there errors. No new with-token/without-token spellings are proposed.

## P7

**`incant!(foo(args) with token_expr, tiers)` → retain compatibility or explicit concrete `_t` call.** Current passthrough inspects exact token identity via `IntoConcreteToken`; it is not ordinary feature detection, and a stronger token is not automatically its weaker ancestor. Do not replace it with cold runtime attuned and claim unchanged behavior.

```rust
// FROM: incant!(foo(x) with token, [v3, scalar]), where token is known concrete v3
// API: Call renamed explicit-token entry; its token is first; keep caller's own signature.
// INLINE: New thin _t boundary is tentative inline(always).
// BEHAVIOR: No probe; use known v3 proof. Only valid when original dispatch selected v3.
foo_v3_t(token, x)
```

```text
// API: Preserve generic T: IntoConcreteToken and all original arguments/bounds.
// INLINE: Existing function attributes stay; no inferred feature context from generic T.
// BEHAVIOR: Retain exact-type passthrough, including unmatched-token behavior, as legacy coverage.
Keep incant!(foo(args) with token_expr, original_tiers) for unresolved generic cases.
```

The `should-fail/incant_passthrough` case concerns its committed expanded output's unstable `panic_internals` diagnostic; do not assert the generic source form is universally rejected. [soundness_exploits.rs:50](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness_exploits.rs#L50).

## P8

**Aliases → same semantics as their underlying macro, while alias tests remain.** `simd_fn` and `token_target_features_boundary` route to arcane (P1); `token_target_features` to rite (P2); `simd_route` and `dispatch_variant` to incant (P5–P7). [lib.rs:210](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/lib.rs#L210), [lib.rs:607](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/lib.rs#L607).

```rust
// FROM: #[token_target_features] fn f(token: X64V3Token, x: f32) -> f32 { x }
// API: Same f name/visibility; concrete proof parameter removed deliberately.
// INLINE: Direct inline matches current rite alias expansion's attribute.
// BEHAVIOR: Keep feature-context requirement; alias compatibility test itself remains legacy.
#[attune(v3)]
fn f(x: f32) -> f32 { x }
```

Alias forwarding into ordinary incant does not prove alias recognition by the **body rewriter**: current body scanners specifically name `incant`/`dispatch_variant`. Do not assume `simd_route` inside a feature body is rewritten identically. Token type aliases (`Desktop64`, `Arm64`, etc.) are also not macro aliases: resolve their registry tier, preserve relevant safety identity tests, and do not rewrite arbitrary user aliases by spelling alone.

## P9

Expected failures and compatibility-only cases: [migration-contracts.md](migration-contracts.md#p9). Preserve rejection intent; do not turn them into successful migration examples.

## P10

Comments, diagnostic strings and embedded fixture strings are not independent macro invocations in the containing Rust file. Raw indices retain them separately. The strings in `soundness_exploits.rs` include actual generated test programs; see [failure classification](migration-contracts.md#p9). Documentation examples are proposals to adapt only when their surrounding prose is updated, never extra code-count occurrences.

## P11

Generated `.expanded.rs` snapshots are evidence of current macro output, not hand-edit targets. Preserve legacy snapshots. New attune expansion tests need separate inputs and regenerated expected outputs after implementation. [The harness](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/macro_expand.rs#L13) also separately compiles input/output families; mere snapshot generation does not establish soundness or compilation.

## P12

Generic parameters, trait bounds, receivers, trait methods and nested functions add requirements to P1–P8: [migration-contracts.md](migration-contracts.md#p12).

## P13

Cfg, inline override and cross-crate visibility requirements: [migration-contracts.md](migration-contracts.md#p13). In particular a public `_t` wrapper around a private implementation is an open alternative static entry path; named public direct functions are **not** mandatory for every public family.

# Additional migration contracts

Read with [migration.md](migration.md). All proposed snippets are provisional. Source references use the pinned baseline, not an implementation of attune.

## P12

### Ordinary generics and where clauses

From [generic_where_clause.rs:4](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/generic_where_clause.rs#L4):

```rust
#[arcane]
fn sum_slice<T>(token: X64V3Token, data: &[T]) -> f32
where T: Copy + Into<f32> { /* original loop */ }
```

To:

```rust
// API: Same sum_slice<T> name/bounds/return/visibility; remove only the concrete proof input.
// INLINE: Direct inline proposed; replaces arcane's inline(always) outer wrapper.
// BEHAVIOR: Caller must cover v3. T stays a real generic type, not a tier placeholder.
#[attune(v3)]
fn sum_slice<T>(data: &[T]) -> f32
where T: Copy + Into<f32> {
    let token = X64V3Token::from_context(); // Only needed if the original body uses it.
    let _ = token;
    let mut s = 0.0f32;
    for &x in data { s += x.into(); }
    s
}
```

Keep lifetimes, const parameters, HRTBs, associated type equality bounds, closure and fn-pointer arguments, `impl Trait` returns, `dyn Trait` data arguments and `unsafe fn` qualifiers. Attribute consolidation alone is not permission to change any of these. This rule covers the similarly named `tests/expand/arcane/generic_*`, `higher_ranked_trait_bound`, `associated_type_bound`, `closure_param`, `fn_ptr_param`, `dyn_trait_param`, `box_dyn_return` and `return_impl_trait` fixtures. Original exact signatures remain in the raw index; these tests should retain their arcane cases and eventually add parallel attune cases.

### A tier-bound generic token is a real API parameter

From [token_trait_bound.rs:3](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/token_trait_bound.rs#L3):

```rust
#[arcane]
fn process(token: impl HasX64V2, a: f32) -> f32 { a + 1.0 }
```

To a conservative design requirement:

```text
// API: Preserve process(token: impl HasX64V2, a: f32) -> f32 if callers depend on the bound.
// INLINE: Legacy wrapper remains inline(always) until a same-signature attune entry exists.
// BEHAVIOR: Features are those of HasX64V2, not every stronger concrete token passed to it.
Retain #[arcane] for compatibility; entry(token) design is unresolved.
```

An explicitly API-changing direct counterpart is:

```rust
// API: process_direct(a) is a NEW name/signature; keep process(token,a) for old callers if needed.
// INLINE: Tentative direct inline; no generic-bound or inlining claim about the old entry.
// BEHAVIOR: Require v2 context. Valid here only because this body does not use token-specific data.
#[attune(v2)]
fn process_direct(a: f32) -> f32 { a + 1.0 }
```

For `fn f<T: HasX64V2 + Other>(token: T, ...) where ...`, deleting `T` can lose trait methods, associated types and caller contracts. Keep T and bounds or retain legacy code; do not mechanically turn every token-generic function into a concrete v2 specialization. `T: F32x8Backend` alone is **not** a feature declaration and does not let a plain Rust helper call `X64V3Token::from_context()`.

### Inherent methods

From [plain_self.rs:5](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/autoversion/plain_self.rs#L5):

```rust
impl P {
    #[autoversion]
    fn apply(&self, x: f32) -> f32 { x * self.f }
}
```

To:

```rust
// API: Preserve P::apply(&self,x), visibility and return; central dispatcher only.
// INLINE: No forced dispatcher inline, matching this attribute-free old dispatcher.
// BEHAVIOR: Runtime dispatch retained; keep self/Self and exact old tiers/cfg policy.
impl P {
    #[attune(make(_))]
    fn apply(&self, x: f32) -> f32 { x * self.f }
}
```

For inherent arcane `fn process(&mut self, token: X64V3Token, ...)`, a direct `attune(v3)` would preserve the receiver and other parameters but remove only the proof. An explicit-token method entry places proof after the receiver, not before `self`. Whether make accepts methods and how helpers are scoped is still an implementation requirement.

Associated functions without a receiver need qualified calls (`Self::foo` / `Type::foo`); do not infer a method call from being inside an impl. Current `arcane(nested)` fixtures exercise this issue: [arcane_inherent_no_receiver.rs:29](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/arcane_inherent_no_receiver.rs#L29). Preserve impl-level type/const generics and `Self` return types. A nested item cannot capture outer generic parameters automatically.

### Trait implementations and object safety

From [nested_self.rs:5](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/nested_self.rs#L5), the `_self` pattern also appears in real trait tests [arcane_macro.rs:413](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/arcane_macro.rs#L413). The safe rule for trait contracts is:

```text
// API: Keep Trait::process(&self, token: X64V3Token, a: f32) -> f32 exactly as declared.
// INLINE: Keep the trait adapter's old policy; prospective helper direct inline is separate.
// BEHAVIOR: A plain trait method must cross a valid proof boundary; it cannot acquire a stronger
//           callable contract or new sibling trait members merely by applying a macro.
Old: #[arcane(_self = Processor)] on the trait impl method.
New: retain that adapter OR design an ordinary adapter calling an inherent/free _t helper.
Unresolved: exact attune placement and entry(token) syntax; no pretend-compiling trait attribute.
```

Do not emit `process_v3` siblings inside a trait impl unless the trait actually declares them. Do not attach target features directly to a trait method whose declared contract is callable from ordinary contexts. Do not remove the trait's token parameter, change `&self`/`&mut self`, introduce new generic methods, or require `Self: Sized` merely to make an expansion compile: these can alter trait compatibility or object safety. A previously object-safe trait must remain object-safe. The current `rite_trait_impl` and `autoversion_trait_impl` known-bug cases specifically catch violations, not successful syntax to emulate.

### Nested free functions and lexical context

Old ordinary inner `fn nested() { incant!(work(x), ...) }` is a separate function item, not a closure inheriting the enclosing feature context. The source rewriter explicitly skips nested functions: [rewrite.rs:43](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/rewrite.rs#L43), tests at [rewrite.rs:361](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/rewrite.rs#L361).

```rust
// FROM: an unannotated nested fn invoke(x: u32) -> u32 using incant!(work(x))
// API: Keep local invoke(x) signature and scope; no new exported items.
// INLINE: Preserve its existing attrs (none here); outer attune does not supply inline.
// BEHAVIOR: This is still cold dispatch, unless invoke itself gets an explicit feature context.
fn invoke(x: u32) -> u32 { attuned!(work(x)) }
```

A locally annotated nested `#[arcane] fn ...` follows P1 independently. Do not hoist it or make it public to simplify generation. `macro_rules!` templates in tests contain real attribute syntax indexed once in the template; expansion multiplicity is not a source occurrence count. Preserve hygiene and `$` substitutions; the compact index marks such lexical signatures instead of pretending to have expanded them.

## P13

### Cfg and tier modifiers

Old `incant!(inner(x), [-neon, -wasm128])` must not become unqualified all-default dispatch that adds ARM/WASM paths back. Old `[+v4]` makes v4 unconditional, while plain/default v4 in incant/magetypes is currently auto-gated. Preserve explicit `v4(cfg(avx512))`, custom cfg feature names, `-scalar`, `default` versus `scalar`, architecture gates and the caller-crate location where features are evaluated.

```text
// API: Keep inner's original selected entry names and visibility; do not add omitted tiers.
// INLINE: Apply direct/_t/dispatcher defaults only to outputs actually requested.
// BEHAVIOR: Preserve caller-crate cfg conditions and explicit no-fallback/error behavior.
Old: incant!(inner(x), [-neon, -wasm128])
New: attuned!(inner(x)), only when its available/selected family has precisely that restriction.
Otherwise retain old call until explicit call-tier grammar is settled.
```

The requested draft `tiers(cfg(...))` spelling is not settled here. No new syntax is invented for it. A runtime `summon` conditional does not make an omitted symbol resolvable; cfg names must align at declaration and call sites.

### Inline attributes and flags

| Existing form | Verified current treatment | Proposed treatment |
|---|---|---|
| arcane user `inline`, `inline(always)`, `inline(never)` | Filtered; wrapper always-inline, inner inline by default | Tentative per-output defaults; override precedence unresolved |
| arcane option `inline_always` | Requests always-inline inner, intended nightly use | No attune syntax specified; retain option tests |
| rite user inline attrs | Filtered; direct inline inserted | Tentative direct inline; explicit override policy unresolved |
| autoversion user inline attrs | Forwarded to dispatcher, not variants | Preserve meaning on central dispatcher if supported; do not erase silently |
| magetypes user attrs | Forwarded into variants; arcane/rite then filter inline on SIMD variants | Resolve per-output placement; fallback handling may differ |
| ordinary generic helper attrs | Ordinary Rust, no macro filtering | Keep unchanged |

Plain `#[inline(never)]` on a helper is not permission to replace it with always-inline. Some old macro forms already ignore that input, which is why the “from” side must be based on expansion, not surface spelling. None of this proves runtime speed.

### Cross-crate visibility and measured update

Current arcane sibling implementation functions are private even for public wrappers ([arcane.rs:512](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/arcane.rs#L512)). An exported macro cannot name a private function in another crate. That is a **name-access restriction**, not a claim that the optimizer cannot see a private body through a public inline function.

Parent-agent measurements, summarized in [the sibling experiment](../cross-crate-inline/README.md) in the eventual draft (not rerun by this read-only inventory): **252 cross-crate Rust 1.98.1 cases, LTO off**. With matching/superset feature callers, inline or always-inline token wrappers matched normalized direct caller assembly in **72 of 72 comparisons**. Direct bodies with inline inlined at **1 and 16 stages**, and remained calls at **128 stages**. Private-name access failed **E0603**, including through an exported macro. Public inline token wrappers could optimize their private body. **No runtime timings were measured.** The sibling report is not present in this isolated archive; the link targets its specified destination in the eventual draft. This lane received the results from the parent and did not independently inspect the measurement artifacts. Do not convert assembly comparisons into timing claims or universal guarantees across compilers/kernels.

This supports an open design alternative for a **token-entry-only family**:

```rust
// FROM: public foo_v3_t(token: X64V3Token, x: f32) -> f32 around a private body
// API: Same public _t entry and signature; no public foo_v3 direct item required.
// INLINE: Keep public _t inline/always-inline; measured matching contexts can optimize private body.
// BEHAVIOR: Proposed attuned static lowering derives proof without probe, then calls public _t.
foo_v3_t(X64V3Token::from_context(), x)
```

The line is a prospective **lowering**, not the syntax users must write or an implemented rule. `attuned!(foo(x))` could select it when a covered direct name is not public but `_t` is available. Design decisions remain: how callee outputs/visibility are known, whether this fallback is automatic or opted in, and how errors are reported if neither path is usable. A dispatcher-only family exposes neither a direct name nor a `_t` entry; do not promise a cross-crate static fast path for that family merely because its central dispatcher is callable.

A family may instead deliberately export direct names:

```rust
// FROM: public dispatcher-only foo(x); this is an intentional larger alternative API.
// API: Add public foo_<tier> and foo_<tier>_t alongside foo; helpers stay private.
// INLINE: Direct inline, _t always-inline, dispatcher no forced inline (all tentative).
// BEHAVIOR: Static callers can name covered direct entries; cold callers dispatch via foo.
#[attune(make(all))]
pub fn foo(x: f32) -> f32 { x * 2.0 }
```

This is optional, not required for all public families. Preserve `pub(crate)` and restricted visibility exactly; `doc(hidden)` is not privacy, and inline eligibility is not a grant of name access.

## P9

### Expected failures and controls

The compact failure index lists files, status and baseline links, including cases with no migration macro. Keep existing tests intact; add analogous attune rejection tests when syntax exists. No failures were executed in this inventory.

| Existing case | Intended status / migration disposition |
|---|---|
| `compile_fail/wrong_token` | Reject incompatible proof; keep legacy signature mismatch test |
| `unknown_trait_bound`, `unknown_generic_bound`, `featureless_simdtoken` | Reject unsupported/featureless feature selection; generic backend bounds are not enough |
| `missing_scalar` | Missing fallback function remains an error under old dispatch contract |
| `scalar_not_in_tier_list` | **Dormant**: invocation commented out while `REQUIRE_EXPLICIT_SCALAR` is false; not an active passing rejection test |
| `scalar_default_mutual_exclusion` | Reject both in one incant selection; do not generalize this to multi-rite generation |
| `autoversion_concrete_token` | Old macro rejects concrete token; preserve rejection test, add separate positive attune single-tier test |
| `token_shadowing`, `token_aliasing` | Keep identity/assertion protections; do not “migrate” by removing the adversarial condition |
| `ui/unsafe_*` | Pointer intrinsics still require unsafe; a new attribute must not relax that rule |
| `soundness/from_context_*` | Missing/weaker/wrong-arch context and fn-pointer conversion reject as asserted by harness |
| `soundness/raw_missing_context`, `raw_weaker_context`, `raw_fn_pointer` | Keep direct-feature safety failures |
| `soundness/raw_matching_context` | Positive control; includes matching and superset feature contexts |
| `soundness/tokenless_uncovered` | No covered callee/fallback: error, not runtime upgrade |
| `soundness/tokenless_misdeclared_callee` | Declared suffix must not bypass actual rustc feature checks |
| `soundness/*shadowing*`, `trait_aliasing_exploit`, `sealed_trait_bypass` | Preserve proof identity, absolute trait-bound and sealing tests |
| `expand/should-fail/rite_trait_impl` | Expanded output expected E0053: incompatible trait method contract |
| `expand/should-fail/autoversion_trait_impl` | Expanded output expected E0407: extra trait methods |
| `expand/should-fail/incant_passthrough` | Expanded output expected E0658 for panic internals; do not label all passthrough source unsupported |
| `avx512-cfg-tests/v4-import-intrinsics-no-feature` | Missing feature/import configuration fixture; retain intentional failure |

Harness evidence: [compile_fail.rs:24](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail.rs#L24), [soundness_exploits.rs:33](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/soundness_exploits.rs#L33). Dynamically built cases in the latter include `suppressed_intrinsic` (E0133), `suppressed_alias` (compile-only success, not safe to execute), baseline intrinsic/SSE2 negative/positive controls, and storage layout/token rejection cases. Embedded macro text is counted as literal text in the raw inventory, not a second source invocation; these test cases are still explicitly covered here.

Legacy suites to retain include `arcane_macro`, `arcane_trait_recognition`, `arcane_inherent_no_receiver`, `autoversion_macro`, `calling_convention_matrix`, `tokenless_context`, alias/deprecated expansion fixtures, tier-modifier/cfg/downstream fixture crates, proof-placement and scalar/default expansions, inline behavior snapshots, all rejection fixtures and positive controls. New examples may use attune once implemented; migration must not erase coverage of supported old APIs. Tests pinning old signatures, upgrade behavior, exact suffix selection, deprecation, or macro options cannot be wholesale renamed and still test the original contract.

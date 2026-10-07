# Attune specification — draft

Status: design only, 2026-10-07. Nothing in this document adds an implemented
API. Keep this specification and its experiments on the draft bookmark; do not
merge them into release branches as implemented functionality.

This is the current design entry point. The earlier
[migration inventory](experiments/attune-migration/README.md) records the existing
API and provisional alternatives. Where this document recommends a newer
spelling, such as `reattune!`, the older sketches remain historical alternatives.

## 1. Status vocabulary and objectives

- **Established direction:** requirements already expressed in the design
  discussion. They still require implementation and tests.
- **Recommendation:** a concrete proposal in this document, not an accepted or
  implemented API merely because it is written here.
- **Spike required:** a correctness or integration question needing an executable
  prototype before we promise the corresponding contract.

Objectives:

1. One function attribute, `#[attune]`, for explicit feature contexts and
   requested families of direct functions, proof entries and central dispatchers.
2. Ordinary source bodies and calls do not need token plumbing just to establish
   a concrete feature context. Tokens remain useful values and generic parameters.
3. `attuned!` composes covered feature contexts without implicit upgrades.
4. Explicit runtime reselection has an obvious spelling. Recommendation:
   `reattune!`, replacing the previously sketched `dispatch` call modifier.
5. Preserve Rust signatures, visibility, generics, trait contracts and attribute
   meaning unless an API change is explicit.
6. Preserve direct-call optimization opportunities; do not promise that every
   body will inline or every route has identical performance.
7. Support migration alongside existing macros. Consolidation does not require
   deleting legacy compatibility or rejection tests.

## 2. Function naming and requested outputs

Established direction, for source `fn process(...)`:

| Attribute/request | Requested Rust items |
|---|---|
| `#[attune(v3)]` | One direct `process(...)`, preserving the name |
| `#[attune] fn process_v3(...)` | Infer V3 from the terminal suffix; preserve `process_v3` |
| `make(_v3)` | Direct `process_v3(...)` |
| `make(_v3_t)` | Explicit-proof `process_v3_t(token, ...)` |
| `make(_)` | Central dispatcher `process(...)` |
| `make(_*)` | Direct variants for the documented default tier set |
| `make(_*_t)` | Explicit-proof variants for that default tier set |
| `make(all)` | Exact shorthand for `make(_*, _*_t, _)` |

Selectors append to the source name; they do not secretly strip an existing tier
suffix. A source already named `process_v3` normally wants bare `#[attune]`, not
another `_v3` selector. Bare inference must use the longest registered terminal
suffix; numeric ordering is not a substitute for registry lookup. A token-entry
suffix such as `_v3_t` must not be misinterpreted as a direct-function suffix.
Whether bare inference also recognizes explicit-proof entries is undecided.

Only requested interface items inherit the source visibility. Necessary
implementation helpers are private. Generated direct bodies should be shared
between requested entries instead of duplicating the operation per entry form.
Internal helpers are generated even when not requested as public interface items.

For example (proposed attribute):

```rust,ignore
#[attune(make(_v3, _v3_t, _))]
pub(crate) fn process(data: &[f32]) -> f32 { /* operation */ }

// API: Requested names all have pub(crate) visibility.
// INLINE: Direct inline; thin proof entry always-inline; dispatcher unforced.
// BEHAVIOR: Direct requires V3 context; _t requires proof; dispatcher detects.
// Resulting interfaces, with bodies omitted:
// pub(crate) fn process_v3(data: &[f32]) -> f32;
// pub(crate) fn process_v3_t(token: X64V3Token, data: &[f32]) -> f32;
// pub(crate) fn process(data: &[f32]) -> f32;
```

The dispatcher also needs a portable fallback implementation. Its presence is
an implementation requirement, not an implicit request for a public scalar name.
Tier-dependent signatures may make a direct family valid but its dispatcher
invalid; see section 7.

`*` means one documented default tier policy, not every registry entry. The exact
policy and optional AVX-512 gates still need agreement: current autoversion and
incant/magetypes defaults differ in their gating. Explicit tier requests provide
a migration path that does not depend on resolving that default immediately.

## 3. Visibility, cfg and output attributes

Established direction: preserve `pub`, `pub(crate)`, restricted visibility or
implicit visibility on each requested output. Containing-module privacy and
re-export rules continue to apply. `doc(hidden)` does not make an item private.

Recommendation: use normal outer `#[cfg(...)]` to gate the whole family and
permit standard cfg attributes on individual make selectors:

```rust,ignore
#[attune(make(
    _v3,
    #[cfg(feature = "avx512")] _v4,
    _
))]
pub fn process(data: &[f32]) -> f32 { /* operation */ }
```

This selector grammar is proposed, not accepted syntax. Reusing the existing
`_v4(cfg(avx512))` style is an alternative with the same feature-gating purpose.
Do not implement both merely to avoid choosing a spelling.

Implementation requirements are clear regardless of spelling:

- Add architecture guards from the registry.
- Gate references, dispatcher arms and helpers consistently with the selected
  output. A reference must never survive after its target definition is omitted.
- A shared implementation used by multiple outputs exists whenever any of those
  outputs needs it; one output's gate must not accidentally remove another's body.
- Keep a portable dispatcher fallback when SIMD selections are cfg-disabled.
- Preserve provider-side Cargo feature decisions across crate boundaries. A
  consumer's identically named Cargo feature is not the provider's feature.
- Known cfg-omitted variants are unavailable selections; a missing declaration
  or inaccessible required entry is a diagnostic, not permission to silently
  invent a different family.

Mixed output visibility is optional additional grammar. If needed, prefer a
Rust-shaped selector override such as `pub(crate) _v3`, inheriting visibility
when absent. An explicit private override needs an unambiguous spelling before
this extension can be complete. The initial inherited-visibility rule needs
none of these overrides.

### Attribute routing recommendation

| Attribute | Intended destination / constraint |
|---|---|
| `cfg`, relevant `cfg_attr` | Family/output conditions, including derived references and metadata |
| Documentation, deprecation, `must_use` | Requested user-facing items where their meaning applies |
| Inline on a direct-only function | Its direct body; preserve an explicit override |
| Inline on a generated family | Explicit output-routing rule required; do not guess from placement |
| `track_caller` | Preserve the visible caller through generated adapters; test panic locations |
| `allow`, `warn`, `deny`, `forbid` | Relevant generated scopes without weakening the user's policy |
| `expect` | Deliberate placement; copying it can create unfulfilled expectations |
| `cold` | Define whether the operation or a particular entry is cold |
| Unknown procedural attribute | Define ordering or reject ambiguous multi-output use; do not duplicate blindly |
| `test`, linkage/export attributes | Output-specific handling; do not duplicate tests or exported symbols |

Current autoversion puts user attributes only on the dispatcher; arcane/rite
filter user inline attributes. A new routing rule can change behavior even when
source spelling is retained. The migration guide must describe that change.

## 4. Calls: attuned and reattune

Established direction for ordinary `attuned!(process(args))`:

| Caller context | Selection |
|---|---|
| Known attune feature context | Choose an eligible covered tier; no detection or implicit upgrade |
| Ordinary code | Runtime selection through safe entries or a central dispatcher |
| Unannotated nested function item | Its own ordinary context; does not inherit the surrounding function's context |
| Closure in a feature context | Preserve the enclosing context where Rust's feature rules permit it |

Coverage is based on the actual feature relation, including crypto branches,
not the numeric value in a tier suffix. For eligible covered variants, the
family's documented selection order determines the choice.

### Recommendation: reattune!

`reattune!` explicitly asks to reconsider the selected tier, including runtime
upgrades. It has the same call-expression shape as `attuned!`:

```rust,ignore
#[attune]
fn outer_v3(data: &[f32]) -> f32 {
    // API: Caller signature unchanged; callee needs an accessible V4 entry.
    // INLINE: V4 remains a feature boundary from V3.
    // BEHAVIOR: Try V4; otherwise use the already-proved V3 implementation.
    reattune!(process(data), [_v4, _v3])
}
```

The bracketed shortlist grammar is a recommendation, chosen to resemble current
incant tier lists. Exact modifier and cfg spellings remain to be finalized.

```rust,ignore
// Illustrative lowering when these are the available named entries:
if let Some(token) = X64V4Token::summon() {
    process_v4_t(token, data)
} else {
    process_v3(data)
}
```

Behavioral requirements:

1. Without a shortlist, use the family's documented runtime tier order.
2. With a shortlist, restrict eligibility to that list, respecting registry
   priority rather than inventing a different priority from token values.
3. When selecting through per-tier entries, use known caller feature coverage
   as proof and only detect what is not proved. Stop once a selected guaranteed
   candidate is reached. Delegating to an opaque central dispatcher follows
   that dispatcher's contract and may recheck already-covered tiers.
4. Enter a stronger/uncovered implementation through an accessible safe proof
   entry or a suitable central dispatcher. Never create an unchecked raw call
   from the suffix alone. A dispatcher must honor an explicit shortlist if used.
5. Infallible selection must have a guaranteed eligible candidate: a covered
   implementation or a portable fallback. V4 alone from V3 supplies neither.
6. An availability-returning exact-tier operation would be a separate API
   decision. Do not silently panic, invent a fallback outside an explicit list,
   or change the return type to Option.
7. Evaluate user arguments and receiver exactly once on the selected path;
   preserve move/borrow behavior and the selected call's argument order.
8. Respect disabled variants and incompatible architectures at compile time.

`reattune!` is not “ignore tokens.” Generated code still obtains proof when it
crosses a feature boundary. The user chooses whether to reconsider the tier,
not whether to pass a hidden proof argument.

Outside an attune context, ordinary attuned already performs runtime selection;
reattune expresses that intent explicitly and permits a shortlist. Inside one,
reattune is the opt-in route for behavior currently provided by tokenful incant
rewriting that probes stronger or unrelated feature branches.

### Accessible entry selection

Recommendation for a covered call:

1. Use an accessible direct entry if the family exposes one.
2. Otherwise, if an accessible `_t` entry exists, derive the covered token with
   `from_context()` and use that entry without detection.
3. If only a dispatcher is accessible, report that static entry is unavailable;
   explicitly calling the dispatcher or reattune retains runtime semantics.

A public inline proof entry can optimize a private implementation body; this
does not allow a source macro to name that private body. The existing assembly
experiment supports this route for its measured cases, not all future kernels.

**Spike required:** how a call discovers family outputs, gates and visibility
across modules/crates, including renamed dependencies, re-exports, aliases and
associated functions. Proc macros must not depend on expansion order, a mutable
global registry, scanning source files, or probing whether arbitrary identifiers
exist. A provider-generated family descriptor is a candidate; its exact export,
namespace, generic and versioning contract must be demonstrated.

## 5. self, Self and placement

Established direction: users write ordinary `self` and `Self` in new bodies.

Current nested arcane converts the receiver into an `_self` parameter and
substitutes capital `Self`; it does not rewrite lowercase self uses in the body.
Existing callers therefore write `_self` explicitly. See
[arcane.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/arcane.rs#L639).

| Placement | Generation approach | Confidence |
|---|---|---|
| Free function | Sibling direct implementation / entries | Clear approach |
| Inherent method with receiver | Sibling methods; retain self and Self | Clear approach |
| Inherent associated function without receiver | Siblings with qualified `Self::...` calls | Clear once context is supplied |
| Trait impl method | Preserve original trait member; place implementation in a legal helper scope | General case needs spike |
| Trait default method | Preserve trait contract and dyn compatibility; helper needs trait-aware generic treatment | Separate spike |
| Nested function | Own feature context; preserve scope and copied generics if helpers are introduced | Scope/hygiene tests required |

Recommendation: an `in_impl` option on a function means **inherent impl context**
and is useful for associated functions without a receiver. Keep it part of
attune rather than inventing another exported attribute macro.

An `in_trait` hint could forbid illegal sibling trait members and select an
ordinary adapter, but it does not supply the self type or enclosing generics.
It must also distinguish a trait declaration/default from a trait implementation.
Do not promise fully general trait support from that flag alone.

Recommendation: allow outer `#[attune]` on an impl to give method expansion its
concrete self type, trait path, generics and where clauses. The outer annotation
provides context to explicitly selected members; it does not automatically
multiversion every method. How this composes with other impl-level macros must
be tested. Trait declarations with default bodies need their own scope analysis.

Nested receiver rewriting must respect expression scopes, nested items, macro
input hygiene, associated types, qualified paths and return-position Self.
Do not perform a blind textual `self`/`Self` substitution. Inner functions do
not capture outer generic parameters; declarations and bounds must be supplied.
Adding `Clone`, `Copy`, `'static` or `Self: Sized` merely to make generated code
compile is not an acceptable shortcut.

Keep legacy `_self` behavior behind its explicit compatibility path. An ordinary
user variable named `_self` must not become reserved globally.

## 6. Proofs, old signatures and token passthrough

Tokens are still ordinary useful values. A concrete feature context can derive
a token with `from_context()` without feature detection. A generic backend
bound by itself does not establish a Rust target-feature context.

There is a clear implementation approach for keeping a published token-taking
signature: retain a safe adapter with exactly that name, visibility, generics,
arguments and return type, and forward into a feature-enabled implementation.
Existing arcane supplies the underlying boundary mechanism.

**Grammar decision:** how the unified attribute requests that adapter while
preserving its name. A proposed `entry(token)` option is not settled. `_t`
generation alone does not preserve an old unsuffixed token entry's name.
Existing arcane can remain the migration bridge while this is decided.

For generic proof parameters such as `T: HasX64V2 + Other`, retain T and its
associated-type/bound contracts when the operation uses them. The feature
guarantee is the declared V2 requirement, not all features of every possible
concrete T. If an existing parameter is nominated as boundary proof, verify its
supported sealed bounds rather than adding a redundant proof parameter or
silently deleting T. Bound collection must cover both inline and where-clause
declarations; the existing inference bugs are tracked in
[issue 122](https://github.com/imazen/archmage/issues/122).

Ordinary `_t` generated entries put proof first among non-receiver parameters.
Compatibility adapters must preserve existing first/middle/last placements.
Receivers remain receivers; proof never precedes `self` in a method signature.

Existing `incant!(... with token)` performs exact-token passthrough. Neither
attuned's feature-context selection nor reattune's runtime reselection is a
drop-in substitute. Its implementation strategy is understood, but whether the
unified API needs a new explicit-proof family operation is a product decision.
Retain the old operation until that choice is made; concrete cases can call
the relevant `_t` entry directly.

## 7. Generics, vectors and dispatcher signatures

Clear approach for ordinary generics: clone the signature with all lifetimes,
type/const parameters, bounds and where clauses; forward the correct generic
arguments; preserve receivers and qualifiers. Do not add type erasure, boxing
or new trait bounds. Test impl-level generics as well as method-level generics.

Current magetypes-style per-tier `Token` substitution and fixed-width
`define(f32x8)` aliases have existing machinery to reuse. Separate that macro
placeholder from a user's real generic parameter T. The unified attribute's
spelling for these facilities remains a grammar decision. Scalar specializations
need a concrete portable backend when the body uses vectors; do not equate
“tokenless default” with “no backend type exists.”

Magetypes `_t` methods remain token-first migration APIs. Adaptive-width aliases,
automatic tail algorithms and tokenless vector constructor releases are separate
designs; generating an attune function must not silently change vector widths,
layouts, slice partitioning, or numerical semantics.

| Signature shape | Direct family | Central dispatcher |
|---|---|---|
| Common ordinary arguments/return | Supported design | Supported design |
| Ordinary type/const/lifetime generics | Preserve all contracts | Supported if every selected arm type-checks under those contracts |
| Tier-specific vector in exposed argument/return | May be valid per variant | Reject unless an explicit common interface exists |
| Distinct opaque return types from separate variants | May be valid individually | Do not implicitly unify or box them; reject incompatible arms |
| Trait method with existing token input | Adapter can preserve it | Must preserve the trait's declared signature |

No new generic trait abstraction is required just to implement naming and
dispatch. A future family-descriptor implementation must justify any traits
it introduces and measure their compilation impact.

## 8. Inlining and codegen contract

Recommended defaults:

| Generated role | Default |
|---|---|
| Direct implementation | `#[inline]` |
| Thin ordinary proof entry | `#[inline(always)]` |
| Runtime dispatcher | No forced inline |

An explicit never-inline operation can still use an always-inline thin adapter;
the adapter may disappear while the operation remains a call. Family-level
attribute syntax must make that routing unambiguous. Never silently discard an
explicit inline policy as existing rite currently does.

Upgrading from V3 to V4 retains a feature boundary. Always-inline on the ordinary
wrapper cannot erase Rust's incompatible-feature constraint. A needed call is
not itself a failure of the design; dispatch placement should amortize it over
the operation rather than repeating it for every tiny arithmetic step.

Measured evidence, Rust 1.98.1, LTO off:

- 252 cross-crate caller cases exercised small/large callers and three callee
  sizes with direct and proof-entry paths.
- 72/72 matching/superset direct-versus-inline-wrapper comparisons had identical
  normalized caller assembly.
- Inline direct bodies at 1 and 16 stages inlined; 128-stage bodies remained calls.
- A tiny unannotated ordinary function inlined across crates; the unannotated
  target-feature kernels in that matrix did not.
- Private-name access failed even through an exported macro.

See [methodology and results](experiments/cross-crate-inline/README.md).
These are assembly observations, not runtime speed or cold-compile measurements
of attune. The experiment run reported peak RSS 0.30 GiB; its full resource line
and compiler/dependency provenance are recorded with those results.

## 9. Invalid and unsupported combinations

Recommended initial diagnostics:

| Request | Disposition |
|---|---|
| Bare attune without an unambiguous registered direct suffix | Require an explicit tier/request |
| Conflicting explicit tier and inferred suffix | Diagnose the conflict |
| Conflicting duplicate output requests/gates | Diagnose rather than silently pick one |
| Identical repeated selectors, including wildcard overlap | Normalize to one output |
| New undeclared sibling methods in a trait impl | Reject; use a legal adapter/helper placement |
| Stronger/uncovered call through ordinary attuned | Diagnose; suggest reattune and a safe entry |
| Infallible reattune without a guaranteed candidate | Require a fallback or an explicit availability-handling API |
| Upgrade when only a raw inaccessible/unsupported entry exists | Require a suitable safe boundary; no unchecked suffix-based call |
| Dispatcher with incompatible tier-specific signatures | Reject; retain direct-only family or require explicit conversion |
| Loss of trait compatibility/dyn compatibility to make helpers compile | Reject the expansion design |
| Special ABI, linkage, async or const combinations without defined semantics | Diagnose unsupported combination until designed and tested |

The last category is not a claim that Rust can never support those combinations.
Do not implement “best effort” emission that changes an API's meaning silently.

## 10. What is solved, chosen, or still needs a spike?

| Remaining area | Assessment | Next concrete work |
|---|---|---|
| Suffixes and default visibility | Clear rules | Implement output model and expansion snapshots |
| Cfg-disabled SIMD with scalar fallback | Clear approach | Settle selector cfg grammar; test gates/references together |
| Explicit upgrade/reselection | Clear approach | Use reattune recommendation; reuse feature-proof dispatch machinery |
| Old names and proof parameter placement | Clear adapter approach | Choose compatibility entry spelling |
| Ordinary generics and fixed-width aliases | Existing machinery | Preserve contracts; finalize placeholder/alias spelling |
| Inherent associated functions without self | Clear with context hint | Implement qualified forwarding; test impl generics |
| Intrinsic/type imports | Existing per-tier machinery | Specify flags and propagation, including scalar/default cases |
| Exact-token passthrough | Known semantics, API choice open | Retain legacy operation until explicit-proof surface is chosen |
| Attribute routing | Mechanism clear, some policy choices open | Decide family-level routing; test expect/track_caller/unknown attrs |
| Full generic trait adapters/default bodies | Spike required | Compile nested/helper placements with associated types and dyn use |
| Scope-correct receiver rewriting | Spike required for generality | Exercise nested items, macros, closures and hygiene |
| Cross-crate family discovery | Spike required | Demonstrate descriptor/re-export/renamed-dependency behavior |
| Cold compile cost and resulting codegen | Measurement required | Compare equivalent expanded output after the prototype exists |

“Clear approach” does not mean implemented or already covered by attune tests.
The inventory accounts for 1,009 existing attributes and 153 dispatch calls;
these are lexical occurrences, including fixtures and templates, not 1,162
passing tests of the new attribute.

## 11. Acceptance gates

Before implementing the public interface, settle the listed grammar/policy
decisions and demonstrate the trait and family-discovery spikes off main.
New public API proceeds through signature review; preserving this draft is not
authorization to merge experimental APIs into main.

Required coverage for a later implementation:

- Existing macro suites retained; analogous attune positive and negative cases
  added without weakening existing expectations.
- Free/inherent/trait contexts; receivers and receiverless associated functions;
  generic impls, method generics, consts, lifetimes and associated types.
- Private, restricted and public items; downstream crates, re-exports, dependency
  renaming, provider/consumer feature combinations and inaccessible entries.
- Covered calls, explicit upgrades, cross-branch features, portable fallback,
  no-fallback rejection and no hidden detection on the covered default path.
- Rust feature checks still reject misdeclared callees; token identity/sealed
  proof checks survive adapters. Generated unsafe boundaries remain small and
  auditable; users retain forbid(unsafe_code).
- Moves, borrows, side-effecting arguments, receivers and closures retain their
  contracts; unannotated nested functions do not inherit feature context.
- cfg and attribute routing; track_caller locations and expect lint behavior;
  snapshot and compile checks for expansion, not snapshots alone.
- Numerical behavior compared where implementations are intended to agree;
  no assumption that changing selected tiers is always numerically invisible.
- Cross-crate assembly checks above/below observed inline heuristics, followed
  by cold compile comparisons on equivalent tier sets and output surfaces.

## 12. References

- [Pinned migration inventory](experiments/attune-migration/report.md)
- [Annotated migration patterns](experiments/attune-migration/migration.md)
- [Trait, generic, visibility and inline contracts](experiments/attune-migration/migration-contracts.md)
- [Assembly experiment](experiments/cross-crate-inline/README.md)
- [Recorded validation scope](experiments/cross-crate-inline/validation.md)
- [Rust attribute macro input](https://doc.rust-lang.org/reference/procedural-macros.html#the-proc_macro_attribute-attribute)
- [Rust target-feature rules](https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute)
- [Nested item generic scope](https://doc.rust-lang.org/error_codes/E0401.html)

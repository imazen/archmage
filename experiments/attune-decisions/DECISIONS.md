# Attune decision worksheet

All choices below are pending. Recommendations are not recorded user decisions.
No proposed attune syntax has been implemented. The
[evidence report](README.md) distinguishes executed lowerings from untested
macro integration. The [specification](../../ATTUNE_SPEC.md) remains a draft.

## First discussion: behavior and necessary context

### Q1. Is an explicit reattune tier list exact?

Example inside a V3 context:

```rust,ignore
reattune!(work(x), [_v4, _v3]) // Try V4, otherwise use proved V3.
reattune!(work(x), [_v4])      // What happens when V4 is unavailable?
```

**A — Exact list, recommended.** The second infallible call is rejected unless
V4 is already proved. Write the fallback explicitly. An availability-returning
API, if desired, is a separate decision; the macro does not silently change
the return type or panic.

**B — Append a fallback automatically.** The second call may run V3 or scalar.
Shorter common calls, but the displayed list no longer fully describes allowed
implementations, and a caller may receive results from a tier it did not name.

Evidence: both upgrade and fallback lowerings execute correctly with one move
and drop, and a covered call does not need a probe. Which implementations are
permitted is a policy choice. Tested lowerings provide no reason to hide it.
The no-list spelling can still use the family's documented defaults.

### Q2. What does the wildcard do with V4?

```rust,ignore
#[attune(make(_*))]
fn work(...) { ... }
```

**A — Gate V4 on the provider's avx512 feature, recommended for continuity
with magetypes.** Retain the documented V3/NEON/WASM/scalar defaults and optional
V4. Existing scalar autoversion users may need to enable a feature to retain
their previous V4 generation. The declaring crate must own/forward this feature.

**B — Always include V4.** Matches current scalar autoversion behavior. Every
body/backend requested by the wildcard must support V4; vector/backend feature
requirements still have to be satisfied. Runtime CPU detection provides hardware
safety, not missing compile-time backend support.

**C — Require explicit V4 selection.** Removes an implicit Cargo-feature policy
from the wildcard; V4-capable users write an extra selector and gate. It changes
the wildcard's relationship to both existing default sets.

Evidence: current autoversion and magetypes generate different V4 surfaces with
the feature disabled. Both references compile when enabled. No performance
ranking or compile-time advantage between these policies was measured.

### Q3. Can generic trait methods receive context from an outer impl attribute?

```rust,ignore
#[attune] // Supplies context; does not transform every method automatically.
impl<T: Copy> Kernel for Processor<T> {
    #[attune(/* chosen operation */)]
    fn apply(&self, data: &[f32]) -> Self::Output { ... }
}
```

**A — Outer annotation when context is needed, recommended.** It supplies the
self type, trait path, impl generics and bounds. Ordinary free functions still
need only their function attribute. A receiverless inherent function can also
use the lightweight in_impl hint.

**B — Function attributes only.** The method must explicitly supply missing
context. An in_trait boolean alone is insufficient; generic declarations,
bounds and self type may need repetition, or the supported shapes are narrower.

Evidence: candidate lowerings preserve mutable receivers, type/const/lifetime
generics, associated output and dyn usage. Implicit nested capture fails E0401;
receiverless bare sibling calls fail E0425. A free-helper strategy also handles
legal foreign-self trait impls where an inherent helper impl fails E0116.

This choice authorizes a context source, not a claim that arbitrary trait-body
rewriting is already solved. The transformer still needs scope and hygiene tests.

## Next discussion: call surface and compatibility

### Q4. What automatic discovery do we promise for sparse families?

Suppose an external family has only V3 and scalar implementations, while its
caller is V4:

```rust,ignore
attuned!(dependency::work(x))          // Must this automatically find V3?
attuned!(dependency::work(x), [_v3])   // Or must sparse selection be explicit?
```

**A — Automatic discovery for declared families.** Simple call sites, with a
provider-generated description of available entries and cfg. Independently
handwritten variants may need an explicit family declaration. The descriptor
becomes an exported protocol for public families and must survive versioning,
renaming, re-exports and associated-function placement.

**B — Conventions plus explicit lists for sparse families.** Less generated
discovery infrastructure. Callers must know/select available tiers, and names,
visibility and gates must agree. No compiler reflection is implied by a missing
function name. This resembles current incant's explicit-tier requirements.

Recommendation: decide the behavioral promise first. Do not choose an internal
macro or enum protocol yet. Both work for the tested free-function namespace
cases, but both have demonstrated limitations: root macro-name collisions,
type-name collisions, and no corresponding declaration position inside an impl.
Neither prototype groups independently annotated variants or implements the
complete versioned protocol. Cold compile cost is not measured.

Trait-object calls are a separate contract: preserving an existing trait method
does not create new statically callable tier members in that trait. Automatic
discovery must not quietly change a trait's vtable or dyn compatibility.

### Q5. How should an existing token-taking public signature migrate?

Old API:

```rust,ignore
#[arcane]
pub fn work(token: X64V3Token, data: &[f32]) -> f32 { ... }
```

**A — Keep legacy attributes as the compatibility bridge, recommended initially.**
New names follow the direct/_t conventions. Existing work(token, data) remains
available and can share the same implementation machinery without inventing
another new syntax immediately.

**B — Add a same-name proof-entry mode to attune.** One attribute can express
the existing signature directly, including generic proof parameters and their
additional contracts. A spelling such as entry(token) must be chosen and its
relationship to make outputs specified.

Both can use a safe ordinary wrapper around a feature-enabled body. Merely
generating work_v3_t changes the old name. Existing first/middle/last proof
positions and generic associated types must not disappear during conversion.
This is an interface choice, not an unsolved safety boundary.

### Q6. What happens to exact-token passthrough?

```rust,ignore
incant!(work(x) with token, [v3, scalar]) // Existing exact-token operation.
```

**A — Preserve it through the legacy API initially, recommended.** Concrete
new callers can use work_v3_t(token, x). No new with-token/without-token mode is
added to normal composition.

**B — Design an explicit-proof family operation now.** Covers generic token
passthrough under the unified surface, at the cost of another explicit contract
and spelling. It must preserve exact-type handling and unmatched-token behavior.

reattune is not a replacement: it may detect a tier unrelated to the supplied
proof. The current exact-token implementation is source evidence; this probe
set did not add a new passthrough implementation.

## Final discussion: output spelling and attribute ownership

### Q7. Which selector cfg spelling should we support?

```rust,ignore
// A: standard attribute-shaped selector decoration
#[attune(make(_v3, #[cfg(feature = "avx512")] _v4, _))]

// B: existing archmage-style feature shorthand
#[attune(make(_v3, _v4(cfg(avx512)), _))]
```

**A — Standard cfg syntax, recommended.** Familiar Rust predicates, including
all/any/not, and a possible common syntax for other per-output attributes.
Longer than the shorthand.

**B — Existing shorthand.** Shorter and familiar to current archmage users.
The current parser is feature-name-oriented; full predicate support would need
a defined extension. Supporting both increases documentation and parser cases.

Either can normalize to the same output model. No parser-cost difference was
measured. The tested semantic requirement is stronger than the spelling choice:
evaluate provider availability in the provider, and preserve an ungated portable
dispatcher fallback when only SIMD outputs are disabled.

### Q8. Do we need mixed visibility in one make request now?

Same visibility is already the established default. An optional extension:

```rust,ignore
#[attune(make(pub(crate) _v3, pub _))]
fn work(...) { ... }
```

**A — Add Rust-shaped per-selector visibility.** One declaration can expose a
public dispatcher and crate-only direct variants. Starting with a private source
function lets selected outputs be widened without needing a private keyword.

**B — Inherited visibility only initially.** Smaller grammar. Mixed surfaces
use a separate wrapper/declaration or wait for the extension. This does not
prevent dispatcher-only make(_) from keeping implementation helpers private.

This is a grammar/product choice. Name privacy still applies across crates;
doc-hidden metadata is not private. The earlier assembly/privacy experiment
already establishes that inlining does not grant source-level access.

### Q9. Where does a written inline override apply?

```rust,ignore
#[inline(never)]
#[attune(make(all))]
pub fn work(...) { ... }
```

**A — Interpret operation-level inline as a body policy.** The body remains a
call while a thin proof wrapper can inline. Document this difference from the
existing autoversion attribute, where user attributes attach to the dispatcher.

**B — Require explicit output placement for multi-output requests, recommended.**
Keep the recommended role defaults, but make overrides unambiguous:

```rust,ignore
#[attune(make(#[inline(never)] _*, _*_t, #[inline(never)] _))]
pub fn work(...) { ... }
```

Single-output direct attributes remain ordinary. The cost is more syntax for
family-level overrides; the benefit is no ambiguous inference of which generated
function a user meant. Wildcard/explicit-selector override precedence needs a
rule before supporting conflicting requests.

Evidence: the prior assembly matrix shows inline/always wrappers can disappear
while a never-inline body remains a call. Copying other attributes indiscriminately
is already demonstrably wrong: duplicated expect fails, and track_caller on only
the wrapper loses the external location. Those have clear routing requirements,
not useful choices to present as arbitrary preferences.

### Q10. Do fixed-width aliases keep their current define spelling?

```rust,ignore
#[attune(/* outputs */, define(f32x8))]
fn work(...) { /* per-tier Token and f32x8 alias */ }
```

**A — Retain define(...) for fixed-width aliases, recommended for migration.**
Reuse existing substitution and preserve real user generics separately.

**B — Introduce a new unified spelling.** Potentially more consistent with a
future vector-context API, but adds migration churn. Adaptive-width aliases and
tail policies should remain a separate decision; neither is implied by this probe.

This is a spelling choice with existing implementation machinery. No new alias
implementation or compile-cost comparison was made here.

## Requirements that do not need a preference vote

- Preserve Rust signature contracts and trait compatibility unless a requested
  API change explicitly says otherwise.
- Do not forge feature proof, suppress feature errors, add implicit boxing,
  invent Clone/Copy/Sized bounds, or emit unchecked suffix-based upgrades.
- Respect nested item scopes when rewriting receivers; copied generics require
  explicit declarations. Use legal free helpers when inherent helpers are illegal.
- Preserve argument ownership and single evaluation on a selected call path.
- Propagate track_caller across forwarding layers; place expect where it is
  fulfilled rather than copying it to every generated function.
- Report unavailable/inaccessible required entries clearly. Keep existing
  compatibility tests and negative tests intact.
- Keep experiments off main and measure compilation after the actual generator
  and discovery protocol exist; namespace probes cannot establish those costs.

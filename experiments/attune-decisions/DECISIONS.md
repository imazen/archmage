# Attune decision worksheet

Q1, feature-gated V4 in Q2, the function-only constraint in Q3, and explicit
sparse-family lists in Q4 are accepted (2026-10-07). Idempotent `-_v4` exclusion and no end-user compile-time regression
are also accepted requirements. Other choices remain pending; recommendations
are not recorded user decisions.
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

**Accepted — Exact list, applying to both call macros.** The second infallible call is rejected unless
V4 is already proved. Write the fallback explicitly. An availability-returning
API, if desired, is a separate decision; the macro does not silently change
the return type or panic.

After cfg/architecture filtering and exclusions, at least one available eligible
callee must have requirements covered by the caller's proved feature context.
Otherwise either call macro emits a compile error. A portable scalar candidate
qualifies; the chance that runtime detection will succeed does not. No implicit
fallback, panic, or Option return. This holds for calls outside attune too.

Evidence: both upgrade and fallback lowerings execute correctly with one move
and drop, and a covered call does not need a probe. Which implementations are
permitted is a policy choice. Tested lowerings provide no reason to hide it.
The no-list spelling can still use the family's documented defaults.

### Q2. What does the wildcard do with V4?

```rust,ignore
#[attune(make(_*))]
fn work(...) { ... }
```

**Accepted — Gate V4 on the provider's avx512 feature, matching magetypes.** Retain the documented V3/NEON/WASM/scalar defaults and optional
V4. Existing scalar autoversion users may need to enable a feature to retain
their previous V4 generation. The declaring crate must own/forward this feature.

**Alternative not selected — Always include V4.** Matches current scalar autoversion behavior. Every
body/backend requested by the wildcard must support V4; vector/backend feature
requirements still have to be satisfied. Runtime CPU detection provides hardware
safety, not missing compile-time backend support.

**Alternative not selected — Require explicit V4 selection.** Removes an implicit Cargo-feature policy
from the wildcard; V4-capable users write an extra selector and gate. It changes
the wildcard's relationship to both existing default sets.

**Accepted independently of the default:** `-_v4` removes V4 if present and is
valid if absent, including when cfg has already disabled it. Unknown tier names
remain invalid. Conflicting-selector precedence still needs a rule.

Evidence: current autoversion and magetypes generate different V4 surfaces with
the feature disabled. Both references compile when enabled. No performance
ranking or compile-time advantage between these policies was measured.

#### Follow-up: V4 versus V4x hardware coverage

Checked 2026-10-07 against the registry and primary sources. V4x is a strict
superset of our V4 tier. The useful population comparison is V4x-capable versus
V4-capable but not V4x-capable, not V4x versus all V4-capable CPUs.

[LLVM's CPU feature definitions](https://github.com/llvm/llvm-project/blob/main/llvm/lib/TargetParser/X86TargetParser.cpp)
include our V4x requirements for AMD Zen 4 and Zen 5. Intel's
[feature introduction table](https://cdrdv2-public.intel.com/855340/361050-004-intel-avx10.2-spec.pdf)
places the remaining V4x additions at Ice Lake; Sapphire Rapids retains them.
Intel's [older Xeon AVX-512 table](https://www.intel.com/content/www/us/en/support/articles/000058341/processors/intel-xeon-processors.html)
documents the narrower Skylake/Cascade Lake/Cooper Lake sets.
These establish hardware capability, not installed-base prevalence. No customer
CPU distribution was measured, so do not claim V4x machines outnumber V4-only
machines in the user population.

A gated V4x default is a candidate worth comparing with the accepted gated V4
policy, not an accepted change. V4x-only would send V4-only CPUs to a lower
fallback such as V3. Generating both preserves their V4 path but adds a tier
body. Neither runtime benefit nor compilation-cost difference was measured.
Keep this hardware-policy question separate from the accepted avx512 gate.

### Q3. How much enclosing context may attune process?

**Accepted — Function attributes only, with a clear supported subset.** Do not
annotate or process an enclosing impl/trait to obtain context. Unsupported
shapes should say "move this kernel outside of the impl" rather than trigger
more source analysis. Inherent sibling generation can retain its original impl
scope; missing generic context for nested trait helpers is not inferred.

```rust,ignore
impl<T: Copy> Kernel for Processor<T> {
    fn apply(&self, data: &[f32]) -> Self::Output {
        // Ordinary adapter: the free kernel declares the bounds it needs.
        attuned!(apply_kernel(self, data))
    }
}

#[attune(make(all))]
fn apply_kernel<T: Copy>(processor: &Processor<T>, data: &[f32]) -> Output<T> {
    // Operation; concrete bounds/output must match the real API.
}
```

This is an illustrative API shape, not an executed fixture. The initial support
boundary is in [spec section 5](../../ATTUNE_SPEC.md#5-self-self-and-placement).
Trait-method transformation requiring additional helpers is excluded initially;
ordinary trait methods may delegate to supported free kernels.

Evidence: the executed handwritten adapters preserve generic and dyn contracts,
but explicitly redeclare required generics. Implicit nested capture fails E0401;
foreign inherent helpers fail E0116. Legal manual adapters do not imply cheap or
complete automatic transformation.

**Accepted performance gate:** no end-user compilation regression. Measure
unchanged legacy and equivalent migrated consumers on matched compiler,
dependencies, outputs and tiers, including cold/incremental builds and generic
macro-heavy cases. Repeated measurements must distinguish noise from a
reproducible regression. Function-only expansion limits scope but is not proof
that this gate passes. No such comparison has yet been measured for attune.

## Next discussion: call surface and compatibility

### Q4. What automatic discovery do we promise for sparse families?

**Accepted — Conventions plus explicit lists for sparse families initially.**
Suppose an external family has only V3 and scalar implementations, while its
caller is V4:

```rust,ignore
attuned!(dependency::work(x), [_v3]) // Explicit covered V3; no detection.
```

The no-list call does not automatically discover V3. Its documented convention
still needs a precise same-tier/default-family contract. Callers must know the
available tiers, and names, visibility and cfg gates must agree. Missing symbols
are compile errors; they cannot be used as reflection or implicit fallback.

Do not require automatic discovery or emit its provider descriptor infrastructure
in the initial API. The prior macro/enum probes remain evidence for possible
future work, not a required component. They demonstrated namespace/cfg behavior
and limitations, not a complete protocol or a compile-time advantage. Entry-form
selection (direct versus _t versus dispatcher) likewise needs an explicit
contract rather than probing which names exist.

Trait-object calls keep their existing method contract and dyn compatibility;
explicit tier lists do not add statically callable members to a trait.

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
- Stay within the annotated function. Preserve inherent sibling scope; reject
  shapes needing unavailable enclosing context. Document explicit free-kernel
  extraction for trait callers, preserving their generic and dyn contracts.
- Preserve argument ownership and single evaluation on a selected call path.
- Propagate track_caller across forwarding layers; place expect where it is
  fulfilled rather than copying it to every generated function.
- Report unavailable/inaccessible required entries clearly. Keep existing
  compatibility tests and negative tests intact.
- Keep experiments off main and measure compilation after the actual generator
  and call lowering exist; namespace probes cannot establish those costs.

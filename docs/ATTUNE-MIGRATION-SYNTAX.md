# Attune migration syntax and spelling cost — 2026-10-09

Checked against draft `5a745872` (parser, emitters and legacy frontends).
This is a spelling comparison and design recommendation, not a completed
migration converter. No macro implementation or defaults change in this document.

The autoversion-shaped form is `#[attune(make(_))]`: a dispatcher with private
implementation bodies. `make(_*, _)` additionally exposes direct variants;
`make(all)` additionally exposes both direct and proof-taking variants.

```rust,ignore
#[attune(make(_))]
pub fn sum(values: &[f32]) -> f32 {
    values.iter().copied().sum()
}
```

| Requested surface | Spelling | Characters |
|---|---|---:|
| Dispatcher only | `#[attune(make(_))]` | 18 |
| Direct variants and dispatcher | `#[attune(make(_*, _))]` | 22 |
| Direct variants and proof wrappers | `#[attune(make(_*, _*_t))]` | 25 |
| Direct variants, proof wrappers and dispatcher | `#[attune(make(all))]` | 20 |

These surfaces differ when the source is public: `all` exports more functions
than autoversion's dispatcher alone. It should not be the automatic migration
replacement merely because it is short.

## Counted spelling cost

Counts are Python `len()` of the exact ASCII spellings shown, including spaces,
brackets and punctuation. A negative delta saves characters. This measures
written syntax, not compiler tokens, LLM tokens, generated code or compile time.
Signature/body changes and rename dictionaries are outside these counts and
must not be inferred to be free. Notes identify semantic/API differences.

| Case | Old | New | Characters old → new | Delta | Migration note |
|---|---|---|---:|---:|---|
| Single V3 direct function | `#[rite(v3)]` | `#[attune(v3)]` | 11 → 13 | +2 | Keeps the written name and covering-feature requirement. |
| Direct function already named work_v3 | `#[rite(v3)]` | `#[attune]` | 11 → 9 | -2 | Tier comes from the existing suffix; no function rename counted. |
| Proof boundary | `#[arcane]` | `#[attune(wrap)]` | 9 → 15 | +6 | Keeps token position, generic proof bounds and callable name. |
| Trait boundary with receiver rewriting | `#[arcane(_self = Foo)]` | `#[attune(wrap, _self = Foo)]` | 22 → 28 | +6 | Retains legacy _self rewriting; not a sibling-generating family. |
| Two direct variants | `#[rite(v3, scalar)]` | `#[attune(make(_v3, _scalar))]` | 19 → 29 | +10 | Same direct suffixes. Tier-bound/token signature changes must be handled separately. |
| Two proof variants | `#[magetypes(v3, scalar)]` | `#[attune(make(_v3_t, _scalar_t))]` | 24 → 33 | +9 | Proof names acquire _t; update callers or use names(...). Token placeholder/aliases need their own review. |
| Two direct magetypes variants | `#[magetypes(rite, v3, scalar)]` | `#[attune(make(_v3, _scalar))]` | 30 → 29 | -1 | Keep any define(...) option; inspect signature/body Token substitutions. |
| Dispatcher-only shape | `#[autoversion]` | `#[attune(make(_))]` | 14 → 18 | +4 | Source API only; new default set omits legacy feature-gated V4. See strict migration notes. |
| V3 + scalar dispatcher, variants private | `#[autoversion(v3, scalar)]` | `#[attune(make(pub(self) _v3, _))]` | 26 → 33 | +7 | Scalar fallback is implicit/private. Written visibility applies to dispatcher, not the explicitly restricted variant. |
| Ordinary family dispatch | `incant!(work(x))` | `attuned!(work(x))` | 16 → 17 | +1 | Target proof names and default tier sets change; this is not a blind rename. |
| Covered-context direct call | `incant!(work(x) without token)` | `attuned!(work(x))` | 30 → 17 | -13 | Legacy is exact-tier; new form selects a covered available tier. Add an explicit list if exact selection matters. |
| Runtime reselection within a context | `incant!(work(x))` | `reattune!(work(x))` | 16 → 18 | +2 | Use only where the legacy call actually permitted an upgrade; not all incant calls need reselection. |
| Use an existing proof | `incant!(work(x) with token)` | `attuned!(work(x), using(token))` | 27 → 31 | +4 | Proof evaluated once; no CPU detection. Preserve explicit tier lists and renamed proof entries. |

The measurements cover individual spellings, not the aggregate cost across the
consumer repositories. A fleet-wide net character count has not been measured.

## Autoversion: API shape is not complete behavioral equivalence

The legacy default set is V4 gated by the declaring crate's `avx512` feature,
V3, NEON, WASM128 and scalar. The new wildcard set is V3, NEON, WASM128 and scalar;
AVX-512 must be requested. `make(_)` uses that smaller default set. It is the
same dispatcher-only API shape, not an exact default-tier migration.

The current grammar can preserve the legacy tiers and keep named helpers private:

```rust,ignore
#[attune(make(pub(self) _*_t, +v4(avx512), _))]
```

`pub(self)` restricts the wildcard outputs, while `_` inherits the source
visibility. This still changes internal helper names to `_t` names, and body
inline differences (especially the scalar fallback) need review. Private helpers
can be referenced elsewhere in their module, so private naming changes are not
necessarily harmless. A name dictionary can preserve required helper names.

There is a grammar gap: `make(_, +v4(avx512))` currently fails because `+tier`
requires a wildcard output form. A dispatcher-only tier-list spelling could
remove this verbosity, but is not implemented or settled here. Do not present
`make(auto)` as existing syntax either; the parser accepts `_`, wildcards and
`all`, not `auto`.

Signature distinctions also matter:

- Tokenless legacy autoversion functions remain tokenless with `make(_)`.
- An unused legacy `_token: SimdToken, ` can be removed: that exact fragment is
  19 characters. If the proof is used, a body/signature migration is necessary.
- The new `Token` parameter is retained as `ScalarToken` on a dispatcher;
  mechanically replacing `SimdToken` with `Token` does not preserve the old
  tokenless public dispatcher signature.
- Legacy real `ScalarToken` dispatcher parameters remain public parameters. A
  `Token` family placeholder can specialize variant proofs while retaining a
  scalar dispatcher parameter; check its position and body uses.
- Type/const generics do not by themselves require extra syntax. Receiverless
  associated families require `in_impl`. Sibling-generating trait families remain
  unsupported: keep the legacy frontend or move the kernel outside the impl.
- Current `define(...)` aliases remain token-first magetypes aliases. This macro
  comparison does not remove constructor token arguments or implement `use(...)`.

## Inline recommendation

Keep operation-body `#[inline]` as the omission default; do not require an
explicit `inline(hint)` on every definition. Keep thin proof wrappers
`#[inline(always)]`. The existing draft already follows those omission defaults
for ordinary operation bodies and proof outputs; central dispatchers currently
have no implicit inline attribute. Existing explicit source attributes also
participate where preserved by the frontend.

The [paired retest](../benchmarks/inline_operation_2026-10-09/README.md) supports
preserving body hints over the visibility heuristic on the measured non-LTO
encoder cases. The [earlier layer comparison](../benchmarks/inline_real_2026-10-08/README.md)
supports preserving wrapper inlining. Neither covers enough processors/workloads
to establish a universal optimum or settle central dispatcher inlining.
These are equivalent legacy-emitter experiments, not whole-rewrite measurements.
The earlier report's statement about magetypes implementations being intact
refers to source bodies; macro-generated backend attributes were changed, as the
[expansion inventory](../benchmarks/inline_operation_inventory_2026-10-09/README.md)
shows.

Proposed exception spelling: `no_inline` means **emit no body inline attribute**,
not `#[inline(never)]`, and does not strip the proof wrapper's separate attribute.
Use `inline(never)` for an explicit out-of-line request. Keep per-output controls
for exceptions on wrappers and dispatchers. `no_inline` is not parsed today;
its existing draft equivalent is `inline(none)`.

If omission means an ordinary hint, the visibility heuristic should not be named
`inline(default)`. Reserve that name for the omission policy; if the heuristic
is kept, a name such as `inline(visibility)` would make its behavior explicit.
This rename is a proposal, not accepted syntax.

| Change under this recommendation | Before | After | Characters old → new | Delta |
|---|---|---|---:|---:|
| Omit redundant body choice | `#[attune(inline(hint), make(_))]` | `#[attune(make(_))]` | 32 → 18 | -14 |
| Proposed no-attribute alias | `#[attune(v3, inline(none))]` | `#[attune(v3, no_inline)]` | 27 → 24 | -3 |


No new benchmark was run for this table. Resource record for the linked paired
runtime test: `rc=0 512s | peak-RSS 0.02GiB | min-avail 26136MiB | peak-load 4.33`.
The rewrite's cold-compile acceptance gate remains open; shorter source spelling
does not establish lower macro expansion or end-user build cost.

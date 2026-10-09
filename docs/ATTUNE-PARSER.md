# Attune definition grammar and validation

The unpublished definition parser has three stages: written syntax, validated
concrete outputs, and Rust emission. Published legacy attributes keep their own
parsers. Inline omission defaults and generated call paths are unchanged.

## Canonical spelling

```rust,ignore
#[attune(
    _*(pub(crate)),
    _*_t(pub),
    _v4x_t(cfg(avx512), pub),
    dispatch(pub),
    inline(hint),
)]
fn work(x: u32) -> u32 { x + 1 }
```

Output selectors can appear directly in `attune(...)`. `make(...)` remains a
compatibility spelling on this draft, and accepts selector-local options too.
Do not mix grouped and flat outputs in one declaration. Existing positional
options and bare feature names remain accepted inside `make(...)` only.

The leading underscore distinguishes an output name from a context:

| Declaration | Meaning for `fn work(...)` |
| --- | --- |
| `#[attune(v3)]` | Keep `work`; establish the V3 context |
| `#[attune(_v3)]` | Generate `work_v3` |
| `#[attune(_v3_t)]` | Generate `work_v3_t` and its private implementation |
| `#[attune(dispatch)]` | Generate `work` dispatching to private implementations |
| `#[attune(all)]` | Generate default direct/proof variants and the dispatcher |
| `#[attune]` | Infer the context or proof boundary from the written suffix |

The old draft accepted a top-level `_v3` as a context alias. It now means a
generated output. Use `v3` to preserve the written function name. Published
`rite`, `arcane`, `autoversion`, `magetypes`, and call-macro syntax are unaffected.

Selector-local options are `pub`/`pub(...)`, `cfg(feature)`, and
`inline(policy)`. Their order is irrelevant; trailing commas are accepted.
`cfg(feature)` means one Cargo feature in the declaring crate, not arbitrary
Rust cfg syntax. Gates belong to named tiers, and all interfaces for one tier
must agree on the gate. Unknown options and repeated options are errors.

Definition-level `inline(...)` controls operation bodies. Direct-selector
inline options override a body's policy. Proof/dispatcher-selector inline
options control those wrappers independently. For example:

```rust,ignore
#[attune(_*_t(pub, inline(always)), inline(never))]
```

This requests always-inline proof wrappers around never-inline bodies.

`+tier` extends selected wildcard forms, inheriting their visibility and inline
policy unless explicitly overridden on that addition. For dispatcher-only
families, additions stay private and accept only a gate. `-tier` removes both
forms; `-tier_t` removes proof outputs. Removals take no options. `+tier_t` is
rejected: use `+tier` to extend forms or `_tier_t` to request a proof output.
These errors replace cases where the old draft silently ignored modifier options
or the `_t` on an addition.

Normalization first gathers outputs, then applies additions/removals in their
written order. A modifier may precede the wildcard or dispatcher it modifies.
Removing an absent tier is valid. Identical output selections coalesce;
conflicting policies fail. A dispatcher always needs an ungated scalar fallback.

## Implementation boundaries

- `archmage-macros/src/attune/syntax/grammar.rs` reads typed selectors, actions,
  options, and source spans. It does not expand wildcards. Duplicate option keys
  are tracked without a heap map; aliases share a key.
- `syntax/resolve.rs` expands registry defaults, applies modifiers, coalesces
  identical outputs, and validates gates, policies, fallback, and mode conflicts.
  Errors tied to an output use its source span.
- `syntax.rs` defines the concrete plan and shared tier/name/inline helpers.
  `attune/mod.rs` consumes that plan. It does not parse selector syntax or
  resolve wildcard conflicts. Function-dependent restrictions, such as safe
  trait placement and opaque dispatcher returns, remain in emission validation.

The grammar uses enums for target and action, with fixed option fields. There is
no generic string-to-value option bag, full function-body parse, source-file scan,
or new dependency. Adding a tier goes through the registry; it does not require
another grammar branch.

## Tests and extension rules

`just attune-parser` runs macro tests plus integration compilation/execution with
the AVX-512 Cargo feature disabled and enabled. It covers these boundaries:

- Parsed selectors remain unexpanded, and grammatical mode conflicts are rejected
  by the separate resolver.
- Context tiers and generated suffixes have different naming behavior.
- Canonical, grouped, and positional spellings produce identical emitted Rust
  for equivalent requests.
- Option order, trailing commas, duplicate handling, unknown syntax, gate
  consistency, modifier order, fallback, and body/wrapper inline scope.
- Every registered tier uses the same direct/proof selector grammar.
- Real generated dispatchers, generic functions, methods, associated functions,
  renamed functions, and composed calls compile under `forbid(unsafe_code)` and
  `deny(warnings)`.

Existing legacy expansion and input/output compile tests remain unchanged and
run through `just attune-compat`. Enabled AVX-512 compilation is not a claim of
AVX-512 runtime coverage.

For an extension, specify its scope and conflicts first; add a typed field or
enum variant; add positive, rejection, and equivalence cases; then measure
expansion allocation and consumer compile cost. Keep unknown syntax an error so
future additions cannot silently reinterpret previously accepted typos. Reuse
the parser for future migration tooling rather than maintaining a second grammar.
No migration converter is implemented by this refactor.

The [allocation and cold-build comparison](../benchmarks/attune_parser_2026-10-09/README.md)
records the refactor's measured costs, source revisions, and coverage limits.

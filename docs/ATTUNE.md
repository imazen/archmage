# Unified macros in 0.9.31-beta

`#[attune]` defines feature bodies, proof wrappers, and dispatchers. `attuned!`
calls a family using the enclosing feature context or proof; `reattune!` permits
runtime reselection. Existing `arcane`, `rite`, `autoversion`, `magetypes`, and
`incant!` spellings remain supported and are not newly deprecated in this beta.

Use matching beta versions of archmage and magetypes. The archmage dependency
pins archmage-macros exactly because generated code refers to archmage APIs.
Magetypes retains its existing token-taking constructors, including `_t` names.

## One function

| Definition | Meaning |
| --- | --- |
| `#[attune(v3)] fn work(...)` | Keep the name and signature; enable V3 features |
| `#[attune] fn work_v3(...)` | Infer direct V3 features from the suffix |
| `#[attune(wrap)] fn work(token: X64V3Token, ...)` | Keep the public name; authenticate proof and call a private feature body |
| `#[attune] fn work_v3_t(...)` | Infer a V3 proof wrapper; insert a concrete token parameter if omitted |

A token parameter alone does not select `wrap`. A written proof parameter keeps
its position; a `Token` placeholder is specialized in generated families and
inferred proof wrappers. A proof suffix must match the written proof's features.
Explicit `wrap` derives its features from the proof, including supported trait
bounds. Explicit tiers override suffix inference.

## Generate a family

```rust,ignore
#[attune(_*, _*_t, dispatch)]
fn work(x: u32) -> u32 { x + 1 }
```

This exposes direct `work_v3`, `work_neon`, `work_wasm128`, and `work_scalar`
functions, matching `_t` proof wrappers, and an ordinary `work` dispatcher.
Architecture-inapplicable variants are omitted. The dispatcher checks available
tiers in priority order and retains an ungated scalar fallback.

| Selector | Outputs |
| --- | --- |
| `_*` | Direct variants for the default portable tiers |
| `_*_t` | Proof wrappers for the default portable tiers |
| `dispatch` or `_` | Dispatcher and private bodies |
| `all` | Direct variants, proof wrappers, and dispatcher |
| `_v3`, `_v3_t` | One explicitly named output |
| `+v4x(cfg(avx512))` | Add the tier to wildcard/dispatcher outputs under the caller's Cargo feature |
| `-_neon` | Remove the tier; removing an absent tier is allowed |

V4 and V4x are explicit additions, not inferred from the implementation crate's
features. A tier's direct and proof outputs must share a Cargo gate.

Selectors accept visibility and inline overrides:

```rust,ignore
#[attune(_*(pub(crate)), _*_t(pub), dispatch(pub), inline(hint))]
fn work(x: u32) -> u32 { x + 1 }
```

`make(...)` grouping and positional visibility remain accepted. `v3` outside
`make` means a single context; `_v3` requests a suffixed output. Keep grouped
outputs together rather than mixing them with flat selectors. Unknown options,
duplicate policies, and incompatible placements are errors.

`define(f32x8)` creates a magetypes alias specialized for each generated tier.
Use `names(_scalar = fallback, _scalar_t = fallback_t)` to override output names.
This beta does not add the previously discussed `use(...)` constructor mode.

## Calls and proofs

```rust,ignore
attuned!(dependency::work(x), [_v3, _scalar]);
attuned!(dependency::work(x), [_v3, _scalar], using(token));
reattune!(dependency::work(x), [_v4x(cfg(avx512)), _v3, _scalar]);
```

Inside an annotated feature body, a covered candidate is a direct call. When a
candidate needs proof, `attuned!` can use one eligible enclosing proof parameter.
Explicit `using(...)` wins and evaluates its expression once without CPU probing.
Ordinary unannotated functions need explicit `using(...)` to supply existing proof.

`reattune!` permits runtime CPU detection for stronger tiers. With explicit
`using(...)`, selection uses that proof instead. Every call requires a guaranteed
fallback after architecture and Cargo gates are applied; a possible runtime match
alone is insufficient. Use an explicit tier list for sparse or external families.
The default list assumes the portable family convention.

Multiple enclosing proofs require an explicit choice only when an active candidate
needs one. Unavailable candidates cannot force ambiguity errors. Calls can use a
`names(...)` dictionary to map tiers to nonconventional paths. A qualified path,
turbofish, or associated path such as `Self::work` is supported; receiver syntax
such as `attuned!(self.work(x))` is not supported in this beta. Definition modifiers
such as `-_neon` are not call-list syntax: spell the remaining tiers explicitly.

## Methods, nesting, and inlining

Inherent receiver methods can use sibling expansion. Receiverless associated
functions need `in_impl` for sibling name resolution. Trait implementations need
`in_trait`/`nested`, with `_self = ConcreteType` for a receiver; `_self` also implies
nesting. The body may keep ordinary `self`. Nested helpers cannot implicitly
capture outer impl generics. Move an unsupported kernel outside the impl and make
its generics explicit. Default trait methods with receivers and no concrete
`_self` remain unsupported; receiverless default methods work.

Operation bodies keep an inline hint when the policy is omitted, as in the legacy
attributes. Proof wrappers and dispatchers default to `inline(always)` separately.
`inline(default)` hints unrestricted public bodies and emits no inline attribute
for restricted/private bodies. Trait visibility cannot be inferred for that
policy. `inline(hint)`, `inline(none)`, and `inline(never)` are explicit choices;
`none` leaves the compiler's normal heuristics in control. Selector-local policies
override the corresponding output. Stable Rust rejects `inline(always)` on
feature bodies; selecting it for proof wrappers or dispatchers is permitted.

## Beta scope and validation

The beta keeps migration manual: there is no complete source converter or exact
replacement deprecation machinery. Matched cold-compile measurements remain a release gate.
The pre-beta measurements and design history remain on the archived
`draft/attune-rewrite` branch; they are not part of the release patch.

The [expansion corpus](../tests/attune_expansion/README.md) documents finite syntax
products, calling contexts, raw replay, and expected rejections. These checks
complement runtime tests and cross-platform CI; passing the matrix does not make
every Rust signature or attribute combination supported.

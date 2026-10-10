# Unified macros in 0.9.31-beta

`#[attune]` defines feature bodies, proof wrappers, and dispatchers. `attuned!`
calls a family using the enclosing feature context or proof; `reattune!` permits
runtime reselection. Existing `arcane`, `rite`, `autoversion`, `magetypes`, and
`incant!` spellings remain supported and are not newly deprecated in this beta.

Use matching beta versions of archmage and magetypes. The archmage dependency
pins archmage-macros exactly because generated code refers to archmage APIs.
Magetypes retains its existing token-taking constructors, including `_t` names;
`_t(token, ...)` stays token-first, and the short names have not become
tokenless. The `use(...)` constructor mode and adaptive-alias experiments are
not part of this beta.

## One function

| Definition | Meaning |
| --- | --- |
| `#[attune(v3)] fn work(...)` | Keep the name and signature; enable V3 features |
| `#[attune] fn work_v3(...)` | Infer direct V3 features from the suffix |
| `#[attune(wrap)] fn work(token: X64V3Token, ...)` | Keep the public name; authenticate proof and call a private feature body |
| `#[attune] fn work_v3_t(...)` | Infer a V3 proof wrapper; insert a concrete token parameter if omitted |

The suffix must be the final `_`-separated segment of the name: `work_v3` and
`work_v3_t` infer V3, `v3_t` infers a V3 proof wrapper, and a name that merely
ends in the same characters (`my_superv3`) does not infer. A `_tier_t` suffix
takes the tier's token as its proof; a `_tier` suffix is a direct feature body
even when a token parameter is written. Explicit tiers override suffix
inference: `#[attune(v3)] fn work_v4(...)` keeps the name `work_v4` and enables
V3 features.

A token parameter alone does not select `wrap`. `#[attune] fn work_v3(token:
X64V3Token, ...)` is a direct feature body that keeps and forwards that
parameter; `#[attune] fn work(token: X64V3Token, ...)` has neither a suffix nor
a tier and fails with "attune needs a tier, a registered tier suffix (optionally
_t), wrap, or make(...)". `#[attune(wrap)]` needs no tier and rejects one
("wrap derives its features from the proof parameter; omit the explicit tier").
A written proof keeps its position; a `Token` placeholder is specialized in
generated families and inferred proof wrappers. A proof suffix must match the
written proof's features ("proof parameter does not match the inferred tier
suffix; use the matching token or explicit wrap"). Explicit `wrap` derives its
features from the proof, including supported trait bounds such as
`impl HasX64V2`.

```rust
use archmage::prelude::*;

// Direct feature body: called only from a matching feature context.
#[cfg(target_arch = "x86_64")]
#[attune(v3)]
fn add_one(x: u32) -> u32 {
    x + 1
}

// Inferred proof wrapper: safe to call from ordinary code after summon().
// Its body is a V3 feature context, so the direct call above is safe.
#[cfg(target_arch = "x86_64")]
#[attune]
fn add_two_v3_t(x: u32) -> u32 {
    add_one(x) + 1
}

// wrap: features come from the proof parameter; the name is unchanged.
#[cfg(target_arch = "x86_64")]
#[attune(wrap)]
fn add_three(token: X64V3Token, x: u32) -> u32 {
    let _ = token;
    add_one(x) + 2
}

#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() {
    assert_eq!(add_two_v3_t(token, 1), 3);
    assert_eq!(add_three(token, 1), 4);
}
```

## Generate a family

```rust
use archmage::prelude::*;

#[attune(_*, _*_t, dispatch)]
fn work(x: u32) -> u32 {
    x + 1
}

// The dispatcher replaces runtime selection.
assert_eq!(work(1), 2);
// `attuned!` selects the same way from any caller.
assert_eq!(attuned!(work(41)), 42);
```

This exposes direct `work_v3`, `work_neon`, `work_wasm128`, and `work_scalar`
functions, matching `_t` proof wrappers, and an ordinary `work` dispatcher.
Architecture-inapplicable variants are omitted. The dispatcher checks available
tiers in priority order and retains an ungated scalar fallback.

| Selector | Outputs |
| --- | --- |
| `_*` | Direct variants for the default portable tiers |
| `_*_t` | Proof wrappers for the default portable tiers |
| `dispatch` or `_` | Dispatcher, plus private bodies for tiers not otherwise selected |
| `all` | Direct variants, proof wrappers, and dispatcher |
| `_v3`, `_v3_t` | One explicitly named output |
| `+v4x(cfg(avx512))` | Add the tier to wildcard/dispatcher outputs under the caller's Cargo feature |
| `-_neon`, `-_neon_t` | Remove the tier (or only its proof wrapper); removing an absent tier is allowed |

The default set is `v3`, `neon`, `wasm128`, and `scalar`. V4 and V4x are
explicit additions, not inferred from the implementation crate's features;
unlike `#[magetypes]` and `incant!`, this default set never includes them.

Modifier rules, each with its diagnostic:

- `+tier` adds only to wildcard (`_*`/`_*_t`/`all`) outputs or to a dispatcher's
  private bodies. On a sparse explicit family it errors ("+tier requires a
  wildcard output form or a dispatcher"). It cannot request a proof wrapper; use
  `_tier_t`.
- Removals accept no options ("removals do not accept options"), may name an
  absent tier, and `-_neon_t` removes only the proof wrapper while `-_neon`
  removes every output of that tier.
- A tier's direct, proof, and dispatcher implementations must share one Cargo
  gate ("a tier's direct, proof and dispatcher implementations must have the
  same feature gate"). A wildcard cannot carry a gate ("cfg belongs on a named
  tier; use outer #[cfg(...)] to gate the whole family").
- `dispatch` requires an ungated scalar fallback ("a dispatcher requires its
  scalar fallback" when removed, "a dispatcher requires an ungated scalar
  fallback" when gated).
- `all` already includes a dispatcher, so selecting it twice is a duplicate
  error. A declaration that selects no outputs errors ("declaration selects no
  outputs").

Grouped and flat spellings cannot be mixed ("keep all outputs inside make(...),
or omit make(...)"), and generated outputs cannot combine with `wrap` or a
single context tier ("generated outputs cannot be combined with wrap or a single
context tier"). Inside `make(...)`, positional `pub`/`inline(...)` before a
selector remain accepted, as does the old `make(_v3(feature))` gate shorthand;
flat selectors take options in parentheses.

### Visibility, inline, and naming

```rust
use archmage::prelude::*;

#[attune(_*(pub(crate)), _*_t(pub), dispatch(pub), inline(hint))]
fn visible(x: u32) -> u32 {
    x + 1
}
```

Selector options are `pub` (any Rust visibility), `cfg(feature)`, and
`inline(policy)`. Duplicates in one selector are errors ("duplicate visibility
option", "duplicate cfg option", "duplicate body inline policy"), and unknown
options are rejected ("expected pub visibility, cfg(feature), or
inline(policy)"). Direct outputs use the source or selector visibility; hidden
implementation bodies and unselected proof wrappers are private.

```rust
use archmage::prelude::*;

#[attune(_scalar, _scalar_t, names(_scalar = fallback, _scalar_t = fallback_t))]
fn leaf(x: u32) -> u32 {
    x + 1
}

fn caller(x: u32) -> u32 {
    // Definition names must be identifiers; call-site names may be paths.
    // The call dictionary maps the tier back to the renamed output.
    attuned!(
        leaf(x),
        [_scalar],
        names(_scalar = fallback, _scalar_t = fallback_t)
    )
}

assert_eq!(caller(1), 2);
```

`define(f32x8)` creates a magetypes alias specialized for each generated tier.
It requires a concrete tier or family ("define(...) requires a concrete tier;
use a direct tier body or a family", "define(...) on a proof wrapper requires a
generated family").

```rust
use archmage::prelude::*;

#[attune(_*_t, define(f32x8))]
fn scale8(token: Token, data: &mut [f32; 8]) {
    (f32x8::load_t(token, data) * f32x8::splat_t(token, 2.0)).store(data);
}

let mut data = [1.0f32; 8];
attuned!(scale8(&mut data));
assert_eq!(data, [2.0; 8]);
```

## Calls and proofs

```text
attuned!(dependency::work(x), [_v3, _scalar]);
attuned!(dependency::work(x), [_v3, _scalar], using(token));
reattune!(dependency::work(x), [_v4x(cfg(avx512)), _v3, _scalar]);
```

Outside an annotated feature body, `attuned!` and `reattune!` can only call
proof wrappers: every candidate is converted to its `_t` name, and the token is
the summoned or supplied proof. Inside an annotated body, a candidate covered by
the enclosing context is a direct call with no probe and no token argument (a
covered context may even omit the scalar fallback, because the covered tier is
itself the guarantee). When a candidate needs proof, `attuned!` can use one
eligible enclosing proof parameter: a concrete token proves its exact tier or a
registry ancestor (`X64V3Token` proves V2 calls, `X64V4xToken` proves V3 and V2
calls); a generic `IntoConcreteToken` parameter selects by exact type at compile
time. `reattune!` skips the enclosing-proof path and probes for stronger tiers;
`using(...)` overrides inference, evaluates its expression once, and never
probes the CPU.

```rust
use archmage::prelude::*;

#[attune(all, +v4x(cfg(avx512)))]
fn leaf(x: u32) -> u32 {
    x + 1
}

// Ordinary caller: every candidate is a proof wrapper, summoned best-first.
fn default_call(x: u32) -> u32 {
    attuned!(leaf(x))
}

// A held token is passed once through using(...); no CPU probe.
fn using_call(x: u32) -> u32 {
    let Some(token) = X64V3Token::summon() else {
        return attuned!(leaf(x), [_scalar]);
    };
    attuned!(leaf(x), [_v3, _scalar], using(token))
}

// One enclosing proof parameter is inferred for uncovered candidates.
#[attune(scalar)]
fn dispatch_to_leaf<P: IntoConcreteToken>(parent: P, x: u32) -> u32 {
    let _ = parent;
    attuned!(leaf(x), [_v3, _neon, _wasm128, _scalar])
}

// reattune! probes for stronger tiers from a covered context.
#[cfg(target_arch = "x86_64")]
#[attune(v3)]
fn widen_impl(x: u32) -> u32 {
    reattune!(leaf(x), [_v4x(cfg(avx512)), _v3, _scalar])
}

#[cfg(target_arch = "x86_64")]
#[attune(wrap)]
fn widen(token: X64V3Token, x: u32) -> u32 {
    widen_impl(x)
}

assert_eq!(default_call(1), 2);
assert_eq!(using_call(1), 2);
assert_eq!(dispatch_to_leaf(ScalarToken, 1), 2);

#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() {
    assert_eq!(widen(token, 1), 2);
}
```

Call syntax:

- A callee list accepts tier names with an optional `_t` suffix and an optional
  gate: `_v3`, `_v3_t`, `_v4x(cfg(avx512))`. The list is exhaustive; no scalar
  fallback and no AVX-512 tier is ever appended implicitly. `_t` forces the
  proof entry even inside a covered context.
- `using(expr)` takes one proof expression: a token, `impl IntoConcreteToken`,
  or `Token::from_context()`. Multiple `using(...)` clauses, multiple callee
  lists, and definition modifiers used as call lists are errors. `-_neon` is
  definition syntax, not call-list syntax; spell the remaining tiers.
- `names(...)` maps tiers to nonconventional paths for external families.
- A qualified path, turbofish, or associated path such as `Self::work` is
  supported; receiver syntax such as `attuned!(self.work(x))` is rejected
  ("expected parentheses"). Method families are called as methods, for example
  `self.work_v3_t(token, x)` or through their dispatcher.
- In an operation whose token is not first, mark its position in the call with
  `Token`; the marker is replaced by the selected proof. Without a marker the
  proof is prepended.

Every call must have a guaranteed fallback after architecture and Cargo gates
are applied. A possible runtime match alone is insufficient; an unannotated
`attuned!(leaf(x), [_v3])` fails with "attuned!/reattune!: no guaranteed
fallback; include scalar or a tier covered by the caller's target features".
Unavailable candidates cannot force an ambiguous parent-proof error before an
available fallback; with two eligible proofs, choose with `using(token)`.

## Methods, nesting, and inlining

Inherent receiver methods can use sibling expansion: `#[attune(wrap)] fn
run(&self, token: X64V3Token, ...)` generates a private sibling and works
unchanged, and families generate methods (`run_v3`, `run_v3_t(&self, token,
...)`, and a dispatcher). Receiverless associated functions need `in_impl` so
generated siblings are called as `Self::name`; this applies to `wrap` and to
family dispatchers alike.

Trait implementations need `in_trait`/`nested`, with `_self = ConcreteType` for
a receiver; `_self` also implies nesting. The body may keep ordinary `self`.
Receiverless default trait methods work with `#[attune(wrap, in_trait)]`.
Default trait methods with receivers and no concrete `_self` remain unsupported
("attune with self receiver in nested mode requires `_self = Type`
argument"); move the kernel to a concrete impl with `_self = Type`, or to a free
function. A direct feature body on a trait method is rejected by rustc
(`#[target_feature(..)]` cannot be applied to safe trait method); use `wrap`
with a proof parameter, or move the kernel outside the impl. Family generation
inside a trait is rejected outright ("family generation adds sibling functions;
move this kernel outside of the impl for a trait method").

Nested helpers cannot implicitly capture outer impl generics: `_self = Wrap<T>`
inside `impl<T> ...` fails with E0401. Move the kernel outside the impl and make
its generics explicit:

```rust
use archmage::prelude::*;

struct Wrap<T>(T);

#[cfg(target_arch = "x86_64")]
#[attune(wrap)]
fn get_free<T: Copy>(this: &Wrap<T>, token: X64V3Token) -> T {
    let _ = token;
    this.0
}

trait Get {
    fn get(&self, token: X64V3Token) -> u32;
}

#[cfg(target_arch = "x86_64")]
impl Get for Wrap<u32> {
    fn get(&self, token: X64V3Token) -> u32 {
        get_free(self, token)
    }
}
```

Operation bodies keep an inline hint when the policy is omitted, as in the
legacy attributes; a written `#[inline(...)]` on an `#[attune]` direct body is
preserved (legacy `#[rite]` replaced it with `#[inline]`). Proof wrappers
default to `inline(always)` separately, and dispatchers carry no inline
attribute unless a policy is selected. `inline(default)` hints unrestricted
public bodies and emits no inline attribute for restricted or private bodies;
trait placement cannot infer visibility for that policy ("inline(default) cannot
infer trait visibility; choose inline(hint), inline(none), or inline(never)").
`inline(none)` leaves the compiler's normal heuristics in control;
`inline(hint)`, `inline(never)`, and `inline(always)` are explicit choices.
Selector-local policies override the corresponding output, and definition-level
policy does not alter proof-wrapper or dispatcher defaults. Stable Rust cannot
take `inline(always)` on a feature body, so it is rejected ("inline(always) on
target-feature bodies requires nightly; use inline(hint) or inline(never)");
selecting it for proof wrappers or dispatchers is permitted.

```rust
use archmage::prelude::*;

#[attune(v3, inline(default))]
pub fn hinted(x: u32) -> u32 {
    x
}

#[attune(v3, inline(none))]
pub fn heuristic(x: u32) -> u32 {
    x
}

#[attune(_v3_t(inline(default)), dispatch(inline(always)))]
fn entry(x: u32) -> u32 {
    x
}
```

## Migrating from the legacy attributes

Every migration below is spelling-only unless the table says otherwise. Legacy
attributes keep their own defaults, so mixed files behave exactly as before.

| Legacy | Unified equivalent | API, name, and behavior changes |
| --- | --- | --- |
| `#[rite(v3)] fn helper(x)` | `#[attune(v3)] fn helper(x)` | None: same emitter, name, visibility, and `#[inline]` default. `#[attune]` preserves a written `#[inline(...)]`; `#[rite]` replaced it with `#[inline]`. |
| `#[rite] fn helper(token: X64V3Token, x)` | `#[attune] fn helper_v3(token: X64V3Token, x)` | Token-based `#[rite]` becomes suffix inference: the name must carry `_v3`, or pass `#[attune(v3)]`. The parameter is still forwarded, not consumed by a wrapper. |
| `#[rite(v3, v4, neon)] fn helper(x)` | `#[attune(_v3, _v4(cfg(avx512)), _neon)] fn helper(x)` | Same `helper_v3`/`helper_v4`/`helper_neon` names. `#[rite]` emits V4 unconditionally; `_v4(cfg(avx512))` is the portable spelling for downstream crates. |
| `#[arcane] fn kernel(token: X64V3Token, x)` | `#[attune(wrap)] fn kernel(token: X64V3Token, x)` or `#[attune] fn kernel_v3_t(token: X64V3Token, x)` | Identical boundary: public `#[inline(always)]` wrapper, private `#[doc(hidden)]` `__arcane_*` body with `#[inline]`. The `_v3_t` form changes the name unless the source already ends in it. Arcane-only `suppress_const_test` and `inline_always` have no unified spelling. |
| `#[magetypes(define(f32x8), v3, neon, wasm128, scalar)] fn gain(token: Token, ...)` | `#[attune(_*_t, define(f32x8))] fn gain(token: Token, ...)` | Variant names gain `_t` (`gain_v3_t`), keeping token-first signatures and source visibility. Dispatch moves from `incant!` to `attuned!` or a dispatcher. |
| `incant!(gain(plane, factor), [v3, scalar])` | `attuned!(gain(plane, factor), [_v3, _scalar])` | Call-list tier names take the `_` prefix. Outside a feature context the proof wrappers are called. `incant!` remains supported for legacy families and cannot dispatch unified direct bodies (E0133). |
| `incant!(m(data) with token, [v3, scalar])` | `attuned!(m(data), [_v3, _scalar], using(token))` | `using(...)` selects by exact token type, evaluates once, and never probes. |
| `#[autoversion] fn sum(data: &[f32]) -> f32` | No equivalent | Scalar auto-vectorization plus a generated dispatcher has no unified form; `#[autoversion]` remains supported. |

The migration is manual: no source converter and no replacement-deprecation
machinery ship in this beta. A family can also be migrated incrementally, one
call at a time, by pointing a call-site dictionary at the legacy names:

```rust
use archmage::prelude::*;

mod legacy {
    use archmage::prelude::*;

    #[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
    pub fn gain(token: Token, data: &mut [f32; 8]) {
        (f32x8::load_t(token, data) * f32x8::splat_t(token, 2.0)).store(data);
    }
}

// New callers can reach the legacy family through a names dictionary.
pub fn apply(data: &mut [f32; 8]) {
    attuned!(
        legacy::gain(data),
        [_v3, _neon, _wasm128, _scalar],
        names(
            _v3_t = legacy::gain_v3,
            _neon_t = legacy::gain_neon,
            _wasm128_t = legacy::gain_wasm128,
            _scalar_t = legacy::gain_scalar
        )
    );
}

let mut data = [1.0f32; 8];
apply(&mut data);
assert_eq!(data, [2.0; 8]);
```

## Beta scope and validation

The beta keeps migration manual: there is no complete source converter or exact
replacement deprecation machinery. The measured cold-compile overhead was
accepted for this beta on 2026-10-10; see the [consumer
measurements](../benchmarks/consumer_compile_beta_2026-10-10.md). The pre-beta
measurements and design history remain on the archived `draft/attune-rewrite`
branch; they are not part of the release patch.

The [expansion corpus](../tests/attune_expansion/README.md) documents finite
syntax products, calling contexts, raw replay, and expected rejections. These
checks complement runtime tests and cross-platform CI; passing the matrix does
not make every Rust signature or attribute combination supported.

### Documentation review status

The pre-publication checklist was executed on 2026-10-10 against the parser and
emitters, not only the examples:

- Definition grammar and resolution: `archmage-macros/src/attune/syntax.rs`,
  `syntax/grammar.rs`, `syntax/resolve.rs`. Emission:
  `archmage-macros/src/attune/mod.rs`, `engine/boundary.rs`,
  `engine/feature.rs`, `engine/inline.rs`. Call selection and proof inference:
  `attune/call.rs`, `attune/parent.rs`.
- Fixtures and contracts: `tests/attune_expansion/` and
  `archmage-macros/src/attune/{convention_tests,inline_tests}.rs`. Every example
  in this guide was expanded locally during review.
- The corrected statements: dispatchers carry no inline attribute unless one is
  selected (only proof wrappers default to `inline(always)`); token parameters
  alone never imply `wrap`; the unified default tier set excludes V4/V4x; and
  the call-side proof, modifier, and receiver rules now name their actual
  diagnostics.

Runnable examples are compiled and executed by `xtask/check_docs.py` with
default features and with `--features avx512`. `just docs-test` runs both. The
guide's reproducibility limits match the corpus: finite syntax products, no
claim of arbitrary signatures, and x86/AArch64 execution for the runnable
examples with WASM covered by the raw expansion matrix.

Remaining gaps, recorded rather than dropped:

- The owner's README review and final publication approval.
- Full remote CI on the final documentation commit.
- No source converter or legacy deprecation warnings ship in this beta.
- Published-site coverage of the unified macros is deferred; this guide is the
  beta documentation until release.
- The syntax and rejection evidence lives in `archmage-macros/src/attune/` and
  `tests/attune_expansion/`; the corpus does not claim the full Cartesian
  product across independent groups.
- `use(...)` constructor modes and adaptive aliases stay on
  `draft/attune-rewrite`.

The [precise semver guard](../xtask/check_semver.py) is integrated in local and
release CI (`94d828e2`); run `just attune-beta-semver` to compare against 0.9.30.
Its [validation record](../benchmarks/attune_beta_validation_2026-10-10.md)
explains the single replaced lint. Full remote CI and the owner's README review
and publication approval remain release gates.

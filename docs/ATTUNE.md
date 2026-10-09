# Unified macros: implementation contract

This is an unpublished rewrite rebased onto main at `52bf060d` (0.9.30),
including the merged PR #123 and harvested consumer signature tests. It
originally started at `de7d28af833b`. The intended next version is 0.9.31;
no release is authorized by this document. Keep it on the
`draft/attune-rewrite` bookmark until the implementation, compatibility and
compile-time gates pass. The existing attribute names and dispatch aliases
remain compatibility frontends; their behavior is the contract, including
effective inline policies and tier defaults.

The new definition spelling is `#[attune]`. `#[attune(v3)]` keeps an arbitrary
function name and adds that tier's features. A bare attribute infers a registered
terminal suffix, longest first. `#[attune(wrap)]` retains an existing proof-taking
signature, including generic bounds and the proof argument's position.

For a source named `work`, explicit outputs are:

| Selector | Interface |
| --- | --- |
| `_v3` | `work_v3(args)` in a covering feature context |
| `_v3_t` | `work_v3_t(token, args)` from ordinary code |
| `_` | `work(args)` with central detection and a private scalar fallback |
| `_*` | Direct outputs for V3, NEON, WASM128 and scalar |
| `_*_t` | Proof outputs for that same set |
| `all` | Both wildcard forms and the dispatcher |

AVX-512 is explicit: `make(_*_t, +v4x)` or
`make(_*_t, +v4x(avx512))`. The latter condition names the declaring crate's
Cargo feature. Removing a registered tier with `-_v4` is valid even if absent.
Per-output visibility inherits the source unless overridden, for example
`make(pub(crate) _*, pub _)`. Inline controls use `inline(never)` inside `make`,
without attribute brackets. Attachment grammar and defaults remain subject to
consumer validation before release.

`attuned!(work(args), [_v3, _scalar])` selects a covered direct function inside
a feature context, and uses proof entries in ordinary code. It does not attempt
an implicit upgrade from a known context. `reattune!` explicitly permits runtime
reselection. Explicit lists never acquire an implicit fallback: cfg/architecture
filtering must leave a guaranteed covered candidate, or compilation fails.
An explicit `_v3_t` selector requests the proof interface even from a covered
context, deriving proof with `from_context()` there.

The supplied-proof spelling is `using(token)` only. Its expression is
evaluated once, and selection does not probe the CPU. Naming dictionaries key
on both tier and interface: `names(_v3 = work_avx2, _v3_t = work_avx2_t)`.
Definition names are identifiers; call-site names may be paths.

Expansion stays function-local. Bodies remain token streams; the macro does not
parse or search an enclosing impl, trait or source file. Inherent methods may
generate siblings; receiverless associated functions say `in_impl`. A trait
family that requires siblings must move its kernel outside the impl. Existing
legacy nested wrappers retain their supported receiver transformations.

## Verification and outstanding work

The original `de7d28af833b` base passed `cargo xtask ci` on the available x86-64 host. That command
reported skipped ARM cross checks because cross tooling was unavailable; it is
not evidence of an ARM pass. Its resource report was:
`done rc=0 230s | peak-RSS 1.19GiB | min-avail 20418MiB | peak-load 7.26`.

The initial unified contracts exercise const generics, tuple destructuring,
methods, associated functions, renamed calls, move-only arguments, explicit
boundaries and proof selection under `forbid(unsafe_code)` and `deny(warnings)`.
These cases do not constitute the complete coverage matrix.

Release gates still include complete frontend consolidation, selector and
attribute-routing rejection coverage, cfg combinations, generic trait methods,
cross-crate code generation, cross-platform compilation, exact migration edits
and warnings, and matched cold/incremental compile-time comparisons. Do not
enable deprecations that suggest replacements until those replacements and the
source converter are validated. Do not describe the current implementation as
the complete rewrite while any of these gates remains open.

The repeatable cold compile harness is `benchmarks/attune_compile.py`;
the fixed baseline is recorded in
[baseline.json](../benchmarks/attune_compile_2026-10-07/baseline.json).
It uses the unchanged downstream compile-cost fixture, fresh Cargo target
directories, and the same lockfile for subsequent comparisons. No compiler
wrapper was configured in the measured environment. These are cold Cargo
builds with warm OS caches, not cold-disk measurements.

## PR #123 integration

The shared boundary emitter preserves the upstream parameter-shadowing
rejection before its unsafe forwarding call. Generated `attune` proof entries
use the same check. A parameter cannot stand in for the generated callee.
Pattern normalization happens before nested-call rewriting, so wildcard proof
parameters have their final forwarding names before a call uses them.

Nested receivers use the common receiver lowering, including explicit
lifetimes and typed receivers such as `self: Box<Self>`. The upstream
`Self` substitutions, nested-impl boundaries, token-position forwarding,
and gated legacy dispatch logic remain in their shared helpers and legacy
frontends. The upstream expansion snapshots remain the compatibility oracle.

Cold-build results recorded against the original base are historical data;
performance acceptance requires a matched comparison against the new base.

The first matched PR #123 comparison is recorded in
[pr123_comparison.json](../benchmarks/attune_compile_2026-10-07/pr123_comparison.json).
Six alternating pairs used identical fixture sources and lockfiles, fresh Cargo
outputs, and warm OS caches. Median cold checks were 1.754408 s baseline versus
1.864871 s rewrite (macros only), and 3.229452 s versus 3.353086 s (magetypes plus
AVX-512). Consumer-only rechecks were 0.039499 s versus 0.039606 s, and 0.042988 s
versus 0.043321 s respectively. These measurements do **not** satisfy the
no-regression requirement. They cover this fixture, not the entire consumer
fleet or migrated applications.

Resource report: `done rc=0 62s | peak-RSS 0.37GiB | min-avail 20683MiB |
peak-load 6.03`. Run `just attune-compare BASE CANDIDATE OUTPUT` to repeat with
pinned revisions. Full Cargo and `/usr/bin/time -v` logs remain in the named
output directory.

Rebase validation on the available host passed the unchanged legacy expansion
snapshots, proof-boundary rejection fixtures, new runtime contracts, macro
clippy with all features and warnings denied, registry regeneration, and
`cargo test -p archmage -p magetypes --features 'std avx512'` including doctests.
The latter reported `done rc=0 112s | peak-RSS 2.20GiB | min-avail 19487MiB |
peak-load 6.00`. The new contracts also compiled for AArch64, WASM32 and i686;
these checks did not run on those targets.

The macro unit suite still has one intermediate-expansion assertion to adapt:
`variant_replacement_keeps_the_token_position` expects an anonymous token
parameter before boundary expansion, while direct lowering returns the final
wrapper with a named forwarding parameter. The corresponding compiler-facing
PR #123 token-position snapshot passes unchanged. This outstanding unit
assertion and the measured cold-build regression keep the full acceptance
gates open.

The larger unchanged-consumer comparison against PR #123 and crates.io 0.9.29
is recorded in [the cold-build report](../benchmarks/consumer_compile_2026-10-07.md).
It covers magetypes, zenav1-svt and rav1d-safe with three cold checks and three
release library builds per stack. It also separates standalone allocation
counts from the cost of compiling the macro crate itself. The no-regression
gate remains open; sub-percent codec deltas overlap observed run variation.

The follow-up [optimization report](../benchmarks/macro_optimization_2026-10-07.md)
records shared-emitter and substitution changes, allocation measurements, and
matched builds including the prior rewrite. The changes keep proof checks and
boundary placement explicit; reduced allocation counts alone do not satisfy
the cold-compile gate.

## Remaining decisions after the main rebase

The current review separates the legacy 0.9.30 release from the unpublished
0.9.31 rewrite. Do not describe a draft limitation as a regression in main.
The independent expansion audit pins main at `52bf060d`; audit findings and
coverage must name their source revision.

The agreed core remains: explicit AVX-512 additions, exact call lists with a
guaranteed fallback, `using(token)`, `wrap`, per-output visibility and inline
controls, explicit lists for sparse families, and function-local expansion.
These do not need another naming vote.

After rebasing, the macro unit suite passes with 129 tests passed, no failures
and one existing ignored allocation-profile test. The forwarding regression
now checks the normalized wrapper argument, inner signature, forwarding
call and dispatcher token position together. Main's and the draft's legacy
expansion suites also pass; the independent expansion audit nevertheless found
legacy safety and attribute-routing defects. Snapshot preservation is not a
correctness proof, and those findings remain release work.
See the [independent expansion audit](audits/EXPANSION_AUDIT_2026-10-07.md)
and its [per-file inventory](audits/EXPANSION_COVERAGE_2026-10-07.md).

The remaining policy questions are narrower:

| Question | Current behavior / recommended first scope |
|---|---|
| Final inline defaults and family-level attributes | Direct bodies default to inline, thin proof wrappers to always-inline, dispatcher unforced. Keep per-output overrides; verify the complete cross-crate matrix and reject ambiguous family-wide overrides instead of guessing. |
| Cold-build acceptance | The previous-base measurements still show a cold macro-crate cost. Rebase alone is not evidence it disappeared. Keep the no-regression gate unless the user explicitly revises it. |
| Method-call expressions | The call parser currently accepts function paths, not `attuned!(self.work(x))`. Either support and test receiver evaluation/borrowing, or make path-only support an explicit first-release restriction. Do not inspect the enclosing impl. |
| Different gates for direct/proof outputs of one tier | Currently rejected. Keeping a shared gate initially is simpler; independent gates require correctly guarding a shared body and every reference. |

Other open items are implementation and verification work, not preference
questions: duplicate/unused rename mappings, generated-name collisions,
portable dispatcher fallback under gated scalar outputs, full attribute
routing (`track_caller`, `expect`, `cfg_attr`, linkage and other proc macros),
complete generic/trait coverage, and a validated migration converter with
exact replacement text. Existing legacy names must remain supported while
those replacements are incomplete.

### Legacy contracts the migration must preserve

| Existing spelling | Observable contract to preserve |
|---|---|
| `#[arcane]` | Existing proof-taking name is callable from ordinary code; wrapper establishes the feature boundary. |
| `#[rite(v3)]` | Written name is retained and requires a covering feature context. Supplying a token alone does not change Rust's caller context. |
| Multi-tier `#[rite(v3, scalar)]` | Generates suffixed direct names; a scalar/default body has no target-feature requirement. |
| `#[magetypes(...)]` versus `#[magetypes(rite, ...)]` | Same family naming convention, but proof wrappers versus direct feature functions. This distinction affects callers, not just implementation style. |
| `#[autoversion]` | Public dispatcher plus generated variants. No proof parameter, legacy `SimdToken`, and real `ScalarToken` have different dispatcher signatures. |
| `incant!(...)` | Ordinary dispatch, context rewriting, supplied-proof and `without token` forms have distinct selection/signature rules. `without token` selects the caller's exact tier and accepts no tier list. |
| `scalar` versus `default` | Both are featureless fallbacks, but token arguments and generated function names differ. |

Tier defaults, provider feature gates, implicit inline attributes and helper
visibility are also part of compatibility. A from/to converter must make them
explicit where new defaults differ. These inconsistencies motivate the unified
model; they are not by themselves evidence that an expansion is unsound.

In particular, the legacy tokenful nested-call rewriter partitions candidates
into covered tiers and runtime upgrades. Replacing that call with `attuned!`
would remove the upgrade behavior; an exact migration may need `reattune!`.
The legacy `without token` form instead selects the exact caller tier. Check
the enclosing attribute and call modifiers together when generating an edit.


## Method calls, migration suggestions and inline policy review (2026-10-07)

These are recommendations and implementation requirements, not a claim that the
current path-only parser already accepts method calls or that dispatcher defaults
have changed.

Support `attuned!(receiver.work(args), [_v3, _scalar])` and the corresponding
`reattune!` form. Preserve an ordinary method-call expression in each selected
branch, changing the method name and inserting proof only when required. Do not
move the receiver into a local or eagerly take `&mut receiver`: either can change
ownership or implicit borrowing. Rust resolves receiver types, autoderef/autoref,
traits and generic arguments; no enclosing-impl scan is necessary. Method rename
mappings should be method identifiers; qualified function paths belong to the
function-call form. Existing restrictions on generating direct trait methods
still apply independently of call syntax.

A standalone lowering probe, [method_lowering.rs](../tests/expansion_audit/method_lowering.rs),
compiled and ran on Rust 1.98.1. Both branch choices preserve `v.push(v.len())`,
a boxed vector's implicit borrow, and one evaluation of a consuming receiver
before its argument. This tests the proposed branch shape, not an implemented
`attuned!` method frontend. Full integration still needs renamed methods,
turbofish, proof position, receiver temporaries, borrowing returns and nested
context calls.

The migration helper should reuse the legacy resolved selection plan. Ordinary
calls and covered-only composition can suggest `attuned!`; calls that actually
perform runtime reselection need `reattune!`. Preserve gates, names, proof
position and evaluation order. A standalone invocation macro does not know its
enclosing attribute; the attribute's contextual rewrite or source converter
must supply that information. Do not present a replacement as exact when the
context or selection equivalence is unproved.

There is a concrete ordering caveat in the current draft. The legacy rewriter
emits all runtime-upgrade arms before covered arms. For a V4 caller with callee
list `[v4, v3_crypto, scalar]`, it probes V3Crypto before calling covered V4.
V4 does not cover V3Crypto. The new call emitter sorts all candidates by priority
and stops at covered V4 (priority 40, versus V3Crypto's 35). Therefore simply
renaming that invocation to `reattune!` would change selection. This conclusion
comes from `rewrite.rs`, `attune/call.rs` and the generated registry; it is not a
runtime measurement. The converter must preserve this plan explicitly or report
that a one-invocation replacement is unavailable.

Keep one Cargo gate per tier shared by its direct body and proof interface.
Independent gates could package an optional public proof API around an always
available direct kernel, but no consumer requirement for that flexibility has
been established here. It adds conditional body/reference routing; it does not
improve feature safety. The current parser's shared-gate restriction is suitable
for the first implementation.

Proposed default policy, reflecting the user's preference:

| Generated function | Default | Override |
|---|---|---|
| Proof wrapper | `#[inline(always)]` | Per-output `inline(hint)` or `inline(never)` |
| Dispatcher | `#[inline(always)]` | Per-output `inline(hint)` or `inline(never)` |
| Operation body, including scalar | `#[inline]` | Per-output `inline(never)`; native feature bodies reject always |

Keep the common spelling short, and attach exceptions to the selected output:

```rust,ignore
#[attune(make(all))]                                  // proposed defaults
#[attune(make(_*, _*_t, inline(never) _))]             // shared dispatcher
#[attune(make(inline(never) _*, _*_t, _))]             // out-of-line bodies
```

The override spellings parse today; default dispatchers currently have no
implicit inline attribute. Forcing dispatcher inlining can duplicate selection
logic across callers, so this preference still needs the agreed multi-repository
codegen and compile-time checks. Keep scalar fallback as a separate operation
body rather than source text embedded in the dispatcher. Covered `attuned!` calls
should bypass dispatch and proof wrappers entirely.

Rust 1.98.1 rejects `#[target_feature]` combined with `#[inline(always)]`.
[inline_always_target_feature.rs](../tests/expansion_audit/inline_always_target_feature.rs)
was compiled as a negative probe and exited 1 with that diagnostic. The current
[Rust codegen reference](https://doc.rust-lang.org/reference/attributes/codegen.html)
also documents this restriction and describes inline attributes as hints.
[Method lookup](https://doc.rust-lang.org/reference/expressions/method-call-expr.html)
performs the receiver's automatic dereference/borrow adjustments.

Probe command: `rustc --edition=2024 <source> -o <output>` for each source;
run only the successfully compiled method probe. Sources were identical to
those preserved here. Full log: `/home/lilith/tmp/attune-method-inline/results.log`.
The serialized run-heavy wrapper checked each expected exit status and reported
`rc=0 0s | peak-RSS 0.11GiB | min-avail 21297MiB | peak-load 0.24`.
No runtime-performance or end-to-end compile-time improvement is claimed.

## Explicit body policy investigation (2026-10-08)

The user clarified that `no_inline` is a proposed escape hatch for legacy
attributes, and asked whether attune should require an explicit choice. No
parser or default changes were made in this investigation. The earlier default
table remains a proposal, not a settled or implemented policy.

The recommendation under investigation is one explicit body policy per attune
definition, inherited by its generated direct bodies. Proof wrappers and
central dispatchers are separate outputs with their own defaults/overrides;
users should not have to repeat the body choice for every tier. For example,
`#[attune(inline(hint), make(all))]` is proposed syntax, not syntax accepted by
the current parser. An explicit policy emitting no attribute must remain
separate from `inline(never)`. The former leaves the optimizer free to inline;
the latter requests out-of-line code.

For migration, legacy body hints must become an explicit hint, and legacy proof
wrappers must retain their effective policy. Whether legacy
`#[arcane(no_inline)]` suppresses only the body hint or also the proof wrapper's
always-inline attribute is still a separate naming/behavior decision.

The [real-consumer experiment](../experiments/inline-real/README.md) changes one
emitted layer at a time in fixed-main source copies. Its
[results](../benchmarks/inline_real_2026-10-08/README.md) cover zenav1-svt encoding
and rav1d-safe decoding, with output parity checks. This measures legacy emitter
policies; it does not establish the performance of the complete attune rewrite
or close that rewrite's compile-time acceptance gate.

## Visibility-based inline policy candidate (2026-10-09)

`inline(default)` could explicitly request an archmage policy that emits a body
hint for unrestricted `pub`, and no body attribute for restricted visibility or
an ordinary private function. This is a candidate, not an implemented default.
Resolve it after each generated output's visibility override, not solely from
the input function's visibility. Keep proof-wrapper policy separate.

Syntactic visibility is only a heuristic. Re-exporting a public type does not
remove `pub` from its public inherent methods, and a public re-export cannot
promote a crate-private item. Trait methods are the important ambiguity: they
have no explicit `pub`, yet can form an externally callable API. A function-only
attribute must not interpret every inherited-visibility method as private or
scan the enclosing impl to resolve this. An explicit hint remains the clear
choice when the enclosing trait context is unavailable. These visibility rules
are described in the [Rust Reference](https://doc.rust-lang.org/reference/visibility-and-privacy.html).

Cross-crate inlining does not absolutely require an inline attribute. Type/const
generic instantiations already make bodies available; Rust also has automatic
cross-crate eligibility for some small functions, and LTO is another route.
For ordinary non-generic library functions, an explicit hint is the predictable
way to make bodies available without relying on those mechanisms. See the
[inline documentation](https://dev-doc.rust-lang.org/stable/core/attribute.inline.html),
[current compiler eligibility implementation](https://doc.rust-lang.org/stable/nightly-rustc/src/rustc_mir_transform/cross_crate_inline.rs.html),
and [LTO options](https://doc.rust-lang.org/rustc/codegen-options/index.html#lto).
Availability still does not guarantee actual inlining.

A visibility change would also change codegen policy under this rule. Internal
hot helpers may benefit from hints regardless of visibility, and helpers called
by exported inline bodies need consideration too. The recorded real-consumer
experiment removed body hints across visibilities; it did not test this mixed
policy. Do not claim those results establish the performance of
`inline(default)`. Measure the visibility-based variant before adopting it as
the recommended default; preserve explicit legacy hints during migration.

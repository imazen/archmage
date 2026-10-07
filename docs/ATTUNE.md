# Unified macros: implementation contract

This is an unpublished rewrite based on `de7d28af833b`. Keep it on the
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

The current supplied-proof spelling is `using(token)`. Its expression is
evaluated once, and selection does not probe the CPU. Naming dictionaries key
on both tier and interface: `names(_v3 = work_avx2, _v3_t = work_avx2_t)`.
Definition names are identifiers; call-site names may be paths.

Expansion stays function-local. Bodies remain token streams; the macro does not
parse or search an enclosing impl, trait or source file. Inherent methods may
generate siblings; receiverless associated functions say `in_impl`. A trait
family that requires siblings must move its kernel outside the impl. Existing
legacy nested wrappers retain their supported receiver transformations.

## Verification and outstanding work

The base passed `cargo xtask ci` on the available x86-64 host. That command
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

# Attune raw expansion audit

This corpus checks definition syntax, invocation selection, and placement across
x86-64, AArch64, WebAssembly, and i686, with Cargo gates disabled and enabled.
It is a finite set of products and targeted cases, not every possible Rust program.

Run from the workspace root under the resource limiter used for builds:

```sh
just attune-expanded-generator-test
just attune-expanded-raw /absolute/path/to/new-artifact-directory --keep-going
```

The output directory must not already exist. `--arches`, `--gates`, and `--groups`
select explicit subsets; `--generate-only` writes inputs without claiming they
passed. `--keep-going` collects every configuration and still exits unsuccessfully
if any check fails. Python 3.11+, cargo-expand, and the selected Rust targets must
be installed. Missing tools or targets fail the run.

## Coverage

| Product | Axes |
| --- | --- |
| Definition policies | Six output forms × flat/grouped/positional spelling × four visibility choices × six body inline choices |
| Selector policies | Six output forms × flat/grouped spelling × five inline choices × inherited/overridden body policy |
| Definition modifiers | Six output forms × flat/grouped spelling × eight addition/removal sequences |
| Definition placement | Six output forms × free/generic/inherent receiver/inherent associated/trait placement |
| Invocation selection | Applicable caller contexts × `attuned!`/`reattune!` × eleven tier lists × inferred/explicit scalar proof |
| Ordinary attributes | Four output forms × seven attributes |
| Registry coverage | Every registry entry with dispatch priority; direct body, proof wrapper, dispatcher, and same-tier callers |

Output forms are direct, proof, both, dispatcher, all, and a sparse V3/scalar
family. Modifiers include `-_neon`, repeated removal, absent `-_v4`, proof-only
removal, gated V4x addition, and removal before/after addition. ARM raw output is
checked for the absence of removed NEON names and token types.

Caller contexts include ordinary functions, scalar and native feature contexts,
lower/higher x86 tiers, generic/where/impl-trait proofs, borrowed/concrete/ambiguous
parent proofs, inferred suffixes, explicit wrappers, inherent methods, associated
functions, trait implementations, receiverless trait defaults, and legacy
`arcane`, `rite`, `autoversion`, and `magetypes` contexts. A manually written
`target_feature` attribute does not provide enclosing signature information to a
function-like macro and is checked as an unknown calling context.

Targeted cases also cover qualified and associated paths, turbofish, closures,
nested functions and invocations, all macro delimiters, rename dictionaries,
Token marker positions, `from_context()`, ordinary attributes, parameter patterns,
move-only values, lifetimes, const generics, and where clauses.

## Rejections and nesting

Rejected cases retain their inputs, diagnostics, and raw expansions. They cover
missing guaranteed fallbacks, ambiguous parent proofs, invalid options, conflicting
gates, unsupported method-call syntax, and definition modifiers used as call lists.
`-_neon` is supported on definitions; it is not currently a call-list modifier.

Both legacy `arcane` and new `attune(wrap)` accept `nested`/`in_trait`; `_self = Type`
also implies nesting. Concrete trait-implementation receivers are passing cases.
Default trait methods with receivers and no `_self` remain rejection cases; the
pre-rewrite implementation at `de7d28af833b` already imposed this restriction.
Receiverless default trait methods exercise successful nested wrappers separately.

A token parameter alone does not infer `attune(wrap)`. A `_tier_t` suffix infers
a proof wrapper; `_tier` infers a direct feature body even with a token parameter.
The corpus checks both this distinction and the missing-context diagnostic.

## Artifacts and checks

Each target/gate directory contains a `README.md` linking every input to its raw
module, a machine-readable `cases.json`, full raw output, per-phase commands and
logs, and diagnostic/inspection findings. The root `results.json` records the
revision and compiler; `artifacts.json` records artifact sizes and SHA-256 hashes.

The runner performs these checks in order:

1. Compile positive inputs with `forbid(unsafe_code)` and warnings denied.
2. Save `cargo expand --ugly` output byte-for-byte and extract individual modules.
3. Check removed symbols and CPU-probe selection where a single caller body makes
   that assertion meaningful. Multi-body families are compiled and captured but
   are not checked with the single-body probe assertion.
4. Compile the standalone raw output, preserving Rust type checking.
5. Compile rejected inputs and require each case's expected diagnostic.
6. Save and extract rejected raw output, including compiler error placeholders.

Raw replay uses `RUSTC_BOOTSTRAP=1` for compiler-generated prelude attributes and
a root-crate `--cap-lints=allow`: serialization loses macro hygiene that allowed
generated unsafe trampoline blocks under source `forbid(unsafe_code)`. The cap is
not applied to dependency builds or Cargo target probing. Source compilation is
the check for the user's unsafe-code policy; raw replay is an additional type
check, not a substitute.

These are cross-target compilation checks, not ARM/WASM/i686 runtime tests or
runtime performance measurements. Existing runtime and legacy snapshot tests
remain separate. There is no claim of arbitrary receiver types, every combination
of Rust attributes, or the full Cartesian product across all independent groups.

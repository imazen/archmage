# Attune decision evidence — 2026-10-07

Missing: this is not an attune implementation or a general receiver transformer.
The trait and call-policy probes are handwritten candidate lowerings using real
archmage attributes. Cross-crate namespace probes use integer marker functions,
not SIMD. No cold-compile comparison, runtime benchmark, ARM or WASM measurement
was made. V4 was unavailable on the execution host, so the V3-to-V4 runtime case
exercised its fallback; successful upgrading was exercised from V2 to V3.

The probes address the questions in [DECISIONS.md](DECISIONS.md), alongside the
[draft specification](../../ATTUNE_SPEC.md). Recommendations there are not user
decisions and do not change the published API.

## Reproduction

Harness commit: `1968373d`. Library/macro implementation baseline:
`cf07592212e96294ef9ca5dca9a364fa8d15d8ad`.
Rust 1.98.1, LLVM 22.1.8, x86_64-unknown-linux-gnu; Intel Core Ultra 7 265K.

Run `just run /absolute/fresh/output/path` here, or invoke `python3 run.py --out
/absolute/fresh/output/path` under the host's resource-limited build wrapper.
Each compiler/runtime command writes a complete log. Existing output directories
are rejected. The runner preserves generated fixtures and a dependency lockfile.
It does not edit archmage sources, existing tests or expectations.

The final run returned 0 with all **41/41 expected command outcomes**. These
include intentional compiler failures with required diagnostics, runtime checks,
formatting and warning-free clippy for the trait/call-policy crate. This is an
enumerated probe set, not complete attune coverage. See [results.csv](results.csv)
and [run.log](run.log).

Resource record for the full probe command:
`done rc=0 6s | peak-RSS 0.30GiB | min-avail 22225MiB | peak-load 0.21`.
The duration is a harness run, not a cold-compile or runtime performance claim.

## Evidence A: cross-crate names and cfg ownership

[provider.rs](provider.rs) and [consumer.rs](consumer.rs) test two possible
descriptors:

1. A function and re-exported helper macro with the same public spelling, reached
   as `work(...)` and `work!(...)`.
2. A function and empty enum with the same spelling, reached as `work(...)` and
   `work::selected(...)`. The enum's methods can carry selection behavior.

Both survived a provider re-export, a consumer alias, and renaming the dependency.
The type descriptor also forwarded ordinary type/const generic arguments.
All four provider/consumer `fast`-flag combinations compiled and ran correctly
when selection was resolved in the provider.

The deliberately incorrect macro that emits `#[cfg(feature = "fast")]` into
its expansion demonstrates the danger of consumer-side evaluation:

| Provider fast | Consumer fast | Result of incorrect macro |
|---|---|---|
| disabled | enabled | E0425: expansion references a provider function that does not exist |
| enabled | disabled | Executes baseline marker 13 although provider-selected result is 14 |

These flag combinations are passed separately to rustc for each crate. The
tests establish cfg evaluation scope; they are not a Cargo resolver benchmark.

Additional required diagnostics:

- Two exported helper macros with the same root name: E0428.
- Publicly re-exporting a local macro without macro_export: E0364.
- Generating a descriptor type where that type name already exists: E0428.
- Declaring an enum or macro_rules definition inside an impl: rejected.

Consequences: neither prototype is a universal, zero-cost API mechanism. The
macro prototype assigns its unique root export name manually. The type
prototype adds a public type namespace item, even if doc-hidden. Associated
families need a different legal placement/protocol. Neither probe implements
automatic sparse-family discovery, version interoperability or grouping of
independently annotated handwritten variants.

Do not ask the user to choose macro versus enum as though that decision were
ready. First choose the desired call-site contract; then test the complete
descriptor protocol against that contract.

## Evidence B: generic and trait adapters

[traits.rs](traits.rs) compiled and ran these real feature-enabled lowerings:

| Shape | Observed result |
|---|---|
| Generic Processor with lifetime, type and const parameters; mutable receiver; associated array output | Correct SIMD result and mutation through dyn Kernel with fixed associated output |
| Nested helper with explicitly redeclared enclosing generics and shared return/input lifetime | Borrowed label returned correctly |
| Default trait body using a helper generic over S: Trait + ?Sized | Correct SIMD result through dyn Trait |
| Foreign trait implemented for Vec using a local marker parameter | Legal implementation; free SIMD helper executes correctly |
| Existing method returning Self under an existing Self: Sized bound | Signature/behavior retained through concrete helper |

Counterexamples:

- A nested function implicitly using its enclosing T fails E0401.
- A free helper returning `P::Output` without trait qualification fails E0223.
- Adding an inherent helper impl for Vec fails E0116.
- A receiverless associated wrapper calling bare `direct(x)` fails E0425;
  `Self::direct(x)` compiles and executes correctly.

Thus there is a concrete legal adapter strategy. Enclosing impl information is
valuable, and free helpers are necessary for some legal trait impls. This does
not establish a complete macro transformation for arbitrary Self paths, macros,
GATs, arbitrary receivers, other proc-macro orderings or every dyn interaction.

## Evidence C: explicit reselection and argument ownership

[call_policy.rs](call_policy.rs) uses actual V2/V3/V4 token types and archmage
boundaries. Each payload is move-only and counts drops; argument factories and
probe closures count calls.

| Candidate lowering | Probe calls | Argument evaluations | Drops | Result |
|---|---:|---:|---:|---|
| V2 context, successful V3 proof | 1 | 1 | 1 | V3 marker 13 |
| V2 context, unavailable V3 proof | 1 | 1 | 1 | V2 marker 12 |
| V3 context, covered V3 selection | 0 | 1 | 1 | V3 marker 13 |
| V3 context, real V4 availability check | 1 | 1 | 1 | V3 marker 13 on this host; V4 unavailable |

The controlled None result tests the fallback branch without forging proof.
The successful V3 proof is real. No fake token or user-written unsafe is used.
These are tests of the candidate control flow, not of an implemented call macro
or of detection overhead. Infallible V4-only selection still needs a product
decision about an absent fallback; Rust does not create that fallback for us.

## Evidence D: current default tiers and disabled configurations

[defaults.rs](defaults.rs) uses the existing main macros:

- Referring to autoversion's generated `_v4` entry succeeds without enabling
  an `avx512` feature on the fixture.
- Referring to magetypes' default-generated `_v4` entry in that configuration
  fails E0425.
- Both references succeed with the fixture's `avx512` feature enabled.
- An autoversion operation with its `cfg(simd_opt)` option still runs its
  fallback with the feature disabled, and runs with the feature enabled.
- An outer `#[cfg(feature = "simd_opt")]` removes the whole operation when
  disabled; trying to call it fails E0425.

The cases selecting the defaults package do not select the separate traits
package, which requests AVX-512 support for its V4 token probe. Package selection
and the fixture's own feature are explicit in every recorded command.

A unified wildcard policy cannot preserve both old defaults automatically.
Choosing provider-owned gates is independent of choosing their surface syntax.

## Evidence E: attributes and receiver scope

- track_caller on both levels of a two-function chain preserved the external
  call line (9). Annotating only the outer function reported the inner definition
  line (6) instead of the external call line (12).
- Copying `expect(unused_variables)` to a wrapper and body caused an unfulfilled
  expectation on the wrapper, rejected with deny(unfulfilled_lint_expectations).
  Keeping the expectation on the body alone compiled and ran.
- An ordinary receiver body containing a nested impl and a capturing closure
  compiled and ran. Blindly replacing every `self` identifier broke compilation.

These tests establish why attribute/receiver handling cannot be unrestricted
copy-and-replace. They do not implement the required scope-aware transformer.
The prior [assembly experiment](../cross-crate-inline/README.md) provides the
separate evidence for direct-body and thin-wrapper inline policies.

## Repository validation

`cargo xtask ci` passed after the harness commit. Miri, ARM cross-testing,
WASM cross-testing and ARM clippy were skipped by the runner because their
required tools were unavailable. This is local validation, not a full
cross-platform result. The full log is retained at
`~/tmp/archmage-attune-decisions-ci-2026-10-07.log`.

Resource record: `done rc=0 137s | peak-RSS 0.39GiB | min-avail 21530MiB |
peak-load 1.22`.

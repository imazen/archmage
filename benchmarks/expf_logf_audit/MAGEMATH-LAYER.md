# Focused Magemath ownership study — 2026-11-10

Magemath should own the configurable f32 powf, expf and logf families. Archmage
provides feature proofs; Magetypes provides vectors and arithmetic primitives.
The dependency is **magemath → magetypes → archmage**. Magetypes retains its old
implementations during deprecation and points users at Magemath; it cannot
forward into a crate that depends on it.

This is a focused migration proposal and executable unpublished prototype. It
is not a general libm replacement, a new global floating-point policy framework,
or an approval of the current expf/logf implementation. [PR148.md](PR148.md)
records the correctness work still needed. No production Magemath public API or
Magetypes deprecation was added by this review.

## Measured build cost

Three alternating, target-cold builds per layout and profile, Rust 1.99.1 on
Ryzen 9 9950X3D, two build jobs, ordinary target features, offline dependencies,
warm OS cache. Each layout has the same consumer and flags. Numbers below are
median wall seconds, with all three cold observations shown. These are build
observations, not zenbench promotion intervals.

| Layout | Debug | Release, no LTO | ThinLTO |
|---|---:|---:|---:|
| Math in Magetypes | **6.04** (6.04, 6.06, 6.04) | **4.21** (4.23, 4.21, 4.17) | **4.88** (4.88, 4.88, 4.78) |
| Thin Magemath re-export | **6.08** (6.08, 6.05, 6.14) | **4.19** (4.19, 4.19, 4.17) | **4.86** (4.92, 4.86, 4.79) |
| Magemath owns math; Magetypes keeps existing APIs | **6.14** (6.11, 6.14, 6.15) | **4.33** (4.41, 4.33, 4.30) | **4.96** (4.94, 4.98, 4.96) |

The re-export is only a lower-bound control. It is not the proposed ownership.
The owner layout deliberately retains the existing configurable implementations
alongside the legacy methods in Magetypes, so it includes transitional duplication.
Removing the deprecated implementations was not measured.

One no-change and one identical semantics-preserving math-edit rebuild per
layout/profile were also measured; these single observations have no confidence
interval. The edit changes the same expf bitcast expression into an equivalent
local block. In the owner layout it rebuilds Magemath and the consumer; the other
layouts rebuild Magetypes and its dependent consumer.

| Layout | Debug edit | Release edit | ThinLTO edit |
|---|---:|---:|---:|
| Math in Magetypes | 0.93 s | 2.51 s | 3.12 s |
| Thin re-export | 0.92 s | 2.47 s | 3.21 s |
| Actual math owner | 0.43 s | 1.40 s | 2.07 s |

No-change builds took 0.01 s, except direct ThinLTO at 0.02 s. This supports
separating math development from vector development; it is not a universal
compile-time guarantee for larger applications.

## Code generation, size and correctness

All 45 build/run checks produced the identical output digest
`332d7cb128d55bca10cd918adeceb150f6a7d1b0295827c17aee292c79dd77db`.
Each execution covers nine configured operations over 256 positive values in
ScalarToken and V3 callers, both Caller and Automatic selection. This includes
MIDP/LOWP/HIGHP powf, MIDP/LOWP exp2 and log2, MIDP log1p and explicit-token powf
lane mapping. It is sampled equivalence, not an exhaustive numerical proof.

Every corresponding inspected consumer/math function has the same instruction
and call counts in all three layouts: 33 functions / 19,240 instructions without
LTO, and 24 functions / 19,139 instructions with ThinLTO. These totals include
software-fusion and HIGHP work. Counts alone do not prove equal timing or
byte-identical assembly. This study did not time the prototype's runtime.

| Layout | Debug executable bytes | Release bytes | ThinLTO bytes |
|---|---:|---:|---:|
| Direct | 18,543,816 | 566,704 | 529,312 |
| Re-export | 18,543,832 | 566,712 | 529,368 |
| Actual owner | 19,089,432 | 566,672 | 529,368 |

These are whole ELF file sizes, including applicable debug information, from
repeat 1 cold builds. They are not resident-memory measurements. A separate
`cargo check --no-default-features --features w512` passes for the prototype
owner. The submitted PR148 itself still fails no_std: the prototype makes the
same no_std-only repair in all three controls.

Private-helper access is a real migration consideration. The controls all use
the same adapter through existing public fused vector primitives for private
powf fusion helpers. This isolates crate placement from an access/API redesign.
The owned module copies the WASM fusion probe and Arm license with their source.
It does not add a new public Magetypes helper. Native std comparisons above do
not establish cross-ISA or no_std arithmetic equivalence for that adapter.

Generic code is [monomorphized in consuming crates](https://rustc-dev-guide.rust-lang.org/backend/monomorph.html).
An extra crate therefore does not inherently create a runtime call for every
polynomial step. [Inlining remains a compiler decision](https://doc.rust-lang.org/reference/attributes/codegen.html#the-inline-attribute),
which is why the actual consumer was inspected with and without LTO.

## Proposed narrow API direction

Keep operation-specific const builders ending in `.bits()`, with the integer
carried through the caller, dispatch entry and typed vector kernel. Publish a
small set of named, documented profiles rather than every Cartesian product of
flags. Each profile states precision target, accepted input/result range,
exceptional handling, validation and whether backend rounding can differ.

Keep automatic whole-operation selection and proof-preserving typed-vector
kernels as entry-point choices. Obtain the feature proof once outside a hot
loop. An external crate cannot access the vectors' private token field; use an
explicit proof argument initially. A zero-cost accessor or extension ergonomics
would require a separately reviewed Magetypes API. Mapping scalar public calls
over lanes is not a substitute for a typed vector kernel; the policy benchmarks
measure substantial overhead for that approach.

Retain the measured portability choice: fused canonical evaluation versus
backend-dependent arithmetic, with native fused implementations internally
sharing code. Keep meaningful range/validation savings; remove public
legacy-fit selectors and implementation-only provider flags. Do not add
correct/native/portable/relaxed/unfused global policy types, new transcendental
families or unmeasured f64 profiles as part of this migration.

The existing powf canonical contracts and regression coverage remain in force.
Move the useful implementation and its scientific tests; do not change its
coefficients simply to share an expf/logf fit. New expf/logf policies need the
corrections in [PR148.md](PR148.md) before becoming replacement APIs.

## Reproduction and resource record

`layer_prepare.py REVISION OUTPUT` creates the three layouts from frozen source
`26eb088b3189c22cd7d8e29ef1ce05ba945a788f`. `layer_build_cost.py SOURCES OUTPUT`
performs the builds; `layer_codegen.py BUILDS` preserves disassembly and counts.
The measured preparation used tools at `f581cb7e`; later formatting and executable
preservation improve replay without changing the measured prepared sources.
The first failed pilot and its error log are retained alongside the successful
second preparation. The measured run predates per-stage executable copying:
all stage output digests/logs and 27 final cold/edit executables are retained,
not a separate binary for every overwritten repeat-0 stage.

Resource wrapper: `run-heavy: done rc=0 154s | peak-RSS 0.95GiB | min-avail
38238MiB | peak-load 2.37`. Individual GNU-time peak RSS records are preserved
for every build; the wrapper's peak covers the complete process tree.
Artifact identity: [PR148-ARTIFACTS.md](PR148-ARTIFACTS.md).

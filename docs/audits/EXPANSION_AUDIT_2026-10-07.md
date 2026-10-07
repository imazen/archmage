# Independent expansion audit — 2026-10-07

## Missing coverage first

This is not a complete consumer or release certification. All **161/161 checked-in
`.expanded.rs` outputs were read** and inventoried, but the following remain missing:

- The magetypes snapshot/compile harness was **run independently by the parent**,
  not this auditor: default and avx512 each passed all three harness tests.
  Its nine outputs were source-reviewed here. Draft magetypes remains untested
  in this audit. The parent log and resource line are recorded below.
- Full consumer-body expansion: **none generated or reviewed** for the archived
  zenav1-svt/rav1d-safe consumers. Harvested signatures are explicitly not a
  substitute. The 233 harvested shape modules use placeholder types and stub
  bodies, and their generator discards distinctions described below.
- Most generated harvested output was not individually read. Four target/feature
  configurations were expanded successfully; selected regions were inspected.
  The coverage inventory states which regions, and records lexical counts only.
- Repository macro uses were inventoried in 324 source files (4,088 lexical
  matches, including possible comments/templates). No exhaustive expansion of
  all examples, tests, or magetypes implementation bodies was performed.
- No full `just ci`, Miri, runtime ARM/WASM execution, MSRV matrix, renamed-Cargo-
  dependency test, or exhaustive attribute/macro-stacking matrix was run.
- Draft AVX-512 snapshots, draft harvested output, and a full new-attune API audit
  were not run. The draft comparison concentrates on legacy lowering. Parent
  unit-test results are independent evidence and are not counted as audit runs.

No existing sources, documentation claims, assertions, or snapshots were edited.
New reproducers live in `tests/expansion_audit/`; they are standalone binaries,
not wired into CI. No UB reproducer was executed. No fix or publication is made.

## Pins and outcome

| Source | Pin | Evidence |
|---|---|---|
| Main, release candidate 0.9.30 | `52bf060d` | clean isolated workspace parent verified |
| Pre-PR123 first parent | `c6e1bc6a4df93311b13be2edfc0c5971aa3bad91` | immutable Git archive |
| Published 0.9.29 | `b04846c743b313a3539119a0a29f381fd39a482e` | downloaded crate archives and `.cargo_vcs_info.json` |
| Rebased unpublished draft | `dd58ba59` | immutable Git archive |

**Release recommendation: address or explicitly adjudicate the unsafe-input
sibling finding before release.** It predates PR123 and exists in published
0.9.29; it is not a regression introduced by PR123 or the draft. Two separately
confirmed attribute issues affect behavior/compilation, not demonstrated memory
safety. Deliberate token forgery is a documented accepted limitation, not a new
contract regression and not classified as an automatic release blocker.

## Main findings

### A1 — unsafe input produces a safe sibling (high, soundness boundary)

Input: `tests/expansion_audit/unsafe_sibling.rs:4–12`. An `unsafe fn` reading a
raw pointer becomes a private **safe** target-feature sibling. A second safe
`#[arcane]` function with a genuine token calls that sibling with a null pointer
without an unsafe block. Compilation succeeds. The pointer read is never run.

The checked-in evidence already exposes the issue:
`tests/expand/arcane/unsafe_fn.rs:3` is unsafe;
`tests/expand/arcane/unsafe_fn.expanded.rs:7` declares safe `__arcane_process`,
while line 11 preserves unsafe only on the wrapper. The emitter at
`archmage-macros/src/arcane.rs:459` spells an unconditional safe `fn`.
`tests/expand/autoversion/unsafe_fn.expanded.rs` has the same pattern through
its generated arcane variants.

The original function's non-feature safety precondition does not disappear
when CPU features are proved. `#[doc(hidden)]` and private visibility do not
stop same-module or descendant code from calling the sibling. This is not an
externally public symbol exploit: the scope of exposure is the module receiving
the expansion. Ordinary calls through the unsafe wrapper remain checked; rite's
unsafe snapshot preserves `unsafe fn`, and WASM's direct signature-preserving
path does not establish this native sibling defect.

Compile-only command (wrapped in run-heavy in this audit):

```sh
cargo check --manifest-path /home/lilith/tmp/archmage-expand-audit/probe/Cargo.toml \
  --bin unsafe_sibling
```

The initial main check compiled both safety reproducers together, exit 0.
The same sources compiled with `--bin unsafe_sibling --bin forged_token` against
published, pre-PR123, and draft manifests, each exit 0. Logs:
`probes.log`, `provenance-published.log`, `provenance-pre-pr123.log`,
`provenance-draft.log`. The published macro source independently contains the
unconditional safe sibling at lines 499 and 538. This is confirmed longstanding
behavior, not a PR123 regression. Suggested correction needs to distinguish
originally safe and originally unsafe inputs; no implementation change was made.
Existing guidance that generated siblings must always be safe needs qualification
for originally unsafe input. That documentation was not edited.

### A2 — track_caller stops at the generated boundary (medium, behavioral)

`tests/expansion_audit/attributes.rs:3–5` annotates the operation with
`#[track_caller]`. Calling it on line 12 returns line **5**, not **12**.
The assertion at line 14 fails. `arcane.rs:401–404` copies ordinary attributes to
the wrapper but only lint controls to the sibling; the operation's
`Location::caller()` runs in the unannotated sibling.

```sh
cargo run --manifest-path /home/lilith/tmp/archmage-expand-audit/probe/Cargo.toml \
  --bin attributes
```

**Individual result: FAIL**, confirmed assertion panic in `behavior.log`.
Its exit code was not separately captured in that first batch. The batch then
ran a successful second binary and returned 0; that is not a passing result for
this check. No safety consequence demonstrated. Legacy draft boundary source
preserves the same attribute split (`engine/boundary.rs:351–352,429–440`);
this audit did not rerun the behavioral binary on draft or 0.9.29.
Autoversion also moves user attrs only to the dispatcher (`autoversion.rs:272`),
so that related track_caller path deserves a dedicated behavioral test.

### A3 — duplicated expect becomes unfulfilled (medium, compile/lint behavior)

`tests/expansion_audit/expect_attribute.rs:4–6` is an ignored token argument under
`#[expect(unused_variables)]` with `deny(unfulfilled_lint_expectations)`.
The operation can fulfill the expectation, but its forwarding wrapper consumes
the argument and cannot. `filter_lint_attrs` includes `expect`, duplicating it.

```sh
cargo check --manifest-path /home/lilith/tmp/archmage-expand-audit/probe/Cargo.toml \
  --bin expect_attribute
```

**Individual result: exit 101**, `expect.log`: “this lint expectation is
unfulfilled” at input line 5. This is an attribute compatibility issue, not a
soundness finding. Draft legacy boundary preserves the duplicate. The new
attune family explicitly removes `expect` from forwarding wrappers/dispatcher;
its distinct policy must not be mistaken for a legacy fix. No draft/published
behavioral probe was run for this case.

### L1 — deliberately forged concrete token (documented limitation)

`tests/expansion_audit/forged_token.rs` defines a local `X64V3Token` with the
expected public associated constant of type `()`. It compiles under `forbid(unsafe_code)` and can enter an AVX2 context
without detection. **Compile only, never executed.** The result initially
received an overbroad blocker label in status.txt; that label was corrected.

There is **no independent sealed-trait or type-identity check on this concrete
path**. `token_discovery.rs:278–295` resolves the type name and supplies an empty
`tier_traits` vector. `common.rs:391–399` returns no trait assertion for that
case. `arcane.rs:23–35` merely references the tier-specific constant on the
signature's supplied type. The real token's constant initializer in
`src/tokens/generated/x86.rs:3759–3764` authenticates that real token's tag;
it does not authenticate another type defining a constant with the same name.

In contrast, generic/impl-trait token forms emit an absolute
`::archmage::Has...` constraint. Real `SimdToken` is sealed
(`src/tokens/mod.rs:34–71`), and the existing forged-trait regression checks
pass. That protection is not invoked for a name-recognized concrete token.

The source explicitly says “accidental-misuse check, not an anti-forgery
boundary.” Public safety docs exclude deliberate shadowing plus copying hidden
constants (`docs/site/content/archmage/concepts/safety.md:167–175`), and macro
docs describe the same limitation (`archmage-macros/src/lib.rs:224–236`).
The identical reproducer compiles on published 0.9.29 and pre-PR123, main, and
draft. It is an accepted contract limitation, not evidence of a new regression.
The opt-out `suppress_const_test` is likewise explicitly documented; no additional
unsafe probes were pursued.

## Ordinary correctness review

- **Feature sets and boundaries:** the source-read snapshots consistently put
  target features on native bodies, with token/trait checks in boundary wrappers;
  rite has no baseline-callable wrapper. V4 token and HasX64V4 trait lists differ
  intentionally: registry trait features do not include the token's crypto extras.
  The validation/soundness commands passed; no new feature-set mismatch found.
- **cfg:** wrong-architecture snapshot bodies disappear on x86. Explicit gated
  nested dispatch falls through to scalar when its gate is absent. V4 calls in
  the default incant snapshots are filtered unless the calling fixture enables
  avx512. Running the package harness with avx512 does not prove every generated
  dependent fixture defines that feature. The negative `#[cfg(any())]` function
  with unavailable names in `attributes.rs:6–8` compiled away successfully;
  the suspected ungated sibling there was disproved. Arbitrary cfg_attr stacks
  remain untested.
- **Generics and receivers:** reviewed type/const turbofish forwarding with
  inferred lifetimes, all represented inline/where-bound positions, token-first/
  middle/last forwarding, tuple/wildcard rebinding, mutable/owned/reference
  receivers, boxed and lifetime-bound trait receivers, and nested impl self.
  Their ordinary snapshots compiled. Ref-binding patterns, generated-name
  collisions beyond the explicit sibling guard, enclosing impl generics, and
  arbitrary attribute-macro ordering were not validated.
- **Evaluation order and move-only values:** `argument_order.rs` passed. Two
  side-effecting String arguments ran once in order `[1, 2]`; dispatch returned
  `ab`. A tokenless autoversioned function forwarded a move-only tuple plus a
  wildcard argument and returned `cd`. This is one concrete case, not every
  dispatch tier, panic/drop order, or passthrough expression combination.
- **Visibility and namespaces:** private generated bodies and public wrappers
  were reviewed; in_impl uses Self-qualified sibling calls. Magetypes aliases
  point through `::magetypes::simd::generic`. Generated code often names
  `archmage` directly; renamed dependencies and custom module shadowing were
  not tested. Successful standalone snapshot compilation covers the harness's
  dependency setup only.
- **Inline:** legacy wrappers stay inline(always), bodies/rite use inline;
  direct inline attributes are filtered. This is existing policy, not a claim
  that arbitrary caller inline policy is preserved. No assembly/performance
  equivalence measurement was made.

### Known-failure snapshots

All seven were read. Receiver-less associated functions without in_impl and
trait impls without the supported nested form are documented macro-context
restrictions, not new bugs. Rite on a safe trait method fails rustc's feature
compatibility rule. `should-fail/incant_passthrough.expanded.rs` fails standalone
recompilation because cargo-expand exposed unstable panic internals (E0658);
this does **not** establish that its original incant input is rejected. The
soundness harness explicitly checks that expanded artifact. Four associated/
sibling limitations have no independent should-fail compilation in the specific
soundness harness run; macrotest output comparison is not a rejection oracle.

## Draft comparison (separate from release findings)

Draft `dd58ba59` routes legacy autoversion/magetypes directly into shared boundary
and feature emitters instead of chained attribute expansion. Reviewed diffs:
`autoversion.rs`, `magetypes.rs`, `common.rs`, `rite.rs`, `rewrite.rs`, and the
new boundary/feature modules. Checked signature/attribute/visibility Token
specialization, generated imports, registry feature strings, cfg feature flow,
scalar/default no-feature emission, private variants, and legacy inline policy.

Draft archmage default snapshots and soundness regression suite pass against
unchanged checked-in expectations. A1 and L1 compile unchanged on the draft;
legacy A2/A3 attribute handling remains in source. The new attune family's
attribute handling differs intentionally (expect only on operations;
track_caller forwarded), but its tests were only source-read by this auditor.
No additional draft-only defect was confirmed. These checks do not establish
full draft API, consumer-body, cross-target, or performance equivalence.

## Commands and exact test scope

All heavy commands were serialized under `TMPDIR=/home/lilith/tmp` and
`run-heavy --mem 16G --jobs 8`. Rust was 1.98.1; cargo-expand 1.0.126.
No `MACROTEST=overwrite` was set. Artifact root:
`/home/lilith/tmp/archmage-expand-audit/`.

| Command/scope | Individual result | Log |
|---|---|---|
| `cargo run -p xtask -- generate`, `validate-registry`, `validate`, `soundness` | all pass; generation left tracked sources unchanged | health.log |
| `cargo test -p archmage --test macro_expand --test soundness_exploits -- --test-threads=1` | 4 snapshot/compile harness tests + 3 soundness harness tests pass | snapshots.log |
| `cargo test -p archmage --features avx512 --test macro_expand -- --test-threads=1` | 4 harness tests pass | snapshots-avx512.log |
| same default macro_expand + soundness command with draft manifest | 4 + 3 harness tests pass | draft-snapshots.log |
| main compile-only safety reproducers | pass | probes.log |
| published/pre-PR123/draft compile-only safety reproducers | each pass | provenance-{published,pre-pr123,draft}.log |
| attributes binary | assertion FAIL, expected 12 / observed 5; individual exit code not separately recorded | behavior.log |
| argument_order binary | PASS | behavior.log |
| expect_attribute check | FAIL, exit 101 | expect.log |
| `cargo expand -p magetypes --test harvest_shapes` | succeeds | harvest.log |
| same with `--features avx512` | succeeds | harvest.log |
| same with `--target aarch64-unknown-linux-gnu` | succeeds, expansion only | harvest.log |
| same with `--target wasm32-wasip1` | succeeds, expansion only; compiler warnings retained | harvest.log |

Archmage harness coverage is **145 ordinary fixture pairs plus 7 known-failure
pairs**. The generic `*.rs` trybuild globs also match `.expanded.rs`, so raw
reported test invocations are not unique fixture counts. The all-files macrotest
also includes should-fail, which is compared again by its dedicated test.
Magetypes has **9 additional pairs**, source-reviewed here and independently tested by the parent on main default
and avx512; its existing compile harness lists define and rite_flag but omits
without_token, although the expansion glob includes it. The parent subsequently
compiled that omitted input/output pair separately; both passed on pinned main.

Health resource line: `rc=0 14s | peak-RSS 0.59GiB | min-avail 18785MiB | peak-load 3.22`.
Main default suite: `rc=0 103s | peak-RSS 0.36GiB | min-avail 20290MiB | peak-load 1.52`.
Main AVX-512 + draft suite batch: `rc=0 193s | peak-RSS 0.36GiB | min-avail 20260MiB | peak-load 2.52`.
Harvest expansion batch: `rc=0 13s | peak-RSS 0.76GiB | min-avail 20046MiB | peak-load 1.53`.
These are run-heavy's recorded resource lines, not compiler optimization claims.

Parent-supplied verification (independent run): main magetypes `macro_expand`
passed all three tests (input compile, expansion match, expanded-output compile),
both default and avx512. Log:
`/home/lilith/tmp/archmage-parent-audit/magetypes-snapshots.log`.
Resource line: `rc=0 32s | peak-RSS 0.39GiB | min-avail 20153MiB | peak-load 1.97`.
The auditor inspected the final log summaries; the parent ran the commands.

The parent also independently reproduced A1, A2, and A3 in a separate archive
of main `52bf060d`. A1 was compiled only (exit 0); A2 ran and failed with
expected line 12 / observed line 5 (exit 101); A3 failed compilation with an
unfulfilled expectation (exit 101). The move-only/order check passed (exit 0).
Each child exit code was checked separately. Log:
`/home/lilith/tmp/archmage-parent-audit/verification.log`.
Resource line: `rc=0 2s | peak-RSS 0.30GiB | min-avail 21207MiB | peak-load 0.16`.
The zero batch status means the observed results matched the expected findings,
not that the failing cases were correct.

The omitted magetypes `without_token/magetypes_body` input and expanded output
were checked as two standalone binaries using the same parent probe manifest.
Both passed; warnings were retained in
`/home/lilith/tmp/archmage-parent-audit/without-token.log`.
Resource line: `rc=0 0s | peak-RSS 0.08GiB | min-avail 21298MiB | peak-load 0.26`.
The parent independently matched every inventory row against `git ls-tree` at
the pinned main revision: all 161 output paths occur exactly once.

## Harvest limitations and learnings

The generator's `coarse_key` collapses runs of ordinary parameters and chooses
one shortest representative. It rewrites user-defined types to P, strips some
bounds/lifetime relationships/attributes, normalizes define lists, can synthesize
generic parameters and replace array lengths, omits bodies, and uses lexical
attribute recognition. The checked-in header says 40 source crates; this audit
did not reconstruct that original capture population. The generated file has
233 modules, which does not prove exhaustive coverage of those consumers.

Snapshot preservation proves stability, not correctness: A1 is plainly visible
in a passing snapshot. Attribute correctness needs operation versus forwarding
layer checks. A successful outer shell status is not a substitute for recording
each child status. Host-filtered output and stub signatures must remain separate
from actual non-x86 and consumer algorithm coverage.

## Artifacts and provenance

The two published archives were fetched directly from
[archmage 0.9.29](https://static.crates.io/crates/archmage/archmage-0.9.29.crate)
and [archmage-macros 0.9.29](https://static.crates.io/crates/archmage-macros/archmage-macros-0.9.29.crate).
Their SHA-256 values are respectively:

```text
8e3a17fc8196cce2902db52de7a21d4407971d080c01a0256037fcd3fc6e536e
1d0c10fb756a60ff9418d3e5dcd2b665c5914e32391d2523995d2c2c09237ac6
```

Full logs, downloaded packages, immutable archives, probe manifests and four
expanded harvest files remain outside git. `artifact-sha256.txt` records the
root artifact hashes; `published/provenance.json` records URLs and VCS IDs.
All 161 checked-in output bytes were independently compared to pinned main
Git objects and remain unchanged. Inventory:
`EXPANSION_COVERAGE_2026-10-07.md`; lexical source inventory:
`EXPANSION_MACRO_USES_2026-10-07.tsv`.

The heavy slot was released after all audit builds exited. The parent reviewed
and integrated these files into the unpublished draft, retaining the original
auditor report and logs outside git. Main's release sources and checked-in
snapshots were not changed. This report is evidence for follow-up fixes, not
release approval.

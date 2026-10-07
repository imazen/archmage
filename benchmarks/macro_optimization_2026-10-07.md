# Macro optimization measurements — 2026-10-07

The optimization keeps one shared boundary emitter and one identifier-substitution implementation. It removes temporary buffers, repeated token discovery, and unnecessary AST reparsing. It adds no global caches, syntax parser, or new public API.

Production source: `e20ffeb41b49c67dc6fdbd5411e1f4b49addbb5f` before, `87e8cd7e435ab5f7f74fc1d648a3fd185499e408` after. Both are unpublished rewrite snapshots on PR #123; this is not a release-to-release API comparison.

## Maintenance and safety

- Token discovery retains the parameter position across pattern normalization. The existing concrete-token, trait-bound and multiple-token checks remain in place.
- Forwarding borrows identifiers instead of allocating one token stream per argument. Nested and sibling wrappers use the same proof call and shadowing rejection.
- `specialize_syntax` retains parsed nodes that contain no `Token`. Magetypes specializes parsed headers and opaque bodies separately. Scalar/default output is emitted directly because no later feature emitter needs its AST.
- The substitution regression covers attributes, restricted visibility, bounds, nested generic arguments, return types, literal text and legacy dispatch markers. Existing compiler-facing expansion snapshots pass unchanged.
- Registry tier suffixes are borrowed; attributes are moved into their output instead of cloned. Raw attune without aliases avoids constructing a token path.
- Private proof wrappers and boundary placement remain unchanged. Removing those wrappers would change generated call structure and needs separate code-generation evidence.

## Standalone allocation probe

These are average allocation/reallocation calls and requested bytes per expansion, from the existing 1,000-iteration probe. The standalone proc_macro2 backend differs from rustc’s bridge. Requested bytes are not peak memory; single-probe nanosecond timings are retained in the JSON but are not presented as consumer speedups.

| Input | Allocation calls before → after | Change | Requested bytes before → after |
|---|---:|---:|---:|
| arcane | 132 → 125 | -5.3% | 10495 → 10163 |
| arcane-dispatch | 177 → 171 | -3.4% | 10863 → 10571 |
| rite | 88 → 88 | +0.0% | 8949 → 8949 |
| attune-raw | 88 → 81 | -8.0% | 9216 → 8726 |
| attune-wrap | 132 → 125 | -5.3% | 10527 → 10195 |
| attune-direct | 195 → 171 | -12.3% | 13228 → 11180 |
| attune-all | 756 → 585 | -22.6% | 51075 → 32539 |
| attune-compose | 913 → 742 | -18.7% | 61235 → 42699 |
| magetypes | 681 → 660 | -3.1% | 54708 → 47040 |
| autoversion | 652 → 631 | -3.2% | 37808 → 36796 |

[Raw allocation measurements](macro_allocations_optimized_2026-10-07.json). The PR #123 deferred family expansions are not used as an allocation denominator; these before/after versions both lower complete families.

```text
run-heavy: done rc=0 2s | peak-RSS 0.29GiB | min-avail 21537MiB | peak-load 0.50
run-heavy: done rc=0 2s | peak-RSS 0.23GiB | min-avail 21522MiB | peak-load 3.25
```

## Matched cold consumer builds

All 72 builds/checks passed: three repetitions of each workload, mode and source stack. The existing [consumer harness](consumer_compile.py) now accepts an optional `--previous` revision so the prior rewrite is measured in the same run. Order rotates between repetitions. Each build has a fresh target directory; downloads and OS caches are warm. Incremental compilation and compiler wrappers are disabled; no `target-cpu=native`. These are default-feature library builds, with each consumer’s original profiles, not final application links.

Stacks: crates.io 0.9.29; [PR #123](https://github.com/imazen/archmage/pull/123) at `d7f85a5bbdc11410524c557b46908fdfdccdd1bd`; prior rewrite `e20ffeb41b49c67dc6fdbd5411e1f4b49addbb5f`; optimized rewrite `87e8cd7e435ab5f7f74fc1d648a3fd185499e408`. Consumers and normalization match [the first comparison](consumer_compile_2026-10-07.md). The encoder/decoder Rust sources are unchanged and still use legacy attributes. This compares full dependency stacks; PR/rewrite include libm while published default magetypes does not.

Host i265, Rust 1.98.1, eight Cargo jobs, serial runs under `run-heavy --mem 16G --jobs 8`. Values below are median seconds. The JSON contains every run, ranges, lockfile/archive provenance and Cargo phase times.

| Workload | Mode | Published | PR #123 | Prior rewrite | Optimized | vs prior | vs PR |
|---|---|---:|---:|---:|---:|---:|---:|
| magetypes | check | 2.508 | 2.583 | 2.674 | 2.676 | +0.07% | +3.57% |
| zenav1-svt | check | 8.204 | 8.274 | 8.447 | 8.422 | -0.30% | +1.79% |
| rav1d-safe | check | 7.757 | 7.773 | 7.794 | 7.770 | -0.31% | -0.04% |
| magetypes | release | 2.780 | 2.889 | 2.986 | 3.006 | +0.68% | +4.05% |
| zenav1-svt | release | 24.020 | 24.222 | 24.609 | 24.361 | -1.01% | +0.57% |
| rav1d-safe | release | 11.548 | 11.581 | 11.900 | 11.719 | -1.52% | +1.19% |

**The allocation reduction did not establish an end-to-end cold-build speedup.** Before/after ranges overlap in all six comparisons. The magetypes penalty against PR #123 remains; the no-regression gate is still open. No result here supports moving the draft to main.

Cargo unit medians for the magetypes check workload (Cargo reports these at 0.01 s precision):

| Phase | Published | PR #123 | Prior rewrite | Optimized |
|---|---:|---:|---:|---:|
| archmage-macros | 0.31 | 0.31 | 0.40 | 0.41 |
| magetypes | 0.89 | 0.96 | 0.96 | 0.97 |

The cold cost still includes compiling the larger macro implementation. Fewer expansion allocations do not remove that cost. Further compile-time work needs to address shared implementation size and duplicated lowering, with the same output/safety oracles; this pass does not introduce caches or change boundary placement to chase those timings.

[Complete cold-build measurements](consumer_compile_optimized_2026-10-07.json). Full logs, archived sources and Cargo timing HTML remain at `/home/lilith/tmp/attune-consumer-optimized/comparison/`.

```text
run-heavy: done rc=0 224s | peak-RSS 0.88GiB | min-avail 20166MiB | peak-load 2.96
run-heavy: done rc=0 467s | peak-RSS 1.19GiB | min-avail 19356MiB | peak-load 6.22
```

## Validation and remaining gates

Full acceptance is still open: the macro unit suite has one existing
intermediate-expansion expectation pending explicit approval to update, and
the cold-build requirement against PR #123 is not met. Nothing from this pass
has been pushed to main.

Passed on the optimized production source:

- The unchanged legacy expansion snapshots, sibling-resolution tests,
  proof-boundary rejection tests, and new attune runtime/attribute contracts.
- `cargo clippy -p archmage-macros --all-targets --all-features -- -D warnings`.
- `cargo test -p archmage -p magetypes --features 'std avx512'`, including doctests.
- The attune contracts compile for AArch64, WASM32 and i686. These are compile
  checks, not execution on those targets.
- All 72 matched consumer builds/checks.

The macro unit suite reports 128 passed, one failed, and one existing opt-in
allocation measurement ignored. The sole failure remains
`variant_replacement_keeps_the_token_position`: it expects an anonymous
parameter before boundary expansion, while the shared emitter has already
normalized it into a forwarding parameter. The compiler-facing PR #123
snapshot for this behavior passes unchanged. The proposed update will check
both final signatures and forwarding order; existing dispatch assertions are
retained. No existing test expectation was changed during this optimization.

Full test logs use `/home/lilith/tmp/attune-opt-` names: `final-compat.log`,
`final-unit.log`, `final-clippy.log`, `packages.log`, and the three `cross-*.log`
files. The package suite is also exposed as `just attune-packages`.

```text
run-heavy: done rc=0 78s | peak-RSS 0.33GiB | min-avail 21137MiB | peak-load 2.58
run-heavy: done rc=0 96s | peak-RSS 0.83GiB | min-avail 19759MiB | peak-load 5.83
```

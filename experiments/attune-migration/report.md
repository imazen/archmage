# Attune migration inventory

MISSING / open design: attune itself has no implementation or codegen measurements here. Trait adapter placement, `entry(token)` compatibility, tier/cfg selection grammar, generic `Token` substitution, explicit inline overrides, and cross-crate callee-surface discovery remain undecided. A public `_t` wrapper around a private body is an **open static-call alternative**, supported by the parent-agent assembly experiment; public direct names are not mandatory for every public family. No runtime timings are reported.

The read-only occurrence accounting is complete for the stated archive scope: **1,162 of 1,162 actual macro attributes/calls**, and **5,448 of 5,448 raw candidates** reconciled with an independent `rg` scan, with **0 unaccounted candidates**. This is lexical source coverage, not proof of compiler expansion coverage or a compiling migration. The complete from/to design is the compact occurrence-to-pattern mapping plus the annotated pattern guide, not 1,162 purportedly compiling replacements.

Baseline: [archmage cf07592212e96294ef9ca5dca9a364fa8d15d8ad](https://github.com/imazen/archmage/tree/cf07592212e96294ef9ca5dca9a364fa8d15d8ad). The supplied archive is the authority for this inventory; its commit identity was supplied by the brief, not independently established through git.

## Deliverables

Read [migration.md](migration.md) for output selection and patterns P0–P13, then [migration-contracts.md](migration-contracts.md) for generics, methods, visibility, inline policy, measured wrapper context and failure dispositions. Every proposed snippet carries API, INLINE and BEHAVIOR comments. Syntax that is undecided is shown as a semantic requirement or retained legacy case.

The compact index contains only actual migration macro attributes and dispatch calls, retaining exact macro/call text, original lexical signatures, explicit visibility, source line numbers and destination rules. Implicit method/trait visibility is marked unresolved rather than assumed private. File heading links target the original baseline; each occurrence includes its line number. The raw index additionally provides an individual baseline link for every record.

Compact index parts:

- [01](source-index-01.md), [02](source-index-02.md), [03](source-index-03.md), [04](source-index-04.md), [05](source-index-05.md), [06](source-index-06.md), [07](source-index-07.md)
- [08](source-index-08.md), [09](source-index-09.md), [10](source-index-10.md), [11](source-index-11.md), [12](source-index-12.md), [13](source-index-13.md)

Also retain [failure-index.md](failure-index.md), [inventory.py](inventory.py), and [inventory-summary.json](inventory-summary.json). Raw generated evidence is `occurrences-001.md` through `occurrences-057.md`; support attributes, prose and snapshots remain there. All 57 parts have been preserved. Scope manifests `scope-01.tsv` through `scope-04.tsv` include every selected file, byte size, SHA-256 and category, including files with zero macro occurrences. `scope-files.txt` is the initial file listing. Raw evidence may stay outside git as requested; it is reproducible from the pinned archive and generator.

Every report/index/generator file is below 30,000 bytes. No source file, test expectation or snapshot was edited. No builds, package installation, git/jj initialization, commits, pushes or delegation occurred in this lane.

## Verified counts

| Classification | Rust files | Actual macro attrs/calls | Other code attributes |
|---|---:|---:|---:|
| Tests/examples and nested fixture crates | 137 | 967 | 3,009 |
| Expansion inputs, excluding known-bug inputs | 102 | 165 | 7 |
| Expected-failure / known-bug inputs | 34 | 28 | 26 |
| Positive soundness control | 1 | 2 | 1 |
| Generated expansion snapshots | 105 | 0 | 662 |
| Total | 379 | 1,162 | 3,705 |

“Other code attributes” deliberately includes a superset of function attributes: test, cfg, lint, derive and module/item attributes. They are raw evidence, not thousands of migration tasks. The 105 snapshots have no surviving code invocations of the inventoried migration macros; their target-feature/inline/support attributes remain accounted for separately.

| Code macro name | Occurrences |
|---|---:|
| arcane | 623 |
| rite | 174 |
| autoversion | 119 |
| magetypes | 87 |
| token_target_features_boundary | 2 |
| token_target_features | 3 |
| simd_fn | 1 |
| incant | 146 |
| dispatch_variant | 4 |
| simd_route | 3 |

This is **1,009 attributes + 153 dispatch calls**. Counts include macro_rules templates once at their source location, all cfg branches, expected failures and positive controls; they do not multiply templates by instantiations or count comments as code.

Raw non-code accounting adds **548 comment mentions**, **13 literal mentions** (including 2 in snapshots), and **20 diagnostic mentions** across `.stderr` files. Thus 1,162 + 3,705 + 548 + 13 + 20 = **5,448**. Non-Rust scope files are in the manifest rather than interpreted as Rust source. All **435 scoped files** are accounted for, including 379 Rust files and 56 other files.

## Scope and limits

The generator discovers every archived file with a `tests` or `examples` path component using `rg --files --hidden`. This includes `tests/`, `examples/`, `magetypes/tests/`, `magetypes/examples/`, downstream-compat fixture crates, AVX-512 cfg fixture crates, soundness fixtures and expansion inputs/snapshots. `archmage-macros/tests/` is absent in this baseline; it contributes zero directory-based occurrences. There are no explicit manifest test/example path overrides in the three package manifests inspected.

Production `src/`, benches, docs/site prose and xtask generator code are outside the occurrence denominator, including inline unit tests within production macro implementation files. Relevant macro source and its inline rewrite tests were inspected for current behavior and linked in the guide; they are not silently mixed into directory-based counts. README prose under examples is manifested but not parsed as Rust. No arbitrary macro import renaming was found in the scoped source that requires an extra dispatch spelling; the supported public aliases are included. No cfg_attr wrapping of the inventoried macro attributes was found in the requested roots.

The scanner is lexical, not a Rust parser. It masks nested comments and literals when finding code regions, preserves original source spans, balances macro delimiters and function headers (including array semicolons), and retains textual macro mentions separately. It cannot resolve trait/inherent ownership, semantic visibility, generated `$` signatures or type aliases. Those items explicitly retain context links and contract rules. The independent `rg` reconciliation proves coverage of the declared attribute/dispatch start patterns, not all conceivable future Rust syntax or macro aliases.

Expected failures are **fixture intent**, not newly observed compiler results. In particular, `scalar_not_in_tier_list` is dormant; `should-fail/incant_passthrough` refers to an expanded-output diagnostic, not universal failure of the source form. Runtime-generated negative/positive programs in `soundness_exploits.rs` are called out in P9 even when their source is a string or read from a fixture file. Positive raw-context controls are not classified as failures.

## Verified current behavior and migration consequences

- [Rite strips inline attributes and adds inline](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/rite.rs#L305). A written `inline(never)` is not the effective old policy for that macro.
- [Autoversion moves user attributes to the dispatcher and makes variants private](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/autoversion.rs#L224). `make(all)` on its public API adds public names; `make(_)` is the conservative surface choice.
- [Magetypes clones signatures and substitutes Token](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/magetypes.rs#L84). Ordinary type/const generics and bounds must survive; backend bounds alone are not feature contexts.
- [Tokenful rewriting can try upgrades](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/rewrite.rs#L164), while [tokenless context rewriting only uses covered tiers](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/rewrite.rs#L279). New static composition is not equivalent to every old incant call.
- [Private arcane bodies remain behind public wrappers](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/archmage-macros/src/arcane.rs#L512). Name access and optimization visibility are different questions.

Parent-agent measurement provenance belongs to [../cross-crate-inline/README.md](../cross-crate-inline/README.md) in the eventual draft. Results reported to this lane: 252 Rust 1.98.1 cross-crate cases with LTO off; 72/72 matching/superset wrapper-vs-direct normalized assembly comparisons matched; inline direct bodies inlined at 1/16 stages, remained calls at 128; private-name access failed E0603, while public inline wrappers could optimize private bodies. This lane did not rerun or inspect that sibling artifact. These are not runtime timings and not measurements of the proposed attune implementation.

## Reproduction and validation

From this directory:

```sh
python3 inventory.py --source source --out .
```

Both arguments accept arbitrary paths; defaults are `source/` beside the script and the script directory. Output inside the source archive is rejected. After relocation to the draft, supply the pinned archive explicitly. The baseline URL is intentionally pinned in the generator; changing source revision requires changing that provenance as well. The generator does not initialize or inspect git.

Completed checks: independent rg candidate reconciliation; compact/raw record totals; all generated part sizes; original manifest SHA-256 matches; destination P0–P13/contract anchor existence; explicit visibility retention; balanced array/const-generic/lifetime signatures; CLI argument/help behavior. Regeneration is deterministic for this archive. No tests/builds were run in this lane.

Parent reported that local `cargo xtask ci` passed, with toolchain-dependent Miri/ARM/WASM jobs skipped. That result is separate from this inventory and does not validate attune, which does not yet exist.

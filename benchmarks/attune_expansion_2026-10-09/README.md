# Attune raw expansion audit — 2026-10-09

All 8 requested target/gate configurations passed. This is compilation and expansion coverage, not a runtime benchmark or closure of the consumer cold-build gate.

Harness: `a100424120af7d94ea0dec247f3ed941cbe450fc`. Macro diagnostic fix: `3d9f8628ea60`.
Compiler: `rustc 1.98.1 (48a229cea 2026-09-01)`.

[Corpus and exact coverage](../../tests/attune_expansion/README.md) · [Final measurements and raw hashes](results.json) · [Earlier run records](earlier-runs.json)

Full indexed raw artifacts are at `/home/lilith/output/archmage/attune-expanded-2026-10-09/README.md`. This host has no `/mnt/v`; the final artifacts use a persistent output directory outside scratch and Git. Every case links its input and raw module. Root `artifacts.json` hashes the full capture.

## Command

```sh
just attune-expanded-raw /home/lilith/output/archmage/attune-expanded-2026-10-09 --keep-going
```

The run used the build resource limiter with a 16 GiB cap and eight jobs, with no concurrent build. Its measured resource summary was:

```text
run-heavy: done rc=0 154s | peak-RSS 0.37GiB | min-avail 27826MiB | peak-load 1.29
```

## Results

Each positive case compiled as source under `forbid(unsafe_code)` with warnings denied. Complete raw output also compiled independently. Each negative case produced its expected diagnostic. Removed-symbol and applicable CPU-probe assertions passed. These counts describe the finite corpus; they do not establish exhaustive Rust-language coverage.

| Target | Cargo gates | Positive cases | Rejection cases | Result |
| --- | --- | ---: | ---: | --- |
| x86_64 | off | 1539 | 466 | pass |
| x86_64 | on | 1572 | 477 | pass |
| aarch64 | off | 1510 | 451 | pass |
| aarch64 | on | 1510 | 451 | pass |
| wasm32 | off | 1453 | 464 | pass |
| wasm32 | on | 1453 | 464 | pass |
| x86 | off | 1072 | 273 | pass |
| x86 | on | 1072 | 273 | pass |

## Findings

- Unavailable foreign or Cargo-gated candidates incorrectly reported ambiguous parent proofs before filtering. The fix applies the same selection guard to diagnostics and call arms. Cross-target invocation cases and the dedicated gated-proof regression now pass.
- Nested wrappers still require `_self = Type` for a receiver. Both `nested` and `in_trait`, plus implied nesting from `_self`, are covered for legacy `arcane` and new `attune(wrap)`. The pre-rewrite `de7d28af833b` implementation already rejected receiverful nested defaults without `_self`. Receiverless defaults and concrete trait implementations pass.
- A token parameter alone does not infer `wrap`; a proof suffix does. A direct suffix remains direct even with a token parameter.
- `-_neon` works as a definition modifier and removes NEON output on AArch64. Definition modifiers inside invocation tier lists and method-call invocation syntax remain rejected.

## Limits and other checks

The corpus splits products by concern instead of taking one Cartesian product across every axis. It does not cover arbitrary receiver types or every Rust attribute. Qualified and associated paths are checked; this audit is not a consumer migration, cross-crate performance measurement, or execution on ARM/WASM/i686. Multi-body callers do not use the single-body CPU-probe assertion. Raw replay allows compiler-generated attributes and applies a lint cap only to the replay crate; source safety lint checks remain independent.

The diagnostic fix also passed macro unit tests, macro-crate Clippy, the gated-proof test with features off/on, allocation instrumentation, and unchanged legacy compatibility/expansion tests. The ten recorded allocation probes retained their counts and bytes. Full logs are preserved in the local capture; this is not evidence about downstream cold-build time.

```text
run-heavy: done rc=0 79s | peak-RSS 0.33GiB | min-avail 28305MiB | peak-load 2.25
```

Earlier-run metadata preserves the initial failing audit and intermediate checks. The initial default-trait receiver success expectation was corrected with user approval; it remains explicitly covered as a rejection alongside successful receiverless defaults. No existing legacy expectations were changed.

# Beta validation, 2026-10-10

The measured cold-compile overhead was accepted for this beta on 2026-10-10.
The [full-consumer follow-up](consumer_compile_beta_2026-10-10.md) records the
Magetypes, encoder, and decoder measurements behind that decision. This small existing legacy-consumer fixture
has higher cold check times in the beta; it is not a measurement of large codec
consumers or of runtime performance.

## Compile comparison

Intel Core Ultra 7 265K, Rust 1.98.1, eight Cargo jobs. Six alternating pairs;
each cold run uses a distinct empty target directory. Consumer-only runs touch
the unchanged fixture after the cold check. No target-cpu override was supplied.
The candidate archive's package versions were normalized to 0.9.30 before
resolving and sharing its lockfile; the working checkout was not changed.

Baseline: `e2dbab66ef5aa08f8e23ed05248e7d1217f58475` (published 0.9.30).
Candidate snapshot: `9e829f5e6b57354a82afdff17877c0b487b48be0`, containing implementation `b22d0336181a`.
Fixture: `tests/downstream-compat/compile-cost`; identical source on both sides.

| Features | Stage | Baseline median (s) | Beta median (s) | Change |
| --- | --- | ---: | ---: | ---: |
| macros | cold | 1.755340 | 1.890423 | +7.70% |
| macros | consumer | 0.038673 | 0.038282 | -1.01% |
| all | cold | 3.153127 | 3.295131 | +4.50% |
| all | consumer | 0.042235 | 0.042852 | +1.46% |

`macros` uses the default fixture; `all` adds `avx512,use_magetypes`.
[Every measured run](attune_beta_compile_2026-10-10.csv) includes maximum RSS
from `/usr/bin/time -v`. A separate Cargo timing diagnostic pair reported
0.39s/0.54s for compiling archmage-macros, 0.10s/0.10s for archmage, and
0.03s/0.03s for the fixture (baseline/beta; Cargo's rounded values).

Command: `just attune-beta-compare v0.9.30 9e829f5e <out> 6`, under run-heavy
with `--mem 16G --jobs 8` and disk-backed TMPDIR.
Full logs, archives, manifest transformations, tool versions, and timings are
retained in `~/output/archmage/attune-beta-compile-2026-10-10/`.
`results.json` SHA-256: `78b5d5303c1cd545f70356bc847ab125bf9c7e5d48ed67e3d95497635c9f6084`.

Resource-limiter summary: `rc=0 62s | peak-RSS 0.37GiB | min-avail 27174MiB | peak-load 1.35`.

## Expansion and packaging

All eight target/gate configurations passed: x86-64, AArch64, wasm32 and i686,
with the AVX-512 Cargo gate disabled/enabled. Each compiles source and raw
expansions, checks the expected rejection diagnostics, and retains per-case
artifacts. Full output: `~/output/archmage/attune-beta-expanded-2026-10-10/`.
Resource-limiter summary: `rc=0 158s | peak-RSS 0.37GiB | min-avail 27462MiB | peak-load 1.79`.

The local all-target tests, doctests, generated-source checks, soundness checks,
and workspace package/publish dry-run passed. No packages were uploaded.
Package-check log: `~/tmp/attune-beta-packages.log`.
Resource-limiter summary: `rc=0 377s | peak-RSS 2.18GiB | min-avail 26175MiB | peak-load 1.11`.

## Semver checker discrepancy

Archmage's check against 0.9.30 passes. Cargo-semver-checks 0.51.0 reports
`safe_inherent_method_requires_more_target_features` for Magetypes, including
when the same current rustdoc JSON is supplied as both baseline and current
with `--release-type patch`. Its [lint query](https://github.com/obi1kenobi/cargo-semver-checks/blob/v0.51.0/src/lints/safe_inherent_method_requires_more_target_features.ron)
matches owner paths and method names without distinguishing impl type arguments.
V4 and V4x `from_raw` methods therefore cross-match.

The diagnostic `xtask/check_target_features.py` retains concrete impl arguments
and compares 2,040 safe inherent methods, including 40 with target features:
zero differences that add requirements against published 0.9.30. Its tests
reject strengthened requirements independently on either specialization,
including a previously unannotated method. This companion does not disable
the existing semver lint; release CI remains blocked on it pending review.

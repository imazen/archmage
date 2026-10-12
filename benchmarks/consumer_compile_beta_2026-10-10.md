# Beta cold consumer compilation — 2026-10-10

These are fresh measurements of the beta implementation against published
0.9.30 source, using the same pinned consumer sources as the earlier survey.
They do not establish a no-regression result: Magetypes and the encoder have
higher cold medians. Decoder ranges overlap in both modes. The measured
overhead was accepted for this beta on 2026-10-10; the compile-time release
gate is satisfied by that explicit acceptance, not by a no-regression claim.

| Workload | Mode | 0.9.30 (s) | Beta (s) | Added time | Change |
| --- | --- | ---: | ---: | ---: | ---: |
| magetypes | check | 2.560 | 2.677 | +0.117 s | +4.57% |
| zenav1-svt | check | 8.052 | 8.202 | +0.150 s | +1.87% |
| rav1d-safe | check | 7.504 | 7.537 | +0.033 s | +0.44% |
| magetypes | release | 2.817 | 2.926 | +0.109 s | +3.87% |
| zenav1-svt | release | 23.308 | 23.568 | +0.260 s | +1.11% |
| rav1d-safe | release | 11.167 | 11.204 | +0.037 s | +0.33% |

Medians of three repetitions, rotating baseline/candidate order. All 36 checks
and release library builds passed. These include building the full dependency
stack; they are not incremental checks, final executable links, or runtime
codec-performance measurements. The encoder and decoder keep legacy attributes.

## Method and inputs

- Baseline: `e2dbab66` (`v0.9.30`); beta: `afa92acb` (PR #129).
- [zenav1-svt](https://github.com/imazen/zenav1-svt): `224c6bbb9a3dc577c64f6e0531bda5c3f60c6b0b`.
- [rav1d-safe](https://github.com/imazen/rav1d-safe): `5b51b144cf73216fffd1100bfc396bc7dc17b5bd`.
- Intel Core Ultra 7 265K, Rust 1.98.1, eight Cargo jobs, serial runs under
  `run-heavy --mem 16G --jobs 8`.
- Default consumer features and original release profiles. Magetypes alone uses
  its defaults (`std`, `w512`); codec dependency features remain as written.
  This rav1d-safe snapshot uses Archmage directly and does not depend on Magetypes.
- Fresh target directory for every run. Compiler wrappers and incremental
  compilation disabled; no target-cpu override. Downloads and OS caches warm.
- Both archived dependency stacks have manifest versions normalized to 0.9.29
  to satisfy the fixed consumer requirements without prerelease resolution.
  This changes only archive manifests; the baseline implementation is 0.9.30
  and the candidate is the beta. No live checkouts were modified.
- The two rav1d-safe Git dependency declarations become exact registry-version
  declarations in isolated copies; Cargo patches select the archived sources.
  Syn 3.x is pinned to 3.0.7 in both stacks.
- Consumer Rust files, manifests, and lockfiles match between each pair. Metadata
  confirms one Archmage/macro package from the intended source archive, and one
  Magetypes package where used. The raw record retains source/archive hashes,
  version normalization, tool versions, every measurement and peak RSS.

## Where the cost occurs

Cargo's rounded check-phase medians locate the increase in compiling the macro
implementation. Magetypes itself remains 0.92s in its standalone workload.
Compilation of different packages can overlap, so phase differences must not be
summed to reconstruct wall time.

| Workload | Cargo unit | 0.9.30 (s) | Beta (s) |
| --- | --- | ---: | ---: |
| magetypes | archmage-macros | 0.31 | 0.44 |
| zenav1-svt | archmage-macros | 0.31 | 0.45 |
| zenav1-svt | magetypes | 1.10 | 1.09 |
| zenav1-svt | zenav1-svt | 0.05 | 0.05 |
| rav1d-safe | archmage-macros | 0.35 | 0.49 |
| rav1d-safe | rav1d-safe | 4.53 | 4.52 |

## Reproduction

[Harness](consumer_compile.py) · [Raw measurements and ranges](consumer_compile_beta_2026-10-10.json).
Full logs, Cargo timing HTML, source archives, metadata and lockfiles:
`~/output/archmage/attune-beta-consumers-2026-10-10/`.

Use `just consumer-compile OUT prepare --sources SOURCES --baseline v0.9.30
--candidate afa92acb`, then `just consumer-compile OUT check` and
`just consumer-compile OUT release` under run-heavy. `SOURCES` is the retained
source capture at `~/tmp/attune-consumer-cold/sources/`, with revisions and hashes
recorded in its `provenance.json`. Each command refuses to overwrite prior runs.

```
run-heavy: done rc=0 110s | peak-RSS 0.88GiB | min-avail 26741MiB | peak-load 1.80
run-heavy: done rc=0 225s | peak-RSS 1.19GiB | min-avail 26559MiB | peak-load 2.13
```

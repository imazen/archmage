# Downstream constructor migration compatibility

Verified 2026-09-27 on `i265`, `x86_64-unknown-linux-gnu`, Rust 1.98.1,
against archmage workspace commit `4401f724b6c8` (implementation `d53d425d`).

All **20 published zen-prefixed direct consumers** in the registry discovery
compiled against the local changes using their latest non-yanked stable releases.
`linear-srgb`, `garb`, and `jxl-encoder-simd` also compiled: **23/23 libraries**.
There were no confirmed compiler regressions in these checks.

This is native `cargo check` with default consumer features and normal warning
settings, not runtime tests, all-features testing, or an ARM/WASM consumer audit.
The deprecations still require migration for callers using `deny(deprecated)` or
`deny(warnings)`. This does not supersede the known ARM/WASM published
jxl-encoder-simd conversion incompatibility documented in
[TOKEN-CONSTRUCTOR-MIGRATION.md](TOKEN-CONSTRUCTOR-MIGRATION.md).

## Published consumers

[cargo-copter](https://github.com/imazen/cargo-copter) at commit
[`2d50bf89`](https://github.com/imazen/cargo-copter/commit/2d50bf89) was built and
run with `--only-check --simple`. Published baseline and local-source rows were
checked separately; successful offered rows resolved to `Local`. The magetypes
runs also patched its archmage and archmage-macros path dependencies. The other
eight consumers received an additional check with all three workspace crates
patched through Cargo configuration.

| Consumer | Version | Result |
|---|---|---|
| `garb` | 0.2.8 | pass |
| `jxl-encoder-simd` | 0.3.0 | pass |
| `linear-srgb` | 0.6.12 | pass |
| `zenanalyze` | 0.1.0 | pass |
| `zenavif` | 0.1.6 | pass |
| `zenbitmaps` | 0.1.5 | pass |
| `zenblend` | 0.1.3 | pass |
| `zenfilters` | 0.1.0 | pass |
| `zenflate` | 0.4.0 | pass |
| `zengif` | 0.7.3 | pass |
| `zenjpeg` | 0.8.4 | pass |
| `zenjxl` | 0.2.1 | pass |
| `zenjxl-decoder-simd` | 0.3.9 | pass |
| `zenpixels-convert` | 0.2.16 | pass |
| `zenpng` | 0.1.4 | pass |
| `zenquant` | 0.1.3 | pass |
| `zenraw` | 0.2.0 | pass |
| `zenresize` | 0.3.1 | pass |
| `zensim` | 0.2.7 | pass |
| `zensim-regress` | 0.3.1 | pass |
| `zentone` | 0.1.0 | pass |
| `zenwebp` | 0.4.4 | pass as a library dependency |
| `zenyuv` | 0.1.3 | pass |

Cargo-copter passed **22/23** current usable releases as standalone packages.
The remaining package, `zenwebp 0.4.4`, cannot freshly resolve its yanked
`webpx 0.1.4` dev-dependency, both with published and local archmage. A separate
crate depending on `zenwebp = "=0.4.4"` compiled in both configurations. Its
normal library dependency graph is therefore covered; its standalone dev graph
is not a passing check.

## Local committed sources

Clean local checkouts were archived before testing. Original consumer checkouts
were not edited. Direct baseline and patched checks retained every original
manifest declaration, including workspace-inherited features. **12/14 packages
passed** both configurations; two could not resolve absent sibling checkouts.
Successful local checks were verified through Cargo's compiler-artifact JSON:
archmage, archmage-macros, and magetypes (where used) came from the current local
archmage workspace.

| Package | Version | Baseline / local |
|---|---|---|
| `zensim-train-core` | 0.0.1 | pass / pass |
| `zenav1-svt-dsp` | 0.1.0 | pass / pass |
| `zenav1-svt-encoder` | 0.1.0 | pass / pass |
| `zenanalyze` | 0.2.0 | pass / pass |
| `zensim-validate` | 0.1.0 | pass / pass |
| `zensim` | 0.3.0 | pass / pass |
| `zenpng` | 0.2.0 | pass / pass |
| `zenjpeg` | 0.9.0 | pass / pass |
| `zenyuv` | 0.1.3 | pass / pass |
| `zenavif` | 0.2.0 | pass / pass |
| `zentone` | 0.2.0 | blocked / blocked |
| `zenpredict` | 0.2.0 | pass / pass |
| `zensim-bench` | 0.0.0 | blocked / blocked |
| `zensim-regress` | 0.4.0 | pass / pass |

`zentone` needs the missing `../zenpixels/zenpixels` sibling; `zensim-bench`
needs `../zenmetrics/crates/zenstats`. These are snapshot setup blockers, not
constructor failures. Local zenmetrics had uncommitted changes and was not
built. Active markers excluded zenbitmaps, zenflate, zengif, zenquant, zenwebp,
and zenzop; zenav1-aom had a stale marker with working-copy changes. Their local
WIP is not covered by the published-package results.

| Snapshot | Commit |
|---|---|
| `zenav1-svt` | `dd3e33709ef2` |
| `zenanalyze` | `b102fa5f3480` |
| `zensim` | `5a4d5fa3c213` |
| `zenpng` | `f7167b7952d3` |
| `zenjpeg` | `f325ca4fc8cf` |
| `zenavif` | `dba8f5ee73cc` |
| `zentone` | `df84ca43dcc4` |
| `zenjxl` | `9226d3a32cfd` |

## Cargo-copter limitations found in this run

1. **“Latest” included yanked releases.** Registry version records confirmed
   `linear-srgb 0.7.0`, `zenavif 0.1.7`, `zenfilters 0.1.1`,
   `zenpixels-convert 0.3.0`, and `zenwebp 0.4.5` were yanked. They were rerun
   with explicit latest non-yanked versions, shown above. Initial failures of
   the latter two must not be reported as failures of their current usable
   releases.
2. **Forcing a workspace-inherited dependency dropped inherited features.**
   In the zensim-train-core snapshot, copter rewrote
   `magetypes = { workspace = true }` to a bare path dependency. The original
   workspace dependency enables `avx512`; the rewritten dependency lost it,
   producing `X64V4Token: F32x16Backend` errors and a false regression report.
   Restoring the original manifest and using Cargo configuration patches passed.
   Compiler-artifact JSON confirmed `avx512` on all three local crates.
3. **Local manifests remained rewritten after testing.** The tool retained
   `Cargo.toml.original.txt` backups. Local checks were rerun in fresh copies
   with these manifests restored, using only Cargo configuration overrides;
   otherwise one workspace member's override can affect another member's
   baseline. The direct-check results above take precedence over the initial
   local copter classifications.

No cargo-copter fixes were included in the archmage change.

## Reproduce and inspect

For a published consumer, use a verified non-yanked version explicitly:

```sh
cargo-copter --path /home/lilith/work/archmage/magetypes \
  --dependents zenjpeg:0.8.4 zenpixels-convert:0.2.16 \
  --only-check --simple
```

For local workspaces, preserve manifests and pass configuration patches:

```sh
cargo check --manifest-path /path/to/snapshot/Cargo.toml \
  --config 'patch.crates-io.archmage.path="/home/lilith/work/archmage"' \
  --config 'patch.crates-io.archmage-macros.path="/home/lilith/work/archmage/archmage-macros"' \
  --config 'patch.crates-io.magetypes.path="/home/lilith/work/archmage/magetypes"'
```

Heavy commands ran serially through `run-heavy --mem 16G --jobs 8`, with
`TMPDIR=/home/lilith/tmp`. Full source snapshots, command argument arrays,
registry metadata, logs, and reports are retained at
`/home/lilith/data/archmage/downstream-2026-09-27/` on `i265`.
The JSON report's `passed_test` summary is not evidence of runtime testing:
these copter runs explicitly used `--only-check` and contain Fetch/Check commands.

Artifact SHA-256 checksums:

```text
52d0a098f74e39bfaaae68ca705ff1df233a1a94655e53284ca4ab86ed264414  verified-published-results.json
1d2b9b62072d481d8b0b2c14887af9575c95c50033d7e6c9d228150ed51ca31f  published-versions.json
e3706396589aafdcb1f2e68cdf03289eb27ec7caa6c684002882c3151c4597bc  direct-results.json
35646b6e420a01952dd1fe28550fd4dc100d409a6ad32bc37e459b7c239c4872  combined-results.json
85aa20db381f6721c9d7b7f80c832667f4440367ba9938669bbe8cb2526c07a9  local-snapshots.json
00eb69e6d96532aee6ace1b8b047056383beb7d4c318a47165f30f7d2719c72e  zenwebp-library-results.json
995438ad3e4e3912f9e59d96cd03159dbad56d551b5f0442a6a31076ce06c799  archmage/copter-report/report.json
7ed28c9c0e7a49a85c1a03a5879c6c5d166d589599323eff1fcba2116cd39ad6  magetypes/copter-report/report.json
0f79e84e289af0ef3a8d4a2267e6c8f1c0886afd8aec9af84a6602ee69caab5f  non-yanked/copter-report/report.json
cfb40d9754690e267d1c4158a86508f9c037a6a8a198b23a89a3b07eacdccad1  local-magetypes/copter-report/report.json
b4966ca9a4e630e4674a68687d35bc4d4767187d09bd7e48d6fefaf86fd69c90  local-archmage/copter-report/report.json
```

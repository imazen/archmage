# Downstream compatibility

## 0.9.30 release check against local checkouts (2026-10-07)

Every local repository under `~/work/zen/` and `~/work/` whose manifests depend
on archmage, archmage-macros or magetypes (50 repositories; jj lane workspaces,
`_*`, `retired/` and the archmage workspaces excluded) was snapshotted with
`git archive HEAD` into an isolated mirror and built with the three crates
patched to the release tree (`--config patch.crates-io.<crate>.path=...`,
shared build directory, Rust 1.99.0, `x86_64-unknown-linux-gnu`, dev). The
live checkouts were not touched: cargo-copter builds local dependents in place
and rewrites their manifests, so it was not pointed at them. Working-copy
changes of other lanes (13 repositories were dirty) are not covered.

**Check:** 37 of 50 snapshots check clean (`cargo check --workspace
--all-targets`, or `--workspace` where noted). None of the 13 failures is an
archmage change: they are build scripts that need git or a C toolchain,
workspace members or test vectors not tracked by git, `[patch]` siblings outside
the mirror, nightly-only code, and one crate (zensr) using
`f32x8::concat_shift`, which exists only on the unmerged `feat/concat-shift`
branch.

| Snapshot (HEAD) | `cargo check --workspace --all-targets` | Note |
|---|---|---|
| `butteraugli` aac925f | pass |  |
| `coefficient` fa1ef73 | fail | its `[patch]` needs sibling repos (ravif, zenjpeg at a version the mirror lacks) |
| `dssim` 711efc8 | pass |  |
| `dvifmish` 8784d3f | pass |  |
| `garb` 0e475de | pass |  |
| `hdr-research` 8985b51 | fail | workspace member `hdr-editor` is not tracked by git |
| `highway-rs-port` 371553e | pass |  |
| `hoisted-bounds` 79d0ac4 | pass |  |
| `imageflow` f49ec638 | fail | `imageflow_types/build.rs` needs a git checkout and `GIT_COMMIT` |
| `imageflow-zencodecs-v2` 9e0ee767 | fail | same build script as imageflow |
| `libimagequant` 39a5edc | fail | `#![feature]` (nightly-only) |
| `moxcms-archmage` f1ec4c4 | pass |  |
| `moxcms-safe` eee52bf | pass |  |
| `quantizr-fast` 7480262 | pass |  |
| `simd-compare` 31756e6 | fail | `[patch]` sibling `third-party/pulp` missing; then its own `extern` blocks (edition) |
| `zen/aom-decoder-rs` b087b30 | pass |  |
| `zen/aom-rs` 1434bc3 | fail | `aom-sys-ref/build.rs` needs a C toolchain and libaom sources |
| `zen/BRAG` d3d66a5 | pass |  |
| `zen/butteraugli` 31e79df | pass |  |
| `zen/fast-ssim2` d4ad4fa | pass |  |
| `zen/heic` bdee718 | fail | needs a backend feature; with `backend-rust` its own `enable_deblock_trace` is missing |
| `zen/jxl-encoder` f738479d | pass |  |
| `zen/linear-srgb` c56e794 | pass | library checks clean (its `benches/kernel_tiers.rs` names functions its own cfg turns off; the library and tests check clean) |
| `zen/mozjpeg-rs` beab085 | fail | `sys-local/build.rs` needs a C toolchain |
| `zen/rav1d-safe` e771c7b | pass | library checks clean (its test target requires `--release`; the library checks clean) |
| `zen/ultrahdr` d77976b | pass |  |
| `zen/zenanalyze` e6c72f9 | pass |  |
| `zen/zenav1-aom` 25ce06c3 | fail | same as aom-rs |
| `zen/zenavif` a7c56be9 | pass | library checks clean (an untracked test vector (`tests/vectors/libavif/...`) ; the library checks clean) |
| `zen/zenbitmaps` 32376ef | pass |  |
| `zen/zenblend` 866a067 | pass |  |
| `zen/zenflate` 3d597d4 | pass |  |
| `zen/zengif` 8cd4c9e | pass |  |
| `zen/zenjpeg` 8f703a6e | pass |  |
| `zen/zenjpegai` c7e1ae4 | pass |  |
| `zen/zenjxl` 4a2c021 | pass |  |
| `zen/zenjxl-decoder` 940d2c51 | pass |  |
| `zen/zenmetrics` 594b014bf | fail | workspace member `crates/zenfleet-vastai` is not tracked by git |
| `zen/zenpipe` aa21cba | pass |  |
| `zen/zenpixels` 82cb6e2 | pass |  |
| `zen/zenpng` cfccd88 | pass |  |
| `zen/zenquant` 7dbde81 | pass |  |
| `zen/zenraw` 50f9c69 | pass |  |
| `zen/zenresize` e3975fb | pass |  |
| `zen/zensim` 298ef4e5 | pass |  |
| `zen/zensr` 12e1054 | fail | uses `f32x8::concat_shift`, which exists only on the unmerged `feat/concat-shift` branch |
| `zen/zensysbench` 67c097c | fail | `[patch]` siblings missing (zencodec, zenav1-svt) |
| `zen/zentone` 9f0dab4 | pass |  |
| `zen/zenwebp` 8aa8a78 | pass |  |
| `zen/zenzstd` 21ba0c2 | pass |  |

**Test:** `cargo test --workspace --no-fail-fast` on the 37 clean snapshots
(1,884 s, peak RSS 22.3 GiB): 26 pass in full; 11 have failing test binaries,
none attributable to archmage:

| Snapshot | Failing | Cause |
|---|---|---|
| `butteraugli`, `zen/butteraugli` | `butteraugli-cli` `tests/cli.rs` (11 tests) | runs the binary from `target/debug/`, which the shared build directory does not provide; 15 other binaries pass |
| `quantizr-fast` | two doctests | its own `load_image` helper is not in the doctest scope |
| `zen/jxl-encoder` | `w44_222_decoder_roundtrip` | the external `djxl` cannot load `libIlmImf-2_5.so.25`; 29 other binaries pass |
| `zen/zenanalyze` | `zenpicker` `metapicker_v1_contract` | needs `ZENPICKER_METAPICKER_V1_BAKE`; 32 other binaries pass |
| `zen/zenjpeg` | `encoder_regression::test_quality_floor`, `recompress_api::preserve_identity_emit_handles_16bit_dqt` | pre-existing at that HEAD: the same score (44.6 below the 47.0 floor) and assertion with the archmage 0.9.28 its lockfile pins; 43 other binaries pass |
| `zen/zenjxl-decoder` | corpus-backed feature tests | `codec-corpus` not found; 15 other binaries pass |
| `zen/zensim` | `zensim-validate` `bake_surface` | pre-existing at that HEAD: identical assertion failures without the patch; 135 other binaries pass |
| `zen/zenzstd` | conformance golden tests | golden files not in the snapshot; 4 other binaries pass |
| `zen/rav1d-safe` | (compile) | its test target requires `cargo test --release` |
| `zen/zenavif` | (compile) | an untracked test vector |

The snapshot list, logs and results are under `~/tmp/downstream-0.9.30/`
(`snapshots.tsv`, `results-check.tsv`, `results-test.tsv`, `logs/`); the driver
is `run-phase.sh` there. The signature shapes harvested from the same snapshots
are compiled by `magetypes/tests/harvest_shapes.rs` (`just harvest-shapes ROOT`).

## Constructor migration compatibility (2026-09-27)

Verified 2026-09-27 on `i265`, `x86_64-unknown-linux-gnu`, Rust 1.98.1,
against archmage workspace commit `4401f724b6c8` (implementation `d53d425d`).

All **20 published zen-prefixed direct consumers** in the registry discovery
compiled against the local changes using their latest non-yanked stable releases.
`linear-srgb`, `garb`, and `jxl-encoder-simd` also compiled: **23/23 libraries**.
There were no confirmed compiler regressions in these checks.

This is native `cargo check` with default consumer features, the additional SIMD
feature checks listed below, and normal warning settings. It is not runtime
testing, an exhaustive feature matrix, or an ARM/WASM consumer audit.
The deprecations still require migration for callers using `deny(deprecated)` or
`deny(warnings)`. This does not supersede the known ARM/WASM published
jxl-encoder-simd conversion incompatibility documented in
[TOKEN-CONSTRUCTOR-MIGRATION.md](TOKEN-CONSTRUCTOR-MIGRATION.md).

### Published consumers

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

The optional SIMD paths were checked explicitly against both published and
local crates: `zenbitmaps --features all`,
`zenjxl-decoder-simd --features all-simd`, and
`zensim-regress --features archmage` all passed. Compiler-artifact JSON verified
the local archmage dependencies were active; zenbitmaps' default empty feature
set alone would not exercise archmage.

### Local committed sources

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

### Cargo-copter limitations found in this run

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

### Reproduce and inspect

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
4d7fefa6b442ac55de5d74da5e842da7d004063c9cbefda4edb52b6959602a9a  feature-results.json
```

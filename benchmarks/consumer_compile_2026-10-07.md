# Cold consumer compilation: published, PR #123, and the rewrite

This records measurements on 2026-10-07, not a claim that the rewrite meets its no-regression gate. All 54 measured checks/builds passed. No consumer Rust source was migrated to `#[attune]`; the rewrite handled the existing legacy attributes.

## Inputs and method

- Published stack: archmage, archmage-macros and magetypes **0.9.29**, resolved from crates.io.
- PR #123: `d7f85a5bbdc11410524c557b46908fdfdccdd1bd` ([pull request](https://github.com/imazen/archmage/pull/123)).
- Rewrite: `e20ffeb41b49c67dc6fdbd5411e1f4b49addbb5f` (local jj snapshot; not published).
- [zenav1-svt](https://github.com/imazen/zenav1-svt): `224c6bbb9a3dc577c64f6e0531bda5c3f60c6b0b`.
- [rav1d-safe](https://github.com/imazen/rav1d-safe): `5b51b144cf73216fffd1100bfc396bc7dc17b5bd`.
- Host: i265, Rust 1.98.1, eight Cargo jobs, serial runs through `run-heavy --mem 16G --jobs 8`.
- Three repetitions per cell, rotating published → PR → rewrite order between repetitions. Tables report medians, not an assertion of statistical significance.
- Fresh target directory for every measurement; dependency downloads were warmed before timing, and OS caches were warm. Incremental compilation and compiler wrappers were disabled. No `target-cpu=native`.
- `cargo check --frozen --lib -p PACKAGE --timings` and `cargo build --frozen --release --lib -p PACKAGE --timings`. These are library builds, not final application links. Each consumer retained its own release profile; all three variants of a consumer used the same profile.
- Default package features were retained. Magetypes used its defaults (`std`, `w512`); the encoder/decoder retained their own AVX-512 dependency feature selections.
- Consumer manifests and locks were unchanged throughout timed runs. Preparation normalized syn 3.x to **3.0.6** in all variants. In isolated rav1d-safe copies only, the two pinned archmage Git dependency declarations became exact 0.9.29 requirements; Cargo config selected the corresponding source stack.
- This compares full dependency stacks, not just replacement macro binaries. The PR/rewrite magetypes stack includes libm 0.2.16 while the published default magetypes stack does not. Other resolved package-name/version sets matched for the codec workloads.

## Results

| Workload | Mode | Published (s) | PR #123 (s) | Rewrite (s) | Rewrite vs PR | Rewrite vs published |
|---|---|---:|---:|---:|---:|---:|
| magetypes | check | 2.468 | 2.536 | 2.635 | +3.93% | +6.76% |
| zenav1-svt | check | 8.002 | 8.135 | 8.196 | +0.75% | +2.43% |
| rav1d-safe | check | 7.619 | 7.594 | 7.631 | +0.49% | +0.16% |
| magetypes | release | 2.666 | 2.783 | 2.887 | +3.72% | +8.26% |
| zenav1-svt | release | 23.323 | 23.611 | 23.760 | +0.63% | +1.87% |
| rav1d-safe | release | 11.282 | 11.352 | 11.274 | -0.69% | -0.07% |

The codec deltas overlap observed run-to-run spread; the negative rav1d-safe release median is not evidence of a speedup. All raw measurements and ranges are in [the JSON record](consumer_compile_2026-10-07.json).

Cargo unit timings for the magetypes check workload put macro-crate compilation at median **0.31 s published, 0.32 s PR, and 0.40 s rewrite**. Magetypes itself checks in **0.86 s published, 0.93 s PR, and 0.93 s rewrite**. Cargo reports these phases at coarser precision than the enclosing wall-clock measurement. This points to compiling the macro implementation as a cold-build cost, separate from allocations while it expands consumer code.

## Allocation probe and optimization candidates

The existing `profile_allocations` measurement was run for PR #123 and the rewrite. Counts include allocation/reallocation calls through its thread-local instrumented allocator, excluding input lexing. It uses proc_macro2’s standalone backend, not rustc’s proc-macro bridge; these counts do not measure peak heap or establish consumer wall-time savings. The family frontends are not directly comparable in this probe: PR #123 returns deferred arcane/rite invocations, while the rewrite already expands them.

| Same-input probe | PR allocation calls | Rewrite allocation calls |
|---|---:|---:|
| arcane | 123 | 132 |
| arcane with nested dispatch | 168 | 177 |
| rite | 88 | 88 |

The new-API probes recorded 88 calls for one raw context, 132 for an explicit wrapper, 195 for a small direct family, 756 for a small `make(all)` family, and 913 for that family with a nested call. These generate different interfaces; their ratios are not regressions or speedup estimates.

Priorities to test, with no savings claimed yet:

1. **Remove intermediate emission buffers in the shared boundary emitter.** Its unchanged simple arcane input gained nine allocation calls. Preserve token authentication and sibling-shadowing rejection; optimize how the same checked wrapper is emitted.
2. **Analyze the signature once and keep it typed.** `attune::proof_entry` and dispatcher lowering serialize, substitute, and reparse signatures. `magetypes_impl` serializes whole variants and reparses `LightFn`. Retain parameter positions, token metadata, receiver form and forwarding arguments in one per-function plan.
3. **Prepare bodies once per family.** Presence scans, dispatch parsing and placeholder processing repeat across variants and frontend/emitter layers. Preserve nested-item context boundaries and exact cfg/fallback semantics while separating invariant parsing from per-tier selection.
4. **Emit only requested interfaces and necessary bodies.** A dispatcher currently creates private proof entries even when none were requested. Evaluate calling the same generated feature body directly after successful proof detection, through the shared checked boundary emission. Public `_t` entries remain when requested.
5. **Reduce duplicate implementation code across frontends.** Lower legacy and new syntax to the same explicit internal plan. This targets compiling archmage-macros itself as well as expansion work. Allocator micro-optimizations alone do not address the measured cold macro-crate build delta.
6. **Reuse registry facts and token paths.** Avoid formatting and reparsing the same canonical paths per variant; avoid constructing owned names used only for diagnostics. Keep tier coverage exact and generated from the registry.

## Reproduction and retained data

The harness is [consumer_compile.py](consumer_compile.py), also exposed as `just consumer-compile OUT MODE ...`. Prepare source captures and their `provenance.json`, then run `prepare`, `check`, `release`, and `report`. The JSON record retains exact revisions, archive hashes, compiler details, per-run wall time and `/usr/bin/time -v` max RSS, and selected Cargo phase timings.

Full logs, source snapshots, lockfiles, and Cargo timing HTML: `/home/lilith/tmp/attune-consumer-cold/comparison/`. Allocation logs: `/home/lilith/tmp/attune-alloc-pr123.log` and `/home/lilith/tmp/attune-alloc-profile.log`.

Allocation command: `cargo test -p archmage-macros --lib profile_allocations -- --ignored --nocapture` (also `just attune-profile`). The existing measurement remains opt-in and is not a correctness test.

```text
run-heavy: done rc=0 167s | peak-RSS 0.88GiB | min-avail 20172MiB | peak-load 2.97
run-heavy: done rc=0 340s | peak-RSS 1.19GiB | min-avail 20034MiB | peak-load 3.15
Allocation probe, PR:      run-heavy: done rc=0 2s | peak-RSS 0.29GiB | min-avail 21294MiB | peak-load 1.37
Allocation probe, rewrite: run-heavy: done rc=0 2s | peak-RSS 0.29GiB | min-avail 21537MiB | peak-load 0.50
```

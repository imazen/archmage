# Artifacts and reproduction

Verified local archive on dev:
`/mnt/v/output/archmage/zenmath-dispatch-2026-11-06/`.
NAS mirror: `/mnt/tower/output/archmage/zenmath-dispatch-2026-11-06/`.
`SHA256SUMS` covers retained files; `just final-audit LOCAL MIRROR` checks both
copies, all required platforms/groups/cells, accepted gates, source archives and
retained executable hashes. Generated JSON, binaries and disassembly stay outside
git. Source/scripts/docs/locks are committed as unpublished drafts.

| Run | Result path | Frozen source/executable identity |
|---|---|---|
| Original native, 94/1,094 | `native/timing-native.json` | `native/frozen-source/identity.json`; production `5a9cc685`, harness `98209208`, exact recovered timing SHA `706088a3d594fd676a4e71cb5b44198b1ac952b98ae02657dca11b1e0c22c7b3` |
| Clean ten-group replacement, 10/119 | `native/timing-native-rerun-v2.json` | `native/frozen-rerun-v2/identity.json`, guarded driver `19f3eadc`; same numerical kernels |
| ARM native, 94/678 | `arm/timing-arm.json` | `arm/frozen-source/identity.json`, timing SHA `4a512da79dc07b8150fd8fc884508734d0494708585bfac838e601b112d8a104` |
| Original full WASM, 94/678 | `wasm/timing-wasm49-final.json` | `wasm/frozen-source/identity.json`; exact timed guest `5bee7e91afe13ca6bda6bc2c057d1c5a8b0dc796689f16587f6006b74c6ce9d3`, guarded host `5e51ef69cd69883382987087d65356c34572f429a9bde579cfce8a70d71531e8` |
| Native ordinary facade, 12/147 + replacement 2/24 | `native/timing-facade-native.json`, `native/timing-facade-repeat.json` | `native/frozen-facade/identity.json`, facade `907e91ed`, additional ordinary adapters |
| Final WASM facade, 3/24 | `wasm/timing-wasm-policy-v2.json` | `wasm/frozen-policy-v2/identity.json`, facade `20281eac`, fresh guest target, required ordinary cases |
| Final strict no-SIMD fallback tests | `native/logs/dispatch-wasm-baseline-fresh-v2.log` | `wasm/frozen-baseline-verified/identity.json`, facade `20281eac`, fresh test target |

Each identity retains source manifest/archive, compiler, exact binaries and hashes.
Native numerical builds use Rust 1.99.0 with actual empty CPU flags and thin LTO.
WASM uses Rust 1.99.0, `+simd128`, link-dead-code/export-all and guest LTO disabled;
its native host uses thin LTO and embedded Wasmtime 49.0.5. Both official validation
CLIs were 49.0.5. Cargo fingerprint flags are retained. The environment recorded
when freezing mixed host/guest snapshots is the freeze invocation's environment;
actual guest build flags come from the retained build script/log/fingerprints.
Analysis uses Rust 1.99.1 and Zenbench's public bootstrap API, separately from the
frozen numerical compiler. Exact locks and dependency source/checksum identities
are retained; no native CPU tuning flags were used.

The original full-run WAT has SHA
`bc34203c4a84bbc475503d8d89be8e6b235aef7b77944d7986f9f608f2e1d2ee` and
corresponds to the `5bee7e91…` guest, not an earlier non-exported build.
Final policy v2 guest/host SHA values are in its immutable identity. All snapshots
refuse reuse before writing; `test-freeze.py` verifies existing bytes survive.

Preserved failures and partials are not accepted coverage: missing-QEMU attempt,
missing-wrapper attempts, early WASM builds with overridden flags/missing exports,
WASI combined aborting test attempt, analyzer compile/initial assertion failures,
the one-group native filter collision, the interrupted `portable_cost` benchmark,
and the 21-cell stale WASM guest. `wasm/policy-v1-status.json` explicitly invalidates
that stale guest's later source mapping. Fresh v2 builds include 37 ordinary catalog
entries and assert ordinary cases in each requested result before save.
The original untimed native proof helper is the explicit retention exception;
its historical hash remains in `retention-status.json`. All timed executables are
retained, including the exact recovered original native executable.

Accepted native policy data combines the 84 clean original groups with the ten
clean replacements. All ARM and original full WASM groups are gate-clean.
Direct all-round CIs are labeled separately from Zenbench filtered comparisons;
ARM has 83 and full WASM 58 filtered comparisons retaining fewer than 30 pairs.
The final WASM tiny check has one such filtered comparison; its direct pairs use
all matched rounds. No dirty rounds are silently filtered or accepted for policy.

Reproduce the generated adapters with `just generate`, primitive APIs with root
`just zenmath-dispatch-generate`, and formatting/lint/adapters with `just check`.
`final-validation.sh` runs explicit library/integration/examples, docs, lint,
five Rust 1.89 configurations and the real external feature-unification consumer.
`build-wasm.sh` specifies guest flags; `test-wasm.py` requires caller tier inventory
and checks the aborting length assertion in a separate process. Use fresh Cargo
target directories after source transport; preserved mtimes can hide changes.
Cold build costs use `build-cost.py` with distinct targets per replicate; the
matched consumer does the same floor/four-exp work under every feature set.

Run remotely through run-heavy with bounded jobs/RAM and quiet timing admission.
Full resource logs are retained. Principal completed runs: native timing 296 s /
0.40 GiB peak RSS; clean ten-group repeat 33 s / 0.13 GiB; ARM timing 138 s /
0.20 GiB; full WASM 257 s / 0.37 GiB; final WASM facade 12 s / 0.09 GiB.
Final x86 suite 19 s / 0.43 GiB; ARM suite 38 s / 0.38 GiB. These are observed
wrapper values, not estimates of isolated kernel memory.

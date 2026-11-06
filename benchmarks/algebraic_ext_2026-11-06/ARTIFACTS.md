# Validation and retained artifacts

Canonical raw archive, checked on **dev**:
`/mnt/v/output/archmage/algebraic-ext-2026-11-06/`.
NAS mirror: `/mnt/tower/output/archmage/algebraic-ext-2026-11-06/` on dev,
backed by `tower:/mnt/user/coefficient/output/archmage/algebraic-ext-2026-11-06/`.
`SHA256SUMS` covers every payload file; `MANIFEST.meta.json` records its digest,
file/byte counts and final source revision. Verification/cleanup logs are retained
outside that payload manifest to avoid self-reference.

Each platform directory has complete timing JSON, output bit triples, MPFR CSV,
compiler IR/assembly, actual timed binary/module disassembly, binary hashes,
full logs and a frozen source archive. `audit.json` checks every expected
workload/form/tier cell, minimum rounds, complete/reliable flags, raw input
identity within oracle groups, frozen binary hashes and unchanged kernel source.
The small tracked CSVs retain results without checking generated dumps or
binaries into git. `published-source-comparison.json` verifies local production
Rust sources against published beta.2: Magetypes 105 files, Archmage 19 files,
macros 30 files, zero differences. This is a Rust-source comparison, not a claim
that package manifests are byte-identical.

| Timing build | Workloads | Cells | Minimum paired rounds | Status |
|---|---:|---:|---:|---|
| x86 off | 8 | 138 | 60 | complete, reliable |
| x86 thin | 8 | 138 | 60 | complete, reliable |
| NEON off | 8 | 92 | 60 | complete, reliable |
| NEON thin | 8 | 92 | 60 | complete, reliable |
| WASM off | 8 | 92 | 30 | complete, reliable |
| WASM thin | 8 | 92 | 32 | complete, reliable |

Tier inventories record scalar/V3/V4x supported on x86, scalar/NEON on ARM,
scalar/WASM128 in WASM, with other requested tiers explicitly unavailable.
Expected tiers are asserted by caller configuration. V4 and generic x1 were
not part of this focused prototype. All six implemented widths have exact
fixtures; timings use generic f32x8 and f64x4 only.

Verification logs and run-heavy resource observations:

| Log | Verified result | Wrapper peak RSS |
|---|---|---:|
| `native/logs/ext-native-check.log` | release fixtures, doctests, clippy, rustdoc, no_std, actual Rust 1.98.1 check, both published unpatched consumers and trees | 0.44 GiB |
| `native/logs/ext-msrv-negative-verified.log` | actual Rust 1.89.0 rejects declared Rust 1.98; expected diagnostic asserted | 0.05 GiB |
| `native/logs/ext-health.log` | xtask generate, validate-registry, validate, soundness | 0.54 GiB |
| `native/logs/ext-final-check.log` | all-feature all-target native tests/clippy, fmt, warning-denied docs; scoped WASM library/kernel/results clippy | 1.58 GiB |
| `arm/logs/ext-arm-build.log` | ARM release fixtures and clippy; dual builds | 0.47 GiB |
| `native/logs/ext-wasm-validate.log` | strict SIMD fixtures and MPFR observations under matching Wasmtime 49.0.5 | 0.06 GiB |
| `native/logs/ext-native-timing.log` | both x86 modes, 65 seconds, exit 0 | 0.09 GiB |
| `arm/logs/ext-arm-timing.log` | both NEON modes, 32 seconds, exit 0 | 0.01 GiB |
| `native/logs/ext-wasm-timing.log` | both WASM modes, 102 seconds, exit 0 | 0.15 GiB |

The first published-fixture attempt lacked a `SimdToken` import. Its failure log
is retained; `ext-native-check.log` records both corrected consumers succeeding.
The ARM default-compiler metadata error and its binary-backed correction are
explicitly retained; see [CODEGEN.md](CODEGEN.md).

Source snapshots are per build. Native, ARM and WASM snapshots can differ in
added wrappers, lockfile resolution or report helpers. The audit verifies the
measured kernels match the final tree. Final documentation/helper edits do not
retroactively change timing provenance. Rust thin-LTO emitted IR is pre-link;
actual linked assembly/WAT is the final-lowering evidence.

No broad rerun follows report edits. Cloud Cargo outputs created by this study
are removed only after binaries, IR and source evidence are archived and the
mirror verified. No source checkout or protected benchmark result is removed.
Draft-only: no push, PR, publish, production source/API change or MSRV change.

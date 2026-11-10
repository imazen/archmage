# Preserved provider evidence

Measurement root: `/mnt/v/output/archmage/powf-external-providers-2026-11-09/`.
Independent mirror: `/mnt/tower/output/archmage/powf-external-providers-2026-11-09/`.
Both copies pass all **8,184** file identities in `CORE-MANIFEST.json`.
Manifest SHA-256:
`e28d6a945f185b5d7cc339f9ab531a30eeacf97b0ea8c7c33fa2a661ea95a07f`.
Later closing records are additions, never replacements for sealed files.

| Directory | Contents |
|---|---|
| `direct-std-rust-libm/` | Direct std/libm runs, literal/opaque traces, twelve execution contexts, complete bit/class/payload differences, failed CXX policy probe |
| `intel/`, `m6/`, `neoverse-n1/`, `amd5900xt/` | Native provider runs or explicit admission failures, raw broadcast/mixed actuals, every stream/chain trajectory, full logs and all retained rounds |
| `inventory-generic-followup/` | Final 234-context ledger: 21,003 differing records, 447,102,610 uncompressed bytes; corrected Gamma predicate |
| `inventory/` | Earlier 199-context inventory retained separately |
| `core-inventory/` | Separate 68-context CORE-MATH/std inventory: 196 differing records, 1,082,571 uncompressed bytes |
| `core-worst/` | Unmodified bounded hard-to-round finder, full output, count preflight and all 515 witnesses |
| Each platform's `core-worst-accuracy/` | All witness outputs and MPFR comparisons |
| `intel/core-fma-only/` | AVX2-disabled execution, raw outputs, instruction trace and executed-FMA proof |
| `glibc-history/` | Tagged scalar-source hashes/diffs, commit history, installed source package, resolved symbol and disassembly |
| `cold-build/` | All twelve fresh-target build logs, client sources/locks, source archives, actual binaries, GNU time/size records |
| `closing/` | Production-head CI record, frozen analyzer and complete source snapshot at 4d0123e5 |

Each platform also retains frozen binaries/DSOs outside Cargo targets, compiler
identities, actual Cargo flags and source manifests. The inherited Neoverse
CPU-target flag remains visible; `accuracy-generic`, `trajectories-generic`
and `identities/generic-followup.json` record the numerical-only generic check.
Failed setup, missing headers, first environment/trace-parser attempts and
partial numerical runs are retained alongside their corrected runs. No dirty
timing round is removed or replaced.

Every accuracy directory has `.actual.gz` files in provider order plus the
input list and summary. Complete per-input output maps preserve all pairwise
variances without a quadratic duplication of records. Trace ledgers retain
every buffer value; chain inputs can diverge after earlier output differences.
Raw outputs also preserve ties. MPFR supplies independent accuracy references;
an observed std reference is explicitly not mathematical truth.

Use `just powf-provider-artifacts verify DIRECTORY MANIFEST` to verify a copy.
The original policy/fused-arithmetic studies remain in their separate sealed
artifact roots; this additional study does not overwrite or requalify them.

## Subsequent isolated C and Rust port study

Separate root: `/mnt/v/output/archmage/powf-arm-c-preport-2026-11-09/`.
Independent mirror: `/mnt/tower/output/archmage/powf-arm-c-preport-2026-11-09/`.
Both copies verify all **5,367 files** in `CORE-MANIFEST.json`, SHA-256
`15f25df1238004d11ea5156536bb9ebeb7932f92d3062bb481dd424a743f8284`.
Later CI/publication evidence is an additive closing manifest.

`intel/` preserves the C-before-port comparison, all five C profiles, Rust
V1/V2/V3 binaries and numerical ledgers, the feature-context callback defect,
its corrected code generation, counters, every noisy pilot and the frozen V1
native retrospective check. `zen5/` preserves locally rebuilt C/native Rust,
full/Gamma and 148 portable cross-context executions, exact Cargo fingerprints,
frozen WASM/i686 executables and runners, actual cargo asm output, hardware
counters, identity negative controls, full trajectories and the first pilot.
`zen5/all-zenbench-bars.md` retains every complete implementation name and raw
ns mean alongside bars copied from zenbench. `neoverse-n1/` contains physical
ARM numerical comparisons, its locally built C profiles and native FMA
disassembly. `sources/` retains versioned source archives.

All result-bit, class and NaN-payload differences remain reconstructible from
the per-width raw outputs and complete variance maps. Numerical comparisons
exercise assumed-input variants only inside their declared Gamma domain.
No timing run qualifies for promotion; setup refusals and flagged rounds are
preserved. Read [ARM-RESULTS.md](ARM-RESULTS.md) for the substantive findings.

## Direct M6 extension

Root `/mnt/v/output/archmage/powf-arm-m6-2026-11-09/` and independent mirror
`/mnt/tower/output/archmage/powf-arm-m6-2026-11-09/` verify **1,452 files** in
`CORE-MANIFEST.json`, SHA-256
`a68d0934d461789a72b1577bcff3da03b1cbb7d1452b86f5eff50c1e3282a19d`.

The root includes the 424-context combined inventory and all paired
comparisons; `m6/` contains full/Gamma/required-NEON raw outputs, every round of
all 32 exploratory cells, the corrected full-native C pilot, exact native
driver features, actual Cargo fingerprints, frozen binaries and disassembly.
`m6/all-zenbench-bars.md` preserves all complete implementation names and raw
ns means. The native Miri failure on both interpreter versions and the stopped
old-snapshot CI attempt are retained. No test skip or threshold change follows.
The counter directory mislabeled x1 explicitly records its actual default x8;
`profile-x1-actual/` contains the corrected explicit-width diagnostic.
Closing publication records are additive. See [M6-NEON.md](M6-NEON.md).

## Canonical HIGHP integration and packed kernels

Root `/mnt/v/output/archmage/powf-highp-simd-2026-11-09/` and independent mirror
`/mnt/tower/output/archmage/powf-highp-simd-2026-11-09/` verify **1,278 files**
in `CORE-MANIFEST.json`, SHA-256
`19a8e9c8d3091c3f2fc3691baa09af63242717aa547095c3a0d8121f8f2170b5`.

The root preserves independent graph answers, every actual output and variance,
MPFR full/hard/Gamma replays, all M6 V1/V2 rounds and bars, both Linux pilots,
N1 native measurements, hardware counters, actual-worker assembly, exhaustive
NaN-family scans, no-FMA/FMA-only execution, both relaxed-engine fusion choices,
default-stack debug WASM checks, full CI and every failed setup attempt.
Binaries/modules and their relocation identities are frozen separately from
Cargo outputs. Cargo caches are excluded. Later publication records are
additive and do not alter this sealed core.

Read the maintained [ISA bounds and performance reference](../../docs/POWF_REFERENCE.md)
and the [integration review](HIGHP-SIMD.md). The earlier provider-only studies
remain separate and immutable. All new performance promotion remains held.

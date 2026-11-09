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

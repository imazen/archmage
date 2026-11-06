# Frozen f32 digest results

Missing coverage at this stage: WASM SIMD128, native V4/V4x8,
and final all-target audits/mirrored archive.
There is no exhaustive f64, arbitrary power-pair, or FMA-ternary claim.

The first audited cross-ISA matrix passes for exp, ln, log2 and pow with exponent
bits `0x4019999a` (f32 2.4). Every cell processes all 4,294,967,296 input patterns
with direct per-output comparison against `zenmath::scalar::reference`.

For each operation, all six paths match in both raw and NaN-canonical output:
ARM scalar reference, packed software, NEON, scalar NEON context; x86 V3 packed
and scalar V3 context. The auditor independently verifies each saved chunk's
checksum, its exact index/start/count, complete ordered coverage, and summary.
The comparator checks matching hash scheme, operation, domain, chunking and
source identity before comparing every canonical chunk and ordered aggregate.

| Operation | Canonical aggregate of chunks |
|---|---|
| exp | `f2a12b6bf62c3027a0aa6ea613dd42de` |
| ln | `1772fc0933b931ce8d3499ea33a943a7` |
| log2 | `8110525308dbc89c3924237426326d95` |
| pow (`y=0x4019999a`) | `09df2fef39e41e89064ab8a83c4a2ed1` |

These are ordered chunk-hash aggregates, not raw output-stream hashes. They are
regression evidence, not an independent accuracy oracle or a correctness proof.
The hash scheme and canonicalization are defined in [README.md](README.md).

Implementation revision: `5a9cc685`. Exact v1 source-manifest SHA-256:
`59e78dde6dee12778662beff2c215a88029e73381bc159fb5dcc77f47a46f64c`.
Both builds use Rust1.99.0 (`b940084d7`), with empty effective native target flags.
The x86 build explicitly disables the harness's AVX-512 feature; V4x is unavailable
on that CPU, a preserved availability failure rather than a numerical mismatch.

| Binary | SHA-256 |
|---|---|
| AArch64 baseline/NEON | `0ec78aeeac918d3ea3ebabffcc5e756667d40a4325d162ccdba4ed13eb713572` |
| x86-64 baseline/V3 | `cce89e4ff23448d76c042c07c499e5557a80bfa13552bc262b3db18a263f8a1c` |

Full audit/comparison log, verified on dev:
`/mnt/v/output/archmage/zenmath-digests-2026-11-06/arm-v3-top-comparison.txt`.
Per-chunk manifests and immutable build identities remain in the run artifacts.
Driver-only failure handling and inspection changes are validated separately;
they do not relabel or replace frozen v1 numerical evidence.

## Primitive cross-ISA audit

All six primitives match canonically across ARM scalar reference, NEON and
x86 V1 (sqrt) / V2 (rounding),18 complete cells. Every cell covers all2^32 inputs
and directly compares every output with the scalar reference. Raw hashes differ
on hardware paths because the numerical contract permits NaN payload/sign
changes; signed zeros and every non-NaN output remain exact.

| Operation | Canonical aggregate of chunks |
|---|---|
| sqrt | `69485215558753e7adc0b302a0f3a431` |
| floor | `b078260f36820b1779c282c0c259ae39` |
| ceil | `af2f0c967ec6608b55106b82598db592` |
| trunc | `9c0153c4b262b99468db79664db63963` |
| round | `6351560f0de86cfad991b2cc25bade50` |
| roundeven | `6797a04535d52f9dfc5f0dd6308f4a6d` |

Audit/comparator log:
`/mnt/v/output/archmage/zenmath-digests-2026-11-06/arm-x86-primitives-comparison.txt`.

The completed native x86 source/identity/results and preserved reference binary
were archived and zstd-verified before deleting its confirmed inactive Cargo
output. Archive SHA-256:
`83d9127e9721b98cb097157f1f50be23322ab05540b2f81a54aadd64e5d2cad1`.
The local artifact and NAS mirror have that same digest:
`/mnt/v/output/archmage/zenmath-digests-2026-11-06/i265-native-source-results.tar.zst`
and `/mnt/tower/output/archmage/zenmath-digests-2026-11-06/i265-native-source-results.tar.zst`.
This is an intermediate archive; WASM and V4 results are not in it.

## Additional fixed exponent families

Each exponent below has complete scalar-reference/NEON/V3 agreement in raw and
canonical hashes, nine full-domain cells total. Every base pattern is covered;
this is not an exhaustive sweep of arbitrary base/exponent pairs.

| Exponent | Exact f32 bits | Canonical aggregate of chunks |
|---|---|---|
| 0.5 | `3f000000` | `33cb939b87e403af3e815cd9aad4a910` |
| 2 | `40000000` | `cdd4ff3a267bfbbb60116c6bf071ddde` |
| -1 | `bf800000` | `38121fc6af1910ef0393d193e7abbfb7` |

Full per-chunk audit/comparator log:
`/mnt/v/output/archmage/zenmath-digests-2026-11-06/arm-v3-pow-families-comparison.txt`.

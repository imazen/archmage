# Frozen f32 digest results

Missing coverage at this stage: WASM SIMD128, native V4/V4x8, audited primitive
and additional power-family cross-ISA comparisons, and final mirrored archive.
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

# Frozen f32 digest results

**89/89 planned full-domain cells pass.** Each cell processes all 4,294,967,296
u32 input patterns in order and directly compares every output with
`zenmath::scalar::reference`. Canonical hashes agree across every selected path.

Not covered: arbitrary power pairs, exhaustive f64/FMA input domains, alternate
rounding modes, or accuracy outside the existing numerical contracts. Hashes are
regression evidence, not an independent accuracy oracle or correctness proof.

| Platform/proof paths | Top-function cells | Primitive cells | Extra power-family cells | Total |
|---|---:|---:|---:|---:|
| ARM reference/software/NEON/scalar context | 16 | 12 | 6 | 34 |
| x86 V1/V2/V3/scalar V3 context | 8 | 6 | 3 | 17 |
| x86 V4x4/V4x8/scalar V4x context | 12 | 0 | 9 | 21 |
| WASM SIMD128/scalar WASM context | 8 | 6 | 3 | 17 |

The V4x8 path uses actual eight-binary64 AVX-512 registers. The final auditor
checks each cell's expected operation/backend/target, exact source, all 4,096
chunk indices/ranges/checksums, complete count, summary and absence of FAILURE.
Its successful log contains 89 AUDITED and 89 MATCH records.

## Accepted canonical aggregates

These are ordered aggregates of chunk hashes, **not raw output-stream hashes**.
NaNs canonicalize to `0x7fc00000`; signed zeros and every non-NaN bit remain exact.
Raw hashes also match for all top functions and power families. Primitive raw
hashes differ only where the contract permits NaN payload/sign changes; every
cell's direct comparison still enforces identical non-NaN outputs.

| Operation / fixed exponent bits | Paths | Canonical aggregate of chunks |
|---|---:|---|
| `exp` | 11 | `f2a12b6bf62c3027a0aa6ea613dd42de` |
| `ln` | 11 | `1772fc0933b931ce8d3499ea33a943a7` |
| `log2` | 11 | `8110525308dbc89c3924237426326d95` |
| `pow-4019999a` | 11 | `09df2fef39e41e89064ab8a83c4a2ed1` |
| `sqrt` | 4 | `69485215558753e7adc0b302a0f3a431` |
| `floor` | 4 | `b078260f36820b1779c282c0c259ae39` |
| `ceil` | 4 | `af2f0c967ec6608b55106b82598db592` |
| `trunc` | 4 | `9c0153c4b262b99468db79664db63963` |
| `round` | 4 | `6351560f0de86cfad991b2cc25bade50` |
| `roundeven` | 4 | `6797a04535d52f9dfc5f0dd6308f4a6d` |
| `pow-3f000000` | 7 | `33cb939b87e403af3e815cd9aad4a910` |
| `pow-40000000` | 7 | `cdd4ff3a267bfbbb60116c6bf071ddde` |
| `pow-bf800000` | 7 | `38121fc6af1910ef0393d193e7abbfb7` |

Power exponent bits are `4019999a` (f32 2.4), `3f000000` (0.5), `40000000` (2),
and `bf800000` (-1). Each covers all bases for that one exponent, not all pairs.

## Source and executable identities

Implementation revision `5a9cc685`; exact frozen v1 source-manifest SHA-256:
`59e78dde6dee12778662beff2c215a88029e73381bc159fb5dcc77f47a46f64c`.
All 89 numerical cells use that identity and Rust1.99.0 (`b940084d7`). Native
builds have no global target CPU flags. Strict WASM enables only `+simd128` and
runs on Wasmtime49.0.5.

| Executable | SHA-256 |
|---|---|
| ARM baseline/NEON | `0ec78aeeac918d3ea3ebabffcc5e756667d40a4325d162ccdba4ed13eb713572` |
| x86 baseline/V3 | `cce89e4ff23448d76c042c07c499e5557a80bfa13552bc262b3db18a263f8a1c` |
| x86 V4x/V4x8 | `255c1bfe4a6b97584eb878c37b06f3a18b3309812b76f43843808360d9fdd368` |
| WASM SIMD128 | `c9a3936ad6e193be0d868f2c803d2249962400ccf9d8ad014e5c917006255819` |

Later bounded integrity/driver/MPFR validation is separate from that matrix:
source stamp `e8d1d37f325c765791ad07c7e61c5c2208cde3b93a19cf3d9d2a694d414f0fc8`.
Eleven native tests, native/WASM all-target clippy with warnings denied, and the
focused WASM test pass. The208 MPFR power pairs preserve independent expected
bits from the existing production fixture. Native V3, V4x and WASM proof paths
pass those expectations; no production fixture or implementation changed.

## Archived evidence

Verified local directory on dev:
`/mnt/v/output/archmage/zenmath-digests-2026-11-06/`.
Verified identical mirror:
`/mnt/tower/output/archmage/zenmath-digests-2026-11-06/`.

The immutable complete archive is `complete-evidence-5a9cc685-v1.tar.zst`
(48,567,128 bytes), SHA-256:
`e3840ba3d02074e5ce413d1024626397624ce144b2ffcfe98895d5d9d4df8e54`.
It retains numerical sources, immutable build identities, executables/runtime,
all per-cell chunks/manifests, full cell logs, historical interrupted attempts,
and the accepted final index/comparison log.

The strengthened `complete-archive-bytes-verification.json` verifies 365,219
required files against originals, including 364,544 chunk files and 91 original
cell logs. There are no missing members. Its compressed per-file SHA-256
manifest is `complete-archive-bytes-verification.files.sha256.gz`; archive,
report, manifest and comparison files have verified identical NAS mirrors.
Verification reports remain sidecars because the archive is immutable.

| Artifact | SHA-256 |
|---|---|
| `final-audit/index.json` | `ed988e3e1957aace18bbfb0fb59b1a9f54d6753040f0af83474b0bd27186ee15` |
| `final-audit/audit-comparison.log` | `aad3987a97f13daf743219d7de38e7603d6bf8766b2aa08431b46f39dad7c0dc` |
| Byte-verification report | `e8daa96adf3f4610e35c50478b3e521cde9c56dcab8cf0d611cba0020243e3fe` |
| Compressed per-file manifest | `3b4e2de6b9e63fe07a3f1ec2eb7afab1105cb53fc9cdbb6980a89887f405443d` |

Intermediate comparator logs are retained and mirrored as
`arm-v3-top-comparison.txt`, `arm-x86-primitives-comparison.txt`, and
`arm-v3-pow-families-comparison.txt`.

## Resource-wrapper evidence

These are harness/job elapsed times and GNU-time peak RSS, not kernel speed
benchmarks. All work used admitted remote hosts with explicit caps and threads.

```text
ARM34-cell queue:     rc=0 947s | peak-RSS 0.03GiB
V3 17-cell queue:     rc=0 166s | peak-RSS 0.02GiB
V4 21-cell build/run: rc=0 376s | peak-RSS 0.31GiB
WASM17-cell queue:    rc=0 832s | peak-RSS 0.28GiB
Final89-cell audit:   rc=0   7s | peak-RSS 0.04GiB
Archive creation:     rc=0  16s | peak-RSS 0.06GiB
Archive byte check:   rc=0  30s | peak-RSS 0.49GiB
```

All changes are confined to this independent unpublished workspace. No push,
release, root justfile edit, production source change or public API change.

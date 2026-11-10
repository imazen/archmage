# Bounded power evidence identities

The 2026-11-09 MIDP/LOWP implementation study has an immutable core containing
2,342 files. All files verify on the independent mirror. Cargo build caches
are excluded; reference binaries, actual WASM modules, source archives, every
retained round, first attempts, failure logs and complete assembly reports remain.
The [readable report](MIDLOW.md) gives the substantive findings and limits.

Resolved on dev, local evidence is `/mnt/v/output/archmage/powf-mid-low-final-2026-11-09/`.
The independently verified mirror is
`/mnt/tower/output/archmage/powf-mid-low-final-2026-11-09/`.
`CORE.json` indexes `core/` with byte lengths and SHA-256 identities:
`5ac5bd029c17e203fbbda50beaa78172e2df3c8e35c0223d052f2afa0220c764`.
The core's final library source is `295001e3`; documentation and reproducibility
harness updates through `f4d766ab` are included in the final tracked-source tar.
Each timing matrix retains its own source and binary identity in `MATRIX.json`.

| Evidence, relative to `core/` | SHA-256 |
|---|---|
| `FINAL-SOURCE.json` | `de67c2fc1e73193c76b3fb8fa2438343c0bc6ebdfbbb9ff9aa86c0fabf4378ae` |
| `all-final-edges-normalized.jsonl` | `05c84d14722da3a3503d1e26d75bc1ce50b63b4c234bee27fcf590f728d91e8e` |
| `native-public-tail-v3-bars.md` | `b1c8cd374b8b51e869cfec08b73152aed0f3701a6abe6fa6b93e0dc7a04c78a5` |
| `mac-native-public-tail-v3-bars.md` | `0162fbae983d946ff81888b4b1ab5768f8a542c54cba01c67a10161384596105` |

The normalized ledger contains every named comparison from the latest 118
cells, including all 50 rounds in each paired confidence interval. Individual
raw JSON files preserve each round's iteration count, execution order,
compensated times and resource-gate observations. Bar reports for earlier
versions, all-policy comparisons and scalar fallback diagnostics sit beside
these two representative native reports. The original absolute-delta diagnostic
is retained; use the normalized ledger for ns per call.

`final-cross/` freezes all five strict and relaxed WASM test modules and engine
logs, including proof of the actual fusion choice in every policy grid.
`final-emulated/` freezes no-FMA Nehalem and i686 binaries and logs. The
`remap-actual-before` and `remap-x86-final` files compare actual frozen x86
entry points; `remap-neon-macro-final-review.json` preserves all scalar and NEON
differences, without loosening instruction normalization. The earlier hand-written
composition comparisons remain separate from the actual-before compiler oracle.

Final CI, warning-sweep and publication records are additive closing evidence;
they do not overwrite or reseal this core. Earlier HIGHP/provider studies retain
their [separate inventories](../powf_providers_2026-11-09/ARTIFACTS.md).

The 61 closing files also verify on the mirror. `CLOSING.json`, indexing
`closing/`, has SHA-256
`68dc055f5dc0dffaf6b5effcee41fd3cfb2a12916278aaa1250e8bc735eb5774`.
It retains the successful 19-stage CI/Miri log, all 24 clean warning records,
initial warning/lint failures, the corrected benchmark verifier, Cargo locks,
compiler observations and tracked-source archive through `58870fcb`.

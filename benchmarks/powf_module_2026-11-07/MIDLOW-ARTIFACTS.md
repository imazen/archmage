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

## Compiler and fresh timing follow-ups

`DEBUG-MEMORY.json` indexes 1710 files in `debug-memory/`, including all 105 actual
WASM modules and the compiler resource trials. Manifest SHA-256:
`90832289b461a4b1ae85f16bee3fa73053f813ab98040badf52885c55f4b530c`.

`QUIET-LOCAL.json` indexes 449 files in `quiet-local/`: the three frozen 64-cell
timing runs, all retained rounds, source/binary identities, complete/partial
analyses, zenbench bars, and the NEON numeric min/max before/after tests and
release assembly. Every file verifies on the independent mirror. Manifest SHA-256:
`17a65a35c3377c424e6ccc3fd1c43ce643186727b8a4f9ab45d0aa5a2a5e3137`.
The [readable follow-up](QUIET-LOCAL.md) names every measured tradeoff.

The pre-squash source history is preserved at
[history/powf-before-squash-2026-11-09](https://github.com/imazen/archmage/tree/history/powf-before-squash-2026-11-09).
PR144 is consolidated before rebasing onto the merged NEON bitmask change.

`FINAL-VERIFY.json` indexes 124 files in `final-verify/`, all independently
verified on the mirror. Manifest SHA-256:
`a8b2b43a49e0bcd20c2fc4e378af939d7178ed195f9bbbe41b0b3e0160e2da05`.
This adds the squashed-source full CI/semver logs, fresh full-policy WASM
(strict and both fusion choices), no-FMA/i686 execution, executed M6 std/no_std
binaries, corrected documentation tests, final native assembly comparison and
the directed legacy exceptional-value ledger. Cargo targets are excluded.
The production-source identities agree across ed20e2b4, eb3ebbdf and 39a95171;
unit-test configuration, documentation and test-fixture updates are recorded
separately in `PROVENANCE.json`.

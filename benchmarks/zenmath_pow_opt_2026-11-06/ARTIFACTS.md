# Artifact and source mapping

Canonical data on the implementation host:
`/mnt/v/output/archmage/zenmath-pow-opt-2026-11-06/`.
NAS mirror:
`/mnt/tower/output/archmage/zenmath-pow-opt-2026-11-06/`.
`MANIFEST.sha256` inventories every retained file except itself; the private
completion report records its final digest and full local/NAS verification.
Large JSON, CSV, TSV, binaries, perf profiles and WAT remain outside git.

Each `frozen-*/identity.json` records exact compiler/flags, source-manifest hash
and copied executable SHA. `source.sha256` paths are repository-relative;
`source.tar.gz` retains those exact files, manifests and locks. The auditor
resolves repository paths from its own location, so the documented just recipe
works from the study directory. Freeze refuses an existing destination and
verifies an atomic binary copy before publishing its identity.

| Evidence | Data path under archive root | Frozen identity |
|---|---|---|
| Initial native power/transfer/feedback | `sources/native/results-native-v1/` | `sources/native/frozen-matrix-v1/` |
| Native replacements | `sources/native/results-native-repeat-v1/`, `timing-transfer-repeat-v2.json` | Same frozen matrix implementation; driver-only group filter |
| Initial strict WASM 60/2,980 | `sources/wasm/timing-wasm-v3.json` | `frozen-wasm-guest-v1` guest `9a2e532978c9a5fa…`; `frozen-wasm-host-v1` host `cd0b2225915b4392…` |
| Final native gamma 28/236 | `sources/native/timing-gamma-native-v6.json`, `timing-gamma-native-repeat-v6.json` | `sources/native/frozen-final-gamma-native-v6/` |
| Final quiet NEON gamma 28/236 | `sources/arm-timing/timing-gamma-arm-v6.json` | `sources/arm-build/frozen-final-gamma-arm-v6/`; timed stripped copy `sources/arm-timing/gamma-timing-arm-v6` |
| Final strict WASM 3/153 on Ryzen 7900X | `sources/native/timing-gamma-wasm-r7900x-v5.json`, `timing-gamma-wasm-r7900x-repeat-v5.json` | Unchanged final v4 guest `acd5c6418a76677d…`, unchanged host `cd0b2225915b4392…` |
| Final exact gamma domains | `sources/{native,arm-build,wasm}/gamma-full-*-v4/` | Matching `frozen-final-gamma-*-v4/`, including separate no-SIMD guest |
| Final 54 cold builds | `sources/wasm/family-build-cost-final-v4/` | Each copied executable SHA in `build-cost.csv`; source `frozen-family-final-v4/` |

Native/NEON v6 source snapshot manifests are respectively
`a1e390fb2ce40d9b33f2469aa87aea6342c69a6019add66ad461d395311d57d9`
and `10f321c962f27bd97288fce01fd2af4c6622d06b8014269997cbd91eec3d21ee`.
The final WASM guest source manifest is
`d63955e9083897fd10cedde8910473285b9f59490c619be49216f4862ce1639c`.
The snapshots' descriptive revision label includes “f7b5f136-plus-unrolled-v4”
where they were frozen before the final commit label; the auditor verifies
all 91 production Rust source files byte-for-byte against `4da80e0a`.
The actual stripped ARM timing file is
`ac0960ee7a24b48d8dfa85c66e9d4bbc2ec638e0e1212b0640ffc501cbf9e753`;
the named unstripped build is `142d25a32d374846…`. Exact `.text` and `.rodata`
identity links the named disassembly to the timed file; see CODEGEN.md.
`run-identities.json` collects the selected original metadata without rewriting
any immutable snapshot.

All numerical/timing builds use pinned Rust 1.99.0 / LLVM 23.1.1. Native
release uses thin LTO, with no target-cpu=native flag. WASM timing exports all
functions using `-Ctarget-feature=+simd128 -Clink-dead-code=yes
-Clink-arg=--export-all` and disables release LTO; that module's actual hash,
not an earlier module with different exports, names the final WAT evidence.
Validation and timing both use retained Wasmtime 49.0.5. The no-SIMD module
has its own binary/source identity and is loaded with SIMD/relaxed SIMD off.

Earlier gamma v1/v2, rejected outlined NEON v3, unrolled v4, once-buffer v5
and final as_chunks v6 timing adapters remain distinct. The v5 array adapter's
variable-length copy cost is diagnostic; final v6 uses checked four-element
chunks. Per-stage source and executable hashes are never replaced by a later
working-tree hash. Earlier i265 final-gamma timing attempts were gate-dirty,
including two targeted encode retries. They remain archived as rejected; the
final targeted set uses one host consistently. Original complete i265 WASM
measurements remain valid for their own earlier source/corpus.

The original freeze build closure retained exact Cargo source/locks but omitted
some validation-only baseline provenance JSON. A separately labeled
`frozen-reproduction-final-v1` source supplement retains the final recipes and
that metadata; it is not another timed binary build and does not overwrite the
old snapshots. The two baseline validators compare all 79/64 frozen Rust files
with committed provenance. Analysis/WAT tools are separately retained under
`tools/`; none are claimed as numerical timing binaries.

Failed build/orchestration logs remain. These include the unavailable initial
runner, missing target, negative feature builds, initial missing cost lock,
rejected dirty timing cells, and the audit's initial metadata typo (902 constants
mistyped as 912 while changing the source-file count). The corrected audit
requires all 902 exact literals; no numerical expectation was loosened. The
first WAT extraction matched a baseline symbol too and stopped in a separate
partial directory; `codegen-final-wasm-v4b` is the corrected exact-module
extraction. No source or result was removed to conceal those failures.

The independent certificate proof archive remains the separately reviewed
[gamma study](../zenmath_gamma_opt_2026-11-06/README.md). Production table
extraction verifies every selected bit against that archived proof/compiled
source. This study preserves its own final full-domain comparisons and streaming
Zenhash digests, with source-pinned test-only hashing and fixed little-endian
output order. Eight-input NEON/WASM adapters are two four-input blocks; V4x
uses the actual wider candidate.

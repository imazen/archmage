# Retained artifacts

Implementation source: `125cdea9672a08238eb7c5d14453f668bfa04183`.
Primary reviewed integration: `8ad71166`. No implementation source changed
following that review. The final report commit additionally hardens the codegen
analyzer to reject missing symbols; the retained-artifact rerun passed with a
byte-identical JSON report. No rebuild was needed.

Local archive root: `/mnt/v/output/archmage/pow-behavior-2026-11-06/`.
Verified NAS mirror: `/mnt/tower/output/archmage/pow-behavior-2026-11-06/`.

- `SHA256SUMS`: 84 files, SHA256
  `04f6e993e97702dd2d1b3d0ee523b62f5438af9325e27f0fd358d24469337f92`.
- `archive-payload.tar.gz`: SHA256
  `7067c5725ee56c74a86ebf04fe876ab656a7996c88dc0280e193421ca4fa92bb`.
  Contains all five target result sets plus guarded reference, corpus, codegen
  binaries/disassemblies/assembly/size/build logs, baseline/V3/WASM checker
  binaries, native benchmark binaries, validation logs, compiler/CPU identity,
  the source-check manifest and complete tracked source archive at `125cdea9`.
- Root timing JSON retains paired rounds and binary SHA256. Root `bin/` retains
  the two ARM timing binaries, verified against their JSON binary identities.
  Native timing binaries likewise match the JSON identities.
- `final-source-before.log` and `final-source-after.log`: all 129 hashes pass,
  byte-identical before/after final native/WASM checks.
- `final-report/`: README, compact CSV/JSON and hardened analyzer. The preceding
  candidate study has its own separately frozen archive and manifest linked
  from its [artifact record](../portable_f32_2026-11-06/ARTIFACTS.md).

Every listed file was SHA256-verified both locally and on the NAS: 84 OK,
zero failures. Mirror used rsync without group/owner changes or deletion.
`verify-local.log` and `verify-mirror.log` retain verification output and are
excluded from their own manifest. Archive packing ran remotely under run-heavy:
exit 0, 8 s, peak RSS 0.01 GiB, minimum available 23,102 MiB, peak load 0.08.

Reverify from either archive root with `sha256sum -c SHA256SUMS`. Binaries,
raw generated result data and corpus remain outside git; compact committed
CSV/JSON files are each below 30 KB. No push, PR or release.

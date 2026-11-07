# Retained evidence and byte integrity

Complete archive, 62,977,246 bytes:
`gamma-evidence-final-v1.tar.gz`
SHA256 `68fcdba50b38deee56daec8d62256794dbccf19c81dbc256611fa88a46c0e78f`.

Copies independently hash-matched on r5900xt, NAS and dev:

- r5900xt:`/home/lilith/tmp/zenmath-gamma-opt-sol/gamma-evidence-final-v1.tar.gz`
- NAS:`/mnt/tower/output/archmage/zenmath-gamma-opt-2026-11-06/r5900xt/evidence/gamma-evidence-final-v1.tar.gz`
- dev:`/mnt/v/output/archmage/zenmath-gamma-opt-2026-11-06/r5900xt/evidence/gamma-evidence-final-v1.tar.gz`

The archive contains every original file in r5900xt task `runs/`: 295 files,
241,691,567 uncompressed bytes. Verification streams the archive and compares
each required member's SHA256 and size against original audited files,
rejecting duplicates, missing members and unexpected types. This includes all
per-path metadata/chunks.tsv/summary/raw logs/auditor logs, the final index,
identities/binaries/source archives, proofs, selected-table extraction,
MPFR inputs/results, complete timing samples and paired analyses. Earlier
failures and synthetic regression fixtures remain separate from accepted runs.

Archive verification under `archive-audits/final-v1/`:

- report.json SHA256 `a4a5f1482d779a8a7a68a8e6b339f9fc71ccb60d56ba33a0f2dcb9a919d9df7e`
- manifest.json SHA256 `bc934707c5550b7dece0a270f02b664999b73d7e14b13054798f3c79a2e19113`

The uncompressed NAS`runs/` mirror also passes every295 original-file hash.
Every manifested source member inside six retained source archives was checked
against its compiled-source identity. No files were skipped.

Mirror verification under `mirror-audits/final-v2/`:

- report.json SHA256 `654459a87fffdba860b4f041be8047ce7ca5f272feb10f926d2286d8d0aff7d8`
- manifest.json SHA256 `8a090872ed2cd48a8f2b7b555aaf8050bd307c7238958ce493d0d67d6a532057`

Both report/manifest pairs independently hash-match r5900xt/NAS/dev copies.
Their common NAS/dev prefix is the r5900xt study directory shown above.
An initial attribute-only rsync failure was preserved; checksum transfer without
permission/group preservation completed rc0 before the successful byte audit.

Resource evidence: archive creation rc0/4s/peakRSS0.00GiB; archive byte audit
rc0/1s/0.03GiB; final NAS mirror byte/source audit rc0/1s/0.03GiB.
All used run-heavy8G/1job and ran after quiet timing/numerical sweeps ended.

Reproduce checks with `verify_evidence_archive.py ORIGINAL_RUNS ARCHIVE NEW_REPORT_DIR`
and`verify_mirror.py ORIGINAL_RUNS NAS_RUNS NEW_REPORT_DIR`. Finalist auditor:
`identity/zenmath-gamma-opt audit KIND FAMILY WIDTH OUT SOURCE_SHA BINARY_SHA`.
Expected plan is `runs/finalist-full-v1/plan.json`; final index and SUCCESS marker
are in that directory. Counter semantics and raw/aggregate scheme are in RESULTS.md.

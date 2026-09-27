# Magetypes token-call inventory, 2026-09-27

Results and migration discussion: [TOKEN-CONTEXT-MIGRATION.md](TOKEN-CONTEXT-MIGRATION.md).

The full source inventory is preserved outside git at `/home/lilith/data/archmage/token-audit/2026-09-27`.
`/mnt/v` and `/mnt/tower` were absent on the audit host (`i265`), so this
capture uses local persistent home storage. R2/Tower mirrors: not created.

- [Detailed findings](/home/lilith/data/archmage/token-audit/2026-09-27/SUMMARY.md)
- [All calls, including enclosing attributes](/home/lilith/data/archmage/token-audit/2026-09-27/calls.csv)
- [Plain enclosing functions](/home/lilith/data/archmage/token-audit/2026-09-27/plain-functions.csv)
- [Snapshot and exclusion manifest](/home/lilith/data/archmage/token-audit/2026-09-27/manifest.json)
- [Reproduction instructions](/home/lilith/data/archmage/token-audit/2026-09-27/README.md)
- [Per-file SHA-256 checksums](/home/lilith/data/archmage/token-audit/2026-09-27/SHA256SUMS)

SHA-256 of `SHA256SUMS`: `c9ee773e51b0cc894b210b7084296f6a6aa0a5eed14397042f0c4a0eec23347a`.

The checksum manifest covers all 2584 preserved files, including the
scanner, source snapshots, original remote archive, and call records.
Absolute snapshot paths embedded in the records refer to the original capture
under `/home/lilith/tmp/magetypes-token-audit-2026-09-27`; the same relative source trees are preserved here under
`snapshots/`. Relative script paths work from this preserved directory.

Provenance: ripgrep 15.2.0; Python 3.12.14; tree-sitter 0.25.2;
tree-sitter-rust 0.24.2. Each source record has its checkout commit and exact
source URL. The primary inventory uses 16 local snapshots plus remote
`zenav1-aom` commit `66f0661e79590cee2a7055bfb028a216bb61c239`.
Three extended snapshots represent two further origin repositories.

This is source inspection, not evidence of downstream compilation. The parser
and alias-inference limitations are recorded in the findings and reproduction
instructions. No downstream source edits were made.

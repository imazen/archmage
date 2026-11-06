# Retained evidence

Study source snapshot: `c94b222f` (full repository tar, including its unpublished
zenmath/magemath dependency sources). Initial result harness snapshot: `1f60c7f9`.
Both are in the archive's `source/` directory. Candidate versions are pinned in
Cargo.lock; `identity/` records actual compiler, libc, OS and runtime versions.

Local archive: `/mnt/v/output/archmage/portable-f32-2026-11-06/`.
NAS archive: `/mnt/tower/output/archmage/portable-f32-2026-11-06/`.

`SHA256SUMS` contains1797files. Every listed NAS file was read and verified:
1797OK,0failures. Manifest SHA256:

```text
ba5b60c32db3240fc41bc17415102bf4965877b7bba277ba0d2fd49b12a59e88
```

The first archive-mode rsync copied content but could not set NAS groups. A
`rsync -rt --omit-dir-times` retry preserved file contents without ownership
changes; full SHA256 verification then passed. No deletion was used.

Inputs/outputs, raw zenbench rounds, source archives, binaries, per-level search
samples, independent directed-MPFR caches, minimized trap and full job logs are
retained. Compact generated text tables are in `results/` (each below30KB).
No image, executable, archive or large generated dataset was committed.

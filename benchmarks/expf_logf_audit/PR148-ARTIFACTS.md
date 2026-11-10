# PR148 immutable evidence

Frozen production: `a052a8ccfee0c463b176265c28392c755b02ee10`.
Merged-main reference: `91a13739b73f46213ecfe7083c77297c531fc8ad`.
Review/source archive: `dce64960` (source sealing and review documentation).

The sealed dataset is outside Git:

- `/mnt/v/output/archmage/expf-logf-pr148-review-2026-11-10`
- durable mirror: `/mnt/tower/output/archmage/expf-logf-pr148-review-2026-11-10`

`CORE-MANIFEST.json` records **1,898 files / 689,375,453 bytes**, excluding
its own bytes. SHA-256:

```text
a99ad8cd3120d9d2bb9399bab51a99905874841e5475e436cd398534bb90fbf0
```

Every listed file's size and hash was verified on both copies. Do not modify
this dataset when adding later evidence. The earlier PR147 dataset is separate
and unchanged; see [ARTIFACTS.md](ARTIFACTS.md).

This set retains submitted/ordered/policy matrices on AMD, Intel and M6, and
restricted-range matrices on AMD and Intel; all rounds, failed admission cells,
raw zenbench bars, paired confidence analyses, CPU placement records, exact-model
outputs, MPFR summaries, full NaN scans, failing contract tests, successful
library checks, source archives, real executed binaries and both failed and
successful build orchestration logs. Prepared layer layouts preserve the common
adapter and no_std-only repair separately from the submitted implementation.

Source/binary distinctions matter:

| Experiment | Source identity |
|---|---|
| AMD original timing / policy tools | `0ea951b5` / `01f91cbd` |
| Intel/M6 original copied source | `38129b882e0d105d527fac5407048e724ac77c16` |
| Ordered comparison patch | `82fa064d` |
| Full NaN scans / copied ordered worker source | `d4006552` |
| Explicit range prototype | `26eb088b3189c22cd7d8e29ef1ce05ba945a788f` |
| Layer preparation tools | `f581cb7e` |

The archived 38129b88 tree contains the same compiled original timing/policy
sources as the earlier AMD builds plus inspection tools. Each `MATRIX.json`
records the actual executable SHA-256; matched retained executables were checked
against those identities. The source archives and pushed review history allow
reconstruction without relying on a moving author branch.

Layer `builds.json` retains all 45 build/run outcomes and stage digests. The 27
retained consumer executables are the final binary in each repeat/profile/layout;
repeat 0 reflects the math-edit rebuild. Earlier stage binary hashes are recorded,
but their overwritten executables are not claimed to be separately retained.
See [MAGEMATH-LAYER.md](MAGEMATH-LAYER.md) for this study's controls and limits.

The expanded Intel range analysis adds two direct relaxed/main comparisons after
the initial 14-edge analysis. Both analyses and the unchanged raw observations
are retained. This is not an independent replication or a new timing run.

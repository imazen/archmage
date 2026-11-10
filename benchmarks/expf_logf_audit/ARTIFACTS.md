# Frozen evidence

Data root: `/mnt/v/output/archmage/expf-logf-audit-2026-11-10/`.

`CORE.sha256` covers 542 files totaling 259,474,947 bytes. Manifest SHA-256:

```text
729143bcd9be053d5b0ed183aec7971eba9cf7a995256b367ea253f675023259
```

All 542 file hashes were independently checked in the working data root and a
second disk copy. AMD and M6 measured executable hashes match their frozen
`MATRIX.json` manifests. The sealed data is preserved without later edits.

The archive contains original and failed check logs, all 96 AMD and 96 M6 timing
cells and raw rounds, paired confidence analyses, exact-graph/MPFR inputs and
outputs, all cross-ISA output dumps, dead-quadratic witnesses, debug-provider
differences, three alternating cold-build pairs, codegen disassembly and symbols,
source archives, executed binaries and representative debug rlibs.

Source identities:

| Role | Commit |
|---|---|
| Reviewed production | `0464774e7c5c63a398041ddaa131b9bd4f7035c7` |
| Frozen main | `91a13739b73f46213ecfe7083c77297c531fc8ad` |
| AMD timing harness | `a9ed1711` |
| M6 timing harness | `55809e38` |
| Wrapper/corpus harness | `cbe970e4` |

The artifact's source archives preserve these revisions independently of branch
movement. Large generated output and compiled executables are deliberately not
committed to Git. Read [REVIEW.md](REVIEW.md) and [BARS.md](BARS.md) for the findings
and readable timing tables.

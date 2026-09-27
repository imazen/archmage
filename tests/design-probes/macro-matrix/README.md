# Macro capability inventory

Compile-only reproduction for the [capability matrix](../../../docs/TIER-SELECTED-TYPES.md#capability-matrix-and-unification-review-2026-09-27).
Library revision: `b7353c1be2b6990a13757839e65740e4230fa687`, Rust 1.98.1,
x86_64 Linux. No SIMD code is executed. The capability source and compiler
inventory are checked in; this excluded probe package is not a regression-test
gate. Individual feature selections deliberately reproduce rejected syntax or
unavailable backend combinations. Running this probe with `--all-features` is
expected to fail.

| Feature selection | Observed compilation |
|---|---|
| None | Accepted: explicit gated autoversion, generic tokenless rite, tokenless magetypes(rite) |
| `avx512` | Accepted |
| `ungated-autoversion` | Accepted: implicit vector backend gate now applied |
| `ungated-autoversion,avx512` | Accepted |
| `signature-alias` | Rejected: body-local f32xN unavailable in signature |
| `tokenless-arcane` | Rejected: arcane does not accept a tier argument |
| `magetypes-without-token` | Rejected: ordinary magetypes variants need arcane's token proof parameter |
| `rite-placeholder` | Rejected: rite does not substitute Token |
| `rite-tier-gate` | Rejected: rite does not parse per-tier cfg gates |

The [CSV](results-gating-fixed-2026-09-27.csv) records actual exit codes and diagnostic
summaries. [Metadata](results-gating-fixed-2026-09-27.meta.json) records the library revision,
Rust version, source hashes, and command. Full diagnostics remain at
`/home/lilith/data/archmage/macro-matrix/2026-09-27-gating-fixed/`.
The [original inventory](results-2026-09-27.csv) and its
[metadata](results-2026-09-27.meta.json) preserve the pre-fix rejection of
`ungated-autoversion` at `e3552516`. The feature name is retained for reproduction;
it now means the source omits an explicit gate, not that the expansion is ungated.

Reproduce from the repository root on an x86_64 host, choosing a new output path:

```sh
TMPDIR="$HOME/tmp" "$HOME/work/zen/scripts/run-heavy" --mem 16G --jobs 8 -- \
  python3 tests/design-probes/macro-matrix/inventory.py \
  --output "$HOME/data/archmage/macro-matrix/new-inventory"
```

Use a memory cap appropriate to the host. `just macro-matrix-inventory OUTPUT`
provides the same inventory command; invoke it under the heavy-job wrapper.
The script records every case's compiler output as it completes and makes no
pass/fail claims about the rejected cases. Inspect the resulting CSV against
the documented capabilities when investigating a change.

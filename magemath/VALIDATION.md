# Extraction validation — 2026-11-10

Scope: configurable f32 powf, expf and logf ownership, explicit caller/vector
entry points, and deprecation of the corresponding retained Magetypes APIs.
No reciprocal/root replacement, new polynomial fit, new range profile or
whole-matrix performance promotion is part of this PR.

Missing evidence first: the complete 19-step `just ci` gate was not rerun after
correcting its inherited Miri oracle failure. That run passed steps 1–14 and
stopped at step 15; steps 16–19 did not execute in that run. Windows ARM64 and
i686 checks are represented in CI, not claimed as local native executions.
No new wall-clock performance comparison was collected for this extraction.

Completed checks:

- `just magemath-check`: std/no_std, debug/release contracts, strict Clippy,
  canonical powf fixture selection, all 32 new exp/log const/runtime encodings,
  exact rejection masks and caller-preserving vector/scalar tails.
- Native Apple M6 std and no_std tests, including its NEON entries.
- Strict WASM std and no_std tests, including its SIMD entries.
- Public API snapshots generated for x86-64, ARM64 and WASM.
- Magetypes compile-fail deprecation documentation, plus its retained policy
  regression witnesses.
- The failed Miri `logf::tests::log1p_basics` test passes after pinning the
  MPFR256-rounded f32 `log1p(0.5)` reference to `0x3ecf991f`. The assertion
  remains bit exact; its native expected value is unchanged.
- Both strict compatibility-driver Clippy commands pass after marking their
  intentional deprecated-API coverage. That addresses the first CI run's
  three Clippy job failures; it does not hide numerical test failures.

Source identities: native ARM production/tests were checked at `4b2ecd25`.
`2aa028a6` then changed only the deterministic test oracle, its recipe and
migration documentation. `640a0de6` adds compatibility-driver lint annotations
and their recipe/documentation; production math sources are unchanged by these
follow-ups. Existing precision fixtures and thresholds remain intact.

Recorded resource-wrapper lines:

```text
Scoped native/std/no_std + strict-WASM checks:
run-heavy: done rc=0 47s | peak-RSS 2.83GiB | min-avail 36310MiB | peak-load 1.79
Strict WASM no_std:
run-heavy: done rc=0 15s | peak-RSS 3.16GiB | min-avail 36243MiB | peak-load 4.05
Apple M6 std/no_std:
run-heavy: done rc=0 14s | peak-RSS 1.02GiB | min-avail 8604MiB | peak-load 1.95
Focused Miri correction + native oracle + deprecation doctests:
run-heavy: done rc=0 8s | peak-RSS 0.67GiB | min-avail 38708MiB | peak-load 0.73
Compatibility diagnostics Clippy:
run-heavy: done rc=0 4s | peak-RSS 0.44GiB | min-avail 39019MiB | peak-load 0.42
```

The Mac uses a monitored process-tree RSS guard, not a Linux cgroup hard cap;
its available-memory value is the wrapper's labeled reclaimable-page proxy.
These are validation resource records, not kernel performance comparisons.

Logs and the source archive are preserved outside Git at
`/mnt/v/output/archmage/magemath-migration-2026-11-10`, mirrored to
`/mnt/tower/output/archmage/magemath-migration-2026-11-10`.
`MANIFEST.json` covers 37 files / 34,744,324 bytes, excluding itself; SHA-256:

```text
208a19b3619d5dbc6143dffb1f839353a8bcc960fd41b20905318e3c31ede66d
```

The preceding [ownership compile study](https://github.com/imazen/archmage/blob/1662daa4d14f/benchmarks/expf_logf_audit/MAGEMATH-LAYER.md)
is separate evidence with its own controls and source identities. It is not
presented as a timing run of this final extraction.

## Compatibility composition follow-up

The subsequent CI WASM job exposed the stack's legacy power-composition mismatch.
Commit `640c07fd` restores legacy coefficients and backend arithmetic in the
retained Magetypes LOWP and unchecked MIDP exponential delegates. Magemath's
configured canonical fits are unchanged. The existing `powf_engine_remap`
bit assertions pass on strict WASM after this correction; no expectation was
changed. The test now includes operation/exponent/input diagnostics on failure.
This supersedes the earlier note that follow-ups only changed test/lint files:
the new compatibility-only change is explicit and separately validated.

The strict-WASM targeted run reports:

```text
run-heavy: done rc=0 2s | peak-RSS 0.37GiB | min-avail 39305MiB | peak-load 0.15
```

The same unchanged composition test and policy regression witnesses pass on
native x86 after the compatibility correction (3s, peak RSS 0.63 GiB,
minimum available 38741 MiB, peak load 0.34). ARM verification of this
compatibility-only follow-up remains with CI; the earlier native M6 evidence
covers the unchanged Magemath production engine. Separate follow-up logs and
source identity are preserved under `magemath-migration-2026-11-10-followup`
in the same local/mirrored output roots; the first archive remains immutable.

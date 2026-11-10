# Ownership validation — 2026-11-10

Scope: rehome the unpublished configurable f32 powf, expf and logf framework,
its tests and oracle drivers into Magemath. Magetypes retains the flat math
implementations actually published in 0.9.31, with deprecation metadata for
replaced families. No configurable compatibility copy remains in Magetypes.

No new wall-clock comparison was collected for this ownership correction.
The complete repository `just ci` gate has not been rerun locally. Native
M6 results from the earlier extraction revision do not qualify the corrected
Magetypes boundary; current ARM execution remains with CI. Windows ARM64 and
i686 are likewise CI checks, not claimed as local executions.

## Current checks

The implementation boundary was corrected in `f6df3ec0`. Subsequent edits remove
unused imports in relocated tests/tools, annotate intentional deprecated flat
method diagnostics, and update documentation/API snapshots.

- Magemath's complete default test selection and doctests pass with `std avx512`
  and without std (`--no-default-features --features w512`). This includes the
  relocated independent power fixtures, HIGHP fixtures, policy rejection masks,
  const/runtime exp/log contracts and scalar/vector/tail identity checks.
- `just magemath-public-boundary` passes: published-body/source-boundary guard,
  retained Magetypes flat-math and power-composition tests, doctests including
  configured-module absence, and strict all-target Clippy for both crates.
- The retained power-composition assertions also pass on strict SIMD WASM via
  `just magemath-legacy-remap-wasm`. No expected bits or error bounds changed.
- `just magemath-compat-clippy` passes for the moved oracle drivers with std
  and without std. Configured drivers call Magemath; the legacy exceptional-value
  driver continues to call Magetypes flat methods.
- Public API listings were regenerated for x86-64, ARM64 and WASM. None of the
  Magetypes supported, feature-added or internal listings contains configured
  powf/expf/logf modules, `PowfOptions` or `try_powf*` receivers. Magemath owns
  those operation modules and explicit token-taking vector functions.
- FMA, bounded-power and HIGHP fixture generators pass their unchanged `--check`
  comparisons at their new owner paths. Formatting and CI workflow contracts pass.

The published-body guard compares all 35 flat transcendental bodies on each
f32x4/x8/x16 width against the crates.io 0.9.31 archive. Its source provenance is
commit `6bfd83d744e41259ec758054631dcb0928fc4ea1`, also tag `v0.9.31`.
The hash normalization removes comments/whitespace and external deprecation
metadata; it retains arithmetic tokens. This establishes body preservation,
not equivalence across arbitrary compilers or changes to lower-level primitives.

The moved Miri unit/HIGHP/error suites are selected at Magemath in `cargo xtask
miri`; the existing numerical assertions remain intact. Full Miri execution is
not claimed for this final correction. The moved log1p fixture retains the
MPFR256-rounded `log1p(0.5)` bits `0x3ecf991f` instead of using Miri's approximate
std-math shim as its oracle.

Final driver-Clippy, formatting and strict-WASM resource-wrapper record:

```text
run-heavy: done rc=0 5s | peak-RSS 0.45GiB | min-avail 41743MiB | peak-load 1.13
```

This is a validation resource record, not a math-kernel speed comparison.
Current logs are preserved in `magemath-public-boundary-2026-11-10` under the
local/mirrored output roots named below, with a source archive and SHA-256 index.

## Earlier evidence and limits

The earlier extraction retained unpublished configured modules in Magetypes.
Its native M6 and strict-WASM runs, compatibility follow-up and source archives
remain preserved under `magemath-migration-2026-11-10` and
`magemath-migration-2026-11-10-followup`, beneath
`/mnt/v/output/archmage` and `/mnt/tower/output/archmage`. They describe their
recorded source revisions and are not presented as new runs of this correction.

The [ownership compile study](https://github.com/imazen/archmage/blob/1662daa4d14f/benchmarks/expf_logf_audit/MAGEMATH-LAYER.md)
is separate evidence, with its own implementations and source identities.
No compile-time or runtime ratio is projected from that prototype onto the final
crate boundary. Historical benchmark harnesses must be replayed at their recorded
source revisions; their old Magetypes module paths are not current Magemath APIs.

LOWP log1p's near-zero bias, Specials subnormal limitations, and the distinction
between component and composed accuracy remain documented in the
[migration guide](../docs/MAGEMATH-MIGRATION.md). This change does not refit or
promote numerical profiles, add reciprocal/root families, or publish a crate.

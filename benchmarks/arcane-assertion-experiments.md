# Arcane constant assertion experiments

Both experiments start at PR #86 commit `15f77a3`. Neither changes the production
macro on PR #86. The local experiment commits and reproducible patches are:

- No assertion: `71c72a1`, branch `experiment/arcane-no-assert`,
  patch `arcane-no-assert.patch`. This deliberately drops validation, for timing only.
- Shared assertion: `94c2b6b`, branch `experiment/arcane-shared-assert`,
  patch `arcane-shared-assert.patch`.

## Shared-assertion code change

The token generator emits the following once per token (including foreign-arch stubs):

```diff
 impl X64V3Token {
     pub const __ARCHMAGE_TIER_TAG: u32 = 0xF38B284B;
+
+    #[doc(hidden)]
+    pub const __ARCHMAGE_ASSERT_TIER_F38B284B: () =
+        [()][!(Self::__ARCHMAGE_TIER_TAG == 0xF38B284B) as usize];
 }
```

The macro substitutes this inside every method:

```diff
- const _ARCHMAGE_TOKEN_MISMATCH: () =
-     [()][!(X64V3Token::__ARCHMAGE_TIER_TAG == 0xF38B284B) as usize];
+ let _: () = X64V3Token::__ARCHMAGE_ASSERT_TIER_F38B284B;
```

The expected tag, not the supplied type's actual tag, determines the constant name.
A weaker token therefore lacks the requested constant. Referencing a constant
still creates work, but avoids a new constant item/body for each expansion.
The full supplied type path is preserved, including dependency renames/re-exports.
This experiment shares checks in the token definitions; ordinary concrete-token
macro uses benefit as well as generated magetypes methods.

## Measurements

Ran removal first, then sharing. Each experiment had its own unchanged-baseline
runs. Six alternating runs per variant and command, cached dependencies, only
magetypes cleaned, incremental and compiler wrapper disabled, no concurrent builds.
Same Ryzen 9 9950X3D machine as earlier PR86 measurements.

Uninstrumented stable rustc 1.98.1, `--locked -p magetypes --features avx512`:

| Experiment | Command | Baseline median (range), seconds | Candidate median (range), seconds | Change |
| --- | --- | --- | --- | --- |
| Remove assertion | check | 1.124 (1.054–1.128) | 1.020 (0.948–1.025) | −103 ms / −9.2% |
| Remove assertion | release build | 1.336 (1.314–1.342) | 1.225 (1.160–1.235) | −111 ms / −8.3% |
| Shared constant | check | 1.119 (1.054–1.127) | 1.031 (0.971–1.042) | −87 ms / −7.8% |
| Shared constant | release build | 1.333 (1.264–1.354) | 1.242 (1.233–1.250) | −92 ms / −6.9% |

Shared checks recover most of the observed saving from removal. These were separate
paired experiments, so the difference between candidates is approximate, not a
head-to-head timing. No cold-dependency build saving is claimed: sharing introduces
constant definitions in the token crate, whose compilation was warmed first.

Nightly-2026-09-02 self-profile medians, dev library build, default events:

| Experiment | Event self time | Baseline | Candidate |
| --- | --- | --- | --- |
| Remove assertion | expand_proc_macro | 90.92 ms | 82.41 ms |
| Remove assertion | typeck_root | 270.34 ms | 244.24 ms |
| Remove assertion | mir_borrowck | 125.09 ms | 116.54 ms |
| Shared constant | expand_proc_macro | 93.32 ms | 86.49 ms |
| Shared constant | typeck_root | 274.64 ms | 249.37 ms |
| Shared constant | mir_borrowck | 126.91 ms | 116.10 ms |

Both candidates reduce `typeck_root` event count from 13,809 to 12,336 and
`mir_borrowck` from 8,666 to 7,193: **1,473 fewer bodies** for each query, matching
the 1,473 arcane invocations on x86. This is consumer-crate work, not total
workspace work. Nightly instrumented and stable wall-clock times are different
measurements and should not be subtracted from one another.

## Checks and adoption limits

Shared-constant probes accepted genuine tokens through a renamed dependency,
including normal parameters and owned/borrowed receivers. They rejected weaker
tokens aliased as X64V3Token in both ordinary and receiver methods. A fake type
copying only the old numeric tag was also rejected. Removal accepted all of these
invalid inputs, confirming that removal is only a performance experiment.
The shared constant is public and can itself be forged; this is not a solution to
the previously discussed authenticity problem.

The shared variant passed all 41 arcane/receiver integration tests, ARM compilation
with avx512 enabled, regeneration idempotence, formatting, and whitespace checks.
Compilation benchmarks cover x86 AVX-512 check/release. Full expansion snapshots and
public API snapshots were not updated for this experiment: both intentionally
change. Existing negative diagnostic snapshots would need to change from a const
evaluation error to a missing-associated-constant error for weaker aliases.

Adoption adds hidden public constants and requires compatible macro/token versions.
An older token crate paired with a newer macro would not have the constants. The
current macro dependency requirement is a compatible version range, so that
publication/versioning issue must be resolved before shipping this form.

## Reproduction and artifacts

Apply either patch to a separate worktree at `15f77a3`, then run:

```sh
python3 benchmarks/pr85-arcane-compile.py /path/to/baseline /path/to/candidate /tmp/wall.json
python3 benchmarks/profile-arcane.py /path/to/baseline /path/to/candidate /tmp/fresh-profiles
```

The shared patch includes generated token output; `cargo run -p xtask -- generate`
reproduces it. No generated source was post-processed.

Raw data: `arcane-no-assert-compile.json`, `arcane-shared-assert-compile.json`,
`arcane-no-assert-profile.json`, `arcane-shared-assert-profile.json`.
Probe outcomes: `arcane-assertion-probes.json`; local source/logs are under
`/tmp/arcane-assert-probes`. Full trace directories are
`/tmp/arcane-no-assert-profile` and `/tmp/arcane-shared-assert-profile`.

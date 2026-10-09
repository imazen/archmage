# Structured attune parser measurements — 2026-10-09

Baseline: `d468e9f2d2fca3021299eb5f5c03f9aaabbf44e9`.
Candidate: `7c45b2a9b10836027cd917ab64cd3c4f86f75a95`.
Harness lock preparation: `625c17f3b85ad278d96c362e4ecc4dce67e4573a`.

Host: i265, Core Ultra 7 265K, x86-64 Linux, 20 logical CPUs.
Rust/Cargo versions and exact commands are recorded in `results.json`.
Runs were serial under run-heavy with a 16 GiB memory cap and eight build jobs.

## Compile fixture

```sh
just attune-compare d468e9f2 7c45b2a9 \
  ~/tmp/attune-parser-cost-refreshed-2026-10-09 3 --refresh-lock
```

Three alternating baseline/candidate pairs per feature configuration. Every cold
run used a new Cargo target directory; the consumer-only stage touched only the
unchanged fixture source. All 24 checks passed. Dependencies and source hashes
are recorded alongside each observation in `results.json`.

| Fixture | Stage | Baseline median (s) | Candidate median (s) | Change |
| --- | --- | ---: | ---: | ---: |
| Macros | Cold | 1.80264 | 1.81086 | +0.46% |
| Macros | Consumer only | 0.03792 | 0.03795 | +0.08% |
| Magetypes + AVX-512 | Cold | 3.18241 | 3.19059 | +0.26% |
| Magetypes + AVX-512 | Consumer only | 0.04149 | 0.04134 | -0.37% |

Observed ranges overlap in each comparison; `summary.csv` contains ranges and
maximum RSS from `/usr/bin/time -v`. This small fixture uses legacy attributes;
it checks the cost of building the larger macro crate and its existing consumer
path. It does not measure a large consumer migrated to selector-local attune
syntax. Three pairs do not establish a general no-regression guarantee, and the
broader rewrite's compile-time acceptance gate remains open.

The first attempt failed before compilation because the fixture lock still named
workspace crates at 0.9.29. The optional preparation step resolved that lock
offline, outside timing, and copied it to both archives. Only the three workspace
package versions changed to 0.9.30; external dependency versions were unchanged.
Timed commands retained `--locked`. The shared lock is committed here.

## Expansion allocations

The existing `just attune-profile` allocator probe runs 1,000 expansions per
case using proc_macro2's standalone backend. `allocations.csv` retains all ten
cases and their single-run timing observations.

- Raw attune, wrapped attune, and every measured legacy case have unchanged
  allocation counts and bytes.
- Direct family: 171 to 170 allocations, 11,180 to 11,179 bytes per expansion.
- All outputs: 585 to 584 allocations, 32,539 to 32,537 bytes.
- Composed family: 743 to 742 allocations, 42,723 to 42,721 bytes.

These are test-allocator counts, not complete process allocation measurements.
They exclude input lexing and may miss dynamically linked library allocations.
The existing cases use grouped syntax; this comparison measures the refactored
parser against equivalent pre-existing declarations, not all possible new forms.

Resource-wrapper output:

```text
baseline checks/profile: rc=0 2s | peak-RSS 0.05GiB | min-avail 28904MiB | peak-load 0
candidate verification:  rc=0 80s | peak-RSS 0.33GiB | min-avail 28482MiB | peak-load 1.63
cold comparison:         rc=0 31s | peak-RSS 0.37GiB | min-avail 28201MiB | peak-load 0.64
```

## Correctness and provenance

`just attune-parser`, `just attune-clippy`, `just attune-profile`, and
`just attune-compat` passed, along with scoped formatting and diff checks.
Existing tests and snapshots were not weakened or rewritten. New grammar tests
cover the parse/resolve boundary, invalid combinations, equivalent spellings,
option permutations, and every registered tier. Integration tests check generic
functions, receivers, associated functions, gates, naming, and composition under
`forbid(unsafe_code)` and `deny(warnings)`, with AVX-512 support off/on.

Full logs and per-process time reports remain at the paths and SHA-256 hashes in
`raw-artifacts.json`. The validation log exceeds the repository's ordinary file
size limit and is referenced rather than copied. Native execution was x86-64;
these results do not claim cross-platform or AVX-512 runtime coverage.

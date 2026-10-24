# Cold consumer compilation after the warning cleanup — 2026-10-24

Question (review, 2026-10-24): the macros now emit more markup (lint attributes,
rewritten-call passthroughs). Does that slow downstream compilation?

Result: no measurable change in consumer crates. Every workload's median is within
±0.3% of the baseline and every baseline/candidate range overlaps. Compiling
`archmage-macros` itself costs 0.01–0.03 s more per cold build.

| Workload | Mode | Before (s) | After (s) | Change | Before range | After range |
| --- | --- | ---: | ---: | ---: | --- | --- |
| magetypes | check | 2.557 | 2.564 | +0.007 s (+0.26%) | 2.554–2.560 | 2.539–2.565 |
| zenav1-svt | check | 7.864 | 7.874 | +0.010 s (+0.13%) | 7.856–7.954 | 7.868–7.956 |
| rav1d-safe | check | 7.244 | 7.257 | +0.013 s (+0.18%) | 7.218–7.246 | 7.214–7.259 |
| magetypes | release | 2.794 | 2.796 | +0.002 s (+0.06%) | 2.784–2.810 | 2.783–2.813 |
| zenav1-svt | release | 21.887 | 21.910 | +0.023 s (+0.11%) | 21.794–21.913 | 21.888–21.918 |
| rav1d-safe | release | 10.546 | 10.524 | −0.022 s (−0.21%) | 10.539–10.592 | 10.473–10.562 |

Medians of three cold library builds per variant, rotating order.
Before: `da0ed5de` (0.9.31-beta.1 state). After: `0d50cb13`.

## Where the time goes

Cargo unit medians (seconds). Units overlap, so they do not sum to wall time.

| Workload | Mode | Unit | Before | After |
| --- | --- | --- | ---: | ---: |
| magetypes | check | archmage-macros | 0.42 | 0.43 |
| magetypes | check | magetypes | 0.87 | 0.86 |
| magetypes | release | archmage-macros | 0.41 | 0.42 |
| magetypes | release | magetypes | 1.09 | 1.10 |
| rav1d-safe | check | archmage-macros | 0.47 | 0.48 |
| rav1d-safe | check | rav1d-safe | 4.35 | 4.37 |
| rav1d-safe | release | archmage-macros | 0.46 | 0.48 |
| rav1d-safe | release | rav1d-safe | 7.48 | 7.45 |
| zenav1-svt | check | archmage-macros | 0.42 | 0.42 |
| zenav1-svt | check | magetypes | 1.18 | 1.16 |
| zenav1-svt | release | archmage-macros | 1.54 | 1.57 |
| zenav1-svt | release | magetypes | 1.41 | 1.40 |

rav1d-safe calls `incant!(... without token)` inside `#[rite]` bodies in hot
files, so it exercises the rewritten-call passthrough
(`incant!(@__archmage_expanded ...)`), which adds one bang-macro expansion per
rewritten call.

## Generated output size

Bytes of the raw expansion snapshots (`tests/expand-raw/`, 161 inputs):
187,370 before; 202,682 at the interim `efef2ceb`; 188,841 after (+0.8%).
The interim growth was 200 `#[allow(clippy::inline_always)]` and 200
`#[allow(clippy::used_underscore_binding)]` on forwarding wrappers. `11d5122c`
removed both: forwarded `_name` parameters take reserved names, and clippy does
not report `inline_always` on boundary wrappers. Generated magetypes sources
shrank from 2,860,247 to 2,805,608 bytes (path shortening, dead scalar namespace).

The interim run (`efef2ceb`, before that change) showed magetypes check +0.83%
and release +1.05% with barely overlapping ranges; the final state removes that
difference. Raw data: [final](consumer_compile_lint_cleanup_2026-10-24.json),
[interim](consumer_compile_lint_cleanup_interim_2026-10-24.json).

## Method

- Intel Core Ultra 7 265K (`i265`), idle (load 0.03 before each phase),
  Rust 1.99.0, eight Cargo jobs, serial runs under
  `run-heavy --mem 16G --jobs 8` (check 106 s, peak RSS 0.86 GiB, peak load
  1.29; release 211 s, peak RSS 1.16 GiB, peak load 2.66).
- Same harness, pinned consumer sources and normalization as the
  [beta study](consumer_compile_beta_2026-10-10.md): zenav1-svt `224c6bbb`,
  rav1d-safe `5b51b144`, magetypes standalone. Fresh target directory per run,
  incremental and compiler wrappers off, no target-cpu override.
- `just consumer-compile OUT prepare --sources ~/tmp/attune-consumer-cold/sources
  --baseline da0ed5de --candidate 0d50cb13`, then `check`, `release`, `report`.
  Logs and Cargo timing HTML: `~/tmp/compile-20261024b/` on i265 (interim run:
  `~/tmp/compile-20261024/`).

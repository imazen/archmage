# Operation visibility: fix and paired retest — 2026-10-09

The draft fix `0176d430` resolves `inline(default)` from the operation's visibility
before generating a private implementation. The body receives that resolved
attribute; proof wrappers retain their separate policy. Direct output visibility
overrides participate, while hidden family bodies use the source visibility.
This fixes the visibility-selection bug without changing legacy omission behavior.

Restoring hints on the 42 affected public operations did **not** recover the
non-LTO encoder loss in this equivalent-emitter experiment. Corrected-policy
encoding took 22.34% longer at 256×256 and 18.90% longer at 512×512 than the
existing-hints baseline. Compared with the buggy rule, changes were +0.01% and
−0.29%, with overlapping process-median ranges. Shipping-profile encoder changes
versus baseline were −0.49% and +0.19%, also with overlapping ranges. Small
changes are not claimed to be statistically significant.

## Paired measurements

Median of five process medians. Positive changes mean more elapsed time.
All three policies were freshly built and measured together, with randomized
policy order in each pass; do not compare absolute times across dated reports.

| Profile | Crop | Workload | Baseline ms | Old rule ms | Fixed rule ms | Fixed vs baseline | Fixed vs old |
|---|---:|---|---:|---:|---:|---:|---:|
| release | 256 | encode | 5.701 | 6.974 | 6.975 | +22.34% | +0.01% |
| release | 256 | decode | 1.585 | 1.598 | 1.594 | +0.60% | -0.23% |
| release | 512 | encode | 125.439 | 149.580 | 149.143 | +18.90% | -0.29% |
| release | 512 | decode | 6.188 | 6.171 | 6.180 | -0.13% | +0.15% |
| ship | 256 | encode | 3.443 | 3.438 | 3.426 | -0.49% | -0.35% |
| ship | 256 | decode | 1.373 | 1.379 | 1.370 | -0.22% | -0.61% |
| ship | 512 | encode | 78.065 | 78.109 | 78.216 | +0.19% | +0.14% |
| ship | 512 | decode | 5.821 | 5.863 | 5.856 | +0.61% | -0.12% |

[Full results and ranges](results.md), [unrounded summary](summary.json), and
[release](release.csv)/[shipping](ship.csv) process statistics are retained.
Every one of the 120 encode/decode groups supplied 20 samples and passed the
harness reliability check. Encoded bytes and decoded pixels matched across all
policies, profiles and repetitions for each crop.

## What changed, and what this cannot establish

The [corrected expansion inventory](../inline_operation_inventory_2026-10-09/README.md)
records 42 restored hints: 36 rav1d-safe and six zenav1-svt-dsp public operations,
all lowered to private sibling kernels. All 1,951 matching proof-wrapper policies
are unchanged. Another 2,360 body events still get no inline attribute under the
visibility ablation, including 1,359 magetypes backend events.

The experiment changes legacy macro emitters in the pinned main 0.9.30 stack;
**it does not measure the complete attune rewrite or a migrated consumer**.
Handwritten consumer attributes and body source remain intact, but macro-generated
magetypes kernel attributes are part of the experiment. Trait placements in attune
reject ambiguous `inline(default)` and require an explicit choice; a faithful
migration should preserve existing body hints with `inline(hint)`. These results
do not attribute cost to a particular function or package. They support keeping
visibility-based policy opt-in, rather than removing hot internal hints by default.

Only two 8-bit still-image crops of one photograph are tested: 256×256 at QP32 /
preset8 and 512×512 at QP20 / preset6. CPU: Intel Core Ultra 7 265K, runtime pinned
to CPU2, decoder threads=1. Rust 1.98.1; no target-cpu=native. Both profiles use
opt-level3: release has LTO off / 16 codegen units; ship has fat LTO / one codegen
unit. Two settings change between profiles, so do not attribute that difference
solely to LTO. No ARM, video, HDR, broad corpus or cross-platform performance
claim is made. Resolve host details with `hostname` and `lscpu -J`; observations
are in [machine.json](machine.json).

## Reproduction and provenance

Tooling: `b3538357`. Stack: main `e2dbab66ef5aa08f8e23ed05248e7d1217f58475`.
Pinned consumers:

- [zenav1-svt](https://github.com/imazen/zenav1-svt): `224c6bbb9a3dc577c64f6e0531bda5c3f60c6b0b`.
- [rav1d-safe](https://github.com/imazen/rav1d-safe): `5b51b144cf73216fffd1100bfc396bc7dc17b5bd`.

All 789 encoder and 256 decoder Rust files are hash-identical across policies;
resolved lockfiles also match. [Plan](plan.json) records full source, compiler,
input, driver and runner hashes. [Artifacts](artifacts.json) records output
hashes and sample counts; [raw index](raw-artifacts.csv) hashes full logs.
[Build observations](builds.json) contain time, binary size and /usr/bin/time -v
maximum RSS for each fresh build. One build per case is not a compile-time
regression estimate or a measurement of the draft macro rewrite's compile cost.

Run serially through run-heavy with a 16G cap, eight jobs and TMPDIR in ~/tmp:

```sh
python3 experiments/inline-real/run.py prepare \
  --out /home/lilith/tmp/attune-inline-operation-2026-10-09 \
  --sources /home/lilith/tmp/attune-consumer-cold/sources \
  --revision e2dbab66ef5aa08f8e23ed05248e7d1217f58475 \
  --image /home/lilith/tmp/png-corpus-sample/2400-unsplash-textures/2404_textures_tree-bark-texture_by-hwang-sanguk-k0tljwrabxg-unsplash_3024x3024.sdr.png \
  --policies baseline body_default body_operation
python3 experiments/inline-real/run.py build --out /home/lilith/tmp/attune-inline-operation-2026-10-09 --policies baseline body_default body_operation
python3 experiments/inline-real/run.py measure --out /home/lilith/tmp/attune-inline-operation-2026-10-09 --policies baseline body_default body_operation --passes 5
python3 experiments/inline-real/run.py export --out /home/lilith/tmp/attune-inline-operation-2026-10-09 --policies baseline body_default body_operation --report /absolute/new/report
```

The `*-operation` recipes in `experiments/inline-real/justfile` encode these
policy selections. The old `body_default` remains unchanged for reproduction;
`body_operation` is the corrected variant. Raw logs and reference binaries are
protected in `/home/lilith/tmp/attune-inline-operation-2026-10-09`, with binaries
outside Cargo target directories. Wrapper logs are
`/home/lilith/tmp/attune-inline-operation-build.log` and
`/home/lilith/tmp/attune-inline-operation-measure.log`.

```text
prepare + six builds: rc=0 224s | peak-RSS 1.18GiB | min-avail 23249MiB | peak-load 6.63
runtime matrix:       rc=0 512s | peak-RSS 0.02GiB | min-avail 26136MiB | peak-load 4.33
```

## Implementation verification

Macro unit tests, all attune integration suites and macro-crate Clippy passed.
The generator, registry/token validators and soundness checks passed. Legacy
expansion snapshots and input/output compile suites passed unchanged. Regression
assertions check public hidden native bodies, proof-only/dispatcher-only families,
scalar fallback, generics, visibility overrides and separate wrapper policy.
The additional generic public-family integration test compiled and ran with
forbid(unsafe_code) and deny(warnings). These are local x86-64 checks; unit tests
also inspect emitted scalar, x86, NEON and WASM attributes without executing all
architectures.

Complete logs:
`/home/lilith/tmp/attune-inline-fix-tests.log`,
`/home/lilith/tmp/attune-inline-fix-compat.log`, and
`/home/lilith/tmp/attune-inline-operation-inventory.log`.

```text
focused tests + Clippy: rc=0 1s  | peak-RSS 0.27GiB | min-avail 25629MiB | peak-load 3.20
health + legacy suites: rc=0 94s | peak-RSS 0.24GiB | min-avail 25312MiB | peak-load 4.31
integration + inventory: rc=0 11s | peak-RSS 0.87GiB | min-avail 25046MiB | peak-load 4.57
```

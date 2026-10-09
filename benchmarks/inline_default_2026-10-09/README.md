# Visibility-based body hints — 2026-10-09

The explicit `inline(default)` policy is implemented on the draft in `b7aceb85`.
It emits hints for unrestricted public bodies and no inline attribute for
restricted/private bodies. It remains opt-in; this experiment does not support
recommending it as a universally fast default. Legacy omission behavior is
unchanged, and `inline(hint)` remains the faithful body-policy migration.

## Measured result

Compared with a freshly rebuilt and remeasured baseline, the visibility policy
increased non-LTO zenav1-svt encoding time by 18.03% at 256×256 and 18.71% at
512×512. With fat LTO / one codegen unit, the measured changes were +0.72% and
+0.43%. Decoder differences ranged from -1.45% to +0.15%, with overlapping
process-median ranges. Small differences are not claimed to be statistically
significant. Visibility alone is insufficient to select body hints for these
non-LTO encoder workloads; this does not attribute the loss to individual
callees.

[Generated tables](results.md) include every baseline/policy comparison and the
range of process medians. [summary.json](summary.json) retains the unrounded
values. The [release](release.csv) and [shipping-profile](ship.csv) CSVs contain
all process-level means, medians and MADs.

## What was measured

This is a controlled legacy-emitter policy experiment, **not a benchmark of the
complete attune rewrite**. Both variants use fixed main
`e2dbab66ef5aa08f8e23ed05248e7d1217f58475` (0.9.30). The native arcane operation
body is private even when its proof wrapper is public, so the visibility policy
omits its body hint. Rite bodies and scalar/wasm arcane bodies use their emitted
visibility. Proof-wrapper attributes, handwritten consumer attributes and
explicit arcane `inline_always` requests remain intact.

Pinned consumers:

- [zenav1-svt](https://github.com/imazen/zenav1-svt): `224c6bbb9a3dc577c64f6e0531bda5c3f60c6b0b`.
- [rav1d-safe](https://github.com/imazen/rav1d-safe): `5b51b144cf73216fffd1100bfc396bc7dc17b5bd`.

All 789 encoder and 256 decoder Rust source files were hash-compared and are
identical between variants. Only macro body-inline emission differs. The decoder
manifest redirects its archmage dependency to the chosen source copy. The
resolved lockfile is identical between policies.

The driver and workload match the [previous experiment](../inline_real_2026-10-08/README.md):
two centered photographic crops from one tree-bark image, 8-bit I420 still
frames; 256×256 QP32/preset8 and 512×512 QP20/preset6. Encoder and single-threaded
decoder API calls operate in memory. Construction, destruction, file I/O and
process startup are outside the timed region. Image and input hashes are in
[plan.json](plan.json).

Machine: Intel Core Ultra 7 265K, x86_64 Linux, rustc 1.98.1 / LLVM 22.1.8,
runtime pinned to CPU 2. No native CPU targeting. `release` uses opt3, LTO off,
16 codegen units; `ship` uses opt3, fat LTO, one codegen unit. Those profiles
change two settings, so cross-profile effects are not attributed solely to LTO.
[Machine observation and resolving commands](machine.json).

Five randomized process-level passes per profile/policy/size produced 40
processes, 80 groups and 20 timed samples per group. Tables report the median
of five process medians; ranges are their minimum/maximum, not confidence
intervals. Zenbench samples are not paired across binaries. Every group passed
the reliability flag, and encoded bytes and decoded pixels matched across all
policies and profiles. The exporter enforces the complete selected matrix and
parity. Correctness hashes are in [artifacts.json](artifacts.json).

No video sequences, broad corpus, HDR, film grain, ARM or AVX-512 runtime paths
were measured. There is no per-callee assembly attribution or cold-compile
regression estimate here. Each build variant was compiled once; the exact
command, wall time, GNU `size` counts, binary hash and `/usr/bin/time -v` maximum
RSS observation are retained in [builds.json](builds.json).

## Reproduction and resources

The [runner](../../experiments/inline-real/run.py) implements the exact edits with
occurrence assertions. Measurement used the runner at `4d7422a2`; subsequent
export improvements only generate tables and record metadata. The exported
runner hash identifies the exporter, not a claim that metadata-only additions
were present during measurement.

From the repository root, use the same prepare/build/measure commands described
in the [harness documentation](../../experiments/inline-real/README.md), passing
`--policies baseline body_default` to **every** stage. This run used:

```sh
python3 experiments/inline-real/run.py prepare \
  --out /home/lilith/tmp/attune-inline-default-2026-10-09 \
  --sources /home/lilith/tmp/attune-consumer-cold/sources \
  --revision e2dbab66ef5aa08f8e23ed05248e7d1217f58475 \
  --image /home/lilith/tmp/png-corpus-sample/2400-unsplash-textures/2404_textures_tree-bark-texture_by-hwang-sanguk-k0tljwrabxg-unsplash_3024x3024.sdr.png \
  --policies baseline body_default
python3 experiments/inline-real/run.py build --out /home/lilith/tmp/attune-inline-default-2026-10-09 --policies baseline body_default
python3 experiments/inline-real/run.py measure --out /home/lilith/tmp/attune-inline-default-2026-10-09 --policies baseline body_default --passes 5
python3 experiments/inline-real/run.py export --out /home/lilith/tmp/attune-inline-default-2026-10-09 --policies baseline body_default --report /absolute/new/report
```

Heavy commands ran serially with a 16G cap, eight build jobs and
`TMPDIR=/home/lilith/tmp`. Resource-wrapper observations:

```text
prepare + four builds: rc=0 145s | peak-RSS 1.18GiB | min-avail 26690MiB | peak-load 3.28
runtime matrix:        rc=0 346s | peak-RSS 0.02GiB | min-avail 29027MiB | peak-load 0.89
```

Full wrapper logs are `/home/lilith/tmp/attune-inline-default-build.log` and
`/home/lilith/tmp/attune-inline-default-measure.log`. The raw artifact root is
`/home/lilith/tmp/attune-inline-default-2026-10-09`; [raw-artifacts.csv](raw-artifacts.csv)
indexes full sample/build logs by hash. Reference binaries are preserved outside
Cargo target directories. The provenance capture also lists older crates.io
archives that were **not** the selected macro stack in this experiment.

## Implementation checks

Macro-crate unit tests, all attune integration suites (including the new inline
suite), and macro-crate Clippy passed. The generator, registry validation, token
validation and soundness checks passed. Legacy macro expansion snapshots and
both the original-input and expanded-output compile suites passed unchanged;
no expected output was updated. Their complete logs are
`/home/lilith/tmp/attune-inline-default-tests-recheck.log` and
`/home/lilith/tmp/attune-inline-default-compat.log`.

```text
focused tests + Clippy: rc=0 1s  | peak-RSS 0.22GiB | min-avail 29021MiB | peak-load 1.20
health + legacy suites: rc=0 94s | peak-RSS 0.24GiB | min-avail 28559MiB | peak-load 0.99
```

These are local x86-64 checks. No new cross-platform CI result or closed
end-user compile-time gate is claimed.

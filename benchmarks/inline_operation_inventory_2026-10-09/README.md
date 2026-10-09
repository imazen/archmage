# Corrected operation-visibility inventory — 2026-10-09

The fix in `0176d430` resolves `inline(default)` from the operation's visibility
before emitting a private implementation. A fresh equivalent legacy-emitter
inventory restores **42 body hints**: 36 in rav1d-safe and six in zenav1-svt-dsp.
All are public source functions lowered to private sibling kernels. Their proof
wrappers retain `inline(always)`.

| Package | Body hint | No body attribute | Unchanged always-inline proof wrappers |
|---|---:|---:|---:|
| magetypes | 0 | 1359 | 1359 |
| rav1d-safe | 36 | 702 | 489 |
| zenav1-svt-dsp | 6 | 278 | 86 |
| zenav1-svt-encoder | 0 | 21 | 17 |
| **Total** | **42** | **2360** | **1951** |

Counts are emission events with matching generated architecture and Cargo-feature
guards, excluding calibration. They are not call counts, monomorphization counts,
linked functions, or measurements of LLVM's actual inlining.

[Changed bodies](changed-bodies.csv) lists every affected function and its source
location. [Summary](summary.csv) separates source, physical body and policy
visibility. [Per-file counts](by-file.csv) include the calibration package as
explicitly labeled rows. The raw inventory retains seven calibration events and
four events excluded by generated architecture guards.

All 4,364 event identities and other old metadata fields match the
[earlier inventory](../inline_default_inventory_2026-10-09/README.md). Only the
42 consumer decisions and the public-wrapper calibration's body decision change
from no attribute to inline. The new `policy_visibility` column explicitly records
which visibility was used. [Comparison provenance](comparison.json) identifies
both complete CSVs by hash; neither large raw file is committed.

## Verification and scope

Tooling is committed in `b3538357`. Calibration requires public native bodies to
receive hints under `body_operation`, and private/restricted bodies to omit them.
The original `body_default` policy remains reproducible and expects the previous
public-native outcome. Positive direct/scalar and explicit-always controls remain.
Resolved package versions and features must match the measured driver.

The inventory uses the pinned legacy 0.9.30 stack from main `e2dbab66`, with the
corrected emitter policy. It does not migrate the consumers to attune. In
particular, the ablation still removes hints from 1,359 magetypes backend kernels;
trait placement in attune rejects ambiguous `inline(default)` and requires an
explicit choice. This inventory does not show that an actual migration should
remove those hints or attribute runtime cost to them.

Run serially through run-heavy with a 16G cap, eight jobs, and TMPDIR in ~/tmp:

```sh
python3 experiments/inline-real/inventory.py \
  --measurement /home/lilith/tmp/attune-inline-operation-2026-10-09 \
  --out /absolute/new/inventory \
  --policy body_operation
```

[Provenance](provenance.json) includes full source pins, compiler version, script
hashes, target configuration and artifact hashes. The raw artifact directory is
`/home/lilith/tmp/attune-inline-operation-inventory-2026-10-09`; the wrapper log is
`/home/lilith/tmp/attune-inline-operation-inventory.log`. This run also compiled
and executed the added generic public-hidden-family regression test.

```text
rc=0 11s | peak-RSS 0.87GiB | min-avail 25046MiB | peak-load 4.57
```

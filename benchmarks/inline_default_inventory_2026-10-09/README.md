# Where inline(default) resolves — 2026-10-09

In the tested consumer build, **all 2,402 matching operation-body emission events
resolve to no inline attribute**. None resolve to an inline hint. This means the
visibility rule made the same body decisions as the remove-all-implicit-body-hints
policy for these recorded expansions. This is an attribute-decision statement,
not a claim of identical linked binaries or attribution of the measured slowdown.

## Counts

These are emission events whose generated architecture and Cargo-feature guards
match the x86-64 build. Calibration functions are excluded. Proof wrappers are
reported separately and retain their existing inline(always).

| Package | Body hint | No body attribute | Unchanged always-inline proof wrappers |
|---|---:|---:|---:|
| magetypes | 0 | 1359 | 1359 |
| rav1d-safe | 0 | 738 | 489 |
| zenav1-svt-dsp | 0 | 284 | 86 |
| zenav1-svt-encoder | 0 | 21 | 17 |
| **Total** | **0** | **2,402** | **1,951** |

The 2,402 bodies divide into **451 direct rite bodies**, **676 private sibling
kernels**, and **1,275 private nested kernels**. All nested kernels are in
magetypes backend implementations. There are also two body events and two wrapper
events filtered out by generated aarch64 guards; they are retained in the raw
inventory rather than counted above. AVX-512 is enabled in this build, so matching
counts include AVX-512 kernels regardless of whether this CPU can execute them.

**42 source functions are written pub but have private generated kernels:**
36 in rav1d-safe and six in zenav1-svt-dsp. Their public proof wrappers keep
inline(always), while the kernels receive no inline attribute.

## Where the decisions occur

Magetypes is part of the macro-policy experiment: **1,359 backend kernel events**
receive no inline attribute. The largest groups are 680 in `x86_v3.rs` and 658 in
`x86_v4.rs`. Handwritten always-inline helpers such as load/store are outside the
modified emitters and are not counted. Source bodies remain unchanged; generated
inline attributes on arcane backend kernels do change.

| File | Matching no-attribute body events |
|---|---:|
| `archmage/magetypes/src/simd/impls/x86_v3.rs` | 680 |
| `archmage/magetypes/src/simd/impls/x86_v4.rs` | 658 |
| `rav1d-safe/src/safe_simd/itx/part09_identity_hybrid_16bpc.rs` | 116 |
| `rav1d-safe/src/safe_simd/mc.rs` | 110 |
| `rav1d-safe/src/safe_simd/itx/part04_rect_idtx_aspect.rs` | 95 |
| `rav1d-safe/src/safe_simd/itx/part03_adst16_and_small_rect.rs` | 80 |
| `rav1d-safe/src/safe_simd/ipred.rs` | 65 |
| `rav1d-safe/src/safe_simd/itx/part02_col_1d_avx512_pmaddwd.rs` | 51 |
| `rav1d-safe/src/safe_simd/loopfilter.rs` | 44 |
| `rav1d-safe/src/safe_simd/looprestoration.rs` | 42 |
| `rav1d-safe/src/safe_simd/itx/part08_rect_dct_adst_16bpc.rs` | 38 |
| `zenav1-svt/rust/crates/svtav1-dsp/src/txfm_simd_ext.rs` | 35 |

The full [per-file table](by-file.csv) and [visibility/body-kind summary](summary.csv)
are committed. [Magetypes method-name counts](magetypes-methods.csv) distinguish
repeated backend methods: `splat`, `zero`, `min`, `max`, each comparison method,
`blend` and `reduce_add` each contribute 40 matching events. These are counts
across backend/type implementations, not 40 calls or monomorphizations.

The name `inner` appears 69 times in magetypes' `sse2_baseline!` expansion. Its
identifier always points to the macro template; its body-source location
separates the instances. The final inventory has no duplicate complete records.

Concrete examples from the pinned sources:

- [`block_sad_v3`](https://github.com/imazen/zenav1-svt/blob/224c6bbb9a3dc577c64f6e0531bda5c3f60c6b0b/rust/crates/svtav1-dsp/src/me_sad.rs#L188) is public; arcane emits a private kernel without a body hint and a public always-inline wrapper.
- [`F32x4Backend::splat`](https://github.com/imazen/archmage/blob/e2dbab66ef5aa08f8e23ed05248e7d1217f58475/magetypes/src/simd/impls/x86_v3.rs#L40) uses arcane in a trait implementation; its generated nested kernel is private.
- [`sse2_baseline!`](https://github.com/imazen/archmage/blob/e2dbab66ef5aa08f8e23ed05248e7d1217f58475/magetypes/src/simd/impls/x86_v3.rs#L22) produces the repeated `inner` functions. Keep their distinct body locations when grouping them.

## Collection and limits

This inventories the exact legacy-emitter visibility-policy variant used by the
[runtime comparison](../inline_default_2026-10-09/README.md). It does not migrate
consumers to attune, count all source declarations in their repositories, or
measure LLVM's actual inlining decisions. A recorded function can be dead,
uninstantiated, folded away, or never reached by the benchmark. No performance
attribution by function or package is established by these counts.

`inherited` in the CSV means no explicit visibility syntax. In particular, it
must not be interpreted as saying a trait method is inaccessible publicly.
`private` for a hidden generated kernel describes the emitted function itself.
Generated guard matching covers the emitter's architecture and Cargo-feature
gates, not a general-purpose evaluator for arbitrary user cfg expressions.

The collector instruments the emitter's selected attribute and records package,
macro, function name, identifier/body source locations, source/body visibility,
body kind, tier/token, guards and selected attribute. It runs a fresh Cargo check
in an isolated copy, leaving timed source trees and binaries untouched. There is
no instrumentation, environment lookup or reporting overhead in normal macro
builds. Source and dependency pins are those of the linked runtime report;
[provenance.json](provenance.json) includes the full measurement plan and hashes.

Calibration adds six driver functions producing seven events: public direct and
scalar functions produce hints; private/restricted direct bodies and the native
hidden kernel produce no attribute; its wrapper and an explicit always-inline
request produce always-inline attributes. Those exact outcomes are asserted.
The collector also asserts that calibration changes no resolved package version
or feature set. The consumer event multiset matches the independent run without
calibration. No complete record is duplicated.

Reproduce using collector commit `2dd66503`, from the repo root, through the
serialized resource-limiting wrapper:

```sh
python3 experiments/inline-real/inventory.py \
  --measurement /home/lilith/tmp/attune-inline-default-2026-10-09 \
  --out /absolute/new/inventory
```

[Collector documentation](../../experiments/inline-real/README.md) describes the
outputs. The complete 4,364-event CSV (including seven calibration events and
four events with nonmatching guards) is retained at:

`/home/lilith/tmp/attune-inline-inventory-calibrated-2026-10-09/decisions.csv`

It contains a row for every observed decision, with source and body locations.
The full compiler log and Cargo metadata are beside it. Their recorded hashes
and sizes are in provenance; the compact summaries above are committed instead
of the large raw files. The full CSV can be regenerated by the collector.

The check and all calibration assertions passed. Python lint/format and Rust
format checks passed for the collector and fixtures. Wrapper resource observation
(not a runtime benchmark):

```text
rc=0 10s | peak-RSS 0.88GiB | min-avail 27358MiB | peak-load 0.90
```

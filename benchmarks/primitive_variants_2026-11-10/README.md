# Configurable primitive study — 2026-11-10

**The overall Zen 5 timing run failed admission. No blanket legacy replacement
is performance-qualified.** Six comparison groups retain 50/50 clean rounds;
reciprocal throughput has only 18/50 clean rounds. All rounds and failed
admission are retained, with no retry or round extension.

The kernels compare existing Magetypes `f32x8::min`, `recip` and `rsqrt` with
explicitly named Magemath configurations. “Legacy working recip/rsqrt” in the
frozen report means those exact methods, with their existing repair/refinement
arithmetic. “Native” means BackendDependent, not a portable graph. Every
approximate profile uses Normal range; unchecked means AssumeValid. HIGHP
is Full-domain division or sqrt-then-division. See the
[contract reference](../../docs/PRIMITIVE-CONTRACTS.md).

## Protocol and identity

- Ryzen 9 9950X3D, V3 kernels, logical x8. No target-cpu=native.
- 64 input values as eight x8 chunks for streams; chains contain 64 dependent x8
  operations, each followed by the same addition. These are hot small kernels,
  not long-buffer application throughput or isolated instruction latency.
- Zenbench 0.1.11, strict admission, exactly 50 interleaved rounds. Enforced
  uniform CPU class via a cpuset on this task's own transient system scope.
- Frozen source `d3c9290ed57d60fc675cb36a8fd3db68fcba1898`, rustc 1.99.1
  (58cf8f9c3 2026-10-13), executable SHA256
  `4346f693a27bee12878ff284ebd3d9696a18e4383eec4ae4d72e0be391794ebe`.
  Zenbench's git field reports colocated Git HEAD `f47f05de`; the source above
  identifies the actual jj snapshot compiled. Source archive and executable
  are preserved with the raw results.
- Later scalar-entry simplifications, NEON raw-seed ablation and strict WASM
  propagating-min ablation are not part of this timing binary. Native V3 typed
  numerical graphs are unchanged; no final-head retiming is claimed.
- The perf pass bound is upper 95% paired time ratio ≤1.02 in every required
  clean cell. Missing ISA/width/workload cells and failed admission prevent
  promotion. A portable result's non-NaN bits must agree, with NaN class only.
- `run-heavy`: rc=1, 13 s, peak-RSS 0.25 GiB, min available 40796 MiB,
  peak load 0.59. Failure is the harness's retained-admission assertion.

[All-round results](zen5-all-rounds.json) use every retained pair without IQR
removal: 10,000 paired bootstrap resamples of the ratio of means, fixed seed
0x531324, percentile 95% interval. Zenbench's printed intervals may use its
outlier analysis; they are not substituted for these all-round gate intervals.

## Results in ns

Each row is **ns per 64 inputs or 64 eight-lane chain steps**, not ns per lane.
The corresponding relative error bound is a separate numerical contract.

| Implementation | recip stream (unqualified) | recip chain | rsqrt stream | rsqrt chain |
|---|---:|---:|---:|---:|
| Existing Magetypes method | 2.515 | 158.832 | 3.581 | 217.749 |
| HIGHP division / sqrt then divide | 3.502 | 86.483 | 10.441 | 237.649 |
| Portable MIDP unchecked | 3.100 | 225.792 | 5.349 | 320.346 |
| Portable LOWP unchecked | 2.033 | 124.974 | 2.985 | 187.526 |
| Backend MIDP unchecked | 2.357 | 125.766 | 2.813 | 187.771 |
| Backend LOWP unchecked | 1.578 | 40.514 | 1.477 | 40.843 |
| Backend MIDP checked | 3.382 | 155.684 | 3.797 | 210.137 |

Selected bars copied from Zenbench's report:

```text
recip/f32x8/chain64 — 50/50 clean
native lowp unchecked    ████████ 40.5ns
highp division           █████████████████ 86.5ns
portable lowp unchecked  █████████████████████████ 125.0ns
native midp unchecked    █████████████████████████ 125.8ns
native midp checked      ███████████████████████████████ 155.7ns
legacy working recip     ████████████████████████████████ 158.8ns
portable midp unchecked  █████████████████████████████████████████████ 225.8ns
```

Portable MIDP chain cost versus the existing method is real: reciprocal ratio
1.421579 [1.420554, 1.422608], rsqrt 1.471171 [1.470033, 1.472604]. Native
unchecked MIDP instead gives 0.791817 [0.791342, 0.792319] and 0.862328
[0.861129, 0.863449]. Exact division winning the reciprocal chain is also real;
accuracy grades are not a monotonic speed ordering. Native checked MIDP rsqrt
throughput is 1.0602 [1.0146, 1.1134], exceeding the 2% upper-bound criterion.
These are individual clean cells inside an overall unqualified run.

| min configuration | stream ns | chain ns |
|---|---:|---:|
| Existing native min | 1.536 | 28.607 |
| Number / ordered zero | 2.488 | 104.923 |
| Propagate / ordered zero | 2.770 | 111.580 |
| Backend NaN / zero | 1.599 | 28.598 |
| Normal checked | 7.586 | 79.316 |
| Normal AssumeValid | 1.289 | 28.650 |

Normal unchecked and legacy native min alias in the compiled V3 probes.
Their small stream timing difference is not evidence of an algorithmic win.
Portable Full policies pay for specified NaN/zero semantics on x86.

The [linear-sRGB inventory](../../docs/LINEAR-SRGB-INVENTORY.md) compares its log2
graph mirror with configured LOWP, BackendDependent, AssumeValid, Caller:
5.765597 ns versus 5.712658 ns, ratio 0.990818 [0.978026, 1.000227], 50/50 clean.
One coefficient differs; this is neither a bit-identical replacement nor a
measurement of the linear-srgb crate binary.

## Reproduction and validation

Use `run-heavy` with resource caps for builds and scans. Commands are in the
root justfile:

```text
just primitive-verify
just primitive-cross
just primitive-wasm-token-testing
just primitive-study-check
just primitive-accuracy
just primitive-codegen
just primitive-codegen aarch64-unknown-linux-gnu
just primitive-codegen wasm32-wasip1 '-Ctarget-feature=+simd128,+relaxed-simd'
```

The full mantissa scan is opt-in; ordinary cargo test keeps directed corners,
independent exact-rational fixtures, validation masks, error bounds and
vector/width-specific tail checks. The scan streams fixed x4 vectors, not an
array containing the full input space. It enumerates every f32 in [1,4) and
uses f64 references; it does not prove every normal input or every CPU seed.

Native x86 and M6 std/no_std directed tests pass. The [1,4) full x4 scans pass
on Scalar, V3 and M6 NEON. Portable maxima match across those backends. M6 raw
NEON LOWP max relative errors are 2.853393554688e-3 (recip) and
3.276822880217e-3 (rsqrt); one-refinement MIDP maxima are 8.228816000135e-6
and 1.618327600566e-5. All are within their advertised bounds. WASM strict
and relaxed directed tests and relaxed-token suppression pass under Wasmtime.
MSRV 1.89 type checking passes. ARM/WASM codegen probes compile.

The timing binary requires `--measure` and `ARCHMAGE_PRIMITIVE_RESULTS` with a
new durable output filename. On Linux, `run_local.py` verifies and constrains
only its own run-heavy system scope before invoking the strict harness. A
failed admission exits nonzero after saving the results; do not edit thresholds
or discard flagged rounds. There is no final-graph Mac/Neoverse/Intel timing,
AVX-512 timing, f64 reciprocal family, or long-buffer timing in this study.

Raw output, binaries, source archives and logs live under
`/mnt/v/output/archmage/primitive-variants-2026-11-10/`, mirrored under
`/mnt/tower/output/archmage/primitive-variants-2026-11-10/`.

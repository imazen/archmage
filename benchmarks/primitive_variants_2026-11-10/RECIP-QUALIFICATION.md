# Reciprocal qualification — 2026-11-11

**Neither research graph is promoted.** Both are bit-portable on the tested
normal domain. Parallel fused correction wins some clean Intel/Zen 5 cells,
but loses other workloads. Canonical RCP14 gives the requested asymmetric
implementation: one native instruction on AVX-512, identical integer emulation
elsewhere. That emulation is expensive. Public Magemath graphs and bounds are
unchanged.

Missing coverage first: every strict suite has at least one dirty group; 6/36
strict groups have all 250 rounds clean. Intel measurements are **Skymont E
cores**, not P cores. M6 timings are unconfined, exploratory and 0/50 clean in
every group. Neoverse storage/compute admission was unavailable. No WASM timing,
first-use dispatch timing, bit-distributed performance corpus or cold-build
comparison is claimed. These gaps prevent a general replacement claim.

## Algorithms and precision

All entries use f32. Approximate inputs satisfy **2^-125 ≤ abs(x) ≤ 2^125**.
The study is unchecked, normal-domain arithmetic; exceptional inputs are outside
this contract. NaN payloads are not a portability promise. HIGHP division has
its existing wider domain, but is measured on the same ordinary inputs.

| Exact label in the report | Implementation | Accuracy / portability |
|---|---|---|
| magetypes recip | Existing `f32xN::recip()` including its refinement and rail handling | Existing working-precision contract; results need not be portable |
| portable parallel MIDP | Integer seed `0x7ef127ea - bits`; `e=fma(-x,y,1)`, `a=fma(y,e,y)`, `b=fma(e,e,1)`, `a*b` | Fixed fused graph; max relative error 1.145700113618e-5 in the full normalized scan; ≤1e-4 required |
| portable serial MIDP | Current public `PORTABLE_MIDP`, Normal, AssumeValid, Caller | Integer seed followed by two `y*fma(-x,y,2)` steps; published relative bound ≤1e-4 |
| portable HIGHP division | Current public HIGHP, Caller: `1/x` | Rounded division; consistent non-NaN bits under RNE and gradual underflow |
| canonical RCP14 | Native VRCP14PS for V4/V4x at **every width**, integer emulation otherwise | Max relative error 5.438657899502e-5; <2^-14, already inside MIDP's 1e-4 bound |

The approximate graphs sacrifice accuracy relative to the existing working
reciprocal. In particular, raw RCP14 is not a correctly rounded reciprocal.
Comparisons show speed at explicitly different precision contracts, not an
accuracy-preserving replacement of `recip()`.

Parallel correction has three FMA and one multiplication, versus the current
graph's two FMA and two multiplications. Its middle nodes can run independently;
the shorter dependency path helps chains, while extra FMA pressure can hurt
streams. Caller mode deliberately retains Scalar software fusion; it does not
measure Automatic's upgrade to a packed/native provider. On WASM the prototype
uses the existing fused primitive. Integrating it into Magemath would reuse the
whole-operation provider selector; no WASM speed claim is made from this prototype.

## A native algorithm that can be emulated exactly

Intel publishes the [underlying RCP14 algorithms and C reference](https://www.intel.com/content/www/us/en/developer/articles/code-sample/reference-implementations-for-ia-approximation-instructions-vrcp14-vrsqrt14-vrcp28-vrsqrt28-vexp2.html).
This differs from trying to infer RCPPS output bits from its error bound. The
unmodified [RECIP14.c](reference/RECIP14.c) is retained under Intel's BSD-3-Clause
notice; [provenance and compilation flags](reference/ORIGIN.md) accompany it.
No C or FFI enters the production libraries.

For our restricted normal domain, its interpolation reduces to 64 packed
coefficients (**256 bytes**) and integer operations:

```text
E = (bits >> 23) & 255; m = bits & 0x7fffff
c = coefficients[m >> 17]; a = c >> 10; b = c & 1023
d = ((m >> 7) & 1023) - 512
s = (a << 8) - b*d
normal_result = sign | ((253-E) << 23) | ((s >> 2) & 0x7fff80)
if m == 0: result = sign | ((254-E) << 23)
```

The safe Rust port is [rcp14.rs](src/rcp14.rs). No floating-point approximation,
FMA availability, or native-estimate common-prefix assumption enters emulation.
AVX-512 x4/x8 explicitly use VRCP14VL; the ordinary Magetypes V4 approximation
at those widths can delegate to V3 RCPPS and is not interchangeable.

The hardware/reference identity is measured on Zen 5, including all signed
normalized mantissas and directed edge exponents. It has **not** been measured
on physical Intel AVX-512 hardware. No claim is made that finite startup probes
certify arbitrary future native hardware. This remains a named research backend,
not an unrestricted production hardware-selection guarantee.

Codegen review: the x8 AVX-512 kernel emits one `vrcp14ps` plus ABI load/store
traffic. AVX2 and NEON emulation need lane-indexed coefficient loads and vector
integer interpolation. The table/repair work explains their observed cost;
this is not merely a dispatch overhead.

## Numerical qualification

The C driver compares native RCP14, the official C implementation and an
independent integer reduction: **16,817,232 cases, zero bit mismatches**.
It also writes a streamed 64 MiB oracle containing every positive f32 in [1,2):
one parallel-fmaf result and one official RCP14 result per input.
SHA256: `5e8e185248b6654fe0e045ef93e4b74039f27c494a9b3d003dcdccf741b038fd`.
Generation uses a fixed 8192-byte buffer; Rust consumes it with a 64 KiB reader.
There is no input-space-sized in-memory array.

Pointwise Rust/C comparisons pass in Scalar, V3, V4, V4x, native M6 NEON with
std and no_std dependencies, and Wasmtime strict/relaxed SIMD. WASM includes
forced non-FMA execution and no_std dependencies in both strict and relaxed
modules. Each dense x4 run checks **16,787,220 signed lanes**, including every
normalized mantissa and directed mantissas at every accepted exponent. x8/x16
check **36,392 / 72,784** directed/sampled lanes, plus same-width tail probes.
Both signs of tails are checked in the final targeted replays. These are full
normalized x4 scans, not a literal exhaustive enumeration of the whole domain
at every width.

Power-of-two scaling explains the exponent reduction. The integer seed scales
exactly inversely with the input exponent. `x*y` and its fused residual are
invariant; the first correction scales with y, while `1+e*e` is invariant.
RNE is invariant under exact power-of-two scaling while rounded outputs stay
normal. The chosen domain keeps the seed, corrected reciprocal and final output
normal. An exact internal FMA product need not itself be normal because it is
not rounded before addition. Reciprocal's odd symmetry covers signs. Directed
exponent checks test this reasoning; the integer RCP14 formula exposes exponent
scaling directly. This assumes RNE and gradual underflow, as do the public graphs.

## Frozen timing protocol

Source `8530969dd44008425a60bc4d84d7bc7f34726455`; rustc 1.99.1.
Zen 5 binary SHA256 `e1a76a96ddec0a0b48d264ffca58dc6856870a9d936713aad209ec93dad93270`.
Intel uses the same source with its own retained compiler/binary identity.
M6 adds the retained exploratory-mode overlay; its arithmetic is unchanged.
No target-cpu=native. No retry or extension of any timing run.

Zenbench 0.1.11 confirmation: five independent processes, each with exactly
50 interleaved paired rounds, 2% margin, family confidence 95%. Every width
x4/x8/x16 measures 64- and 4096-element streams plus 64 dependent vector steps.
Streams repeat 64 ordinary mixed-sign values; chains start at 0.75 and add
0.5 after every reciprocal. Buffers are fixed and bounded, at most 16 KiB.
Linux uses an enforced uniform cpuset. The launcher chose the Intel efficiency
class; this is explicitly an E-core measurement. M6 uses the owner-authorized
`--cpu-policy=unrestricted` exploratory mode. The initial strict placement
refusal is retained too.

**Promotion requires every mandatory cell clean and paired upper95 ratio ≤1.02;
a speed claim additionally needs upper95 <1 in an intended cell.** Every strict
suite fails aggregate confirmation, with all dirty rounds retained. Resource
traces identify foreign work; no process was killed or excluded to get a pass.
M6's resource observations exceed the strict 5% CPU-load threshold, so its
exploratory result is also resource-flagged despite `unreliable=false` under
non-strict reporting. Read its per-round flags, not that Boolean alone.

The all-round analysis keeps every round and bootstraps paired ratios of
arithmetic means within each process, 10,000 draws, fixed seed 0x531324.
The additional process analysis uses five paired log process means with
Student-t df4, Bonferroni family adjustment and timer floor. It retains admission
failure; these are diagnostic intervals, not manufactured zenbench confirmation
verdicts. Clean-cell intervals do not rescue dirty mandatory cells elsewhere.

## Full measured matrix

All times are **ns per stream or per 64 vector chain steps**, not ns/lane.
The columns use the exact implementations above. “Clean” is retained clean
rounds across five processes (250), or the one M6 exploratory process (50).

| CPU / tier | Width / work | Clean | Existing recip | Parallel MIDP | Current serial MIDP | HIGHP division | Canonical RCP14 |
|---|---|---:|---:|---:|---:|---:|---:|
| Zen 5 V3 | f32x4/stream64 | 125/250 | 5.288 | 5.650 | 5.648 | 7.108 | 24.447 |
| Zen 5 V3 | f32x4/stream4096 | 150/250 | 378.738 | 372.931 | 373.279 | 466.455 | 1565.619 |
| Zen 5 V3 | f32x4/chain64 | 161/250 | 159.761 | 163.319 | 188.999 | 78.923 | 453.448 |
| Zen 5 V3 | f32x8/stream64 | 195/250 | 2.457 | 2.829 | 2.855 | 3.667 | 14.321 |
| Zen 5 V3 | f32x8/stream4096 | 122/250 | 155.161 | 189.515 | 189.167 | 236.297 | 907.571 |
| Zen 5 V3 | f32x8/chain64 | 235/250 | 159.067 | 154.949 | 177.218 | 79.431 | 537.263 |
| Zen 5 V3 | f32x16/stream64 | 200/250 | 2.455 | 2.841 | 2.863 | 3.540 | 14.276 |
| Zen 5 V3 | f32x16/stream4096 | 179/250 | 194.715 | 188.536 | 188.955 | 235.912 | 911.661 |
| Zen 5 V3 | f32x16/chain64 | 112/250 | 182.642 | 175.005 | 199.961 | 114.697 | 560.060 |
| Zen 5 V4 | f32x4/stream64 | 154/250 | 5.224 | 5.634 | 5.635 | 7.116 | 1.913 |
| Zen 5 V4 | f32x4/stream4096 | 101/250 | 313.679 | 378.521 | 373.701 | 468.037 | 136.155 |
| Zen 5 V4 | f32x4/chain64 | 200/250 | 159.804 | 163.422 | 189.222 | 79.335 | 48.757 |
| Zen 5 V4 | f32x8/stream64 | 155/250 | 2.489 | 2.874 | 2.804 | 3.718 | 1.502 |
| Zen 5 V4 | f32x8/stream4096 | 197/250 | 156.228 | 189.124 | 188.852 | 236.088 | 81.075 |
| Zen 5 V4 | f32x8/chain64 | 104/250 | 159.494 | 155.326 | 177.675 | 79.314 | 46.991 |
| Zen 5 V4 | f32x16/stream64 | 172/250 | 1.503 | 1.641 | 1.568 | 1.884 | 1.386 |
| Zen 5 V4 | f32x16/stream4096 | 200/250 | 95.050 | 116.115 | 98.058 | 120.096 | 72.157 |
| Zen 5 V4 | f32x16/chain64 | 250/250 | 163.585 | 155.219 | 177.753 | 79.176 | 46.491 |
| Zen 5 Scalar Caller | f32x4/stream64 | 177/250 | 7.068 | 82.038 | 72.455 | 7.078 | 20.354 |
| Zen 5 Scalar Caller | f32x4/stream4096 | 151/250 | 464.266 | 5250.494 | 4620.641 | 464.019 | 1239.864 |
| Zen 5 Scalar Caller | f32x4/chain64 | 102/250 | 93.932 | 654.586 | 516.539 | 93.951 | 437.640 |
| Zen 5 Scalar Caller | f32x8/stream64 | 151/250 | 7.124 | 97.077 | 109.446 | 7.121 | 19.520 |
| Zen 5 Scalar Caller | f32x8/stream4096 | 127/250 | 466.069 | 6212.485 | 7004.614 | 466.222 | 1245.743 |
| Zen 5 Scalar Caller | f32x8/chain64 | 200/250 | 152.575 | 758.530 | 963.054 | 152.517 | 469.895 |
| Zen 5 Scalar Caller | f32x16/stream64 | 201/250 | 7.151 | 81.363 | 86.457 | 7.131 | 19.460 |
| Zen 5 Scalar Caller | f32x16/stream4096 | 194/250 | 626.549 | 5224.761 | 5445.370 | 624.212 | 1251.471 |
| Zen 5 Scalar Caller | f32x16/chain64 | 220/250 | 147.104 | 1154.394 | 1154.414 | 147.032 | 509.181 |
| Skymont E V3 | f32x4/stream64 | 250/250 | 10.042 | 7.349 | 7.168 | 17.246 | 43.196 |
| Skymont E V3 | f32x4/stream4096 | 250/250 | 635.278 | 476.549 | 486.956 | 1115.097 | 2783.887 |
| Skymont E V3 | f32x4/chain64 | 226/250 | 254.557 | 221.585 | 256.782 | 89.340 | 480.876 |
| Skymont E V3 | f32x8/stream64 | 250/250 | 10.018 | 5.956 | 7.777 | 17.258 | 36.191 |
| Skymont E V3 | f32x8/stream4096 | 220/250 | 669.609 | 392.797 | 497.607 | 1114.405 | 2290.900 |
| Skymont E V3 | f32x8/chain64 | 200/250 | 270.795 | 237.081 | 290.800 | 90.555 | 584.735 |
| Skymont E V3 | f32x16/stream64 | 200/250 | 10.062 | 5.907 | 7.360 | 17.239 | 33.968 |
| Skymont E V3 | f32x16/stream4096 | 250/250 | 667.900 | 382.967 | 480.595 | 1114.590 | 2152.320 |
| Skymont E V3 | f32x16/chain64 | 250/250 | 310.317 | 258.147 | 308.802 | 278.406 | 721.398 |
| M6 NEON exploratory | f32x4/stream64 | 0/50 | 4.438 | 5.895 | 5.866 | 3.296 | 21.080 |
| M6 NEON exploratory | f32x4/stream4096 | 0/50 | 303.819 | 390.481 | 388.979 | 244.273 | 1360.902 |
| M6 NEON exploratory | f32x4/chain64 | 0/50 | 117.889 | 106.070 | 123.158 | 43.368 | 489.174 |
| M6 NEON exploratory | f32x8/stream64 | 0/50 | 4.408 | 5.885 | 5.870 | 3.285 | 23.150 |
| M6 NEON exploratory | f32x8/stream4096 | 0/50 | 298.569 | 389.362 | 389.019 | 221.645 | 1496.785 |
| M6 NEON exploratory | f32x8/chain64 | 0/50 | 177.496 | 154.605 | 172.590 | 65.988 | 563.870 |
| M6 NEON exploratory | f32x16/stream64 | 0/50 | 4.450 | 5.859 | 5.832 | 3.299 | 23.196 |
| M6 NEON exploratory | f32x16/stream4096 | 0/50 | 297.879 | 393.295 | 387.003 | 220.318 | 1477.463 |
| M6 NEON exploratory | f32x16/chain64 | 0/50 | 220.979 | 197.910 | 220.281 | 97.158 | 629.904 |

Selected clean-cell diagnostic family-adjusted intervals versus the existing
Magetypes method:

- Skymont E V3 x8 stream64: parallel 5.956 vs 10.018 ns; ratio
  **0.584611–0.604684**, 250/250 clean. Current serial is 7.777 ns.
- Zen 5 V4 x16 chain64: parallel 155.219 vs 163.585 ns; ratio
  **0.941479–0.956297**, 250/250 clean. Hardware RCP14 is 46.491 ns,
  ratio **0.280605–0.287840**. Its accuracy is lower but inside MIDP.
- The same Zen 5 x16 stream64 is 1.641 vs 1.503 ns, only 172/250 clean;
  no stream win or qualified regression magnitude is claimed from it.
- M6 x8 stream64: parallel 5.885 vs 4.408 ns; RCP14 emulation 23.150 ns;
  HIGHP division 3.285 ns. All 0/50 clean and unconfined: exploratory only.

Selected bars from zenbench (process descriptive means; admission caveats above):

```text
i265/v3 v3/f32x8/stream64
  portable parallel MIDP   ████████ 6.0ns
  portable serial MIDP     ██████████ 7.8ns
  magetypes recip          █████████████ 10.0ns
  portable HIGHP division  ██████████████████████ 17.3ns
  canonical RCP14          ██████████████████████████████████████████████ 36.2ns

zen5/v4 v4/f32x16/chain64
  canonical RCP14          ████████████ 46.5ns
  portable HIGHP division  ████████████████████ 79.2ns
  portable parallel MIDP   ███████████████████████████████████████ 155.2ns
  magetypes recip          █████████████████████████████████████████ 163.6ns
  portable serial MIDP     █████████████████████████████████████████████ 177.8ns

```

## Decision and reproducibility

Keep the public portable serial MIDP graph for now. Parallel correction is a
useful latency/Intel-throughput option, not a universal improvement. RCP14
shows that asymmetric canonical algorithms are practical, at a substantial
emulation cost on the measured non-AVX-512 platforms. HIGHP division is already
portable and often wins dependency chains; select it when that workload and
precision make sense rather than assuming an estimate/refinement must be faster.
M6 suggests division is worth particular attention there, without a qualified
M6 speed claim.

```sh
just primitive-rcp14-reference OUTPUT
just primitive-recip-reference-data OUTPUT
just primitive-recip-qualify OUTPUT/recip-reference.u32le v3 --dense
just primitive-recip-qualify-wasm OUTPUT/recip-reference.u32le
# Prebuild, freeze source/binary, and launch through run-heavy and its owned scope:
just primitive-recip-qualify-perf BINARY v3
just primitive-recip-qualification-analyze RUN_DIRECTORY
just primitive-recip-qualification-process-analyze RUN_DIRECTORY
```

Durable raw results, all processes/rounds, trace logs, source, binaries, reference
oracle, compiler identities and analysis live under
`/mnt/v/output/archmage/recip-qualification-2026-11-11/`, mirrored with a checked
SHA256 manifest to `/mnt/tower/output/archmage/recip-qualification-2026-11-11/`.
Large generated results are external; these tools and this readable report are
tracked.

Resource wrappers: Zen 5 V3 **84 s, peak-RSS 0.26 GiB**; V4 **80 s, 0.25 GiB**;
Scalar **87 s, 0.25 GiB**. Intel E-core timing **61 s, 0.03 GiB**, build
**29 s, 0.46 GiB**. M6 timing **12 s, BSD wrapper peak-RSS 0.02 GiB**, with
monitored process-tree guard, not a cgroup peak claim. C native qualification
**1 s, 0.12 GiB**; streamed WASM validation **9 s, 0.33 GiB**. All failed
admissions and placement refusals remain preserved.

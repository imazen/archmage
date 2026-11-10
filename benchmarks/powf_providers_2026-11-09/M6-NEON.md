# M6: Arm scalar powf through NEON-capable callers

**A dedicated packed NEON algorithm remains unimplemented. Promotion remains held.**
This is the scalar table graph inside a batch, with guaranteed AArch64 FMA and
an independently required NEON-token evaluator. LLVM may vectorize a caller;
that does not make this a separately implemented packed backend.

## Numerical results

The M6 adds 60 full provider/width executions, 76 Gamma executions and 48
explicit scalar/required-NEON executions. All use the same 27,909 tagged pairs
or the independently filtered 8,698 Gamma records, broadcast and mixed lanes
at widths 1/4/8/16. The Rust port has zero sampled non-NaN bit, NaN-class or
payload differences against the explicitly unscaled Arm C reference on M6.
The separate 515 hard-to-round witnesses match that reference as well.

Apple std differs from the canonical Arm graph on 122 non-NaN tagged results,
eight NaN-class cases and twelve payload cases. Gamma still has 57 result-bit
differences. These are observed platform differences, not accepted port
mismatches. A finite startup probe does not establish arbitrary-input std
canonicality. MPFR remains the independent mathematical reference.

The combined inventory covers **424 provider/width executions** across Intel,
Zen 5, Neoverse-N1, M6, i686, no-FMA/AVX2-disabled emulation and strict/relaxed
WASM contexts. The Rust graph has zero non-NaN or NaN-class differences across
those samples. Every observed variance, including NaN payloads and rounded
MPFR differences, is retained: 196 tagged records, 7,291,911 uncompressed bytes.
This is bounded sampled coverage, not exhaustive binary-function coverage.

The C errno-enabled profiles also differ from Apple C powf on 7,097 errno
observations and 36 FP-exception observations. Raw result-bit differences total
142, decomposed above. The Rust port promises result bits under RNE/gradual
underflow, not C errno or FP flags.

## Interleaved timings

All 32 frozen cells completed with 50 retained paired rounds each. They remain
**exploratory**: the strict launch refuses macOS process confinement; the
separate unrestricted run preserves every resource observation. All 32 cells
are flagged and all have zero resource-clean pairs. For the x8/4096 stream,
the first observation records CPU load 0.1084307055. No timing qualifies under
the 2% promotion gate. No noisy rounds are removed.

Units throughout are **ns per 1,048,576 actual powers**, including one socket
request. Rust std and the Rust graph share the same Rust adapter. The C adapter
uses resolved function pointers and runtime widths; these results include its
wrapper work. EMPTY transport is retained and never subtracted. In particular,
the expensive C x1 wrapper must not become a claim about raw scalar powf latency.

Selected bars are copied from zenbench, with full names and raw ns means
replacing terminal truncation and automatic ms formatting. x8/4096 stream,
opaque exponent 2.4:

```text
Rust fused Gamma-assumed loop      ███████████████████                        1,385,673 ns
Rust fused Gamma-assumed unrolled  ████████████████████                       1,440,369 ns
Rust full fused loop               ███████████████████████                    1,645,329 ns
Rust full fused unrolled           ██████████████████████████                 1,915,617 ns
Rust std (Apple)                   █████████████████████████████              2,124,612 ns
C powf (Apple)                     ██████████████████████████████████████     2,779,547 ns
```

Full fused loop / Rust std paired 95% ratio CI is 0.773656–0.775222.
Explicit unrolling / Rust std is 0.898367–0.906519. These tight intervals do
not cure resource flags or unverified core-class placement.

Ranking changes with the dependency structure. At x8/4096 chain/2.4,
full fused loop takes 2,978,298 ns versus Rust std's 2,890,101 ns, ratio CI
1.028878–1.032192. Explicit unrolling takes 2,195,616 ns, CI
0.758546–0.760854 versus std. Conversely, unrolling x16/4096 chain/0.5 costs
2,444,386 ns versus std's 2,089,716 ns, CI 1.167514–1.172199.
The ordinary x1/4096 stream/2.4 full loop has CI 1.142182–1.145996 versus std.
These costs are logged observations, not hidden by wins elsewhere.

Cost-bound-only intervals for the distribution build, before the failed
resource/placement gates: full fused loop has 21 cells at upper CI <=1.02,
ten entirely above 1.02, one inconclusive; full unrolled has 24 and eight;
Gamma-assumed loop has 29 and three; Gamma-assumed unrolled has all 32 upper
intervals <=1.02. Separate arithmetic has only four such cells and 28 entirely
above 1.02. These counts are not promotion passes. Restricted assumptions can
save work, but no universal ordering of caller layouts follows.

Complete full-range comparisons (all ns per batch; bracketed values are paired
95% ratios versus Rust std). `s` = stream, `c` = normalized dependency chain:

| Width / buffer / mode / exponent | Rust std ns | Full fused loop ns [ratio] | Full fused unrolled ns [ratio] |
|---|---:|---:|---:|
| x1/64/s/2.4 | 2,122,831 | 2,361,151 [1.1079–1.1167] | 1,511,120 [0.7089–0.7149] |
| x1/64/s/0.5 | 2,125,611 | 2,371,416 [1.1126–1.1190] | 1,515,392 [0.7110–0.7149] |
| x1/64/c/2.4 | 13,884,379 | 11,839,067 [0.8496–0.8559] | 11,202,782 [0.8042–0.8096] |
| x1/64/c/0.5 | 13,887,948 | 11,724,514 [0.8418–0.8468] | 11,102,672 [0.7973–0.8018] |
| x1/4096/s/2.4 | 2,086,898 | 2,387,459 [1.1422–1.1460] | 1,508,901 [0.7219–0.7241] |
| x1/4096/s/0.5 | 2,088,371 | 2,382,602 [1.1391–1.1427] | 1,511,148 [0.7220–0.7253] |
| x1/4096/c/2.4 | 16,211,453 | 12,928,168 [0.7946–0.8006] | 13,009,768 [0.8003–0.8048] |
| x1/4096/c/0.5 | 15,420,053 | 12,434,846 [0.8024–0.8106] | 12,469,945 [0.8057–0.8118] |
| x4/64/s/2.4 | 2,167,492 | 1,705,014 [0.7850–0.7883] | 1,886,157 [0.8686–0.8716] |
| x4/64/s/0.5 | 2,169,371 | 1,716,694 [0.7865–0.7971] | 1,886,834 [0.8668–0.8732] |
| x4/64/c/2.4 | 4,913,417 | 5,294,714 [1.0731–1.0831] | 3,625,698 [0.7347–0.7418] |
| x4/64/c/0.5 | 4,930,411 | 5,267,464 [1.0620–1.0752] | 3,629,590 [0.7336–0.7394] |
| x4/4096/s/2.4 | 2,143,606 | 1,684,142 [0.7817–0.7908] | 1,882,040 [0.8738–0.8826] |
| x4/4096/s/0.5 | 2,149,025 | 1,689,209 [0.7810–0.7913] | 1,886,652 [0.8723–0.8840] |
| x4/4096/c/2.4 | 5,567,902 | 5,612,242 [1.0033–1.0124] | 4,006,291 [0.7178–0.7214] |
| x4/4096/c/0.5 | 5,412,290 | 5,514,831 [1.0150–1.0228] | 3,916,655 [0.7220–0.7256] |
| x8/64/s/2.4 | 2,112,098 | 1,660,034 [0.7850–0.7869] | 1,920,139 [0.9078–0.9105] |
| x8/64/s/0.5 | 2,111,100 | 1,660,717 [0.7851–0.7880] | 1,919,814 [0.9077–0.9109] |
| x8/64/c/2.4 | 2,750,248 | 2,845,907 [1.0320–1.0375] | 2,126,852 [0.7713–0.7753] |
| x8/64/c/0.5 | 2,765,514 | 2,847,991 [1.0257–1.0338] | 2,129,445 [0.7666–0.7736] |
| x8/4096/s/2.4 | 2,124,612 | 1,645,329 [0.7737–0.7752] | 1,915,617 [0.8984–0.9065] |
| x8/4096/s/0.5 | 2,124,806 | 1,644,247 [0.7730–0.7747] | 1,911,283 [0.8982–0.9012] |
| x8/4096/c/2.4 | 2,890,101 | 2,978,298 [1.0289–1.0322] | 2,195,616 [0.7585–0.7609] |
| x8/4096/c/0.5 | 2,823,219 | 2,957,703 [1.0458–1.0494] | 2,205,468 [0.7798–0.7825] |
| x16/64/s/2.4 | 2,084,350 | 1,657,345 [0.7944–0.7959] | 2,338,689 [1.1211–1.1230] |
| x16/64/s/0.5 | 2,086,801 | 1,657,007 [0.7923–0.7958] | 2,344,325 [1.1205–1.1268] |
| x16/64/c/2.4 | 2,090,028 | 1,811,068 [0.8648–0.8680] | 2,417,969 [1.1548–1.1586] |
| x16/64/c/0.5 | 2,090,462 | 1,816,134 [0.8666–0.8713] | 2,418,115 [1.1556–1.1578] |
| x16/4096/s/2.4 | 2,096,523 | 1,647,928 [0.7848–0.7877] | 2,326,492 [1.1085–1.1108] |
| x16/4096/s/0.5 | 2,094,994 | 1,646,669 [0.7851–0.7870] | 2,326,989 [1.1097–1.1118] |
| x16/4096/c/2.4 | 2,085,942 | 1,835,381 [0.8776–0.8831] | 2,405,848 [1.1515–1.1551] |
| x16/4096/c/0.5 | 2,089,716 | 1,834,016 [0.8750–0.8818] | 2,444,386 [1.1675–1.1722] |

## Code generation and native flags

Frozen actual Mach-O binaries were inspected with cargo asm 0.2.64, rebuilt
with its disassembly feature, and llvm-objdump. The scalar fused probe is
1,004 bytes with nine FMADD instructions, no FMA libcalls and no bounds-check
panic; the separate probe is 1,088 bytes with zero FMADD. Actual x8 worker
execution code is retained as well; its shared function includes other methods
and is not used to attribute their packed instructions to this graph.

The first C native profile requested native CPU scheduling while retaining
`-march=armv8-a`. The driver confirms that baseline ISA restriction. It remains
explicitly labeled in the first matrix. The corrected full-native profile
removes that flag, passes the numerical replay, and has its own preregistered
50-pair exploratory pilot: C full-native 2,836,032 ns, Rust std 2,102,732 ns,
Rust full fused loop 1,620,307 ns. The changed C configuration does not turn
these control measurements into a favorable rerun of the original matrix.

The hardware identifies as Apple M6. The installed C compiler maps native CPU
to apple-m4; pinned Rust maps it to apple-m5. Exact driver features, compiler
identities and Cargo flags are retained. No M6-specific compiler scheduling
claim is inferred from the word native.

Whole-process hardware counters for 104,857,600 fixed powers, x8/4096 stream,
include startup and adapter work. Darwin's native time utility reports retired
instructions and cycles; these are not per-call latency estimates:

| Implementation | Instructions | Cycles |
|---|---:|---:|
| Apple C powf | 9,876,906,325 | 1,277,020,836 |
| Arm C unscaled FMA | 10,427,887,576 | 1,305,121,668 |
| Rust full fused loop | 6,575,207,921 | 751,073,489 |
| Rust full fused unrolled | 8,593,841,507 | 874,131,281 |
| Rust Gamma-assumed unrolled | 5,513,651,948 | 658,589,594 |

The first directory labeled profile-x1 actually retained default x8 requests;
its immutable manifest exposes that. It remains as an explicitly marked setup
error. The corrected profile-x1-actual uses explicit --width 1. It reports
20,979,672,957 instructions for the C adapter, 10,297,118,862 for the full Rust
loop and 8,070,849,344 for Rust std. This reinforces the wrapper/layout caveat.

## Native tests and the Miri limit

Native M6 tests, validation, API snapshots and documentation checks pass through
Step 14 of full CI. Step 15 fails because AArch64 Miri cannot interpret
`llvm.aarch64.neon.sshl.v4i32`, reached by automatic powf dispatch. Both the
pinned October 6 interpreter and the November 7 nightly reproduce it. The
existing zero-shift contract and all assertions are retained; no test is
skipped or weakened. Thus **native M6 full just ci does not pass**. x86 full CI
including Miri passes, and the pushed a29d7a35 GitHub run succeeds. The ARM
interpreter limitation is an explicit coverage gap, not a native-result defect.
An earlier local CI launch resolved an old bundle instead of the PR revision;
that attempt was stopped, preserved, and excluded. The correct attempt verifies
its exact parent a29d7a35 before running.

Reported run-heavy records: build/numerical comparison 77s, peak RSS 0.03GiB,
minimum available-memory proxy 9,070MiB, peak load 2.81; required NEON replay
18s/0.04GiB/9,185MiB/2.18; 32-cell matrix
224s/0.03GiB/9,777MiB/2.86; counter diagnostics
3s/0.02GiB/10,004MiB/1.74; failed native full CI
92s/0.76GiB/8,458MiB/5.23. macOS uses a monitored RSS guard and a labeled
available-memory proxy, not a Linux cgroup hard-cap claim.

All variances, paired rounds, failed attempts, actual binaries, C sources,
compiler flags and disassembly are preserved separately from the sealed earlier
study. See [ARTIFACTS.md](ARTIFACTS.md), [ARM-RESULTS.md](ARM-RESULTS.md) and
[ARM-PORT-PLAN.md](ARM-PORT-PLAN.md). No production math source changed.

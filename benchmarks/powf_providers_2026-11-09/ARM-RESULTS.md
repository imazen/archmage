# Scalar Arm C and experimental Rust powf port

**Promotion remains held. No production backend changed.** The isolated C
comparison completed before the Rust port began. The Rust implementation now
has matching sampled results and competitive observed timings, but **zero
qualified timing cells**. The complete 32-cell performance matrix, physical
cross-platform cost coverage, first-use costs and an arbitrary-input accuracy
proof remain missing. The preregistered pilot stop rule was honored.

## What was compiled

Arm Optimized Routines [v26.07 source](https://github.com/ARM-software/optimized-routines/blob/v26.07/math/powf.c), revision
`4be260a5117480382690c6d8c300bc784e927d76`, supplies `powf.c`, its two data
translation units and error helpers. The isolated entry is renamed after
including math.h, avoiding symbol interposition. Source hashes, commands,
compiler identities, disassembly and all five DSOs are retained.

The x86 C profiles are baseline separate arithmetic with errno; AVX2+FMA with
errno; AVX+FMA without AVX2 with errno; AVX2+FMA without errno; and the owner's
requested host-native build with errno. All use O3, rounding support, standard
excess precision and no fast math. The native diagnostic is rebuilt on each
physical CPU; frozen manifests reject moving it to a different CPU context.

The safe Rust port lives only in this unpublished benchmark package. It uses
the unscaled table reduction, exact binary64 constants extracted by compiling
the pinned C data, f64 intermediates and explicit fused/separate arithmetic.
Arm's MIT license is preserved in [ARM-LICENSE.txt](ARM-LICENSE.txt).
Rust [cold_path](https://doc.rust-lang.org/beta/std/hint/fn.cold_path.html) is stable since 1.95; this standalone nightly package uses it
without raising the production workspace's 1.89 MSRV. Actual linked binary
disassembly comes from cargo asm 0.2.62, rather than a separately rebuilt proxy.

`Arm_Rust_fused_cold` and `Arm_Rust_fused_warm` include full special/range
handling; the name's final word means cold hints enabled/disabled.
`Arm_Rust_separate_cold` uses separate multiply/add. `fused_unrolled` explicitly
expands up to sixteen lanes. `fused_assume_Gamma` and
`assume_Gamma_unrolled` omit classification/range work only on the established
Gamma domain: +0 or normal bases in [2^-16,2^16], normal-or-zero exponents in
[0,4]. Those assumed-input arms are validated only within that domain.

## Correctness and observable differences

The original Intel C comparison has zero result-bit differences against
installed glibc 2.43 on 27,909 tagged pairs, broadcast and mixed lanes at widths
1/4/8/16, and on the separate 515 hard-to-round witnesses. All errno-enabled
builds also match errno and FP exception flags. The errno-disabled build has
7,097 errno differences, zero exception-flag differences and identical result
bits. This is sampled equivalence, not proof for every binary32 pair.

The Rust port promises result bits under RNE and gradual underflow; it does
**not** promise C errno or FP exception side effects. Intel and Zen 5 full
sample inventories contain zero non-NaN bit or NaN-class differences against
the observed C/std reference. NaN payload/sign differences are retained in
complete ledgers: Zen 5 full profiles have 46, 47 or 49 differing tagged cases,
depending on code generation. The explicit unrolling changes some NaN payloads;
it does not acquire a payload guarantee. All 8,698 Gamma records, including
38 +0 cases, match exactly across the checked full and assumed-input arms.

Independent MPFR-256 enclosures remain the mathematical reference. The broad
sample's maximum finite-output error upper bound is 0.506155662239 ULP; Gamma's
is 0.501247502864 ULP. The 515 hard witnesses have 110 rounded-result differences
for both glibc and the port, with maximum 0.500007618219 ULP. These sampled
maxima are not global accuracy guarantees or correctly-rounded claims.

The portable evaluator additionally passed 148 provider/width executions on
the same 27,909 tagged pairs, with zero non-NaN or NaN-class differences:
baseline scalar, required V3, i686, emulated Nehalem without FMA, emulated
Haswell with AVX2 disabled, strict SIMD WASM scalar/native, and relaxed-module
WASM scalar/native under both Cranelift FMA settings. Required token contexts
fail when unavailable. Emulation supplies no speed evidence. This scalar port
uses std f64 mul_add in WASM; it has not integrated Magetypes' relaxed-FMA
probe and therefore does not demonstrate that acceleration path.

Real Neoverse-N1 hardware adds 40 full-domain provider/width executions and
48 Gamma executions, again with zero sampled non-NaN or NaN-class differences.
Its C references include Armv8 fused/separate, an explicitly unscaled fused
profile and a locally compiled native profile. The baseline AArch64 Rust build
uses the architecture's guaranteed FMA support; it is not an x86-native binary
moved to ARM. No new Neoverse timing was collected. The earlier frozen Intel
V1 native binary was also replayed retrospectively: 56 provider/width
executions plus all hard witnesses pass the same sampled result contract.

## C-to-Rust code generation and counters

The first Rust worker used array::from_fn inside the V3 operation. LLVM outlined
its callback outside the feature region, producing nine indirect FMA calls per
ordinary lane. Inspecting only the scalar V3 probe had missed this. Its actual
worker assembly and hardware counters exposed the cause. An explicit inlined
array loop preserves the V3 feature context: no FMA libcalls or bounds-check
panics remain on that path. Intel whole-process instructions for 104,857,600
fixed powers fell from 19,157,264,786 to 6,626,755,364; both complete records
are retained. This was a source fix, not a favorable rerun of an unchanged build.

Zen 5 whole-process counters, same 104,857,600 powers per profile, including
startup and adapters (these are not single-call latency counts):

| Implementation | Instructions | Cycles |
|---|---:|---:|
| C glibc | 9,330,745,513 | 1,784,803,141 |
| Arm C AVX2+FMA errno1 | 9,330,961,622 | 1,783,368,698 |
| Rust full fused loop, distribution build | 6,626,801,089 | 1,398,867,230 |
| Rust full fused unrolled, distribution build | 6,377,768,047 | 1,306,559,133 |
| Rust full fused unrolled, native build | 6,464,731,818 | 1,313,394,845 |
| Rust full fused loop, native build | 2,571,685,436 | 2,279,366,960 |

The full x8 loop kernel is 4,711 bytes; explicit unrolling expands it to 16,527
bytes. Unrolling saves loop work at a substantial code-size cost. Native
compilation changes the full loop into vectorized code: the linked x8 execution
function contains ZMM operations and gathers. Its lower instruction count comes
with a higher measured cycle count. The native assumed-Gamma loop needs
789,115,777 instructions and 609,998,326 cycles; removing exceptional work enables
a different optimization result. Thus native compilation and manual unrolling
are not unconditional improvements. No instruction-count ratio is presented as
a wall-clock speed ratio.

Direct cold_path hints and the unhinted scalar V3 probe merge to the same
861-byte function in the inspected build. There is no demonstrated standalone
benefit from that hint. The experiment preserves both arms.

## Interleaved zenbench timings

Every number below is ns per **1,048,576 actual powers**, one persistent-worker
socket request, logical x8, 4096-element stream, opaque exponent bits 4019999a
(2.4). The C and Rust adapters differ; EMPTY controls are retained and never
subtracted. The C adapter calls resolved function pointers; the Rust port can
inline the graph into its batch. Thus this comparison includes dispatch/call
granularity, rather than claiming equal raw scalar function latency. Rust std
in the same Rust worker is the stronger replacement-cost
comparison. All 50 interleaved paired rounds are retained.

Intel's pre-port C study: glibc 3,855,679 ns; Arm C AVX2+FMA errno1 3,874,101 ns;
Arm C native errno1 3,890,703 ns. AVX2+FMA/C-glibc paired 95% ratio CI is
0.992418–1.018908. Only 47/50 pairs were resource-clean, so the cell is
**inadmissible**, despite that interval fitting the 2% cost bound.

Zen 5's first launch refused before timing because the performance policy could
not choose between equal-ranked cache classes. The corrected launch used the
documented auto policy plus an enforced 96-MiB-class cpuset, preserving all
resource gates. The resulting timing run has **0/50 clean pairs** and
`unreliable=true`; the first recorded gate observation reports CPU load
0.0806069520. It stopped the matrix. No unchanged source was rerun for a more
favorable result.

Selected bars below are copied from zenbench's complete sorted bar chart;
full implementation names and raw ns means replace terminal truncation and
automatic ms formatting. They are **inadmissible observations**:

```text
Rust full fused unrolled, native        ████████████████████████            2,590,222 ns
Rust full fused unrolled, distribution  █████████████████████████           2,616,181 ns
Rust std, distribution                 ██████████████████████████          2,708,202 ns
Rust std, native                       ██████████████████████████          2,795,424 ns
Rust full fused loop, distribution      ██████████████████████████          2,797,357 ns
Arm C native errno1                    █████████████████████████████████   3,488,227 ns
Arm C AVX2+FMA errno1                   ██████████████████████████████████  3,632,406 ns
C glibc                                ███████████████████████████████████ 3,659,248 ns
Rust full fused loop, native            █████████████████████████████████████████ 4,392,119 ns
```

Distribution Rust unrolled / Rust std has paired 95% ratio CI
0.940239–1.007889; versus C glibc it is 0.698149–0.754727. Native Rust unrolled /
distribution Rust std is 0.932587–0.995698. **None passes admission**, regardless
of the interval. Native full-loop and separate variants show substantial costs;
all profiles, payload variances and noisy rounds remain available.

Resource records: Intel C verification 14s, peak RSS 0.37GiB, minimum available
22,598MiB, peak load 1.95; Intel C pilot 2s/0.02GiB/23,014MiB/1.62. Zen 5 build
48s/0.50GiB/42,688MiB/3.14; numerical verification
41s/0.72GiB/43,117MiB/3.68; counter run
2s/0.02GiB/43,833MiB/3.55; timing pilot
5s/0.26GiB/44,113MiB/2.09. These are run-heavy summaries, not estimated memory.
Neoverse build plus numerical verification: 177s, peak RSS 0.51GiB, minimum
available 27,011MiB, peak load 7.35. Its existing background work was preserved.

The evidence supports continuing the same-algorithm approach through Archmage.
It does not yet qualify replacing std, LOWP or MIDP. See
[ARM-C-PLAN.md](ARM-C-PLAN.md), [ARM-PORT-PLAN.md](ARM-PORT-PLAN.md) and
[ARTIFACTS.md](ARTIFACTS.md) for the frozen protocol and evidence locations.

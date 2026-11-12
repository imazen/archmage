# Bounded widening and canonical reciprocal rounding — 2026-11-11

This is a benchmark-only study in an unpublished Cargo package. Public Magemath graphs, options,
accuracy expectations and Magetypes methods are unchanged.

**The numerical approach works; a faster-than-native portable replacement is
not established.** Bounded widening preserves the current fused graph and
reduces the measured software cost. It remains slower than division and native
FMA in the recorded workloads. Canonical RN14/RN8 agrees across the tested ISAs,
but rounding/certification adds work and spends accuracy relative to current
MIDP. The fixed runs retain admission failures; none licenses a new default.

## Target and independent checks

The domain is **2^-125 ≤ abs(x) ≤ 2^125**, round-to-nearest-even and gradual
underflow. The widened graph must match current portable LOWP/MIDP fused bits.
The new canonical targets are the mathematical reciprocal rounded directly to
14 or 8 significant binary digits, encoded as f32. Their relative bounds are
2^-14 / 2^-8 on this domain, inside MIDP1e-4 / LOWP6e-3 respectively.
These are separate targets; canonical rounding does not preserve Newton bits.

The streamed C oracle writes four columns: libc-fmaf MIDP, libc-fmaf LOWP,
integer-division RN14, integer-division RN8. It holds 8,388,608 positive
normalized inputs in a 128 MiB file, generated with a fixed 16 KiB buffer and
read with a 64 KiB buffer. No input-space-sized array is allocated.
SHA256: `7f48c84de3459ab75da0ffe7ba63c69556fe89a76ff3085065c6c1a18ebafc4d`.
The repository's independent Fraction-based FMA rounder and rational reciprocal
rounder agree with all four C columns on 1,029 directed/random inputs.

Dense x4/x8/x16 checks enumerate every signed normalized mantissa, plus directed
mantissas at every accepted exponent and MIDP/RN14 both-sign same-width tail
spot checks. The lane
counts per context are 16,787,220 / 16,797,224 / 16,817,232, excluding additional
tail evaluations. They pass Scalar, V3, V4, V4x, physical Intel V3, native M6
NEON std/no_std, Wasmtime strict/relaxed, forced non-FMA and no_std modules.
Normalized enumeration plus exact power-of-two homogeneity covers the exponent
reduction; this is not a literal full-domain sweep at every exponent.

Maximum measured relative errors, using the exact f64 product x*y minus one:
current/widened MIDP **1.1415493993638393e-5**, RN14
**6.1027705669403076e-5**, RN8 **0.003891050349920988**. All observed widened
and certified output bits match their respective independent C columns.

## Why bounded widening is exact

A binary32 product has at most 48 significant bits. The exact product x*y in
each Newton correction lies in [0.5,2], so subtracting it from two needs at
most 49 significant bits. Both operations therefore fit exactly in binary64;
only conversion back to binary32 rounds. The following multiplication by y
remains binary32, preserving the original graph's rounding points.

The C audit enumerates the seed and first refinement and records these exact
product intervals:

- Seed: `[0x1.e24fd4p-1, 0x1.0a2a0b6690f2p+0]`.
- First refinement: `[0x1.fe474b094p-1, 0x1.000001cabd6p+0]`.

The bit seed scales inversely with the input exponent, so the products are
invariant under normal-domain power-of-two scaling. Rounded outputs remain
normal. This justifies the exponent reduction and the second step by induction.
This specialized computation is not a general-purpose f32 FMA replacement.

WASM expresses packed promote/multiply/subtract/demote/shuffle operations.
The inspected LLVM output still contains scalar extract/demote/insert sequences;
the source spelling does not establish the emitted instruction count. Both
versions pass the same oracle; their binaries are retained.
Hardware-FMA contexts are measured as comparisons, but the intended use of
widening is a software fallback, not replacement of native FMA.

## Why canonical correction is necessary

Division then naïve bit rounding disagrees with direct mathematical rounding
on **3,980 RN14 inputs** and **54 RN8 inputs** in the normalized scan. Retained
witnesses include input `3ffffc00` / `3fff00ff`. The new paths correct these
rather than treating a finite probe as a universal guarantee.

All three implementations select the same canonical result:

1. Native working `recip()` proposal, certify every vector.
2. Native working proposal, certify vectors with any lane within 16 binary32
   encoding units of a coarse midpoint. This conservative guard relies on the
   existing working reciprocal's ≤4-ULP normal-domain contract; the guard and
   proposal error are much smaller than an RN14/RN8 bin.
3. Correctly rounded division proposal, certify only coarse midpoint results,
   which are the ambiguous double-rounding cases.

Here “native” in the retained benchmark labels names the existing Magetypes
working `recip()` proposal, not a raw hardware estimate instruction. Scalar and
WASM implement that proposal with division. The division-proposal variant can
use its tighter ambiguity guard there; both variants still target the same RN14
or RN8 result.

Certification normalizes x and the proposed rounded answer. Midpoints then
have at most 16 significant bits. A hardware FMA computes the sign of
`1 - x*midpoint`; the smallest nonzero normalized residual is well above f32
underflow, so rounding cannot erase its sign. Without FMA, the ≤40-bit product
and subtraction are exact in f64. The sign chooses the adjacent bin; parity
handling retains RNE at equality. Input/output scaling is restored with integer
bit operations. Each width and scalar tail targets the same integer quotient /
remainder result, regardless of the original estimate bits.

## Frozen measurement plan

Use zenbench pinned to **73302a1c115836b2646f6a9adc32058c20d1a66f**, including
observer-affinity and macOS CPU-counter fixes. No target-cpu=native. Strict
confirmation uses five independent processes and exactly 50 retained paired
rounds, a 2% margin and family confidence95%. Promotion requires every mandatory
ordinary-input cell clean and upper95 ratio≤1.02; an optimization claim also
requires upper95<1 in an intended cell. Dirty and boundary-stress results remain
visible. There is one fixed run per source/context, with no favorable retries.

Seven named implementations:

- Existing Magetypes `recip()`.
- Current portable serial MIDP, Normal, AssumeValid, Caller.
- Bounded f64 MIDP, same fused graph bits.
- HIGHP division, Caller.
- RN14 native always certified.
- RN14 native guarded.
- RN14 division guarded.

Each context has 13 groups: x4/x8/x16 ordinary signed streams64/4096 and positive
chain64; x8 bit-distributed streams64/4096 over the normal domain, and x8 streams
concentrated at RN14 rounding boundaries. The latter deliberately exercise the
correction path. RN8 is numerically qualified here; this phase times RN14 only.
Contexts: Zen5 Scalar/V3/V4, Intel Skymont E V3, M6 NEON exploratory, and hosted
WASM strict/fusing/forced-nonfusing. M6 remains excluded from strict promotion
because macOS does not enforce one core class.

Linux runs enforce a uniform CPU-class cpuset. Their measurement thread is
unpinned; traces report observer `no_pin` and inherited affinity. Dedicated
spare-observer-CPU isolation and verified frequency are not claimed. The cold
ablation enables gate tracing, retaining its observer work for every arm.

The WASM host uses patched zenbench and Wasmtime49.0.5. Compilation and instance
setup are outside samples. Calls batch 4096 complete short-stream workloads or
256 long-stream/chain workloads, with guest black_box barriers inside every
repetition. Tables report ns/workload after dividing by that fixed batch;
raw packets/bars retain ns/host call, including host-call and guest-lock costs.
The actual relaxed-FMA probes are 0/1/0 for strict/fusing/nonfusing respectively.
Before timing, the guest checks every result lane for every arm and workload.
The forced-nonfusing host requests a baseline x86_64 engine target; this is not
an isolated FMA-only codegen switch. Compare implementations within each mode,
and do not attribute every timing difference between engine modes solely to FMA.

The first matrix is complete. Every strict suite fails admission. Ten of its
91 strict groups have all 250 observations clean individually, all on Intel;
none of the 13 M6 groups has a clean round. M6 remains exploratory regardless
of the packet's aggregate flag. All 104 groups and every timing round are kept.
The [complete first matrix](CERTIFIED-RECIP-FIRST-TIMINGS.md) includes every
width, ordinary stream/chain and bit/boundary-stress result.
The corrective matrix is complete and reported separately, rather than
replacing the first measurements.

### First matrix: same fused graph, different implementation

All values below are ns per **64-element ordinary stream**, logical x8, including
the stream's loads/stores. Intervals compare bounded f64 with current portable
serial MIDP, using all retained data and five independent-process log means,
Student-t intervals adjusted over all 91 reported comparisons plus the timer floor. These are diagnostic
intervals on failed suites, not confirmed optimization claims. M6 has one
exploratory process, so no independent-process interval is available.

| Context | Current portable serial MIDP | Bounded f64 MIDP | HIGHP division | Bounded / current family95 interval | Clean pairs |
|---|---:|---:|---:|---:|---:|
| Zen 5 Scalar Caller | 109.341 | 27.608 | 7.097 | 0.249777–0.255241 | 200/250 |
| Zen 5 V3 | 2.823 | 19.876 | 3.495 | 6.525589–7.600840 | 140/250 |
| Zen 5 V4 | 2.887 | 9.916 | 3.566 | 2.902692–4.068793 | 199/250 |
| Intel Skymont E V3 | 7.802 | 30.883 | 17.254 | 3.869132–4.049459 | 250/250 |
| M6 NEON exploratory | 5.789 | 18.007 | 3.240 | Not measured across processes | 0/50 |
| WASM strict | 231.842 | 70.194 | 7.309 | 0.298115–0.307494 | 148/250 |
| WASM relaxed, fusing | 53.666 | 70.678 | 7.360 | 1.229345–1.411055 | 169/250 |
| WASM relaxed, nonfusing | 268.096 | 87.311 | 7.299 | 0.322154–0.329227 | 133/250 |

This is a **software fallback specialization**: preserve native FMA and the
cached fusing relaxed-WASM provider. Replacing those paths with widening is a
regression in these measurements. Calling widened arithmetic portable does not
make it faster than the already portable HIGHP division.

Raw zenbench bars from the first Scalar x8 stream64 report (display rounding):

```text
portable HIGHP division              ███ 7.1ns
magetypes recip                      ███ 7.1ns
bounded f64 MIDP (same fused bits)    ███████████ 27.6ns
portable serial MIDP                 ██████████████████████████████████████████ 109.3ns
```

The raw complete bar reports and packets remain in the artifact directories.
The table uses all-retained-round arithmetic means. Bootstrap diagnostics use
10,000 paired within-process resamples (PCG64 seed 0x531324); no IQR or flagged
round removal is used. Separate process intervals do not clear resource flags.

### What is and is not covered

This phase times **MIDP reciprocal**, RN14 and existing HIGHP/native reciprocal,
not LOWP/RN8 timing, rsqrt optimization, f64 reciprocals, first-use dispatch,
consumer application speed, or cold-build comparison. LOWP/RN8 numerical checks
are included. The accepted domain excludes zeros, subnormals, infinities and
NaNs. No claim is made for arbitrary rounding/flush modes or future estimates
that violate the proposal's stated error contract.

Chain64 begins with the same opaque 0.75 in every lane, then repeats reciprocal
plus 0.5. Scalar optimization may share identical lane work; those timings are
for this exact workload, not evidence of independent-lane throughput.
The native proposal uses existing working `recip()`, including its rails; these
results are not a lower bound on every possible domain-specialized algorithm.
WASM canonical certificates use exact f64 even in a fusing engine in this
prototype. RN14's measured error is larger than existing fused MIDP, although
both meet MIDP1e-4. No accuracy contract is silently traded for a timing result.


## Corrective implementation ablation

The first scalar assembly speculates normalization and saturating float-to-int
conversions before the guard, with a large stack frame. The corrective version
keeps the same proposal and rounding guard, but invokes a cold, non-inlined
integer certificate only for uncertain vectors. It compares the normalized
24-bit input significand times each midpoint numerator against a power of two;
these products fit in u64. Original inline and new cold variants are both
retained. Their independent oracle targets are identical.

For the inspected Scalar x8 division-proposal stream64 function, the explicit
stack allocation decreases from `0x468` (1128) to `0x68` (104) bytes. This is
compiler stack reservation, not a heap/RSS measurement. The cold call is behind
the uncertainty branch and the hot loop has no float-to-int repair conversion.
Both disassemblies and the exact executable are preserved. This establishes the
codegen change; timings still determine whether it helps each workload.

This is a new source qualification, not a timing retry. The corrective matrix
uses the same 13 groups, five processes and 50 rounds, with six named arms:
existing recip, HIGHP division, native/division guarded inline, and native/division
guarded cold. It covers the same contexts and retains the original study data.

### Cold integer repair: measured tradeoff

All values below are ns per x8 ordinary stream64. Intervals compare cold
division-proposal repair with inline division-proposal repair, using the same
91-comparison family correction. Every strict suite fails admission; only one
of 91 strict groups is individually all-clean (V4 x4 chain64). M6 has 0/50
clean rounds per group and remains exploratory. The [complete corrective
matrix](CERTIFIED-RECIP-COLD-TIMINGS.md) retains all 104 groups and all six arms.

| Context | HIGHP division | RN14 division inline | RN14 division cold | Cold / inline family95 interval | Clean pairs |
|---|---:|---:|---:|---:|---:|
| Zen 5 Scalar Caller | 7.067 | 107.428 | 13.284 | 0.122528–0.124784 | 183/250 |
| Zen 5 V3 | 3.470 | 6.820 | 6.826 | 0.995840–1.005859 | 141/250 |
| Zen 5 V4 | 3.494 | 6.935 | 6.576 | 0.931150–0.965701 | 212/250 |
| Intel Skymont E V3 | 17.219 | 19.471 | 19.773 | 1.009395–1.021705 | 82/250 |
| M6 NEON exploratory | 3.302 | 9.671 | 9.597 | One exploratory process | 0/50 |
| WASM strict | 7.333 | 17.069 | 14.694 | 0.853794–0.867999 | 169/250 |
| WASM relaxed, fusing | 7.337 | 17.154 | 14.734 | 0.850199–0.867712 | 140/250 |
| WASM relaxed, nonfusing | 7.329 | 18.411 | 15.135 | 0.815570–0.828635 | 178/250 |

Cold division repair helps the Scalar hot path, but HIGHP division still costs
less. Cold **working-recip proposal** repair regresses ordinary x8 streams:
20.399 → 72.034 ns on Zen 5 V3, 58.078 → 107.544 ns on Intel Skymont E,
and 27.896 → 45.349 ns on exploratory M6. Moving repair out of line is not a
universal improvement. Boundary-heavy input makes cold calls frequent; both
cold variants are substantially slower there on Intel/NEON. No variant is
promoted on the strength of a favorable average or the Scalar stack reduction.

## Validation, reproduction and resources

The benchmark package passes strict Clippy with std/defaults, no-default w512,
and the optional native WASM host. Its default targets pass MSRV1.89; the
optional Wasmtime49 host was built/tested with Rust1.99.1. Dense i686 Scalar
checks pass at all three widths. The i686 probe/import warnings found during
qualification were fixed with target cfgs; no lint or assertion was relaxed.
WASM numerical CLI tests use Wasmtime40.0.1, while hosted timing/verification
uses Wasmtime49.0.5. Those engine identities are deliberately separate.

The no_std x86 baseline correctly refuses to summon V3 dynamically. Its Scalar
check passes; the V3 no_std check passes when compiled explicitly for x86-64-v3.
This correctness build is not a target-cpu=native timing run. M6 no_std and all
five WASM std/no_std/engine-choice numerical checks pass unchanged expectations.
The oracle and test readers use fixed buffers, and timings reuse fixed short/long
inputs in paired interleaved rounds; they do not allocate an input-space array.

Library, generator, workspace manifest/lockfile, workflow and existing test
sources are unchanged from complete [green CI67dead91](https://github.com/imazen/archmage/actions/runs/42956052496).
The separate worker replay passes generation, registry/token validation,
soundness and the first 14 full-suite stages. Its redundant unchanged-library
Miri run was stopped and its partial log retained; a fresh complete local suite
is not claimed. The new unpublished package's exactness/codegen/lint evidence is
recorded separately above. The PR's new-head CI runs separately.

Reproduction entry points (use a new external output directory for the 128 MiB
reference; preserve existing fixtures):

```text
just primitive-recip-round-reference REF_DIR
just primitive-recip-round-audit REF_DIR/recip-round-reference.u32le REF_DIR
just primitive-recip-round-check REF_DIR/recip-round-reference.u32le scalar --dense
just primitive-recip-round-check REF_DIR/recip-round-reference.u32le v3 --dense
just primitive-recip-round-wasm REF_DIR/recip-round-reference.u32le
just primitive-recip-guest GUEST_DIR strict
just primitive-recip-guest GUEST_DIR relaxed
just primitive-recip-wasm-host-build
just primitive-recip-certified-analyze RESULT_DIR first
just primitive-recip-certified-analyze COLD_RESULT_DIR cold
just primitive-recip-certified-table RESULT_DIR
```

`--certified-study` selects the seven-arm matrix; `--cold-study` selects the
six-arm repair ablation. `run_certified_matrix.py` expects already frozen native
and host binaries, strict/relaxed guests and SOURCE.json. Supply the absolute
resource-wrapper path through ARCHMAGE_HEAVY_WRAPPER; zenbench still owns
admission and benchmark serialization. Every failed cell and its return code is
kept, with no automatic retry. Analysis writes a new file and refuses overwrite.

Each timing job has an 8 GiB Linux cgroup cap; M6 uses its monitored RSS guard
and BSD-time peak, without a cgroup hard-cap claim. These are the actual wrapper
resource summaries, separate from stack reservations and source buffer sizes:

| Context | First study seconds / peak-RSS GiB | Cold ablation seconds / peak-RSS GiB |
|---|---:|---:|
| zen5/scalar | 135 / 0.27 | 122 / 0.27 |
| zen5/v3 | 141 / 0.26 | 139 / 0.26 |
| zen5/v4 | 143 / 0.26 | 146 / 0.27 |
| i265/v3 | 114 / 0.03 | 101 / 0.03 |
| mac/neon | 23 / 0.02 | 19 / 0.02 |
| wasm/strict | 250 / 0.28 | 215 / 0.28 |
| wasm/fusing | 229 / 0.28 | 209 / 0.28 |
| wasm/nonfusing | 248 / 0.28 | 210 / 0.28 |

The initial optional-host build used 193 s / 2.14 GiB peak-RSS. Cold-variant
multi-ISA qualification used 36 s / 0.53 GiB; the final i686 build and dense
check used 14 s / 0.45 GiB (its following documentation whitespace check was
corrected). Final std/no_std/host Clippy, fmt and source integrity checks pass
in 3 s / 0.33 GiB. No cold-build speedup or binary-size comparison is claimed.

## Artifacts and source identity

First frozen inputs: `d5026a6d4a40aecec00df4c6ac196baffdcedac3`.
Cold-repair inputs: `29f36d84dd258d4cab64f412cc08027f52c67e6a`.
Later changes add analysis/docs and scope imports/probes on i686; the timed math
and frozen binaries are unchanged. Zenbench's git field comes from the launcher
checkout, not the compiled source. SOURCE.json, source.tar, binary/module hashes
and per-host compiler records identify the actual inputs.

External directories:

- `/mnt/v/output/archmage/recip-certified-2026-11-11/`
- `/mnt/v/output/archmage/recip-certified-cold-2026-11-11/`

They preserve source archives, exact binaries/modules, the C reference, compiler
and flags, complete paired packets, child-process logs, all failures, disassembly,
all-retained-round analysis and resource lines. The earlier analysis with only
78 baseline comparisons is retained as v1; final intervals correctly include
all 91 reported comparisons. No timing data changed for that correction.

Verified NAS mirrors use the same two directory names under
`/mnt/tower/output/archmage/`. Every file was read back and checked against its
SHA256 manifest on 2026-11-11:

- `recip-certified-2026-11-11`: 110 files; manifest SHA256 `7d44379dbc7247e2557f875fe353b15c217d62182eebeca191392555aa87a30e`.
- `recip-certified-cold-2026-11-11`: 94 files; manifest SHA256 `c64c541696a427a72e9dae9f329df311f76560facee539541abe12bd23899557`.

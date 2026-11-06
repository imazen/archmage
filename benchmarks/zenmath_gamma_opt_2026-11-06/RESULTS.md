# Measured scalar candidates (initial and refined)

No exhaustive domain sweep or production integration yet. Whole-transfer controls
and the separate f64 curve oracle are still pending.
All measurements below are native scalar stream timings on r5900xt, local
source/target/toolchain/temp, pinned Rust1.99.1, empty actual Cargo rustflags,
release thin LTO/codegen-units1. Std is a system-libm throughput control;
independent CORE and directed MPFR provide correctness gates.

The exact-v full-mantissa certificate passed52 combinations in smoke and directed
MPFR sampling. Coarse degrees8/12/16/20: best decode1.04–1.15x std time,
encode1.33–1.42x. Refinement degrees11..17: best decode1.03–1.09x std,
encode1.18–1.27x. Estrin13 decode and15 encode are retained as global controls.
All speed ratios compare measured operations, not extrapolations.

The new16-center certificate passed64 combinations (52 global+12 piecewise),
with MPFR containment required separately from output equality, including
rescued cases. Piecewise degrees3,4,5 and both layouts were sampled; no claim
of64 full-domain sweeps. The corpus is boundary-biased (directed inverse
midpoint inputs and interval centers/edges), so oracle rescue fractions are
not throughput workload estimates. Actual throughput input rescue counts are
retained separately in each run's throughput-rescues.csv.

| Kind/distribution/length | Best piecewise method | Time/std |
|---|---|---:|
| decode/values/64 | Estrin4 | 0.818509 |
| decode/bits/64 | Estrin4 | 0.825780 |
| decode/values/4096 | Estrin5 | 0.845333 |
| decode/bits/4096 | Estrin4 V3 | 0.834757 |
| encode/values/64 | Estrin4 | 0.936434 |
| encode/bits/64 | Estrin5 | 0.863325 |
| encode/values/4096 | Estrin5 | 0.854393 |
| encode/bits/4096 | Estrin5 V3 | 0.856410 |

One fixed baseline Estrin5 candidate also wins all8 groups: decode0.845205–
0.864196x std time, encode0.854393–0.977835x. All eight paired difference
confidence intervals lie below zero. This is a scalar result; SIMD ranking
is not inferred. Every polynomial V3 timing context was checked in linked
assembly: arithmetic inline, separate multiply/add, only conditional CORE
rescue. LLVM merged identical cold helper functions across the two families.

All three suites cover two lengths64/4096 and values/bit-uniform distributions,
with the same warmed/preallocated buffers and std/CORE/directV3/MIDP/LOWP
controls. Shared default zenbench locks and gates stayed enabled. All groups
have at least50 paired rounds, every retained pre-round gate clean, and
unreliable=false. Full per-cell means, paired difference CIs, resolution flags
and raw samples remain in timing.json (632 total timing cells across suites).

| Run | Binary SHA256 | run-heavy |
|---|---|---|
| initial-v3 | 935622b2dcc84fda918cc796801602a190bfa4cf818478efcf674866995218f3 | rc0,70s,RSS0.12GiB,min47943MiB,load1.11 |
| refine-v1 | fa4837107ed911713421239d54a3c76001d930c036a6a872e96aa4ccab1cad76 | rc0,104s,RSS0.13GiB,min47686MiB,load1.46 |
| piecewise-smoke-v1 | bf1498de1f704fd5854a579e3bfd89340fc1ab1b8d57431c6580aa5e5819dcbd | rc0,74s,RSS0.12GiB,min47936MiB,load0.97 |

Each run retains an independent executable outside Cargo, source.tar.gz,
source/compiler/build identities, generated proofs, oracle input/results,
linked.asm/asm-review.txt and timing.log/json. Earlier health PATH and generated
clippy failures remain explicitly separate. Local run root on r5900xt:
`/home/lilith/tmp/zenmath-gamma-opt-sol/runs/`; NAS mirror under
`/mnt/tower/output/archmage/zenmath-gamma-opt-2026-11-06/r5900xt/runs/`.
The initial mirror's attribute-only rsync failure was corrected using checksum
transfer without group/permission preservation; byte-manifest audit pending.

## Packed array4/8 follow-up

Frozen vector-smoke-v4 source manifest
`5b8794523d203d68a4d08e7a125bee42cfbfa67ecbe017c4c6d2743d9f3ac94b`, binary
`68fa480a4e1ceca7dec86d062b80a52aca0f115e5f17fd7327f0c4b173bebe7b`.
The first vector prototype passed numerical checks but kept piecewise block
helpers out of line. It was never timed. Same-body expansion in the proof
context removed those calls. All12 linked array4/8 contexts have packed
vmulpd/vaddpd, separate rounding, and no block/Evaluation helper calls.
Frames200–248bytes and8–16 static YMM stack references remain; scalar
certificate/exceptions retain live values across conditional CORE rescue.
The source21-vector temporary does not survive as a672-byte stack array.
Remaining calls include conditional CORE rescue, scalar-tail reference and
cold length-assertion failure. They are listed in vector-asm-review.txt.

Mixed-lane directed MPFR checks passed on six chosen kind/degree/family
combinations, with lane intervals checked independently of returned bits.
Both actual array4/8 stream functions also passed expected-bit comparisons
and non-multiple-length tails. Scalar64-combination gates remain passing.

The packed suite has8groups152timingcells, >=50rounds each, all gates clean,
unreliable=false; run-heavy rc0/62s/peakRSS0.12GiB/min47920MiB/load1.21.
Fixed array8 globalE13 decode ratios:0.530510,0.530097,0.561014,0.544314;
globalE15 encode:0.682879,0.576528,0.658145,0.649088 (values64,bits64,
values4096,bits4096). Thus scalar ranking does not transfer to packed code.

Direct all-round paired MeanCi uses zenbench0.1.11,10000resamples with
all retained pairs and no additional exclusions. Fixed scalar pieceE5/std
CIs remain negative in all8groups; the narrowest encode/values/64 difference
is-0.0826943ns/element,95%CI[-0.0996249,-0.0587046],100/100pairs retained
(built-in analysis had excluded10). All-round data are
paired-validation-v1/fixed-e5-all-round.csv and vector-smoke-v4/all-round-paired.csv.
Analysis-only binary SHA25de576c36c079cca70503d70f704313fb794189fd7bcd0090c29e25f03dc6b5.
Six negative CLI fixtures (incomplete JSON text,empty list,zero/NaN/infinite
scale,sample-width mismatch) and two caller completeness checks passed.
The caller requires the exact eight-group inventory. No changed numerical
binary or timing rerun was needed for these analysis checks.

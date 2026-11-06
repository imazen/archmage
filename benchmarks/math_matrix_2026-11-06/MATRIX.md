# Planned matrix (2026-11-06)

Work in progress; **no timings complete yet**. `api-matrix.csv` is the exact
source-expression inventory. Each vector family expands scalar/V3/V4x/NEON/
WASM128; only successfully summoned tiers run. Baseline compilation has no
`target-cpu=native`. Whole-buffer loops stay inside one feature context.

Operations: f32 general powf, uniform power 2.4, exp, ln, log2, log10, sqrt,
floor, ceil, ties-away round; separately ties-even roundeven; f64 log2.
Widths: f32 4/8/16 and f64 2/4/8 for magetypes. Magemath x8 only.

Scalar: std, zenmath correct, libm 0.2.16, fpmath 0.2.0, micromath 2.1.0,
metallic 0.3.1 where available; zenmath fused/unfused approximations.
Vector: Magemath correct and portable/unfused/relaxed approximations;
Magetypes lowp/midp/precise and native primitives. General packed approximate
power uses existing PowBehavior; never a splat/extract adapter.

Explicit unavailable APIs: Magemath correct log10 and f64 log2; Magemath
widths other than f32x8; Magetypes packed per-lane exponent pow; Magemath
and Magetypes ties-away round; metallic floor/ceil; micromath f64;
fpmath/micromath/metallic vectors; core scalar float transcendental and
sqrt/floor/ceil/round APIs on stable Rust. Missing functions are not std adapters.

Datasets: ordinary, finite edge-heavy, specials. Approximate outputs outside
published domains are findings rather than contract failures. Throughput uses
1024 identical inputs/observable outputs and paired interleaved zenbench rounds.
Feedback controls are separate, with explicitly mapped bounded recurrences.

Hosts: Core Ultra 7 265K; Ryzen 7900X V3/V4x; real Neoverse-N1; strict WASM.
No Windows execution currently available; mac unavailable per owner. No Zen5
performance claim. Algebraic reduction/polynomial/array probes are separate
from numerical bit contracts; inspect actual flags and packed lowering.

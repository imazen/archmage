# Existing math inventory and migration review

Reviewed 2026-11-05. This reconciles work already implemented in
`zenmetrics-math`; it is not a proposal to reimplement that crate. The later owner request authorizes draft Rust ports of the consumed CORE-MATH
subset; see [port coverage and validation](CORE_MATH.md). Consumer migration
remains separate.

## What exists and where it belongs

The source review used clean zenmetrics commit `097b5e4de9f8e5322b0f0abd4efb2b7055e05931`.
Zensim's `det_math.rs` was inspected at `a3de7bc48044760cab59695bd33761d5178a08a7`;
its checkout has unrelated manifest/lockfile edits, which were not changed.
Existing downstream tests and numerical claims have not been rerun for this review.

| Existing implementation | Present home | Migration assessment |
|---|---|---|
| f32/f64 sqrt, floor, ceil, round-away, round-even; f32 fused multiply-add | `zenmath::scalar` | Already here. Preserve primitive semantics independently of any host libm. f64 FMA has since been ported; validation is tracked in CORE_MATH.md. |
| f32 lowp/midp log2, exp2, ln, exp, log10, pow, cbrt; full-range midp cbrt; sigmoid and SiLU | zenmath scalar mirrors; magemath portable/relaxed f32x8 kernel, array and slice surfaces | Already here as draft. Portable rounding remains fused pending the owner's decision. These approximations do not reproduce named libms. |
| f64 lowp log2/exp2/ln/exp/log10/pow | zenmath scalar | Already here, but not full-range, high-accuracy replacements for statistics. `exp2_lowp_f64` clamps to [-1022, 1022]; do not advertise a universal relative-error bound. |
| `libm 0.2.16` f32 pow and exp lane ports | `zenmetrics-math/src/fdlibm.rs` | Reuse the eight-lane implementations in magemath's future compatibility family; supply pinned scalar fallback through zenmath. Sixteen-lane code remains in its existing consumer until width expansion is approved. |
| `libm 0.2.16` f64 pow, log and log2 lane ports | `zenmetrics-math/src/fdlibm64.rs` | Preserve existing consumer code. Scalar dependencies are migration candidates now; f64 vector extraction remains caller-driven, outside the current f32x8 scope. |
| glibc-derived f32 pow | `zenmetrics-math/src/glibc.rs` | Reuse the fast-path work, but complete its exceptional scalar branches before calling it portable. It currently calls host `x.powf(y)` for negative/subnormal/special inputs and large exponent products. |
| exp/log/pow policy kernels; scaling, AXPY, absolute differences, offset powers | `zenmetrics-math/src/{policy,kernels}.rs` | Existing reusable composition. Do not replace policy-specific tails or operation order during extraction. Introduce thin consumer adapters after primitive parity is proven. |
| LUT interpolation, Weber contrast, temporal FIR, masking and pooling | `zenmetrics-math` | Keep metric-specific compositions there. They are not missing libm primitives. Reduction order and native vector width affect results. |
| erfc, normal CDF/log-CDF, log-gamma, incomplete beta, Student-t CDF/tail/quantile | `zenmetrics-stats/src/special.rs` | Reuse in optional zenmath `special`, with source attribution and independent error/tail tests. Current routines call std exp/ln and gamma reflection calls sin: a no_std extraction needs those dependencies first. Quantile-of-sorted-samples stays with statistics. |
| erf, GELU, softplus, logistic transformations | `zenmetrics-nn/src/afine.rs` | Reuse scalar functions where useful, leaving model-specific scaling in the consumer. Their current approximation contracts are not correctly rounded libm contracts. |

`zenmath::exact` and `zenmath::special` are still empty. The existing ports have
not disappeared; they have not yet moved into these new crates. Preserve the
consumer entry points through adapters when moving implementations.

## Which numerical contracts to preserve

There are three independent promises:

1. **Correct rounding** to a specified IEEE format under round-to-nearest,
   ties-to-even, with gradual underflow. Use an independent high-precision or
   exact-rational oracle. Hardware/std is an implementation option, not the
   definition. This review reproduced Rust 1.99.1 WASI `mul_add` returning
   `0x00010002` where `0x00010001` is required; zenmath now uses software there and on native builds without compile-time
   hardware FMA, which reproduced the same failure under Nehalem emulation.
2. **Compatibility with a named algorithm/version**. Preserve its constants,
   evaluation order, fusion, exceptional paths and final rounding, even when
   another algorithm has lower error. Equality to a libm is not correct rounding.
3. **Portable bounded-error approximation**, or its explicitly relaxed variant.
   This is the existing magemath surface. Do not route compatibility callers
   through it merely because the function names match.

NaN payload preservation, errno, exception flags and dynamic rounding modes are
separate contracts. The present draft normalizes comparison of NaNs to NaN-ness
and assumes default rounding and gradual underflow; it does not emulate a C
floating-point environment. A compatibility family needs explicit per-function
zero/infinity/NaN rules, especially `pow(NaN, 0)` and `pow(1, NaN)`.

## Implementation and version choices

Recommendations below are design guidance, not additional exported modules.

| Source/version | Role and recommendation |
|---|---|
| Rust `libm 0.2.16` | First compatibility target: existing pow/exp/log/log2 ports already name it, and zensim `PureRust` plus zenfleet's deterministic logarithm are concrete callers. Pin the scalar algorithm and its fallback too; Cargo `"0.2.16"` alone permits later patches. Retain Sun/FreeBSD notices. Audit architecture-specific paths before claiming whole-crate bit portability. |
| glibc 2.40 source lineage in the existing powf port | Preserve as the historical candidate, but its current mixed glibc/Arm/musl provenance is not yet a frozen compatibility specification. Capture exact source/table hashes and the consumer's actual glibc build and selected symbol path. Do not promise “glibc >=2.28” equality. |
| glibc 2.43 oracle environment | The existing `zenmetrics/benchmarks/clean_panel_2026-10-19/ORACLE_NATIVE_CONTRACT.md` records 2.43. Use that retained consumer environment as the first compatibility comparison, rather than assuming the port's 2.40 source label identifies the oracle. This review did not rerun that oracle or verify its library binary. |
| glibc 2.44 | Current upstream release at review time; use as a comparison target, not an automatic replacement for the historical contract. Keep a second implementation only where a consumer needs differing bits. Pin architecture, IFUNC/FMA path and compilation choices as well as version. |
| Arm optimized-routines v26.07 (`4be260a`) | Candidate source for modern exp/log/pow algorithms and native SIMD work, after the existing ports are reused. It includes correctly rounded AdvSIMD expf. Evaluate the actual function, not a blanket accuracy claim for the library. |
| CORE-MATH, per-function immutable commit | Preferred research source for new correctly rounded functions; no moving-master contract. Start with required f32 functions, then required f64 functions. Bring hard cases and proofs/tests; specials need our explicit wrapper semantics. The first eight-f32 compiler study pins `660067dfd3e751c8f07acc31c899ff410aa7206e`; this is a research baseline, not a promoted implementation. See [the measured study](../../benchmarks/core_math_cost_2026-11-05/README.md). |
| LLVM libc, per-function release/commit | Alternative correctly rounded implementation and independent comparison. Check its function-specific accuracy table and build configuration; do not assume every routine or configuration is correctly rounded. |
| musl 1.2.6 | Useful independent comparison and source lineage. Do not add a separate public compatibility family without a consumer requiring its results. Rust libm is derived from musl but is not synonymous with this release. |
| SLEEF 3.9.0 | Candidate for future bounded-error SIMD/range-reduction work. No portable-bit or correct-rounding promise follows from using SLEEF; inspect the chosen variant and fusion policy. |
| OpenLibm 0.8.8 / standalone fdlibm or FreeBSD msun | Comparison sources only for now. The existing Rust-libm compatibility target already reuses that lineage; adding another near-duplicate family needs an observed compatibility requirement. |

Upstream verification: [Rust libm releases](https://github.com/rust-lang/compiler-builtins/releases),
[glibc releases](https://sourceware.org/glibc/),
[glibc accuracy contract](https://sourceware.org/glibc/manual/latest/html_node/Errors-in-Math-Functions.html),
[Arm v26.07](https://github.com/ARM-software/optimized-routines/releases/tag/v26.07),
[CORE-MATH](https://core-math.gitlabpages.inria.fr/),
[LLVM libc math](https://libc.llvm.org/math/index.html),
[musl release history](https://musl.libc.org/releases.html),
[SLEEF releases](https://github.com/shibatch/sleef/releases),
[OpenLibm releases](https://github.com/JuliaMath/openlibm/releases).
Versions are observations on this review date, not update policies. The glibc
website lists 2.44, but the bminor GitHub mirror returned 404 for that tag during
this review; no 2.44 source comparison or compatibility claim has been made. Per-file
license/provenance follows each reused function; ownership of the surrounding
consumer does not replace third-party notices.

## Recommended order

1. Preserve and extract the existing Rust-libm-compatible scalar pow/exp/log/log2
   dependencies and f32x8 ports, retaining exceptional fallbacks and coverage gates.
   Verify against the pinned upstream implementation and existing consumer goldens.
2. Complete the existing glibc powf port only after identifying the consumer's
   required version/path. Compare historical and current glibc before multiplying
   public compatibility families.
3. Extract existing scalar statistics/activation functions, after supplying their
   f64 exp/log/sin dependencies. Prioritize `log1p`, `expm1`, `hypot`, and stable
   tail/complement forms where cancellation or overflow makes composed formulas
   unsuitable. These are missing capabilities, not permission to change existing
   model results during extraction.
4. Add sin/cos/sincos, sinpi/cospi, atan2, tanh, and correctly rounded exp/log/pow
   as concrete callers require. Full-range trig needs range reduction and hard-case
   testing; it is not just another polynomial. Consider trunc/scalbn/frexp/nextafter
   as internal support first. Leave gamma-family extensions, Bessel, complex,
   long-double, and f16/bf16 arithmetic outside the initial portable surface.

No new scalar or vector implementations were added for this inventory review.
The source inventory is established work; the remaining job is contract-preserving
extraction, completion of the host-dependent glibc fallback, and targeted gaps.

Source check: glibc 2.40 has both `e_powf.c` and `e_powf-fma.c` in
`sysdeps/x86_64/fpu/multiarch`. Its
[generic powf file](https://github.com/bminor/glibc/blob/glibc-2.40/sysdeps/ieee754/flt-32/e_powf.c)
carries LGPL-2.1-or-later, so the existing mixed-source port needs file-level
provenance review before extraction into MIT/Apache crates. The Arm algorithm
lineage is a candidate source, not evidence that every glibc line has its license.

## Default versus compatibility: measured follow-up

The owner proposed correctly rounded core functions as the default, with named
`compat` choices for callers reproducing a libc. The [CORE-MATH compiler study](../../benchmarks/core_math_cost_2026-11-05/README.md)
supports that separation: all eight tested CORE-MATH f32 functions matched
MPFR on 200,000 inputs each under eight compiler/ISA/contraction configurations,
while the host glibc differed on six functions. This is sampled validation, not
exhaustive certification or permission to migrate existing consumers.

Baseline x86-64 remains the proposed deployment floor. ISA-specific FMA paths
can preserve the same correctly rounded final bits; they need not expose another
numerical contract. Hardware selection matters most for the measured powf:
GCC baseline took 7.441 ns/value with the hardware-dispatched libc FMA helper,
25.989 with its software helper, and contracted V3 took 5.366 in the normal run.
The separate software-helper run is not a paired CPU comparison.

The standalone safe-Rust exp2 translation also exposes a baseline LLVM conversion
dependency: exact bit widening improved 2.684 to 1.702 ns/value, but lost on V3
(1.715 versus the ordinary cast's 1.466). Keep optimization choices per target
and function, with identical result gates. No new exports or defaults were added.

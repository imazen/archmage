# CORE-MATH follow-up: frozen before timing

This is an additional comparison prompted by the glibc source-history audit.
It neither replaces nor reruns the prior provider matrices. Keep every earlier
run and its original flags, including the CPU-targeted Neoverse pilot.

Source: the `cr_powf` C file shipped in core-math-sys 1.4.0; retain its MIT
license, source hash and crate provenance. Build only this function. Baseline
uses explicit x86-64 or Armv8-A, never `native`. On x86 a second DSO enables
AVX+FMA and explicitly disables AVX2. Both retain explicit fused operations;
baseline may call system fma while the feature kernel emits FMA directly.
Use -frounding-math, -fno-fast-math, -ffp-contract=off and standard excess
precision. CORE_MATH_SUPPORT_ERRNO remains undefined and is recorded.

Reuse the prior 32 batch axes and 1,048,576 actual evaluations/request, with
opaque exponents and full traces. Compare only Rust std, adapter-matched system
C powf, CORE-MATH baseline and, on x86, CORE-MATH FMA-without-AVX2. Record EMPTY
transport arms. Fifty interleaved retained rounds, unchanged strict common
gates and all-round paired 95% confidence intervals. Primary ratios are each
CORE-MATH implementation/system C powf; feature/baseline and other pairs are
descriptive. M6 gets one exploratory full matrix; Linux gets one proper pilot
and a full matrix only after a clean pilot. No favorable retry.

Independently check MPFR-256 on the unchanged 27,909 pairs, broadcast/mixed
widths 1/4/8/16 and every ordinary stream/chain trace. Preserve all bits and
classes, including NaN payloads. Test the FMA-only DSO with AVX2 disabled under
the existing no-AVX2 emulation control; emulation supplies no speed evidence.
Finite samples verify those executions, not a new proof of whole-domain
correct rounding. RNE and gradual underflow are required; no production
provider or threshold changes. The additive native worker also rejects a
truncated evaluator request before reading its values.

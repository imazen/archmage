# Frozen study plan, 2026-11-07

Baseline: `6bfd83d744e41259ec758054631dcb0928fc4ea1`. The baseline
extension traits are mechanically extracted from that commit's generated
source; they use the same backend operations and preserve every arithmetic
expression. SHA256 manifests retain the extraction and executable identities.

Three arms: frozen main, the exceptional-only live candidate, and an
experimental fixed-fused graph extracted from the candidate with only
`mul_add_relaxed` replaced by `mul_add_fused`. The third arm is never a
change to the shipped default. It includes the exceptional correction so
its finite arithmetic cost can be compared with the corrected second arm.

Mandatory native cells:

- f32 widths 4, 8, 16; scalar baseline caller, scalar inside a V3 caller,
  V3 vectors in a baseline caller, and V3 vectors inside a matching caller.
  NEON uses scalar and matching NEON contexts.
- 64 and 4096 elements; independent output streams and dependent feedback.
  Feedback is `x = abs(op(x))*0.125+0.5`, repeated once per input vector.
  It measures amortized dependent work, including normalization, and can
  converge. It does not represent instruction latency or exceptional density
  after the first step.
- Ordinary positive values (alternating signs for exp/activation),
  bit-distributed valid values, sparse and dense NaN lanes. Log/pow ordinary
  inputs are positive normal. Exceptional stream cells preserve the declared
  density throughout the pass.
- Canonical changed operations: exp2/exp/pow at lowp and checked midp,
  sigmoid and SiLU. Fixed-fused additionally includes log2/ln/log10 at both
  precisions. Aliases are correctness-checked against their canonical bodies
  rather than multiplying timing cells for identical implementations.

Every cell retains 50 shuffled paired rounds, all admission flags, with
zenbench's default shared lock and strict resource gate. The report applies
`MeanCi` to every retained per-round candidate/baseline ratio, without
outlier filtering. Required: unreliable=false, every round gate_clean=true,
upper paired 95% endpoint <=1.02. An interval above 1.02 fails promotion;
one straddling 1.02 is inconclusive. A single declared extension to 200
pairs may preserve the first 50; no favorable-run selection. This study
starts with 50 and does not silently extend or replace populations.

Busy or incomplete results are diagnostic evidence, preserved with their
reason, and cannot establish a pass or regression. All mandatory cells are
reported individually; missing native Intel/NEON/WASM coverage prevents an
unqualified promotion recommendation. Preliminary historical context is
[PR128](https://github.com/imazen/archmage/pull/128), which used an earlier
FMA implementation and is not a current performance measurement.

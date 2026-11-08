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


## Supplemental controls, frozen before their first timing

The original fourteen-operation runs stay intact. `combined_patch` adds the
left-associated sum of the eight changed canonical operations on the same input,
with the same widths, buffers, contexts and distributions. It is a workload
comparison, not an additional accuracy claim. The independent graph dump keeps
the original fourteen operations; the sum introduces no different polynomial.

FMA controls compare the existing public fused primitive, mechanically copied
array software arithmetic, and a hardware projection dispatched once per buffer.
Native widths are f32 1/4/8/16 and f64 1/2/4/8. They use the same context,
size, shape and distribution axes. A forced no-V3 run measures fallback overhead;
it prints the actual detector state. These controls do not alter library dispatch.
WASM representatives are f32 1/4 and f64 1/2; scalar hardware uses a packed
broadcast because WASM has no scalar FMA instruction. All share prepared inputs.

WASM uses a host-verified f32/f64 fusion probe. The strict guest lacks relaxed
instructions. The relaxed nonfusing x86 engine clears ISA autodetection, which
also disables other optional ISA features: cross-engine differences cannot be
attributed solely to FMA. Deterministic relaxed selection is a separate control.
Each steady-state timed call processes 128 complete buffers. A separate fresh
guest-instance group times the first complete 64-element exp2_midp buffer;
instance construction/preparation/destruction remain outside the timer.
Native fresh-process perf-stat diagnostics include process startup and cannot
be described as isolated primitive latency.


Steady-cache audit correction, before the remaining WASM timings: zenbench
0.1.11 invokes the bench closure anew for each round. Each guest instance must
therefore process one untimed buffer before `b.iter`. The original strict
exp2_lowp attempt, which amortized an unprimed instance over the timed calls,
is retained separately as diagnostic; it cannot establish the cached gate.
The fresh-instance first-use group deliberately keeps its cache unprimed.


First-use extension preregistration: the unchanged fresh-instance unit hit the
60s wall safety cap at 28 strict / 25 fusing pairs. All those pairs passed every
resource gate. They are insufficient for the 50-pair minimum and are retained.
Use the one allowed extension to 200 total pairs per mode: append172 strict /
175 fusing pairs, wall safety budget600s. Keep guest bytes, instance preparation,
timed work, sample-duration target and every inference/admission bound unchanged.
Pool ALL original and appended paired ratios; keep each source/binary identity.
No further extension or favorable-run selection. Steady-state cells proceed
independently; an insufficient first-use count does not imply host contention.

The combined ordinary workload uses positive values so that all eight
operations, including pow, receive valid inputs. Individual exp/activation
cells separately exercise alternating signs.

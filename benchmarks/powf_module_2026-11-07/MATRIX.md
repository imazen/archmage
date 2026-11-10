# Bounded powf admission/dispatch study, frozen before timing

The library and its independent rational corpus establish the result contract.
Performance controls mechanically extract the actual powf graph, removing only
the input and output rejection conditions. Inputs are validated and every arm's
result bits compared before timing. No std powf supplies baseline arithmetic.
The scalar software control copies the exact existing array FMA implementation.

Native mandatory pilot: scalar entry width 1, plus widths 4/8/16 in scalar and
matching native-feature callers; 64/4096 logical elements; throughput streams
and normalized dependent chains; PositiveNormal, NonNegativeNormal and Gamma
profiles; ordinary inputs and bit-distributed valid inputs in [0.5,2]. This is
168 cells on NEON. The uniform exponent is the exact f32 value 2.4, matching a
gamma consumer. PQ and other exponents are in the independent correctness and
MPFR corpora; this pilot does not measure their performance.

Arms: prevalidated graph, checked Caller, checked Automatic. The native loop
with a matching token keeps the full operation in the feature region. The
scalar/Automatic arm can upgrade the whole pow graph. Scalar Caller forces the
software graph. All caches are warmed explicitly outside the timer. Width1
measures the actual public scalar function against its mechanically extracted
width 4 broadcast core, retaining only the first lane in both arms.

Retain 50 interleaved paired rounds per cell; use zenbench's common lock and
strict resource gates. All retained gates must be clean, unreliable=false.
Analyze every ratio using MeanCi with 100,000 bootstrap samples. Upper 95% CI
<=1.02 is PASS, lower>1.02 FAIL, crossing 1.02 INCONCLUSIVE. This pilot makes no
global promotion claim; failing cells and unavailable hosts remain explicit.
No repeated favorable-run selection or deletion of dirty rounds.

Dependent chains apply `abs(result)*0.125+0.5`, as part of equal work in every
arm. They can converge and do not measure isolated instruction latency.
The selected valid range stays in all three domains throughout the chain.

Separate required evidence: admitted AMD and Intel, forced below-V3 callers,
no-FMA overhead, strict/relaxed-fusing/relaxed-nonfusing WASM, cold dispatch and
the std correctness screen. Emulation supplies correctness only. Missing
timings block an unqualified optimization or no-regression claim.

Ratios measure total checked API cost, including validation, dispatch, and
its return/call boundary; they do not isolate the mask branch alone. One strict
admission pilot per host runs before the full matrix. A dirty pilot blocks
that host; no repeated attempts until favorable.

Separate rejection matrix: widths 4/8/16, scalar and native callers, 64/4096
elements, sparse (one bad lane per 16 vectors) and dense (every lane bad),
signaling NaN and nonzero subnormal. These 48 Gamma stream cells compare
checked Automatic against checked Caller; they do not compare rejection with
a graph that would perform invalid arithmetic. Expected masks are checked
before timing.

WASM: the same 168 ordinary cells per engine control, plus first use of the
complete x4/64 Gamma buffer in a fresh instance. The control mechanically
extracts the same graph but uses the existing fused primitive at each FMA;
this retains its per-node cached probe, versus the new whole-operation route.
All lane bits are checked before timing. Steady samples invoke 128 complete
buffers per host call; first use invokes one. Instance creation, preparation,
and destruction stay outside first-use timing. Strict, relaxed fusing and
relaxed nonfusing projections are asserted at runtime. The nonfusing host
control selects Wasmtime's whole baseline x86-64 ISA, disabling other optional
host features too; timing differences cannot be attributed solely to FMA.

# Power replacement comparisons — frozen before timing, 2026-11-08

The target is replacing existing power methods within explicitly reported
input/accuracy domains. Performance controls are actual `pow_midp`,
`pow_midp_unchecked`, `pow_midp_precise`, `pow_lowp`, `pow_lowp_unchecked`,
per-lane `f32::powf`, and per-lane promoted `f64::powf` rounded to f32.
Integer-only cells also include `f32::powi`. No approximation supplies
mathematical truth: accuracy uses directed MPFR reference enclosures.

The new candidate uses a const builder ending in `.bits()`, carried as a u32
const generic from the consumer loop through the public vector/scalar method,
backend and graph. Compare Caller and Automatic. Diagnostic competitors retain
the runtime adapter with either literal or opaque runtime options, both Caller
and Automatic, plus the same prevalidated fused graph. Thus the primary ratios
are replacement versus actual legacy/std implementations; the prevalidated
ratio only isolates total checked-API overhead and arithmetic/dispatch effects.

Native matrix: scalar x1 and x4/x8/x16, matching V3 or NEON x4/x8/x16;
64/4096 logical elements; streams and normalized dependency chains.
For x1 the legacy methods use a scalar-token x4 broadcast and extract lane 0,
since those methods have no scalar-f32/x1 API. Std and the new scalar API are
called as scalar functions. No four-call std surrogate is used.

672 ordinary cells per native host:

- 504: three domains × literal f32 exponents 2.4, reciprocal-2.4 and 0.5 ×
  unorm8 and bit-distributed [2^-16,2^16] inputs × two lengths × stream/chain ×
  seven width/caller pairs.
- 56: two PQ exponents (0.1593017578125 and 78.84375), PositiveNormal,
  [0.5,2] ordinary inputs × lengths × stream/chain × width/caller pairs.
- 56: Gamma / 2.4 / the first two distributions, with an opaque runtime
  exponent instead of a literal in the consumer × remaining axes.
- 56: Gamma / literal exponents 2 and 4 / nonzero unorm8 inputs, including
  std powi × lengths × stream/chain × width/caller pairs.

Every ordinary input lies in the common accepted domain. Nonzero unorm8 means
1/255 through 1, rounded to f32. Wide inputs are sampled deterministically by
f32 bit pattern. Chains normalize each result by retaining its fraction and
setting its binade to [0.5,1). Approximate implementations can therefore follow
different chains; operation counts match, but these are not identical numeric
trajectories or isolated instruction latencies. Separate accuracy measurements
use identical input pairs for every implementation.

Before timing, all selected outputs must be finite. Both legacy alias pairs
must match bits. Every runtime/const candidate must match the extracted exact
fused graph at every output, including chains. No legacy/std bit equality with
that graph is assumed. Rejection and special-value semantics remain a separate
comparison: these methods have different contracts.

Retain 50 interleaved paired rounds, no filtering or selected reruns. Require
all zenbench resource flags clean, `unreliable=false`, and its common admission
lock. Report candidate/control ratios with all-round paired MeanCi bootstrap,
100,000 resamples. For each replacement target, upper 95% CI <=1.02 passes,
lower >1.02 fails, otherwise inconclusive. Accuracy/domain compatibility is a
separate requirement. Dirty or unavailable hosts supply no speed claim.
One predeclared pilot per host runs before the complete matrix; preserve it.
A failed admission or dirty cell stops that source/host run. A later source or
verified external-state change is recorded rather than silently replacing it.

No `target-cpu=native`. Archive source/compiler/flags, all rounds and failures,
reference executables outside Cargo output, instruction probes and resource
summaries. Raw results live in `powf-replacements-2026-11-08`, separate from the
original prevalidated-control study and its immutable evidence.

## Recorded orchestration amendments

After the original Neoverse-N1 run stopped at cell 155, the unmeasured suffix
may run once in a new directory with the identical executable and cell list.
The invalid cell remains invalid, with every sample retained; it is never
retried or removed from the mandatory denominator. The continuation records
its parent manifest and original cell indices. Promotion requirements stay fixed.

The owner authorized a separate exploratory Apple M6 comparison after strict
placement refused before timing. This uses explicit unrestricted CPU placement,
50 interleaved rounds and the common admission/resource gates. The OS schedules
its asymmetric core classes. Every result is labeled exploratory and cannot
satisfy the strict promotion gate, even when resource flags are clean. Dirty
cells remain reported and the exploratory run proceeds through unmeasured cells
without retrying them. The original strict refusal is retained separately.

The N1 suffix also stopped at cell 302. To complete descriptive coverage, the
remaining suffix may record each unmeasured cell once without stopping on a
resource-dirty verdict (`--record-all`). No completed cell is rerun, no dirty
round is removed, and no admission or promotion condition is relaxed. Both
invalid cells (155 and 302) remain in the mandatory denominator. Functional
verification failures and admission errors still stop execution.

The AMD worker was quiet on reinspection after the original run stopped at
cell 0. Its previously unmeasured suffix may run once with the same executable,
cell list and 50 retained rounds, using `--continue-unmeasured --record-all`.
The original invalid cell 0 stays in the denominator and is never retried.
Dirty verdicts remain invalid; admission/functional failures still stop.

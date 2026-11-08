# Backend-dependent power rounding study, 2026-11-08

The new experiment compares LOWP and MIDP with forced single-rounding FMA
against separate RNE multiply/add, on the same scalar/native backend. Kernels
are extracted from the owning production expressions. The bit-exact library
default, coefficient bits, existing fixtures and promotion gates stay fixed.

## Numerical requirements

- Every actual result/rejection must match its independent rational expression
  model. Scalar/native versions of each forced rounding mode must agree.
- On common positive accepted inputs, separate evaluation must reproduce the
  existing scalar legacy method; fused evaluation must reproduce the existing
  FMA-native method. Zero rails and out-of-range arguments are recorded
  separately because the legacy methods have different handling.
- Mathematical accuracy uses directed MPFR-256, with direct f32 endpoint
  conversion. Keep every rejection and special/reference overflow classification.
  Sampled maxima never become global error guarantees.
- Reuse the preserved 26,857 domain-tagged pairs, add deterministic integer and
  half-integer range-reduction boundary witnesses, and inspect both normal-only
  profiles and Gamma. Adaptive search is bounded and Gamma-specific for all
  four methods, both actual backend contexts and three search heuristics.
- Gamma is +0 or positive normal x in [2^-16,2^16], with normal-or-zero n in
  [0,4]. Independent monotone RNE interval certificates must bound the computed
  argument below 68 for both coefficient families and rounding modes.
- Test whole-vector validation/masks at widths 4/8/16, explicit +0 semantics,
  invalid uniform exponents and checked/assumed equality on valid inputs.
- Inspect selected scalar/V3/NEON/WASM probe bodies. Separate multiplication
  and addition must remain separate; no fast-math or inner-loop barriers.

## Timing cells (freeze before timing)

96 cells: widths 4/8/16 × scalar/matching-native caller × 64/4096 values ×
stream/normalized dependency chain × literal exponents 2.4/0.5 × checked/assumed
Gamma validation. Each cell contains all four extracted LOWP/MIDP × fused/split
implementations plus actual legacy lowp, legacy midp and per-lane std f32 powf.
Dependency chains normalize fractions into [0.5,1) after each result, preserving
equal operation counts; trajectories can differ. Accuracy compares identical
input pairs separately. Include one preregistered x4/native/64/stream/2.4/checked
pilot per host. A failed or dirty pilot blocks that host's full matrix.

Use zenbench common admission, all 50 retained interleaved paired rounds,
100,000-resample paired confidence analysis and the unchanged 2% pass bound.
Dirty cells remain invalid; no retries or selected rounds. M6 is explicitly
exploratory under unrestricted OS scheduling, preserving all resource flags.
Report actual function/family/rounding/range/validation/context and ns per buffer
with zenbench bars; identify any ns/value normalization as amortized throughput.

## Preset requirements

Every proposed preset states portability, accuracy family, base interval,
exponent interval, output behavior, validation and dispatch. Unchecked means a
documented numerical precondition; it must not introduce Rust memory unsoundness.
Do not infer a general LOWP error budget from one fixed-exponent band or admit
std as a portable/one-ULP backend from finite screening alone. New combinations
remain experimental until their requirements pass; unsupported public const
options retain their compile errors.

Artifacts live in `powf-nonfused-expectations-2026-11-08`, independently of the
earlier immutable replacement study. Preserve source, references, raw results,
commands and failures, and verify the NAS mirror.

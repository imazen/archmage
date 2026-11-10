# Accuracy evidence

The pre-gamma optimization stage emitted 403 cells × 4,096 inputs on native
and NEON. Its independent MPFR analysis uses directed 256-bit endpoints for
correctly rounded f32 power: both endpoints must round to identical f32 bits,
then the implementation must match. This is separate from the sampled error
estimates below. MPFR cache keys contain raw input bits and type, retaining
signed zero. NaN payload differences alone are outside the contract.

For native gamma power (decode/encode × uniform/log/boundary, 24,576 samples
per row), observed maximum errors were:

| Candidate | Maximum observed ULP estimate | Rounded-bit differences |
|---|---:|---:|
| `current/correct` | 0.499949 | 0 |
| `current/midp-full-fused` | 8.25987 | 10,711 |
| `current/lowp-full-fused` | 93350.6 | 23,617 |
| `current/midp-full-separate` | 9.74347 | 11,716 |
| `current/lowp-full-separate` | 93350.6 | 23,620 |
| `std/control` | 0.500787 | 6 |

These are observed estimates, not outward-rounded upper bounds. The ULP
scale is the spacing at the correctly rounded result's magnitude; zero and
subnormal magnitudes use 2^-149. Representable bit-step distance is recorded
separately. Approximation-function bounds do not automatically become bounds
for composed power. No global error bound is inferred from this sample.

Whole-transfer controls use MPFR evaluation of the distinct f64-constant C0
curve, including its actual non-textbook constants from linear-srgb. This is
not the exact-f32-exponent power oracle and not a claim that stage-rounded f32
composition equals a once-rounded ideal curve. Across 49,152 transfer samples
per candidate (including both composition modes), native observations were:

| Candidate | 8-bit quantization differences | 16-bit differences |
|---|---:|---:|
| `current/correct` | 0 | 49 |
| `current/midp-full-fused` | 0 | 56 |
| `current/lowp-full-fused` | 5,504 | 17,247 |
| `current/midp-full-separate` | 0 | 54 |
| `current/lowp-full-separate` | 5,504 | 17,247 |
| `std/control` | 0 | 49 |

Zeros in the raw CSV's quantization columns for pow-only rows mean not
applicable; only whole-transfer rows perform quantization comparison.

The final gamma specialization has a stronger, separately scoped gate:
all 29,945,741 admitted decode values and 70,823,872 admitted encode values
are directly compared by bits with independent original CORE on native,
NEON, strict SIMD128 WASM and a no-SIMD WASM module. The selected final source
passes 22 cells / 1,108,465,743 comparisons, with matching raw output and
fixed-chunk Zenhash XXH3 digests. Signed zeros and NaNs do not occur in those
positive admitted ranges; outside-range rails retain exact reference checks.
This validates the two exact exponent families, not all 2^64 binary inputs.

The certificate's analytic remainder and propagated rounding proof is
independent of the original CORE comparison; see
[Zenmath's gamma design](../../zenmath/docs/GAMMA.md). The scalar reference
remains unchanged. Source/binary identities and both earlier/final stages
remain separate in [ARTIFACTS.md](ARTIFACTS.md).

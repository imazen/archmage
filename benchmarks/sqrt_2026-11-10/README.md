# Scalar sqrt performance and portability

Scope: improve the scalar f32/f64 sqrt route while preserving the existing
correctly rounded result contract. Native SIMD sqrt implementations and all
published signatures stay unchanged. No approximate sqrt profile is introduced.

The pre-change software controls are the unchanged public
`magetypes::nostd_math::{sqrtf,sqrt}` at main `bd83f034`. ScalarToken vectors,
single-lane wrappers and their exact rsqrt compositions previously used them
unconditionally, even when the compilation target had hardware sqrt.

Acceptance checks:

- Independent exact-rational squared-midpoint fixtures, hardware/std and the
  retained software controls agree on non-NaN output bits. Signed zeros remain
  bit exact; NaN classification is checked without promising payload/sign bits.
- Check scalar and native vector widths with std/no_std; include subnormals,
  exponent boundaries, large values, infinities and negative inputs. Keep the
  f32 streaming sweep constant-memory; stride one is explicit opt-in.
- Verify no_std/MSRV builds and inspect baseline and native assembly. Retain
  software fallback where the build cannot safely select hardware.
- Use zenbench interleaved 50-round comparisons, requiring every retained round
  to pass admission. Compare named software, scalar x1, scalar packed and std
  paths on identical data; report ns per 64 values or dependent steps. No
  target-cpu=native flags. Preserve raw results and resource-wrapper records.

The numerical contract assumes round-to-nearest and gradual underflow. Changing
FP control registers is outside it. NaN payloads/signs and floating exception
flags are not cross-backend contracts. Correctness observations do not establish
performance, and sampled f64 tests do not constitute an exhaustive proof.

Results and supported-target coverage will be recorded here after validation.

Admission setup correction: initial zenbench 0.1.10 runs counted benchmark CPU
as foreign activity. They remain retained and unqualified. The measurement
harness now uses published zenbench 0.1.11 (source `983eafaa49ce`) with owned
launcher registration and trace evidence. All thresholds, 50 retained rounds
and the all-rounds-clean requirement stay unchanged. Both numerical paths are
interleaved under the corrected harness; production math is unchanged.

# NEON numeric min/max and legacy cascades

[Issue145](https://github.com/imazen/archmage/issues/145) changes float min/max
and every float reduction stage to ARM numeric instructions. The owning
generator and generated backends change together. All six float widths pass
directed quiet/signaling-NaN, signed-zero, infinity and subnormal tests on M6,
in std and no_std builds. The old implementation fails all six width tests.

A quiet NaN yields the number in either operand order. All-quiet-NaN reductions
yield NaN. Opposite zero ties yield -0 for min and +0 for max. Signaling NaNs
propagate through the primitive under the normal floating-point environment.
Rust's scalar min/max instead ignore both quiet and signaling NaNs; native
NEON numeric instructions do not implement that entire scalar contract.

## Measured legacy consequences

The independent before source is main e93e6898, including PR114. The after
library source is ed20e2b4, whose production code is unchanged by the later
private test-feature rename. `legacy_edge_variance` runs 12 directed `(base,n)`
cases over seven operations, with optimizer barriers. Of 84 NEON observations,
20 move to scalar's exact bits and none move away from scalar's bits. This is
a bounded witness set, not exhaustive binary power coverage.

| Input / operation | Old NEON | New NEON | Scalar on the same M6 |
|---|---|---|---|
| qNaN / `exp2_midp`, `exp_midp`, `exp2_lowp`, `exp_lowp` | NaN | 2^-126 | 2^-126 |
| qNaN / `sigmoid_midp` | NaN | 1 | 1 |
| `pow_midp(0,0)`, `pow_midp(-1,0)`, `pow_midp(-1,2.4)`, `pow_midp(+inf,0)` | NaN | 2^-126 | 2^-126 |
| sNaN / `exp_midp`, `exp_lowp` | NaN | 2^-126 | 2^-126 |
| sNaN / `exp2_midp` | NaN | bits `0x7f7fff7a` | 2^-126 |
| sNaN / `exp2_lowp` | NaN | bits `0x7f7cdda8` | 2^-126 |

Quiet NaN uses bits 0x7fc00123 and signaling NaN 0x7f800001. In a two-clamp
sequence, the first numeric operation quiets signaling NaN; the second can then
select its numeric bound. Multiplication by LOG2_E quiets it earlier in exp.
This explains the remaining signaling-NaN exp2 differences. Strict WASM still
propagates NaNs through its min/max. x86 remains operand-order-dependent.
These changes improve the named native/scalar agreement but do not establish
global ISA equality or correct mathematical exceptional handling.

The bounded checked powf API rejects these unsupported inputs. AssumeValid
requires the declared domain. Canonical Full HIGHP has its separate tested
special-value graph. No floating-point environment is changed.

## Code generation and resources

All 18 native legacy-power probes differ from frozen main solely by numeric
min/max opcode substitutions; instruction counts and all other operands and
control flow match. 14/18 scalar ARM probes are identical; four retain the
previous scheduling differences, with the same or fewer instructions. The
comparison preserves operands, constants, branches, calls and ordering. No
global assembly gate or expected numerical threshold is loosened.

M6 full 64-policy std/no_std replay plus Clippy: 67 s, peak RSS 0.78 GiB, minimum
reclaimable-page proxy 7005 MiB, peak load 3.36. Numeric-min/max release tests:
1 s, peak RSS 0.12 GiB. Assembly build: 18 s, peak RSS 0.25 GiB. Edge witness run:
11 s, peak RSS 0.01 GiB. macOS reports the wrapper's monitored process-tree RSS,
without a cgroup hard cap. These timings are validation/build costs, not
primitive runtime speed measurements.

The reporter lives in [the validation package](../powf_validation/README.md).
Reproduce observations with `just powf-legacy-edge-variance`; use the same
example source against each frozen library.
[Artifact identities](MIDLOW-ARTIFACTS.md) preserve raw bits, source/compiler
identities, assembly reports and failure logs.

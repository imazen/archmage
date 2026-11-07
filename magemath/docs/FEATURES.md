# Math families

The default enables `correct-rounding`, `normal`, and `configurable`, together
with Magemath's existing `std` default. Existing default APIs keep their result
bits. Cargo features add availability; enabling another family never selects a
different precision, range rule or multiply-add sequence for an existing call.

| Family | Zenmath API | Magemath API | Implementation today |
|---|---|---|---|
| Always | `scalar` primitives | Primitive kernels, arrays and slices | Existing sqrt, rounding and fused arithmetic contracts |
| `correct-rounding` | Existing precise `scalar` names; `simd` precise entries with `archmage` | Existing unsuffixed `portable::array` and `portable::slice` exp/ln/log2/pow | Pinned precise CORE-MATH ports |
| `normal` | `normal` scalar names; `normal::simd` with `archmage` | `normal::{array,slice}` | Shares the precise implementation; no cheaper libm approximation is claimed |
| `configurable` | Named lowp/midp functions, `unfused`, `PowBehavior`, `powf_with` | Named lowp/midp kernels, arrays/slices and policy power | Existing declared ranges, rounding and exceptional-value contracts |

The draft migration makes `default-features = false` primitives-only. Existing
consumers that disabled defaults must now select their math family explicitly.
For example, a power-policy consumer selects `features = ["configurable"]`;
add `correct-rounding` if it also calls the unsuffixed precise API.

`std`, `archmage`, `runtime-dispatch`, and `avx512` remain independent backend
choices. They do not enable a math family. Zenmath only probes CPUs under its
own `runtime-dispatch` feature, including when another dependency enables
Archmage's `std` feature. Magemath retains its existing `std` dispatch model.
Per-call power precision/range/consistency stays in `PowBehavior`.

Configurable Full power still requires an accurate private power implementation
for rescue and correctly rounded policies. It can compile that core even with
the public `correct-rounding` API disabled. Normal currently shares the entire
precise implementation. These are implementation dependencies, not quality
changes selected by Cargo feature unification.

Zenmath does not depend on Magetypes. Magetypes does not need a Zenmath edge for
deprecation. Magemath connects both crates and forwards the requested family
features to Zenmath with dependency defaults disabled. Their dependency graph
permits independent compilation after shared prerequisites; actual overlap and new family
cold-build costs remain pending measurement in the power optimization study.
The dependency graph alone is not evidence of a compile-time improvement.

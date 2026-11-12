# Complete option permutation matrix

Generated from the actual Rust builders and validity predicates, in std and no_std builds.
Each cell fixes the row's three axes plus the column's validation/execution choices.
This covers **72 raw combinations**, **72 supported with std**,
and **24 distinct supported flag sets**. A flag ID identifies a configuration, not a timing result.

- **Supported ID**: legal configuration; see its canonical row below.
- **→ ID**: `.bits()` returns exactly the same flags as that canonical row. This is a configuration alias, not just similar accuracy.
- **Unsupported**: rejected option combination; const API compilation fails. Transcendental runtime adapters return UnsupportedOptions.
- **std only**: that combination is unsupported without std.
- Execution Caller/Automatic remains a distinct flag when not canonicalized. Typed entries retain their supplied token; array dispatch is a separate timing dimension.

<details><summary>Full: all precision/result-policy × validation × execution choices</summary>

| NaN | Zero | Checked / Auto | Checked / Caller | AssumeValid / Auto | AssumeValid / Caller |
|---|---|---|---|---|---|
| Number | Ordered | Supported [M00](#m00) | Supported [M40](#m40) | → [M00](#m00) | → [M40](#m40) |
| Number | Backend | Supported [M10](#m10) | Supported [M50](#m50) | → [M10](#m10) | → [M50](#m50) |
| Propagate | Ordered | Supported [M04](#m04) | Supported [M44](#m44) | → [M04](#m04) | → [M44](#m44) |
| Propagate | Backend | Supported [M14](#m14) | Supported [M54](#m54) | → [M14](#m14) | → [M54](#m54) |
| Backend | Ordered | Supported [M08](#m08) | Supported [M48](#m48) | → [M08](#m08) | → [M48](#m48) |
| Backend | Backend | Supported [M18](#m18) | Supported [M58](#m58) | → [M18](#m18) | → [M58](#m58) |

</details>

<details><summary>Finite: all precision/result-policy × validation × execution choices</summary>

| NaN | Zero | Checked / Auto | Checked / Caller | AssumeValid / Auto | AssumeValid / Caller |
|---|---|---|---|---|---|
| Number | Ordered | Supported [M01](#m01) | Supported [M41](#m41) | Supported [M21](#m21) | Supported [M61](#m61) |
| Number | Backend | Supported [M11](#m11) | Supported [M51](#m51) | Supported [M31](#m31) | Supported [M71](#m71) |
| Propagate | Ordered | → [M01](#m01) | → [M41](#m41) | → [M21](#m21) | → [M61](#m61) |
| Propagate | Backend | → [M11](#m11) | → [M51](#m51) | → [M31](#m31) | → [M71](#m71) |
| Backend | Ordered | → [M01](#m01) | → [M41](#m41) | → [M21](#m21) | → [M61](#m61) |
| Backend | Backend | → [M11](#m11) | → [M51](#m51) | → [M31](#m31) | → [M71](#m71) |

</details>

<details><summary>Normal: all precision/result-policy × validation × execution choices</summary>

| NaN | Zero | Checked / Auto | Checked / Caller | AssumeValid / Auto | AssumeValid / Caller |
|---|---|---|---|---|---|
| Number | Ordered | Supported [M02](#m02) | Supported [M42](#m42) | Supported [M22](#m22) | Supported [M62](#m62) |
| Number | Backend | → [M02](#m02) | → [M42](#m42) | → [M22](#m22) | → [M62](#m62) |
| Propagate | Ordered | → [M02](#m02) | → [M42](#m42) | → [M22](#m22) | → [M62](#m62) |
| Propagate | Backend | → [M02](#m02) | → [M42](#m42) | → [M22](#m22) | → [M62](#m62) |
| Backend | Ordered | → [M02](#m02) | → [M42](#m42) | → [M22](#m22) | → [M62](#m62) |
| Backend | Backend | → [M02](#m02) | → [M42](#m42) | → [M22](#m22) | → [M62](#m62) |

</details>

## Canonical cells

| ID | Domain | NaN | Zero | Validation | Execution | Features |
|---|---|---|---|---|---|---|
| <a id="m00"></a>M00 | Full | Number | Ordered | Checked | Automatic | std or no_std |
| <a id="m01"></a>M01 | Finite | Number | Ordered | Checked | Automatic | std or no_std |
| <a id="m02"></a>M02 | Normal | Number | Ordered | Checked | Automatic | std or no_std |
| <a id="m04"></a>M04 | Full | Propagate | Ordered | Checked | Automatic | std or no_std |
| <a id="m08"></a>M08 | Full | Backend | Ordered | Checked | Automatic | std or no_std |
| <a id="m10"></a>M10 | Full | Number | Backend | Checked | Automatic | std or no_std |
| <a id="m11"></a>M11 | Finite | Number | Backend | Checked | Automatic | std or no_std |
| <a id="m14"></a>M14 | Full | Propagate | Backend | Checked | Automatic | std or no_std |
| <a id="m18"></a>M18 | Full | Backend | Backend | Checked | Automatic | std or no_std |
| <a id="m21"></a>M21 | Finite | Number | Ordered | AssumeValid | Automatic | std or no_std |
| <a id="m22"></a>M22 | Normal | Number | Ordered | AssumeValid | Automatic | std or no_std |
| <a id="m31"></a>M31 | Finite | Number | Backend | AssumeValid | Automatic | std or no_std |
| <a id="m40"></a>M40 | Full | Number | Ordered | Checked | Caller | std or no_std |
| <a id="m41"></a>M41 | Finite | Number | Ordered | Checked | Caller | std or no_std |
| <a id="m42"></a>M42 | Normal | Number | Ordered | Checked | Caller | std or no_std |
| <a id="m44"></a>M44 | Full | Propagate | Ordered | Checked | Caller | std or no_std |
| <a id="m48"></a>M48 | Full | Backend | Ordered | Checked | Caller | std or no_std |
| <a id="m50"></a>M50 | Full | Number | Backend | Checked | Caller | std or no_std |
| <a id="m51"></a>M51 | Finite | Number | Backend | Checked | Caller | std or no_std |
| <a id="m54"></a>M54 | Full | Propagate | Backend | Checked | Caller | std or no_std |
| <a id="m58"></a>M58 | Full | Backend | Backend | Checked | Caller | std or no_std |
| <a id="m61"></a>M61 | Finite | Number | Ordered | AssumeValid | Caller | std or no_std |
| <a id="m62"></a>M62 | Normal | Number | Ordered | AssumeValid | Caller | std or no_std |
| <a id="m71"></a>M71 | Finite | Number | Backend | AssumeValid | Caller | std or no_std |

IDs expose the current encoded flags for traceability; construct options with the public builder, not handwritten integers.
Numerical contracts and measured error are distinct from timing. Timing identities also include operation, scalar type, logical width, ISA, entry form, workload, compiler and source.
A redirected configuration may reuse its canonical cell's exact evidence. Other reuse needs a separately verified graph or compiled-code identity.

## Precision and timing evidence

The [primitive permutation reference](https://github.com/imazen/archmage/blob/feat/primitive-contracts/docs/PRIMITIVE-PERMUTATIONS.md) joins these IDs to numerical contracts, measured error and per-ISA/width/workload timings.
Missing measurements are explicitly unmeasured; dirty retained runs are descriptive; M6 scheduler-selected runs are exploratory. Those states do not mean a demonstrated slowdown.

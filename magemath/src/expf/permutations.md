# Complete option permutation matrix

Generated from the actual Rust builders and validity predicates, in std and no_std builds.
Each cell fixes the row's three axes plus the column's validation/execution choices.
This covers **32 raw combinations**, **32 supported with std**,
and **32 distinct supported flag sets**. A flag ID identifies a configuration, not a timing result.

- **Supported ID**: legal configuration; see its canonical row below.
- **→ ID**: `.bits()` returns exactly the same flags as that canonical row. This is a configuration alias, not just similar accuracy.
- **Unsupported**: rejected option combination; const API compilation fails. Transcendental runtime adapters return UnsupportedOptions.
- **std only**: that combination is unsupported without std.
- Execution Caller/Automatic remains a distinct flag when not canonicalized. Typed entries retain their supplied token; array dispatch is a separate timing dimension.

<details><summary>Bounded: all precision/result-policy × validation × execution choices</summary>

| Precision | Portability | Checked / Auto | Checked / Caller | AssumeValid / Auto | AssumeValid / Caller |
|---|---|---|---|---|---|
| Lowp | BitExact | Supported [E08](#e08) | Supported [E0c](#e0c) | Supported [E28](#e28) | Supported [E2c](#e2c) |
| Lowp | BackendDependent | Supported [E18](#e18) | Supported [E1c](#e1c) | Supported [E38](#e38) | Supported [E3c](#e3c) |
| Midp | BitExact | Supported [E00](#e00) | Supported [E04](#e04) | Supported [E20](#e20) | Supported [E24](#e24) |
| Midp | BackendDependent | Supported [E10](#e10) | Supported [E14](#e14) | Supported [E30](#e30) | Supported [E34](#e34) |

</details>

<details><summary>Full: all precision/result-policy × validation × execution choices</summary>

| Precision | Portability | Checked / Auto | Checked / Caller | AssumeValid / Auto | AssumeValid / Caller |
|---|---|---|---|---|---|
| Lowp | BitExact | Supported [E09](#e09) | Supported [E0d](#e0d) | Supported [E29](#e29) | Supported [E2d](#e2d) |
| Lowp | BackendDependent | Supported [E19](#e19) | Supported [E1d](#e1d) | Supported [E39](#e39) | Supported [E3d](#e3d) |
| Midp | BitExact | Supported [E01](#e01) | Supported [E05](#e05) | Supported [E21](#e21) | Supported [E25](#e25) |
| Midp | BackendDependent | Supported [E11](#e11) | Supported [E15](#e15) | Supported [E31](#e31) | Supported [E35](#e35) |

</details>

## Canonical cells

| ID | Domain | Precision | Portability | Validation | Execution | Features |
|---|---|---|---|---|---|---|
| <a id="e00"></a>E00 | Bounded | Midp | BitExact | Checked | Automatic | std or no_std |
| <a id="e01"></a>E01 | Full | Midp | BitExact | Checked | Automatic | std or no_std |
| <a id="e04"></a>E04 | Bounded | Midp | BitExact | Checked | Caller | std or no_std |
| <a id="e05"></a>E05 | Full | Midp | BitExact | Checked | Caller | std or no_std |
| <a id="e08"></a>E08 | Bounded | Lowp | BitExact | Checked | Automatic | std or no_std |
| <a id="e09"></a>E09 | Full | Lowp | BitExact | Checked | Automatic | std or no_std |
| <a id="e0c"></a>E0c | Bounded | Lowp | BitExact | Checked | Caller | std or no_std |
| <a id="e0d"></a>E0d | Full | Lowp | BitExact | Checked | Caller | std or no_std |
| <a id="e10"></a>E10 | Bounded | Midp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="e11"></a>E11 | Full | Midp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="e14"></a>E14 | Bounded | Midp | BackendDependent | Checked | Caller | std or no_std |
| <a id="e15"></a>E15 | Full | Midp | BackendDependent | Checked | Caller | std or no_std |
| <a id="e18"></a>E18 | Bounded | Lowp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="e19"></a>E19 | Full | Lowp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="e1c"></a>E1c | Bounded | Lowp | BackendDependent | Checked | Caller | std or no_std |
| <a id="e1d"></a>E1d | Full | Lowp | BackendDependent | Checked | Caller | std or no_std |
| <a id="e20"></a>E20 | Bounded | Midp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="e21"></a>E21 | Full | Midp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="e24"></a>E24 | Bounded | Midp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="e25"></a>E25 | Full | Midp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="e28"></a>E28 | Bounded | Lowp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="e29"></a>E29 | Full | Lowp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="e2c"></a>E2c | Bounded | Lowp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="e2d"></a>E2d | Full | Lowp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="e30"></a>E30 | Bounded | Midp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="e31"></a>E31 | Full | Midp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="e34"></a>E34 | Bounded | Midp | BackendDependent | AssumeValid | Caller | std or no_std |
| <a id="e35"></a>E35 | Full | Midp | BackendDependent | AssumeValid | Caller | std or no_std |
| <a id="e38"></a>E38 | Bounded | Lowp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="e39"></a>E39 | Full | Lowp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="e3c"></a>E3c | Bounded | Lowp | BackendDependent | AssumeValid | Caller | std or no_std |
| <a id="e3d"></a>E3d | Full | Lowp | BackendDependent | AssumeValid | Caller | std or no_std |

IDs expose the current encoded flags for traceability; construct options with the public builder, not handwritten integers.
Numerical contracts and measured error are distinct from timing. Timing identities also include operation, scalar type, logical width, ISA, entry form, workload, compiler and source.
A redirected configuration may reuse its canonical cell's exact evidence. Other reuse needs a separately verified graph or compiled-code identity.

## Measurement coverage

This expansion inventories every option for this family. New timing coverage is currently scoped to min/max, reciprocal and rsqrt; the transcendental measurement gaps are not presented as completed work.

# Complete option permutation matrix

Generated from the actual Rust builders and validity predicates, in std and no_std builds.
Each cell fixes the row's three axes plus the column's validation/execution choices.
This covers **48 raw combinations**, **32 supported with std**,
and **22 distinct supported flag sets**. A flag ID identifies a configuration, not a timing result.

- **Supported ID**: legal configuration; see its canonical row below.
- **→ ID**: `.bits()` returns exactly the same flags as that canonical row. This is a configuration alias, not just similar accuracy.
- **Unsupported**: rejected option combination; const API compilation fails. Transcendental runtime adapters return UnsupportedOptions.
- **std only**: that combination is unsupported without std.
- Execution Caller/Automatic remains a distinct flag when not canonicalized. Typed entries retain their supplied token; array dispatch is a separate timing dimension.

<details><summary>Full: all precision/result-policy × validation × execution choices</summary>

| Precision | Portability | Checked / Auto | Checked / Caller | AssumeValid / Auto | AssumeValid / Caller |
|---|---|---|---|---|---|
| Lowp | BitExact | Unsupported | Unsupported | Unsupported | Unsupported |
| Lowp | BackendDependent | Unsupported | Unsupported | Unsupported | Unsupported |
| Midp | BitExact | Unsupported | Unsupported | Unsupported | Unsupported |
| Midp | BackendDependent | Unsupported | Unsupported | Unsupported | Unsupported |
| Highp | BitExact | Supported [R00](#r00) | Supported [R20](#r20) | → [R00](#r00) | → [R20](#r20) |
| Highp | BackendDependent | → [R00](#r00) | → [R20](#r20) | → [R00](#r00) | → [R20](#r20) |

</details>

<details><summary>Normal: all precision/result-policy × validation × execution choices</summary>

| Precision | Portability | Checked / Auto | Checked / Caller | AssumeValid / Auto | AssumeValid / Caller |
|---|---|---|---|---|---|
| Lowp | BitExact | Supported [R05](#r05) | Supported [R25](#r25) | Supported [R15](#r15) | Supported [R35](#r35) |
| Lowp | BackendDependent | Supported [R0d](#r0d) | Supported [R2d](#r2d) | Supported [R1d](#r1d) | Supported [R3d](#r3d) |
| Midp | BitExact | Supported [R03](#r03) | Supported [R23](#r23) | Supported [R13](#r13) | Supported [R33](#r33) |
| Midp | BackendDependent | Supported [R0b](#r0b) | Supported [R2b](#r2b) | Supported [R1b](#r1b) | Supported [R3b](#r3b) |
| Highp | BitExact | Supported [R01](#r01) | Supported [R21](#r21) | Supported [R11](#r11) | Supported [R31](#r31) |
| Highp | BackendDependent | → [R01](#r01) | → [R21](#r21) | → [R11](#r11) | → [R31](#r31) |

</details>

## Canonical cells

| ID | Domain | Precision | Portability | Validation | Execution | Features |
|---|---|---|---|---|---|---|
| <a id="r00"></a>R00 | Full | Highp | BitExact | Checked | Automatic | std or no_std |
| <a id="r01"></a>R01 | Normal | Highp | BitExact | Checked | Automatic | std or no_std |
| <a id="r03"></a>R03 | Normal | Midp | BitExact | Checked | Automatic | std or no_std |
| <a id="r05"></a>R05 | Normal | Lowp | BitExact | Checked | Automatic | std or no_std |
| <a id="r0b"></a>R0b | Normal | Midp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="r0d"></a>R0d | Normal | Lowp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="r11"></a>R11 | Normal | Highp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="r13"></a>R13 | Normal | Midp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="r15"></a>R15 | Normal | Lowp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="r1b"></a>R1b | Normal | Midp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="r1d"></a>R1d | Normal | Lowp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="r20"></a>R20 | Full | Highp | BitExact | Checked | Caller | std or no_std |
| <a id="r21"></a>R21 | Normal | Highp | BitExact | Checked | Caller | std or no_std |
| <a id="r23"></a>R23 | Normal | Midp | BitExact | Checked | Caller | std or no_std |
| <a id="r25"></a>R25 | Normal | Lowp | BitExact | Checked | Caller | std or no_std |
| <a id="r2b"></a>R2b | Normal | Midp | BackendDependent | Checked | Caller | std or no_std |
| <a id="r2d"></a>R2d | Normal | Lowp | BackendDependent | Checked | Caller | std or no_std |
| <a id="r31"></a>R31 | Normal | Highp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="r33"></a>R33 | Normal | Midp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="r35"></a>R35 | Normal | Lowp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="r3b"></a>R3b | Normal | Midp | BackendDependent | AssumeValid | Caller | std or no_std |
| <a id="r3d"></a>R3d | Normal | Lowp | BackendDependent | AssumeValid | Caller | std or no_std |

IDs expose the current encoded flags for traceability; construct options with the public builder, not handwritten integers.
Numerical contracts and measured error are distinct from timing. Timing identities also include operation, scalar type, logical width, ISA, entry form, workload, compiler and source.
A redirected configuration may reuse its canonical cell's exact evidence. Other reuse needs a separately verified graph or compiled-code identity.

## Precision and timing evidence

The [primitive permutation reference](https://github.com/imazen/archmage/blob/feat/primitive-contracts/docs/PRIMITIVE-PERMUTATIONS.md) joins these IDs to numerical contracts, measured error and per-ISA/width/workload timings.
Missing measurements are explicitly unmeasured; dirty retained runs are descriptive; M6 scheduler-selected runs are exploratory. Those states do not mean a demonstrated slowdown.

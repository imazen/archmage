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

<details><summary>Positive: all precision/result-policy × validation × execution choices</summary>

| Precision | Portability | Checked / Auto | Checked / Caller | AssumeValid / Auto | AssumeValid / Caller |
|---|---|---|---|---|---|
| Lowp | BitExact | Supported [L08](#l08) | Supported [L0c](#l0c) | Supported [L28](#l28) | Supported [L2c](#l2c) |
| Lowp | BackendDependent | Supported [L18](#l18) | Supported [L1c](#l1c) | Supported [L38](#l38) | Supported [L3c](#l3c) |
| Midp | BitExact | Supported [L00](#l00) | Supported [L04](#l04) | Supported [L20](#l20) | Supported [L24](#l24) |
| Midp | BackendDependent | Supported [L10](#l10) | Supported [L14](#l14) | Supported [L30](#l30) | Supported [L34](#l34) |

</details>

<details><summary>Specials: all precision/result-policy × validation × execution choices</summary>

| Precision | Portability | Checked / Auto | Checked / Caller | AssumeValid / Auto | AssumeValid / Caller |
|---|---|---|---|---|---|
| Lowp | BitExact | Supported [L09](#l09) | Supported [L0d](#l0d) | Supported [L29](#l29) | Supported [L2d](#l2d) |
| Lowp | BackendDependent | Supported [L19](#l19) | Supported [L1d](#l1d) | Supported [L39](#l39) | Supported [L3d](#l3d) |
| Midp | BitExact | Supported [L01](#l01) | Supported [L05](#l05) | Supported [L21](#l21) | Supported [L25](#l25) |
| Midp | BackendDependent | Supported [L11](#l11) | Supported [L15](#l15) | Supported [L31](#l31) | Supported [L35](#l35) |

</details>

## Canonical cells

| ID | Domain | Precision | Portability | Validation | Execution | Features |
|---|---|---|---|---|---|---|
| <a id="l00"></a>L00 | Positive | Midp | BitExact | Checked | Automatic | std or no_std |
| <a id="l01"></a>L01 | Specials | Midp | BitExact | Checked | Automatic | std or no_std |
| <a id="l04"></a>L04 | Positive | Midp | BitExact | Checked | Caller | std or no_std |
| <a id="l05"></a>L05 | Specials | Midp | BitExact | Checked | Caller | std or no_std |
| <a id="l08"></a>L08 | Positive | Lowp | BitExact | Checked | Automatic | std or no_std |
| <a id="l09"></a>L09 | Specials | Lowp | BitExact | Checked | Automatic | std or no_std |
| <a id="l0c"></a>L0c | Positive | Lowp | BitExact | Checked | Caller | std or no_std |
| <a id="l0d"></a>L0d | Specials | Lowp | BitExact | Checked | Caller | std or no_std |
| <a id="l10"></a>L10 | Positive | Midp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="l11"></a>L11 | Specials | Midp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="l14"></a>L14 | Positive | Midp | BackendDependent | Checked | Caller | std or no_std |
| <a id="l15"></a>L15 | Specials | Midp | BackendDependent | Checked | Caller | std or no_std |
| <a id="l18"></a>L18 | Positive | Lowp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="l19"></a>L19 | Specials | Lowp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="l1c"></a>L1c | Positive | Lowp | BackendDependent | Checked | Caller | std or no_std |
| <a id="l1d"></a>L1d | Specials | Lowp | BackendDependent | Checked | Caller | std or no_std |
| <a id="l20"></a>L20 | Positive | Midp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="l21"></a>L21 | Specials | Midp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="l24"></a>L24 | Positive | Midp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="l25"></a>L25 | Specials | Midp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="l28"></a>L28 | Positive | Lowp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="l29"></a>L29 | Specials | Lowp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="l2c"></a>L2c | Positive | Lowp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="l2d"></a>L2d | Specials | Lowp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="l30"></a>L30 | Positive | Midp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="l31"></a>L31 | Specials | Midp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="l34"></a>L34 | Positive | Midp | BackendDependent | AssumeValid | Caller | std or no_std |
| <a id="l35"></a>L35 | Specials | Midp | BackendDependent | AssumeValid | Caller | std or no_std |
| <a id="l38"></a>L38 | Positive | Lowp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="l39"></a>L39 | Specials | Lowp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="l3c"></a>L3c | Positive | Lowp | BackendDependent | AssumeValid | Caller | std or no_std |
| <a id="l3d"></a>L3d | Specials | Lowp | BackendDependent | AssumeValid | Caller | std or no_std |

IDs expose the current encoded flags for traceability; construct options with the public builder, not handwritten integers.
Numerical contracts and measured error are distinct from timing. Timing identities also include operation, scalar type, logical width, ISA, entry form, workload, compiler and source.
An equal-flag redirect uses the same instantiation. A source-level graph redirect does not license copying timings: the compiled execution path and call context must match too.

## Measurement coverage

This expansion inventories every option for this family. New timing coverage is currently scoped to min/max, reciprocal and rsqrt; the transcendental measurement gaps are not presented as completed work.

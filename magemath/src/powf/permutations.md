# Complete option permutation matrix

Generated from the actual Rust builders and validity predicates, in std and no_std builds.
Each cell fixes the row's three axes plus the column's validation/execution choices.
This covers **120 raw combinations**, **104 supported with std**,
and **104 distinct supported flag sets**. A flag ID identifies a configuration, not a timing result.

- **Supported ID**: legal configuration; see its canonical row below.
- **→ ID**: `.bits()` returns exactly the same flags as that canonical row. This is a configuration alias, not just similar accuracy.
- **Unsupported**: rejected option combination; const API compilation fails. Transcendental runtime adapters return UnsupportedOptions.
- **std only**: that combination is unsupported without std.
- Execution Caller/Automatic remains a distinct flag when not canonicalized. Typed entries retain their supplied token; array dispatch is a separate timing dimension.

<details><summary>PositiveNormal: all precision/result-policy × validation × execution choices</summary>

| Precision | Portability | Checked / Auto | Checked / Caller | AssumeValid / Auto | AssumeValid / Caller |
|---|---|---|---|---|---|
| Lowp | BitExact | Supported [P08](#p08) | Supported [P0c](#p0c) | Supported [P28](#p28) | Supported [P2c](#p2c) |
| Lowp | BackendDependent | Supported [P18](#p18) | Supported [P1c](#p1c) | Supported [P38](#p38) | Supported [P3c](#p3c) |
| Midp | BitExact | Supported [P00](#p00) | Supported [P04](#p04) | Supported [P20](#p20) | Supported [P24](#p24) |
| Midp | BackendDependent | Supported [P10](#p10) | Supported [P14](#p14) | Supported [P30](#p30) | Supported [P34](#p34) |
| Highp | BitExact | Supported [P40](#p40) | Supported [P44](#p44) | Supported [P60](#p60) | Supported [P64](#p64) |
| Highp | BackendDependent | Supported [P50](#p50) (std only) | Supported [P54](#p54) (std only) | Supported [P70](#p70) (std only) | Supported [P74](#p74) (std only) |

</details>

<details><summary>NonNegativeNormal: all precision/result-policy × validation × execution choices</summary>

| Precision | Portability | Checked / Auto | Checked / Caller | AssumeValid / Auto | AssumeValid / Caller |
|---|---|---|---|---|---|
| Lowp | BitExact | Supported [P09](#p09) | Supported [P0d](#p0d) | Supported [P29](#p29) | Supported [P2d](#p2d) |
| Lowp | BackendDependent | Supported [P19](#p19) | Supported [P1d](#p1d) | Supported [P39](#p39) | Supported [P3d](#p3d) |
| Midp | BitExact | Supported [P01](#p01) | Supported [P05](#p05) | Supported [P21](#p21) | Supported [P25](#p25) |
| Midp | BackendDependent | Supported [P11](#p11) | Supported [P15](#p15) | Supported [P31](#p31) | Supported [P35](#p35) |
| Highp | BitExact | Supported [P41](#p41) | Supported [P45](#p45) | Supported [P61](#p61) | Supported [P65](#p65) |
| Highp | BackendDependent | Supported [P51](#p51) (std only) | Supported [P55](#p55) (std only) | Supported [P71](#p71) (std only) | Supported [P75](#p75) (std only) |

</details>

<details><summary>Gamma: all precision/result-policy × validation × execution choices</summary>

| Precision | Portability | Checked / Auto | Checked / Caller | AssumeValid / Auto | AssumeValid / Caller |
|---|---|---|---|---|---|
| Lowp | BitExact | Supported [P0a](#p0a) | Supported [P0e](#p0e) | Supported [P2a](#p2a) | Supported [P2e](#p2e) |
| Lowp | BackendDependent | Supported [P1a](#p1a) | Supported [P1e](#p1e) | Supported [P3a](#p3a) | Supported [P3e](#p3e) |
| Midp | BitExact | Supported [P02](#p02) | Supported [P06](#p06) | Supported [P22](#p22) | Supported [P26](#p26) |
| Midp | BackendDependent | Supported [P12](#p12) | Supported [P16](#p16) | Supported [P32](#p32) | Supported [P36](#p36) |
| Highp | BitExact | Supported [P42](#p42) | Supported [P46](#p46) | Supported [P62](#p62) | Supported [P66](#p66) |
| Highp | BackendDependent | Supported [P52](#p52) (std only) | Supported [P56](#p56) (std only) | Supported [P72](#p72) (std only) | Supported [P76](#p76) (std only) |

</details>

<details><summary>PositiveGamma: all precision/result-policy × validation × execution choices</summary>

| Precision | Portability | Checked / Auto | Checked / Caller | AssumeValid / Auto | AssumeValid / Caller |
|---|---|---|---|---|---|
| Lowp | BitExact | Supported [P8a](#p8a) | Supported [P8e](#p8e) | Supported [Paa](#paa) | Supported [Pae](#pae) |
| Lowp | BackendDependent | Supported [P9a](#p9a) | Supported [P9e](#p9e) | Supported [Pba](#pba) | Supported [Pbe](#pbe) |
| Midp | BitExact | Supported [P82](#p82) | Supported [P86](#p86) | Supported [Pa2](#pa2) | Supported [Pa6](#pa6) |
| Midp | BackendDependent | Supported [P92](#p92) | Supported [P96](#p96) | Supported [Pb2](#pb2) | Supported [Pb6](#pb6) |
| Highp | BitExact | Supported [Pc2](#pc2) | Supported [Pc6](#pc6) | Supported [Pe2](#pe2) | Supported [Pe6](#pe6) |
| Highp | BackendDependent | Supported [Pd2](#pd2) (std only) | Supported [Pd6](#pd6) (std only) | Supported [Pf2](#pf2) (std only) | Supported [Pf6](#pf6) (std only) |

</details>

<details><summary>Full: all precision/result-policy × validation × execution choices</summary>

| Precision | Portability | Checked / Auto | Checked / Caller | AssumeValid / Auto | AssumeValid / Caller |
|---|---|---|---|---|---|
| Lowp | BitExact | Unsupported | Unsupported | Unsupported | Unsupported |
| Lowp | BackendDependent | Unsupported | Unsupported | Unsupported | Unsupported |
| Midp | BitExact | Unsupported | Unsupported | Unsupported | Unsupported |
| Midp | BackendDependent | Unsupported | Unsupported | Unsupported | Unsupported |
| Highp | BitExact | Supported [P43](#p43) | Supported [P47](#p47) | Supported [P63](#p63) | Supported [P67](#p67) |
| Highp | BackendDependent | Supported [P53](#p53) (std only) | Supported [P57](#p57) (std only) | Supported [P73](#p73) (std only) | Supported [P77](#p77) (std only) |

</details>

## Canonical cells

| ID | Domain | Precision | Portability | Validation | Execution | Features |
|---|---|---|---|---|---|---|
| <a id="p00"></a>P00 | PositiveNormal | Midp | BitExact | Checked | Automatic | std or no_std |
| <a id="p01"></a>P01 | NonNegativeNormal | Midp | BitExact | Checked | Automatic | std or no_std |
| <a id="p02"></a>P02 | Gamma | Midp | BitExact | Checked | Automatic | std or no_std |
| <a id="p04"></a>P04 | PositiveNormal | Midp | BitExact | Checked | Caller | std or no_std |
| <a id="p05"></a>P05 | NonNegativeNormal | Midp | BitExact | Checked | Caller | std or no_std |
| <a id="p06"></a>P06 | Gamma | Midp | BitExact | Checked | Caller | std or no_std |
| <a id="p08"></a>P08 | PositiveNormal | Lowp | BitExact | Checked | Automatic | std or no_std |
| <a id="p09"></a>P09 | NonNegativeNormal | Lowp | BitExact | Checked | Automatic | std or no_std |
| <a id="p0a"></a>P0a | Gamma | Lowp | BitExact | Checked | Automatic | std or no_std |
| <a id="p0c"></a>P0c | PositiveNormal | Lowp | BitExact | Checked | Caller | std or no_std |
| <a id="p0d"></a>P0d | NonNegativeNormal | Lowp | BitExact | Checked | Caller | std or no_std |
| <a id="p0e"></a>P0e | Gamma | Lowp | BitExact | Checked | Caller | std or no_std |
| <a id="p10"></a>P10 | PositiveNormal | Midp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="p11"></a>P11 | NonNegativeNormal | Midp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="p12"></a>P12 | Gamma | Midp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="p14"></a>P14 | PositiveNormal | Midp | BackendDependent | Checked | Caller | std or no_std |
| <a id="p15"></a>P15 | NonNegativeNormal | Midp | BackendDependent | Checked | Caller | std or no_std |
| <a id="p16"></a>P16 | Gamma | Midp | BackendDependent | Checked | Caller | std or no_std |
| <a id="p18"></a>P18 | PositiveNormal | Lowp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="p19"></a>P19 | NonNegativeNormal | Lowp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="p1a"></a>P1a | Gamma | Lowp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="p1c"></a>P1c | PositiveNormal | Lowp | BackendDependent | Checked | Caller | std or no_std |
| <a id="p1d"></a>P1d | NonNegativeNormal | Lowp | BackendDependent | Checked | Caller | std or no_std |
| <a id="p1e"></a>P1e | Gamma | Lowp | BackendDependent | Checked | Caller | std or no_std |
| <a id="p20"></a>P20 | PositiveNormal | Midp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="p21"></a>P21 | NonNegativeNormal | Midp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="p22"></a>P22 | Gamma | Midp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="p24"></a>P24 | PositiveNormal | Midp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="p25"></a>P25 | NonNegativeNormal | Midp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="p26"></a>P26 | Gamma | Midp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="p28"></a>P28 | PositiveNormal | Lowp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="p29"></a>P29 | NonNegativeNormal | Lowp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="p2a"></a>P2a | Gamma | Lowp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="p2c"></a>P2c | PositiveNormal | Lowp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="p2d"></a>P2d | NonNegativeNormal | Lowp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="p2e"></a>P2e | Gamma | Lowp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="p30"></a>P30 | PositiveNormal | Midp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="p31"></a>P31 | NonNegativeNormal | Midp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="p32"></a>P32 | Gamma | Midp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="p34"></a>P34 | PositiveNormal | Midp | BackendDependent | AssumeValid | Caller | std or no_std |
| <a id="p35"></a>P35 | NonNegativeNormal | Midp | BackendDependent | AssumeValid | Caller | std or no_std |
| <a id="p36"></a>P36 | Gamma | Midp | BackendDependent | AssumeValid | Caller | std or no_std |
| <a id="p38"></a>P38 | PositiveNormal | Lowp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="p39"></a>P39 | NonNegativeNormal | Lowp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="p3a"></a>P3a | Gamma | Lowp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="p3c"></a>P3c | PositiveNormal | Lowp | BackendDependent | AssumeValid | Caller | std or no_std |
| <a id="p3d"></a>P3d | NonNegativeNormal | Lowp | BackendDependent | AssumeValid | Caller | std or no_std |
| <a id="p3e"></a>P3e | Gamma | Lowp | BackendDependent | AssumeValid | Caller | std or no_std |
| <a id="p40"></a>P40 | PositiveNormal | Highp | BitExact | Checked | Automatic | std or no_std |
| <a id="p41"></a>P41 | NonNegativeNormal | Highp | BitExact | Checked | Automatic | std or no_std |
| <a id="p42"></a>P42 | Gamma | Highp | BitExact | Checked | Automatic | std or no_std |
| <a id="p43"></a>P43 | Full | Highp | BitExact | Checked | Automatic | std or no_std |
| <a id="p44"></a>P44 | PositiveNormal | Highp | BitExact | Checked | Caller | std or no_std |
| <a id="p45"></a>P45 | NonNegativeNormal | Highp | BitExact | Checked | Caller | std or no_std |
| <a id="p46"></a>P46 | Gamma | Highp | BitExact | Checked | Caller | std or no_std |
| <a id="p47"></a>P47 | Full | Highp | BitExact | Checked | Caller | std or no_std |
| <a id="p50"></a>P50 | PositiveNormal | Highp | BackendDependent | Checked | Automatic | std required |
| <a id="p51"></a>P51 | NonNegativeNormal | Highp | BackendDependent | Checked | Automatic | std required |
| <a id="p52"></a>P52 | Gamma | Highp | BackendDependent | Checked | Automatic | std required |
| <a id="p53"></a>P53 | Full | Highp | BackendDependent | Checked | Automatic | std required |
| <a id="p54"></a>P54 | PositiveNormal | Highp | BackendDependent | Checked | Caller | std required |
| <a id="p55"></a>P55 | NonNegativeNormal | Highp | BackendDependent | Checked | Caller | std required |
| <a id="p56"></a>P56 | Gamma | Highp | BackendDependent | Checked | Caller | std required |
| <a id="p57"></a>P57 | Full | Highp | BackendDependent | Checked | Caller | std required |
| <a id="p60"></a>P60 | PositiveNormal | Highp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="p61"></a>P61 | NonNegativeNormal | Highp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="p62"></a>P62 | Gamma | Highp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="p63"></a>P63 | Full | Highp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="p64"></a>P64 | PositiveNormal | Highp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="p65"></a>P65 | NonNegativeNormal | Highp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="p66"></a>P66 | Gamma | Highp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="p67"></a>P67 | Full | Highp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="p70"></a>P70 | PositiveNormal | Highp | BackendDependent | AssumeValid | Automatic | std required |
| <a id="p71"></a>P71 | NonNegativeNormal | Highp | BackendDependent | AssumeValid | Automatic | std required |
| <a id="p72"></a>P72 | Gamma | Highp | BackendDependent | AssumeValid | Automatic | std required |
| <a id="p73"></a>P73 | Full | Highp | BackendDependent | AssumeValid | Automatic | std required |
| <a id="p74"></a>P74 | PositiveNormal | Highp | BackendDependent | AssumeValid | Caller | std required |
| <a id="p75"></a>P75 | NonNegativeNormal | Highp | BackendDependent | AssumeValid | Caller | std required |
| <a id="p76"></a>P76 | Gamma | Highp | BackendDependent | AssumeValid | Caller | std required |
| <a id="p77"></a>P77 | Full | Highp | BackendDependent | AssumeValid | Caller | std required |
| <a id="p82"></a>P82 | PositiveGamma | Midp | BitExact | Checked | Automatic | std or no_std |
| <a id="p86"></a>P86 | PositiveGamma | Midp | BitExact | Checked | Caller | std or no_std |
| <a id="p8a"></a>P8a | PositiveGamma | Lowp | BitExact | Checked | Automatic | std or no_std |
| <a id="p8e"></a>P8e | PositiveGamma | Lowp | BitExact | Checked | Caller | std or no_std |
| <a id="p92"></a>P92 | PositiveGamma | Midp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="p96"></a>P96 | PositiveGamma | Midp | BackendDependent | Checked | Caller | std or no_std |
| <a id="p9a"></a>P9a | PositiveGamma | Lowp | BackendDependent | Checked | Automatic | std or no_std |
| <a id="p9e"></a>P9e | PositiveGamma | Lowp | BackendDependent | Checked | Caller | std or no_std |
| <a id="pa2"></a>Pa2 | PositiveGamma | Midp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="pa6"></a>Pa6 | PositiveGamma | Midp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="paa"></a>Paa | PositiveGamma | Lowp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="pae"></a>Pae | PositiveGamma | Lowp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="pb2"></a>Pb2 | PositiveGamma | Midp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="pb6"></a>Pb6 | PositiveGamma | Midp | BackendDependent | AssumeValid | Caller | std or no_std |
| <a id="pba"></a>Pba | PositiveGamma | Lowp | BackendDependent | AssumeValid | Automatic | std or no_std |
| <a id="pbe"></a>Pbe | PositiveGamma | Lowp | BackendDependent | AssumeValid | Caller | std or no_std |
| <a id="pc2"></a>Pc2 | PositiveGamma | Highp | BitExact | Checked | Automatic | std or no_std |
| <a id="pc6"></a>Pc6 | PositiveGamma | Highp | BitExact | Checked | Caller | std or no_std |
| <a id="pd2"></a>Pd2 | PositiveGamma | Highp | BackendDependent | Checked | Automatic | std required |
| <a id="pd6"></a>Pd6 | PositiveGamma | Highp | BackendDependent | Checked | Caller | std required |
| <a id="pe2"></a>Pe2 | PositiveGamma | Highp | BitExact | AssumeValid | Automatic | std or no_std |
| <a id="pe6"></a>Pe6 | PositiveGamma | Highp | BitExact | AssumeValid | Caller | std or no_std |
| <a id="pf2"></a>Pf2 | PositiveGamma | Highp | BackendDependent | AssumeValid | Automatic | std required |
| <a id="pf6"></a>Pf6 | PositiveGamma | Highp | BackendDependent | AssumeValid | Caller | std required |

IDs expose the current encoded flags for traceability; construct options with the public builder, not handwritten integers.
Numerical contracts and measured error are distinct from timing. Timing identities also include operation, scalar type, logical width, ISA, entry form, workload, compiler and source.
A redirected configuration may reuse its canonical cell's exact evidence. Other reuse needs a separately verified graph or compiled-code identity.

## Measurement coverage

This expansion inventories every option for this family. New timing coverage is currently scoped to min/max, reciprocal and rsqrt; the transcendental measurement gaps are not presented as completed work.

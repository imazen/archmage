# Generated initial numeric budgets

MPFR4.2.2,1024-bit directed operations; exact rational error budgets.
Source generator and runtime remain under root review; no full sweep yet.

## Exact power domains (inclusive)

- decode: exponent`0x4019999a`, inputbits`0x3db71074..=0x3f800000`, count29945741.
- encode: exponent`0x3ed55555`, inputbits`0x3b475041..=0x3f800000`, count70823872.

Rust actual-C0 preprocessing equality is a smoke gate, not assumed from this script.

| Kind | Degree | Layout | Tail upper | Eval upper | Largest absolute pad | Max suffix upper |
|---|---:|---|---:|---:|---|---:|
| decode | 8 | horner | 6.533248156e-08 | 3.544595331e-15 | `0x1.0000000000000p-22` | 1.995429776e+00 |
| decode | 12 | horner | 1.946498366e-10 | 5.316893145e-15 | `0x1.0000000000000p-30` | 1.995429832e+00 |
| decode | 16 | horner | 8.864497460e-13 | 7.089190861e-15 | `0x1.0000000000000p-38` | 1.995429832e+00 |
| decode | 20 | horner | 5.069812574e-15 | 8.861488576e-15 | `0x1.0000000000000p-44` | 1.995429832e+00 |
| decode | 8 | estrin | 6.533248156e-08 | 1.019392234e-15 | `0x1.0000000000000p-22` | 1.995429776e+00 |
| decode | 12 | estrin | 1.946498366e-10 | 1.019392386e-15 | `0x1.0000000000000p-30` | 1.995429832e+00 |
| decode | 16 | estrin | 8.864497460e-13 | 1.240929601e-15 | `0x1.0000000000000p-38` | 1.995429832e+00 |
| decode | 20 | estrin | 5.069812574e-15 | 1.240929601e-15 | `0x1.0000000000000p-45` | 1.995429832e+00 |
| encode | 8 | horner | 9.557922294e-07 | 2.052478599e-15 | `0x1.0000000000000p-19` | 1.155442732e+00 |
| encode | 12 | horner | 6.934095150e-09 | 3.078720265e-15 | `0x1.0000000000000p-26` | 1.155443620e+00 |
| encode | 16 | horner | 5.821512082e-11 | 4.104960376e-15 | `0x1.0000000000000p-33` | 1.155443626e+00 |
| encode | 20 | horner | 5.309561574e-13 | 5.131200470e-15 | `0x1.0000000000000p-40` | 1.155443626e+00 |
| encode | 8 | estrin | 9.557922294e-07 | 5.326522496e-16 | `0x1.0000000000000p-19` | 1.155442732e+00 |
| encode | 12 | estrin | 6.934095150e-09 | 5.326543206e-16 | `0x1.0000000000000p-26` | 1.155443620e+00 |
| encode | 16 | estrin | 5.821512082e-11 | 6.609343484e-16 | `0x1.0000000000000p-33` | 1.155443626e+00 |
| encode | 20 | estrin | 5.309561574e-13 | 6.609343485e-16 | `0x1.0000000000000p-40` | 1.155443626e+00 |

Every suffix/node is bounded below8, every coefficient below2. Generator proves
y-Delta positive and y+Delta finite before using bit-adjacent outward endpoints.
Other degrees8..20 are retained in the full CSV/JSON. All scalar52 combinations
will be smoke-tested; exhaustive sweeps will cover timing finalists only.

Full numerical certificates (generated data, not committed):
`/mnt/tower/output/archmage/zenmath-gamma-opt-2026-11-06/r5900xt/runs/initial-bound-proof.json`
and `initial-bound-summary.csv`.

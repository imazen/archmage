+++
title = "Precision Levels"
weight = 2
+++

Choose precision from the algorithm's input domain and error budget. `_lowp`
and `_midp` name approximation families; they are not a universal relative-error
or ULP guarantee for every function, backend, and exceptional input.

| Suffix | Decision |
|---|---|
| `_lowp` | Lower-cost approximation where its measured domain error is acceptable |
| `_midp` | More accurate approximation family; validate the relevant function and range |
| `_midp_precise` where provided | Additional correction; inspect its documented domain and cost |
| `_unchecked` where provided | Omits domain repair under documented numerical preconditions |

Unchecked numerical methods remain memory-safe. Out-of-domain values have
unspecified or unsuitable numerical results, not permission to violate Rust
memory safety. Repair can involve several comparisons, blends, and arithmetic
operations; do not describe its cost as universally one comparison.

Checked logarithms distinguish zero from negative values: for example the
intended logarithmic zero rail is negative infinity, not a blanket NaN rule.
Use the [method-specific transcendental tables](@/magetypes/math/transcendentals.md)
and tests for exact behavior. `pow_midp` is not a complete scalar `powf`
replacement for all negative bases and exponents.

For color work, record encoding, range, transfer curve, alpha policy, and error
metric. A nominal bit count does not prove an error is invisible. For reductions,
test cancellation and accumulation length. For bit-exact codec requirements,
compare exact expected output on every supported tier.

The complete [linear-srgb gamma chain](@/magetypes/math/transcendentals.md)
uses a vector approximation and scalar `powf` tail. Those are not bit-identical
implementations. Test the accepted error at the vector/tail boundary and across ISAs.

# Four-lane f64 certificate, array4/8 contexts

The experiment applies the existing scalar certificate independently to each
lane. Reduction and table selection are scalar exact operations. Out-of-domain
lanes receive a safe temporary x=1 for bounded table indices and arithmetic,
and their final result always comes from independent CORE.

The global kernel broadcasts common stored coefficients; the piecewise kernel
loads four independently selected coefficients into each f64 vector. All
polynomial operations use explicit AVX2 multiply and add intrinsics, never
FMA. Horner has the same ordered operations per lane. Estrin has exactly the
generator's pair/carry/square DAG: reduce adjacent coefficient pairs, carry an
odd final node, square the power between stages. Thus every lane uses its
existing generated coefficient, evaluation, tail and scale error budget.

The final vector multiply and y±Delta arithmetic are separate binary64
operations. After storing these endpoints, each lane moves one binary64
pattern outward and compares the two f32 results, preserving its own CORE
rescue. Numerical oracle checks inspect every lane's interval even on rescue.
No vector-mask aggregation can discard an ambiguous lane.

Array4 and array8 timing contexts enter V3 once for the entire buffer; array8
uses two four-lane f64 groups. Scalar tails preserve arbitrary buffer lengths.
Linked assembly must establish whether block helpers inline and whether the
polynomial remains packed. Any helper call or scalarized lowering is reported
and investigated before attributing a SIMD speed result. No Magetypes dependency.

# Sixteen dyadic centers: independent certificate

This is a new candidate with new bounds. The full-mantissa certificate and its
frozen evidence remain unchanged. Domain and exact f32 exponents are identical.

Let j be the upper four stored mantissa bits, 0..15. Set
`c_j=(33+2j)/32`, `v=m-c_j`. Both m and c_j are exactly representable binary64;
`m/c_j` lies in [1/2,2], so subtraction is exact by Sterbenz. Each interval has
`|v|<=h=1/32` (the upper edge is excluded, but use the closed bound). Hence
`|v/c_j|<=r_j=1/(33+2j)<=1/33`, all exact rational bounds.

For exact f32 p, generate exact rational
`a_k=binom(p,k)/c_j^k`, rounded stored binary64 b_k, and
`T_(e,j)=(c_j*2^e)^p` with directed MPFR endpoints. The table argument is exact
binary64 because c_j and the power of two are dyadic in this bounded range.
Degrees N=3,4,5 all satisfy N>=ceil(p). The tail term ratio beyond N+1 is
`|(p-k)/(k+1)|*|v/c_j| <= r_j`. Therefore

`R=|a_(N+1)| h^(N+1)/(1-r_j)`.

Coefficient uncertainty is exact `C=sum_(k=0..N)|b_k-a_k|h^k`;
`S=sum|b_k|h^k`; Horner evaluation error is
`H=gamma_(2N)*S+8N*eta`, u=2^-53 and eta=2^-1075. The eta allowance
covers rounded subnormal results without assuming all intermediates normal.
Every suffix is checked separately:
`B_j=sum_(k=j..N)|b_k|h^(k-j)` and
`(1+gamma_(2(N-j)))*B_j+8(N-j)*eta < 8`.

New coarse invariant: generated `|b_k|<=4`, not the previous <=2. For
`|q|<=8`, a Horner step (separately rounded multiply and add) is bounded by
`((8/32)*(1+u)+eta+4)*(1+u)+eta < 8`.
This separately proves finite intermediate operations. Estrin uses its own
DAG and the add/multiply error propagation in BOUND.md, with every polynomial
magnitude computed using h=1/32 and every node plus error certified below8.
No old Horner error budget is reused for Estrin.

The generator certifies every stored table `0<t<8` (the previous <4 check is
inapplicable), true lower endpoint >2^-32, and table error deltaT<t/2^48.
All bounds are generated separately per p, center, degree, layout and exponent.
Set `E=R+C+eval`, `B=sum|a_k|h^k+R`. The exact final absolute budget is

`D=t*E+deltaT*B+u*t*(S+eval)+eta`.

Round D upward to a power-of-two Delta, using exact rational comparison.
For the endpoint positivity proof, `(1-r_j)^p >= (32/33)^3 > 1/4`
(both p in (0,3)). Thus computed product is bounded below by
`t*(1/4-E)-u*t*(S+eval)-eta`. Generate/check that this exceeds t/8,
Delta<t/32, and the upper product bound plus Delta is below64. The runtime
endpoints are adjacent outward binary64 patterns after rounded y±Delta,
exactly as in BOUND.md. Both f32 endpoint conversions must agree; otherwise
rescue with independent CORE. Any sampled containment failure is a proof bug.

Storage tradeoff: all two-kind coefficient, table and degree/layout error
arrays are emitted explicitly and their exact byte sizes reported by the
generator. No speed or exhaustive coverage claim exists before measurement.

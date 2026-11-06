# Initial fixed-gamma enclosure derivation

Status: analytic candidate for independent review; no exhaustive sweep or speed
claim yet. Implementation base is4ccaf1cf. The reference is always
`zenmath::scalar::reference::powf(x,p)` for the exact binary32 exponent, not std.
Only the admitted positive gamma domain uses the candidate; every other input
(signed zeros,negative,subnormal,nonfinite,out-of-range) goes to CORE unchanged.
The two exponent bit patterns and domain endpoints will be checked against the
actual Rust preprocessing before any sweep.

## Exact reduction and rational coefficients

For a positive normal binary32 input, extract exponent e and exact mantissa
m in[1,2), so x=2^e*m exactly. Set v=m-3/2. Sterbenz gives exact binary64
subtraction; in fact v is a multiple of2^-23 and |v|<=1/2.

For the exact dyadic binary32 p, define the exact rational coefficients

    a[k] = binom(p,k)*(2/3)^k
    P(v) = (1+(2/3)*v)^p
    T[e] = ((3/2)*2^e)^p
    x^p = T[e]*P(v).

No rounded2/3 or runtime u multiplication is used. The rational2/3 exists only
in coefficient generation/proof. Coefficients are computed as exact Python
fractions and enclosed independently with directed MPFR; both directed bounds
must round to the same binary64 constant. Tables T[e] likewise use directed
MPFR pow with exact input and exponent. Generated source retains the input
fractions, MPFR version/precision, interval-derived budgets and provenance.

For degrees N in8..20, N>=ceil(p). For every k>=N+1,

    |a[k+1]*v^(k+1)| / |a[k]*v^k|
      = |p-k|/(k+1) * |(2/3)*v| <= 1/3.

Thus the exact infinite-tail bound is

    R = (3/2)*|a[N+1]|*(1/2)^(N+1).

This bounds both signs of v. It uses the exact analytic1/3, which is legitimate
because v is exact and no computed rounded u exists.

## Separate-arithmetic evaluation budget

Let stored binary64 coefficients be b[k]. Compute these as exact rationals:

    S = sum |b[k]|*2^-k
    C = sum |b[k]-a[k]|*2^-k
    u = 2^-53
    gamma = (2*N*u)/(1-2*N*u).

Strict Horner uses N multiplications and N additions, no implicit contraction.
For normal intermediates the usual absolute bound is gamma*S. To avoid assuming
away cancellation/subnormal intermediates, use the general nearest-even model

    |RN(z)-z| <= u*|z| + eta, eta=2^-1075.

The weighted S is only a final-polynomial bound. Generate every suffix bound
B[j]=sum_(k=j..N)|b[k]|*2^-(k-j), and check its rounding-inflated value
(1+gamma_(2*(N-j)))*B[j]+8*(N-j)*eta below8. Independently check every
|b[k]|<=2. Starting at|b[N]|<=2, the coarse invariant|q|<=8 is preserved by
multiply-by-|v|<=1/2 then add-|b[k]|<=2, including roundoff; this first rules
out overflow without circularly relying on the normal-arithmetic theorem. Each additive eta term
propagates through at most2N factors(1+u) and powers of |v|<=1. Since
(1+u)^(2N)<2 and2+u<3, a deliberately larger8*N*eta covers all such terms.
The stored-coefficient evaluation budget is therefore

    H = gamma*S + 8*N*eta.
    Epoly = R + C + H.

The exact full P is bounded by

    B = sum |a[k]|*2^-k + R,

while the computed Horner result q has |q|<=S+H. Fractions keep all terms exact,
including eta; generation never rounds a bound downward.

## Table and final multiply

Let stored t be the correctly rounded binary64 T[e]. Obtain rigorous binary64
endpoints Tlo<=T[e]<=Thi from directed MPFR and take

    deltaT = max(|t-Tlo|,|Thi-t|).

For y=RN64(t*q), the following absolute budget includes coefficient, truncation,
evaluation, table and final multiply errors:

    D = |t|*Epoly + deltaT*B + u*|t|*(S+H) + eta.

All quantities on the right are exact rational upper bounds. Generate a simple
power-of-two Delta>=D by integer comparisons (no sampled-error substitution).
Table entries/degree choices may have distinct Delta. More conservative rounding
is allowed only in this newly introduced bound, not by relaxing existing tests.

## Runtime certificate and endpoints

Form L=next_down(RN64(y-Delta)) and U=next_up(RN64(y+Delta)). These outward
endpoints enclose the exact real x^p, including the endpoint-add/subtract error.
If RN32(L)==RN32(U), monotonic nearest-even rounding makes that common value the
correct result. Otherwise call independent CORE. Signed-zero/special handling
is inherited entirely through the out-of-domain CORE path.

A branchless positive-finite nextafter implementation may replace the general
helpers only after proving y-Delta>0 and y+Delta finite over every admitted
input, degree and exponent. The generator must check conservative lower margins
(P>1/4 for these0<p<3 and m/1.5>=2/3) and upper margins. Otherwise retain the
general next_down/up helpers. Endpoint handling is part of the certificate,
not a sampling assumption.

## Layout alternatives and validation gates

Estrin is generated as an explicit separate-multiply/add DAG, including each
rounded v^2/v^4/v^8/v^16 node. Every node carries its exact stored-coefficient
polynomial, absolute bound B=sum|coeff|*(1/2)^k and error E. For addition,

    Eadd=Ea+Eb+u*(Ba+Bb+Ea+Eb)+eta.

For multiplication,

    Emul=Ba*Eb+Bb*Ea+Ea*Eb+u*(Ba+Ea)*(Bb+Eb)+eta.

These follow by expanding computed operands around their exact node expressions
and applying the same general rounding model. The generator checks B+E<8 for
every node, and checks the final ideal polynomial equals the stored coefficient
array exactly. This separately bounds Estrin's real operation order, including
shared-power errors; no Horner budget is reused. Explicit FMA would require a
separate DAG operation/error rule and separate measurements.

Before exhaustive runs: publish generated numeric budgets; test reduction,
coefficient/table intervals and endpoint containment against independent MPFR,
including directed halfways and range neighbors. Exhaustive checks cover every
binary32 input bit pattern in each declared gamma domain, use exact CORE output
comparisons and streaming hashes, and fail on any non-NaN bit difference.
Fixed-gamma domains are not arbitrary-pow or all2^32 input coverage.

The actual C0 transfer constants come from clean linear-srgb revision
8900ffc87cd50b249bae19c99d874431e055d888, src/scalar.rs. F32 fixed-power truth,
f32 composed transfer truth and the independent f64-parameter C0 curve oracle
remain separately labeled; their differences are never assigned to the wrong
contract.

MPFR rounding/API reference: [MPFR4.2.2 manual](https://www.mpfr.org/mpfr-current/mpfr.html#Rounding).

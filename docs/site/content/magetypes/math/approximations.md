+++
title = "Approximations"
weight = 3
+++

Reciprocal and reciprocal-square-root methods offer different numerical contracts.
These are separate from the transcendental approximation families.

| Method | Intended use |
|---|---|
| `rcp_approx()` / `rsqrt_approx()` | Estimate tier; exceptional rails are not generally promised |
| `recip()` / `rsqrt()` | Refined working tier with documented zero/infinity/NaN repairs |
| `recip_portable()` / `rsqrt_portable()` | Division-based portable path when the wider input/result range matters |

Do not substitute a handwritten Newton step without carrying over its special
cases. Algebraically equivalent products can behave differently at zero and
infinity; NEON refinement instructions also have useful hardware special cases.

The working reciprocal has a known limitation: V3 can return zero for
`recip(f32::MAX)` instead of the division result's nonzero subnormal. Its nominal
working-precision target therefore does not cover every normal input. Use
`recip_portable()` when this range matters. Subnormal inputs also require care.

See [ISA quirks and measured fixup overhead](@/magetypes/isa-quirks.md) for exact
method calls, rails, and the hardware evidence. Use the
[transcendental guide](@/magetypes/math/transcendentals.md) for pow/log/exp.

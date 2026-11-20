+++
title = "Math"
description = "Transcendental functions, precision levels, and fast approximations"
sort_by = "weight"
weight = 5

[extra]
sidebar = true
+++

SIMD polynomial approximations for common math functions. Faster than scalar `f32::exp()` and friends, with configurable precision.

1. [Transcendentals](@/magetypes/math/transcendentals.md) — Exponentials, logarithms, power, roots
2. [Precision Levels](@/magetypes/math/precision.md) — `_lowp`, `_midp`, full precision — when to use each
3. [Approximations](@/magetypes/math/approximations.md) — `rcp`, `rsqrt`, Newton-Raphson refinement

For configurable input domains (`SingleOctave`, `AroundZero`, `HdrGamma`), explicit subnormal policies, `const_exponent` specialization, cross-ISA `BitExact` rounding, `Highp` `powf`, whole-slice 16-lane AVX-512 widening, and matching scalar tails, see [Magemath](@/magemath/_index.md), [Tuning Performance & Subnormals](@/magemath/tuning-and-subnormals.md), and [Magetypes Fast Paths & Contract Tiers](@/magemath/magetypes-fast-paths.md).

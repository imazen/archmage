+++
title = "Conversions"
description = "Type conversions between SIMD vector types"
sort_by = "weight"
weight = 4

[extra]
sidebar = true
+++

Convert between SIMD types, change widths, reinterpret bits, and safely cast slices.

1. [Float / Integer](@/magetypes/conversions/float-int.md) — `to_i32`, `to_i32_saturating`, `to_i32_round`, `to_f32`
2. [Width Conversions](@/magetypes/conversions/width.md) — Narrowing (pack), widening (extend), half-width (`split`, `from_halves_t`)
3. [Bitcast](@/magetypes/conversions/bitcast.md) — Bit reinterpretation, signed/unsigned conversion
4. [Slice Casting](@/magetypes/conversions/slice-casting.md) — Token-gated `cast_slice_t`, `as_bytes`, `from_bytes_t` (and why not bytemuck)

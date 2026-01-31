# Bitcast NxN Tracking

Currently allowlisted bitcast pairs in `xtask/src/simd_types/ops_bitcast.rs`:

## Implemented

- Float ↔ same-size signed int: f32↔i32, f64↔i64
- Float ↔ same-size unsigned int: f32↔u32, f64↔u64
- Signed ↔ unsigned (same element size): i8↔u8, i16↔u16, i32↔u32, i64↔u64

## Not Yet Allowlisted (cross-element-size pairs)

These require different lane counts (e.g., f32x4 → u8x16). The infrastructure
supports them; just add pairs to `ALLOWED_BITCAST_PAIRS`.

- f32x4 ↔ u8x16, i8x16, u16x8, i16x8, u64x2, i64x2
- f64x2 ↔ u8x16, i8x16, u16x8, i16x8, u32x4, i32x4
- i32x4 ↔ u8x16, i8x16, u16x8, i16x8, u64x2, i64x2, f64x2
- All other cross-width pairs at W256/W512

Same-width constraint: source and target must have the same total byte width.
The generator already enforces this (same SimdWidth).

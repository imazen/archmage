# Magetypes token-taking API inventory

Audited 2026-09-27 against the current working tree, including the pending raw-interoperability additions. Explicit-token vector methods are separate from traits implemented by token types, whose `self` receiver is the proof. Private helpers and archmage re-exports are excluded.

## Vector methods with an explicit token argument

There are 42 distinct method names. All signatures below put the token first; `partition_slice` and `partition_slice_mut` name the unused argument `_`.

| Method | Types / source |
|---|---|
| [cast_slice](../magetypes/src/simd/generic/generated/block_ops_f32x4.rs) | f32x4, f32x8, f64x2, f64x4, i32x4, i32x8, i8x16, u32x4 |
| [cast_slice_mut](../magetypes/src/simd/generic/generated/block_ops_f32x4.rs) | f32x4, f32x8, f64x2, f64x4, i32x4, i32x8, i8x16, u32x4 |
| [from_array](../magetypes/src/simd/scalar.rs) | All 30 generic vector types; all 10 single-lane scalar types |
| [from_bytes](../magetypes/src/simd/generic/generated/block_ops_f32x4.rs) | f32x4, f32x8, f64x2, f64x4, i32x4, i32x8, i8x16, u32x4 |
| [from_bytes_owned](../magetypes/src/simd/generic/generated/block_ops_f32x4.rs) | f32x4, f32x8, f64x2, f64x4, i32x4, i32x8, i8x16, u32x4 |
| [from_float32x4_t](../magetypes/src/simd/generic/generated/f32x4_impl.rs) | f32x4 |
| [from_float64x2_t](../magetypes/src/simd/generic/generated/f64x2_impl.rs) | f64x2 |
| [from_halves](../magetypes/src/simd/generic/cross_width.rs) | f32x8 and f32x16 |
| [from_i32](../magetypes/src/simd/generic/generated/f32x16_impl.rs) | f32x16, f32x4, f32x8 |
| [from_i32_bitcast](../magetypes/src/simd/generic/generated/f32x16_impl.rs) | f32x16, f32x4, f32x8 |
| [from_i32x16](../magetypes/src/simd/generic/generated/f32x16_impl.rs) | f32x16 |
| [from_i32x4](../magetypes/src/simd/generic/generated/f32x4_impl.rs) | f32x4 |
| [from_i32x8](../magetypes/src/simd/generic/generated/f32x8_impl.rs) | f32x8 |
| [from_int16x8_t](../magetypes/src/simd/generic/generated/i16x8_impl.rs) | i16x8 |
| [from_int32x4_t](../magetypes/src/simd/generic/generated/i32x4_impl.rs) | i32x4 |
| [from_int64x2_t](../magetypes/src/simd/generic/generated/i64x2_impl.rs) | i64x2 |
| [from_int8x16_t](../magetypes/src/simd/generic/generated/i8x16_impl.rs) | i8x16 |
| [from_m128](../magetypes/src/simd/generic/generated/f32x4_impl.rs) | f32x4 |
| [from_m128d](../magetypes/src/simd/generic/generated/f64x2_impl.rs) | f64x2 |
| [from_m128i](../magetypes/src/simd/generic/generated/i16x8_impl.rs) | i16x8, i32x4, i64x2, i8x16, u16x8, u32x4, u64x2, u8x16 |
| [from_m256](../magetypes/src/simd/generic/generated/f32x8_impl.rs) | f32x8 |
| [from_m256d](../magetypes/src/simd/generic/generated/f64x4_impl.rs) | f64x4 |
| [from_m256i](../magetypes/src/simd/generic/generated/i16x16_impl.rs) | i16x16, i32x8, i64x4, i8x32, u16x16, u32x8, u64x4, u8x32 |
| [from_m512](../magetypes/src/simd/generic/generated/f32x16_impl.rs) | f32x16 |
| [from_m512d](../magetypes/src/simd/generic/generated/f64x8_impl.rs) | f64x8 |
| [from_m512i](../magetypes/src/simd/generic/generated/i16x32_impl.rs) | i16x32, i32x16, i64x8, i8x64, u16x32, u32x16, u64x8, u8x64 |
| [from_repr](../magetypes/src/simd/generic/generated/f32x16_impl.rs) | All 30 generic vector types |
| [from_slice](../magetypes/src/simd/generic/generated/f32x16_impl.rs) | All 30 generic vector types |
| [from_u8](../magetypes/src/simd/generic/generated/block_ops_f32x4.rs) | f32x4, f32x8 |
| [from_uint16x8_t](../magetypes/src/simd/generic/generated/u16x8_impl.rs) | u16x8 |
| [from_uint32x4_t](../magetypes/src/simd/generic/generated/u32x4_impl.rs) | u32x4 |
| [from_uint64x2_t](../magetypes/src/simd/generic/generated/u64x2_impl.rs) | u64x2 |
| [from_uint8x16_t](../magetypes/src/simd/generic/generated/u8x16_impl.rs) | u8x16 |
| [from_v128](../magetypes/src/simd/generic/generated/f32x4_impl.rs) | f32x4, f64x2, i16x8, i32x4, i64x2, i8x16, u16x8, u32x4, u64x2, u8x16 |
| [load](../magetypes/src/simd/scalar.rs) | All 30 generic vector types; scalar f32x1 and f64x1 |
| [load_4_rgba_u8](../magetypes/src/simd/generic/generated/block_ops_f32x4.rs) | f32x4 |
| [load_8_rgba_u8](../magetypes/src/simd/generic/generated/block_ops_f32x8.rs) | f32x8 |
| [load_8x8](../magetypes/src/simd/generic/generated/block_ops_f32x8.rs) | f32x8 |
| [partition_slice](../magetypes/src/simd/generic/generated/f32x16_impl.rs) | All 30 generic vector types |
| [partition_slice_mut](../magetypes/src/simd/generic/generated/f32x16_impl.rs) | All 30 generic vector types |
| [splat](../magetypes/src/simd/scalar.rs) | All 30 generic vector types; all 10 single-lane scalar types |
| [zero](../magetypes/src/simd/scalar.rs) | All 30 generic vector types; all 10 single-lane scalar types |

512-bit generic types require `w512`; native V4/V4x raw constructors also require `avx512`. NEON `from_*x*_t`, WASM `from_v128`, and x86 `from_m512*` are additions in the pending change. Existing x86 `from_m128*`/`from_m256*` signatures are preserved.

## Token-receiver trait methods

Every method named below takes the token as `self`. Types exposing identical method-name sets are grouped. Availability still follows each trait’s target and Cargo feature gates.

**[F16Convert](../magetypes/src/simd/generic/convert_f16.rs)**

`f16_to_f32_into`, `f16_to_f32_slice`, `f32_to_f16_into`, `f32_to_f16_slice`.

**[F32x16Backend](../magetypes/src/simd/backends/f32x16.rs), [F64x2Backend](../magetypes/src/simd/backends/f64x2.rs), [F64x4Backend](../magetypes/src/simd/backends/f64x4.rs), [F64x8Backend](../magetypes/src/simd/backends/f64x8.rs)**

`abs`, `add`, `bitand`, `bitor`, `bitxor`, `blend`, `ceil`, `clamp`, `div`, `floor`, `from_array`, `load`, `max`, `min`, `mul`, `mul_add`, `mul_sub`, `neg`, `not`, `rcp_approx`, `recip`, `reduce_add`, `reduce_max`, `reduce_min`, `round`, `rsqrt`, `rsqrt_approx`, `simd_eq`, `simd_ge`, `simd_gt`, `simd_le`, `simd_lt`, `simd_ne`, `splat`, `sqrt`, `store`, `sub`, `to_array`, `zero`.

**[F32x4Backend](../magetypes/src/simd/backends/f32x4.rs)**

`abs`, `add`, `bitand`, `bitor`, `bitxor`, `blend`, `ceil`, `clamp`, `div`, `floor`, `from_array`, `load`, `max`, `min`, `mul`, `mul_add`, `mul_sub`, `neg`, `not`, `rcp_approx`, `recip`, `reduce_add`, `reduce_max`, `reduce_min`, `round`, `rsqrt`, `rsqrt_approx`, `simd_eq`, `simd_ge`, `simd_gt`, `simd_le`, `simd_lt`, `simd_ne`, `splat`, `sqrt`, `store`, `store_rgba_bytes`, `sub`, `to_array`, `to_u8_bytes`, `zero`.

**[F32x4Convert](../magetypes/src/simd/backends/convert.rs), [F32x8Convert](../magetypes/src/simd/backends/convert.rs), [F32x16Convert](../magetypes/src/simd/backends/convert.rs)**

`bitcast_f32_to_i32`, `bitcast_i32_to_f32`, `convert_f32_to_i32`, `convert_f32_to_i32_round`, `convert_f32_to_i32_saturating`, `convert_i32_to_f32`.

**[F32x8Backend](../magetypes/src/simd/backends/f32x8.rs)**

`abs`, `add`, `bitand`, `bitor`, `bitxor`, `blend`, `ceil`, `clamp`, `div`, `floor`, `from_array`, `load`, `max`, `min`, `mul`, `mul_add`, `mul_sub`, `neg`, `not`, `rcp_approx`, `recip`, `reduce_add`, `reduce_max`, `reduce_min`, `round`, `rsqrt`, `rsqrt_approx`, `simd_eq`, `simd_ge`, `simd_gt`, `simd_le`, `simd_lt`, `simd_ne`, `splat`, `sqrt`, `store`, `store_rgba_bytes`, `sub`, `to_array`, `to_u8_bytes`, `transpose_8x8_repr`, `zero`.

**[F32x8FromHalves](../magetypes/src/simd/generic/cross_width.rs), [F32x16FromHalves](../magetypes/src/simd/generic/cross_width.rs)**

`from_halves`, `high`, `low`.

**[I16x16Backend](../magetypes/src/simd/backends/i16x16.rs), [I16x32Backend](../magetypes/src/simd/backends/i16x32.rs), [I16x8Backend](../magetypes/src/simd/backends/i16x8.rs)**

`abs`, `abs_diff`, `add`, `all_true`, `any_true`, `bitand`, `bitmask`, `bitor`, `bitxor`, `blend`, `clamp`, `from_array`, `load`, `madd_adjacent`, `max`, `min`, `mul`, `narrow_saturating_i16_to_i8`, `narrow_saturating_i16_to_u8`, `neg`, `not`, `reduce_add`, `saturating_add`, `saturating_sub`, `shl_const`, `shl_uniform`, `shr_arithmetic_const`, `shr_arithmetic_uniform`, `shr_logical_const`, `shr_logical_uniform`, `simd_eq`, `simd_ge`, `simd_gt`, `simd_le`, `simd_lt`, `simd_ne`, `splat`, `store`, `sub`, `to_array`, `widen_high_i16_to_i32`, `widen_low_i16_to_i32`, `zero`.

**[I16x8Bitcast](../magetypes/src/simd/backends/convert_int.rs), [I16x16Bitcast](../magetypes/src/simd/backends/convert_int.rs)**

`bitcast_i16_to_u16`, `bitcast_u16_to_i16`.

**[I32x16Backend](../magetypes/src/simd/backends/i32x16.rs), [I32x4Backend](../magetypes/src/simd/backends/i32x4.rs), [I32x8Backend](../magetypes/src/simd/backends/i32x8.rs)**

`abs`, `add`, `all_true`, `any_true`, `bitand`, `bitmask`, `bitor`, `bitxor`, `blend`, `clamp`, `from_array`, `load`, `max`, `min`, `mul`, `narrow_saturating_i32_to_i16`, `narrow_saturating_i32_to_u16`, `neg`, `not`, `reduce_add`, `shl_const`, `shl_uniform`, `shr_arithmetic_const`, `shr_arithmetic_uniform`, `shr_logical_const`, `shr_logical_uniform`, `simd_eq`, `simd_ge`, `simd_gt`, `simd_le`, `simd_lt`, `simd_ne`, `splat`, `store`, `sub`, `to_array`, `zero`.

**[I64x2Backend](../magetypes/src/simd/backends/i64x2.rs), [I64x4Backend](../magetypes/src/simd/backends/i64x4.rs), [I64x8Backend](../magetypes/src/simd/backends/i64x8.rs)**

`abs`, `add`, `all_true`, `any_true`, `bitand`, `bitmask`, `bitor`, `bitxor`, `blend`, `clamp`, `from_array`, `load`, `max`, `min`, `neg`, `not`, `reduce_add`, `shl_const`, `shr_arithmetic_const`, `shr_logical_const`, `simd_eq`, `simd_ge`, `simd_gt`, `simd_le`, `simd_lt`, `simd_ne`, `splat`, `store`, `sub`, `to_array`, `zero`.

**[I64x2Bitcast](../magetypes/src/simd/backends/convert.rs), [I64x4Bitcast](../magetypes/src/simd/backends/convert.rs)**

`bitcast_f64_to_i64`, `bitcast_i64_to_f64`.

**[I8x16Backend](../magetypes/src/simd/backends/i8x16.rs), [I8x32Backend](../magetypes/src/simd/backends/i8x32.rs), [I8x64Backend](../magetypes/src/simd/backends/i8x64.rs)**

`abs`, `add`, `all_true`, `any_true`, `bitand`, `bitmask`, `bitor`, `bitxor`, `blend`, `clamp`, `from_array`, `load`, `max`, `min`, `neg`, `not`, `reduce_add`, `saturating_add`, `saturating_sub`, `shl_const`, `shr_arithmetic_const`, `shr_logical_const`, `simd_eq`, `simd_ge`, `simd_gt`, `simd_le`, `simd_lt`, `simd_ne`, `splat`, `store`, `sub`, `to_array`, `widen_high_i8_to_i16`, `widen_low_i8_to_i16`, `zero`.

**[I8x16Bitcast](../magetypes/src/simd/backends/convert_int.rs), [I8x32Bitcast](../magetypes/src/simd/backends/convert_int.rs)**

`bitcast_i8_to_u8`, `bitcast_u8_to_i8`.

**[U16x16Backend](../magetypes/src/simd/backends/u16x16.rs), [U16x8Backend](../magetypes/src/simd/backends/u16x8.rs)**

`add`, `all_true`, `any_true`, `bitand`, `bitmask`, `bitor`, `bitxor`, `blend`, `clamp`, `from_array`, `load`, `max`, `min`, `mul`, `not`, `pairwise_widen_add`, `reduce_add`, `saturating_add`, `saturating_sub`, `shl_const`, `shl_uniform`, `shr_logical_const`, `shr_logical_uniform`, `simd_eq`, `simd_ge`, `simd_gt`, `simd_le`, `simd_lt`, `simd_ne`, `splat`, `store`, `sub`, `to_array`, `widen_high_u16_to_u32`, `widen_low_u16_to_u32`, `zero`.

**[U16x32Backend](../magetypes/src/simd/backends/u16x32.rs)**

`add`, `all_true`, `any_true`, `bitand`, `bitmask`, `bitor`, `bitxor`, `blend`, `clamp`, `from_array`, `load`, `max`, `min`, `mul`, `neg`, `not`, `pairwise_widen_add`, `reduce_add`, `saturating_add`, `saturating_sub`, `shl_const`, `shl_uniform`, `shr_arithmetic_const`, `shr_arithmetic_uniform`, `shr_logical_const`, `shr_logical_uniform`, `simd_eq`, `simd_ge`, `simd_gt`, `simd_le`, `simd_lt`, `simd_ne`, `splat`, `store`, `sub`, `to_array`, `widen_high_u16_to_u32`, `widen_low_u16_to_u32`, `zero`.

**[U32x16Backend](../magetypes/src/simd/backends/u32x16.rs)**

`add`, `all_true`, `any_true`, `bitand`, `bitmask`, `bitor`, `bitxor`, `blend`, `clamp`, `from_array`, `load`, `max`, `min`, `mul`, `neg`, `not`, `reduce_add`, `shl_const`, `shl_uniform`, `shr_arithmetic_const`, `shr_arithmetic_uniform`, `shr_logical_const`, `shr_logical_uniform`, `simd_eq`, `simd_ge`, `simd_gt`, `simd_le`, `simd_lt`, `simd_ne`, `splat`, `store`, `sub`, `to_array`, `zero`.

**[U32x4Backend](../magetypes/src/simd/backends/u32x4.rs), [U32x8Backend](../magetypes/src/simd/backends/u32x8.rs)**

`add`, `all_true`, `any_true`, `bitand`, `bitmask`, `bitor`, `bitxor`, `blend`, `clamp`, `from_array`, `load`, `max`, `min`, `mul`, `not`, `reduce_add`, `shl_const`, `shl_uniform`, `shr_logical_const`, `shr_logical_uniform`, `simd_eq`, `simd_ge`, `simd_gt`, `simd_le`, `simd_lt`, `simd_ne`, `splat`, `store`, `sub`, `to_array`, `zero`.

**[U32x4Bitcast](../magetypes/src/simd/backends/convert.rs), [U32x8Bitcast](../magetypes/src/simd/backends/convert.rs)**

`bitcast_i32_to_u32`, `bitcast_u32_to_i32`.

**[U64x2Backend](../magetypes/src/simd/backends/u64x2.rs), [U64x4Backend](../magetypes/src/simd/backends/u64x4.rs)**

`add`, `all_true`, `any_true`, `bitand`, `bitmask`, `bitor`, `bitxor`, `blend`, `clamp`, `from_array`, `load`, `max`, `min`, `not`, `reduce_add`, `shl_const`, `shr_logical_const`, `simd_eq`, `simd_ge`, `simd_gt`, `simd_le`, `simd_lt`, `simd_ne`, `splat`, `store`, `sub`, `to_array`, `zero`.

**[U64x2Bitcast](../magetypes/src/simd/backends/convert_int.rs), [U64x4Bitcast](../magetypes/src/simd/backends/convert_int.rs)**

`bitcast_i64_to_u64`, `bitcast_u64_to_i64`.

**[U64x8Backend](../magetypes/src/simd/backends/u64x8.rs)**

`add`, `all_true`, `any_true`, `bitand`, `bitmask`, `bitor`, `bitxor`, `blend`, `clamp`, `from_array`, `load`, `max`, `min`, `neg`, `not`, `reduce_add`, `shl_const`, `shr_arithmetic_const`, `shr_logical_const`, `simd_eq`, `simd_ge`, `simd_gt`, `simd_le`, `simd_lt`, `simd_ne`, `splat`, `store`, `sub`, `to_array`, `zero`.

**[U8x16Backend](../magetypes/src/simd/backends/u8x16.rs), [U8x32Backend](../magetypes/src/simd/backends/u8x32.rs)**

`abs_diff`, `add`, `all_true`, `any_true`, `bitand`, `bitmask`, `bitor`, `bitxor`, `blend`, `clamp`, `from_array`, `load`, `max`, `min`, `not`, `pairwise_widen_add`, `reduce_add`, `reduce_add_u32`, `saturating_add`, `saturating_sub`, `shl_const`, `shr_logical_const`, `simd_eq`, `simd_ge`, `simd_gt`, `simd_le`, `simd_lt`, `simd_ne`, `splat`, `store`, `sub`, `sum_abs_diff`, `to_array`, `widen_high_u8_to_u16`, `widen_low_u8_to_u16`, `zero`.

**[U8x64Backend](../magetypes/src/simd/backends/u8x64.rs)**

`abs_diff`, `add`, `all_true`, `any_true`, `bitand`, `bitmask`, `bitor`, `bitxor`, `blend`, `clamp`, `from_array`, `load`, `max`, `min`, `neg`, `not`, `pairwise_widen_add`, `reduce_add`, `reduce_add_u32`, `saturating_add`, `saturating_sub`, `shl_const`, `shr_arithmetic_const`, `shr_logical_const`, `simd_eq`, `simd_ge`, `simd_gt`, `simd_le`, `simd_lt`, `simd_ne`, `splat`, `store`, `sub`, `sum_abs_diff`, `to_array`, `widen_high_u8_to_u16`, `widen_low_u8_to_u16`, `zero`.

**[WidthDispatch](../magetypes/src/width.rs)**

`f32x16_load`, `f32x16_splat`, `f32x16_zero`, `f32x4_load`, `f32x4_splat`, `f32x4_zero`, `f32x8_load`, `f32x8_splat`, `f32x8_zero`, `f64x2_load`, `f64x2_splat`, `f64x2_zero`, `f64x4_load`, `f64x4_splat`, `f64x4_zero`, `f64x8_load`, `f64x8_splat`, `f64x8_zero`, `i16x16_load`, `i16x16_splat`, `i16x16_zero`, `i16x32_load`, `i16x32_splat`, `i16x32_zero`, `i16x8_load`, `i16x8_splat`, `i16x8_zero`, `i32x16_load`, `i32x16_splat`, `i32x16_zero`, `i32x4_load`, `i32x4_splat`, `i32x4_zero`, `i32x8_load`, `i32x8_splat`, `i32x8_zero`, `i64x2_load`, `i64x2_splat`, `i64x2_zero`, `i64x4_load`, `i64x4_splat`, `i64x4_zero`, `i64x8_load`, `i64x8_splat`, `i64x8_zero`, `i8x16_load`, `i8x16_splat`, `i8x16_zero`, `i8x32_load`, `i8x32_splat`, `i8x32_zero`, `i8x64_load`, `i8x64_splat`, `i8x64_zero`, `u16x16_load`, `u16x16_splat`, `u16x16_zero`, `u16x32_load`, `u16x32_splat`, `u16x32_zero`, `u16x8_load`, `u16x8_splat`, `u16x8_zero`, `u32x16_load`, `u32x16_splat`, `u32x16_zero`, `u32x4_load`, `u32x4_splat`, `u32x4_zero`, `u32x8_load`, `u32x8_splat`, `u32x8_zero`, `u64x2_load`, `u64x2_splat`, `u64x2_zero`, `u64x4_load`, `u64x4_splat`, `u64x4_zero`, `u64x8_load`, `u64x8_splat`, `u64x8_zero`, `u8x16_load`, `u8x16_splat`, `u8x16_zero`, `u8x32_load`, `u8x32_splat`, `u8x32_zero`, `u8x64_load`, `u8x64_splat`, `u8x64_zero`.

**[i8x64PopcntBackend](../magetypes/src/simd/backends/popcnt.rs), [u8x64PopcntBackend](../magetypes/src/simd/backends/popcnt.rs), [i16x32PopcntBackend](../magetypes/src/simd/backends/popcnt.rs), [u16x32PopcntBackend](../magetypes/src/simd/backends/popcnt.rs), [i32x16PopcntBackend](../magetypes/src/simd/backends/popcnt.rs), [u32x16PopcntBackend](../magetypes/src/simd/backends/popcnt.rs), [i64x8PopcntBackend](../magetypes/src/simd/backends/popcnt.rs), [u64x8PopcntBackend](../magetypes/src/simd/backends/popcnt.rs)**

`popcnt`.

## Tokenless calls

`from_raw(raw)` uses a compiler-checked feature context in the pending change. Ordinary vector operations such as arithmetic, comparisons, stores, and conversions on `self` reuse the token stored in that vector. Their corresponding backend trait operations still require a token receiver.

`from_i32`, `from_i32_bitcast`, `from_i32x4`, `from_i32x8`, `from_i32x16`, and `from_halves` currently request an additional token even though their vector arguments already carry the same token type. `partition_slice` and `partition_slice_mut` only split scalar slices into array chunks and do not use the token argument.

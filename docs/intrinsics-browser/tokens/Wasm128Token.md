# Wasm128Token — WASM SIMD128

Proof that WASM SIMD128 is available.

**Architecture:** wasm32 | **Features:** simd128
**Total intrinsics:** 302 (277 safe, 25 unsafe, 302 stable, 0 unstable/unknown)

## Usage

```rust
use archmage::prelude::*;

if let Some(token) = Wasm128Token::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: Wasm128Token, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(4) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: Wasm128Token, chunk: &mut [f32; 4]) {
    let v = v128_load(chunk);  // safe!
    let doubled = f32x4_add(v, v);  // value intrinsic (safe inside #[rite])
    v128_store(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```

## Safe Memory Operations (via import_intrinsics)

| Function | Safe Signature |
|----------|---------------|
| `i16x8_load_extend_i8x8` | `fn i16x8_load_extend_i8x8<T: Is8BytesUnaligned>(t: &T) -> v128` |
| `i16x8_load_extend_u8x8` | `fn i16x8_load_extend_u8x8<T: Is8BytesUnaligned>(t: &T) -> v128` |
| `i32x4_load_extend_i16x4` | `fn i32x4_load_extend_i16x4<T: Is8BytesUnaligned>(t: &T) -> v128` |
| `i32x4_load_extend_u16x4` | `fn i32x4_load_extend_u16x4<T: Is8BytesUnaligned>(t: &T) -> v128` |
| `i64x2_load_extend_i32x2` | `fn i64x2_load_extend_i32x2<T: Is8BytesUnaligned>(t: &T) -> v128` |
| `i64x2_load_extend_u32x2` | `fn i64x2_load_extend_u32x2<T: Is8BytesUnaligned>(t: &T) -> v128` |
| `u16x8_load_extend_u8x8` | `fn u16x8_load_extend_u8x8<T: Is8BytesUnaligned>(t: &T) -> v128` |
| `u32x4_load_extend_u16x4` | `fn u32x4_load_extend_u16x4<T: Is8BytesUnaligned>(t: &T) -> v128` |
| `u64x2_load_extend_u32x2` | `fn u64x2_load_extend_u32x2<T: Is8BytesUnaligned>(t: &T) -> v128` |
| `v128_load` | `fn v128_load<T: Is16BytesUnaligned>(t: &T) -> v128` |
| `v128_load16_splat` | `fn v128_load16_splat<T: Is2BytesUnaligned>(t: &T) -> v128` |
| `v128_load32_splat` | `fn v128_load32_splat<T: Is4BytesUnaligned>(t: &T) -> v128` |
| `v128_load32_zero` | `fn v128_load32_zero<T: Is4BytesUnaligned>(t: &T) -> v128` |
| `v128_load64_splat` | `fn v128_load64_splat<T: Is8BytesUnaligned>(t: &T) -> v128` |
| `v128_load64_zero` | `fn v128_load64_zero<T: Is8BytesUnaligned>(t: &T) -> v128` |
| `v128_load8_splat` | `fn v128_load8_splat<T: Is1ByteUnaligned>(t: &T) -> v128` |
| `v128_store` | `fn v128_store<T: Is16BytesUnaligned>(t: &mut T, v: v128) -> ()` |


## All Intrinsics

### Stable, Safe (277 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `f32x4_abs` | Calculates the absolute value of each lane of a 128-bit vect... |  | — |
| `f32x4_add` | Lane-wise addition of two 128-bit vectors interpreted as fou... |  | — |
| `f32x4_ceil` | Lane-wise rounding to the nearest integral value not smaller... |  | — |
| `f32x4_convert_i32x4` | Converts a 128-bit vector interpreted as four 32-bit signed ... |  | — |
| `f32x4_convert_u32x4` | Converts a 128-bit vector interpreted as four 32-bit unsigne... |  | — |
| `f32x4_demote_f64x2_zero` | Conversion of the two double-precision floating point lanes ... |  | — |
| `f32x4_div` | Lane-wise division of two 128-bit vectors interpreted as fou... |  | — |
| `f32x4_eq` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `f32x4_extract_lane` | Extracts a lane from a 128-bit vector interpreted as 4 packe... |  | — |
| `f32x4_floor` | Lane-wise rounding to the nearest integral value not greater... |  | — |
| `f32x4_ge` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `f32x4_gt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `f32x4_le` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `f32x4_lt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `f32x4_max` | Calculates the lane-wise minimum of two 128-bit vectors inte... |  | — |
| `f32x4_min` | Calculates the lane-wise minimum of two 128-bit vectors inte... |  | — |
| `f32x4_mul` | Lane-wise multiplication of two 128-bit vectors interpreted ... |  | — |
| `f32x4_ne` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `f32x4_nearest` | Lane-wise rounding to the nearest integral value; if two val... |  | — |
| `f32x4_neg` | Negates each lane of a 128-bit vector interpreted as four 32... |  | — |
| `f32x4_pmax` | Lane-wise maximum value, defined as `a < b ? b : a` |  | — |
| `f32x4_pmin` | Lane-wise minimum value, defined as `b < a ? b : a` |  | — |
| `f32x4_replace_lane` | Replaces a lane from a 128-bit vector interpreted as 4 packe... |  | — |
| `f32x4_splat` | Creates a vector with identical lanes.  Constructs a vector ... |  | — |
| `f32x4_sqrt` | Calculates the square root of each lane of a 128-bit vector ... |  | — |
| `f32x4_sub` | Lane-wise subtraction of two 128-bit vectors interpreted as ... |  | — |
| `f32x4_trunc` | Lane-wise rounding to the nearest integral value with the ma... |  | — |
| `f64x2_abs` | Calculates the absolute value of each lane of a 128-bit vect... |  | — |
| `f64x2_add` | Lane-wise add of two 128-bit vectors interpreted as two 64-b... |  | — |
| `f64x2_ceil` | Lane-wise rounding to the nearest integral value not smaller... |  | — |
| `f64x2_convert_low_i32x4` | Lane-wise conversion from integer to floating point |  | — |
| `f64x2_convert_low_u32x4` | Lane-wise conversion from integer to floating point |  | — |
| `f64x2_div` | Lane-wise divide of two 128-bit vectors interpreted as two 6... |  | — |
| `f64x2_eq` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `f64x2_extract_lane` | Extracts a lane from a 128-bit vector interpreted as 2 packe... |  | — |
| `f64x2_floor` | Lane-wise rounding to the nearest integral value not greater... |  | — |
| `f64x2_ge` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `f64x2_gt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `f64x2_le` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `f64x2_lt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `f64x2_max` | Calculates the lane-wise maximum of two 128-bit vectors inte... |  | — |
| `f64x2_min` | Calculates the lane-wise minimum of two 128-bit vectors inte... |  | — |
| `f64x2_mul` | Lane-wise multiply of two 128-bit vectors interpreted as two... |  | — |
| `f64x2_ne` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `f64x2_nearest` | Lane-wise rounding to the nearest integral value; if two val... |  | — |
| `f64x2_neg` | Negates each lane of a 128-bit vector interpreted as two 64-... |  | — |
| `f64x2_pmax` | Lane-wise maximum value, defined as `a < b ? b : a` |  | — |
| `f64x2_pmin` | Lane-wise minimum value, defined as `b < a ? b : a` |  | — |
| `f64x2_promote_low_f32x4` | Conversion of the two lower single-precision floating point ... |  | — |
| `f64x2_replace_lane` | Replaces a lane from a 128-bit vector interpreted as 2 packe... |  | — |
| `f64x2_splat` | Creates a vector with identical lanes.  Constructs a vector ... |  | — |
| `f64x2_sqrt` | Calculates the square root of each lane of a 128-bit vector ... |  | — |
| `f64x2_sub` | Lane-wise subtract of two 128-bit vectors interpreted as two... |  | — |
| `f64x2_trunc` | Lane-wise rounding to the nearest integral value with the ma... |  | — |
| `i16x8_abs` | Lane-wise wrapping absolute value |  | — |
| `i16x8_add` | Adds two 128-bit vectors as if they were two packed eight 16... |  | — |
| `i16x8_add_sat` | Adds two 128-bit vectors as if they were two packed eight 16... |  | — |
| `i16x8_all_true` | Returns true if all lanes are non-zero, false otherwise |  | — |
| `i16x8_bitmask` | Extracts the high bit for each lane in `a` and produce a sca... |  | — |
| `i16x8_eq` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i16x8_extadd_pairwise_i8x16` | Integer extended pairwise addition producing extended result... |  | — |
| `i16x8_extadd_pairwise_u8x16` | Integer extended pairwise addition producing extended result... |  | — |
| `i16x8_extend_high_i8x16` | Converts high half of the smaller lane vector to a larger la... |  | — |
| `i16x8_extend_high_u8x16` | Converts high half of the smaller lane vector to a larger la... |  | — |
| `i16x8_extend_low_i8x16` | Converts low half of the smaller lane vector to a larger lan... |  | — |
| `i16x8_extend_low_u8x16` | Converts low half of the smaller lane vector to a larger lan... |  | — |
| `i16x8_extmul_high_i8x16` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `i16x8_extmul_high_u8x16` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `i16x8_extmul_low_i8x16` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `i16x8_extmul_low_u8x16` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `i16x8_extract_lane` | Extracts a lane from a 128-bit vector interpreted as 8 packe... |  | — |
| `i16x8_ge` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i16x8_gt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i16x8_le` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i16x8_lt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i16x8_max` | Compares lane-wise signed integers, and returns the maximum ... |  | — |
| `i16x8_min` | Compares lane-wise signed integers, and returns the minimum ... |  | — |
| `i16x8_mul` | Multiplies two 128-bit vectors as if they were two packed ei... |  | — |
| `i16x8_narrow_i32x4` | Converts two input vectors into a smaller lane vector by nar... |  | — |
| `i16x8_ne` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i16x8_neg` | Negates a 128-bit vectors interpreted as eight 16-bit signed... |  | — |
| `i16x8_q15mulr_sat` | Lane-wise saturating rounding multiplication in Q15 format |  | — |
| `i16x8_replace_lane` | Replaces a lane from a 128-bit vector interpreted as 8 packe... |  | — |
| `i16x8_shl` | Shifts each lane to the left by the specified number of bits... |  | — |
| `i16x8_shr` | Shifts each lane to the right by the specified number of bit... |  | — |
| `i16x8_shuffle` |  |  | — |
| `i16x8_splat` | Creates a vector with identical lanes.  Construct a vector w... |  | — |
| `i16x8_sub` | Subtracts two 128-bit vectors as if they were two packed eig... |  | — |
| `i16x8_sub_sat` | Subtracts two 128-bit vectors as if they were two packed eig... |  | — |
| `i32x4_abs` | Lane-wise wrapping absolute value |  | — |
| `i32x4_add` | Adds two 128-bit vectors as if they were two packed four 32-... |  | — |
| `i32x4_all_true` | Returns true if all lanes are non-zero, false otherwise |  | — |
| `i32x4_bitmask` | Extracts the high bit for each lane in `a` and produce a sca... |  | — |
| `i32x4_dot_i16x8` | Lane-wise multiply signed 16-bit integers in the two input v... |  | — |
| `i32x4_eq` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i32x4_extadd_pairwise_i16x8` | Integer extended pairwise addition producing extended result... |  | — |
| `i32x4_extadd_pairwise_u16x8` |  |  | — |
| `i32x4_extend_high_i16x8` | Converts high half of the smaller lane vector to a larger la... |  | — |
| `i32x4_extend_high_u16x8` | Converts high half of the smaller lane vector to a larger la... |  | — |
| `i32x4_extend_low_i16x8` | Converts low half of the smaller lane vector to a larger lan... |  | — |
| `i32x4_extend_low_u16x8` | Converts low half of the smaller lane vector to a larger lan... |  | — |
| `i32x4_extmul_high_i16x8` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `i32x4_extmul_high_u16x8` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `i32x4_extmul_low_i16x8` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `i32x4_extmul_low_u16x8` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `i32x4_extract_lane` | Extracts a lane from a 128-bit vector interpreted as 4 packe... |  | — |
| `i32x4_ge` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i32x4_gt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i32x4_le` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i32x4_lt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i32x4_max` | Compares lane-wise signed integers, and returns the maximum ... |  | — |
| `i32x4_min` | Compares lane-wise signed integers, and returns the minimum ... |  | — |
| `i32x4_mul` | Multiplies two 128-bit vectors as if they were two packed fo... |  | — |
| `i32x4_ne` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i32x4_neg` | Negates a 128-bit vectors interpreted as four 32-bit signed ... |  | — |
| `i32x4_replace_lane` | Replaces a lane from a 128-bit vector interpreted as 4 packe... |  | — |
| `i32x4_shl` | Shifts each lane to the left by the specified number of bits... |  | — |
| `i32x4_shr` | Shifts each lane to the right by the specified number of bit... |  | — |
| `i32x4_shuffle` | Same as |  | — |
| `i32x4_splat` | Creates a vector with identical lanes.  Constructs a vector ... |  | — |
| `i32x4_sub` | Subtracts two 128-bit vectors as if they were two packed fou... |  | — |
| `i32x4_trunc_sat_f32x4` | Converts a 128-bit vector interpreted as four 32-bit floatin... |  | — |
| `i32x4_trunc_sat_f64x2_zero` | Saturating conversion of the two double-precision floating p... |  | — |
| `i64x2_abs` | Lane-wise wrapping absolute value |  | — |
| `i64x2_add` | Adds two 128-bit vectors as if they were two packed two 64-b... |  | — |
| `i64x2_all_true` | Returns true if all lanes are non-zero, false otherwise |  | — |
| `i64x2_bitmask` | Extracts the high bit for each lane in `a` and produce a sca... |  | — |
| `i64x2_eq` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i64x2_extend_high_i32x4` | Converts high half of the smaller lane vector to a larger la... |  | — |
| `i64x2_extend_high_u32x4` | Converts high half of the smaller lane vector to a larger la... |  | — |
| `i64x2_extend_low_i32x4` | Converts low half of the smaller lane vector to a larger lan... |  | — |
| `i64x2_extend_low_u32x4` | Converts low half of the smaller lane vector to a larger lan... |  | — |
| `i64x2_extmul_high_i32x4` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `i64x2_extmul_high_u32x4` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `i64x2_extmul_low_i32x4` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `i64x2_extmul_low_u32x4` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `i64x2_extract_lane` | Extracts a lane from a 128-bit vector interpreted as 2 packe... |  | — |
| `i64x2_ge` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i64x2_gt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i64x2_le` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i64x2_lt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i64x2_mul` | Multiplies two 128-bit vectors as if they were two packed tw... |  | — |
| `i64x2_ne` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i64x2_neg` | Negates a 128-bit vectors interpreted as two 64-bit signed i... |  | — |
| `i64x2_replace_lane` | Replaces a lane from a 128-bit vector interpreted as 2 packe... |  | — |
| `i64x2_shl` | Shifts each lane to the left by the specified number of bits... |  | — |
| `i64x2_shr` | Shifts each lane to the right by the specified number of bit... |  | — |
| `i64x2_shuffle` | Same as |  | — |
| `i64x2_splat` | Creates a vector with identical lanes.  Construct a vector w... |  | — |
| `i64x2_sub` | Subtracts two 128-bit vectors as if they were two packed two... |  | — |
| `i8x16_abs` | Lane-wise wrapping absolute value |  | — |
| `i8x16_add` | Adds two 128-bit vectors as if they were two packed sixteen ... |  | — |
| `i8x16_add_sat` | Adds two 128-bit vectors as if they were two packed sixteen ... |  | — |
| `i8x16_all_true` | Returns true if all lanes are non-zero, false otherwise |  | — |
| `i8x16_bitmask` | Extracts the high bit for each lane in `a` and produce a sca... |  | — |
| `i8x16_eq` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i8x16_extract_lane` | Extracts a lane from a 128-bit vector interpreted as 16 pack... |  | — |
| `i8x16_ge` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i8x16_gt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i8x16_le` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i8x16_lt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i8x16_max` | Compares lane-wise signed integers, and returns the maximum ... |  | — |
| `i8x16_min` | Compares lane-wise signed integers, and returns the minimum ... |  | — |
| `i8x16_narrow_i16x8` | Converts two input vectors into a smaller lane vector by nar... |  | — |
| `i8x16_ne` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `i8x16_neg` | Negates a 128-bit vectors interpreted as sixteen 8-bit signe... |  | — |
| `i8x16_popcnt` | Count the number of bits set to one within each lane |  | — |
| `i8x16_replace_lane` | Replaces a lane from a 128-bit vector interpreted as 16 pack... |  | — |
| `i8x16_shl` | Shifts each lane to the left by the specified number of bits... |  | — |
| `i8x16_shr` | Shifts each lane to the right by the specified number of bit... |  | — |
| `i8x16_shuffle` |  |  | — |
| `i8x16_splat` | Creates a vector with identical lanes.  Constructs a vector ... |  | — |
| `i8x16_sub` | Subtracts two 128-bit vectors as if they were two packed six... |  | — |
| `i8x16_sub_sat` | Subtracts two 128-bit vectors as if they were two packed six... |  | — |
| `i8x16_swizzle` | Returns a new vector with lanes selected from the lanes of t... |  | — |
| `u16x8_add` | Adds two 128-bit vectors as if they were two packed eight 16... |  | — |
| `u16x8_add_sat` | Adds two 128-bit vectors as if they were two packed eight 16... |  | — |
| `u16x8_all_true` | Returns true if all lanes are non-zero, false otherwise |  | — |
| `u16x8_avgr` | Lane-wise rounding average |  | — |
| `u16x8_bitmask` | Extracts the high bit for each lane in `a` and produce a sca... |  | — |
| `u16x8_eq` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u16x8_extadd_pairwise_u8x16` | Integer extended pairwise addition producing extended result... |  | — |
| `u16x8_extend_high_u8x16` | Converts high half of the smaller lane vector to a larger la... |  | — |
| `u16x8_extend_low_u8x16` | Converts low half of the smaller lane vector to a larger lan... |  | — |
| `u16x8_extmul_high_u8x16` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `u16x8_extmul_low_u8x16` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `u16x8_extract_lane` | Extracts a lane from a 128-bit vector interpreted as 8 packe... |  | — |
| `u16x8_ge` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u16x8_gt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u16x8_le` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u16x8_lt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u16x8_max` | Compares lane-wise unsigned integers, and returns the maximu... |  | — |
| `u16x8_min` | Compares lane-wise unsigned integers, and returns the minimu... |  | — |
| `u16x8_mul` | Multiplies two 128-bit vectors as if they were two packed ei... |  | — |
| `u16x8_narrow_i32x4` | Converts two input vectors into a smaller lane vector by nar... |  | — |
| `u16x8_ne` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u16x8_replace_lane` | Replaces a lane from a 128-bit vector interpreted as 8 packe... |  | — |
| `u16x8_shl` | Shifts each lane to the left by the specified number of bits... |  | — |
| `u16x8_shr` | Shifts each lane to the right by the specified number of bit... |  | — |
| `u16x8_shuffle` |  |  | — |
| `u16x8_splat` | Creates a vector with identical lanes.  Construct a vector w... |  | — |
| `u16x8_sub` | Subtracts two 128-bit vectors as if they were two packed eig... |  | — |
| `u16x8_sub_sat` | Subtracts two 128-bit vectors as if they were two packed eig... |  | — |
| `u32x4_add` | Adds two 128-bit vectors as if they were two packed four 32-... |  | — |
| `u32x4_all_true` | Returns true if all lanes are non-zero, false otherwise |  | — |
| `u32x4_bitmask` | Extracts the high bit for each lane in `a` and produce a sca... |  | — |
| `u32x4_eq` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u32x4_extadd_pairwise_u16x8` |  |  | — |
| `u32x4_extend_high_u16x8` | Converts high half of the smaller lane vector to a larger la... |  | — |
| `u32x4_extend_low_u16x8` | Converts low half of the smaller lane vector to a larger lan... |  | — |
| `u32x4_extmul_high_u16x8` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `u32x4_extmul_low_u16x8` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `u32x4_extract_lane` | Extracts a lane from a 128-bit vector interpreted as 4 packe... |  | — |
| `u32x4_ge` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u32x4_gt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u32x4_le` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u32x4_lt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u32x4_max` | Compares lane-wise unsigned integers, and returns the maximu... |  | — |
| `u32x4_min` | Compares lane-wise unsigned integers, and returns the minimu... |  | — |
| `u32x4_mul` | Multiplies two 128-bit vectors as if they were two packed fo... |  | — |
| `u32x4_ne` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u32x4_replace_lane` | Replaces a lane from a 128-bit vector interpreted as 4 packe... |  | — |
| `u32x4_shl` | Shifts each lane to the left by the specified number of bits... |  | — |
| `u32x4_shr` | Shifts each lane to the right by the specified number of bit... |  | — |
| `u32x4_shuffle` | Same as |  | — |
| `u32x4_splat` | Creates a vector with identical lanes.  Constructs a vector ... |  | — |
| `u32x4_sub` | Subtracts two 128-bit vectors as if they were two packed fou... |  | — |
| `u32x4_trunc_sat_f32x4` | Converts a 128-bit vector interpreted as four 32-bit floatin... |  | — |
| `u32x4_trunc_sat_f64x2_zero` | Saturating conversion of the two double-precision floating p... |  | — |
| `u64x2_add` | Adds two 128-bit vectors as if they were two packed two 64-b... |  | — |
| `u64x2_all_true` | Returns true if all lanes are non-zero, false otherwise |  | — |
| `u64x2_bitmask` | Extracts the high bit for each lane in `a` and produce a sca... |  | — |
| `u64x2_eq` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u64x2_extend_high_u32x4` | Converts high half of the smaller lane vector to a larger la... |  | — |
| `u64x2_extend_low_u32x4` | Converts low half of the smaller lane vector to a larger lan... |  | — |
| `u64x2_extmul_high_u32x4` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `u64x2_extmul_low_u32x4` | Lane-wise integer extended multiplication producing twice wi... |  | — |
| `u64x2_extract_lane` | Extracts a lane from a 128-bit vector interpreted as 2 packe... |  | — |
| `u64x2_mul` | Multiplies two 128-bit vectors as if they were two packed tw... |  | — |
| `u64x2_ne` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u64x2_replace_lane` | Replaces a lane from a 128-bit vector interpreted as 2 packe... |  | — |
| `u64x2_shl` | Shifts each lane to the left by the specified number of bits... |  | — |
| `u64x2_shr` | Shifts each lane to the right by the specified number of bit... |  | — |
| `u64x2_shuffle` | Same as |  | — |
| `u64x2_splat` | Creates a vector with identical lanes.  Construct a vector w... |  | — |
| `u64x2_sub` | Subtracts two 128-bit vectors as if they were two packed two... |  | — |
| `u8x16_add` | Adds two 128-bit vectors as if they were two packed sixteen ... |  | — |
| `u8x16_add_sat` | Adds two 128-bit vectors as if they were two packed sixteen ... |  | — |
| `u8x16_all_true` | Returns true if all lanes are non-zero, false otherwise |  | — |
| `u8x16_avgr` | Lane-wise rounding average |  | — |
| `u8x16_bitmask` | Extracts the high bit for each lane in `a` and produce a sca... |  | — |
| `u8x16_eq` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u8x16_extract_lane` | Extracts a lane from a 128-bit vector interpreted as 16 pack... |  | — |
| `u8x16_ge` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u8x16_gt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u8x16_le` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u8x16_lt` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u8x16_max` | Compares lane-wise unsigned integers, and returns the maximu... |  | — |
| `u8x16_min` | Compares lane-wise unsigned integers, and returns the minimu... |  | — |
| `u8x16_narrow_i16x8` | Converts two input vectors into a smaller lane vector by nar... |  | — |
| `u8x16_ne` | Compares two 128-bit vectors as if they were two vectors of ... |  | — |
| `u8x16_popcnt` | Count the number of bits set to one within each lane |  | — |
| `u8x16_replace_lane` | Replaces a lane from a 128-bit vector interpreted as 16 pack... |  | — |
| `u8x16_shl` | Shifts each lane to the left by the specified number of bits... |  | — |
| `u8x16_shr` | Shifts each lane to the right by the specified number of bit... |  | — |
| `u8x16_shuffle` |  |  | — |
| `u8x16_splat` | Creates a vector with identical lanes.  Constructs a vector ... |  | — |
| `u8x16_sub` | Subtracts two 128-bit vectors as if they were two packed six... |  | — |
| `u8x16_sub_sat` | Subtracts two 128-bit vectors as if they were two packed six... |  | — |
| `u8x16_swizzle` | Returns a new vector with lanes selected from the lanes of t... |  | — |
| `v128_and` | Performs a bitwise and of the two input 128-bit vectors, ret... |  | — |
| `v128_andnot` | Bitwise AND of bits of `a` and the logical inverse of bits o... |  | — |
| `v128_any_true` | Returns `true` if any bit in `a` is set, or `false` otherwis... |  | — |
| `v128_bitselect` | Use the bitmask in `c` to select bits from `v1` when 1 and `... |  | — |
| `v128_not` | Flips each bit of the 128-bit input vector |  | — |
| `v128_or` | Performs a bitwise or of the two input 128-bit vectors, retu... |  | — |
| `v128_xor` | Performs a bitwise xor of the two input 128-bit vectors, ret... |  | — |

### Stable, Unsafe (25 intrinsics) — use import_intrinsics for safe versions

| Name | Description | Safe Variant |
|------|-------------|--------------|
| `i16x8_load_extend_i8x8` | Load eight 8-bit integers and sign extend each one to a 16-b... | `i16x8_load_extend_i8x8` (safe via import_intrinsics) |
| `i16x8_load_extend_u8x8` | Load eight 8-bit integers and zero extend each one to a 16-b... | `i16x8_load_extend_u8x8` (safe via import_intrinsics) |
| `i32x4_load_extend_i16x4` | Load four 16-bit integers and sign extend each one to a 32-b... | `i32x4_load_extend_i16x4` (safe via import_intrinsics) |
| `i32x4_load_extend_u16x4` | Load four 16-bit integers and zero extend each one to a 32-b... | `i32x4_load_extend_u16x4` (safe via import_intrinsics) |
| `i64x2_load_extend_i32x2` | Load two 32-bit integers and sign extend each one to a 64-bi... | `i64x2_load_extend_i32x2` (safe via import_intrinsics) |
| `i64x2_load_extend_u32x2` | Load two 32-bit integers and zero extend each one to a 64-bi... | `i64x2_load_extend_u32x2` (safe via import_intrinsics) |
| `u16x8_load_extend_u8x8` | Load eight 8-bit integers and zero extend each one to a 16-b... | `u16x8_load_extend_u8x8` (safe via import_intrinsics) |
| `u32x4_load_extend_u16x4` | Load four 16-bit integers and zero extend each one to a 32-b... | `u32x4_load_extend_u16x4` (safe via import_intrinsics) |
| `u64x2_load_extend_u32x2` | Load two 32-bit integers and zero extend each one to a 64-bi... | `u64x2_load_extend_u32x2` (safe via import_intrinsics) |
| `v128_load` | Loads a `v128` vector from the given heap address.  This int... | `v128_load` (safe via import_intrinsics) |
| `v128_load16_lane` | Loads a 16-bit value from `m` and sets lane `L` of `v` to th... | — |
| `v128_load16_splat` | Load a single element and splat to all lanes of a v128 vecto... | `v128_load16_splat` (safe via import_intrinsics) |
| `v128_load32_lane` | Loads a 32-bit value from `m` and sets lane `L` of `v` to th... | — |
| `v128_load32_splat` | Load a single element and splat to all lanes of a v128 vecto... | `v128_load32_splat` (safe via import_intrinsics) |
| `v128_load32_zero` | Load a 32-bit element into the low bits of the vector and se... | `v128_load32_zero` (safe via import_intrinsics) |
| `v128_load64_lane` | Loads a 64-bit value from `m` and sets lane `L` of `v` to th... | — |
| `v128_load64_splat` | Load a single element and splat to all lanes of a v128 vecto... | `v128_load64_splat` (safe via import_intrinsics) |
| `v128_load64_zero` | Load a 64-bit element into the low bits of the vector and se... | `v128_load64_zero` (safe via import_intrinsics) |
| `v128_load8_lane` | Loads an 8-bit value from `m` and sets lane `L` of `v` to th... | — |
| `v128_load8_splat` | Load a single element and splat to all lanes of a v128 vecto... | `v128_load8_splat` (safe via import_intrinsics) |
| `v128_store` | Stores a `v128` vector to the given heap address.  This intr... | `v128_store` (safe via import_intrinsics) |
| `v128_store16_lane` | Stores the 16-bit value from lane `L` of `v` into `m`  This ... | — |
| `v128_store32_lane` | Stores the 32-bit value from lane `L` of `v` into `m`  This ... | — |
| `v128_store64_lane` | Stores the 64-bit value from lane `L` of `v` into `m`  This ... | — |
| `v128_store8_lane` | Stores the 8-bit value from lane `L` of `v` into `m`  This i... | — |



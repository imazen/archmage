+++
title = "Float / Integer"
weight = 1
+++

Convert between floating-point and integer vector types.

## Float to Integer

### Truncate (toward zero)

Behaves like `as i32` in Rust — drops the fractional part:

```rust
let floats = f32x8::from_array(token, [1.5, 2.7, -3.2, 4.0, 5.9, 6.1, 7.0, 8.5]);

let ints = floats.to_i32x8();
// [1, 2, -3, 4, 5, 6, 7, 8]
```

### Round to nearest

Rounds to the nearest integer (banker's rounding — ties go to even):

```rust
let rounded = floats.to_i32x8_round();
// [2, 3, -3, 4, 6, 6, 7, 8]
```

## Integer to Float

```rust
let ints = i32x8::from_array(token, [1, 2, 3, 4, 5, 6, 7, 8]);
let floats = ints.to_f32x8();
// [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

For values that don't fit exactly in f32 (integers above 2^24), the result is the nearest representable float.

## 128-bit Variants

The same methods exist on 128-bit types:

```rust
let floats = f32x4::from_array(token, [1.5, 2.7, -3.2, 4.0]);
let ints = floats.to_i32x4();        // Truncate
let rounded = floats.to_i32x4_round(); // Round

let back = ints.to_f32x4();
```

## Lane Access

Vectors implement `Index<usize>` and `IndexMut<usize>` for single-lane access:

```rust
let v = f32x8::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

let third = v[2];   // 3.0

let mut v = v;
v[2] = 99.0;        // [1.0, 2.0, 99.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

Lane indices are runtime values with bounds checking — out-of-bounds panics.

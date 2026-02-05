# Construction & Operators

magetypes provides natural Rust syntax for SIMD operations.

## Construction

### From Array

```rust
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let v = f32x8::from_array(token, data);
```

### From Slice

```rust
let slice = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let v = f32x8::from_slice(token, slice);
```

### Splat (Broadcast)

```rust
let v = f32x8::splat(token, 3.14159);  // All lanes = π
```

### Zero

```rust
let v = f32x8::zero(token);  // All lanes = 0
```

### Load from Memory

```rust
// Unaligned load
let v = f32x8::load(token, ptr);

// From reference
let v = f32x8::from_array(token, *array_ref);
```

## Extraction

### To Array

```rust
let arr: [f32; 8] = v.to_array();
```

### Store to Memory

```rust
v.store(ptr);          // Unaligned store
v.store_aligned(ptr);  // Aligned store (UB if misaligned)
```

### Extract Single Lane

```rust
let first = v.extract::<0>();
let third = v.extract::<2>();
```

## Arithmetic Operators

All standard operators work:

```rust
let a = f32x8::splat(token, 2.0);
let b = f32x8::splat(token, 3.0);

let sum = a + b;        // [5.0; 8]
let diff = a - b;       // [-1.0; 8]
let prod = a * b;       // [6.0; 8]
let quot = a / b;       // [0.666...; 8]
let neg = -a;           // [-2.0; 8]
```

### Compound Assignment

```rust
let mut v = f32x8::splat(token, 1.0);
v += f32x8::splat(token, 2.0);  // v = [3.0; 8]
v *= f32x8::splat(token, 2.0);  // v = [6.0; 8]
```

## Fused Multiply-Add

FMA is faster and more precise than separate multiply and add:

```rust
// a * b + c (single instruction on AVX2/NEON)
let result = a.mul_add(b, c);

// a * b - c
let result = a.mul_sub(b, c);

// -(a * b) + c  (negated multiply-add)
let result = a.neg_mul_add(b, c);
```

## Comparisons

Comparisons return mask types:

```rust
let a = f32x8::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
let b = f32x8::splat(token, 4.0);

let lt = a.simd_lt(b);   // [true, true, true, false, false, false, false, false]
let eq = a.simd_eq(b);   // [false, false, false, true, false, false, false, false]
let ge = a.simd_ge(b);   // [false, false, false, true, true, true, true, true]
```

### Blend with Mask

```rust
let mask = a.simd_lt(b);
let result = mask.blend(true_values, false_values);
```

## Min/Max

```rust
let min = a.min(b);  // Element-wise minimum
let max = a.max(b);  // Element-wise maximum

// With scalar
let clamped = v.max(f32x8::splat(token, 0.0))
               .min(f32x8::splat(token, 1.0));
```

## Absolute Value

```rust
let abs = v.abs();  // |v| for each lane
```

## Reductions

Horizontal operations across lanes:

```rust
let sum = v.reduce_add();      // Sum of all lanes
let max = v.reduce_max();      // Maximum lane
let min = v.reduce_min();      // Minimum lane
```

## Integer Operations

For integer types (`i32x8`, `u8x16`, etc.):

```rust
let a = i32x8::splat(token, 10);
let b = i32x8::splat(token, 3);

// Arithmetic
let sum = a + b;
let diff = a - b;
let prod = a * b;

// Bitwise
let and = a & b;
let or = a | b;
let xor = a ^ b;
let not = !a;

// Shifts
let shl = a << 2;           // Shift left by constant
let shr = a >> 1;           // Shift right by constant
let shr_arith = a.shr_arithmetic(1);  // Sign-extending shift
```

## Example: Dot Product

```rust
use archmage::{Desktop64, SimdToken, arcane};
use magetypes::simd::f32x8;

#[arcane]
fn dot_product(token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = f32x8::from_array(token, *a);
    let vb = f32x8::from_array(token, *b);
    (va * vb).reduce_add()
}
```

## Example: Vector Normalization

```rust
#[arcane]
fn normalize(token: Desktop64, v: &mut [f32; 8]) {
    let vec = f32x8::from_array(token, *v);
    let len_sq = (vec * vec).reduce_add();
    let len = len_sq.sqrt();

    if len > 0.0 {
        let inv_len = f32x8::splat(token, 1.0 / len);
        let normalized = vec * inv_len;
        *v = normalized.to_array();
    }
}
```

+++
title = "Construction & Extraction"
weight = 1
+++

All construction methods take a token as the first argument. Once you have a vector, extraction methods don't need the token.

Constructor calls are always turbofished with the type parameter: `f32x8::<T>::splat(token, 1.0)`. The `T` is resolved by the function's generic bound.

## Construction

### From Array

```rust
use magetypes::simd::{generic::f32x8, backends::F32x8Backend};

// given a token: T where T: F32x8Backend
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let v = f32x8::<T>::from_array(token, data);
```

### From Slice

```rust
// given a token: T where T: F32x8Backend
let slice = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
let v = f32x8::<T>::from_slice(token, slice);
```

The slice must have at least as many elements as the vector width.

### Splat (Broadcast)

Fill every lane with the same value:

```rust
// given a token: T where T: F32x8Backend
let v = f32x8::<T>::splat(token, 3.14159);  // All 8 lanes = pi
```

### Zero

```rust
// given a token: T where T: F32x8Backend
let v = f32x8::<T>::zero(token);  // All lanes = 0.0
```

### Load from Array Reference

```rust
// given a token: T where T: F32x8Backend, data: &[f32; 8]
let v = f32x8::<T>::load(token, data);
```

`load` takes a reference to a fixed-size array, not a raw pointer. This is safe by design.

## Extraction

### To Array

```rust
let arr: [f32; 8] = v.to_array();
```

### Store to Array

```rust
let mut buf = [0.0f32; 8];
v.store(&mut buf);  // Takes &mut [f32; 8]
```

### Access Single Lane

Vectors implement `Index<usize>` and `IndexMut<usize>`:

```rust
let first = v[0];   // Read lane 0
let third = v[2];   // Read lane 2

v[2] = 99.0;        // Write lane 2
```

Lane access is runtime-indexed with bounds checking.

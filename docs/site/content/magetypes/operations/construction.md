+++
title = "Construction & Extraction"
weight = 1
+++

All construction methods take a token as the first argument. Once you have a vector, extraction methods don't need the token.

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

The slice must have at least as many elements as the vector width.

### Splat (Broadcast)

Fill every lane with the same value:

```rust
let v = f32x8::splat(token, 3.14159);  // All 8 lanes = pi
```

### Zero

```rust
let v = f32x8::zero(token);  // All lanes = 0.0
```

### Load from Array Reference

```rust
// Load from fixed-size array reference
let v = f32x8::load(token, &data);  // data: &[f32; 8]
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

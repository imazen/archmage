+++
title = "Load & Store"
weight = 1
+++

Moving data between memory and SIMD registers. For most cases, `from_array`/`from_slice` and `to_array`/`store` are what you want.

## Unaligned Load

The default and recommended approach. Modern CPUs handle unaligned access with minimal or no penalty:

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

fn example<T: F32x8Backend>(token: T) {
    // From an array
    let arr = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let v = f32x8::<T>::from_array(token, arr);

    // From a slice (must have enough elements)
    let slice = &[1.0f32; 16];
    let v = f32x8::<T>::from_slice(token, &slice[0..8]);
}
```

## Aligned Load

If you know your data is aligned to the vector width (32 bytes for `f32x8`), you can use the aligned variant:

```rust
// Aligned load — UB if pointer is not aligned to 32 bytes
let v = unsafe { f32x8::<T>::load_aligned(ptr) };
```

In practice, the performance difference between aligned and unaligned loads is negligible on modern CPUs (Haswell+, all ARM Cortex-A). Prefer unaligned loads unless profiling says otherwise.

## Unaligned Store

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

fn example<T: F32x8Backend>(token: T) {
    let v = f32x8::<T>::splat(token, 42.0);

    // To an array
    let arr: [f32; 8] = v.to_array();

    // Store to a mutable array reference
    let mut buf = [0.0f32; 8];
    v.store(&mut buf);
}
```

## Aligned Store

```rust
// Aligned store — UB if pointer is not aligned
unsafe { v.store_aligned(ptr) };
```

## Streaming Stores

Non-temporal stores bypass the cache hierarchy. Use for large sequential writes where the data won't be read back soon:

```rust
// Non-temporal store (bypasses cache)
unsafe { v.stream(ptr) };
```

Streaming stores are useful when:
- Writing large arrays sequentially (image processing, audio buffers)
- Data won't be read again in the near future
- You want to avoid evicting useful data from cache

They are counterproductive when:
- The buffer fits in cache and will be read soon
- Access is random rather than sequential
- The write volume is small

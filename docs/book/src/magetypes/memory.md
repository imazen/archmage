# Memory Operations

Efficiently moving data between memory and SIMD registers is critical for performance.

## Load Operations

### Unaligned Load

```rust
use magetypes::f32x8;

// From array reference
let arr = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let v = f32x8::from_array(token, arr);

// From slice (must have enough elements)
let slice = &[1.0f32; 16];
let v = f32x8::from_slice(token, &slice[0..8]);
```

### Aligned Load

If you know your data is aligned:

```rust
// Aligned load (UB if not aligned to 32 bytes for f32x8)
let v = unsafe { f32x8::load_aligned(ptr) };
```

### Partial Load

Load fewer elements than the vector width:

```rust
// Load 4 elements into lower half, zero upper half
let v = f32x8::load_low(token, &[1.0, 2.0, 3.0, 4.0]);
// v = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
```

## Store Operations

### Unaligned Store

```rust
let v = f32x8::splat(token, 42.0);

// To array
let arr: [f32; 8] = v.to_array();

// To slice
let mut buf = [0.0f32; 8];
v.store_slice(&mut buf);
```

### Aligned Store

```rust
// Aligned store (UB if not aligned)
unsafe { v.store_aligned(ptr) };
```

### Partial Store

Store only some elements:

```rust
// Store lower 4 elements
v.store_low(&mut buf[0..4]);
```

## Streaming Stores

For large data where you won't read back soon:

```rust
// Non-temporal store (bypasses cache)
unsafe { v.stream(ptr) };
```

Use streaming stores when:
- Writing large arrays sequentially
- Data won't be read again soon
- Avoiding cache pollution is important

## Gather and Scatter

Load/store non-contiguous elements:

```rust
// Gather: load from scattered indices
let data = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
let indices = i32x8::from_array(token, [0, 2, 4, 6, 8, 1, 3, 5]);
let gathered = f32x8::gather(&data, indices);
// gathered = [0.0, 20.0, 40.0, 60.0, 80.0, 10.0, 30.0, 50.0]

// Scatter: store to scattered indices
let mut output = [0.0f32; 10];
let values = f32x8::splat(token, 1.0);
values.scatter(&mut output, indices);
```

**Note**: Gather/scatter may be slow on some CPUs. Profile before using.

## Prefetch

Hint the CPU to load data into cache:

```rust
use std::arch::x86_64::*;

// Prefetch for read
unsafe { _mm_prefetch(ptr as *const i8, _MM_HINT_T0) };

// Prefetch levels:
// _MM_HINT_T0  - All cache levels
// _MM_HINT_T1  - L2 and above
// _MM_HINT_T2  - L3 and above
// _MM_HINT_NTA - Non-temporal (don't pollute cache)
```

## Interleaved Data

For RGBARGBA... or similar interleaved formats:

```rust
// Deinterleave 4 channels (RGBA)
let (r, g, b, a) = f32x8::deinterleave_4ch(
    token,
    &rgba_data[0..8],
    &rgba_data[8..16],
    &rgba_data[16..24],
    &rgba_data[24..32]
);

// Process channels separately
let r_bright = r + f32x8::splat(token, 0.1);

// Reinterleave
let (out0, out1, out2, out3) = f32x8::interleave_4ch(token, r_bright, g, b, a);
```

## Chunked Processing

Process large arrays in SIMD-sized chunks:

```rust
#[arcane]
fn process_large(token: Desktop64, data: &mut [f32]) {
    // Process full chunks
    for chunk in data.chunks_exact_mut(8) {
        let v = f32x8::from_slice(token, chunk);
        let result = v * v;  // Process
        result.store_slice(chunk);
    }

    // Handle remainder
    for x in data.chunks_exact_mut(8).into_remainder() {
        *x = *x * *x;
    }
}
```

## Alignment Tips

1. **Use `#[repr(align(32))]`** for AVX2 data:
   ```rust
   #[repr(C, align(32))]
   struct AlignedData {
       values: [f32; 8],
   }
   ```

2. **Allocate aligned memory**:
   ```rust
   use std::alloc::{alloc, Layout};

   let layout = Layout::from_size_align(size, 32).unwrap();
   let ptr = unsafe { alloc(layout) };
   ```

3. **Check alignment at runtime**:
   ```rust
   fn is_aligned<T>(ptr: *const T, align: usize) -> bool {
       (ptr as usize) % align == 0
   }
   ```

## Performance Tips

1. **Minimize loads/stores** — Keep data in registers
2. **Prefer unaligned** — Modern CPUs handle it well
3. **Use streaming for large writes** — Saves cache space
4. **Batch operations** — Load once, do multiple ops, store once
5. **Avoid gather/scatter** — Sequential access is faster

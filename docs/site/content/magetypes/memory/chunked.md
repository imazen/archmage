+++
title = "Chunked Processing"
weight = 4
+++

The standard pattern for processing large arrays with SIMD: iterate in fixed-size chunks, handle the remainder with scalar code.

## The Pattern

```rust
use archmage::{Desktop64, arcane};
use magetypes::simd::f32x8;

#[arcane]
fn process_large(token: Desktop64, data: &mut [f32]) {
    let (chunks, remainder) = data.split_at_mut(data.len() - data.len() % 8);

    // Process full 8-element chunks
    for chunk in chunks.chunks_exact_mut(8) {
        let chunk_arr: &mut [f32; 8] = chunk.try_into().unwrap();
        let v = f32x8::from_array(token, *chunk_arr);
        let result = v * v;  // Your SIMD operation
        result.store(chunk_arr);
    }

    // Handle leftover elements (0-7) with scalar code
    for x in remainder {
        *x = *x * *x;
    }
}
```

`chunks_exact_mut(8)` yields slices of exactly 8 elements. The remainder (if the array length isn't a multiple of 8) is handled separately.

## Reduction Over Chunks

When reducing an entire array to a single value:

```rust
#[arcane]
fn sum_array(token: Desktop64, data: &[f32]) -> f32 {
    let chunks = data.chunks_exact(8);
    let remainder = chunks.remainder();

    // Accumulate in a SIMD register
    let mut acc = f32x8::zero(token);
    for chunk in chunks {
        let chunk_arr: &[f32; 8] = chunk.try_into().unwrap();
        let v = f32x8::from_array(token, *chunk_arr);
        acc = acc + v;
    }

    // Reduce the accumulator to scalar
    let mut total = acc.reduce_add();

    // Add remainder
    for &x in remainder {
        total += x;
    }

    total
}
```

Accumulating in a SIMD register and reducing once at the end is faster than reducing each chunk individually.

## Alignment Tips

### Align your structs

For AVX2 data (256-bit), align to 32 bytes:

```rust
#[repr(C, align(32))]
struct AlignedData {
    values: [f32; 8],
}
```

### Allocate aligned memory

```rust
use std::alloc::{alloc, Layout};

let layout = Layout::from_size_align(size, 32).unwrap();
let ptr = unsafe { alloc(layout) };
```

### Check alignment at runtime

```rust
fn is_aligned<T>(ptr: *const T, align: usize) -> bool {
    (ptr as usize) % align == 0
}
```

In practice, unaligned access on modern CPUs is fast enough that explicit alignment is rarely worth the complexity. Profile before adding alignment boilerplate.

## Performance Tips

1. **Minimize loads and stores.** Keep data in SIMD registers as long as possible. Load once, do multiple operations, store once.

2. **Prefer unaligned access.** Modern CPUs (Haswell+, all Cortex-A) handle unaligned loads/stores with negligible penalty. Don't complicate your code for alignment unless profiling shows a bottleneck.

3. **Use streaming stores for large writes.** When writing large sequential buffers (image frames, audio buffers) that won't be read back soon, streaming stores avoid polluting the cache.

4. **Batch operations.** Instead of processing one pixel or one sample at a time, accumulate a batch and process it in one SIMD pass.

5. **Avoid gather in hot loops.** Sequential memory access is 3-10x faster than scattered access. If you find yourself using gather in a tight loop, consider restructuring your data layout.

6. **Enter `#[arcane]` once.** Put your loop inside the `#[arcane]` function, not the other way around. Each `#[arcane]` call from non-SIMD code crosses a target-feature boundary that LLVM can't optimize across. See [Target-Feature Boundaries](@/archmage/concepts/target-feature-boundaries.md).

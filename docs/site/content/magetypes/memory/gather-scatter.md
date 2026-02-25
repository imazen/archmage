+++
title = "Gather & Scatter"
weight = 2
+++

Non-contiguous memory access patterns and how to handle them in SIMD code.

## The Problem

SIMD loads and stores are contiguous — `from_slice` reads 4 or 8 consecutive elements. But real data often needs non-sequential access: lookup tables, sparse arrays, indexed structures.

## Approach: Manual Gather via Lane Access

Magetypes vectors support `Index<usize>` for lane access, so you can build a vector from scattered positions:

```rust
let data = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
let indices = [0, 2, 4, 6, 8, 1, 3, 5];

let mut arr = [0.0f32; 8];
for i in 0..8 {
    arr[i] = data[indices[i]];
}
let gathered = f32x8::from_array(token, arr);
// gathered = [0.0, 20.0, 40.0, 60.0, 80.0, 10.0, 30.0, 50.0]
```

This is straightforward and safe. The compiler may auto-vectorize it, but don't count on it.

## Approach: Raw Intrinsics for Gather

If you need hardware gather (`vgatherdps`), use raw intrinsics inside `#[arcane]`:

```rust
use std::arch::x86_64::*;

#[arcane]
fn gather_example(token: Desktop64, data: &[f32], indices: &[i32; 8]) -> [f32; 8] {
    let idx = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);
    let gathered = unsafe {
        _mm256_i32gather_ps::<4>(data.as_ptr(), idx)
    };
    let mut out = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(out.as_mut_ptr(), gathered) };
    out
}
```

## Performance Note

Gather and scatter are convenient but often slow compared to sequential access. On x86-64, `vgatherdps` issues one memory request per lane (up to 8 separate cache line accesses for `f32x8`). Sequential loads from contiguous memory benefit from cache line prefetching and can be 3-10x faster depending on the access pattern.

Use gather when:
- The access pattern is genuinely non-contiguous (lookup tables, sparse arrays)
- Restructuring the data layout to be sequential isn't practical

Avoid gather when:
- You could rearrange your data to enable sequential access
- You're in a tight inner loop — consider whether transposing the data once outside the loop would eliminate the gather

## Prefetch

Hint the CPU to load data into cache before you need it:

```rust
use std::arch::x86_64::*;

// Prefetch for read, all cache levels
unsafe { _mm_prefetch(ptr as *const i8, _MM_HINT_T0) };
```

Prefetch hints:

| Hint | Cache Level | Use |
|------|-------------|-----|
| `_MM_HINT_T0` | All levels (L1+) | Data needed very soon |
| `_MM_HINT_T1` | L2 and above | Data needed soon |
| `_MM_HINT_T2` | L3 and above | Data needed later |
| `_MM_HINT_NTA` | Non-temporal | Streaming data, don't pollute cache |

Prefetching is most effective when issued 100-300 cycles before the data is needed. In a loop, prefetch 2-4 iterations ahead. Excessive prefetching can hurt performance by evicting useful data from cache.

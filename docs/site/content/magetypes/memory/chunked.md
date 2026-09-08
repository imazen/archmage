+++
title = "Chunked Processing"
weight = 4
+++

The standard pattern for processing large arrays with SIMD: iterate in fixed-size chunks, handle the remainder with scalar code.

See the complete [`#[magetypes]` image-plane loop](@/magetypes/memory/load-store.md)
and [reusable generic helper](@/magetypes/examples/generic-kernels.md).
Dispatch once, iterate array chunks, and handle the remainder explicitly.
For a reduction, accumulate in a vector inside that same context and call
`reduce_add()` once after the loop; floating-point association can differ by ISA.

## Alignment Tips

### Align your structs

For AVX2 data (256-bit), align to 32 bytes:

```rust
#[repr(C, align(32))]
struct AlignedData {
    values: [f32; 8],
}
```

### Allocate aligned memory safely

When a measured consumer needs 32-byte alignment, let Rust allocate an aligned
type and own its lifetime:

```rust
#[repr(C, align(32))]
struct AlignedBlock([f32; 8]);
let blocks = vec![AlignedBlock([0.0; 8]), AlignedBlock([0.0; 8])];
assert_eq!(blocks.as_ptr() as usize % 32, 0);
```

The vector drops its allocation normally, and every element is initialized.
Ordinary magetypes loads do not require this stronger alignment.

## Performance tips

Keep the loop inside one generated context; keep intermediate vectors in
registers; retain a scalar tail. Test cache-resident and streaming working sets.
Use measurements to choose layout and width rather than assuming that alignment,
gather avoidance, or non-temporal stores always improve performance.

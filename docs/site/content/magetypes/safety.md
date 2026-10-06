+++
title = "Safety Model"
description = "Why magetypes vectors are safe to use, and the one module that holds their unsafe"
weight = 10
+++

magetypes vectors are safe to use from safe code, including crates that
forbid `unsafe`. That rests on two things: every vector carries the archmage
token it was built with, and all of magetypes' own `unsafe` sits in one module,
where each block states what it relies on. The
[archmage safety model](@/archmage/concepts/safety.md) covers tokens and
`#[arcane]`;
[SOUNDNESS.md](https://github.com/imazen/archmage/blob/main/docs/SOUNDNESS.md)
has the complete inventory.

## Vectors carry their token

[`f32x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html)
stores the platform register next to the token, so no vector exists without a
proof. Constructors take the token, as in `f32x8::splat_t(token, 1.0)`, or,
like `from_raw(raw)`, require a matching `#[target_feature]` context that rustc
checks. Each operation calls a backend trait method that takes the token by
value. The x86 and NEON implementations of those methods are `#[arcane]`
regions for that token, so the compiler checks each intrinsic they use against
the token's features. WASM SIMD intrinsics are safe to call anywhere, because a
module that loads has its features.

There is no route to a vector without a token. The vector types implement no
`Default`, serde or bytemuck traits, and calling a backend method without a
token value does not compile:

```rust,compile_fail
#![forbid(unsafe_code)]
use magetypes::simd::backends::F32x8Backend;

// The backend methods take the token as `self`; there is no tokenless form.
let v = <archmage::X64V3Token as F32x8Backend>::splat(2.0);
```

## One module holds all of the `unsafe`

magetypes denies `unsafe_code` at the crate root and allows it in one module,
[`simd_storage.rs`](https://github.com/imazen/archmage/blob/main/magetypes/src/simd_storage.rs).
In 0.9.30 it contains 14 `unsafe` blocks, each with a `// SAFETY:` comment:

| What | Why it holds |
|---|---|
| Copies and views between plain-data types (`copy`, `cast`, `view`, `view_mut`, `store`) | Both types implement `Pod`, so every bit pattern is valid, and each `Pod` registration asserts the type has no padding. Sizes and alignment are checked at compile time. |
| Treating storage as vectors (`vector_view`, `vector_slice`) | A token value is required, each vector's layout is asserted at compile time, and slice length and alignment are checked at run time; a mismatch returns `None`. |
| AVX-512 gather and scatter | Every lane that touches memory has its index bounded against the borrowed slice first. |
| `TokenStorage` | Implemented only by a macro that asserts each vector type is exactly the register followed by the token, with the register at offset 0. |

The backend implementations, 5,395 intrinsic calls across x86, NEON and WASM
in 0.9.30, contain no `unsafe`. The module also declares `cast::Upcast`, a deprecated
trait with an `unsafe fn` that nothing implements.

## Gather and scatter

The gather and scatter intrinsics read and write `base + 4 * index` for sixteen
signed 32-bit indices, so a borrowed slice alone proves nothing about the
addresses. `gather_wrapping` masks each index with `N - 1` for a power-of-two
table of `N` elements. `gather_or` and `scatter_select` enable only the lanes
whose unsigned index is below `min(len, 2^31)`, and a disabled lane reads or
writes nothing. The module quotes Intel's description and pseudocode for each
intrinsic it uses next to that argument; [Gather & Scatter](@/magetypes/memory/gather-scatter.md)
has the same quotes and the method-to-intrinsic table.

## How it's checked

- The compiler: `#![deny(unsafe_code)]`, with one module allowed.
- `cargo xtask soundness` rejects the `unsafe` keyword, any other
  `allow(unsafe_code)`, bare `transmute`, gather and scatter intrinsics, and
  `Default`, serde or bytemuck on vector types anywhere else in magetypes,
  including code compiled out on the host. It also checks every backend
  intrinsic against its token's features.
- Miri runs the magetypes tests to catch layout and pointer mistakes in the
  `unsafe` blocks.
- [`tests/gather_scatter_v4.rs`](https://github.com/imazen/archmage/blob/main/magetypes/tests/gather_scatter_v4.rs)
  drives gathers and scatters with hostile indices: negative values, the 2^31
  boundary, indices at and just past the end, empty slices and duplicate
  scatter targets.

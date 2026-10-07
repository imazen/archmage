+++
title = "Safety Model"
description = "Why using magetypes takes no unsafe, and how little unsafe it needs inside"
weight = 10
+++

Using magetypes takes no `unsafe` in your code, so your crate can keep
`#![forbid(unsafe_code)]`. Inside, magetypes stacks its own proofs on
archmage's tokens until all that is left for `unsafe` is 14 one-line blocks:
loads, stores and views of vector storage, and the AVX-512 gathers and
scatters. The [archmage safety model](@/archmage/concepts/safety.md) covers
tokens and `#[arcane]`;
[SOUNDNESS.md](https://github.com/imazen/archmage/blob/main/docs/SOUNDNESS.md)
has the complete inventory.

## Proofs on proofs

Each layer relies on the one below it:

1. **The token.** Every vector holds the archmage token it was built with, so
   code that has a vector has the proof that the CPU supports its operations.
2. **The compiler checks the intrinsics.** Each x86 and NEON backend method is
   an `#[arcane]` region for its token, so rustc rejects any intrinsic that the
   token's features don't cover. WASM SIMD intrinsics are safe to call
   anywhere, because a module that loads has its features. `cargo xtask
   soundness` re-checks all 5,485 magetypes intrinsic calls in 0.9.30 against
   the intrinsic table extracted from Rust's `stdarch`.
3. **Compile-time layout proofs.** Every reinterpretation of memory checks at
   compile time that the sizes match. Alignment is checked wherever a
   reference is formed: at compile time for single values, at run time for
   slices. Each vector type asserts that it is exactly its register followed
   by its token, with the register at offset 0. Each plain-data type asserts
   that it has no padding. Shift counts and gather table sizes are checked at
   compile time too.
4. **What remains.** The compiler cannot check memory access through a pointer.
   The 14 `unsafe` blocks do exactly that, and each states what it relies on in
   a `// SAFETY:` comment.

## Vectors carry their token

[`f32x8<T>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html)
stores the platform register next to the token, so no vector exists without a
proof. Constructors take the token, as in `f32x8::splat_t(token, 1.0)`, or,
like `from_raw(raw)`, require a matching `#[target_feature]` context that rustc
checks. Each operation calls a backend trait method that takes the token by
value.

Nothing builds a vector without that proof. The vector types implement no
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
In 0.9.30 it contains 14 `unsafe` blocks, one line each, and each with a
`// SAFETY:` comment:

| Blocks | What they do | Why they hold |
|---|---|---|
| 4: `copy`, `view`, `view_mut`, `store` | Copy, borrow or store plain data as another plain-data type: the loads, stores and bit casts under the vector types | Both types implement `Pod`, so every bit pattern is valid, and each `Pod` registration asserts the type has no padding. Sizes are checked at compile time, and alignment too wherever a reference is formed. |
| 4: `vector_view`, `vector_view_mut`, `vector_slice`, `vector_slice_mut` | Borrow plain storage as vectors | A token value is required, each vector's layout is asserted at compile time, and slice length and alignment are checked at run time; a mismatch returns `None`. |
| 6: the AVX-512 gathers and scatters | Read and write table elements by per-lane index | Every lane that touches memory has its index bounded against the borrowed slice first. |

The blocks rely on two `unsafe` marker traits defined in the same module:

- `Pod` marks plain data where every bit pattern is valid. A macro implements
  it for each scalar and vector type and asserts that the type has no padding.
  Arrays of `Pod` types are `Pod` too, and add no padding.
- `TokenStorage` marks a vector that is exactly its register followed by its
  token. Only a macro implements it, and the macro checks each vector's layout
  at compile time.

The module also declares `cast::Upcast`, a deprecated trait with an `unsafe fn`
that nothing implements. The backend implementations contain no `unsafe`.

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
- `cargo xtask soundness` reads the rest of magetypes, including code compiled
  out on the host. It rejects the `unsafe` keyword, any other
  `allow(unsafe_code)`, bare `transmute`, and gather and scatter intrinsics. It
  rejects `Default`, serde and bytemuck on vector types. It also checks every
  backend intrinsic against its token's features.
- Miri runs the magetypes tests to catch layout and pointer mistakes in the
  `unsafe` blocks.
- [`tests/gather_scatter_v4.rs`](https://github.com/imazen/archmage/blob/main/magetypes/tests/gather_scatter_v4.rs)
  drives gathers and scatters with hostile indices: negative values, the 2^31
  boundary, indices at and just past the end, empty slices and duplicate
  scatter targets.

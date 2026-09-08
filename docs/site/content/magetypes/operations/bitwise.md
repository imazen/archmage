+++
title = "Integer & Bitwise"
weight = 4
+++

Integer vectors support `&`, `|`, `^`, and `!`. The
[zenwebp add-green transform](@/magetypes/examples/byte-transforms.md) shows a
complete bitwise/shift kernel and its scalar tail. Packed channels require
masking before additions so carries do not leak between channels.

| Need | Method |
|---|---|
| Constant left shift | `shl_const::<N>()` |
| Constant zero-filling right shift | `shr_logical_const::<N>()` |
| Constant sign-extending right shift (signed types) | `shr_arithmetic_const::<N>()` |
| Same runtime count for every lane (supported wider element types) | `shl_uniform(count)`, `shr_logical_uniform(count)`, `shr_arithmetic_uniform(count)` |
| Different count per lane (supported shapes) | Consult the vector's method reference |

The shorter `shl::<N>()`, `shr_logical::<N>()`, and `shr_arithmetic::<N>()`
spellings are compatibility aliases. There is no general promise that a signed
`shr` spelling means the same thing across every historical API. Choose the
explicit logical or arithmetic method.

Runtime byte shifts are not part of the current portable surface. Shift-count
normalization and canonical comparison masks have contracts recorded in
[ISA quirks](@/magetypes/isa-quirks.md). For arbitrary bit patterns, do not use
`blend` as if it were a bitwise mux; construct a mux with AND/OR explicitly.

Widening and saturating narrowing have their own
[reference and complete example](@/magetypes/conversions/width.md).

+++
title = "Memory"
description = "Load, store, gather, scatter, and data layout patterns"
sort_by = "weight"
weight = 6

[extra]
sidebar = true
+++

Moving data between memory and SIMD registers efficiently. The difference between fast and slow SIMD code is usually in the memory access patterns, not the arithmetic.

1. [Load & Store](@/magetypes/memory/load-store.md) — Unaligned, aligned, partial, streaming
2. [Gather & Scatter](@/magetypes/memory/gather-scatter.md) — Non-contiguous access and prefetch hints
3. [Interleaved Data](@/magetypes/memory/interleaved.md) — `deinterleave_4ch`, `interleave_4ch` for RGBA and similar
4. [Chunked Processing](@/magetypes/memory/chunked.md) — Processing large arrays in SIMD-sized chunks, alignment, performance

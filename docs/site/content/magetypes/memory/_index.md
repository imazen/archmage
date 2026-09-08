+++
title = "Memory"
description = "Load, store, gather, scatter, and data layout patterns"
sort_by = "weight"
weight = 6

[extra]
sidebar = true
+++

Moving data between memory and SIMD registers efficiently. Data layout, arithmetic, and target-feature context all affect performance.

1. [Load & Store](@/magetypes/memory/load-store.md) — Array loads/stores, length proofs, scalar tails
2. [Gather & Scatter](@/magetypes/memory/gather-scatter.md) — Checked access and the limits of the current API
3. [Interleaved Data](@/magetypes/memory/interleaved.md) — `deinterleave_4ch`, `interleave_4ch` for RGBA and similar
4. [Chunked Processing](@/magetypes/memory/chunked.md) — Processing large arrays in SIMD-sized chunks, alignment, performance

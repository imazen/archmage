+++
title = "Which Tokens to Target"
weight = 1
+++

Choose tiers from operations, logical width, and measured workload performance.
Use `summon()` to check the required feature set; CPU model years and brand names
are not substitutes for detection.

| Workload | Starting point |
|---|---|
| Portable eight-lane f32 or 32-byte loops | `v3, neon, wasm128, scalar` |
| Sixteen-lane f32 loops | `w512` logical types; optionally native V4 with `avx512` |
| Existing scalar loops with regular memory access | `#[autoversion]`, then inspect vectorization |
| ISA-specific shuffles, crypto, or dot products | A concrete token whose registry features cover the instructions |
| Very short inputs | Compare dispatch and tail costs with a scalar implementation |

V4 need not outperform V3 for a given kernel. Width, register pressure, data
layout, memory bandwidth, and numerical fixups all affect throughput. Measure
at realistic row/strip sizes and include the scalar tail.

The [token reference](@/archmage/reference/tokens.md) describes capabilities;
[ISA quirks](@/magetypes/isa-quirks.md) describes semantic costs. Detection support
also depends on the OS and target. Test the actual deployment platform rather
than assuming every machine of a given generation can summon a tier.

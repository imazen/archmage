+++
title = "Polyfills"
weight = 1
+++

A logical vector can use several hardware vectors. For example, [`f32x8<NeonToken>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html)
uses two 128-bit halves. The public shape and lane order remain eight f32 values.
No algorithm-level runtime dispatch is needed merely to split those halves.

`w512` enables logical 512-bit types, including polyfills, and is a default
feature. `avx512` additionally enables native AVX-512 implementations. A wider
logical shape is not a promise that the CPU executes it as one instruction.

Use the same complete [generated kernel](@/archmage/getting-started/first-simd.md)
for supported backends. Do not call a bare generic SIMD loop from baseline code
and assume the polyfill will establish the feature context.

Some operations split cheaply; others require cross-half shuffles, scalar
fallbacks, or numerical fixups. Neither constant overhead nor a universal
speedup over scalar code is guaranteed. Benchmark the operation in its actual
loop. [ISA quirks](@/magetypes/isa-quirks.md) is the semantic reference, including
cases where identically named operations retain different backend behavior.

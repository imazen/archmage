+++
title = "Cross-Platform"
description = "Polyfill strategy and known behavioral differences across architectures"
sort_by = "weight"
weight = 7

[extra]
sidebar = true
+++

Magetypes aims for API parity across x86-64, AArch64, and WASM. Wider types are polyfilled on narrower hardware, and the API surface is identical — but a few behavioral differences exist between architectures.

1. [Polyfills](@/magetypes/cross-platform/polyfills.md) — How `f32x8` works on 128-bit hardware, `implementation_name()`
2. [Behavioral Differences](@/magetypes/cross-platform/differences.md) — Operators, shift semantics, blend signatures

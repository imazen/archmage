+++
title = "Dispatch"
description = "Using magetypes with incant! and #[magetypes] for multi-platform dispatch"
sort_by = "weight"
weight = 8

[extra]
sidebar = true
+++

Magetypes vectors work with archmage's dispatch system. The generic backend pattern means one function covers all architectures — `incant!` handles runtime selection automatically.

1. [Types and Dispatch](@/magetypes/dispatch/types-and-dispatch.md) — `incant!` with generic SIMD code, `#[magetypes]` for generated bodies

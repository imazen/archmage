# Performance

Archmage generates identical assembly to hand-written `#[target_feature]` + `unsafe` code. The safety abstractions compile away. At runtime, you get raw SIMD instructions.

The only thing that costs you performance is calling `#[arcane]` from the wrong place.

## Zero overhead: archmage = bare `#[target_feature]`

Every benchmark pattern shows archmage and bare `#[target_feature]` producing the same timings. There is no wrapper cost, no token cost, no abstraction tax.

| What you write | Time (1000 x 8-float add) | What LLVM sees |
|----------------|---------------------------|----------------|
| `#[rite]` helper in `#[arcane]` | 547 ns | One function, one target, full inlining |
| Bare `#[target_feature]` (no archmage) | 544 ns | Same |
| `#[arcane]` per loop iteration | 2209 ns (4x) | Boundary crossing per call |
| Bare `#[target_feature]` per loop iteration | 2222 ns (4x) | Same boundary, same cost |

The 4x penalty comes from LLVM, not archmage. Read on.

## The target-feature boundary

`#[arcane]` and `#[rite]` read the token type from your function signature to decide which `#[target_feature]` to emit. A function taking `Desktop64` gets `#[target_feature(enable = "avx2,fma,...")]`. A function taking `X64V4Token` gets AVX-512 features. The token type *is* the feature selector.

`#[arcane]` generates a wrapper: an outer function that calls an inner `#[target_feature]` function via `unsafe`. This is how you cross into SIMD code without writing `unsafe` yourself — but the wrapper creates an LLVM optimization boundary. LLVM won't inline across mismatched `#[target_feature]` attributes: no load hoisting, no store sinking, no cross-iteration vectorization.

`#[rite]` applies `#[target_feature]` + `#[inline]` directly to the function, with no wrapper. When the caller already has matching features (from its own `#[arcane]` or `#[rite]`), LLVM inlines freely — no boundary.

The boundary has nothing to do with archmage. A bare `#[target_feature]` function has the same cost. `#[arcane]` just makes the wrapper safe; the boundary is LLVM's.

**The fix:** use `#[arcane]` once at your API entry point, and `#[rite]` for everything called from within SIMD code. Pass the same token type through your call hierarchy — the macros emit the same `#[target_feature]` on every function that takes it. LLVM sees matching targets, inlines freely, and there's no boundary.

```rust
// WRONG: boundary every iteration (4x slower)
fn process_all(points: &[[f32; 8]]) {
    let token = Desktop64::summon().unwrap();
    for p in points {
        process_one(token, p);  // #[arcane] — boundary crossing
    }
}

// RIGHT: one boundary, loop inside
fn process_all(points: &[[f32; 8]]) {
    if let Some(token) = Desktop64::summon() {
        process_all_simd(token, points);  // one #[arcane] entry
    }
}

#[arcane]
fn process_all_simd(token: Desktop64, points: &[[f32; 8]]) {
    for p in points {
        process_one(token, p);  // #[rite] — inlines, no boundary
    }
}
```

## Benchmark results

All benchmarks from `cargo bench --bench asm_inspection --features "std macros avx512"`, run on x86-64 with AVX-512 support. Source: [`benches/asm_inspection.rs`](../benches/asm_inspection.rs).

### Simple vector add (1000 iterations, 8-float add)

Seven patterns isolating the target-feature boundary effect:

| # | Pattern | Time | Ratio | Boundary? |
|---|---------|------|-------|-----------|
| 1 | `#[arcane]` per iteration | 2209 ns | 4.1x | yes — baseline caller, AVX2 callee |
| 2 | `#[rite]` in `#[arcane]` | 547 ns | 1.0x | no — features match, LLVM inlines |
| 3 | Manual inline in `#[arcane]` | 544 ns | 1.0x | no — same function body |
| 4 | `#[rite]` called directly (unsafe, no wrapper) | 2227 ns | 4.1x | yes — proves it's not the wrapper |
| 5 | Scalar via wrapper fn | 542 ns | 1.0x | no — no `#[target_feature]` at all |
| 6 | Scalar inline | 537 ns | 1.0x | no — baseline |
| 7 | Bare `#[target_feature]` (no archmage) | 2222 ns | 4.1x | yes — same boundary, archmage not involved |

Patterns 1, 4, and 7 all cross the boundary per iteration and land at the same ~2.2 us. Pattern 4 has no wrapper at all (calls `#[rite]` directly with `unsafe`), proving the overhead is the boundary, not the wrapper. Patterns 2, 3, 5, and 6 avoid the boundary and land at ~544 ns.

### DCT-8 (100 rows, 8 dot products per row)

A realistic signal-processing workload. Each row computes 8 coefficient dot products using `_mm256_mul_ps` + horizontal sum. Higher computational density amplifies the boundary effect.

| Pattern | Time | Ratio |
|---------|------|-------|
| `#[rite]` in `#[arcane]` | 61 ns | 1.0x |
| `#[arcane]` per row | 376 ns | 6.2x |
| Bare `#[target_feature]` per row | 374 ns | 6.1x |

Archmage and bare `#[target_feature]` produce identical numbers here too (376 vs 374 ns — noise). The boundary costs 6.2x instead of 4x because DCT-8 has more optimization potential per call: FMA fusion, register reuse across coefficient loads, instruction scheduling. When LLVM can inline, it exploits all of that. When the boundary forces a separate call, it can't. The multiplier depends on how much work LLVM loses at the boundary, not on which mechanism creates it.

### Cross-token nesting (1000 iterations, 8-float add)

What happens when `#[arcane]` functions call other `#[arcane]` functions at different feature levels. When both functions take the same token type, their `#[target_feature]` strings match and LLVM inlines freely — no boundary. When the token types differ, direction matters.

| Pattern | Time | Ratio | Why |
|---------|------|-------|-----|
| V3 `#[arcane]` calling V3 `#[arcane]` | 547 ns | 1.0x | Caller has V3 features; callee needs V3. LLVM inlines. |
| V3 `#[arcane]` calling V3 `#[rite]` | 544 ns | 1.0x | Control — `#[rite]` always inlines. |
| V4 entry calling V3 `#[arcane]` (downgrade) | 547 ns | 1.0x | Caller has V4 superset; V3 callee inlines freely. |
| V3 down to V2 (AVX2 calling SSE) | 544 ns | 1.0x | Caller has V3 superset; V2 callee inlines. |
| V2 up to V3 (SSE calling AVX2) | 2209 ns | 4.1x | Caller lacks AVX2; boundary per call. |
| V3 up to V4 (AVX2 calling AVX-512) | 2222 ns | 4.1x | Caller lacks AVX-512; boundary per call. |

Every downgrade pattern matched its bare `#[target_feature]` equivalent exactly. Every upgrade pattern hit the same ~4x boundary as calling from baseline code.

**The rule:** downgrades are free (caller's superset features enable inlining), upgrades hit the boundary (callee needs features the caller doesn't have).

## Rules

These are distilled from the benchmark data above.

1. **Enter `#[arcane]` once.** Put loops inside it. Each call from non-SIMD code crosses the boundary.

2. **Use `#[rite]` for everything inside.** `#[rite]` adds `#[target_feature]` + `#[inline]` directly. LLVM inlines it into any caller with matching features. No boundary.

3. **Never call `#[arcane]` from `#[arcane]`.** Use `#[rite]` for functions called from SIMD code. `#[arcane]` creates a wrapper that re-crosses the boundary.

4. **Downcasting is free.** A V4 function calling a V3 helper inlines because V4 is a superset of V3. Same for V3 calling V2.

5. **Upcasting hits the boundary.** A V3 function calling a V4 helper can't inline because the caller lacks AVX-512 features. Dispatch at the entry point, not deep in hot code.

6. **Use concrete tokens, not generics.** Generic bounds (`impl HasX64V2`) create optimization barriers. Use `X64V3Token` directly for hot paths.

## Reproducing

```bash
# Simple vector add + DCT-8
cargo bench --bench asm_inspection --features "std macros"

# Cross-token nesting (needs avx512 feature + AVX-512 hardware)
cargo bench --bench asm_inspection --features "std macros avx512"
```

Results will vary by CPU. The *ratios* between patterns are stable: archmage always matches bare `#[target_feature]` on the same workload. The boundary multiplier itself (4x on simple adds, 6.2x on DCT-8) depends on how much optimization LLVM loses when it can't inline — denser workloads lose more.

`summon()` overhead is separate: ~1.3 ns cached, 0 ns with `-Ctarget-cpu=haswell` (compiles away). See [`benches/summon_overhead.rs`](../benches/summon_overhead.rs).

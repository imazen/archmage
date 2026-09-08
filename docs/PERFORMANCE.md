# Performance

Archmage generates identical assembly to hand-written `#[target_feature]` + `unsafe` code. The safety abstractions compile away. At runtime, you get raw SIMD instructions.

Calling `#[arcane]` across a target-feature boundary can prevent optimization. Compile-time costs are measured separately in the proc-macro section below.

## Zero overhead: archmage = bare `#[target_feature]`

Every benchmark pattern shows archmage and bare `#[target_feature]` producing the same timings. There is no wrapper cost, no token cost, no abstraction tax.

| What you write | Time (1000 x 8-float add) | What LLVM sees |
|----------------|---------------------------|----------------|
| `#[arcane]` calling `#[arcane]` (matching features) | 547 ns | Features match — LLVM inlines wrapper |
| Bare `#[target_feature]` (no archmage) | 544 ns | Same |
| `#[arcane]` per loop iteration from non-SIMD code | 2209 ns (4x) | Boundary crossing per call |
| Bare `#[target_feature]` per loop iteration | 2222 ns (4x) | Same boundary, same cost |

The 4x penalty comes from LLVM, not archmage. Read on.

## The target-feature boundary

`#[arcane]` reads the token type from your function signature to decide which `#[target_feature]` to emit. A function taking `X64V3Token` gets `#[target_feature(enable = "avx2,fma,...")]`.

`#[arcane]` generates a wrapper: an outer function that calls an inner `#[target_feature]` function via `unsafe`. This is how you cross into SIMD code without writing `unsafe` yourself. When an `#[arcane]` function calls another `#[arcane]` function with matching features, LLVM inlines the wrapper away — no boundary. The boundary only exists when the caller has fewer features than the callee (e.g., non-SIMD code calling an `#[arcane]` function).

LLVM won't inline across mismatched `#[target_feature]` attributes: no load hoisting, no store sinking, no cross-iteration vectorization. But matching features = no mismatch = full inlining.

The boundary has nothing to do with archmage. A bare `#[target_feature]` function has the same cost. `#[arcane]` just makes the wrapper safe; the boundary is LLVM's.

**The fix:** enter `#[arcane]` once from non-SIMD code, put the loop inside. From within `#[arcane]`, call other `#[arcane]` functions freely — matching features means LLVM inlines the wrapper away. `#[rite]` is also available (adds `#[target_feature]` + `#[inline]` directly, no wrapper) but isn't necessary when features match.

```rust
// WRONG: boundary every iteration (4x slower)
fn process_all(points: &[[f32; 8]]) {
    let token = X64V3Token::summon().unwrap();
    for p in points {
        process_one(token, p);  // #[arcane] — boundary crossing
    }
}

// RIGHT: one boundary, loop inside
fn process_all(points: &[[f32; 8]]) {
    if let Some(token) = X64V3Token::summon() {
        process_all_simd(token, points);  // one #[arcane] entry
    }
}

#[arcane(import_intrinsics)]
fn process_all_simd(token: X64V3Token, points: &[[f32; 8]]) {
    for p in points {
        process_one(token, p);  // #[arcane] — features match, LLVM inlines
    }
}

#[arcane(import_intrinsics)]
fn process_one(_token: X64V3Token, p: &[f32; 8]) {
    // ...
}
```

## Benchmark results

All benchmarks from `cargo bench --bench asm_inspection --features "std avx512"`, run on x86-64 with AVX-512 support. Source: [`benches/asm_inspection.rs`](../benches/asm_inspection.rs).

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

### Generic magetypes types: zero overhead inside `#[arcane]` (with `#[inline(always)]`)

A common concern: does using `f32x8::<T>` with a generic `T: F32x8Backend` produce worse code than concrete `f32x8::<x64v3>`? **No — but only if the generic function can inline into the `#[arcane]` caller.**

The backend trait methods are all `#[inline(always)]`, but that's not enough on its own. **Your generic helper function must also inline** into the `#[arcane]` caller so LLVM compiles it within the `#[target_feature]` region. The generic function itself has no `#[target_feature]` — it gets the right features only by being inlined into a function that does.

Source: [`benches/generic_vs_concrete.rs`](../benches/generic_vs_concrete.rs).

| Pattern | Time | Assembly |
|---------|------|----------|
| `f32x8::<T>` generic `#[inline(always)]` inside `#[arcane]` | 1.35 ns | `vmovups` + `vaddps` + horizontal sum |
| `f32x8::<T>` generic (no annotation) inside `#[arcane]` | 1.37 ns | identical — LLVM chose to inline (not guaranteed) |
| `f32x8::<x64v3>` concrete inside `#[arcane]` | 1.16 ns | identical instructions |
| Concrete via `#[rite]` in `#[arcane]` | 1.40 ns | identical |
| `f32x8::<T>` generic `#[inline(never)]` inside `#[arcane]` | **23.7 ns (18x)** | `call _mm256_add_ps` — forced no-inline proves it |
| `f32x8::<T>` generic **without** `#[target_feature]` | **24.7 ns (18x)** | `call _mm256_add_ps` (function calls!) |

The `#[inline(never)]` row is the smoking gun: even inside `#[arcane]`, a generic function that can't inline is just as slow as having no `#[target_feature]` at all. The generic function body is compiled without target features — it only gets them by being inlined into the `#[arcane]` caller's `#[target_feature]` region.

**Mark generic SIMD helpers `#[inline(always)]`.** For small same-crate functions, LLVM usually inlines without annotation (the "no annotation" row above). But this is an LLVM heuristic, not a guarantee — LLVM can decline to inline any function without `#[inline(always)]`. Cross-crate, without at least `#[inline]`, the function body isn't even available to the caller's compilation unit. `#[inline(always)]` removes all ambiguity: the function will always inline, and the generic code will always get the caller's target features.

```rust
// CORRECT: #[inline(always)] guarantees the generic body inlines into the caller
#[inline(always)]
fn generic_sum<T: F32x8Backend>(token: T, data: &[f32; 8]) -> f32 {
    let v = f32x8::<T>::from_array(token, *data);
    (v + v).reduce_add()  // ~1.35 ns from #[arcane(import_intrinsics)]
}

// The #[arcane] caller provides #[target_feature] — generic_sum inlines into it
#[arcane(import_intrinsics)]
fn entry(token: X64V3Token, data: &[f32; 8]) -> f32 {
    generic_sum(token, data)  // Inlines → full AVX2 codegen
}
```

```asm
; #[inline(always)] generic inside #[arcane] — identical to concrete:
vmovups ymm0, [rdi]          ; load 8 floats
vaddps  ymm0, ymm0, ymm0    ; v + v
vextractf128 xmm1, ymm0, 1  ; horizontal sum...
vaddps  xmm0, xmm1, xmm0
vhaddps xmm0, xmm0, xmm0
vmovshdup xmm1, xmm0
vaddss  xmm0, xmm0, xmm1

; #[inline(never)] generic inside #[arcane] — still catastrophic:
movups  xmm0, [rdi]          ; SSE2 load, not AVX!
call    _mm256_add_ps         ; FUNCTION CALL
call    _mm256_extractf128_ps ; FUNCTION CALL
call    _mm_hadd_ps           ; FUNCTION CALL
```

**The rule:** generic magetypes code is zero-cost when it inlines into an `#[arcane]` or `#[rite]` caller. Mark generic SIMD helpers `#[inline(always)]` to guarantee this. The generic function has no `#[target_feature]` of its own — it inherits the caller's features through inlining.

## Rules

These are distilled from the benchmark data above.

1. **Enter `#[arcane]` once.** Put loops inside it. Each call from non-SIMD code crosses the boundary.

2. **Use `#[arcane]` for helpers too.** When an `#[arcane]` function calls another with matching features, LLVM inlines the wrapper — no boundary. `#[rite]` (adds `#[target_feature]` + `#[inline]` directly) and plain `#[inline(always)]` functions also work.

3. **Don't cross feature boundaries in hot loops.** Calling `#[arcane]` from `#[arcane]` with matching features is free — LLVM inlines the wrapper (benchmark: V3→V3 = 1.0x). The boundary only hurts when the caller has fewer features than the callee (V3→V4 = 4x).

4. **Downcasting is free.** A V4 function calling a V3 helper inlines because V4 is a superset of V3. Same for V3 calling V2.

5. **Upcasting hits the boundary.** A V3 function calling a V4 helper can't inline because the caller lacks AVX-512 features. Dispatch at the entry point, not deep in hot code.

6. **Generic magetypes types are zero-cost inside `#[arcane]` — if they inline.** `f32x8::<T>` and `f32x8::<x64v3>` produce identical assembly when the generic function inlines into the `#[arcane]` caller. Mark generic SIMD helpers `#[inline(always)]` to ensure this. The generic function has no `#[target_feature]` of its own — if it doesn't inline, intrinsics become function calls (18x slower). The backend methods are `#[inline(always)]`, but that only helps once the generic body is inside the `#[target_feature]` region.

## Reproducing

```bash
# Simple vector add + DCT-8
cargo bench --bench asm_inspection --features "std"

# Cross-token nesting (needs avx512 feature + AVX-512 hardware)
cargo bench --bench asm_inspection --features "std avx512"
```

Results will vary by CPU. The *ratios* between patterns are stable: archmage always matches bare `#[target_feature]` on the same workload. The boundary multiplier itself (4x on simple adds, 6.2x on DCT-8) depends on how much optimization LLVM loses when it can't inline — denser workloads lose more.

`summon()` overhead is separate: ~1.3 ns cached, 0 ns with `-Ctarget-cpu=haswell` (compiles away). See [`benches/summon_overhead.rs`](../benches/summon_overhead.rs).

## Proc-macro parsing and allocation pass (2026-09-07)

Compared `0373579` with the parsing cleanup in `9382f9d` on the `wsl` Ryzen 9
7950X, Rust 1.98.1. This changes macro execution, not the generated algorithms.
Bodies were already opaque token streams; `syn` remains responsible for
signatures, generics, bounds, and dispatch arguments.

The changes borrow temporary tier names, avoid sorting already ordered default
and filtered tier lists, preserve groups that need no rewriting, stream argument
emission without intermediate vectors, and remove unreachable stub generation.
The default-order test compares the fast path with general resolution, checks
uniqueness and fallback placement, and separately checks caller-order stability
for equal-priority and duplicate explicit tiers. Arbitrary user lists still sort.

### Allocation evidence

The ignored `profile_allocations` test counts allocation/reallocation calls while
parsing and expanding pre-tokenized input on its own thread. These are
**standalone proc_macro2 measurements**, not rustc's total allocations; input
lexing is outside the measurement. It uses the default unoptimized test profile.

| Input | Before allocations | After allocations |
|---|---:|---:|
| `arcane`, ordinary kernel | 128 | 116 |
| `arcane`, nested dispatch | 166 | 163 |
| `rite`, ordinary kernel | 89 | 81 |
| `magetypes`, local vector alias | 300 | 240 |
| `autoversion`, scalar loop | 380 | 370 |

### Consumer builds

`xtask/macro_perf.py` tests linear-srgb 0.6.12 (with `transfer`) and
zenpixels-convert 0.2.16 from the same source archives used in earlier consumer
comparisons: linear-srgb `6bae33ee657d0cc29eaae6fd869894ec78a41b1c` and zenpixels
`03bedaa73ae2b8c013765fd7eef59104fd307127`. The archived manifests and edited
source files were checked against those commits. The harness records archive
hashes and resolved lockfiles; set `BENCH_LOCKS_DIR` to that results directory
to reuse the dependency versions. Each configuration has six paired baseline/candidate runs in
alternating order. Cold runs start with empty Cargo artifact directories;
registry sources and OS caches are warm. The script verifies identical dependency
trees and that source edits rebuild the consumer while keeping archmage,
archmage-macros, and magetypes fresh. Cargo's default dev and release profiles
apply; release source-edit builds do not enable incremental compilation.

Seconds below are medians. The last column uses the geometric mean of paired
ratios and an approximate 95% Student-t interval on log ratios; it is not the
ratio of the displayed medians. Small-sample intervals do not account for every
possible machine-load effect.

| Consumer | Profile | Cold before → after | Kernel edit before → after | Paired kernel-edit change (95% interval) |
|---|---|---:|---:|---:|
| linear-srgb | dev | 5.084 → 5.169 | 0.397 → 0.380 | −5.5% [−7.6%, −3.2%] |
| linear-srgb | release | 5.296 → 5.265 | 1.346 → 1.338 | −0.6% [−1.6%, +0.5%] |
| zenpixels-convert | dev | 5.901 → 5.875 | 0.621 → 0.586 | −5.1% [−10.2%, +0.3%] |
| zenpixels-convert | release | 7.662 → 7.463 | 3.282 → 3.214 | −3.2% [−6.1%, −0.3%] |

All four cold-build paired intervals include zero: **no demonstrated cold-build
improvement or regression**. Ordinary-function edits were also measured: paired
changes were −5.6% (linear dev), −0.6% (linear release), −2.1% (zenpixels dev), and
−1.4% (zenpixels release); only linear dev's interval excluded zero.

Four alternating nightly self-profile runs per variant measured all 608
proc-macro expansions in magetypes. Median `expand_proc_macro` self time was
64.25 ms before and 62.20 ms after, with overlapping samples. Macro invocation
counts and `-Zmacro-stats` output were unchanged. This does not establish a
significant aggregate expansion-time win or justify replacing syn wholesale.

### Correctness and reproduction

The macro library now runs its implementation directly as well as through rustc.
136 unit/contract tests pass in both feature configurations; the allocation probe
is deliberately opt-in. Existing expansion snapshots, compilation of both inputs
and outputs, negative soundness cases, and behavioral tests pass. LLVM's fresh combined library/compiler-backed line report is 96.9%, not 100%
(shared helpers and tier resolution: 100%; dispatch rewriter: 99.3%). Remaining
gaps include thin entry-point error paths, defensive branches, and test-helper
failure paths. Coverage is a gap-finding tool, not a soundness proof. The ISA comparison found identical resolved instruction bodies in all
3,155 probes (1,591 x86, 782 ARM, 782 WASM).

```bash
cargo test -p archmage-macros --lib
cargo test -p archmage-macros --lib --features avx512
cargo test --test macro_expand --test soundness_exploits
cargo test -p archmage-macros --lib profile_allocations -- --ignored --nocapture
cargo llvm-cov -p archmage -p archmage-macros --lib \
  --test macro_behavioral_contracts --test incant_macro --test tokenless_context \
  --include-build-script
python3 xtask/codegen.py 0373579 WORKTREE
# See the harness docstring for source archive and before/after tree preparation.
python3 xtask/macro_perf.py "$HOME/tmp/archmage-macro-perf"

# Profile the crate USING the macros, not just archmage-macros itself.
mkdir -p "$HOME/tmp/macro-profile"
CARGO_INCREMENTAL=0 cargo +nightly rustc -p magetypes --lib -- \
  -Zmacro-stats -Zself-profile="$HOME/tmp/macro-profile"
summarize summarize "$HOME"/tmp/macro-profile/*.mm_profdata
```

`-Zmacro-stats` describes expansion sizes/counts; it is not a timer.
`-Zself-profile` records rustc query/event time; use `perf` with call stacks for
hotspots inside the proc-macro library. The current measureme command is
`summarize summarize FILE.mm_profdata` (not a hard-coded `.pft` suffix).
See the [measureme instructions](https://github.com/rust-lang/measureme/tree/master/summarize)
and [rustc profiling guide](https://rustc-dev-guide.rust-lang.org/profiling.html).

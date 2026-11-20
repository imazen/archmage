# Performance

Archmage generates identical assembly to hand-written `#[target_feature]` + `unsafe` code. The safety abstractions compile away. At runtime, you get raw SIMD instructions.

Calling a proof wrapper (`#[attune(wrap)]` / `_tier_t`, or legacy `#[arcane]`) across a target-feature mismatch prevents inlining across that boundary. Compile-time costs are measured separately in the proc-macro section below.

## Zero overhead: archmage = bare `#[target_feature]`

Every benchmark pattern shows archmage and bare `#[target_feature]` producing the same timings. There is no wrapper cost, no token cost, no abstraction tax.

| What you write | Time (1000 x 8-float add) | What LLVM sees |
|----------------|---------------------------|----------------|
| Proof wrapper calling proof wrapper (matching features) | 547 ns | Features match — LLVM inlines wrapper |
| Bare `#[target_feature]` (no archmage) | 544 ns | Same |
| Proof wrapper per loop iteration from non-SIMD code | 2209 ns (4x) | Boundary crossing per call |
| Bare `#[target_feature]` per loop iteration | 2222 ns (4x) | Same boundary, same cost |

The 4x penalty comes from LLVM, not archmage.

## The target-feature boundary

`#[attune(wrap)]` (like legacy `#[arcane]`) reads the token type from your function signature to decide which `#[target_feature]` to emit. A function taking `X64V3Token` gets `#[target_feature(enable = "avx2,fma,...")]`.

A proof wrapper (`#[attune(wrap)]` or `#[attune] fn work_v3_t(...)`) emits an outer safe function that calls a private `#[target_feature]` inner function (`__arcane_*`) via `unsafe`. That is how you cross into SIMD code without writing `unsafe` yourself. When one V3 wrapper calls another V3 wrapper, LLVM inlines the wrapper away — no boundary. The boundary exists only when the caller has fewer features than the callee (for example, baseline code calling an AVX2 proof wrapper).

LLVM cannot inline a stronger-feature callee into a weaker-feature caller: no load hoisting, no store sinking, no cross-iteration vectorization across the mismatch. Matching or superset caller features inline normally.

A bare `#[target_feature]` function has the exact same cost. Archmage makes the wrapper safe; the boundary is LLVM's.

**The fix:** enter `#[attune(wrap)]` once from non-SIMD code, put the loop inside, and write internal helpers as direct feature bodies (`#[attune(v3)]` or `#[attune] fn helper_v3(...)`, which emit `#[target_feature]` + `#[inline]` directly with no wrapper). A proof-wrapper helper with matching features also inlines, as the cross-token table below shows, without saving the extra wrapper generation.

```rust
use archmage::prelude::*;

// WRONG: boundary every iteration (4x slower)
fn process_all_slow(points: &[[f32; 8]]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = X64V3Token::summon() {
        for p in points {
            let _ = process_one_wrapped(token, p); // #[attune(wrap)] — boundary crossing
        }
    }
}

// RIGHT: one boundary, loop inside
fn process_all(points: &[[f32; 8]]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = X64V3Token::summon() {
        process_all_simd(token, points); // one #[attune(wrap)] entry
    }
}

#[attune(wrap, import_intrinsics)]
fn process_all_simd(_token: X64V3Token, points: &[[f32; 8]]) {
    for p in points {
        let _ = process_one(p); // #[attune(v3)] — inlines, no boundary
    }
}

#[attune(v3, import_intrinsics)]
fn process_one(p: &[f32; 8]) -> __m256 {
    _mm256_loadu_ps(p)
}
```

## Benchmark results

All benchmarks from `cargo bench --bench asm_inspection --features "std avx512"`, run on x86-64 with AVX-512 support. Source: [`benches/asm_inspection.rs`](../benches/asm_inspection.rs) (the benchmark harness uses the legacy `#[arcane]` / `#[rite]` spellings, which expand through the same proof-wrapper and direct-body emitters as `#[attune(wrap)]` and `#[attune(v3)]`).

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

### Generic magetypes types: zero overhead inside `#[attune]` (with `#[inline(always)]`)

A common concern: does using `f32x8::<T>` with a generic `T: F32x8Backend` produce worse code than concrete `f32x8::<X64V3Token>`? **No — provided the generic function inlines into the `#[attune]` caller.**

The backend trait methods are all `#[inline(always)]`, which is necessary but not sufficient on its own. **Your generic helper function must also inline** into the `#[attune]` caller so LLVM compiles it within the `#[target_feature]` region. The generic function itself has no `#[target_feature]` — it inherits target features only by inlining into a function that has them.

Source: [`magetypes/benches/generic_vs_concrete.rs`](../magetypes/benches/generic_vs_concrete.rs). Measured 2026-10-05 on a Ryzen 9 9950X3D with Rust 1.99.0, median of three runs; record in [`benchmarks/generic_vs_concrete_zen5-9950x3d_2026-10-05.md`](../benchmarks/generic_vs_concrete_zen5-9950x3d_2026-10-05.md).

| Pattern | Time | Assembly |
|---------|------|----------|
| `f32x8::<T>` generic `#[inline(always)]` inside V3 wrapper | 1.08 ns | `vmovups` + `vaddps` + horizontal sum |
| `f32x8::<T>` generic (no annotation) inside V3 wrapper | 1.09 ns | identical — LLVM chose to inline (not guaranteed) |
| `f32x8::<X64V3Token>` concrete inside V3 wrapper | 1.09 ns | identical instructions |
| Concrete via direct V3 helper in V3 wrapper | 1.06 ns | identical |
| `f32x8::<T>` generic `#[inline(never)]` inside V3 wrapper | **8.13 ns (7.5x)** | SSE loads, stack spills, a call per backend method |
| `f32x8::<T>` generic **without** `#[target_feature]` | **7.95 ns (7.4x)** | the same calls |

The `#[inline(never)]` row isolates the cause: even when called from a V3 wrapper, a generic function that cannot inline is as slow as having no `#[target_feature]` at all. The generic function body is compiled without target features — it gets them only by inlining into the caller's `#[target_feature]` region. The multiplier belongs to this 8-lane kernel, where the calls dominate; the bench's first version, in February 2026, measured 18x on an older build.

**Mark generic SIMD helpers `#[inline(always)]`.** For small same-crate functions, LLVM usually inlines without annotation (the "no annotation" row above). That is an LLVM heuristic, not a guarantee. Cross-crate, without at least `#[inline]`, the function body isn't available to the caller's compilation unit. `#[inline(always)]` guarantees inlining so the generic body compiles with the caller's target features.

```rust
use archmage::{X64V3Token, attune};
use magetypes::simd::{backends::F32x8Backend, generic::f32x8};

// CORRECT: #[inline(always)] guarantees the generic body inlines into the caller
#[inline(always)]
fn generic_sum<T: F32x8Backend>(token: T, data: &[f32; 8]) -> f32 {
    let v = f32x8::<T>::from_array_t(token, *data);
    (v + v).reduce_add()  // ~1.1 ns from #[attune(wrap)]
}

// The #[attune(wrap)] caller provides #[target_feature] — generic_sum inlines into it
#[attune(wrap)]
fn entry(token: X64V3Token, data: &[f32; 8]) -> f32 {
    generic_sum(token, data)  // Inlines → full AVX2 codegen
}
```

```asm
; #[inline(always)] generic inside #[attune(wrap)] — identical to concrete:
vmovups ymm0, ymmword ptr [rdi]   ; load 8 floats
vaddps ymm0, ymm0, ymm0           ; v + v
vextractf128 xmm1, ymm0, 1        ; horizontal sum...
vaddps xmm0, xmm0, xmm1
vmovshdup xmm1, xmm0
vaddps xmm0, xmm0, xmm1
vshufpd xmm1, xmm0, xmm0, 1
vaddss xmm0, xmm0, xmm1

; #[inline(never)] generic inside #[attune(wrap)]:
movups xmm0, xmmword ptr [rdi]    ; SSE loads, not AVX
movups xmm1, xmmword ptr [rdi + 16]
; ... both halves spilled to the stack ...
call <X64V3Token as F32x8Backend>::add::__simd_inner_add
call <X64V3Token as F32x8Backend>::reduce_add::__simd_inner_reduce_add
```

**The rule:** generic magetypes code is zero-cost when it inlines into an `#[attune]` feature region (for example `#[attune(_*_t, define(f32x8))]`, `#[attune(wrap)]`, or `#[attune(v3)]`). Mark generic SIMD helpers `#[inline(always)]` to guarantee this.

## Rules

These are distilled from the benchmark data above.

1. **Enter a proof wrapper (`#[attune(wrap)]` / `_tier_t`) once.** Put loops inside it. Each call from non-SIMD code crosses the target-feature boundary.

2. **Use direct feature bodies (`#[attune(v3)]` / `_*`) for helpers.** They add `#[target_feature]` + `#[inline]` directly, with no wrapper. A proof-wrapper helper with matching features also inlines (V3→V3 = 1.0x above), as does a plain `#[inline(always)]` function.

3. **Don't cross feature mismatches in hot loops.** Calling a V3 wrapper from a V3 context is free — LLVM inlines the wrapper (V3→V3 = 1.0x). The boundary hurts when the caller has fewer features than the callee (V3→V4 = 4x).

4. **Downcasting is free.** A V4 function calling a V3 helper inlines because V4 is a superset of V3. Same for V3 calling V2.

5. **Upcasting hits the boundary.** A V3 function calling a V4 wrapper (`reattune!`) cannot inline because the caller lacks AVX-512 features. Dispatch at the entry point, not deep in hot code.

6. **Generic magetypes types are zero-cost inside `#[attune]` — if they inline.** `f32x8::<T>` and `f32x8::<X64V3Token>` produce identical assembly when the generic function inlines into the `#[attune]` caller. Mark generic SIMD helpers `#[inline(always)]` to ensure this. The generic function has no `#[target_feature]` of its own — if it doesn't inline, every backend method becomes a call (7.4x slower for the 8-lane kernel above; the cost depends on the kernel).

## Reproducing

```bash
# Simple vector add + DCT-8
cargo bench --bench asm_inspection --features "std"

# Cross-token nesting (needs avx512 feature + AVX-512 hardware)
cargo bench --bench asm_inspection --features "std avx512"

# Generic vs concrete magetypes
cargo bench -p magetypes --bench generic_vs_concrete
```

Results will vary by CPU. The *ratios* between patterns are stable: archmage always matches bare `#[target_feature]` on the same workload. The boundary multiplier itself (4x on simple adds, 6.2x on DCT-8) depends on how much optimization LLVM loses when it can't inline — denser workloads lose more.

`summon()` overhead is separate: ~1.3 ns cached, 0 ns with `-Ctarget-cpu=haswell` (compiles away). See [`benches/summon_overhead.rs`](../benches/summon_overhead.rs).

## Proc-macro parsing and allocation pass (2026-09-07)

Compared `0373579` with the parsing cleanup in `9382f9d` on the `wsl` Ryzen 9
9950X3D, Rust 1.98.1. This changes macro execution, not the generated algorithms.
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
These counts cover calls routed through the test allocator. Heaptrack subsequently
confirmed that dynamically linked libstd bypasses this allocator for some paths,
so the table is **partial allocation counts**, not complete per-expansion totals.
The consumer wall-time measurements below are independent of this limitation.

| Input | Before allocations | After allocations |
|---|---:|---:|
| `arcane`, ordinary kernel | 128 | 116 |
| `arcane`, nested dispatch | 166 | 163 |
| `rite`, ordinary kernel | 89 | 81 |
| `magetypes`, local vector alias | 300 | 240 |
| `autoversion`, scalar loop | 380 | 370 |

### Static-data follow-up

Compared `3860cb7` with `89fd76a`: borrow registry feature lists with `Cow`,
generate token-name-to-CSV lookups from the same registry, retain attributes once before
cloning variants, scan each magetypes body once for dispatch, move bodies into
the rewriter, and build identical dispatch arguments once outside the tier loop.
Feature-list uniqueness and first-seen union ordering are checked by tests.
Generic bounds retain the ordinary CSV join, preserving combination order.

Heaptrack, filtered to stacks containing `expansion_tests::expand`, counted
3,249,000 → 3,125,000 allocation calls over 1,000 repetitions of all five inputs
(**3.8% fewer**). This includes libstd allocations bypassing the test allocator.
Attribution to the nearest macro/dependency frame above the standard library:

| Requesting crate | Calls per five-input suite before → after |
|---|---:|
| `archmage-macros` | 69 → 52 |
| `proc_macro2` | 2,594 → 2,487 |
| `syn` | 569 → 569 |
| `quote` | 17 → 17 |

These are standalone-backend allocation calls, not peak memory, actual rustc
bridge counts, or evidence of a compile-time speedup. Dependency attribution
does not mean unavoidable cost: most of the reduction comes from asking
proc_macro2 to do less copying. The magetypes probe does not recursively expand
its emitted arcane attributes. All 138 unit/contract tests pass; expansion
snapshots and 3,155 resolved x86/ARM/WASM codegen probes are unchanged.

The follow-up repeated all 144 consumer builds on `wsl` with the same pinned
sources and dependency locks (six alternating pairs per configuration).
Paired geometric-mean changes and 95% intervals:

| Consumer | Profile | Cold build | Consumer arcane-body edit |
|---|---|---:|---:|
| linear-srgb | debug | -1.6% [-3.0, -0.2] | +3.5% [-0.5, +7.6] |
| linear-srgb | release | -0.3% [-2.0, +1.4] | -2.7% [-5.3, +0.0] |
| zenpixels-convert | debug | +0.3% [-1.4, +2.1] | -3.1% [-7.7, +1.8] |
| zenpixels-convert | release | -1.7% [-5.7, +2.5] | -2.6% [-9.1, +4.4] |

No configuration demonstrates a regression; the clearest improvement is the
small linear-srgb cold-debug result. Ordinary-function edit intervals all
include zero. This does not establish a broad compile-time speedup. Four alternating nightly
self-profile runs per version measured 608 proc-macro expansions in both:
66.55 → 65.14 ms median, with overlapping samples. All eight `-Zmacro-stats`
reports are identical.

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

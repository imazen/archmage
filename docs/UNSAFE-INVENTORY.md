# Remaining unsafe sites after PR #86

Audited commit: `9be0637` (native arcane generation and checked storage).
`from_context()` is independently proposed in [PR #87](https://github.com/imazen/archmage/pull/87).
That addition does not change the counts below.

## Scope and counting

Counts are lexical Rust source sites across all architectures and feature
configurations. Comments, string literals, and `cfg(test)` items are excluded.
Tests, examples, benches, expansion snapshots, dependencies, and xtask's emitted
string templates are not included in library totals. Macro expansion is not
counted as if it were checked-in source.

There are **377 explicit unsafe blocks** in `src/` and `magetypes/src/`:

| Area | Blocks | What they do |
| --- | ---: | --- |
| Generic SIMD wrapper implementations | 148 | 60 slice partitions, 60 indexing operations, 28 cross-wrapper reference casts |
| Generic block/view operations | 71 | 32 array/byte views, 16 byte constructors, 16 slice-to-wrapper casts, 6 reference bitcasts, 1 by-value bitcast |
| WASM backend | 60 | 20 loads, 20 stores, 10 array constructors, 10 array conversions |
| Token implementation generator output | 90 | 32 summon paths, 14 cold detection paths, 44 ancestor-token extractions |
| Cross-width x86 operations | 6 | Combine/extract halves using AVX/AVX-512 intrinsics |
| Checked storage module | 2 | One unaligned bit-copy primitive and one unaligned store primitive |
| Native V3/V4/NEON backend files | 0 | Intrinsics checked through arcane; memory operations use checked helpers |
| **Total** | **377** | |

Other source occurrences are **54 unsafe function declarations** (53 token
forge declarations and the unused `Upcast::upcast` trait method), **1 unsafe
trait declaration**, and **2 unsafe impl source sites** in the POD machinery.
One impl site is a macro expanded for the permitted primitive/vector types;
these are source-site counts, not expanded impl counts. Token forge declarations
include mutually exclusive visibility configurations and architecture stubs.

That is **434 code-keyword occurrences** in the two library source trees,
versus **454 raw word occurrences** including comments/literals. The proc-macro
sources have another **6 unsafe-emission sites**: four alternative arcane wrapper
templates and two autoversion templates. These are not six runtime operations:
the native backend alone has 2,206 arcane methods, each with a generated unsafe
feature-entry boundary. xtask has no executable unsafe sites; its remaining
mentions are generator strings or commentary.

The previous WASM figure of 62 counted two documentation mentions. The actual
number of blocks is 60. Historic counts in SOUNDNESS.md were also from an older
repository revision; the table above is the current full library inventory.

## Recommended order

### 1. Replace slice partitioning: remove 60 blocks without adding new unsafe

Change `gen_partition_slice` and `gen_partition_slice_mut` in
[`generic_gen/type_impl.rs`](../xtask/src/simd_types/generic_gen/type_impl.rs)
to emit `data.as_chunks::<LANES>()` and `data.as_chunks_mut::<LANES>()`.
Their results already have the exact existing shape: array chunks plus remainder.
All generated lane counts are nonzero.

Both APIs are [stable since Rust 1.88](https://doc.rust-lang.org/std/primitive.slice.html#method.as_chunks),
below this repository's Rust 1.89 minimum. This also removes the hand-written
pointer/length arithmetic and the redundant formatting branches in the generator.
Verify empty slices, short slices, exact multiples, tails, and mutable writeback.

### 2. Extend checked storage to WASM: centralize another 60 blocks

Permit the private storage module on wasm32, implement its POD contract for
`v128`, and update the WASM generator templates to emit checked loads/stores
and array casts, just as the native generators do. These 60 blocks are all
whole-array memory operations; the value intrinsics are already safe.

This adds a POD implementation to audit, but needs no new unsafe block beyond
the existing two primitives. Verify WASM round trips, unaligned input/output,
wide polyfill lane order, and optimized load/store codegen.

**These first two changes remove 120 of 377 blocks (31.8%) from library source.**
That is a smaller local audit surface, not a claim that the underlying memory
operations disappear.

### 3. Rebuild byte/value constructors from storage plus the real token: 17 sites

The 16 `from_bytes`/`from_bytes_owned` methods currently transmute the entire
wrapper and discard the supplied token value. Copy bytes into the known scalar
lane array with checked storage, then use the existing token-gated array
constructor. Likewise, replace the one whole-wrapper `u32x4` → `f32x4` value
transmute with a representation/array conversion that carries the existing token.

This makes the capability proof explicit and avoids depending on the token's
zero-sized layout for construction. Test NaN payloads, signed zero, arbitrary
integer bits, and optimized codegen before accepting the conversion path.

### 4. Consolidate views and indexing: 92 sites, with stronger layout checks

The 60 Index/IndexMut sites and 32 array/byte view sites should share checked
reference-cast helpers analogous to fearless_simd's `checked_cast_ref/mut`.
Use the representation field, not the entire token-bearing wrapper. The helper
must enforce equal size, sufficient source alignment, initialized storage, and
valid destination bit patterns. Preserve the existing index bounds behavior.

This requires making the representation's POD/layout guarantee available to
generic code (through a private/sealed contract or backend accessors). Only eight
wrapper types currently have block-op array views, so simply calling `as_array()`
from every Index impl would not work without extending that support.

Expect a pair of shared unsafe reference primitives instead of 92 separate
casts. This is centralization, with new trait/layout obligations to verify,
not an unsafe-free implementation. Run Miri for aliasing, alignment, bounds,
mutable views, and representation changes across backends.

### 5. Treat wrapper-reference/slice conversions separately: 50 sites

There are 34 cross-wrapper reference bitcasts and 16 casts from scalar slices
into token-bearing wrapper slices. These need a capability-aware, sealed layout
contract, not the plain storage POD trait. Preserve the same token tier, require
an existing token for slice-to-wrapper conversion, and check alignment, length,
size, and exclusive mutable borrowing.

**Do not implement generic POD for tokens or public token-bearing SIMD wrappers.**
That would allow checked storage helpers to manufacture feature proofs from
arbitrary bytes. Consolidation here is worthwhile only if it makes these
obligations easier to enforce and review.

### 6. Convert the six cross-width intrinsic methods to arcane

[`generic/cross_width.rs`](../magetypes/src/simd/generic/cross_width.rs) still
has six feature-justified unsafe blocks. Its receivers are concrete tokens,
so the same `_self = Token` form can make the compiler check the feature lists.
Validate half ordering and inlining. This removes manual intrinsic auditing,
but moves the feature-entry boundary into the macro rather than eliminating it.

## Unsafe worth keeping

- **The two storage primitives and their narrow POD contract.** They are the
  intentional low-level implementation behind safe memory APIs. Keep them small
  and audited; fewer keywords alone would not improve safety.
- **Runtime feature-entry boundaries.** Runtime detection cannot generally be
  proved by rustc's target-feature checker. arcane's boundary is the bridge.
  `from_context()` does not remove it when entering from a baseline caller.
- **Token detection and extraction.** Of the 90 blocks, 46 concern detection or
  caches and 44 concern ancestor extraction. After both PRs land, extraction
  could use arcane plus `AncestorToken::from_context()` for compiler-checked
  feature inclusion, but that relocates its boundary into the macro. Replacing
  forge calls with private-field literals merely hides the same proof obligation.
- **Explicit unsafe forge APIs.** Their unsafety is the public contract. Most
  of the 53 declarations are generated/cfg duplication, not distinct algorithms.
  Consolidating their syntax has much less value than the reductions above.
- **Upcast::upcast.** No implementations or uses were found in the repository.
  Consider deprecating it or requiring an explicit destination token when the
  API is revisited; removing/changing a public trait is not a routine cleanup.

This report recommends follow-ups; it does not implement those additional reductions.

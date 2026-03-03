# Feature Requests from zenjxl-decoder

Compiled March 2026 from a soundness audit of `zenjxl-decoder-simd` (the
JPEG XL decoder's SIMD abstraction layer). These are operations the decoder
uses in hot paths that magetypes does not yet provide.

Context: zenjxl-decoder has a hand-rolled SIMD abstraction (`jxl_simd`) with
298 `unsafe` uses across 7 files. The safe public API is unsound — `load`,
`store`, and offset helpers use `debug_assert!` + raw pointer ops, so safe
code can trigger UB in release builds. We want to migrate to magetypes for
provable soundness, but these gaps must be closed first.

Priority scale: **P0** = blocks migration, **P1** = needed for full feature
parity, **P2** = nice to have / can workaround.

---

## P0 — Blocks migration

### 1. `neg_mul_add(self, mul, add)` → `add - self * mul` (FNMADD)

**Backend trait method on f32xN.**

The decoder uses this 60+ times across XYB color transform, EPF (edge-preserving
filter), transfer function curves, and DCT butterflies. It maps directly to
x86 `vfnmadd{132,213,231}ps` and ARM `vfms`. Without a dedicated method, the
workaround is `add - self * mul`, which the compiler *might* fuse but cannot
guarantee — breaking floating-point determinism and losing ~1 cycle/vector on
the critical path.

`mul_sub` (= `self * a - b`) already exists. This is its complement:
`neg_mul_add` = `-(self * mul) + add` = `add - self * mul`.

### 2. Interleave/deinterleave for 2, 3, and 8 channels (f32)

**Block ops on f32x4, f32x8, f32x16.**

The decoder interleaves/deinterleaves pixel channels constantly:
- **2-channel**: stereo/alpha pairs (squeeze transforms)
- **3-channel**: RGB output without alpha
- **4-channel**: RGBA (already in magetypes as `interleave_4ch`)
- **8-channel**: used in the DCT coefficient save/restore pipeline

Current signatures:
```rust
fn store_interleaved_2(a: Self, b: Self, dest: &mut [f32]);
fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [f32]);
fn store_interleaved_8(a..h: Self, dest: &mut [f32]);
fn load_deinterleaved_2(d: D, src: &[f32]) -> (Self, Self);
fn load_deinterleaved_3(d: D, src: &[f32]) -> (Self, Self, Self);
```

The 4-channel variants already exist. Extending to 2/3/8 completes the set.

### 3. Interleave for u8 and u16 types (2, 3, 4 channels)

**Block ops on u8xN and u16xN.**

The output pipeline writes pixels directly as interleaved u8 or u16 depending
on output bit depth. These are final-stage operations:

```rust
// On u8xN:
fn store_interleaved_2(a: Self, b: Self, dest: &mut [u8]);
fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u8]);
fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u8]);

// Same for u16xN.
```

### 4. `round_store_u8` / `round_store_u16` on f32xN

**Cross-type SIMD operation: f32 → round → saturate → pack → store as u8/u16.**

12 call sites in the output conversion stage. This is a single SIMD pipeline
on every architecture:
- x86: `vroundps` + `vcvttps2dq` + `vpackssdw` + `vpackuswb` + `vmovdqu`
- ARM: `vcvtns` + `vqmovn` + `vst1`
- WASM: `f32x4_nearest` + `i32x4_trunc_sat` + narrowing + store

The magetypes `to_u8()` method returns `[u8; N]` via scalar extraction — it
doesn't use the SIMD pack instructions. A proper `round_store_u8(&self, dest)`
or `round_to_u8(self) -> u8xN` would be significantly faster.

### 5. `MaybeUninit` output variants for interleave stores

**All interleave store methods need `_uninit` variants.**

The decoder pre-allocates output buffers as `Vec<MaybeUninit<T>>` and writes
into them without zeroing first. This avoids a memset on multi-megapixel
buffers. The interleave stores need variants that accept `&mut [MaybeUninit<T>]`:

```rust
fn store_interleaved_3_uninit(a, b, c, dest: &mut [MaybeUninit<f32>]);
```

The `unsafe trait` contract is that implementations must write initialized
data to all positions they claim to fill. This is the same pattern as
`slice::write` on `MaybeUninit`.

---

## P1 — Needed for full feature parity

### 6. `mul_wide_take_high` on i32xN

**`(self as i64 * rhs as i64) >> 32` per lane, returning i32xN.**

Used in the modular transform pipeline for fixed-point arithmetic. Maps to
`_mm_mul_epi32` + shift on x86, `vmull` + `vshrn` on ARM.

20 call sites in squeeze/modular transforms.

### 7. `store_u16` on i32xN (truncate + pack + store)

**Truncates each i32 lane to u16 and stores to `&mut [u16]`.**

12 call sites in the modular squeeze transform. Maps to `vpackusdw` + store
on x86, `vqmovun` + `vst1` on ARM.

### 8. `load_f16_bits` / `store_f16_bits` (f16 ↔ f32 conversion)

**Loads u16 f16 bit patterns and converts to f32xN, and the reverse.**

Used in the HDR output pipeline. Maps to F16C `vcvtph2ps`/`vcvtps2ph` on x86
(V3 tokens have F16C), NEON `vcvt_f32_f16`/`vcvt_f16_f32` on ARM, and
software fallback on WASM.

```rust
fn load_f16_bits(token: T, mem: &[u16]) -> f32xN<T>;
fn store_f16_bits(self, dest: &mut [u16]);
```

### 9. `prepare_table_bf16_8` / `table_lookup_bf16_8`

**8-entry f32 lookup table using BF16 packing for fast SIMD lookup.**

The decoder pre-packs 8 f32 values as BF16 (upper 16 bits of each f32) into
a SIMD register, then uses `pshufb`/`tbl` to look up by index. This gives
approximate f32 lookups in a single SIMD instruction. Used in transfer function
curves.

The BF16 precision loss (~0.4% relative error) is acceptable for the use case.

```rust
type Bf16Table8; // Prepared table type (opaque per-backend)
fn prepare_table_bf16_8(token: T, table: &[f32; 8]) -> Bf16Table8;
fn table_lookup_bf16_8(token: T, table: Bf16Table8, indices: i32xN<T>) -> f32xN<T>;
```

### 10. `copysign(self, sign)` on f32xN

**Copies the sign bit from `sign` to `self`, preserving magnitude.**

15 call sites in XYB inverse transform. Maps to `vblendvps` with sign mask
on x86, `vbsl` on ARM.

Can be built from bitwise ops (`self.abs() | (sign & splat(-0.0))`), but a
dedicated method would be cleaner and lets backends use optimal instructions.

### 11. `andnot` on mask/comparison results

**`!self & rhs` — bitwise AND-NOT.**

Used 15+ times for mask composition in EPF and conditional blend logic. Maps
directly to `vandn` on x86, `vbic` on ARM.

Currently buildable as `self.not() & rhs` (assuming NOT exists), but the
fused instruction is a single cycle vs two on all architectures.

---

## P2 — Nice to have / can workaround

### 12. `test_all_instruction_sets!` macro

**Generates per-ISA test variants from a single generic test function.**

The decoder test suite uses this extensively (155 test functions generated from
~50 generic test bodies). Without it, test authors must manually write per-token
test wrappers. Low priority since it's just test ergonomics.

### 13. `bench_all_instruction_sets!` macro

Same as above but for criterion benchmarks.

### 14. `maskz_i32(mask, v)` — zero-masked i32

**`if mask { zero } else { v }` — equivalent to `v & ~mask`.**

Trivially buildable as `blend(mask, zero, v)`. Dedicated method is a minor
ergonomic improvement.

### 15. `lt_zero` / `eq_zero` convenience methods on i32xN

Trivially buildable as `self.simd_lt(zero)` / `self.simd_eq(zero)`.

---

## Structural notes for migration

### Mask type difference

jxl_simd has a separate `SimdMask` trait. magetypes uses the vector type
itself as mask (comparisons return f32xN/i32xN with all-1s/all-0s lanes).
This is fine — the migration just replaces `mask.if_then_else_f32(t, f)` with
`f32xN::blend(mask, t, f)`. The type system actually gets simpler.

### Descriptor bundling

jxl_simd bundles all associated types in one `SimdDescriptor` trait:
```rust
fn foo<D: SimdDescriptor>(d: D, ...) { D::F32Vec::load(d, ...); }
```

magetypes uses separate backend traits per type:
```rust
fn foo<T: F32x8Backend + I32x8Backend + F32x8Convert>(t: T, ...) {
    f32x8::load(t, ...);
}
```

A `SimdTypes` supertrait or a convenience alias combining the common bounds
would reduce boilerplate at call sites. Something like:
```rust
trait JxlSimdToken: F32x4Backend + I32x4Backend + U32x4Backend
    + U8x16Backend + U16x8Backend + F32x4Convert { }
```

### Width dispatch

jxl_simd uses `Descriptor256` / `Descriptor128` associated types for code that
needs to operate at a specific width regardless of the "best" ISA. magetypes
uses `WidthDispatch` + `.low()` / `.high()` / `from_halves()`. The magetypes
approach is actually cleaner since width is in the type, not a runtime downgrade.

### `load_from` / `store_at` (unchecked offset operations)

jxl_simd has these as safe methods with `debug_assert!` + `get_unchecked` —
they're unsound (103 call sites). magetypes correctly omits them. The
migration should replace them with checked slicing or, where the bounds are
provably correct, `unsafe` blocks with proper safety comments at each call site.
This makes the safety obligation explicit rather than hidden behind a "safe" API.

**Do NOT add unchecked offset load/store to magetypes.** The whole point of
migrating is to make unsoundness impossible through the safe API.

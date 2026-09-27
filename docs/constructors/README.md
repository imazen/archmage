# Complete constructor signatures

Each page contains every public inherent method whose first argument is a token,
plus the feature-context `from_raw` constructor. Signatures, bounds, and explicit
`cfg` / `target_feature` attributes are extracted by the alias generator.
`Self` refers to the surrounding impl. These are declarations, not compilable impl bodies.

The existing names retain their arguments. Migration methods append `_t` and
**keep the token first**: `splat_t(token, value)`, not `splat_t(value, token)`.
Native names already ending in `_t` use the uniform `from_raw_t(token, raw)`;
there are no `_t_t` methods. `from_raw(raw)` alone requires the displayed CPU
features in its caller. Explicit-token constructors require no caller attributes.
The new spellings are unreleased additions to the published 0.9.29 surface;
restored NEON/WASM/AVX-512 native names were missing in that published version.

Generic vector paths are `magetypes::simd::generic::TYPE<T>`; single-lane paths
are `magetypes::simd::scalar::TYPE`. The 512-bit types require `w512` even when a
particular impl below has no explicit gate. Their native AVX-512 raw constructors
additionally require `avx512`. `T` must satisfy the listed backend/conversion bounds.
Trait methods, receiver operations, and implicit conversions are outside this
explicit-construction inventory. See [the migration guide](../TOKEN-CONSTRUCTOR-MIGRATION.md).

- [f32x1](f32x1.md) · [f32x4](f32x4.md) · [f32x8](f32x8.md) · [f32x16](f32x16.md)
- [f64x1](f64x1.md) · [f64x2](f64x2.md) · [f64x4](f64x4.md) · [f64x8](f64x8.md)
- [i8x1](i8x1.md) · [i8x16](i8x16.md) · [i8x32](i8x32.md) · [i8x64](i8x64.md)
- [u8x1](u8x1.md) · [u8x16](u8x16.md) · [u8x32](u8x32.md) · [u8x64](u8x64.md)
- [i16x1](i16x1.md) · [i16x8](i16x8.md) · [i16x16](i16x16.md) · [i16x32](i16x32.md)
- [u16x1](u16x1.md) · [u16x8](u16x8.md) · [u16x16](u16x16.md) · [u16x32](u16x32.md)
- [i32x1](i32x1.md) · [i32x4](i32x4.md) · [i32x8](i32x8.md) · [i32x16](i32x16.md)
- [u32x1](u32x1.md) · [u32x4](u32x4.md) · [u32x8](u32x8.md) · [u32x16](u32x16.md)
- [i64x1](i64x1.md) · [i64x2](i64x2.md) · [i64x4](i64x4.md) · [i64x8](i64x8.md)
- [u64x1](u64x1.md) · [u64x2](u64x2.md) · [u64x4](u64x4.md) · [u64x8](u64x8.md)

# archmage examples

## In this directory

| Example | Description |
|---------|-------------|
| `alpha_blend` | Premultiply, unpremultiply, Porter-Duff compositing with raw AVX2 intrinsics |
| `vertical_reduce` | Fixed-point vertical image reduction (separable resampling kernel) |
| `cpu_survey` | Print detected CPU features and available tokens |
| `detect_features` | Feature detection diagnostics |

Run any example with:
```sh
cargo run --example alpha_blend --release
```

## magetypes examples

For examples using the higher-level `f32x8` / `i32x8` types (from the [magetypes](https://docs.rs/magetypes) crate) alongside `#[arcane]`, see [`magetypes/examples/`](https://github.com/imazen/archmage/tree/main/magetypes/examples):

| Example | Description |
|---------|-------------|
| [alpha_blend](https://github.com/imazen/archmage/blob/main/magetypes/examples/alpha_blend.rs) | Same operations as above, using `f32x8` for ergonomic load/store/blend |
| [color_convert](https://github.com/imazen/archmage/blob/main/magetypes/examples/color_convert.rs) | YUV ↔ RGB with BT.601 coefficients |
| [convolution](https://github.com/imazen/archmage/blob/main/magetypes/examples/convolution.rs) | Vertical reduction, box filter, Gaussian blur |
| [fast_dct](https://github.com/imazen/archmage/blob/main/magetypes/examples/fast_dct.rs) | DCT-8x8 for JPEG encoding via vectorized matrix multiply |
| [simd_kernels](https://github.com/imazen/archmage/blob/main/magetypes/examples/simd_kernels.rs) | 8 production image processing kernels (DCT, color, quantization, sRGB) |
| [cross_platform](https://github.com/imazen/archmage/blob/main/magetypes/examples/cross_platform.rs) | Portable SIMD with auto-dispatch across x86/ARM/WASM |
| [generic_simd](https://github.com/imazen/archmage/blob/main/magetypes/examples/generic_simd.rs) | Backend-generic algorithms (dot product, cosine similarity) |
| [magetypes_showcase](https://github.com/imazen/archmage/blob/main/magetypes/examples/magetypes_showcase.rs) | Side-by-side raw intrinsics vs magetypes API |

Run magetypes examples with:
```sh
cargo run -p magetypes --example color_convert --release
```

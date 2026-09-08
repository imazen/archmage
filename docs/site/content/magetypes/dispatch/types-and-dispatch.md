+++
title = "Types and Dispatch"
weight = 1
+++

Use Rust generics for data representation and algorithm parameters, and
`#[magetypes]` for ISA specialization. They compose in the same function.
`define(f32x8)` is optional shorthand for the generic vector type with the
current `Token`; it does not replace Rust's function generics.

## A generic input type and a const mode

This teaching extraction follows `zenanalyze/src/tier1.rs`:
`accumulate_row` selects const modes, then `incant!` calls
`accumulate_row_simd<const BT601: bool, const FULL: bool, const SKIN: bool,
R: ChunkInput>`. The production kernel accumulates several image statistics.
Here we retain the input-type and luma-coefficient specialization, and compute
only the luma sum. The small `Pixel` trait replaces the production chunk-loader
trait so the complete example needs no image framework.

```rust
use archmage::prelude::*;

pub trait Pixel: Copy {
    fn rgb(self) -> [f32; 3];
}
impl Pixel for [u8; 3] {
    fn rgb(self) -> [f32; 3] { self.map(f32::from) }
}
impl Pixel for [u8; 4] {
    fn rgb(self) -> [f32; 3] {
        [f32::from(self[0]), f32::from(self[1]), f32::from(self[2])]
    }
}

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn luma_impl<const BT601: bool, P: Pixel>(
    token: Token, pixels: &[P], weights: [f32; 3],
) -> f64 {
    let [kr, kg, kb] = if BT601 { [0.299, 0.587, 0.114] } else { weights };
    let mut sum = 0.0_f64;
    let (chunks, tail) = pixels.as_chunks::<8>();
    for chunk in chunks {
        let rgb = chunk.map(Pixel::rgb);
        let r = f32x8::from_array(token, core::array::from_fn(|i| rgb[i][0]));
        let g = f32x8::from_array(token, core::array::from_fn(|i| rgb[i][1]));
        let b = f32x8::from_array(token, core::array::from_fn(|i| rgb[i][2]));
        let y = r * f32x8::splat(token, kr)
              + g * f32x8::splat(token, kg)
              + b * f32x8::splat(token, kb);
        sum += f64::from(y.reduce_add());
    }
    for &pixel in tail {
        let [r, g, b] = pixel.rgb();
        sum += f64::from(r * kr + g * kg + b * kb);
    }
    sum
}

pub fn luma_sum<const BT601: bool, P: Pixel>(pixels: &[P], weights: [f32; 3]) -> f64 {
    incant!(luma_impl::<BT601, P>(pixels, weights), [v3, neon, wasm128, scalar])
}

// Values are encoded RGB channel samples; this is a signal statistic,
// not a linear-light color conversion. RGBA alpha is ignored.
let rgb = [[10u8, 20, 30]; 11];
let rgba = [[10u8, 20, 30, 128]; 11];
assert_eq!(luma_sum::<false, _>(&rgb, [1.0, 0.0, 0.0]), 110.0);
assert_eq!(luma_sum::<false, _>(&rgba, [0.0, 1.0, 0.0]), 220.0);
assert!((luma_sum::<true, _>(&rgb, [0.0; 3]) - 199.65).abs() < 0.001);
assert_eq!(luma_sum::<true, [u8; 3]>(&[], [0.0; 3]), 0.0);
```

`P` is statically resolved. `BT601` selects coefficients at specialization time.
The ISA token is independently substituted by the macro. `define(f32x8)` binds
the same generic vector implementation you could spell [`f32x8::<Token>`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html).
There is no trait-object dispatch here. The array construction is a simple
teaching loader; production `ChunkInput` implementations specialize loading and
layout too. Verify their assembly rather than assuming this loader is optimal.

## Other production combinations

| Source | Generic specialization |
|---|---|
| `zenavif/src/yuv_convert.rs`, `yuv420_strip_kernel<S: YuvSample, P: StripPixel>` | u8/u16 input samples and RGB/RGBA output; macro supplies ISA features for ordinary loops |
| `zenpixels-convert/src/scan.rs`, `fused_cg_impl<const A: bool, const B: bool>` | Opacity and grayscale checks; caller passes `::<A, B>` through `incant!` |
| `zenpng/src/simd/scan.rs`, `fused_cg_impl<const A: bool, const B: bool, const C: bool>` | Adds binary-alpha scanning; same const-plus-ISA arrangement |
| `zenpixels-convert/src/hdr/measure.rs`, `accumulate_strip_max_rgb_tier<const N: usize>` | RGB/RGBA stride with alpha ignored |
| `linear-srgb/src/simd.rs` → `tf/srgb.rs` | Concrete generated slice loop calls a backend-generic vector-to-vector helper |

Specialization multiplies the reachable combinations of tiers, types, and
constants. Use const parameters when they remove meaningful work, not for every
runtime setting. Inspect cold compile time and code size as well as runtime.

## Reusing the backend-generic part

Extract an algorithm generic over `T: F32x8Backend` when several entries need it.
Keep the feature-enabled caller and inspect inlining. The complete
[zenblend chain](@/magetypes/examples/pixel-blending.md) and
[zenfilters helper chain](@/magetypes/examples/generic-kernels.md) show this.
`#[magetypes]` can call ordinary helpers and other per-tier helpers; it is not
restricted to leaf functions.

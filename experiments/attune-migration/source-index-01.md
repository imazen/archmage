# Compact source migration index

Exact old macro/call → destination pattern. Signatures are lexical, including macro templates, not compiler-resolved. Original file links plus L numbers locate every item. All support attributes and textual mentions remain in the raw inventory.

## [examples/alpha_blend.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/examples/alpha_blend.rs#L28)

- L28 [source]: `#[arcane(import_intrinsics)]` on `fn premultiply_2px(_token: X64V3Token, pixels: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L51 [source]: `#[arcane(import_intrinsics)]` on `fn unpremultiply_2px(_token: X64V3Token, pixels: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L77 [source]: `#[arcane(import_intrinsics)]` on `fn composite_over_2px(_token: X64V3Token, src: &[f32; 8], dst: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [examples/vertical_reduce.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/examples/vertical_reduce.rs#L35)

- L35 [source]: `#[arcane(import_intrinsics)]` on `pub fn reduce_vertical_u8( _token: X64V3Token, inputs: &[&[u8]], weights: &[i16], output: &mut [u8], )` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/examples/accuracy_test.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/accuracy_test.rs#L118)

- L118 [source]: `#[arcane]` on `fn exp2_lowp_chunk(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L124 [source]: `#[arcane]` on `fn log2_lowp_chunk(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L130 [source]: `#[arcane]` on `fn pow_lowp_chunk(token: X64V3Token, input: &[f32; 8], exp: f32) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L136 [source]: `#[arcane]` on `fn pow_midp_chunk(token: X64V3Token, input: &[f32; 8], exp: f32) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/examples/alpha_blend.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/alpha_blend.rs#L37)

- L37 [source]: `#[arcane]` on `fn premultiply_2px(token: X64V3Token, pixels: f32x8) -> f32x8` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L61 [source]: `#[arcane]` on `pub fn premultiply_alpha_simd(token: X64V3Token, data: &mut [f32])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L95 [source]: `#[arcane]` on `fn unpremultiply_2px(token: X64V3Token, pixels: f32x8) -> f32x8` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L133 [source]: `#[arcane]` on `pub fn unpremultiply_alpha_simd(token: X64V3Token, data: &mut [f32])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L173 [source]: `#[arcane]` on `fn composite_over_2px(token: X64V3Token, src: f32x8, dst: f32x8) -> f32x8` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L196 [source]: `#[arcane]` on `pub fn composite_over_simd(token: X64V3Token, src: &[f32], dst: &mut [f32])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/examples/color_convert.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/color_convert.rs#L44)

- L44 [source]: `#[arcane]` on `fn yuv_to_rgb_f32x8(token: X64V3Token, y: f32x8, u: f32x8, v: f32x8) -> (f32x8, f32x8, f32x8)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L79 [source]: `#[arcane]` on `fn rgb_to_yuv_f32x8(token: X64V3Token, r: f32x8, g: f32x8, b: f32x8) -> (f32x8, f32x8, f32x8)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L139 [source]: `#[arcane]` on `fn load_hi_16(token: X64V2Token, src: &[u8]) -> __m128i` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L156 [source]: `#[arcane]` on `fn yuv_to_rgb_fixed_8( token: X64V2Token, y: &[u8], u: &[u8], v: &[u8], r_out: &mut [u8], g_out: &mut [u8], b_out: &mut [u8], )` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/examples/convolution.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/convolution.rs#L35)

- L35 [source]: `#[arcane]` on `pub fn reduce_vertical_f32_simd( token: X64V3Token, inputs: &[&[f32]], weights: &[f32], output: &mut [f32], )` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L92 [source]: `#[arcane]` on `pub fn reduce_vertical_u8_simd( token: X64V3Token, inputs: &[&[u8]], weights: &[i16], output: &mut [u8], )` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L187 [source]: `#[arcane]` on `pub fn box_filter_3x3_f32( token: X64V3Token, input: &[f32], output: &mut [f32], width: usize, height: usize, )` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/examples/cross_platform.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/cross_platform.rs#L66)

- L66 [source]: `#[arcane]` on `fn sum_of_squares_avx2(token: archmage::X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L89 [source]: `#[arcane]` on `fn sum_of_squares_sse(token: archmage::X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L112 [source]: `#[arcane]` on `fn sum_of_squares_neon(token: archmage::NeonToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L173 [source]: `#[arcane]` on `fn polynomial_eval_avx2(token: archmage::X64V3Token, data: &mut [f32], a: f32, b: f32, c: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L203 [source]: `#[arcane]` on `fn polynomial_eval_neon(token: archmage::NeonToken, data: &mut [f32], a: f32, b: f32, c: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/examples/edge_case_test.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/edge_case_test.rs#L14)

- L14 [source]: `#[arcane]` on `fn test_exp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L19 [source]: `#[arcane]` on `fn test_exp2(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L24 [source]: `#[arcane]` on `fn test_log2(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L29 [source]: `#[arcane]` on `fn test_ln(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L34 [source]: `#[arcane]` on `fn test_cbrt(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/examples/fast_dct.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/fast_dct.rs#L45)

- L45 [source]: `#[arcane]` on `fn dct1d_8( token: X64V3Token, v0: f32x8, v1: f32x8, v2: f32x8, v3: f32x8, v4: f32x8, v5: f32x8, v6: f32x8, v7: f32x8, ) -> [f32x8; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L174 [source]: `#[arcane]` on `fn transpose_8x8_vecs(token: X64V3Token, rows: &[f32x8; 8]) -> [f32x8; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L231 [source]: `#[arcane]` on `fn load_block(token: X64V3Token, block: &[f32; 64]) -> [f32x8; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L246 [source]: `#[arcane]` on `fn store_block(token: X64V3Token, vecs: &[f32x8; 8], block: &mut [f32; 64])` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L261 [source]: `#[arcane]` on `pub fn fast_dct8x8(token: X64V3Token, block: &mut [f32; 64])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L287 [source]: `#[arcane]` on `pub fn fast_dct8x8_batch(token: X64V3Token, blocks: &mut [[f32; 64]])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/examples/generic_simd.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/generic_simd.rs#L323)

- L323 [source]: `#[archmage::arcane]` on `fn bench_dot_avx2( _token: archmage::X64V3Token, a: &[f32], b: &[f32], iters: u32, ) -> std::time::Duration` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/examples/idiomatic_patterns_all.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/idiomatic_patterns_all.rs#L50)

- L50 [source]: `#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]` on `fn scale_plane_impl(token: Token, plane: &mut [f32], factor: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L67 [source]: `incant!( scale_plane_impl(plane, factor), [v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar] )` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L101 [source]: `#[magetypes(v4, v3, neon, wasm128, scalar)]` on `fn dot_impl(token: Token, a: &[f32], b: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L107 [source]: `incant!(dot_impl(a, b))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L132 [source]: `#[arcane]` on `fn scale_plane_impl_v4x(token: X64V4xToken, plane: &mut [f32], factor: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L153 [source]: `#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]` on `fn clamp01_impl(token: Token, plane: &mut [f32])` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L173 [source]: `#[autoversion]` on `fn apply_color_matrix(rgb: &mut [f32], mat: [[f32; 3]; 3])` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L193 [source]: `#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]` on `fn pipeline_impl(token: Token, plane: &mut [f32], bias: f32, factor: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L208 [source]: `incant!(clamp01_impl(plane))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L212 [source]: `incant!(scale_plane_impl(plane, factor))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L216 [source]: `incant!(pipeline_impl(plane, bias, factor))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [magetypes/examples/isa_fixups.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/isa_fixups.rs#L111)

- L111 [source]: `#[arcane(import_intrinsics)]` on `pub fn run<const OP: u8, const FIX: bool>( token: X64V3Token, input: &[u32], output: &mut [u32], )` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L184 [source]: `#[arcane(import_intrinsics)]` on `pub fn run<const OP: u8, const FIX: bool>( token: X64V4Token, input: &[u32], output: &mut [u32], )` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L259 [source]: `#[arcane]` on `pub fn run<const OP: u8, const FIX: bool>(token: NeonToken, input: &[u32], output: &mut [u32])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L307 [source]: `#[arcane]` on `pub fn run<const OP: u8, const FIX: bool>( token: Wasm128Token, input: &[u32], output: &mut [u32], )` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/examples/magetypes_showcase.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/magetypes_showcase.rs#L32)

- L32 [source]: `#[arcane]` on `pub fn dot_product_raw(_token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L48 [source]: `#[arcane]` on `pub fn dot_product_clean(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L56 [source]: `#[arcane]` on `pub fn fma_raw(_token: X64V3Token, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L68 [source]: `#[arcane]` on `pub fn fma_clean(token: X64V3Token, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L85 [source]: `#[arcane]` on `pub fn vector_math(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L101 [source]: `#[arcane]` on `pub fn integer_ops(token: X64V3Token, a: &[i32; 8], b: &[i32; 8]) -> [i32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L123 [source]: `#[arcane]` on `pub fn statistics(token: X64V3Token, data: &[f32; 8]) -> (f32, f32, f32)` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L134 [source]: `#[arcane]` on `pub fn clamped_normalize(token: X64V3Token, data: &[f32; 8], lo: f32, hi: f32) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L148 [source]: `#[arcane]` on `pub fn abs_values(token: X64V3Token, data: &[f32; 8]) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L154 [source]: `#[arcane]` on `pub fn sqrt_and_reciprocals( token: X64V3Token, data: &[f32; 8], ) -> ([f32; 8], [f32; 8], [f32; 8])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L168 [source]: `#[arcane]` on `pub fn floor_ceil_round(token: X64V3Token, data: &[f32; 8]) -> ([f32; 8], [f32; 8], [f32; 8])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L189 [source]: `#[arcane]` on `pub fn softmax_chunk(token: X64V3Token, logits: &[f32; 8], max_val: f32) -> ([f32; 8], f32)` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L203 [source]: `#[arcane]` on `pub fn gamma_correction(token: X64V3Token, pixels: &[f32; 8], gamma: f32) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L210 [source]: `#[arcane]` on `pub fn log_sum_exp(token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L224 [source]: `#[arcane]` on `pub fn log_values(token: X64V3Token, data: &[f32; 8]) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L231 [source]: `#[arcane]` on `pub fn log2_values(token: X64V3Token, data: &[f32; 8]) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L247 [source]: `#[arcane]` on `pub fn relu(token: X64V3Token, x: &[f32; 8]) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L255 [source]: `#[arcane]` on `pub fn leaky_relu(token: X64V3Token, x: &[f32; 8], alpha: f32) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L270 [source]: `#[arcane]` on `pub fn threshold(token: X64V3Token, x: &[f32; 8], thresh: f32) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L282 [source]: `#[arcane]` on `pub fn clamp(token: X64V3Token, x: &[f32; 8], lo: f32, hi: f32) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L308 [source]: `#[arcane]` on `fn batch_norm_avx2( token: X64V3Token, data: &mut [f32], mean: f32, var: f32, gamma: f32, beta: f32, eps: f32, )` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L359 [source]: `#[arcane]` on `fn layer_norm_avx2(token: X64V3Token, data: &mut [f32], gamma: f32, beta: f32, eps: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L384 [source]: `#[rite]` on `fn compute_mean(token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L398 [source]: `#[rite]` on `fn compute_variance(token: X64V3Token, data: &[f32], mean: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L446 [source]: `#[arcane]` on `fn cosine_sim_avx2(token: X64V3Token, a: &[f32], b: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L507 [source]: `#[arcane]` on `fn softmax_avx2(token: X64V3Token, data: &mut [f32])` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L540 [source]: `#[rite]` on `fn find_max(token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

## [magetypes/examples/plane_gain.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/plane_gain.rs#L7)

- L7 [source]: `#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]` on `fn gain_impl(token: Token, plane: &mut [f32], gain: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L20 [source]: `incant!(gain_impl(plane, gain), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [magetypes/examples/polyfill_demo.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/polyfill_demo.rs#L25)

- L25 [source]: `#[arcane]` on `fn sum_polyfill(token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L45 [source]: `#[arcane]` on `fn sum_native_sse(token: archmage::X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L67 [source]: `#[arcane]` on `fn sum_native_avx2(token: archmage::X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/examples/simd_kernels.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/simd_kernels.rs#L34)

- L34 [source]: `#[arcane]` on `pub fn dct4x4_vp8(token: X64V2Token, block: &mut [i32; 16])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L77 [source]: `#[arcane]` on `pub fn dct8_butterfly(token: X64V3Token, m: &mut [f32x8; 8])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L161 [source]: `#[arcane]` on `pub fn downsample_2x2_row(token: X64V3Token, row0: &[f32], row1: &[f32], output: &mut [f32])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L210 [source]: `#[arcane]` on `pub fn rgb_to_ycbcr_8px( token: X64V3Token, r: f32x8, g: f32x8, b: f32x8, ) -> (f32x8, f32x8, f32x8)` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L253 [source]: `#[arcane]` on `pub fn convolve_horizontal_u8( token: X64V2Token, input: &[u8], output: &mut [u8], kernel: &[i16], scale_shift: u32, )` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L298 [source]: `#[arcane]` on `pub fn srgb_to_linear_8px(token: X64V3Token, srgb: f32x8) -> f32x8` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L327 [source]: `#[arcane]` on `pub fn linear_to_srgb_8px(token: X64V3Token, linear: f32x8) -> f32x8` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L361 [source]: `#[arcane]` on `pub fn blend_multiply_2px(token: X64V3Token, src: f32x8, dst: f32x8) -> f32x8` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L372 [source]: `#[arcane]` on `pub fn blend_screen_2px(token: X64V3Token, src: f32x8, dst: f32x8) -> f32x8` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

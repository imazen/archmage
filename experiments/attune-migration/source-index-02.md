# Compact source migration index

Exact old macro/call → destination pattern. Signatures are lexical, including macro templates, not compiler-resolved. Original file links plus L numbers locate every item. All support attributes and textual mentions remain in the raw inventory.

## [magetypes/examples/simd_kernels.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/simd_kernels.rs#L388) (continued)

- L388 [source]: `#[arcane]` on `pub fn blend_overlay_2px(token: X64V3Token, src: f32x8, dst: f32x8) -> f32x8` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L419 [source]: `#[arcane]` on `pub fn reduce_horizontal_f32( token: X64V3Token, input: &[f32], output: &mut [f32], weights: &[f32], stride: usize, )` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/examples/transcendental_test.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/transcendental_test.rs#L14)

- L14 [source]: `#[arcane]` on `pub fn test_log2(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L19 [source]: `#[arcane]` on `pub fn test_exp2(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L24 [source]: `#[arcane]` on `pub fn test_pow(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L29 [source]: `#[arcane]` on `pub fn test_ln(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L34 [source]: `#[arcane]` on `pub fn test_exp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L39 [source]: `#[arcane]` on `pub fn test_log10(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L44 [source]: `#[arcane]` on `pub fn test_cbrt_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/examples/u32_shift_anomaly.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/examples/u32_shift_anomaly.rs#L72)

- L72 [source]: `#[arcane(import_intrinsics)]` on `pub fn copy_128(_t: X64V3Token, src: &[u32], dst: &mut [u32])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L82 [source]: `#[arcane(import_intrinsics)]` on `pub fn shr_const_128(_t: X64V3Token, src: &[u32], dst: &mut [u32])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L93 [source]: `#[arcane(import_intrinsics)]` on `pub fn shr_uniform_128(_t: X64V3Token, src: &[u32], dst: &mut [u32], count: u32)` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L108 [source]: `#[arcane(import_intrinsics)]` on `pub fn shr_uniform_128_licm(_t: X64V3Token, src: &[u32], dst: &mut [u32], count: u32)` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L119 [source]: `#[arcane(import_intrinsics)]` on `pub fn copy_256(_t: X64V3Token, src: &[u32], dst: &mut [u32])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L129 [source]: `#[arcane(import_intrinsics)]` on `pub fn shr_const_256(_t: X64V3Token, src: &[u32], dst: &mut [u32])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L140 [source]: `#[arcane(import_intrinsics)]` on `pub fn shr_uniform_256(_t: X64V3Token, src: &[u32], dst: &mut [u32], count: u32)` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L153 [source]: `#[arcane(import_intrinsics)]` on `pub fn shr_srlv_128(_t: X64V3Token, src: &[u32], dst: &mut [u32], count: u32)` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L171 [source]: `#[arcane(import_intrinsics)]` on `pub fn shr_add_const_128(_t: X64V3Token, src: &[u32], dst: &mut [u32])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L184 [source]: `#[arcane(import_intrinsics)]` on `pub fn shr_add_uniform_128(_t: X64V3Token, src: &[u32], dst: &mut [u32], count: u32)` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L200 [source]: `#[arcane(import_intrinsics)]` on `pub fn micro_lat_srl(_t: X64V3Token, iters: usize, seed: u32, count: u32) -> u32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L212 [source]: `#[arcane(import_intrinsics)]` on `pub fn micro_tpt_srl(_t: X64V3Token, iters: usize, seed: u32, count: u32) -> u32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L239 [source]: `#[arcane(import_intrinsics)]` on `pub fn micro_lat_srlv(_t: X64V3Token, iters: usize, seed: u32, count: u32) -> u32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L250 [source]: `#[arcane(import_intrinsics)]` on `pub fn micro_tpt_srlv(_t: X64V3Token, iters: usize, seed: u32, count: u32) -> u32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L276 [source]: `#[arcane(import_intrinsics)]` on `pub fn shr_srlv_256(_t: X64V3Token, src: &[u32], dst: &mut [u32], count: u32)` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/accuracy_test.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/accuracy_test.rs#L15)

- L15 [source]: `#[arcane]` on `fn simd_cbrt_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L20 [source]: `#[arcane]` on `fn simd_cbrt_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L25 [source]: `#[arcane]` on `fn simd_pow_lowp(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L30 [source]: `#[arcane]` on `fn simd_pow_midp(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L35 [source]: `#[arcane]` on `fn simd_exp2_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L40 [source]: `#[arcane]` on `fn simd_exp2_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L45 [source]: `#[arcane]` on `fn simd_log2_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L50 [source]: `#[arcane]` on `fn simd_log2_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L55 [source]: `#[arcane]` on `fn simd_ln_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L60 [source]: `#[arcane]` on `fn simd_ln_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L65 [source]: `#[arcane]` on `fn simd_exp_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L70 [source]: `#[arcane]` on `fn simd_exp_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L75 [source]: `#[arcane]` on `fn simd_log10_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/archmage_doc_examples.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/archmage_doc_examples.rs#L29)

- L29 [source]: `#[arcane]` on `fn process_simd(token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L38 [source]: `#[rite]` on `fn process_chunk(token: X64V3Token, chunk: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L68 [source]: `#[arcane]` on `fn process_all_simd(token: X64V3Token, pairs: &[([f32; 8], [f32; 8])]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L76 [source]: `#[rite]` on `fn process_pair_simd(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L108 [source]: `#[arcane(import_intrinsics)]` on `fn load_and_square_intrinsics(token: X64V3Token, data: &[f32; 8]) -> __m256` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L114 [source]: `#[arcane]` on `fn load_and_square_magetypes(token: X64V3Token, data: &[f32; 8]) -> f32x8` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L148 [source]: `#[arcane]` on `fn process_simd(token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L189 [source]: `#[arcane]` on `fn process_avx2(token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L199 [source]: `#[arcane]` on `fn process_neon(token: Arm64, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L228 [source]: `#[arcane]` on `fn x86_kernel(token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L281 [source]: `#[arcane]` on `fn sum_squares_avx2(token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L297 [source]: `#[arcane]` on `fn scale_avx2(token: X64V3Token, data: &mut [f32], scale: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L342 [source]: `#[arcane]` on `fn softmax_avx2(token: X64V3Token, data: &mut [f32])` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L378 [source]: `#[rite]` on `fn reduce_max(token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L447 [source]: `#[arcane]` on `fn dot_avx2(token: X64V3Token, a: &[f32], b: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L469 [source]: `#[arcane]` on `fn dot_neon(token: Arm64, a: &[f32], b: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/cbrt_range.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/cbrt_range.rs#L38)

- L38 [source]: `#[magetypes(v3, neon, wasm128, scalar)]` on `fn run(token: Token)` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L81 [source]: `incant!(run(), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/doc_examples.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/doc_examples.rs#L1011)

- L1011 [source]: `#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]` on `fn gain_impl(token: Token, plane: &mut [f32], gain: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L1024 [source]: `incant!(gain_impl(plane, gain), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L1054 [source]: `#[magetypes(v3, neon, wasm128, scalar)]` on `fn gain_entry(token: Token, plane: &mut [f32], gain: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L1060 [source]: `incant!(gain_entry(plane, factor), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L1077 [source]: `#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]` on `fn square_impl(token: Token, plane: &mut [f32])` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L1090 [source]: `incant!(square_impl(plane), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L1108 [source]: `#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]` on `fn lookup_impl(token: Token, table: &[f32], indices: &[usize; 8], gain: f32) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L1116 [source]: `incant!( lookup_impl(table, indices, gain), [v3, neon, wasm128, scalar] )` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L1141 [source]: `#[rite(v3)]` on `fn double(value: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L1146 [source]: `#[arcane]` on `fn test_entry(token: X64V3Token, value: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L1175 [source]: `#[archmage::magetypes(define(f32x16), v4(cfg(avx512)), v3, neon, wasm128, scalar)]` on `fn gamma_to_linear_slice_tier(token: Token, values: &mut [f32], gamma: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L1191 [source]: `incant!( gamma_to_linear_slice_tier(values, gamma), [v4, v3, neon, wasm128, scalar] )` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L1232 [source]: `#[magetypes(v3, neon, wasm128, scalar)]` on `fn blend_entry(token: Token, fg: &mut [f32], bg: &[f32])` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L1238 [source]: `incant!(blend_entry(fg, bg), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L1290 [source]: `#[magetypes(v3, neon, wasm128, scalar)]` on `fn add_green_entry(token: Token, rgba: &mut [u8])` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L1296 [source]: `incant!(add_green_entry(rgba), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/exp2_lowp_range.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/exp2_lowp_range.rs#L10)

- L10 [source]: `#[magetypes(v3, neon, wasm128, scalar)]` on `fn run(token: Token)` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L44 [source]: `incant!(run(), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/expand/define/empty_list.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/expand/define/empty_list.rs#L5)

- L5 [expansion-input]: `#[magetypes(define(), v3, scalar)]` on `fn kernel(_token: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/expand/define/multiple_types.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/expand/define/multiple_types.rs#L4)

- L4 [expansion-input]: `#[magetypes(define(f32x8, f32x4, u8x16), v3, scalar)]` on `fn kernel( token: Token, data_8: &[f32; 8], data_4: &[f32; 4], bytes: &[u8; 16], ) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/expand/define/position_middle.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/expand/define/position_middle.rs#L4)

- L4 [expansion-input]: `#[magetypes(v3, define(f32x8), scalar)]` on `fn kernel(token: Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/expand/define/single_type.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/expand/define/single_type.rs#L4)

- L4 [expansion-input]: `#[magetypes(define(f32x8), v3, scalar)]` on `fn kernel(token: Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/expand/define/strict_lints.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/expand/define/strict_lints.rs#L14)

- L14 [expansion-input]: `#[magetypes(define(f32x8, u8x16), v3, scalar)]` on `fn kernel(token: Token, data: &[f32; 8], bytes: &[u8; 16]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/expand/define/with_rite_flag.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/expand/define/with_rite_flag.rs#L5)

- L5 [expansion-input]: `#[magetypes(rite, define(f32x8), v3, scalar)]` on `fn kernel(token: Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/expand/rite_flag/basic.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/expand/rite_flag/basic.rs#L5)

- L5 [expansion-input]: `#[magetypes(rite, v3, scalar)]` on `fn kernel(token: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/expand/rite_flag/with_magetypes_body.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/expand/rite_flag/with_magetypes_body.rs#L7)

- L7 [expansion-input]: `#[magetypes(rite, v3, scalar)]` on `fn kernel(token: Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/expand/without_token/magetypes_body.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/expand/without_token/magetypes_body.rs#L6)

- L6 [expansion-input]: `#[rite(v3, scalar)]` on `fn dbl(x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L11 [expansion-input]: `#[magetypes(v3, scalar)]` on `fn run(token: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L14 [expansion-input]: `incant!(dbl(x) without token)` → [P6](migration.md#p6); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/fused_arithmetic.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/fused_arithmetic.rs#L217)

- L217 [source]: `#[magetypes(v3, v4, v4x, neon, wasm128, scalar)]` on `fn vectors(token: Token, inputs32: &[[f32; 3]], inputs64: &[[f64; 3]])` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L221 [source]: `incant!(narrow_f64(inputs64), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L229 [source]: `#[magetypes(v3, neon, wasm128, scalar)]` on `fn narrow_f64(token: Token, inputs64: &[[f64; 3]])` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L299 [source]: `incant!(vectors(&a, &b), [v4, v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L318 [source]: `#[archmage::arcane]` on `fn probe(_token: archmage::Wasm128RelaxedToken) -> (u32, u64)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/import_params.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/import_params.rs#L25)

- L25 [source]: `#[arcane(import_magetypes)]` on `fn arcane_magetypes_basic(token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L46 [source]: `#[arcane(import_intrinsics, import_magetypes)]` on `fn arcane_both_imports(token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

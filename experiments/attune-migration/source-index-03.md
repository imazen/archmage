# Compact source migration index

Exact old macro/call → destination pattern. Signatures are lexical, including macro templates, not compiler-resolved. Original file links plus L numbers locate every item. All support attributes and textual mentions remain in the raw inventory.

## [magetypes/tests/import_params.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/import_params.rs#L70) (continued)

- L70 [source]: `#[rite(import_magetypes)]` on `fn rite_magetypes(token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L76 [source]: `#[rite(import_intrinsics, import_magetypes)]` on `fn rite_both(token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L85 [source]: `#[arcane]` on `fn call_rite_variants(token: X64V3Token, data: &[f32; 8]) -> (f32, f32)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L119 [source]: `#[arcane(import_intrinsics, import_magetypes)]` on `fn process_batch(token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L129 [source]: `#[rite(import_magetypes)]` on `fn scale_chunk(token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L156 [source]: `#[arcane(import_intrinsics, import_magetypes)]` on `fn fma_then_reduce(token: X64V3Token, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L192 [source]: `#[arcane(import_magetypes)]` on `fn multi_width_types(token: X64V3Token, data4: &[f32; 4], data8: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L219 [source]: `#[arcane(import_magetypes)]` on `fn integer_types(token: X64V3Token, data: &[i32; 8]) -> i32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L243 [source]: `#[arcane(import_intrinsics, import_magetypes)]` on `fn process(&self, token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L273 [source]: `#[arcane(_self = Reducer, import_magetypes)]` on `fn reduce_sum(&self, token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L301 [source]: `#[rite(import_magetypes)]` on `fn normalize(token: X64V3Token, data: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L309 [source]: `#[rite(import_magetypes)]` on `fn scale(token: X64V3Token, data: &[f32; 8], factor: f32) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L316 [source]: `#[arcane(import_magetypes)]` on `fn normalize_and_scale(token: X64V3Token, data: &[f32; 8], factor: f32) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L349 [source]: `#[arcane(import_magetypes)]` on `fn use_backend_trait(token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L374 [source]: `#[arcane(import_magetypes)]` on `fn use_width_constants(token: X64V3Token) -> (usize, usize)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L395 [source]: `#[arcane(import_magetypes)]` on `fn use_natural_width(token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L417 [source]: `#[arcane(import_magetypes)]` on `fn use_token_alias(token: X64V3Token) -> &'static str` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L440 [source]: `#[arcane(import_magetypes)]` on `fn neon_magetypes(token: NeonToken, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L447 [source]: `#[rite(import_intrinsics, import_magetypes)]` on `fn neon_helper(token: NeonToken, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L471 [source]: `#[arcane(import_magetypes)]` on `fn wasm_magetypes(token: Wasm128Token, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/incant_chain_combinations.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/incant_chain_combinations.rs#L73)

- L73 [source]: `#[arcane]` on `fn mixed_v3(token: X64V3Token, d: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L77 [source]: `#[arcane]` on `fn mixed_neon(token: NeonToken, d: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L81 [source]: `#[arcane]` on `fn mixed_wasm128(token: Wasm128Token, d: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L89 [source]: `incant!(mixed(d), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L93 [source]: `#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]` on `fn mt(token: Token, d: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L98 [source]: `incant!(mt(d), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L102 [source]: `#[arcane]` on `fn av_top_v4(token: X64V4Token, d: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L107 [source]: `#[autoversion(v3, neon, wasm128)]` on `fn av_top_default(d: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L112 [source]: `incant!(av_top(d), [v4(cfg(avx512)), default])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L147 [source]: `#[arcane]` on `fn leaf_v3(token: X64V3Token, d: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L151 [source]: `#[arcane]` on `fn leaf_neon(token: NeonToken, d: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L155 [source]: `#[arcane]` on `fn leaf_wasm128(token: Wasm128Token, d: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L164 [source]: `#[arcane]` on `pub fn caller_arcane_v3(_token: X64V3Token, d: &[f32; 8]) -> f32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L166 [source]: `incant!(leaf(d), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L170 [source]: `#[rite(import_intrinsics)]` on `pub fn caller_rite_v3(_token: X64V3Token, d: &[f32; 8]) -> f32` [visibility: pub] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L172 [source]: `incant!(leaf(d), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L176 [source]: `#[autoversion(v3, neon, wasm128)]` on `pub fn caller_autoversion(d: &[f32; 8]) -> f32` [visibility: pub] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L178 [source]: `incant!(leaf(d), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L182 [source]: `#[magetypes(v3, neon, wasm128, scalar)]` on `fn caller_magetypes(token: Token, d: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L185 [source]: `incant!(leaf(d), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L188 [source]: `incant!(caller_magetypes(d), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L220 [source]: `#[rite(import_intrinsics)]` on `fn chain_rite_v3(token: X64V3Token, d: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L228 [source]: `#[arcane]` on `pub fn chain_arcane_to_rite(token: X64V3Token, d: &[f32; 8]) -> f32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L235 [source]: `#[arcane]` on `pub fn chain_v4_downcast_to_rite(token: X64V4Token, d: &[f32; 8]) -> f32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L242 [source]: `#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]` on `fn chain_mt(token: Token, d: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L248 [source]: `incant!(chain_mt(d), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/int_widen_narrow.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/int_widen_narrow.rs#L828)

- L828 [source]: `#[archmage::magetypes(define(i16x8, i32x4, u8x16), v3, neon, wasm128, scalar)]` on `fn integer_kernel(token: Token, a: &[i16; 8], b: &[i16; 8], bytes: &[u8; 16]) -> ([i32; 4], u32)` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L845 [source]: `archmage::incant!(integer_kernel(&a, &b, &bytes), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L851 [source]: `#[archmage::magetypes(define(u8x64, i16x32), v4(cfg(avx512)), v3, neon, wasm128, scalar)]` on `fn w512_byte_dot(token: Token, bytes: &[u8; 64], rhs: &[i16; 32]) -> [i32; 16]` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L870 [source]: `archmage::incant!( w512_byte_dot(&bytes, &rhs), [v4(cfg(avx512)), v3, neon, wasm128, scalar] )` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/magetypes_define_flag.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/magetypes_define_flag.rs#L15)

- L15 [source]: `#[magetypes(define(f32x8), v3, scalar)]` on `fn scale_impl(token: Token, plane: &mut [f32], factor: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L29 [source]: `incant!(scale_impl(plane, factor), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L43 [source]: `#[magetypes(define(f32x4, f32x8), v3, scalar)]` on `fn mixed_widths_impl(token: Token, data_4: &[f32; 4], data_8: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L51 [source]: `incant!(mixed_widths_impl(data_4, data_8), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L66 [source]: `#[magetypes(define(u8x16, i16x8), v3, scalar)]` on `fn integer_ops_impl(token: Token, bytes: &[u8; 16]) -> i32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L76 [source]: `incant!(integer_ops_impl(bytes), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L89 [source]: `#[magetypes(rite, define(f32x8), v3, scalar)]` on `fn rite_with_define_impl(token: Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L109 [source]: `#[magetypes(define(f32x8), v3, scalar)]` on `fn scope_isolation_impl(token: Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L129 [source]: `#[magetypes(define(), v3, scalar)]` on `fn empty_define_impl(_t: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L143 [source]: `#[magetypes(v3, define(f32x8), scalar)]` on `fn order_define_middle_impl(token: Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L148 [source]: `#[magetypes(v3, scalar, define(f32x8))]` on `fn order_define_last_impl(token: Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/magetypes_macro.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/magetypes_macro.rs#L29)

- L29 [source]: `#[magetypes]` on `pub fn add_one(token: Token, x: f32) -> f32` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L68 [source]: `#[magetypes]` on `pub fn multiply(token: Token, a: f32, b: f32) -> f32` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L98 [source]: `#[magetypes]` on `pub fn sum_slice(token: Token, data: &[f32]) -> f32` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L129 [source]: `#[magetypes]` on `pub fn double(token: Token, x: f32) -> f32` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L137 [source]: `incant!(double(x))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L167 [source]: `#[magetypes]` on `pub fn fill_array(token: Token, data: &mut [f32], val: f32)` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L201 [source]: `#[magetypes]` on `pub fn count(token: Token, data: &[f32]) -> usize` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L213 [source]: `#[magetypes]` on `pub fn is_empty(token: Token, data: &[f32]) -> bool` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L234 [source]: `#[magetypes]` on `pub fn public_fn(token: Token, x: f32) -> f32` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L240 [source]: `#[magetypes]` on `fn private_fn(token: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L265 [source]: `#[magetypes]` on `pub fn inner(token: Token, x: f32) -> f32` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L272 [source]: `incant!(inner(x) with token)` → [P7](migration.md#p7); apply [contract rules](migration-contracts.md#p12).
- L302 [source]: `#[magetypes]` on `pub fn uses_token_substring(token: Token, x: f32) -> f32` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L350 [source]: `#[arcane]` on `fn sum_impl_v3(token: archmage::X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L360 [source]: `incant!(sum_impl(data), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L384 [source]: `#[magetypes]` on `pub fn documented(token: Token, x: f32) -> f32` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L392 [source]: `#[magetypes]` on `pub fn with_allow(token: Token, x: f32, unused_arg: i32) -> f32` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L399 [source]: `#[magetypes(v3, neon, scalar)]` on `pub fn documented_explicit(token: Token, x: f32) -> f32` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/raw_interop.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/raw_interop.rs#L30)

- L30 [source]: `#[archmage::rite(v3)]` on `fn x86_roundtrips()` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L66 [source]: `#[arcane]` on `fn entry(_token: archmage::X64V3Token)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L76 [source]: `#[arcane]` on `fn entry(token: archmage::NeonToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/token_aliases.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/token_aliases.rs#L177)

- L177 [source]: `#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]` on `fn defined(token: Token) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L187 [source]: `archmage::incant!(defined(), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/transcendental_accuracy.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/transcendental_accuracy.rs#L150)

- L150 [source]: `#[arcane]` on `fn eval_f32x8(token: X64V3Token, inputs: &[f32; 8], op: &str, param: f32) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L182 [source]: `#[arcane]` on `fn eval_generic_f32x4(token: X64V3Token, inputs: &[f32; 4], op: &str, param: f32) -> [f32; 4]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [magetypes/tests/transcendental_edge_cases.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/magetypes/tests/transcendental_edge_cases.rs#L21)

- L21 [source]: `#[arcane]` on `fn direct_cbrt_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L26 [source]: `#[arcane]` on `fn direct_cbrt_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L31 [source]: `#[arcane]` on `fn direct_cbrt_midp_precise(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L38 [source]: `#[arcane]` on `fn direct_pow_lowp(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L43 [source]: `#[arcane]` on `fn direct_pow_midp(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L48 [source]: `#[arcane]` on `fn direct_pow_midp_precise(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L55 [source]: `#[arcane]` on `fn direct_pow_midp_unchecked(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L62 [source]: `#[arcane]` on `fn direct_pow_lowp_unchecked(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L69 [source]: `#[arcane]` on `fn direct_log2_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L74 [source]: `#[arcane]` on `fn direct_log2_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L79 [source]: `#[arcane]` on `fn direct_exp2_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L84 [source]: `#[arcane]` on `fn direct_ln_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L89 [source]: `#[arcane]` on `fn direct_exp_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L98 [source]: `#[arcane]` on `fn generic_cbrt_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

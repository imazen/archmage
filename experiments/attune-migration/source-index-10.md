# Compact source migration index

Exact old macro/call → destination pattern. Signatures are lexical, including macro templates, not compiler-resolved. Original file links plus L numbers locate every item. All support attributes and textual mentions remain in the raw inventory.

## [tests/macro_behavioral_contracts.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/macro_behavioral_contracts.rs#L244) (continued)

- L244 [source]: `#[rite]` on `fn helper_mul(token: X64V3Token, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L250 [source]: `#[arcane]` on `fn combined(token: X64V3Token, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L282 [source]: `#[arcane]` on `fn arm_cfgout(_token: NeonToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L297 [source]: `#[arcane]` on `fn x86_cfgout(_token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L327 [source]: `#[archmage::arcane]` on `fn compute_v3(_token: archmage::X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L333 [source]: `#[archmage::arcane]` on `fn compute_v4(_token: archmage::X64V4Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L339 [source]: `#[archmage::arcane]` on `fn compute_neon(_token: archmage::NeonToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L345 [source]: `#[archmage::arcane]` on `fn compute_wasm128(_token: archmage::Wasm128Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L352 [source]: `incant!(compute(data))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L363 [source]: `incant!(compute(data) with token)` → [P7](migration.md#p7); apply [contract rules](migration-contracts.md#p12).
- L375 [source]: `incant!(compute(data), [v3, neon, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/magetypes_rite_flag.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/magetypes_rite_flag.rs#L40)

- L40 [source]: `#[magetypes(rite, v3, scalar)]` on `fn rite_clamp(_t: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L56 [source]: `#[magetypes(rite, v3, scalar)]` on `fn rite_square(_t: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L64 [source]: `#[archmage::arcane]` on `fn call_rite_from_arcane_v3(token: archmage::X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L87 [source]: `#[magetypes(rite, v3, scalar)]` on `fn rite_double(_t: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L110 [source]: `#[archmage::arcane]` on `fn chain_rite_v3(token: archmage::X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L130 [source]: `#[magetypes(rite)]` on `fn rite_default_tiers(_t: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L147 [source]: `#[magetypes(rite, v3, scalar)]` on `fn token_aware(_t: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L163 [source]: `#[archmage::arcane]` on `fn wrap(token: archmage::X64V3Token) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/magetypes_scalar_dispatch.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/magetypes_scalar_dispatch.rs#L9)

- L9 [source]: `#[magetypes(rite, v3, neon, wasm128, scalar)]` on `fn leaf(_token: Token) -> &'static str` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L14 [source]: `#[magetypes(rite, v3, neon, wasm128, scalar)]` on `fn tokenless_scalar() -> &'static str` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L16 [source]: `incant!(leaf(), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L19 [source]: `#[magetypes(rite, v3, neon, wasm128, default)]` on `fn tokenless_default() -> &'static str` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L21 [source]: `dispatch_variant!(leaf(), [v3, neon, wasm128, scalar])` → [P8](migration.md#p8); apply [contract rules](migration-contracts.md#p12).
- L24 [source]: `#[magetypes(rite, v3, neon, wasm128, -scalar)]` on `fn default_leaf(_token: Token) -> i32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L33 [source]: `#[magetypes(rite, default)]` on `fn default_to_default() -> i32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L35 [source]: `incant!(default_leaf(), [v3, neon, wasm128, default])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L38 [source]: `#[magetypes(v3, neon, wasm128, scalar)]` on `fn boundary(_token: Token) -> &'static str` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L45 [source]: `#[magetypes(rite, scalar)]` on `fn tokenful_placeholder(x: i32, proof: Token) -> (&'static str, i32)` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L48 [source]: `incant!(boundary(), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L51 [source]: `#[magetypes(rite, scalar)]` on `fn tokenful_concrete(_: ScalarToken) -> &'static str` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L53 [source]: `incant!(boundary(), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L66 [source]: `incant!(boundary(), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/name_mangling.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/name_mangling.rs#L16)

- L16 [source]: `#[autoversion]` on `fn av_default(_token: SimdToken, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L54 [source]: `#[autoversion(v1, v2, x64_crypto, v3, v3_crypto, v3_gfni_crypto, v4, v4x, scalar)]` on `fn av_all_x86(_token: SimdToken, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L126 [source]: `#[autoversion(neon, arm_v2, arm_v3, neon_aes, neon_sha3, neon_crc, scalar)]` on `fn av_all_arm(_token: SimdToken, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L184 [source]: `#[autoversion(wasm128, wasm128_relaxed, scalar)]` on `fn av_all_wasm(_token: SimdToken, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L213 [source]: `#[autoversion(v3, neon, default)]` on `fn av_with_default(x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L238 [source]: `#[rite(v2, v3)]` on `fn rite_multi(x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L244 [source]: `#[arcane]` on `fn call_v3(_token: X64V3Token) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L257 [source]: `#[arcane]` on `fn call_v2(_token: X64V3Token) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L274 [source]: `#[magetypes]` on `fn mt_compute(_token: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L301 [source]: `#[arcane]` on `fn incant_test_v3(_: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L307 [source]: `incant!(incant_test(x), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L329 [source]: `#[arcane]` on `fn incant_default_test_v3(_: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L335 [source]: `incant!(incant_default_test(x), [v3, default])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L360 [source]: `#[arcane]` on `fn us_incant_v3(_: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L366 [source]: `incant!(us_incant(x), [_v3, _scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L383 [source]: `#[magetypes(_v3, _neon)]` on `fn us_mt(_token: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L403 [source]: `#[autoversion(_v3, _neon)]` on `fn us_av(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L418 [source]: `#[rite(_v3)]` on `fn us_rite_helper(x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L423 [source]: `#[arcane]` on `fn call_us_rite(_token: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L441 [source]: `#[autoversion(+v1)]` on `fn additive_av(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L463 [source]: `#[arcane]` on `fn add_incant_v4(_: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L468 [source]: `#[arcane]` on `fn add_incant_v3(_: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L477 [source]: `#[arcane]` on `fn add_incant_neon(_: NeonToken, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L487 [source]: `incant!(add_incant(x), [+_v1])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L497 [source]: `#[magetypes(+v1)]` on `fn additive_mt(_token: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).

## [tests/no-features-crate/src/lib.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/no-features-crate/src/lib.rs#L23)

- L23 [source]: `#[autoversion]` on `pub fn sum_squares(_token: SimdToken, data: &[f32]) -> f32` [visibility: pub] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L32 [source]: `#[autoversion]` on `pub fn scale_vec(_token: SimdToken, data: &[f32], factor: f32) -> Vec<f32>` [visibility: pub] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L41 [source]: `#[autoversion(v3, v4, neon)]` on `pub fn dot_product(_token: SimdToken, a: &[f32], b: &[f32]) -> f32` [visibility: pub] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L67 [source]: `#[autoversion(v3, neon, wasm128)]` on `pub fn entropy_score(_token: SimdToken, data: &[u8]) -> u32` [visibility: pub] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L76 [source]: `#[autoversion(v3, neon, wasm128)]` on `pub fn premul_u8_impl(_token: SimdToken, buf: &mut [u8])` [visibility: pub] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L95 [source]: `#[autoversion]` on `pub fn total(&self, _token: SimdToken) -> f32` [visibility: pub] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).

## [tests/prelude_docs.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/prelude_docs.rs#L129)

- L129 [source]: `#[arcane]` on `fn arcane_via_prelude(_token: Desktop64, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L134 [source]: `#[rite]` on `fn rite_via_prelude(_token: Desktop64, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L175 [source]: `#[arcane]` on `fn setzero_via_prelude(_token: Desktop64) -> __m256` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L193 [source]: `#[arcane]` on `fn value_ops_via_prelude(_token: Desktop64, a: __m256, b: __m256) -> __m256` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L221 [source]: `#[arcane]` on `fn safe_load(_token: Desktop64, data: &[f32; 8]) -> __m256` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L226 [source]: `#[arcane]` on `fn safe_store(_token: Desktop64, v: __m256, out: &mut [f32; 8])` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L295 [source]: `#[arcane]` on `fn multiply_and_add_safe( _token: Desktop64, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8], ) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/rite_macro.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/rite_macro.rs#L9)

- L9 [source]: `#[rite]` on `fn add_vectors(_token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L21 [source]: `#[rite]` on `fn mul_vectors(_token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L33 [source]: `#[rite]` on `fn horizontal_sum(_token: X64V3Token, v: __m256) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L44 [source]: `#[arcane]` on `fn dot_product(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L55 [source]: `#[arcane]` on `fn weighted_sum( token: X64V3Token, a: &[f32; 8], b: &[f32; 8], weight_a: f32, weight_b: f32, ) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L123 [source]: `#[rite(v3)]` on `fn add_vectors_tierless(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L136 [source]: `#[rite(v3, import_intrinsics)]` on `fn mul_vectors_tierless(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L147 [source]: `#[arcane]` on `fn dot_product_tierless(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L190 [source]: `#[rite(v3)]` on `fn negate_tierless(a: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L203 [source]: `#[rite(v2)]` on `fn popcount_tierless(val: i32) -> i32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L227 [source]: `#[rite]` on `fn scale_vector(_: X64V3Token, a: &[f32; 8], factor: f32) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L268 [source]: `#[rite(v1)]` on `fn add_i32x4_v1(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L280 [source]: `#[rite(v1)]` on `fn f64_add_v1(a: &[f64; 2], b: &[f64; 2]) -> [f64; 2]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L317 [source]: `#[rite(v2)]` on `fn blend_i16_v2(a: &[i16; 8], b: &[i16; 8]) -> [i16; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L332 [source]: `#[rite(v2, import_intrinsics)]` on `fn crc32_step_v2(crc: u32, data: u8) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L366 [source]: `#[rite(v3)]` on `fn fma_f32x8(a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L394 [source]: `#[rite(v3, import_intrinsics)]` on `fn abs_f32x8_all_options(a: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L417 [source]: `#[rite(v3, import_intrinsics)]` on `fn sum_f32x8_tierless(data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L427 [source]: `#[rite(v3, import_intrinsics)]` on `fn scale_f32x8_tierless(data: &[f32; 8], factor: f32) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L438 [source]: `#[arcane(import_intrinsics)]` on `fn normalize_f32x8(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L463 [source]: `#[rite(v3, import_intrinsics)]` on `fn square_f32x8(data: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L472 [source]: `#[rite(v3, import_intrinsics)]` on `fn sum_of_squares_tierless(data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L480 [source]: `#[arcane]` on `fn l2_norm_squared(_token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L497 [source]: `#[rite]` on `fn mixed_caller_token_based(_token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L520 [source]: `#[arcane(import_intrinsics)]` on `fn compose_mixed_flavors(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L552 [source]: `#[rite(v3, import_intrinsics)]` on `fn process_chunk(&self, data: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L563 [source]: `#[rite(v3, import_intrinsics)]` on `fn reduce_sum(&self, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L603 [source]: `#[rite(v3, import_intrinsics)]` on `fn minmax_f32x8(data: &[f32; 8]) -> (f32, f32)` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L634 [source]: `#[rite(v3, import_intrinsics)]` on `fn dot_with_offset(a: &[f32; 8], b: &[f32; 8], offset_a: f32, offset_b: f32, scale: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L666 [source]: `#[rite(v3, import_intrinsics)]` on `fn sum_first_n<const N: usize>(data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

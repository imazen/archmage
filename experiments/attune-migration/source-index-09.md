# Compact source migration index

Exact old macro/call → destination pattern. Signatures are lexical, including macro templates, not compiler-resolved. Original file links plus L numbers locate every item. All support attributes and textual mentions remain in the raw inventory.

## [tests/expand/should-fail/autoversion_trait_impl.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/should-fail/autoversion_trait_impl.rs#L13)

- L13 [expected-failure]: `#[autoversion(_self = Filter)]` on `fn process(&self, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/should-fail/incant_passthrough.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/should-fail/incant_passthrough.rs#L4)

- L4 [expected-failure]: `#[arcane]` on `fn inner_v3(_token: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).
- L9 [expected-failure]: `#[arcane]` on `fn inner_neon(_token: NeonToken, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).
- L19 [expected-failure]: `incant!(inner(x) with token)` → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/should-fail/rite_trait_impl.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/should-fail/rite_trait_impl.rs#L13)

- L13 [expected-failure]: `#[rite]` on `fn compute(&self, token: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/from_context.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/from_context.rs#L32)

- L32 [source]: `#[rite(v3)]` on `fn sum_via_forged_v3(data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L43 [source]: `#[rite(import_intrinsics)]` on `fn double_and_sum(_token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L54 [source]: `#[rite(v3)]` on `fn forge_weaker_tiers() -> (X64V1Token, X64V2Token)` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L61 [source]: `#[arcane]` on `fn entry(_token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L73 [source]: `#[rite(v3)]` on `fn recursive_sum(data: &[f32], depth: u32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L90 [source]: `#[rite(v3)]` on `fn closure_forges(data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L96 [source]: `#[arcane]` on `fn entry_recursive(_token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L101 [source]: `#[arcane]` on `fn entry_closure(_token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L153 [source]: `#[rite(neon)]` on `fn sum_via_forged_neon(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L159 [source]: `#[rite(import_intrinsics)]` on `fn double_and_sum(_token: NeonToken, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L165 [source]: `#[arcane]` on `fn entry(_token: NeonToken, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L199 [source]: `#[rite(wasm128)]` on `fn sum_via_forged_wasm128(data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L205 [source]: `#[rite(import_intrinsics)]` on `fn double_and_sum(_token: Wasm128Token, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L214 [source]: `#[arcane]` on `fn entry(_token: Wasm128Token, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/idiomatic_patterns.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/idiomatic_patterns.rs#L41)

- L41 [source]: `#[arcane(import_intrinsics)]` on `pub fn sum_f32x8(token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L54 [source]: `#[arcane(import_intrinsics)]` on `pub fn fma_f32x8(token: Desktop64, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L111 [source]: `#[arcane(import_intrinsics)]` on `pub fn popcnt_array(token: impl HasX64V2, data: &[u64; 4]) -> u32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L122 [source]: `#[arcane(import_intrinsics)]` on `pub fn sum_sse<T: HasX64V2>(token: T, data: &[f32; 4]) -> f32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L132 [source]: `#[arcane(import_intrinsics)]` on `pub fn dot_sse<T>(token: T, a: &[f32; 4], b: &[f32; 4]) -> f32 where T: HasX64V2,` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L199 [source]: `#[arcane(import_intrinsics)]` on `pub fn add_f32x8_deprecated(token: impl Has256BitSimd, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L211 [source]: `#[arcane(import_intrinsics)]` on `pub fn fma_f32x8_correct( token: X64V3Token, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8], ) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L271 [source]: `#[arcane(_self = Vec8f32, import_intrinsics)]` on `fn double(&self, _token: X64V3Token) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L281 [source]: `#[arcane(_self = Vec8f32, import_intrinsics)]` on `fn square(self, _token: X64V3Token) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L291 [source]: `#[arcane(_self = Vec8f32, import_intrinsics)]` on `fn scale(&mut self, _token: X64V3Token, factor: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L341 [source]: `#[arcane(import_intrinsics)]` on `fn add_vectors(token: X64V3Token, a: __m256, b: __m256) -> __m256` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L347 [source]: `#[arcane(import_intrinsics)]` on `fn mul_vectors(token: X64V3Token, a: __m256, b: __m256) -> __m256` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L353 [source]: `#[arcane(import_intrinsics)]` on `pub fn dot_product(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L371 [source]: `#[arcane(import_intrinsics)]` on `pub fn polynomial(token: X64V3Token, x: &[f32; 8], a: f32, b: f32, c: f32) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L428 [source]: `#[arcane(import_intrinsics)]` on `pub fn sum_v3(token: X64V3Token, data: &[f32]) -> f32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L500 [source]: `#[arcane(import_intrinsics)]` on `pub fn process_x86(token: X64V3Token, data: &mut [f32])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L524 [source]: `#[arcane(import_intrinsics)]` on `pub fn process_arm(token: NeonToken, data: &mut [f32])` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L586 [source]: `#[arcane(import_intrinsics)]` on `fn sse_operation(token: X64V2Token, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L595 [source]: `#[arcane(import_intrinsics)]` on `pub fn flexible_sum(token: X64V3Token, data: &[f32; 4]) -> f32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L606 [source]: `#[arcane(import_intrinsics)]` on `pub fn avx512_with_fallback(token: X64V4Token, data: &[f32; 4]) -> f32` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/import_intrinsics.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/import_intrinsics.rs#L21)

- L21 [source]: `#[arcane(import_intrinsics)]` on `fn arcane_intrinsics_basic(token: X64V3Token, data: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L47 [source]: `#[rite(import_intrinsics)]` on `fn rite_intrinsics(token: X64V3Token, data: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L62 [source]: `#[arcane(import_intrinsics)]` on `fn trait_bound_impl(token: impl HasX64V2, data: &[f32; 4]) -> [f32; 4]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L84 [source]: `#[arcane(import_intrinsics)]` on `fn generic_bound_intrinsics<T: HasX64V2>(token: T, data: &[f32; 4]) -> [f32; 4]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L106 [source]: `#[arcane(import_intrinsics)]` on `fn wildcard_token(_: X64V3Token, data: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L130 [source]: `#[arcane(import_intrinsics)]` on `fn v2_intrinsics(token: X64V2Token, data: &[f32; 4]) -> [f32; 4]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L158 [source]: `#[arcane(import_intrinsics)]` on `fn v4_intrinsics(token: X64V4Token, data: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L187 [source]: `#[arcane(import_intrinsics)]` on `fn with_existing_imports(token: X64V3Token) -> bool` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L213 [source]: `#[arcane(import_intrinsics)]` on `fn neon_intrinsics(token: NeonToken, data: &[f32; 4]) -> [f32; 4]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/incant_cfgout.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/incant_cfgout.rs#L25)

- L25 [source]: `#[archmage::arcane]` on `fn add_v3(_token: archmage::X64V3Token, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L31 [source]: `#[archmage::arcane]` on `fn add_v4(_token: archmage::X64V4Token, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L38 [source]: `#[archmage::arcane]` on `fn add_neon(_token: archmage::NeonToken, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L45 [source]: `#[archmage::arcane]` on `fn add_wasm128(_token: archmage::Wasm128Token, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L59 [source]: `incant!(add(a, b))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L73 [source]: `incant!(add(a, b), [v3, neon, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L92 [source]: `incant!(trivial(x), [scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/incant_macro.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/incant_macro.rs#L82)

- L82 [source]: `incant!(sum(data))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L119 [source]: `incant!(add_one(data), [v1, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L131 [source]: `incant!(add_one(data), [v2, v1, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L182 [source]: `incant!(gfni_available(), [v3_gfni_crypto, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L206 [source]: `incant!(sum(data) with token)` → [P7](migration.md#p7); apply [contract rules](migration-contracts.md#p12).
- L273 [source]: `incant!(double(x))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L323 [source]: `incant!(dot(a, b))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L369 [source]: `incant!(make_array(val))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L412 [source]: `simd_route!(add(a, b))` → [P8](migration.md#p8); apply [contract rules](migration-contracts.md#p12).
- L462 [source]: `incant!(super::simd_impls::triple(x))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L475 [source]: `incant!(super::simd_impls::triple(x) with token)` → [P7](migration.md#p7); apply [contract rules](migration-contracts.md#p12).
- L550 [source]: `#[arcane]` on `fn entry_v3(_: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L556 [source]: `incant!(entry(x), [v3, default])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L579 [source]: `incant!(default_only(x), [default])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L593 [source]: `#[arcane]` on `fn multi_arg_v3(_: X64V3Token, a: f32, b: f32, c: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L599 [source]: `incant!(multi_arg(a, b, c), [v3, default])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L621 [source]: `#[arcane]` on `fn passthrough_v3(_: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L627 [source]: `incant!(passthrough(x) with token, [v3, default])` → [P7](migration.md#p7); apply [contract rules](migration-contracts.md#p12).
- L654 [source]: `incant!(no_token(x), [default])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/incant_variants.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/incant_variants.rs#L19)

- L19 [source]: `#[arcane]` on `fn add_one_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L26 [source]: `#[arcane]` on `fn add_one_v4(_t: archmage::X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L38 [source]: `incant!(add_one(x))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L63 [source]: `#[arcane]` on `fn double_v3(_t: X64V3Token, x: i32) -> i32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L68 [source]: `#[arcane]` on `fn double_v4(_t: X64V4Token, x: i32) -> i32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L78 [source]: `incant!(double(x))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L116 [source]: `#[arcane]` on `fn square_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L122 [source]: `#[arcane]` on `fn square_v4(_t: archmage::X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L132 [source]: `incant!(square(x) with token)` → [P7](migration.md#p7); apply [contract rules](migration-contracts.md#p12).
- L158 [source]: `#[arcane]` on `fn step_a_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L164 [source]: `#[arcane]` on `fn step_a_v4(_t: archmage::X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L173 [source]: `#[arcane]` on `fn step_b_v3(_t: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L179 [source]: `#[arcane]` on `fn step_b_v4(_t: archmage::X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L189 [source]: `incant!(step_a(x))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L190 [source]: `incant!(step_b(intermediate))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L208 [source]: `#[arcane]` on `fn weighted_sum_v3(_t: X64V3Token, a: f32, b: f32, weight: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L214 [source]: `#[arcane]` on `fn weighted_sum_v4(_t: archmage::X64V4Token, a: f32, b: f32, weight: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L224 [source]: `incant!(weighted_sum(a, b, weight))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/macro_behavioral_contracts.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/macro_behavioral_contracts.rs#L19)

- L19 [source]: `#[arcane]` on `fn double_values(token: X64V3Token, data: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L39 [source]: `#[arcane]` on `fn negate_desktop(token: Desktop64, data: &[f32; 4]) -> [f32; 4]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L58 [source]: `#[arcane]` on `fn sum_wildcard(_: X64V3Token, data: &[f32; 4]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L72 [source]: `#[arcane]` on `fn add_arrays(token: X64V3Token, a: &[f32; 4], b: &[f32; 4]) -> [f32; 4]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L89 [source]: `#[arcane]` on `fn scalar_return(token: X64V3Token, val: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L101 [source]: `#[arcane]` on `fn bool_return(token: X64V3Token, val: f32) -> bool` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L126 [source]: `#[arcane(_self = SimdVec8)]` on `fn sum_ref(&self, token: X64V3Token) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L131 [source]: `#[arcane(_self = SimdVec8)]` on `fn scale_mut(&mut self, token: X64V3Token, factor: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L138 [source]: `#[arcane(_self = SimdVec8)]` on `fn into_sum(self, token: X64V3Token) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L144 [source]: `#[arcane(_self = SimdVec8)]` on `fn doubled(&self, token: X64V3Token) -> Self` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L190 [source]: `#[arcane]` on `fn generic_impl_trait(token: impl HasX64V2, val: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L202 [source]: `#[arcane]` on `fn generic_type_param<T: HasX64V2>(token: T, val: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L215 [source]: `#[arcane]` on `fn generic_where_clause<T>(token: T, val: f32) -> f32 where T: HasX64V2,` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L239 [source]: `#[rite]` on `fn helper_add(_: X64V3Token, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

# Compact source migration index

Exact old macro/call → destination pattern. Signatures are lexical, including macro templates, not compiler-resolved. Original file links plus L numbers locate every item. All support attributes and textual mentions remain in the raw inventory.

## [tests/avx512_intrinsics_exercise.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/avx512_intrinsics_exercise.rs#L719) (continued)

- L719 [source]: `#[arcane]` on `fn exercise_avx512vnni(token: X64V4xToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L805 [source]: `#[arcane]` on `fn exercise_avx512bitalg(token: X64V4xToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L831 [source]: `#[arcane]` on `fn exercise_avx512ifma(token: X64V4xToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L881 [source]: `#[arcane]` on `fn exercise_gfni(token: X64V4xToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L936 [source]: `#[arcane]` on `fn exercise_vaes(token: X64V4xToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L960 [source]: `#[arcane]` on `fn exercise_vpclmulqdq(token: X64V4xToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/avx512fp16_intrinsics.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/avx512fp16_intrinsics.rs#L83)

- L83 [source]: `#[rite]` on `fn splat512(_t: Avx512Fp16Token, bits: u16) -> __m512h` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L90 [source]: `#[rite]` on `fn low_lanes512(_t: Avx512Fp16Token, v: __m512h) -> [f32; 2]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L100 [source]: `#[arcane]` on `fn exercise_fp16_512(token: Avx512Fp16Token)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L146 [source]: `#[rite]` on `fn lane0_256(_t: Avx512Fp16Token, v: __m256h) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L153 [source]: `#[rite]` on `fn lane0_128(_t: Avx512Fp16Token, v: __m128h) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L159 [source]: `#[arcane]` on `fn exercise_fp16_vl(token: Avx512Fp16Token)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/calling_convention_matrix.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/calling_convention_matrix.rs#L6)

- L6 [source]: `#[magetypes(rite, v3, neon, wasm128, scalar)]` on `fn proof_leaf<const N: usize>(x: [u32; N], _proof: Token) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L11 [source]: `#[rite(v3, neon, wasm128, scalar, default)]` on `fn plain_leaf<const N: usize>(x: [u32; N]) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L17 [source]: `#[rite(v3, neon, wasm128, scalar, default)]` on `fn plain_caller<const N: usize>(x: [u32; N]) -> (u32, u32)` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L20 [source]: `incant!(proof_leaf::<N>(x, Token), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L21 [source]: `incant!(plain_leaf::<N>(x) without token)` → [P6](migration.md#p6); apply [contract rules](migration-contracts.md#p12).
- L27 [source]: `#[magetypes(v3, neon, wasm128, scalar)]` on `fn boundary_leaf<const N: usize>(x: [u32; N], _proof: Token) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L33 [source]: `#[magetypes(v3, neon, wasm128, scalar)]` on `fn boundary<const N: usize>(_proof: Token, x: [u32; N]) -> (u32, u32)` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L35 [source]: `incant!(boundary_leaf::<N>(x, Token), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L36 [source]: `incant!(plain_caller::<N>(x) without token)` → [P6](migration.md#p6); apply [contract rules](migration-contracts.md#p12).
- L41 [source]: `#[autoversion(v3, neon, wasm128)]` on `fn automatic(x: [u32; 3]) -> (u32, u32)` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L43 [source]: `incant!(plain_caller::<3>(x) without token)` → [P6](migration.md#p6); apply [contract rules](migration-contracts.md#p12).
- L46 [source]: `#[autoversion(v3, neon, wasm128)]` on `fn automatic_legacy(_proof: SimdToken, x: [u32; 3]) -> (u32, u32)` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L48 [source]: `incant!(plain_caller::<3>(x) without token)` → [P6](migration.md#p6); apply [contract rules](migration-contracts.md#p12).
- L51 [source]: `#[arcane]` on `fn scalar_boundary(_proof: ScalarToken, x: [u32; 3]) -> (u32, u32)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L53 [source]: `incant!(plain_caller::<3>(x) without token)` → [P6](migration.md#p6); apply [contract rules](migration-contracts.md#p12).
- L71 [source]: `incant!(boundary::<3>(x), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L75 [source]: `incant!(proof_leaf::<3>(x, Token), [scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L76 [source]: `incant!(plain_leaf::<3>(x, Token), [default])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/cfg_elision.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/cfg_elision.rs#L542)

- L542 [source]: `#[archmage::arcane]` on `fn sum_avx2(_token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/cfg_feature_syntax.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/cfg_feature_syntax.rs#L12)

- L12 [source]: `#[arcane]` on `fn add_v3(_token: X64V3Token, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L21 [source]: `#[arcane]` on `fn add_neon(_token: NeonToken, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L34 [source]: `#[arcane]` on `fn add_v4(_token: X64V4Token, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L42 [source]: `incant!(add(1.0, 2.0), [v4(avx512), v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L49 [source]: `incant!(add(1.0, 2.0), [v4(cfg(avx512)), v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L56 [source]: `incant!(add(1.0, 2.0))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L66 [source]: `#[arcane(cfg(avx512))]` on `fn guarded_v4(_token: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L72 [source]: `#[arcane]` on `fn always_v3(_token: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L93 [source]: `#[rite(v3, cfg(avx512))]` on `fn rite_guarded() -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L99 [source]: `#[rite(v3)]` on `fn rite_always() -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L108 [source]: `#[autoversion(cfg(avx512))]` on `fn auto_sum(_token: SimdToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L129 [source]: `#[autoversion(v4(avx512), v3, neon)]` on `fn auto_with_tier_gate(_token: SimdToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L142 [source]: `#[autoversion(v4(cfg(avx512)), v3, neon)]` on `fn auto_with_cfg_gate(_token: SimdToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L160 [source]: `#[autoversion]` on `fn $name(_token: SimdToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L183 [source]: `#[autoversion(cfg(avx512))]` on `fn $name(_token: SimdToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).

## [tests/compile_fail/autoversion_concrete_token.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/autoversion_concrete_token.rs#L4)

- L4 [expected-failure]: `#[archmage::autoversion]` on `fn process(_token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/compile_fail/featureless_simdtoken.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/featureless_simdtoken.rs#L8)

- L8 [expected-failure]: `#[arcane]` on `fn bad_simdtoken(token: impl archmage::SimdToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/compile_fail/missing_scalar.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/missing_scalar.rs#L15)

- L15 [expected-failure]: `incant!(add(1, 2))` → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/compile_fail/scalar_default_mutual_exclusion.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/scalar_default_mutual_exclusion.rs#L8)

- L8 [expected-failure]: `#[archmage::arcane]` on `fn add_v3(_: archmage::X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).
- L12 [expected-failure]: `incant!(add(1.0), [v3, scalar, default])` → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/compile_fail/scalar_not_in_tier_list.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/scalar_not_in_tier_list.rs#L17)

- L17 [expected-failure]: `incant!(add(1, 2), [v3])` → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/compile_fail/token_aliasing.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/token_aliasing.rs#L8)

- L8 [expected-failure]: `#[arcane]` on `fn evil(token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/compile_fail/token_shadowing.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/token_shadowing.rs#L11)

- L11 [expected-failure]: `#[arcane]` on `fn evil(_token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/compile_fail/unknown_generic_bound.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/unknown_generic_bound.rs#L7)

- L7 [expected-failure]: `#[arcane]` on `fn bad_generic<T: HasFma>(token: T, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/compile_fail/unknown_trait_bound.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/compile_fail/unknown_trait_bound.rs#L7)

- L7 [expected-failure]: `#[arcane]` on `fn bad_trait(token: impl HasAvx2, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/downstream-compat/compile-cost/src/lib.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/downstream-compat/compile-cost/src/lib.rs#L17)

- L17 [source]: `#[autoversion]` on `fn sum_squares(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L26 [source]: `#[autoversion(-wasm128, +v1)]` on `fn dot_product(a: &[f32], b: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L44 [source]: `#[rite(v3)]` on `fn add_chunk(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8]` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L53 [source]: `#[arcane]` on `pub fn add_arrays(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8]` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L63 [source]: `#[arcane]` on `fn scale_v3(_: X64V3Token, data: &mut [f32], factor: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L70 [source]: `#[arcane]` on `fn scale_neon(_: NeonToken, data: &mut [f32], factor: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L84 [source]: `incant!(scale(data, factor), [v3, neon, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L103 [source]: `#[arcane]` on `fn sum_simd_v3(token: X64V3Token, data: &[f32; 8]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L113 [source]: `incant!(sum_simd(data), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/downstream-compat/mixed-avx512/crate-with-avx512/src/lib.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/downstream-compat/mixed-avx512/crate-with-avx512/src/lib.rs#L7)

- L7 [source]: `#[arcane]` on `fn add_v4(_token: X64V4Token, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L11 [source]: `#[arcane]` on `fn add_v3(_token: X64V3Token, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L17 [source]: `incant!(add(a, b), [v4, v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/downstream-compat/mixed-avx512/crate-without-avx512/src/lib.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/downstream-compat/mixed-avx512/crate-without-avx512/src/lib.rs#L14)

- L14 [source]: `#[arcane]` on `fn mul_v4(_token: X64V4Token, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L18 [source]: `#[arcane]` on `fn mul_v3(_token: X64V3Token, a: f32, b: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L26 [source]: `incant!(mul(a, b), [v4(avx512), v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L31 [source]: `incant!(mul(a, b))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/downstream-compat/tier-modifiers/with-avx512/src/lib.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/downstream-compat/tier-modifiers/with-avx512/src/lib.rs#L13)

- L13 [source]: `#[autoversion(+v1)]` on `fn add_v1(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L33 [source]: `#[autoversion(+v4)]` on `fn unconditional_v4(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L54 [source]: `#[autoversion(+default)]` on `fn with_default(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L74 [source]: `#[autoversion(-neon, -wasm128)]` on `fn x86_only(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L88 [source]: `#[arcane]` on `fn inc_v3(_: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L102 [source]: `incant!(inc(x), [-neon, -wasm128, -v4, +v1])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L119 [source]: `#[magetypes(-neon, -wasm128, +v1)]` on `fn mt_mod(_token: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L148 [source]: `#[arcane]` on `fn cfg_v4(_: X64V4Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L153 [source]: `#[arcane]` on `fn cfg_v3(_: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L163 [source]: `incant!(cfg(x), [v4(cfg(avx512)), v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/downstream-compat/tier-modifiers/without-avx512/src/lib.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/downstream-compat/tier-modifiers/without-avx512/src/lib.rs#L14)

- L14 [source]: `#[autoversion]` on `fn defaults_no_avx512(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L28 [source]: `#[autoversion(+v1)]` on `fn plus_v1_no_avx512(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L48 [source]: `#[autoversion(+default)]` on `fn default_fallback_no_avx512(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L67 [source]: `#[autoversion(-neon, -wasm128)]` on `fn stripped_no_avx512(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L81 [source]: `#[autoversion(-v4)]` on `fn minus_v4_explicit(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L95 [source]: `#[arcane]` on `fn noavx_v3(_: X64V3Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L107 [source]: `incant!(noavx(x), [-neon, -wasm128])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L124 [source]: `#[magetypes(-neon, -wasm128)]` on `fn mt_noavx(_token: Token, x: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/associated_type_bound.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/associated_type_bound.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn sum_iter<I>(token: X64V3Token, iter: I) -> f32 where I: Iterator<Item = f32>,` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/box_dyn_return.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/box_dyn_return.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn make_adder(token: X64V3Token, offset: f32) -> Box<dyn Fn(f32) -> f32>` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/cfg_feature.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/cfg_feature.rs#L3)

- L3 [expansion-input]: `#[arcane(cfg(my_feature))]` on `fn process(token: X64V3Token, a: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/expand/arcane/closure_param.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/expand/arcane/closure_param.rs#L4)

- L4 [expansion-input]: `#[arcane]` on `fn apply(token: X64V3Token, data: &[f32; 4], f: impl Fn(f32) -> f32) -> [f32; 4]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

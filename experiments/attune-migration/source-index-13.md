# Compact source migration index

Exact old macro/call → destination pattern. Signatures are lexical, including macro templates, not compiler-resolved. Original file links plus L numbers locate every item. All support attributes and textual mentions remain in the raw inventory.

## [tests/tokenless_context.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/tokenless_context.rs#L23) (continued)

- L23 [source]: `incant!(entry(&[1, 2, 3]), [v3, neon, wasm128, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L27 [source]: `#[rite]` on `fn lower_v2(_token: X64V2Token, value: u32) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L31 [source]: `#[rite(v3)]` on `fn downgrade(value: u32) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L33 [source]: `incant!(lower(value), [v2, -scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L38 [source]: `#[rite(v3)]` on `fn skip_upgrades(value: u32) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L40 [source]: `incant!(lower(value), [v4, v3_crypto, v2, -scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L42 [source]: `#[rite(v3)]` on `fn explicit_position(value: u32) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L44 [source]: `dispatch_variant!(position(value, Token), [v3, -scalar])` → [P8](migration.md#p8); apply [contract rules](migration-contracts.md#p12).
- L46 [source]: `#[rite]` on `fn position_v3(value: u32, _token: X64V3Token) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L50 [source]: `#[arcane]` on `fn check_x86(_token: X64V3Token)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L67 [source]: `#[rite]` on `fn lower_v3(_token: X64V3Token, value: u32) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L71 [source]: `#[rite(v3)]` on `fn gated(value: u32) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L73 [source]: `incant!(lower(value), [v3(cfg(std)), v2, -scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L78 [source]: `#[rite(v3)]` on `fn fallback(value: u32) -> u32` [visibility: implicit; method/trait context unresolved, see source] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L80 [source]: `incant!(missing(value), [v4, default])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/v2_integration.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/v2_integration.rs#L10)

- L10 [source]: `#[magetypes]` on `pub fn dot(token: Token, a: &[f32], b: &[f32]) -> f32` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L17 [source]: `incant!(dot(a, b))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L20 [source]: `#[magetypes]` on `pub fn normalize(token: Token, data: &mut [f32])` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L33 [source]: `incant!(normalize(data))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L36 [source]: `#[magetypes]` on `pub fn scale(token: Token, data: &mut [f32], factor: f32)` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L45 [source]: `incant!(scale(data, factor) with token)` → [P7](migration.md#p7); apply [contract rules](migration-contracts.md#p12).
- L48 [source]: `#[magetypes]` on `pub fn min_max(token: Token, data: &[f32]) -> (f32, f32)` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L68 [source]: `incant!(min_max(data))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L71 [source]: `#[magetypes]` on `pub fn square(token: Token, data: &mut [f32])` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L79 [source]: `#[magetypes]` on `pub fn add_const(token: Token, data: &mut [f32], val: f32)` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L88 [source]: `incant!(square(data))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L89 [source]: `incant!(add_const(data, 1.0))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L92 [source]: `#[magetypes]` on `pub fn identity(token: Token, x: f32) -> f32` [visibility: pub] → [P4](migration.md#p4); apply [contract rules](migration-contracts.md#p12).
- L99 [source]: `incant!(identity(x))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L152 [source]: `simd_route!(dot(a, b))` → [P8](migration.md#p8); apply [contract rules](migration-contracts.md#p12).

## [tests/wasm_intrinsics_exercise.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/wasm_intrinsics_exercise.rs#L41)

- L41 [source]: `#[arcane]` on `fn exercise_integer_arithmetic(token: Wasm128Token)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L98 [source]: `#[arcane]` on `fn exercise_float_arithmetic(token: Wasm128Token)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L143 [source]: `#[arcane]` on `fn exercise_comparisons(token: Wasm128Token)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L208 [source]: `#[arcane]` on `fn exercise_bitwise(token: Wasm128Token)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L228 [source]: `#[arcane]` on `fn exercise_shifts(token: Wasm128Token)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L255 [source]: `#[arcane]` on `fn exercise_conversions(token: Wasm128Token)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L311 [source]: `#[arcane]` on `fn exercise_lane_ops(token: Wasm128Token)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L350 [source]: `#[arcane]` on `fn exercise_boolean_reductions(token: Wasm128Token)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L386 [source]: `#[arcane]` on `fn exercise_shuffle_swizzle(token: Wasm128Token)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L405 [source]: `#[arcane]` on `fn exercise_extended_multiply(token: Wasm128Token)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L445 [source]: `#[arcane]` on `fn exercise_saturating_arithmetic(token: Wasm128Token)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/x86_crypto_intrinsics.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/x86_crypto_intrinsics.rs#L40)

- L40 [source]: `#[arcane]` on `fn exercise_pclmulqdq(token: X64CryptoToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L71 [source]: `#[arcane]` on `fn exercise_aes_enc_dec(token: X64CryptoToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L100 [source]: `#[arcane]` on `fn exercise_aes_key_assist(token: X64CryptoToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L139 [source]: `#[arcane]` on `fn exercise_vpclmulqdq(token: X64V3CryptoToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L167 [source]: `#[arcane]` on `fn exercise_vaes_256(token: X64V3CryptoToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L215 [source]: `#[arcane]` on `fn exercise_gfni(token: X64V3GfniCryptoToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

# Compact source migration index

Exact old macro/call → destination pattern. Signatures are lexical, including macro templates, not compiler-resolved. Original file links plus L numbers locate every item. All support attributes and textual mentions remain in the raw inventory.

## [tests/autoversion_macro.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/autoversion_macro.rs#L190) (continued)

- L190 [source]: `#[autoversion(_self = Buffer)]` on `fn scale_all(&mut self, _token: SimdToken, factor: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L207 [source]: `#[autoversion]` on `fn sum(&self, _token: SimdToken) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L212 [source]: `#[autoversion]` on `fn double_all(&mut self, _token: SimdToken)` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L251 [source]: `#[autoversion]` on `fn sum_wildcard(_: SimdToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L307 [source]: `#[autoversion(v3, neon)]` on `fn product(&self, _token: SimdToken) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L326 [source]: `#[autoversion]` on `fn weighted_sum(&self, _token: SimdToken, weight: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L349 [source]: `#[autoversion]` on `fn into_sum(self, _token: SimdToken) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L375 [source]: `#[autoversion]` on `fn add_wildcards(_: SimdToken, _: &[f32], _: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L392 [source]: `#[autoversion]` on `fn min_max(_token: SimdToken, data: &[f32]) -> (f32, f32)` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L419 [source]: `#[autoversion]` on `fn find_first_negative(_token: SimdToken, data: &[f32]) -> Option<usize>` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L434 [source]: `#[autoversion]` on `fn sum_i64(_token: SimdToken, data: &[i64]) -> i64` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L449 [source]: `#[autoversion]` on `fn all_positive(_token: SimdToken, data: &[f32]) -> bool` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L499 [source]: `#[autoversion]` on `fn sum_large(_token: SimdToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L523 [source]: `#[autoversion]` on `fn clamp_inplace(_token: SimdToken, data: &mut [f32], lo: f32, hi: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L547 [source]: `#[autoversion]` on `fn values_ref(&self, _token: SimdToken) -> &[f32]` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L565 [source]: `#[autoversion]` on `fn noop(_token: SimdToken, _data: &[f32])` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L580 [source]: `#[autoversion]` on `fn sum_array<const N: usize>(_token: SimdToken, data: &[f32; N]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L607 [source]: `#[autoversion]` on `fn make_zeros<const N: usize>(_token: SimdToken) -> [f32; N]` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L619 [source]: `#[autoversion]` on `fn reshape<const M: usize, const N: usize>(_token: SimdToken, data: &[f32; M]) -> [f32; N]` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L640 [source]: `#[autoversion]` on `fn sum_generic<const N: usize, T: Default + Copy + core::ops::Add<Output = T>>( _token: SimdToken, data: &[T; N], ) -> T` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L662 [source]: `#[autoversion]` on `fn chunk_sum<const CHUNK: usize>(_token: SimdToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L686 [source]: `#[autoversion]` on `fn extract<const N: usize>(&self, _token: SimdToken) -> [f32; N]` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L698 [source]: `#[autoversion(_self = ConstGenericBuf)]` on `fn extract_nested<const N: usize>(&self, _token: SimdToken) -> [f32; N]` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L730 [source]: `#[autoversion(v3, neon)]` on `fn const_sum_explicit<const N: usize>(_token: SimdToken, data: &[f32; N]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L749 [source]: `#[autoversion]` on `fn first_n_sum<'a, const N: usize>(_token: SimdToken, data: &'a [f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L774 [source]: `#[autoversion]` on `fn fill_row<const BPP: usize>(&self, _token: SimdToken, out: &mut Vec<u8>)` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L781 [source]: `#[autoversion(_self = PixelRow)]` on `fn fill_row_nested<const BPP: usize>(&self, _token: SimdToken, out: &mut Vec<u8>)` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L838 [source]: `#[autoversion]` on `fn sum_plain(&self, _token: SimdToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L844 [source]: `#[autoversion(_self = Accum)]` on `fn sum_nested(&self, _token: SimdToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L869 [source]: `#[autoversion]` on `fn inner_product(a: &[f32], b: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L907 [source]: `#[autoversion(v3, neon)]` on `fn scale_sum(data: &[f32], factor: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L924 [source]: `#[autoversion]` on `fn fill_chunked<const N: usize>(data: &mut [f32], val: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L949 [source]: `#[autoversion]` on `fn total(&self) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L954 [source]: `#[autoversion]` on `fn scale_all(&mut self, factor: f32)` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L961 [source]: `#[autoversion]` on `fn into_total(self) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L966 [source]: `#[autoversion]` on `fn values_ref(&self) -> &[f32]` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L971 [source]: `#[autoversion]` on `fn weighted_sum(&self, weight: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L976 [source]: `#[autoversion(v3, neon)]` on `fn product(&self) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L1061 [source]: `#[autoversion(_self = TokenlessNested)]` on `fn biased_sum(&self, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L1066 [source]: `#[autoversion(_self = TokenlessNested)]` on `fn biased_scale(&mut self, data: &[f32], factor: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L1090 [source]: `#[autoversion]` on `fn parity_explicit(_token: SimdToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L1095 [source]: `#[autoversion]` on `fn parity_tokenless(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L1128 [source]: `#[arcane]` on `fn nested_dispatch_v3(_token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L1135 [source]: `#[autoversion(v3, neon)]` on `fn nested_dispatch_fallback(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L1147 [source]: `incant!(nested_dispatch(data), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L1226 [source]: `#[arcane]` on `fn process_v3(&self, _token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L1235 [source]: `#[autoversion(v3, neon)]` on `fn process_auto(&self, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L1277 [source]: `#[arcane]` on `fn bridgeless_v3(_token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L1284 [source]: `#[autoversion(v3, neon)]` on `fn bridgeless_scalar(_: ScalarToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L1291 [source]: `incant!(bridgeless(data), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L1351 [source]: `#[arcane]` on `fn process_v3(&self, _token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L1357 [source]: `#[autoversion(v3, neon)]` on `fn process_scalar(&self, _: ScalarToken, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L1388 [source]: `#[arcane]` on `fn default_tier_v3(_token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L1394 [source]: `#[autoversion(v3, neon)]` on `fn default_tier_default(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L1401 [source]: `incant!(default_tier(data), [v3, default])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L1434 [source]: `#[autoversion(v3, neon, default)]` on `fn auto_with_default(data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L1468 [source]: `#[arcane]` on `fn process_v3(&self, _token: X64V3Token, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L1473 [source]: `#[autoversion(v3, neon)]` on `fn process_default(&self, data: &[f32]) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L1508 [source]: `#[arcane]` on `fn resample_scalar_sol1_v3(_: X64V3Token, data: &[f32], factor: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L1513 [source]: `#[autoversion(v3, neon)]` on `fn resample_scalar_sol1_scalar(_: ScalarToken, data: &[f32], factor: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L1519 [source]: `incant!(resample_scalar_sol1(data, factor), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L1534 [source]: `#[arcane]` on `fn resample_default_sol2_v3(_: X64V3Token, data: &[f32], factor: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L1539 [source]: `#[autoversion(v3, neon)]` on `fn resample_default_sol2_default(data: &[f32], factor: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L1545 [source]: `incant!(resample_default_sol2(data, factor), [v3, default])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).
- L1560 [source]: `#[arcane]` on `fn resample_bridge_sol3_v3(_: X64V3Token, data: &[f32], factor: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L1565 [source]: `#[autoversion(v3, neon)]` on `fn resample_bridge_sol3_auto(data: &[f32], factor: f32) -> f32` [visibility: implicit; method/trait context unresolved, see source] → [P3](migration.md#p3); apply [contract rules](migration-contracts.md#p12).
- L1576 [source]: `incant!(resample_bridge_sol3(data, factor), [v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/avx512-cfg-tests/v4-all-token-forms/src/lib.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/avx512-cfg-tests/v4-all-token-forms/src/lib.rs#L22)

- L22 [source]: `#[arcane(import_intrinsics)]` on `pub fn f(_token: X64V4Token) -> core::arch::x86_64::__m512` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L32 [source]: `#[arcane(import_intrinsics)]` on `pub fn f(_token: Avx512Token) -> core::arch::x86_64::__m512` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L42 [source]: `#[arcane(import_intrinsics)]` on `pub fn f(_token: Server64) -> core::arch::x86_64::__m512` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L52 [source]: `#[arcane(import_intrinsics)]` on `pub fn f(_token: X64V4xToken) -> core::arch::x86_64::__m512` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L65 [source]: `#[arcane(import_intrinsics)]` on `pub fn f(_token: Avx512Fp16Token) -> core::arch::x86_64::__m512` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L75 [source]: `#[arcane(import_intrinsics)]` on `pub fn f(_token: impl HasX64V4) -> core::arch::x86_64::__m512` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L85 [source]: `#[arcane(import_intrinsics)]` on `pub fn f<T: HasX64V4>(_token: T) -> core::arch::x86_64::__m512` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L95 [source]: `#[rite(v4, import_intrinsics)]` on `pub fn f() -> core::arch::x86_64::__m512` [visibility: pub] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L105 [source]: `#[rite(v4, v3, import_intrinsics)]` on `pub fn f() -> f32` [visibility: pub] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).
- L117 [source]: `#[arcane(import_intrinsics)]` on `pub fn v3_import(_token: X64V3Token) -> core::arch::x86_64::__m256` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L124 [source]: `#[arcane]` on `pub fn v4_no_import(_token: X64V4Token) -> core::arch::x86_64::__m512` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L131 [source]: `#[rite(v3, import_intrinsics)]` on `pub fn v3_rite() -> core::arch::x86_64::__m256` [visibility: pub] → [P2](migration.md#p2); apply [contract rules](migration-contracts.md#p12).

## [tests/avx512-cfg-tests/v4-import-intrinsics-no-feature/src/lib.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/avx512-cfg-tests/v4-import-intrinsics-no-feature/src/lib.rs#L13)

- L13 [expected-failure]: `#[arcane(import_intrinsics)]` on `pub fn v4_load_add(token: X64V4Token, a: &[f32; 16], b: &[f32; 16]) -> [f32; 16]` [visibility: pub] → [P9](migration.md#p9); apply [contract rules](migration-contracts.md#p12).

## [tests/avx512-cfg-tests/v4-no-import-intrinsics/src/lib.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/avx512-cfg-tests/v4-no-import-intrinsics/src/lib.rs#L13)

- L13 [source]: `#[arcane]` on `pub fn v4_add_values(_token: X64V4Token, a: core::arch::x86_64::__m512, b: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L20 [source]: `#[arcane]` on `pub fn v4_setzero(_token: X64V4Token) -> core::arch::x86_64::__m512` [visibility: pub] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

## [tests/avx512-cfg-tests/with-avx512-feature/src/lib.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/avx512-cfg-tests/with-avx512-feature/src/lib.rs#L19)

- L19 [source]: `#[arcane(import_intrinsics)]` on `fn v4_value_ops(_token: X64V4Token, a: &[f32; 16], b: &[f32; 16]) -> [f32; 16]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L33 [source]: `#[arcane(import_intrinsics)]` on `fn add_v4(_token: X64V4Token, a: &[f32; 16], b: &[f32; 16]) -> [f32; 16]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L43 [source]: `#[arcane(import_intrinsics)]` on `fn add_v3(_token: X64V3Token, a: &[f32; 16], b: &[f32; 16]) -> [f32; 16]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L62 [source]: `incant!(add(a, b), [v4, v3, scalar])` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/avx512-cfg-tests/with-avx512-passthrough/src/lib.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/avx512-cfg-tests/with-avx512-passthrough/src/lib.rs#L16)

- L16 [source]: `#[arcane(import_intrinsics)]` on `fn add_v4(_token: X64V4Token, a: &[f32; 16], b: &[f32; 16]) -> [f32; 16]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L27 [source]: `#[arcane(import_intrinsics)]` on `fn add_v3(_token: X64V3Token, a: &[f32; 16], b: &[f32; 16]) -> [f32; 16]` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L47 [source]: `incant!(add(a, b))` → [P5](migration.md#p5); apply [contract rules](migration-contracts.md#p12).

## [tests/avx512_intrinsics_exercise.rs](https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/tests/avx512_intrinsics_exercise.rs#L49)

- L49 [source]: `#[arcane]` on `fn exercise_avx512f(token: X64V4xToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L262 [source]: `#[arcane]` on `fn exercise_avx512f_vl(token: X64V4xToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L326 [source]: `#[arcane]` on `fn exercise_avx512bw(token: X64V4xToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L429 [source]: `#[arcane]` on `fn exercise_avx512bw_vl(token: X64V4xToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L458 [source]: `#[arcane]` on `fn exercise_avx512dq(token: X64V4xToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L531 [source]: `#[arcane]` on `fn exercise_avx512dq_vl(token: X64V4xToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L568 [source]: `#[arcane]` on `fn exercise_avx512cd(token: X64V4xToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L604 [source]: `#[arcane]` on `fn exercise_avx512cd_vl(token: X64V4xToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L640 [source]: `#[arcane]` on `fn exercise_avx512vbmi(token: X64V4xToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).
- L674 [source]: `#[arcane]` on `fn exercise_avx512vbmi2(token: X64V4xToken)` [visibility: implicit; method/trait context unresolved, see source] → [P1](migration.md#p1); apply [contract rules](migration-contracts.md#p12).

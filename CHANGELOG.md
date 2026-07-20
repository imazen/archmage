# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES

- Remove `guaranteed()` from `SimdToken` trait — use `compiled_with()` instead (deprecated since 0.6.0, zero callers)
- Remove width traits `Has128BitSimd`, `Has256BitSimd`, `Has512BitSimd` — use concrete tokens or tier traits (`HasX64V2`, `HasX64V4`) instead (deprecated since 0.9.9; `Has256BitSimd` only enables AVX, not AVX2/FMA)
- Remove `SimdToken` parameter support from `#[autoversion]` — use tokenless (recommended) or `ScalarToken` for `incant!` nesting (deprecated since 0.9.11)
- Remove `_self = Type` from `#[autoversion]` — plain `self` works in sibling mode; `#[autoversion]` can't do trait impls anyway
- Deprecate `incant!` passthrough mode (`with token`) — zero downstream uses; `#[rite]` multi-tier or direct `IntoConcreteToken` dispatch are better alternatives
- Require `scalar` or `default` in explicit `incant!` tier lists (currently auto-appended with deprecation warning)
- Require explicit `tier(cfg(feature))` syntax — remove implicit `cfg_feature` auto-gating on v4/v4x
- Make `w512` non-default in magetypes — users who need 512-bit types add `features = ["w512"]`; saves ~25% build time for the majority who don't

## [0.9.28] - 2026-07-20

### Added

- **Structure-aware intrinsic soundness scanner** (`xtask/src/soundness.rs`, runs in `just soundness` / `validate` / `generate` / `ci`): derives gating contexts from the code itself (impl-for-token blocks, `#[target_feature]` fns, token-typed params — contexts union), detects intrinsics by membership in the 10,884-entry stdarch database instead of prefix whitelists, and hard-fails on feature mismatches, ungated intrinsics (incl. trait default bodies), unknown intrinsic-shaped names, structural-rule breaches (`MaybeUninit`/`mem::zeroed`/forging in magetypes, bare `transmute` outside backend impls, `Default`/serde/bytemuck on SIMD wrappers, backend-trait methods without a `self` receiver), and missing SAFETY discipline. Vacuous-pass guards (global floor 4,000 verified calls + per-file floors; 4,478 measured at introduction) make an empty scan a failure, and the scanner's own unit tests plant every violation class and assert it fires (`cargo test -p xtask`, now CI step 6). (e30bfea, 865ba31)
- `docs/SOUNDNESS.md` — canonical audit entry point: the token-as-proof invariant, the complete `unsafe` inventory (81 forge sites + ~1,960 contract-covered backend blocks + 225 commented layout casts), what each mechanical verifier proves, trust boundaries, the by-hand audit procedure, and an incident log. (865ba31)
- `tests/apple_fallback_guard.rs` — source-scan guards (run on every CI lane) pinning the Apple Silicon detection fallback to provably-M1+ hosts, plus a macOS-host execution check. (c2d92c2)

### Changed

- **magetypes: out-of-range shift-by-const `N` is now a uniform compile-time failure on every backend** ([#63](https://github.com/imazen/archmage/issues/63)). The generic front-ends for `shl_const`, `shr_arithmetic_const`, and `shr_logical_const` const-assert `N ∈ 0..=lane_bits-1` at monomorphization. Previously out-of-range `N` was backend-specific: x86 clamped/zeroed, WASM masked `N mod lane_bits`, scalar split between compile-error (i32/i64) and wrap (i8/i16), NEON errored. What compiles is now bit-identical on all backends; in-range behavior is unchanged. Strictly a tightening: code relying on the undocumented out-of-range behavior no longer compiles. Enforcement pinned by `compile_fail` doctests in `simd/mod.rs`. (54b34bc)

### Fixed

- **magetypes: four right-shift-by-const backend bugs** ([#63](https://github.com/imazen/archmage/issues/63)). NEON rejected `N == 0` at compile time (`vshrq_n_*` immediate encoding has no shift-0 form; now lowered through `vshlq_*` with a negated splat count, which LLVM folds to the immediate `sshr`/`ushr` forms — plus a `const` assert making the portable contract `0..=lane_bits-1` explicit, which also rejects the formerly-NEON-only `N == lane_bits`). x86 v3 `i8x16`/`i8x32` `shr_arithmetic_const::<0>` silently turned negative lanes into `-1` (`wrapping_shl` wraps the shift amount at `N == 0`, producing a full fill mask). AVX-512 `i8x64` `shr_arithmetic_const` (V4 and V4x tokens) was byte-identical to the logical shift — zero-fill instead of sign-fill at every `N`. WASM signed `i8`/`i16` `shr_logical_const` called the i-prefixed (arithmetic) intrinsic. All four are locked in by `magetypes/tests/shift_const_boundaries.rs`: `N == 0` and `N == lane_bits-1` vs scalar-computed references, 24 integer generic types, scalar/X64V3/X64V4/NEON/WASM backends. Out-of-range-`N` behavior on non-NEON backends is still divergent — tracked in #63. (0dc8fbe)
- **archmage: Apple aarch64 detection fallback was unsound on device iOS/tvOS/watchOS/visionOS.** `__impl_aarch64_apple_or_runtime_check!` returned `true` unconditionally under `target_vendor = "apple"` with no OS narrowing, so on device targets (hardware back to A7–A12 that lacks crc/rdm/dotprod/fp16/fhm/fcma/sha3) safe code could summon `Arm64V2Token`/`NeonCrcToken`/`NeonSha3Token` without the CPU backing them — undefined behavior (in practice SIGILL). The unconditional arm is now gated to macOS, Mac Catalyst, and the aarch64 simulators (all exclusively Apple Silicon hosts); device targets use genuine runtime detection and otherwise fail closed. (c2d92c2)
- **xtask: both intrinsic-vs-token validators had gone vacuous** — they iterated `[[magetypes_file]]` registry mappings that no longer exist after the generic-backend refactor, so `just soundness` and `just validate-tokens` passed while verifying **zero** intrinsic calls (and the old ARM regex was a prefix whitelist that silently missed `vrecpe`/`vqtbl`/`vclz`-class intrinsics even when it ran). Replaced by the structure-aware scanner above; dead registry plumbing and its stale failing test removed. (e30bfea)
- magetypes: generated byte/slice casts (`as_bytes`, `from_bytes`, `cast_slice`, `bitcast_*`, …) now open with an inline-const assert tying `size_of::<Self>()` to the exact literal size they cast to, evaluated per backend at monomorphization — a future mis-sized `Repr` becomes a compile error instead of an out-of-bounds view. Their SAFETY comments also now state the true invariant (`#[repr(C)]` + size assert + all-bit-patterns-valid lanes) instead of claiming `repr(transparent)`. (30a6b44)
- magetypes: `cargo test -p magetypes` (default features) failed to compile — `tests/reciprocal_precision.rs` referenced the V4-native `f64x8` backend, which only exists under the non-default `avx512` feature; the check is now feature-gated, so the default-features lane compiles and runs 1,590 tests it previously couldn't. (1711c56)
- Generated backends docs claimed trait methods are "associated functions (no `self`/token parameter)"; they take `self`, and that receiver is the anti-UFCS soundness mechanism (30 doc sites regenerated). (865ba31)

### Removed

- magetypes: the dead `#[doc(hidden)]` `#[macro_export]` op-impl templates (`impl_arithmetic_ops!`, `impl_int_arithmetic_ops!`, `impl_int_mul_op!`, `impl_bitwise_ops!`, `impl_assign_ops!`, `impl_float_assign_ops!`, `impl_neg!`, `impl_index!`). Zero invocations since the concrete per-platform types were retired, never documented, and each expanded to token-less `unsafe { $intrinsic }` blocks — an unsound-by-construction footgun for any downstream invoker. (865ba31)

### Documentation

- magetypes: documented all 60 aarch64/wasm32 bare-name type aliases in `simd/mod.rs` — every target now compiles `missing_docs`-clean (previously 30 warnings per ARM/WASM build). (9843557)
- Every generated `forge_token_dangerously()` call site (81) now carries a site-specific `// SAFETY:` comment, and every intrinsic-bearing generated impls file carries a `# Safety (audit contract)` header stating the uniform justification for its unsafe blocks; both enforced by the soundness scanner so coverage cannot regress. (865ba31)
- README overhaul for both `archmage` and `magetypes`, conformed to the zen README conventions: badge row now includes `&label=CI`, an MSRV 1.89 badge, and a `license` → `#license` anchor (all badges linked); added a `## Quick start` section (copy-paste `#[arcane]` + `incant!` runtime dispatch for archmage, `#[magetypes]` for magetypes) and made every body link absolute. Split each crate's crates.io README into a generated, badge-free `README.crates.md` (`readme = "README.crates.md"`, `include` updated) so version-pinned crates.io pages no longer show HEAD-reflecting badges, while the GitHub READMEs keep the full row. Replaced the hand-maintained crosslink footer with the canonical registry-rendered footer (repo links), placed last after License. (49023ce)

## [0.9.27] - 2026-06-23

### Added

- Versioned public-API surface snapshots at `docs/public-api/{archmage,magetypes}.txt`, regenerated by `tests/public_api_doc.rs` on every `cargo test` (`ZEN_API_DOC=check` verifies in CI's clippy job, `=off` skips; `just api-doc` / `just api-doc-check` locally).
- macros: **`incant!(foo(args) without token)`** — a tokenless variant-call modifier (sibling to `with token`). Inside a tier-macro body (`#[rite]`/`#[arcane]`/`#[magetypes]`/`#[autoversion]`) it rewrites to the caller's exact-tier variant called with **no token and no `summon()`**: `foo_<caller_tier>(args)`. This is how you compose **tokenless** multi-tier helpers (tier-based `#[rite(v3, neon)]`) from inside another tier body — the token-first nested-`incant!` rewrite can't reach them (no token to thread). The call is a direct, safe matching-feature call (the caller is already in that `#[target_feature]` region); a missing variant or feature mismatch is a compile error. Only valid inside a tier-macro body (clear error in plain code); the target must be multi-tier/suffixed (always emits a `_<tier>` suffix); takes no tier list (resolves to the caller's own tier). The body rewriter also now leaves non-rewritten `incant!` calls byte-identical (emits the original tokens verbatim instead of reconstructing), and tokenless tier bodies (`#[rite(v3)]`) + `#[autoversion]` scalar/default variants are scanned so `without token` resolves there too. Spec + table: `docs/SPEC-INCANT-REWRITING.md`; expand tests in `tests/expand/rewrite/without_token_*` (#57).
- magetypes: **deterministic, cross-platform bit-identical reciprocal / reciprocal-sqrt** for f32 (`f32x4`/`f32x8`/`f32x16`), for callers who need the *same bits on every machine* (x86/ARM/WASM): `rcp_approx_portable` / `rsqrt_approx_portable` (~8-bit estimate), `recip_newton_portable` / `rsqrt_newton_portable` (one opt-in Newton step, ~16-bit), and `recip_portable` / `rsqrt_portable` (full precision via IEEE div/sqrt). Determinism comes from an integer bit-trick seed (pure integer math) plus Newton steps in plain IEEE-754 `mul`/`sub` — never `mul_add` (WASM SIMD has no fused FMA, and FMA-vs-non-FMA itself diverges). Hardware estimate instructions (`rsqrtps`/`vrsqrte`) are deliberately avoided — their bits differ across vendors/generations. Full precision uses correctly-rounded div/sqrt (already bit-identical, true ~24-bit). Implemented once on the generic type from existing ops, so no per-backend code. **Measured bit-identical x86 == aarch64 (QEMU) over 73,692 inputs, all 6 variants** (`magetypes/tests/portable_reciprocal.rs`): rsqrt 9.2/17.7/23.5-bit, rcp 8.2/16.4/24.0-bit. Additive (`cargo semver-checks` clean); added alongside the unchanged per-platform fast methods. f64 not yet (92ec698).

### Changed

- magetypes: **the fast per-platform `rcp_approx` / `rsqrt_approx` (f32) now guarantee a ≥~12-bit floor** by the cheapest *measured* path on each target, instead of forwarding a bare hardware estimate everywhere. x86 keeps the raw `rcpps`/`rsqrtps` estimate (~12-bit); **ARM adds one fused FRECPS/FRSQRTS step** to the ~8-bit `vrecpe`/`vrsqrte` estimate (~16-bit). On **WASM/scalar** (no hardware estimate): `rcp_approx` is exact division (~24-bit — benchmarked as fast as, and more accurate than, a bit-hack), while **`rsqrt_approx` is an integer bit-hack seed + 2 Newton steps (~17-bit)** replacing the expensive `sqrt`+division — **measured 1.9× faster on WASM and 11× faster on scalar** (`benchmarks/wasm_scalar_reciprocal_2026-06-21.md`), and bit-identical to `rsqrt_approx_portable` + one `rsqrt_newton_portable`. Previously ARM/WASM callers got the unrefined ~8-bit estimate (or identity), and WASM/scalar `rsqrt_approx` paid for a full `sqrt`+division. **Each backend owns its own ≥12-bit estimate**; the generic wrapper is a thin forward (no dispatch consts, no new trait bounds). Measured floors (`magetypes/tests/reciprocal_precision.rs`): x86 11.9/12.2-bit, ARM 17.1/16.5-bit, WASM/scalar 24.3/17.7-bit. For the *same bits on every machine* use the `_portable` family above; for full precision use `recip`/`rsqrt`.
- magetypes: **full-precision ARM/NEON `recip` / `rsqrt` now use the native Newton-Raphson assist instructions** FRECPS (`vrecpsq`) and FRSQRTS (`vrsqrtsq`) for the two refinement steps, instead of hand-rolled `mul`/`sub` + 2.0/3.0/0.5 splats. The fused `2 − a·y` / `(3 − a·y·y)/2` is one rounding and measurably faster on real silicon — **1.32× (rcp) / 1.51× (rsqrt) on an Ampere Neoverse-N1** (`benchmarks/rsqrt_arm_neoverse-n1_2026-06-21.md`) — at the same full f32 (~24-bit) precision. (At this commit the fast `_approx` methods were still the raw `vrecpe`/`vrsqrte` estimate; the ≥12-bit floor above was added separately.) The full-precision NEON `rsqrt` now does two Newton steps (it previously did only one in one path, ~16-bit rather than full). (2ad2f17)
- magetypes: **the generic types reproduce the retired concrete types' native block-op codegen** — `to_u8` / `store_4_rgba_u8` / `store_8_rgba_u8` / `transpose_8x8` (f32x4/f32x8) regained per-backend native sequences via sealed-trait overrides (no public-API change): x86 `cvtps`+`packs`+`packus`(+`pshufb` interleave); ARM `FCVTNS`+`SQXTN`+`SQXTUN` and the `R|G<<8|B<<16|A<<24` shift-or pack; AVX2 `unpck`+`shuffle`+`permute2f128` for the 8×8 transpose. LLVM does **not** recover those scalar fallbacks (the scalar `roundevenf`, and the cross-lane gather that bloated to ~60 AVX2 ops), so this keeps the retirement perf-neutral. Verified native asm (x86 + aarch64) + value parity on x86 / aarch64(qemu) / wasm(wasmtime); `from_u8` and the shuffle round-trips were measured to recover natively and left as-is (398660da).

### Fixed

- magetypes: **generic `log2_midp(+inf)` now returns `+inf`** instead of `128` (infinity's raw unbiased exponent that the polynomial path produces). The retired concrete `log2_midp` special-cased `+inf`; the generic one — now the only implementation, reached via `simd::v3::f32x8` etc. — blended in only the zero→`-inf` and negative→`NaN` cases. Adding the `+inf → +inf` blend also corrects the `ln` / `log10` / `pow` paths built on it. Surfaced by the full concrete-type retirement routing `transcendental_edge_cases::log2_midp_inf_is_inf` through the generic type.
- magetypes: **f64 `recip` / `rsqrt` now reach full precision (~52–53 bit) on every width and backend**, and the **scalar `f32x16`/`f64x8` reciprocals (previously the identity) are fixed.** Four latent defects, each verified across x86 / aarch64 (QEMU) / wasm32 (wasmtime) and now including `ScalarToken` at every width by `magetypes/tests/reciprocal_precision.rs`: **(1)** x86 `f64x2` / `f64x4` have no hardware reciprocal estimate below AVX-512 and fell through to the trait-default `recip = rcp_approx = a` — they **returned the input unchanged** (the identity); they now use exact IEEE division / sqrt. **(2)** the AVX-512 `f64x8` did a single Newton step from the 14-bit `rcp14`/`rsqrt14` estimate (~28-bit, *half* of full f64) — now two steps (~52-bit), and `rcp_approx`/`rsqrt_approx` expose the raw 14-bit estimate. **(3)** the ARM/WASM `f64x8` (and `f32x16`) W512 polyfills omitted the reciprocal methods entirely, so they too were the identity; they now delegate per-lane-group to the narrower backend. **(4)** the **scalar `f32x16` and `f64x8`** W512 backends *also* never overrode the reciprocals, so on the no-SIMD fallback they returned the input unchanged — now overridden (exact division / sqrt, with the f32 `rsqrt_approx` bit-hack). ARM native `f64x2` / polyfill `f64x4` also go from two to three Newton steps (the f64 mantissa needs three from the 8-bit NEON estimate). Before this, only the AVX-512 `f64x8` path produced even approximately-right f64 reciprocals.
- docs: archmage README now has a runtime-dispatched slice-body example — an element-wise op over a whole `&mut [f32]` (windowing into `&[f32; 8]` via `try_into` + scalar tail + `incant!`), with a verified x86 `_mm256_loadu_ps`/`_mm256_storeu_ps` body, a NEON `vld1q_f32`/`vst1q_f32`/`vmulq_f32` variant, and the unaligned-vs-NEON-element-aligned memory caveats. Prior examples only showed `&[f32; N]` fixed arrays. Found via an insulated external-developer usability test.
- docs: magetypes README now has a `[dependencies]` block with real versions (`magetypes`/`archmage` `0.9.26`) stating that `archmage` is a required direct dependency for the macros and tokens, an import-path table, and an explicit note that `load`/`store` are **unaligned** (`_mm256_loadu_ps`/`_mm256_storeu_ps`, only `f32` alignment required) so `partition_slice_mut` works on any slice. Previously there was no dependency block, making a compilable Cargo.toml impossible from the README alone (build-blocker). Found via an insulated external-developer usability test.

### Removed

- magetypes: **fully retired the legacy concrete per-platform SIMD type structs (BREAKING)** — the hand-written `#[repr(transparent)]` structs at `simd::x86::w128`/`w256`/`w512`, `simd::arm::w128`, `simd::wasm::w128`, and the entire `simd::polyfill` module are gone (~1.5 MB of generated code + their xtask generators). Every SIMD type is now the single generic strategy-pattern type `simd::generic::TYPE<Token>`, which carries the token as a field and is structurally sound. The per-token namespaces (`simd::v3`, `v4`, `v4x`, `neon`, `wasm128`, `scalar`) are now thin **generic aliases** — `simd::v3::f32x8` is `generic::f32x8<X64V3Token>` rather than a re-exported concrete struct (x86 128/256-bit aliases all use `X64V3Token`, the sole x86 ≤256-bit backend; AVX-512 adds nothing below 512-bit). The bare `simd::fNxM` names already resolved to the generic types. `implementation_name()` and all results are preserved; the migration surfaced and fixed one divergence the concrete path had masked — generic `log2_midp(+inf)` now returns `+inf` (see Fixed). `WidthDispatch`'s x86 512-bit associated types are now the always-available `[f32x8; 2]`/`[i8x32; 2]`/… array form (the `w512`-gated generic 512-bit types can't back an unconditional trait), matching the ARM/WASM array fallback. No downstream consumer used the concrete full paths or the `polyfill` module (verified across the workspace).
- magetypes: **removed the unsound concrete raw-memory constructors (BREAKING, no deprecation)** — `from_u8(&[u8; N])`, `load_4_rgba_u8(&[u8; 16])`, `load_8_rgba_u8(&[u8; 32])`, `load_8x8(&[f32; 64])` on the *concrete* per-platform float types (`simd::x86::w128::f32x4`, `simd::v3::f32x8`, `simd::arm::w128::f32x4`, `simd::wasm::w128::f32x4`). They built a SIMD register from raw scalar memory with **no token and no `Self` to borrow proof from**, so on x86 they executed SSE4/AVX2 (e.g. `_mm256_cvtepu8_epi32`) with no guarantee the CPU has the tier — a safe `fn` that could **SIGILL**. The **generic** equivalents (`generic::fNxM<T>::from_u8(token, …)` / `load_*(token, …)`) take the token (proof) explicitly and are the supported, sound path; the bare `simd::fNxM` names already route to them. Sound siblings (`to_u8`, `store_*`, `deinterleave_4ch`, `transpose_4x4` — each takes a `Self`/`[Self; N]`, a transitive proof) are retained. Zero downstream consumers (verified across the workspace), so this narrow break — together with the full concrete retirement above — ships on the **0.9 chain** (next `0.9.x`) rather than forcing a `0.10`.
- magetypes: **`magetypes::prelude` no longer re-exports the platform-"best" aliases (BREAKING)** — `F32Vec`/`F64Vec`/`I32Vec`/`I64Vec`/`U32Vec`/`U8Vec`/`U16Vec`/`I8Vec`/`I16Vec`/`U64Vec`, `RecommendedToken`, and `LANES`/`F32_LANES`/`F64_LANES`/`I32_LANES` are gone. They hid the lane width **and** token, so the same source compiled to different lane counts per architecture — the project bans them outside `#[magetypes]`. The prelude now re-exports the generic `simd::generic::*` types plus `SimdToken`; name the token explicitly (`f32x8::<X64V3Token>`) and pair with `use archmage::prelude::*;` for tokens and macros. **`archmage::prelude` also drops the deprecated width-trait re-exports** `Has128BitSimd`/`Has256BitSimd`/`Has512BitSimd` (still reachable by full path; deprecated since 0.9.9). Zero downstream consumers; ships on the 0.9 chain.

## [0.9.26] - 2026-06-01

### Added

- magetypes: **AVX-512F 16-wide f16 slice path** (`token.f16_to_f32_slice` / `token.f32_to_f16_slice`). A slice is a whole-buffer op, so it summons-up to the best tier **once per call** — amortized over every lane (the cached `summon()` is ~1.3 ns); single-vector register methods (`i32x4::f16_to_f32` / `f32x4::to_f16`) never summon. A `X64V3Token` slice runs 8-wide F16C (`_mm256_cvtph_ps` / `_mm256_cvtps_ph`) and, with the `avx512` feature on a CPU that proves it, summons-up to the 16-wide AVX-512F `_mm512_cvtph_ps` / `_mm512_cvtps_ph` for the slice bulk (8/4-wide F16C tail); a V4-tier holder reaches the same 16-wide path via `token.v3()`. Verified **bit-identical to the software (and 8-wide F16C) path** over the full f16 sweep with the same benign **NaN-only** divergences (16-wide zmm `vcvtph2ps`/`vcvtps2ph` emission confirmed in `objdump`; `tests/convert_f16_exhaustive.rs` adds the `x86_avx512f` 16-lane-boundary tests). AVX-512 **FP16** (`avx512fp16`, Sapphire Rapids / Zen 5+) is intentionally **not** used — its `vcvtph2psx` / `vcvtps2phx` match AVX-512F throughput for f16↔f32 while being far less available; the FP16 ISA's value is native half-precision *arithmetic*, which a converter does not need. Public API unchanged (`cargo semver-checks` clean); purely additive. **Measured (Zen 4 / 7950X, `benchmarks/f16_convert_zen4-7950x_2026-06-01.md`):** the 16-wide path is a *modest* win over 8-wide F16C, **not 2×** — decode ≈1.2–1.5×, encode ≈1.5–1.8× when compute-bound (L1-resident), ≈parity once memory-bandwidth-bound (Zen 4 double-pumps AVX-512 on 256-bit units; native-512-bit Intel server parts should gain more, **unmeasured** here); the once-per-slice summon is amortized to negligible (9c63dc2).

### Changed

- archmage: anchor root-level paths in `include` with a leading `/` to prevent `docs/site/themes/goyo` git-submodule files from leaking into published tarballs via gitignore-style glob depth-matching (e.g. `LICENSE*` was matching `docs/site/themes/goyo/LICENSE`) (4fb9e7c).
- archmage: stop publishing the `tests/` directory in the crate tarball (727 KB / 276 files removed from the published package) (beb38dd).

## [0.9.25] - 2026-05-31

This patch release adds an f16 (IEEE-754 binary16) ↔ f32 conversion arc to
magetypes. It is purely additive — a new converter trait and hardware
backends, no breaking changes (`cargo semver-checks` clean against 0.9.24).
MSRV is unchanged at 1.89, and the AVX-512 (v4) tier remains in the
`#[autoversion]` default tier list.

### Added

- magetypes: branchless, vectorized **f16 (IEEE-754 binary16) ↔ f32** converters in `simd::generic::convert_f16` (no free functions — methods only), generic over the backend token, so NEON / WASM128 / SSE / scalar all get a straight-line SIMD path instead of an element-by-element software decode. The **register-level** conversions are **inherent methods on the value types** — `i32x4::f16_to_f32(self) -> f32x4` (decode one lane) and `f32x4::to_f16(self) -> i32x4` (encode one lane) — matching the magetypes value-method convention (`f32x4::sqrt`, `f32x4::min`, …) for value→value ops. The **slice** conversions stay **token methods on the sealed `F16Convert` trait** (re-exported at `simd::F16Convert` / `simd::generic::F16Convert`): `token.f16_to_f32_slice(in, out)` / `token.f32_to_f16_slice(in, out)` convert whole `&[u16]` ↔ `&mut [f32]` slices (inherently token-keyed — they take slices, not vector values). Decode uses Fabian Giesen's magic-multiply (`2^112` rescale, branchless Inf/NaN fixup); encode uses Giesen's `float_to_half_full_rtne` (RTNE magic-add, subnormal-flush, ±Inf saturation, NaN→canonical qNaN). Verified **bit-identical to a scalar IEEE reference exhaustively**: all 65 536 f16 decode inputs (incl. NaN bit patterns) and **all 2³² f32 encode inputs** (finite/Inf; NaN→NaN per the documented payload-may-differ contract) — `tests/convert_f16_exhaustive.rs`. The register kernels are pure safe arithmetic (no `unsafe`, no intrinsics), so correctness is arch-independent (b3862b9).
- magetypes: **native x86-64 F16C hardware backend** for the f16 slice converters. The `F16Convert` slice methods dispatch through the trait's overridable `f16_to_f32_into` / `f32_to_f16_into` methods: an `X64V3Token` (x86-64-v3, whose tier includes `f16c`) routes through the native `vcvtph2ps` / `vcvtps2ph` instructions, processing 8 lanes per AVX op; every other token keeps the branchless software kernel. AVX-512 (`X64V4Token`) holders reach the F16C path via `token.v3()`. The intrinsics are encapsulated behind `#[archmage::arcane]` boundaries (the framework's blessed `#[target_feature]` trampoline), so the module stays free of module-level `unsafe` and the public methods remain safe to call from ordinary code; memory ops use the safe `safe_unaligned_simd` re-exports (no raw pointers). The hardware path is verified **bit-identical to the software path** over all 65 536 f16 (decode) and the boundary-band + dense-sweep encode coverage, with two documented benign **NaN-only** divergences: decode hardware-quiets the 1022 f16 *signaling* NaNs (`vcvtph2ps` sets the f32 mantissa MSB; the software path widens the signaling payload — both valid f32 NaNs), and encode may emit a different (still valid) f16 NaN payload. F16C is stable since Rust 1.68. The whole f16 surface is methods (slice ops on the sealed `F16Convert` trait, register ops as inherent `i32x4`/`f32x4` methods — no free functions), and 0.9.24 shipped no f16 API at all, so `cargo semver-checks` reports **no breaking change** (purely additive) against the last published version (e25aa84).
- magetypes: **toolchain-version gate** (`rustversion` + `target_arch`) for hardware paths that use a `core::arch` intrinsic stabilized *above* the crate MSRV — replacing the trialed build-script capability-probe for the stable case. A stable intrinsic has a nameable stabilization version, so the path is selected by `#[rustversion::since(X)]` / `#[rustversion::before(X)]` inside its `#[cfg(target_arch = "…")]` scope: on rustc ≥ X the hardware kernel compiles and a runtime token decides whether to use it; on rustc < X only the branchless software fallback compiles — **MSRV unchanged (1.89), no missing-intrinsic error**. `rustversion` 1.0.22 is a dep-light, trusted proc-macro (no build script, no bespoke-probe gotchas like a cross-`core` false-negative). Both sides of the bound are exercised by the **normal** CI matrix — the MSRV-1.89 `aarch64 Linux` job `cargo check`s the `before(1.94)` software arm (below the bound) and the `test-aarch64` (native) + `test-cross` (QEMU) stable jobs compile and run the `since(1.94)` HW arm (above it). The dropped build-script probe was measured (~80 ms one-time cold compile, ~0 hot — `benchmarks/build_script_overhead_*.md`); `rustversion` was chosen over it for maintainability. **No build script ships** — the f16 arc adds zero build-time/complexity cost for downstream consumers (c6efa9f, 40bd485).
- magetypes: **native aarch64 NEON-f16 hardware backend** for the f16 slice converters, un-deferred via the version gate above. When compiled with Rust ≥ 1.94 (the `#[rustversion::since(1.94)]`-gated kernels compile) **and** the CPU presents `fp16` at runtime (the `Arm64V2Token` tier: NEON+FP16, i.e. Cortex-A55+/Apple M1+/Graviton 2+), a `NeonToken` routes the slice converters through native `vcvt_f32_f16` / `vcvt_f16_f32` (4 lanes per op); otherwise it keeps the branchless software kernel — same runtime-token dispatch the F16C path uses. **Measured on Neoverse-N1** (generic aarch64, full dispatch path incl. the `summon()` check): the HW path runs **2.6–4.7× faster than the NEON-software fallback** (and 1.6–5.2× over scalar) across 16…1 M elements — including the memory-bound 1 MP case (2.6×); the runtime dispatch is negligible (at n=16 it's within noise of the software path and well under scalar). Bench + full table: `benchmarks/f16_convert_neoverse-n1_2026-06-01.md`. The intrinsics are encapsulated behind `#[archmage::arcane]` (the `Arm64V2Token` tier enables `fp16` in the emitted `#[target_feature]` region) and memory ops use the safe `safe_unaligned_simd` re-exports, so no module-level `unsafe`. Verified **bit-identical to the software path** under QEMU (`-cpu max`): all 65 536 f16 (decode) and the boundary-band + dense-sweep encode coverage, with the same benign **NaN-only** divergences as F16C — decode hardware-quiets exactly the 1022 f16 *signaling* NaNs, encode may emit a different (valid) f16 NaN payload. On a toolchain < 1.94 the same source compiles cleanly with the software path and **no MSRV bump** (proven by `cargo +1.93 check --target aarch64-unknown-linux-gnu`). Public API unchanged; `cargo semver-checks` reports no breaking change (c6efa9f, 40bd485).

## [0.9.24] - 2026-05-26

### Added

- Windows-on-ARM detection now routes unconditionally through `winarm-cpufeatures` ≥ 0.1.2 via `Features::current_full().has(Feature::from_name(...))`. The `__winarm_cpufeatures_detected!` bridge macro folds each string-literal feature name to a `Feature` discriminant at compile time (using winarm's new `pub const fn from_name`), then bit-tests a single cached snapshot — no per-call name dispatch, and the full snapshot path includes the registry-decoded `ID_AA64*_EL1` layer. Every aarch64 token's `summon()` slow path inherits the wider coverage automatically: recovers ~30 feature names that stdarch's IPFP-only Windows backend cannot see (`fhm`, `fcma`, `sha3`, `i8mm`, `bf16`, `paca`, `bti`, `dpb`, `flagm`, `mte`, `frintts`, `sm4`, the SVE2 variants, SME, FP8, …). Sandboxed callers can disable the registry layer at runtime via `winarm_cpufeatures::set_registry_enabled(false)`. The dep is target-scoped to `cfg(all(target_os = "windows", target_arch = "aarch64"))` and the crate is internally cfg-gated to that combo, so it never resolves on any other target. No cargo feature flag required (962fceb, e13907d, 7fd6749).
- `cobalt100_runner_must_summon_full_arm64_v3` test in `arm_feature_intrinsics`. Hardware assertion that the GH `windows-11-arm` and `ubuntu-24.04-arm` runners (both Neoverse N2 / Cobalt 100) detect the full V2 + V3 token set. CI runs it via `--ignored` on those two matrix entries; without it a detection regression on Windows would silently degrade `summon()` to `None` and the existing implication tests would skip with no failure (e13907d).
- Tier lists across `incant!`, `#[autoversion]`, `#[magetypes]`, and `#[rite]` now accept plain tiers **mixed** with `+`/`-` modifiers, and `#[rite]` gains `+`/`-` support. Any `+` makes the list additive (a plain tier is treated as `+tier`); a plain list with `-` removals and no `+` overrides the defaults with the plain tiers and drops the named fallback — so `#[magetypes(v3, -scalar)]` resolves to just `v3`, replacing the brittle `[-v4, -neon, -wasm128, -scalar]` workaround (#48, e365a6b).
- `magetypes/tests/incant_chain_combinations.rs` — combinatorial coverage threading `incant!` through chains that mix every macro kind; documents that `incant!` cold-dispatch targets must be safe-wrapped (`#[arcane]`/`#[magetypes]`/plain/`#[autoversion]`-default), not `#[rite]` (67fdb89).

### Changed

- `f32` `reduce_add` now uses the adjacent-pair tree `(v0+v1)+(v2+v3)` consistently across x86 v3 (128/256-bit), NEON, WASM, scalar, and the polyfill. x86 replaces `_mm_hadd_ps`/`_mm256_hadd_ps` with `shuffle+add` (same tree, ~6–8 % faster on Zen 4: 3 µops → 2 µops). **x86 v3, NEON, and AVX-512 numeric results are unchanged; WASM, scalar, and polyfilled `f32` `reduce_add` now match x86/NEON** — wide-magnitude `f32` sums that previously diverged across those backends are now consistent. `f64` and AVX-512 512-bit reductions are untouched (f6452e4, a9ef9f2).

### Fixed

- Hardened the Windows-on-ARM bridge against silently reverting to the IPFP-only `_fast` path (which skips the registry decoder and drops the ~30 registry-classified features above). The bridge funnels through a single `registry_aware_detected()` → `Features::current_full()` entry point, guarded by a cross-platform source-scan test (`winarm_registry_path_guard`) that fails on every CI lane — not just the lone `windows-11-arm` runner — if the fast path creeps back (6ade322).

### Docs

- README: add the `_v4x` tier and the `X64V4xToken` / `Avx512Fp16Token` rows; correct the tier-mixing rules (#48 made mixing legal); replace the `_mm256_hadd_ps` horizontal-sum example; stop recommending the deprecated `_token: SimdToken` `#[autoversion]` parameter. docs/site: sweep `hadd` reductions out of the concept/dispatch examples, fix a raw-pointer load, and note that `incant!` dispatch targets must be safe-wrapped — `#[arcane]`, not `#[rite]` (78fbe95, 5dd73c8).

## 0.9.23 — 2026-04-26

### Fixed

- `#[magetypes(..., -scalar)]` (and `incant!` / `#[autoversion]` tier lists with `-scalar` or `-default`) now actually drop the fallback variant. Previously the resolver auto-appended `scalar` *after* `-scalar` had removed it, silently undoing the removal and breaking the documented "piecewise tier blocks" pattern with `the name foo_scalar is defined multiple times`. Pure additive lists and override mode are unchanged — auto-append still protects users who omit the fallback without opting out (#47, closes #46, 1eae110).
- `archmage-macros` doctest in `src/magetypes.rs` no longer fails: an indented snippet in a doc comment was being parsed as a runnable doctest referencing `::magetypes::` (a circular dep) and an unbound `Token`. Moved to a fenced `text` block (#47, 1eae110).

## 0.9.22 — 2026-04-23

### Added

- `#[magetypes(define(f32x8, u8x16, ...), ...)]` — inject `type <name> = ::magetypes::simd::generic::<name><Token>;` aliases at the top of each per-tier variant body. Eliminates the `#[allow(non_camel_case_types)] type f32x8 = GenericF32x8<Token>;` boilerplate users previously wrote at the top of every `#[magetypes]` body. `Token` is substituted per tier (`f32x8<X64V3Token>` in the v3 variant, `f32x8<ScalarToken>` in scalar, etc.) (40617c6).
- `#[magetypes(rite, ...)]` — flag that makes each per-tier variant use `#[archmage::rite(import_intrinsics)]` (direct `#[target_feature]` + `#[inline]`) instead of `#[archmage::arcane]` (safe wrapper + inner trampoline). For inner helpers called from matching-feature contexts, this eliminates the optimization boundary entirely. Not for public API dispatch — the scalar variant of an arcane-flavored magetypes falls through to the standard incant! dispatcher, which can't safely call a bare `#[target_feature]` rite variant (e735aef).
- `#[rite(scalar)]` and `#[rite(default)]` — two tokenless/featureless tiers for `#[rite]`. `scalar` takes `ScalarToken` (tokenful — fits incant!'s token-passing rewriting); `default` is tokenless. Both emit `#[inline]` with no `#[target_feature]` and no cfg-gating, callable from any context. Enables `#[rite(v3, v4, neon, wasm128, scalar)]` multi-tier as a full suffix-convention family for incant! routing (7c4e932).
- `implementation_name()` emitted on every concrete `{Type}Backend` impl, not just `X64V3Token`. Now available on `f32xN<NeonToken>`, `f32xN<Wasm128Token>`, `f32xN<ScalarToken>`, and AVX-512 variants. Names follow the `<arch>::<tier>::<type>` / `polyfill::<tier>::<type>` / `scalar::<type>` scheme documented in polyfills.md — which was aspirational before this commit and now matches the code (34793c0, closes the gap in polyfills.md:67-88).
- `magetypes/tests/expand/` — parallel macro-expansion snapshot test harness. Covers features whose emitted code references `::magetypes::simd::generic::*` (specifically `#[magetypes(define(...))]` and `#[magetypes(rite, ...)]`) — these can't be tested in archmage's `tests/expand/` because archmage has no `magetypes` dependency. Three test groups: snapshot diff, unexpanded compile, and expanded compile (32031cb).
- `magetypes/examples/idiomatic_patterns_all.rs` — runnable reference for every idiomatic magetypes + archmage pattern in one file. Self-tests A: inline `#[magetypes]`, B: extracted generic kernel, C: hand-tuned `_v4x` slotted by suffix, D: `#[autoversion]`, E: nested `incant!` rewriting, F: polyfill `implementation_name()` assertions. Passes on x86_64 (±avx512), aarch64, wasm32-wasip1 (9161d7c, updated by 40617c6).

### Changed

- `docs/site/content/magetypes/dispatch/types-and-dispatch.md` — rewritten. Led with "generic `fn<T: F32x8Backend>` + hand-written `#[arcane]` wrapper per tier + `incant!`" as the canonical pattern, which was a consistent source of user misread. `#[magetypes]` IS the per-tier `#[arcane]` wrapper generator; users don't hand-write wrappers. Also dropped the stale "X64V4Token doesn't implement F32x8Backend" limitation (it does, via delegation — see 0.9.21 changelog) (38785cc).
- `docs/site/content/archmage/concepts/arcane.md`, `CLAUDE.md`, `magetypes/README.md` — coordinated doc fixes to eliminate the `#[magetypes]` wrapper confusion across every reader entry point (e6123e4).
- `docs/site/content/archmage/dispatch/magetypes-macro.md` — added sections documenting the `rite` flag and the `define(...)` flag as they ship (e735aef, 40617c6).
- Release-prep doc updates: top-level `README.md` SIMD-types section surfaces `define()` + `rite` flags; magetypes first-types tutorial page closes with "One Body, Every Platform" section linking to types-and-dispatch (639602b).

## 0.9.21 — 2026-04-20

### Changed — narrow breaking

- Every `F32xN` / `I*xN` / `U*xN` backend trait method now takes `self` as its first parameter. Closes the UFCS path where `<X64V3Token as F32x8Backend>::splat(7.0)` could invoke a backend primitive without holding a token. The generic wrapper changes from `f32xN<T>(Repr, PhantomData<T>)` to `#[repr(C)] f32xN<T>(Repr, T)`; token storage is inline and `const _: ()` asserts keep the layout identical to `T::Repr` (#40, 7876c81). The backend traits are sealed, so the only public methods whose signatures actually changed are the five conversion helpers in the next bullet. `cargo semver-checks` reports this as a major break; the shipping surface is narrow enough that a patch was preferred over forcing every downstream off `^0.9`.
- `f32x4::from_u8`, `f32x4::load_4_rgba_u8`, `f32x8::from_u8`, `f32x8::load_8_rgba_u8`, `f32x8::load_8x8` — each gained a leading `token: T` parameter. These were the five associated functions on the generic wrappers that previously allowed constructing a SIMD value without a token (part of the UFCS gap fixed above). Callers on `^0.9` will see a compile error pointing at the missing first argument; the fix is to pass the token used elsewhere in the surrounding code (#40, 7876c81).

### Added

- Cross-width raise/lower for the f32 chain: `F32x8FromHalves` and `F32x16FromHalves` traits with `from_halves` / `low` / `high` / `split`, plus `f32x8<T>::from_halves(token, lo, hi)` and `f32x16<T>::from_halves(token, lo, hi)` generic constructors. Native AVX `vinsertf128` / AVX-512 `vinsertf32x8` / NEON + Wasm128 polyfill as appropriate. Closes the moxcms migration blocker for Double-variant interpolators (#38, 3ccf38d, closes #36).
- Backend delegation: `X64V4Token`, `X64V4xToken`, and `Avx512Fp16Token` now implement `F32x4Backend` and `F32x8Backend` by delegating to `X64V3Token` via the `.v3()` extractor. Previously those tokens only had `F32x16Backend`, so generic code on narrower widths couldn't accept a V4 token at all (#38, 3ccf38d).
- `magetypes/tests/cross_width_adversarial.rs` — 13 adversarial tests including `v4_native_matches_v3_polyfill` (AVX-512 `_mm512_insertf32x8` vs V3 polyfill bit-parity gate), lane-order checks, NaN / ±0 / ±inf round-trips, and a permutation sweep (#38, 3ccf38d).
- `magetypes/tests/bypass_closed.rs` + `magetypes/src/bypass_adversarial.rs` — adversarial soundness suite. 20 `compile_fail` doctests (construction, memory, arithmetic, math, comparison, reduction, bitwise, shift, boolean, bitcast) paired with 12 runtime-sanctioned counterparts. Uses `ScalarToken` so every target exercises the closure (#40, b635ae3).
- Compile-time layout assertions in every generic `*_impl.rs`. Build fails if a token ever gains a non-ZST field (#40, 4197620).

### Fixed

- NEON `f32x8` polyfill `recip` / `rsqrt` use two-step Newton-Raphson for precision parity with single-width NEON (#40, 6ea5448).
- `_approx` tolerance widened from 1e-3 → 4e-3 to match the ARM Architecture Reference Manual bound for `frecpe` / `frsqrte`; earlier value was tighter than the spec permits and failed under QEMU (#40, 9c48311).
- ARM and WASM polyfill UFCS trait calls thread `self` correctly through every `recip` / `rsqrt` / `rcp_approx` / `rsqrt_approx` path (#40, f2a76fd, 4e31ec9).
- Bench CI: matrix entries like `summon_overhead|archmage ...` were expanded into bash as bare pipe tokens, causing `syntax error near unexpected token '|'` on every runner. Assigning the matrix string to a single-quoted variable first makes bash treat `|` as literal (8089d5f).
- `tests/token_permutations.rs` and `tests/token_infrastructure.rs` gated on `feature = "std"` — both imported `archmage::testing`, which is std-only, so they failed to compile under `cargo test --no-default-features` in `just ci`'s full-integration-test mode. CI's own no-default matrix used `--lib` and sidestepped the issue (fdd502e).

### Changed

- CI runs magetypes integration tests on aarch64 (via `cross`) and wasm32-wasip1 (via `wasmtime`) targets, not just x86 (#39, 9d34c81).
- Moved magetypes-dependent benches (`asm_patterns`, `cbrt_variants`, `generic_vs_concrete`, `safe_memory_overhead`) from archmage to magetypes. The bench workflow was failing because the magetypes dev-dep was removed from archmage in 0.9.18 (83519ca) but these benches were missed (8c8c9e5).
- `magetypes` allows `clippy::wrong_self_convention` crate-wide. Backend trait methods thread the CPU-feature token through `self` on every method (including `from_*` / `to_*`), which the lint's constructor heuristic doesn't apply to (4fd8f02).

## 0.9.20 — 2026-04-15

### Fixed

- `#[autoversion]` and `incant!` no longer emit `unused_imports` warnings on archs where every dispatch arm is cfg'd out (notably 32-bit x86 with `[v3, neon, wasm128]`-style tier lists). Downstream crates no longer need to sprinkle `#[cfg_attr(target_arch = "x86", allow(unused_imports))]` on every call site (#34, cae6284)

## 0.9.19 — 2026-04-14

### Added

- `w512` cargo feature for magetypes gating 512-bit SIMD types (`f32x16`, `f64x8`, `i*x64`, `u*x64`, etc.). Default-on for backwards compatibility. Users who only need W128/W256 can disable default features and skip `w512` for ~25% faster builds. `avx512` implies `w512`. (75a32d6)

### Changed

- Token tier-tag assertions use per-token `__ARCHMAGE_TIER_TAG` constants instead of `::archmage::` path checks, enabling re-exporters to use `#[arcane]` without requiring downstream crates to depend on archmage directly (6b34824)

### Fixed

- `#[magetypes]` now propagates doc comments, `#[allow]`, and other attributes to all generated variants — previously stripped them, causing `missing_docs` warnings (#32, f2f8b94)
- Token aliasing compile errors now show `_ARCHMAGE_TOKEN_MISMATCH` in the error instead of anonymous `_`, making the failure immediately diagnosable (ba5ef4d)

## 0.9.16 — 2026-04-01

### Scalar rounding now matches hardware (ties-to-even)

The scalar backend's `round()` and `to_i32_round()` now use IEEE 754 round-to-nearest-even, matching the behavior of SSE `cvtps2dq`, AVX2 `vcvtps2dq`, NEON `vcvtnq_s32_f32`, and WASM `f32x4.nearest`. Previously, the scalar fallback used ties-away-from-zero (`f32::round()` semantics), causing dispatch parity failures — e.g., a 47-byte divergence in zenjpeg's encoder when comparing scalar vs SIMD output.

New `nostd_math::roundevenf` and `nostd_math::roundeven` functions are available for `no_std` code that needs IEEE 754 default rounding.

### Token disable tests no longer flaky

Tests that call `dangerously_disable_token_process_wide()` now hold `lock_token_testing()`, preventing races when tests run in parallel.

## 0.9.11 — 2026-03-24

### `#[autoversion]` — tokenless mode + ScalarToken nesting

`#[autoversion]` no longer requires a `SimdToken` parameter. Write plain functions:

```rust
#[autoversion]
fn sum(data: &[f32]) -> f32 { data.iter().sum() }

let result = sum(&data);
```

Three token forms:

| Parameter | Dispatcher signature | Use case |
|-----------|---------------------|----------|
| *(none)* | Token-free | **Recommended** for new code |
| `_: ScalarToken` | Keeps ScalarToken | **incant! nesting** — no bridge needed |
| `_: SimdToken` | Token stripped | **Deprecated** — use tokenless or ScalarToken |

**ScalarToken nesting** — hand-written SIMD + autoversioned fallback, zero boilerplate:

```rust
fn process(data: &[f32]) -> f32 {
    incant!(process(data), [v4, scalar])
}

#[arcane(import_intrinsics)]
fn process_v4(_: X64V4Token, data: &[f32]) -> f32 { /* AVX-512 intrinsics */ }

#[autoversion(v3, neon)]
fn process_scalar(_: ScalarToken, data: &[f32]) -> f32 {
    data.iter().sum()  // auto-vectorized, dispatches v3/neon/scalar internally
}
```

`process_scalar` IS the autoversion dispatcher. `incant!` calls it with ScalarToken — signature matches directly.

### Deprecations

- **`SimdToken` in `#[autoversion]`**: Emits deprecation warning. `SimdToken` is a trait, not a type — it can't appear in compiled signatures. Use tokenless or `ScalarToken`.

### Errors

- **Concrete tokens in `#[autoversion]`** (`X64V3Token`, `NeonToken`, etc.): Now produces a clear compile error directing users to `#[arcane]` or `#[rite]` for single-token functions.

### `default` tier — tokenless fallback

New `default` tier for `incant!`, `#[autoversion]`, and `#[magetypes]`. Like `scalar` but calls `_default(args)` without any token:

```rust
fn process(data: &[f32]) -> f32 {
    incant!(process(data), [v4, default])
}

#[arcane(import_intrinsics)]
fn process_v4(_: X64V4Token, data: &[f32]) -> f32 { /* intrinsics */ }

#[autoversion(v3, neon)]
fn process_default(data: &[f32]) -> f32 {
    data.iter().sum()  // tokenless — no ScalarToken, no bridge
}
```

`scalar` and `default` are mutually exclusive. If neither is listed, `scalar` is auto-appended for backwards compatibility.

### Docs

Comprehensive autoversion docs: name collision patterns, incant! nesting (bridgeless via `default` and via `ScalarToken`), feature-gated tiers, const generics, method patterns, when-to-use comparison.

## 0.9.10 — 2026-03-24

### Deprecations

- **Width traits deprecated**: `Has128BitSimd`, `Has256BitSimd`, `Has512BitSimd` now emit `#[deprecated]` warnings. `Has256BitSimd` is actively misleading (enables AVX, not AVX2). Use concrete tokens (`X64V3Token`) or tier traits (`HasX64V2`, `HasX64V4`). Will be removed in v1.0.

- **`Avx2FmaToken` alias deprecated**: Misleading name — V3 includes BMI1/2, F16C, and more, not just AVX2+FMA. Use `X64V3Token` or `Desktop64`.

- **Missing `scalar` in explicit `incant!` tier lists**: Now emits a deprecation warning. `incant!` always calls `fn_scalar()` as the final fallback — not listing it hides this requirement. Will become a compile error in v1.0.

### Infrastructure

- Trait deprecation driven from `token-registry.toml` via `deprecated` field.
- Token alias deprecation via `deprecated_aliases` map in `token-registry.toml`.
- Generator handles `#[allow(deprecated)]` on all internal references.

## 0.9.9 — 2026-03-24

### Backwards compat fix for explicit tier lists

`incant!` with explicit `[v4, v3, neon]` tier lists now always auto-applies the `(avx512)` feature gate to v4/v4x — matching 0.9.5 behavior. 0.9.8 only applied this for default tier lists, breaking zenresize and zensim whose published code uses `[v4, v3]` with cfg-gated `_v4` functions.

### CI

Added downstream compat tests for zensim and zenpixels-convert (both depend on linear-srgb which uses the `[v4, v3, neon]` pattern).

## 0.9.8 — 2026-03-24

### Feature-gated tiers: `tier(feature)` syntax

New syntax for conditionally dispatching to tiers based on the calling crate's cargo features. Works across all dispatch macros:

```rust
// incant! — dispatch gated on calling crate's "avx512" feature
incant!(foo(x), [v4(avx512), v3, neon, scalar])

// #[arcane] — combined arch + feature cfg guard (replaces manual #[cfg(all(...))])
#[arcane(import_intrinsics, cfg(avx512))]
fn process_v4(_token: X64V4Token, data: &mut [f32]) { ... }

// #[rite] — same for single and multi-tier
#[rite(v4, import_intrinsics, cfg(avx512))]
fn helper() { ... }

// #[autoversion] — full dispatch when feature on, scalar-only when off
#[autoversion(cfg(simd))]
fn process(_token: SimdToken, data: &[f32]) -> f32 { ... }
```

- `tier(feature)` wraps dispatch in `#[cfg(feature = "feature")]` — checks the *calling crate's* features, not archmage's. No `#[allow(unexpected_cfgs)]` emitted (the user explicitly declared the feature).
- **Default tiers** auto-apply `(avx512)` to v4/v4x with `#[allow(unexpected_cfgs)]` — graceful degradation for crates that don't define an avx512 feature.
- **`#[arcane(cfg(feat))]`** generates `#[cfg(all(target_arch = "...", feature = "feat"))]` — replaces the manual `#[cfg(all(target_arch = "x86_64", feature = "avx512"))]` pattern.
- **`#[autoversion(cfg(feat))]`** emits two dispatchers: full dispatch under `#[cfg(feature)]`, scalar-only under `#[cfg(not(feature))]`.

### `#[autoversion]` macro_rules! hygiene fix

`#[autoversion]` now uses `return` instead of `break '__dispatch` in the generated dispatcher. This fixes a label hygiene issue where `#[autoversion]` applied inside `macro_rules!` would fail with "undeclared label `'__dispatch`". The labeled block's span was in the proc macro context while the function body was in the `macro_rules!` context. `return` has no hygiene issues since it's a keyword.

`incant!` still uses labeled blocks (it's used as an expression where `return` would exit the enclosing function).

## 0.9.7 — 2026-03-24

### Backwards compatibility fix

- **`incant!`/`#[magetypes]` with explicit `[v4, ...]` tier lists** — AVX-512 tiers are now silently skipped when the `avx512` feature is off, even in explicit tier lists. This matches the old behavior where v4 dispatch arms were cfg-gated, but via the correct mechanism (expansion-time check on archmage-macros, not `#[cfg(feature)]` in output). Crates like linear-srgb that use `incant!(foo(x), [v4, v3, neon])` with cfg-gated `_v4` functions now work without changes.

## 0.9.6 — 2026-03-24

### Bug fixes

- **`i16x16`/`u16x16` bitmask correctness** — `bitmask()` was returning incorrect results on x86_64 AVX2: lanes 8-15 were always zero. Root cause: `_mm256_packs_epi16(shifted, shifted)` interleaves within 128-bit lanes, producing wrong lane ordering. Fix: extract 128-bit halves first, then use `_mm_packs_epi16(lo, hi)` for correct order. Fixed in both the raw W256 types and the generic backend implementations. ([#16])

- **`#[arcane]` lint attribute propagation** — `#[allow(clippy::too_many_arguments)]` and similar lint-control attributes (`#[expect]`, `#[deny]`, `#[warn]`, `#[forbid]`) now propagate to the generated dispatch wrapper in both sibling mode (default) and nested mode. Previously, clippy would lint the generated code even when the user explicitly suppressed the warning. ([#17])

- **`#[autoversion]` v4/v4x variants no longer require `avx512` feature** — `#[autoversion]` generates scalar code compiled with `#[target_feature]`, so the `avx512` cargo feature was never needed. Previously, v4/v4x variants were silently eliminated by `#[cfg(feature = "avx512")]` in macro output — which checked the *calling crate's* features (always wrong for downstream crates). Now v4/v4x variants are always generated.

### avx512 feature gating overhaul

The `avx512` cargo feature handling was redesigned. The old approach emitted `#[cfg(feature = "avx512")]` in proc-macro output, which checked the calling crate's features instead of archmage's — always wrong for downstream crates, and triggering `unexpected_cfgs` warnings on modern rustc.

**New behavior:**

- **`avx512` feature propagated to `archmage-macros`** — macros check `cfg!(feature = "avx512")` at expansion time on their own crate, not via `#[cfg]` in output.
- **`#[autoversion]`** — always generates v4/v4x (scalar code, no safe memory ops needed).
- **`incant!`/`#[magetypes]` default tiers** — include v4 only when `avx512` is enabled. Without it, dispatch gracefully skips v4.
- **`#[arcane(import_intrinsics)]`/`#[rite(import_intrinsics)]` with AVX-512 tokens** — clear `compile_error!` when `avx512` is not enabled, telling users exactly what to add to `Cargo.toml`. Works for all token spellings: concrete (`X64V4Token`, `Avx512Token`, `Server64`), trait bounds (`impl HasX64V4`), and generics (`T: HasX64V4`).
- **`#[arcane]`/`#[rite]` without `import_intrinsics`** — always works with any token. Value intrinsics don't need the cargo feature.
- **No `#[cfg(feature = "...")]` ever emitted in macro output** — eliminates `unexpected_cfgs` warnings entirely.

### Testing

- 192 bitmask correctness tests covering all 24 integer SIMD types (W128/W256/W512).
- No-features integration test crate (`archmage-no-features-test`) with `#![deny(warnings)]`.
- 5 standalone avx512-cfg-test crates verifying every feature gating scenario.
- Token infrastructure tests now serialized in CI to prevent races from process-wide state mutations.

[#16]: https://github.com/imazen/archmage/issues/16
[#17]: https://github.com/imazen/archmage/issues/17

## 0.9.5 — 2026-03-09

### Transcendental accuracy improvements

- **`exp2_midp`: floor → round-to-nearest split** — Splitting the input into integer and fractional parts now uses round-to-nearest instead of floor, keeping |frac| ≤ 0.5 instead of [0, 1). This eliminates the accuracy hot spot near integer boundaries where the polynomial was evaluating at frac ≈ 1.0. The integer part is clamped to 127 to prevent the `(n+127)<<23` bit trick from overflowing. Accuracy is now uniform across all input regions (1 ULP for evenly-spaced inputs, 63 ULP worst case overall). Applied on all platforms (x86, ARM, WASM, generic).

- **`exp2_midp` overflow threshold**: changed from `> 128` to `>= 128`. Since 2^128 > f32::MAX, `exp2_midp(128.0)` now correctly returns inf instead of ~3.4e38.

- **`exp2_midp` underflow limit**: tightened from -150 to -126. Inputs in [-150, -126] were producing garbage via the bit trick (which can't construct denormal floats). Now returns 0 for all inputs below -126.

- **`pow_lowp(0, n)` for positive n**: now returns 0 instead of NaN. The zero-mask was being applied before the exp2 computation instead of after.

- **`cbrt` zero handling**: `cbrt(±0)` now returns ±0 (preserving sign) instead of producing small nonzero values from the bit-hack initial guess.

- **Accuracy improvements** (max ULP vs std, measured on x86-64):
  - `exp2_midp`: 132 → 63 ULP
  - `exp_midp`: 256+ → 58 ULP
  - `pow_midp` (n=0.5): 149 → 9 ULP
  - `pow_midp` (n=2): ~130 → 16 ULP
  - `pow_midp` (n=3): ~150 → 55 ULP

- **No performance impact**: round instruction costs the same as floor (same opcode, different rounding mode bit). The added `min(xi, 127)` is one extra SIMD min instruction.

## 0.9.4 — 2026-03-08

Multi-tier `#[rite]`, `#[inline(always)]` wrappers, improved cbrt, docs overhaul.

### `#[rite]` multi-tier support

`#[rite]` now supports three modes:

- **Token-based** (`#[rite]`): reads the token type from the function signature (existing behavior)
- **Tier-based** (`#[rite(v3)]`): specifies features via tier name, no token parameter needed (existing behavior)
- **Multi-tier** (`#[rite(v3, v4, neon)]`): generates a suffixed variant for each tier (`fn_v3`, `fn_v4`, `fn_neon`), each with its own `#[target_feature]` and `#[cfg(target_arch)]`

Multi-tier lets you write one function body and get per-tier compiled variants — like `#[autoversion]` but for internal SIMD functions (no dispatcher generated). Each variant is safe to call from matching `#[arcane]` or `#[rite]` contexts (Rust 1.86+). Single-tier behavior is unchanged — no suffix is added.

```rust
#[rite(v3, v4, neon)]
fn scale(data: &[f32; 4], factor: f32) -> [f32; 4] {
    [data[0] * factor, data[1] * factor, data[2] * factor, data[3] * factor]
}
// Generates: scale_v3(), scale_v4(), scale_neon()
```

### `#[inline(always)]` on `#[arcane]` wrappers

`#[arcane]` now generates `#[inline(always)]` on the safe wrapper function. Previously the wrapper had no inline hint, which could prevent LLVM from inlining the dispatch trampoline. If you had `#[inline(always)]` on the function yourself, the macro strips it to avoid the duplicate-attribute warning (which Rust is phasing into a hard error).

### `incant!` explicit tiers: `scalar` recommended

When using explicit tier lists with `incant!`, always include `scalar`: `incant!(sum(data), [v3, neon, scalar])`. This documents the mandatory fallback path. Currently `scalar` is auto-appended if omitted for backwards compatibility; this will become a compile error in v1.0.

### Improved `cbrt` (cube root)

- `cbrt_midp` now uses 2-iteration Halley refinement (was Newton-Raphson). ~2 ULP max error across the full f32 range, down from ~4 ULP.
- `cbrt_lowp` uses 1-iteration Halley. ~22 ULP max error, faster than midp.
- Added `cbrt_lowp`/`cbrt_midp` with `_unchecked` and `_precise` variants.
- All variants handle negative inputs, zero, NaN, and infinity correctly.
- Added scalar `ScalarToken` implementations for all cbrt variants.

### `macros` feature is now always-on

The `macros` cargo feature is now a no-op — macros (`#[arcane]`, `#[rite]`, `incant!`, etc.) are always available. The feature flag still exists so `features = ["macros"]` doesn't break existing code.

### Documentation overhaul

- **README**: safety model diagram showing Rust's `#[target_feature]` call rules and how archmage makes dispatch sound. Macro selection flowchart (`#[arcane]` vs `#[rite]` vs `#[autoversion]` vs `incant!`). Tier naming conventions table. Both `#[rite]` syntaxes with code examples. Expanded testing section with `testable_dispatch`, `CompileTimePolicy`, and `lock_token_testing()`.
- **All docs**: updated `incant!` examples to include `scalar` in explicit tier lists.

### Other fixes

- Use `f64::clamp()` instead of manual min/max pattern.
- User `#[inline(always)]` on `#[arcane]`/`#[rite]` functions no longer causes duplicate attribute warnings.
- `#[rite]` strips user `#[inline]` attributes to avoid conflicts with its own `#[inline]`.

## 0.9.3 — 2026-03-05

Fixed `no_std` compilation on bare-metal targets, added `no_std` CI enforcement.

- **Fixed `no_std` on aarch64/WASM bare metal** — ARM and WASM `f64x2` transcendentals (`log2_lowp`, `exp2_lowp`, `ln_lowp`, `exp_lowp`, `log10_lowp`, `pow_lowp`) used `f64` inherent methods (`.log2()`, `.exp2()`, etc.) that only exist with `std`. Added scalar polynomial approximations to `nostd_math` using the same coefficients as the x86 SIMD implementations.

- **Mandatory `no_std` CI checks** — CI now auto-installs and compiles against `aarch64-unknown-none` and `thumbv7m-none-eabi` targets. Host-target `--no-default-features` checks don't catch `std` leaks because libstd is always linkable on the host; cross-target checks are required to catch them.

- **`just test-nostd`** — new justfile target runs `no_std` compilation checks and tests for all crates.

- **Bitmask tests handle missing runtime detection** — tests now skip gracefully when `summon()` returns `None` (happens under `no_std` without `-Ctarget-cpu`) instead of panicking.

## 0.9.2 — 2026-03-05

Const generic support for `#[autoversion]` and `#[arcane]`, semver-checks CI.

- **Const generic support** — `#[autoversion]`, `#[arcane]` (sibling mode), and `#[arcane]` (nested mode) now forward const generic parameters via turbofish in generated dispatch/wrapper calls. Previously, const generics that couldn't be inferred from argument types alone (e.g., `<const BPP: usize>` used only in the function body) caused `E0282: type annotations needed`. This matches `multiversion`'s `#[multiversed]` behavior.

  ```rust
  // Now works — CHUNK forwarded via turbofish in dispatcher
  #[autoversion]
  fn fill_row<const BPP: usize>(_token: SimdToken, data: &[u8]) { ... }

  fill_row::<3>(&data); // Dispatcher calls fill_row_v3::<3>(...)
  ```

- **Semver-checks CI** — new `semver-checks.yml` workflow runs `cargo-semver-checks` on every PR for all three crates, catching accidental breaking changes before merge.

- **22 new const generic tests** — covers `#[autoversion]` and `#[arcane]` with: basic const generics, body-only const generics, return-type-only, multiple const generics, mixed type+const generics, lifetimes, self receivers, `_self = Type` nested mode, explicit tiers, and direct variant calls with turbofish.

## 0.9.1 — 2026-03-05

Generic `f32x16<T>` transcendentals, bitmask bug fix, `#[autoversion]` improvements.

- **Fixed i16x16/u16x16 bitmask** — `_mm256_packs_epi16` lane interleaving caused lanes 8-15 to be dropped on x86_64. The fix uses per-lane `_mm256_extract_epi16` + manual bit assembly, matching the pattern used for other element sizes.

- **`#[autoversion]` self receiver fix** — inherent methods with `self`/`&self`/`&mut self` now work without `_self = Type`. The `_self` parameter is only needed for trait impl delegation (nested mode). Previously, `#[autoversion]` incorrectly required `_self` for all self receivers.

- **Generated functions are private** — `#[autoversion]` variants and `#[arcane]` sibling functions no longer inherit the user's visibility. Only the dispatcher (for `#[autoversion]`) or the safe wrapper (for `#[arcane]`) gets the original visibility. This prevents leaking internal implementation functions.

- **`#[autoversion]` reference page** — comprehensive documentation at `docs/site/content/archmage/dispatch/autoversion.md` covering all parameters, tier tables, dispatch flow, and usage patterns.

- **`F32x16Convert` trait** — new backend trait enabling bitcast and numeric conversion between `f32x16` and `i32x16`. Implemented for all backends: X64V3Token (2×256-bit polyfill), X64V4Token/X64V4xToken (native AVX-512), NeonToken (4×128-bit polyfill), Wasm128Token (4×128-bit polyfill), ScalarToken.

- **Generic `f32x16<T>` transcendentals** — `pow_midp`, `log2`, `exp2`, `ln`, `exp`, `log10`, `cbrt` (all with `_lowp`/`_midp`, `_unchecked`, `_precise` variants). Same polynomial approximations as `f32x4<T>` and `f32x8<T>`, works on any backend that implements `F32x16Convert`.

- **`f32x16<T>` ↔ `i32x16<T>` conversion methods** — `bitcast_to_i32`, `from_i32_bitcast`, `to_i32`, `to_i32_round`, `from_i32` on `f32x16<T>`; `bitcast_to_f32`, `to_f32` on `i32x16<T>`.

- **Comprehensive bitmask tests** — correctness tests for all 24 generic SIMD integer types (W128/W256/W512, signed/unsigned, 8/16/32/64-bit) covering individual lanes, cross-boundary patterns, all-set, and all-clear.

- **30 `#[autoversion]` integration tests** — plain self receivers, owned self, explicit tiers, wildcards, tuple/Option returns, in-place mutation, scalar/v3 variant direct calls.

- **34 f32x16 tests** covering transcendentals, conversions, edge cases, roundtrips, cross-backend consistency, and generic function usage.

## 0.9.0 — 2026-03-04

Sibling expansion, cfg-out default, macro options, `import_intrinsics`.

**BREAKING:** `#[arcane]` and `#[rite]` now cfg-out functions on non-matching architectures by default (no unreachable stub). Code referencing these functions on wrong platforms without `#[cfg]` guards will fail to compile. **Migration:** Add `#[arcane(stub)]` / `#[rite(stub)]` to restore old behavior, or use `#[cfg(target_arch)]` guards, or use `incant!` (unaffected — already cfg-gates dispatch calls).

**BREAKING:** The `safe_unaligned_simd` cargo feature is gone — the dependency is now always included. `import_intrinsics` (new in 0.9) generates combined intrinsics modules that re-export `safe_unaligned_simd`'s reference-based memory ops alongside `core::arch`, so the dependency is no longer optional. A no-op `safe_unaligned_simd` feature flag exists for backwards compatibility — `features = ["safe_unaligned_simd"]` won't break, it just does nothing. If you were using `default-features = false` to exclude it, note that the `safe_unaligned_simd` crate is now always pulled in (it's small and `no_std`).

- **Sibling expansion (default)** — `#[arcane]` now generates a sibling `__arcane_fn` with `#[target_feature]` at the same scope, plus a safe wrapper. Both functions share the `impl` block, so `self`, `Self`, and associated constants work naturally in methods. No more `_self` boilerplate for inherent methods.

- **Nested expansion (opt-in)** — `#[arcane(nested)]` uses the old inner-function approach. `#[arcane(_self = Type)]` implies nested. **Required for trait impls** — sibling expansion adds `__arcane_fn` which isn't in the trait definition.

- **Cfg-out default** — On wrong architectures, `#[arcane]` and `#[rite]` emit no code at all. Less dead code. `incant!` is unaffected (already cfg-gates each tier).

- **`stub` param** — `#[arcane(stub)]` and `#[rite(stub)]` generate `unreachable!()` stubs on wrong architectures, restoring previous cross-platform behavior for dispatch patterns that reference functions without `#[cfg]` guards.

- **LightFn (internal)** — Proc macro now parses only the function signature into AST, leaving the body as opaque `TokenStream`. Saves ~2ms per function. Token-level `Self`/`self` replacement instead of syn `Fold`.

- **`import_intrinsics` parameter** — `#[arcane(import_intrinsics)]` and `#[rite(import_intrinsics)]` inject `use archmage::intrinsics::{arch}::*` into the function body. This brings all `core::arch` types and value intrinsics into scope alongside [`safe_unaligned_simd`](https://crates.io/crates/safe_unaligned_simd)'s reference-based memory ops. Rust's name resolution makes explicit re-exports shadow glob imports, so `_mm256_loadu_ps` resolves to the safe version (takes `&[f32; 8]`) automatically. Combined with Rust 1.87+ making value intrinsics safe inside `#[target_feature]`, this means zero `unsafe` in your SIMD code — `#![forbid(unsafe_code)]` compatible.

- **Combined intrinsics modules** — `archmage::intrinsics::{x86_64, x86, aarch64, wasm32}` glob-import `core::arch` and explicitly re-export every `safe_unaligned_simd` function. Auto-generated by `cargo xtask generate`. These modules are what `import_intrinsics` injects.

- **`#[autoversion]`** — new single-attribute macro that generates architecture-specific function variants *and* a runtime dispatcher from one annotated function. Write a plain scalar loop with a `SimdToken` placeholder parameter; `#[autoversion]` clones it per tier (v4, v3, neon, wasm128, scalar by default), replaces `SimdToken` with each concrete token type in the signature, wraps non-scalar variants in `#[arcane]` for `#[target_feature]`, and emits a dispatcher function (same name, `SimdToken` param removed) that calls the best variant at runtime via `Token::summon()`.

  - **Signature-only replacement** — only the `SimdToken` type annotation in the parameter list is swapped. The function body is never reparsed (uses `LightFn`'s opaque body), keeping compile times low. Compare with `#[magetypes]` which does full text substitution including the body.

  - **Explicit tiers** — `#[autoversion(v3, v4, v4x, neon, arm_v2, wasm128)]`. `scalar` is always appended implicitly. Unknown tier names produce a compile error. Tiers are sorted by dispatch priority automatically.

  - **Self receivers** — inherent methods with `self`/`&self`/`&mut self` work naturally (fixed in 0.9.1 — originally required `_self = Type`). For trait impl delegation, use `#[autoversion(_self = Type)]` with `_self` in the body.

  - **Trait method delegation** — trait impl methods can't expand to multiple siblings, so `#[autoversion]` can't be used directly on them. Documented delegation pattern: trait method calls an autoversioned inherent method.

  - **Generated variants are private** — individually callable within the module, `incant!`-compatible, with `#[cfg(target_arch)]` and `#[cfg(feature)]` guards matching each tier. Only the dispatcher inherits the user's visibility.

  - **74 unit tests** — argument parsing, `SimdToken` parameter discovery, tier resolution, AST replacement for all known tiers, dispatcher parameter removal and wildcard renaming, tier descriptor properties, suffix_path.

- **MSRV 1.89** — required for stabilized target features and intrinsics. On x86, Rust 1.89 stabilizes `avx512fp16`, `sm3`, `sm4`, `kl`, and `widekl` target features, plus additional AVX-512 intrinsics and target features. These are needed for archmage's token-to-feature mappings and `#[target_feature]` attributes emitted by `#[arcane]`.

## 0.8.3 — 2026-02-19

Complete `X64CryptoToken` integration with `incant!` dispatch.

- **`x64_crypto` tier in `incant!`** — dispatch to `X64CryptoToken` via explicit tier lists: `incant!(func(data), [v4x, x64_crypto, arm_v2, neon_aes])`. Priority 25 (between v3=30 and v2=20), so crypto is tried before plain v2.

- **`IntoConcreteToken::as_x64_crypto()`** — safe downcasting to `X64CryptoToken` for passthrough dispatch.

- **Prelude** — `X64CryptoToken` now re-exported from `archmage::prelude::*`.

## 0.8.2 — 2026-02-19

New `X64CryptoToken` for PCLMULQDQ + AES-NI.

- **`X64CryptoToken`** — new leaf token off `X64V2Token` providing `pclmulqdq` and `aes` features. PCLMULQDQ and AES-NI are not part of the psABI v2 spec (Nehalem 2008 and some VMs lack them), so they belong in a dedicated token rather than in `X64V2Token`. Available on Westmere (2010)+, Bulldozer+, Silvermont+, all Zen. Use for CRC-32 folding, AES encryption, and GF(2) polynomial arithmetic. Dispatch tier name: `x64_crypto`.

- **Reverts 0.8.1** — removed `pclmulqdq` and `aes` from `X64V2Token` and all higher tokens (V3, V4, V4x, FP16). V2 now matches the psABI spec exactly.

## 0.8.1 — 2026-02-18 [YANKED]

Incorrectly added PCLMULQDQ/AES-NI to V2 baseline. These are not in the psABI v2 spec — Nehalem (2008) and QEMU's x86-64-v2 CPU model lack them. Use 0.8.2's `X64CryptoToken` instead.

## 0.8.0 — 2026-02-18

ARM compute tiers, better macro diagnostics, edition 2024.

- **`Arm64V2Token` and `Arm64V3Token`** — new compute-tier tokens for AArch64, analogous to x86's V2/V3/V4 hierarchy. V2 covers Cortex-A55+, Apple M1+, Graviton 2+ (adds CRC, RDM, DotProd, FP16, AES, SHA2 over baseline NEON). V3 covers Cortex-A510+, Apple M2+, Snapdragon X, Graviton 3+ (adds FHM, FCMA, SHA3, I8MM, BF16). Both tokens work with `#[arcane]`, `#[rite]`, `incant!`, and `#[magetypes]`.

- **`HasArm64V2` and `HasArm64V3` traits** — tier traits for generic bounds over the new ARM compute tiers, matching the pattern of `HasX64V2`/`HasX64V4`.

- **`incant!` tiers `arm_v2` and `arm_v3`** — dispatch to ARM compute tiers in explicit tier lists: `incant!(process(data), [v3, arm_v2, arm_v3, neon])`.

- **`IntoConcreteToken` gains `as_arm_v2()` and `as_arm_v3()`** — safe downcasting to the new ARM tokens.

- **Featureless trait rejection** — `#[arcane]` and `#[rite]` now give a specific error when you use `SimdToken` or `IntoConcreteToken` as a token bound. These traits carry no CPU features, so the macros can't emit `#[target_feature]`. The error message explains why and suggests concrete tokens or feature traits like `HasX64V2`.

- **`detect_features` example** — new example (`cargo run --example detect_features`) that prints all detected SIMD capabilities on the current CPU.

- **Edition 2024** — `archmage-macros` upgraded from Rust edition 2021 to 2024 (requires rustc 1.89+). `archmage` and `magetypes` were already on edition 2024.

## 0.7.1 — 2026-02-14

Docs, warnings, and magetypes 0.7.0.

- **Documentation: token type is the feature selector** — all docs (lib.rs, README, PERFORMANCE.md, spec.md, magetypes README) now explain that `#[arcane]` and `#[rite]` parse the token type from your function signature to determine which `#[target_feature]` to emit. Passing the same token through your call hierarchy keeps features consistent; mismatched types create optimization boundaries.

- **Documentation: `#[arcane]` wrapper vs `#[rite]` direct** — clarified that `#[arcane]` generates a wrapper function to cross the `#[target_feature]` boundary without `unsafe` at the call site, but that wrapper *is* the optimization boundary. `#[rite]` applies `#[target_feature]` + `#[inline]` directly with no wrapper. `#[rite]` should be the default; `#[arcane]` only at entry points.

- **Zero compiler warnings** — fixed all warnings across xtask (14), tests, and examples. Removed unused imports, unnecessary `unsafe` blocks (safe since Rust 1.87), and minor clippy lints.

- **Fixed rustdoc warning** — escaped `#[arcane]` doc link in X64V1Token.

- **`magetypes` 0.7.0** — version aligned with archmage 0.7.0 dependency.

## 0.7.0 — 2026-02-13

New token, explicit dispatch control, and docs refresh.

- **`X64V1Token` / `Sse2Token`** — baseline x86_64 token covering SSE + SSE2. Rust 1.87+ made intrinsics safe inside `#[target_feature]` functions, but that means even `_mm_add_ps` requires a `#[target_feature(enable = "sse2")]` context. Without a token to enter that context, `#![forbid(unsafe_code)]` crates couldn't call baseline SIMD intrinsics at all. `X64V1Token::summon()` succeeds on every x86_64 CPU (SSE2 is mandatory for the architecture), so it compiles down to nothing — but it gives you the `#[target_feature]` gate you need.

- **Explicit tier lists for `incant!`** — control which dispatch tiers are attempted:

  ```rust
  // Only dispatch to V1, V3, and NEON (plus implicit scalar fallback)
  pub fn sum(data: &[f32]) -> f32 {
      incant!(sum(data), [v1, v3, neon])
  }
  ```

  Without a tier list, `incant!` tries all tiers and you need `_v2`, `_v3`, `_v4`, `_neon`, `_wasm128`, and `_scalar` variants. With a tier list, you only write the variants you care about. Scalar is always implicit.

- **Explicit tier lists for `#[magetypes]`** — same idea, applied to code generation:

  ```rust
  #[magetypes([v3, neon])]
  fn process(token: Token, data: &[f32]) -> f32 { ... }
  // Generates: process_v3, process_neon, process_scalar
  ```

- **`testable_dispatch` feature** — renamed from `disable_compile_time_tokens`. The old name described the mechanism; the new name says what it's for. Enable it in dev-dependencies so `for_each_token_permutation()` and `dangerously_disable_token_process_wide()` work even when compiled with `-Ctarget-cpu=native`.

- **Documentation refresh** — updated safety model docs, token reference, and README to cover V1 token, tier lists, and the `dangerously_disable_tokens_except_wasm` API.

## 0.6.1 — 2026-02-12

- **`archmage::testing` module** — `for_each_token_permutation()` runs a closure for every unique combination of SIMD tokens disabled, testing all dispatch fallback tiers on native hardware. Handles cascade hierarchy, mutex serialization, panic-safe re-enable, and deduplication of equivalent effective states. On an AVX-512 machine this produces 5–7 permutations; on Haswell-era, 3.

  ```rust
  use archmage::testing::{for_each_token_permutation, CompileTimePolicy};

  #[test]
  fn dispatch_works_at_all_tiers() {
      let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
          let result = sum_squares(&data);
          assert!((result - expected).abs() < 1e-1, "failed at: {perm}");
      });
      assert!(report.permutations_run >= 2);
  }
  ```

- **`CompileTimePolicy` enum** — `Warn` (silent, collect in report), `WarnStderr` (also prints), `Fail` (panics with exact compiler flags to fix). Wire an env var for CI enforcement.

## 0.6.0 — 2026-02-12

Cross-platform hardening, testability, and CI infrastructure.

- **Test every dispatch path without cross-compilation.** Disable individual tokens or kill all SIMD at once to force your code through scalar and lower-tier fallbacks — on your native hardware, in your existing test suite:

  ```rust
  use archmage::{X64V3Token, SimdToken, dangerously_disable_tokens_except_wasm};

  #[test]
  fn scalar_fallback_produces_same_results() {
      let result_simd = my_function(&data);

      // Kill V3 (AVX2+FMA) — summon() now returns None
      X64V3Token::dangerously_disable_token_process_wide(true).unwrap();
      let result_scalar = my_function(&data);
      X64V3Token::dangerously_disable_token_process_wide(false).unwrap();

      assert_eq!(result_simd, result_scalar);
  }

  #[test]
  fn everything_works_without_simd() {
      dangerously_disable_tokens_except_wasm(true).unwrap();
      // entire test runs through scalar fallbacks
      run_full_integration_test();
      dangerously_disable_tokens_except_wasm(false).unwrap();
  }
  ```

  Disabling V2 cascades to V3/V4/Modern/Fp16. Disabling NEON cascades to Aes/Sha3/Crc. Per-token `manually_disabled()` lets you query the current state.

- **Cross-architecture SIMD API consistency** — final alignment pass across all platforms
- **Coverage tests** — targeted tests for stubs, `forge()`, `disable()`, `detect` helpers, and `IntoConcreteToken` on ARM/WASM stubs
- **Feature combination CI** — 8 feature combos tested (no-default, individual, pairs, all-features) plus aarch64 coverage with codecov flag-based merging
- **Cross-platform CI** — ARM, WASM, and i686 compilation verified; arch guards on platform-specific tests
- **Performance documentation** — DCT-8 and cross-token nesting benchmarks consolidated into `docs/PERFORMANCE.md`
- **SIMD reference mdbook** — searchable docs with ASM-verified load/store patterns
- **Feature flag strings and `AUDITING.md`** — `DISABLE_TARGET_FEATURES` string per token tells auditors exactly which RUSTFLAGS to set
- **Codegen quality** — replaced 330 `writeln!` chains with `formatdoc!` across token_gen.rs and main.rs
- **Miri CI stability** — isolated target dirs, pinned nightly, gated platform-specific tests

## 0.5.0 — 2025-12-20

Macro system overhaul and performance infrastructure.

- **`#[rite]` macro** — inner SIMD helpers with `#[target_feature]` + `#[inline]`. Use this by default; `#[arcane]` only at entry points. Benchmarked: calling `#[arcane]` per iteration costs 4-6x vs `#[rite]` inlining.
- **`incant!` macro** — runtime dispatch across `_v3`, `_v4`, `_neon`, `_wasm128`, `_scalar` suffixed functions
- **`#[magetypes]` macro** — replaces `#[multiwidth]`. Generates platform-specific variants from a generic `Token` parameter.
- **`ScalarToken`** — always-available fallback token for `incant!` dispatch
- **`IntoConcreteToken`** — safe upcasting with `as_x64v3()`, `as_x64v4()`, etc.
- **Atomic `summon()` caching** — 2-6x faster detection after first call (~1.3 ns cached, 0 ns when compiled away)
- **`compiled_with()` rename** — `guaranteed()` → `compiled_with()` for clarity
- **Token disable mechanism** — per-token `.disable(true)` for testing
- **NEON runtime detection fix** — no longer assumes NEON on AArch64 (broken on some Android/Linux kernels)
- **`SimdToken` sealed** — removed unsound `From` impls, added `forge()` for `unsafe` token construction
- **Per-token namespace modules** — `archmage::x64v3::Token`, `archmage::neon::Token`, etc.
- **Prelude module** — `use archmage::prelude::*` for tokens, traits, macros, intrinsics, and memory ops
- **`implementation_name()`** — all magetypes vectors report their backing implementation
- **Cross-architecture stubs** — all tokens compile on all platforms; `summon()` returns `None` on wrong arch
- **Removed `#[multiwidth]`** — replaced by `#[magetypes]`
- **Removed `bytemuck` dependency** — token-gated cast methods instead
- **`safe_unaligned_simd` integration** — re-exported via prelude, reference-based loads/stores

## 0.4.0 — 2025-11-15

Full cross-platform parity.

- **`token-registry.toml`** — single source of truth for all token definitions, feature sets, and trait mappings. Code generation reads this; validation checks against it.
- **API parity: 270 → 0 issues** — every W128 SIMD type has identical methods across x86, ARM, and WASM
- **ARM transcendentals** — full lowp + midp coverage: log2, exp2, ln, exp, log10, pow, cbrt for f32x4; lowp for f64x2
- **WASM transcendentals** — cbrt_midp, f64x2 log10_lowp, complete `_unchecked` and `_precise` suffix variants
- **ARM/WASM block ops** — interleave, deinterleave, transpose for all W128 types
- **x86 byte shift polyfills** — i8x16/u8x16 shl, shr, shr_arithmetic
- **WASM u64x2 ordering comparisons** — simd_lt/le/gt/ge via bias-to-signed polyfill
- **AVX-512 fast-path methods** — `_fast` variants for i64/u64 min/max/abs accepting `X64V4Token`
- **`cargo xtask parity`** — detects API variance between architectures
- **`cargo xtask validate`** — static soundness verification for all magetypes intrinsics
- **Miri boundary tests** — exhaustive load/store verification under Miri
- **Proptest fuzzing** — divergence detection across implementations
- **Codegen refactor** — generated files moved to `generated/` subfolders; all codegen uses `formatdoc!`

## 0.3.0 — 2025-10-28

Architecture cleanup.

- **Microarchitecture-level tokens** — replaced granular per-feature tokens with `X64V2Token`, `X64V3Token`, `X64V4Token` matching LLVM's x86-64 levels
- **Full WASM SIMD128 support** — `Wasm128Token` with optimized codegen
- **Intrinsic safety validation** — complete intrinsic database with automated safety checks
- **Intrinsic reference docs** — auto-generated docs organized by token tier
- **Removed ~2200 lines of dead wrapper code** and 6 unused dependencies

## 0.2.0 — 2025-10-15

Types and cross-platform.

- **`magetypes` crate** — token-gated SIMD types with natural operators (`f32x4`, `f32x8`, `i32x4`, etc.)
- **WASM SIMD support** — `Wasm128Token` and W128 types
- **AArch64 NEON support** — `NeonToken` with polyfilled wide types
- **`#[multiwidth]` macro** — generate multi-width SIMD variants (later replaced by `#[magetypes]`)
- **Token-gated bytemuck replacements** — `cast_slice`, `as_bytes`, `from_bytes` without `unsafe`
- **AVX-512 types** — 512-bit SIMD types behind `avx512` feature
- **`WidthDispatch` trait** — associated-type-based SIMD width dispatch

## 0.1.0 — 2025-10-01

Initial release.

- **Token-based SIMD capability proof** — `X64V3Token::summon()` returns `Some` only if CPU supports AVX2+FMA
- **`#[arcane]` macro** — generates `#[target_feature]` functions with cross-arch stubs
- **Zero overhead** — identical assembly to hand-written `unsafe` code
- **`#[forbid(unsafe_code)]` compatible** — all unsafety is inside the macro expansion
- **`no_std` + `alloc` by default** — `std` opt-in via feature flag

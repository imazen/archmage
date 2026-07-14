# Soundness: the model, the machinery, and how to audit it

This is the entry point for verifying — mechanically or by hand — that
archmage and magetypes are sound. It states the one invariant everything
rests on, inventories every place `unsafe` lives, lists what each tool
proves, and gives the audit procedure for reviewing changes.

Last full audit: 2026-07-14 (all counts below measured then).

## The invariant

> **Every call to a feature-gated CPU intrinsic is enclosed by a proof that
> the feature is available on the executing CPU.**

A *proof* is one of:

1. **A token value.** Token types (`X64V3Token`, `NeonToken`, …) are
   zero-sized and unforgeable from safe code: private field, sealed
   `SimdToken` supertrait, no `Default`/`new`. The only constructors are
   `summon()` (runtime CPU detection), the compile-time
   `cfg(target_feature)` fast path (the binary only runs where the features
   exist), and `forge_token_dangerously()` (`unsafe`, caller asserts the
   claim). Holding a token therefore implies the CPU has that tier's
   features.
2. **A `#[target_feature(enable = …)]` region.** Reaching such a function
   means the caller discharged the feature obligation — via a token-gated
   `#[arcane]` wrapper, a matching-feature safe call (Rust 1.86+), or an
   explicit `unsafe` call that took on the obligation.

Proofs union: a method taking `X64V4Token` inside an
`impl … for X64V3Token` block may use V3 ∪ V4 features.

Since Rust 1.87, value-based `core::arch` intrinsics are *safe* inside a
matching `#[target_feature]` region; everywhere else they require `unsafe`
with exactly this feature-availability obligation. magetypes' backend impls
take the second route: `unsafe { intrinsic }` justified by the token
receiver.

## Where `unsafe` lives (the complete inventory)

| Surface | Count | Invariant | Discipline |
|---|---|---|---|
| `src/tokens/generated/{x86,arm,wasm}.rs` forge call sites | 81 | summon/detect just verified the features, the features are compile-time guaranteed, or the source token's feature set is a registry-verified superset (extraction methods) | per-block `// SAFETY:` comments, generator-emitted, checker-enforced |
| `src/tokens/mod.rs` (`ScalarToken`, forge definitions) | 2 | `ScalarToken` proves the empty feature set; forge fns are `unsafe` with `# Safety` docs | doc sections |
| `magetypes/src/simd/impls/{x86_v3,x86_v4,arm_neon,wasm128}.rs` | ~1,960 blocks | uniform: token receiver proves intrinsic features (mechanically re-verified per run); loads/stores go through sized references; transmutes are same-size POD | file-header audit contract (generator-emitted, checker-enforced); per-block comments deliberately omitted as noise |
| `magetypes/src` outside `impls/` (byte casts, cross-width, slice reshape) | 225 blocks | size/align-guarded layout casts on all-bit-patterns-valid element types; token-gated construction | per-block `// SAFETY:` comments, checker-enforced |
| `archmage-macros` emitted code (`#[arcane]` wrappers etc.) | 1 `unsafe` block per wrapper | the token parameter (tier-tag const-asserted) proves the sibling's `#[target_feature]` set | justified in macro source; expansion snapshots under `tests/expand/` are re-verified by the intrinsic scanner (comments cannot survive tokenization, so snapshots carry no SAFETY text) |

Notable absences, enforced by structural rules: no `MaybeUninit`, no
`mem::zeroed`, no forging, no bare `transmute` outside the backend impls,
no `Default`/serde/bytemuck construction of SIMD wrappers anywhere in
magetypes.

## The mechanical verifiers

Run everything with `just ci`. Individually:

| Command | What it proves |
|---|---|
| `just soundness` (= `cargo xtask soundness`, also inside `generate`/`validate`/`ci`) | The structure-aware scanner (`xtask/src/soundness.rs`): every intrinsic call in `src/`, `magetypes/src/`, and `tests/expand/*.expanded.rs` sits inside a gating context whose feature set (from `token-registry.toml`) covers the intrinsic's requirements (from the stdarch-extracted `docs/intrinsics/complete_intrinsics.csv`, 10,884 entries). Also enforces the structural rules and SAFETY-comment discipline above. **Vacuous-pass guards:** global floor (4,000 verified calls; 4,478 measured at introduction) plus per-file floors — if the scanner stops seeing the backends, it fails rather than passing empty. |
| `cargo test -p xtask` (CI step 6) | The verifiers themselves: unit tests plant every violation class (feature mismatch, ungated intrinsic, trait-default-body intrinsic, unknown intrinsic, structural-rule breaches, missing SAFETY comments) and assert the scanner fires; plus a full-repo scan meeting the floors. |
| `just validate-tokens` | Every token's `summon()` checks exactly the features the registry declares (parses the generated detection code). |
| `just parity` | API parity across x86/ARM/WASM backends (0 issues). |
| `just miri` | UB detection over magetypes under Miri (layout casts, transmutes, pointer ops — the obligations the intrinsic scanner does *not* prove). |
| `just audit` | Scans the safety-critical non-generated areas listed in `docs/SAFETY-CRITICAL.md`. |
| `cargo test` (all platforms in CI) | Exercise tests: every token's claimed features drive real intrinsics on x86-64, ARM64 (cross/QEMU), WASM (wasmtime), Windows ARM64, macOS — see `tests/*_intrinsics*.rs`, `tests/feature_consistency.rs`. |
| Compile-fail suites (`tests/compile_fail.rs`, `magetypes/tests/bypass_adversarial.rs`, `tests/soundness/*`) | Negative space: tokenless UFCS calls, token shadowing/aliasing around `#[arcane]`, raw-pointer intrinsics without `unsafe` — all fail to compile. |
| Source-guard tests (`tests/apple_fallback_guard.rs`, `tests/winarm_registry_path_guard.rs`) | Platform detection paths that CI hardware cannot execute are pinned at the source level so known-bad patterns can't silently return. |

### Trust boundaries (what is asserted, not proven, and by whom)

- **`token-registry.toml`** is the axiom set: which features each token
  claims, and the tier hierarchy. Everything (tokens, macros, magetypes,
  the scanner) is generated from or checked against it. Auditing a token
  change = auditing the registry diff.
- **`complete_intrinsics.csv`** is extracted from rust stdarch sources
  (`just intrinsics-refresh`); intrinsic-shaped names missing from it are
  hard errors, so staleness surfaces instead of hiding.
- **Platform detection truths**: x86-64 CPUID; aarch64 via std_detect
  (Linux), `winarm-cpufeatures` registry decoding (Windows), and the Apple
  Silicon fallback **only** on provably-M1+ hosts (macOS, Catalyst,
  simulators — device iOS/tvOS/watchOS/visionOS use genuine runtime
  detection and otherwise fail closed). WASM is compile-time-only by
  design: a validated module implies the features.
- **The scanner is textual, not an AST.** Comments are stripped; braces are
  matched; generated code is formatting-stable. The floors plus the
  scanner's own unit tests are the defense against it silently rotting —
  which is exactly how the previous (registry-file-mapping) checker died:
  it passed for weeks while verifying zero calls.

## Auditing a change by hand

1. **`token-registry.toml` changed?** Verify each feature list against
   vendor documentation, and that every feature is exercised by an
   intrinsic test (`tests/*_intrinsics*.rs`). The registry is the axiom set
   — nothing downstream can catch a wrong axiom.
2. **`src/detect.rs` or generated token files changed?** Re-derive the
   summon proof: detection must positively verify *every* registry feature
   before any forge runs; caches must only reach "available" from a
   positive check (`grep 'store(if available'`); re-enable paths must reset
   to "unknown", never "available". Any unconditional-`true` arm must be
   justified by an ABI/target guarantee, narrowed with `cfg` to exactly the
   targets where it holds (see the Apple incident below).
3. **Backend impls or their generators changed?** `just generate` must
   leave a clean worktree; `just soundness` re-verifies every intrinsic.
   If a new impls file appears, add it to `REQUIRED_FILE_FLOORS`.
4. **New `unsafe` anywhere else?** The checker will demand a `// SAFETY:`
   comment; the comment must name the invariant, not restate the code. If
   the obligation is layout/pointer validity, extend the Miri tests
   (`magetypes/tests/miri_boundary_tests.rs`).
5. **Macros changed?** Regenerate expansion snapshots (`cargo xtask
   gen-expand`), then read the `.expanded.rs` diff — the sibling must stay
   a *safe* `fn` with `#[target_feature]`, the wrapper's `unsafe` call must
   remain justified by a token parameter or tier-tag assertion, and the
   scanner re-checks any intrinsics in the snapshots.
6. **Run `just ci` before push.** It chains all of the above.

## Incident log (why the guards exist)

- **2026-07-14 — vacuous checker.** The original intrinsic checker iterated
  `[[magetypes_file]]` registry mappings; a refactor emptied the list and
  the checker passed while verifying 0 calls. Replaced by the
  structure-aware scanner with coverage floors and self-tests.
- **2026-07-14 — Apple device fallback.** The Apple Silicon detection
  fallback was gated on `target_vendor = "apple"` alone, unconditionally
  asserting ten features on device iOS/tvOS/watchOS/visionOS targets whose
  hardware baseline only guarantees three. Fixed by narrowing to
  macOS/Catalyst/simulators; guarded by `tests/apple_fallback_guard.rs`.
- **(historical) token-by-self refactor.** Backend trait methods were
  associated functions callable via UFCS without a token value; PR #40 made
  every method take `self` (see `docs/SOUNDNESS_HANDOFF.md` for the design
  record). Now enforced mechanically by a structural rule plus the
  `bypass_adversarial.rs` compile-fail tests.

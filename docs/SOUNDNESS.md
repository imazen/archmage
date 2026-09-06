# Soundness: the model, the machinery, and how to audit it

This is the entry point for verifying — mechanically or by hand — that
archmage and magetypes are sound. It states the one invariant everything
rests on, inventories every place `unsafe` lives, lists what each tool
proves, and gives the audit procedure for reviewing changes.

Last full audit: 2026-07-14. The inventory below was re-measured 2026-09-05.

## The invariant

> **Every call to a feature-gated CPU intrinsic is enclosed by a proof that
> the feature is available on the executing CPU.**

A *proof* is one of:

1. **A token value.** Token types (`X64V3Token`, `NeonToken`, …) are
   zero-sized and unforgeable from safe code: private field, sealed
   `SimdToken` supertrait, no `Default`/`new`. The constructors are
   `summon()` (runtime CPU detection, including the compile-time
   `cfg(target_feature)` fast path — the binary only runs where the features
   exist) and `from_context()` (see below), which `summon()` is itself built
   from. Holding a token therefore implies the CPU has that tier's features.
2. **A `#[target_feature(enable = …)]` region.** Reaching such a function
   means the caller discharged the feature obligation — via a token-gated
   `#[arcane]` wrapper, a matching-feature safe call (Rust 1.86+), or an
   explicit `unsafe` call that took on the obligation.

Both proofs are authenticated, because both are selected **by name** from the
signature and a name is not a proof:

* A **concrete token** (`token: X64V3Token`) is checked against its generated
  `__ARCHMAGE_ASSERT_TIER_<tag>` const, which a same-named local struct does not
  have (`tests/soundness/token_shadowing_exploit.rs`) and a lower tier aliased to
  a higher tier's name does not match (`token_aliasing_exploit.rs`).
* A **tier trait bound** (`impl HasX64V2`, `<T: HasX64V2>`, or a where-clause) is
  re-stated through an absolute `::archmage::` path that a local trait cannot
  shadow, and required of the token value:

  ```rust
  const fn __archmage_assert_tier_trait<__T: ?Sized + ::archmage::HasX64V2>(_: &__T) {}
  __archmage_assert_tier_trait(&token);
  ```

  Since the tier traits are sealed through `SimdToken`, only a genuine token of
  that tier or stronger satisfies it. A `const fn` so it provably has no runtime
  body; by reference so the form also works for `impl Trait` in argument
  position, where the type cannot be named; all bounds on one helper so a
  multi-trait bound is checked as the same union of tiers the feature list was
  built from. Pinned by `trait_shadowing_exploit.rs` (plus its generic and
  `#[rite]` variants) and `trait_aliasing_exploit.rs`.

Proofs union: a method taking `X64V4Token` inside an
`impl … for X64V3Token` block may use V3 ∪ V4 features.

### Converting proof 2 back into proof 1

`from_context()` is a **safe** `#[target_feature]` function carrying its
tier's complete feature list, which makes rustc the arbiter of each call
site:

* From a caller whose own `#[target_feature]` attribute enables the tier's
  features or a superset — an `#[arcane]` / `#[rite]` / `#[magetypes]` body —
  the call needs no `unsafe`. The caller's attribute *is* proof 2, and the
  compiler checks the subset relation; the result is proof 1. This is the
  only way to move from a feature region back to a token without either a
  redundant runtime check or a hand-written obligation.
* From every other caller the call is `unsafe` and the caller carries the
  obligation by hand. That is how every generated internal call site uses
  it — `summon()`, the cold detect functions, the extraction methods and
  `IntoConcreteToken` are all feature-free functions.

`forge_token_dangerously()` is a deprecated alias with the identical gate.

Three properties are load-bearing and pinned by tests:

* Features enabled *globally* (`-C target-feature=+avx2`, `-C
  target-cpu=native`) do **not** make the call safe — rustc requires them on
  the caller's own attribute
  (`tests/soundness/from_context_missing_context.rs`).
* A weaker context cannot construct a stronger token
  (`tests/soundness/from_context_weaker_context.rs`).
* The function cannot be coerced to a safe function pointer — there would be
  no call site left to check (`tests/soundness/from_context_fn_pointer.rs`).

On a **foreign architecture** the stub constructor stays `unsafe fn`: no
`#[target_feature]` context for those features can exist on that target, so
there is nothing for rustc to check, so no `from_context()` is generated at
all — only the `unsafe fn` alias
(`tests/soundness/from_context_wrong_arch.rs`). On **WASM**, Rust permits safe
calls to `#[target_feature]` functions from any context; the engine validates
the required instructions when the module is loaded, so a module that runs at
all has the features. `ScalarToken` asserts the empty feature set, so its
constructor carries no gate.

Forging performs no runtime detection, so it also bypasses process-wide token
disabling (including `testable_dispatch`). That is intentional: the caller's
feature context is already the proof, and there is no way to make a
`#[target_feature]` region stop having its features. Use `summon()` when
dispatch must respond to runtime state.

The whole safe path is pinned by `tests/from_context.rs`, which is
`#![forbid(unsafe_code)]` — it compiles only if the safe route is genuinely
safe. The four rejection cases are driven by `tests/soundness_exploits.rs`
rather than trybuild: rustc's diagnostic names the target features enabled in
the *build configuration*, and that set differs per platform (Linux x86-64
says "the sse and sse2"; macOS-Intel says "the cmpxchg16b, sse, sse2, sse3,
sse4.1, and ssse3"), so a committed `.stderr` snapshot cannot pass on every
runner. The exploit harness asserts an error code plus message fragments
instead — and the fragments are our own identifiers, not rustc prose, so a
reworded diagnostic cannot break them either. `cargo xtask validate` enforces
the boundary: it rejects any committed trybuild `.stderr` that names target
features, quotes the build configuration, embeds an absolute or toolchain path,
carries a rustc version, or depends on pointer width. The soundness scanner's structural rules ban both `from_context` and
`forge_token_dangerously` from magetypes.

**The feature gate is free.** `#[inline(always)]` is not permitted on a
`#[target_feature]` function, so `from_context()` is plain `#[inline]`, and
LLVM will not inline it into the feature-free functions that call it. It does
not need to: the body constructs a ZST and emits no instructions, so the call
is dead-code-eliminated instead. Measured with `cargo asm` (0.2.62, rustc
1.97.1) against the pre-gate constructor across five probes — the entry
`summon()`-then-dispatch pattern, an extraction downcast, `summon()` in a
loop, `IntoConcreteToken`, and bare `summon()` — built both generically and
with `-C target-cpu=x86-64-v3`:

* no call to the constructor survives in **any** probe;
* the `call` count is unchanged everywhere (the one that remains is the cold
  `x64_v3_detect`, which was always there);
* under `-C target-cpu=x86-64-v3` every probe is instruction-identical, and
  bare `summon()` is still `mov eax, 1; ret` — it compiles away entirely;
* generically the instruction counts move by −2/+0/+1/+0/−1, which is branch
  layout and register allocation, not work.

Since Rust 1.87, value-based `core::arch` intrinsics are *safe* inside a
matching `#[target_feature]` region; everywhere else they require `unsafe`
with exactly this feature-availability obligation. magetypes' backend impls
take the second route: `unsafe { intrinsic }` justified by the token
receiver.

## Where `unsafe` lives (the complete inventory)

| Surface | Count | Invariant | Discipline |
|---|---|---|---|
| `src/tokens/generated/{x86,arm,wasm}.rs` `from_context()` call sites | 90 (58 x86 + 29 arm + 3 wasm) | summon/detect just verified the features, the features are compile-time guaranteed, or the source token's feature set is a registry-verified superset (extraction methods) | per-block `// SAFETY:` comments, generator-emitted, checker-enforced |
| `src/tokens/mod.rs` (`ScalarToken` constructors) | 0 | `ScalarToken` proves the empty feature set, so `from_context()` and its deprecated alias are ungated safe `const fn`s | doc sections |
| `src/tokens/generated/{x86,arm,wasm}_stubs.rs` forge definitions | 17 total (9 x86 + 6 arm + 2 wasm); 8–15 visible per target | foreign-architecture constructors: `unsafe fn` with an *unsatisfiable* `# Safety` contract — they exist so cross-architecture code compiles, not to be called | doc sections; `tests/soundness/from_context_wrong_arch.rs` |
| `magetypes/src/simd/impls/{x86_v3,x86_v4,arm_neon,wasm128}.rs` | **1 block** (was ~1,960) | per-method `#[arcane(_self = Token)]` turns each body into a `#[target_feature]` region, so the 5,142 value intrinsics in these files need no `unsafe` at all; the one remaining block is `x86_v3.rs`'s `sse2_baseline!` macro, which calls a *narrower* SSE2-only inner fn from the AVX tier | file-header audit contract (generator-emitted, checker-enforced); every intrinsic re-verified against the registry per run |
| `magetypes/src` outside `impls/` | **8 blocks, all in `simd_storage.rs`** (was 225) | size/align-guarded layout casts over `Pod` (all-bit-patterns-valid) storage; the four token-taking helpers additionally require a token value and const-assert the token is a 1-ZST | per-block `// SAFETY:` comments, checker-enforced; `unsafe impl Pod` is banned outside this file and every `TokenStorage` type must be `#[repr(C)]` |
| `archmage-macros` emitted code (`#[arcane]` wrappers etc.) | 1 `unsafe` block per wrapper | the token parameter (tier-tag const-asserted) proves the sibling's `#[target_feature]` set | justified in macro source; expansion snapshots under `tests/expand/` are re-verified by the intrinsic scanner (comments cannot survive tokenization, so snapshots carry no SAFETY text) |

Notable absences, enforced by structural rules: no `MaybeUninit`, no
`mem::zeroed`, no token construction (neither `from_context` nor
`forge_token_dangerously`), no bare `transmute` outside the backend impls,
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

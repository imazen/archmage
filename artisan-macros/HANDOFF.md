# artisan-macros handoff

Draft state snapshot. Captured for review at a later date.

## What exists

A working draft of `artisan-macros` — two proc-macros (`#[cpu_tier]`, `#[chain]`) with a thread-local test-hook layer, compile-time feature-string equality check, real-SIMD-kernel test, and dedicated CI workflow. 9 tests pass on x86_64 Linux.

File inventory:

| File | Purpose | Reviewed? |
|---|---|---|
| `Cargo.toml` | Proc-macro crate with `artisan_test_hooks` feature, `syn`/`quote`/`proc-macro2` deps | draft |
| `src/lib.rs` | Single-file implementation — parsing, arch inference, cpu_tier expansion, chain expansion, feature-string const + assertion, test-hooks codegen. ~840 LoC. | draft |
| `DESIGN.md` | Design rationale: trampoline-chain model, tri-state cache, compile-time entry, user-owned suffixes, forbid(unsafe_code) compat, constant-CPUID axiom | draft |
| `SPEC-CPU-TIER.md` | Normative spec for `#[cpu_tier]` — grammar, output, arch inference, validation rules | draft (pre-feature-const; update needed) |
| `SPEC-CHAIN.md` | Normative spec for `#[chain]` — grammar, expansion shape, cache semantics, compile-time elision, feature-string duplication rules | draft (pre-feature-const; update needed) |
| `SPEC-TEST-HOOKS.md` | Normative spec for the thread-local `force_max_tier` RAII API and its concurrency properties | draft |
| `README.md` | User-facing overview, shows the canonical usage shape | draft |
| `tests/smoke.rs` | 3-test smoke suite: dispatch correctness, idempotence, test-hook scope isolation. Scalar bodies in every tier. | passing on x86_64 |
| `tests/real_kernel.rs` | 6-test real-SIMD test: sums an `&[f32]` using AVX2+FMA on x86_64 and NEON on aarch64. Unsafe intrinsics inside tier bodies (not forbid-compat; by design for this test). | passing on x86_64 |
| `tests/expand.rs` | macrotest driver: globs `tests/expand/*.rs` and compares expansions to committed `.expanded.rs` snapshots. Gated on `target_arch = "x86_64"` because `cargo expand` strips non-host cfg branches. | passing on x86_64 |
| `tests/expand/*.rs` + `*.expanded.rs` | 4 snapshot pairs: simple cpu_tier, cpu_tier with explicit arch, single-arch chain, multi-arch chain. Any drift in macro output fails the expand test. | committed |
| `.github/workflows/artisan-macros.yml` | Dedicated CI: test on ubuntu/windows/windows-11-arm/macos-intel/macos-latest; cross to i686/aarch64/armv7; clippy/fmt/docs; expansion-snapshot drift check; deliberate-mismatch self-test to prove the compile-time check fires. | unverified (not yet run in CI) |
| `PARITY_HARNESS_ISSUE_DRAFT.md` | GitHub issue text for archmage enhancement: "add cross-tier parity test harness" | draft, unposted |
| `ARCHMAGE-THREAD-LOCAL-ISSUE.md` | GitHub issue text for archmage enhancement: "add thread-local override layer to token caches" (prerequisite to the parity harness) | draft, unposted |

## Build status (local, 2026-04-20)

```
cargo check -p artisan-macros                                     ✓
cargo clippy -p artisan-macros --all-targets -- -D warnings       ✓
cargo clippy -p artisan-macros --all-targets --features artisan_test_hooks -- -D warnings   ✓
cargo test -p artisan-macros                                      ✓ (10 tests pass: 3 smoke + 6 real_kernel + 1 expand snapshot)
cargo fmt -p artisan-macros -- --check                            ✓
RUSTDOCFLAGS=-D warnings cargo doc -p artisan-macros --no-deps    ✓
```

The feature-string compile-time check was sanity-verified by deliberately mutating `tests/smoke.rs` to declare `"avx2,fma,bmi2"` in `#[chain]` while `#[cpu_tier]` still said `"avx2,fma"`. Build failed with the expected `artisan-macros feature-string mismatch for tier \`dot_v3\` on arch \`x86_64\`` error. Reverted.

Not tested / not verified:

- **aarch64 cross-build** and test execution via `cross`. CI matrix includes it but hasn't been run yet.
- **wasm32 build**. `#[chain]` rejects wasm32 explicitly. `#[cpu_tier(enable = "simd128")]` alone should work but is untested.
- **Release builds** (`cargo check --release`). Untested locally; CI does debug only today.
- **`#![forbid(unsafe_code)]` downstream** — no test crate asserts this yet. The claim is unchanged (macro expansions don't require user `unsafe`); verification is pending.
- **Windows ARM64 (`windows-11-arm`)** in CI — included in the matrix but not yet run.

## Decisions locked in (from this session)

1. `chain` is an attribute macro (`#[chain(...)]`), not function-like.
2. Fallback keyword is `default`, not `scalar`.
3. `#[cpu_tier]` infers `target_arch` from feature names when unambiguous; fails loudly when all features are ambiguous. Users can override with `arch = "..."`.
4. Feature-set resolution is Option A: inline strings at both `#[cpu_tier]` and `#[chain]` sites. No hidden registry. Mismatches are user bugs documented in `SPEC-CHAIN.md`.
5. Test hooks use a thread-local `Option<u8>` override consulted by each trampoline, with an RAII scope guard. Per-thread isolation means no `#[serial]` or `RUST_TEST_THREADS=1` needed for parity tests.
6. Feature flag for test hooks: `artisan_test_hooks` (underscored). Unit tests inside the declaring crate also get hooks via `cfg(test)`.

## Open questions for review

Enumerated at the end of each SPEC file. Most salient:

- **SPEC updates pending.** `SPEC-CPU-TIER.md` and `SPEC-CHAIN.md` predate the feature-string const + compile-time assertion. Needs a Revisions section describing the emitted `__ARTISAN_CPU_TIER_FEATS_<fn>` const and the `const _: () = { assert!(str_eq(...)) }` blocks in chain output.
- **Lower-tier compile-time elision** (SPEC-CHAIN.md) — current only elides the top tier per arch. Nice-to-have, not correctness.
- **Generic fn / async fn support** (SPEC-CHAIN.md) — currently rejected for simplicity. Decide per concrete need.
- **Trampoline mangling collision hash** (SPEC-CHAIN.md) — current mangle is `__artisan_{chain_fn}__{arch}__{tier_fn}__chain`. Unlikely to collide in practice.
- **Process-wide force as an escape hatch** (SPEC-TEST-HOOKS.md) — currently not exposed. If rayon parity tests need it, decide on a feature-gated opt-in.
- **CI matrix coverage.** `windows-11-arm`, `macos-latest` (Apple Silicon), and the cross targets haven't run yet. The workflow is committed and will fire on first push.

## Next session, if resuming

1. **Read `DESIGN.md`, then each SPEC-*.md in order.** They're written to be picked up cold.
2. **Run `cargo test -p artisan-macros --manifest-path ../archmage-artisan/Cargo.toml`** to confirm the smoke + real_kernel suite still passes (9 tests).
3. **Check CI status** once pushed. The first push of `.github/workflows/artisan-macros.yml` will trigger the workflow; verify all jobs pass (native x86_64/ARM64/macOS, cross i686/aarch64/armv7, lint, docs, deliberate-mismatch test).
4. **Update SPEC-CPU-TIER.md and SPEC-CHAIN.md** to reflect the feature-string const + compile-time assertion. Both specs currently describe the pre-check state. The impl is correct; the specs are stale.
5. **Add a downstream `#![forbid(unsafe_code)]` test crate** at `tests/forbid_unsafe/` to prove the compat claim holds (compile, not just docs).
6. **Bench the trampoline overhead.** Expected ~1-3 ns per dispatch in the cached-hit case; the test-hooks `cfg(...)` block in each trampoline should be a zero-cost branch in release (the cfg strips it entirely).
7. **Promote to `publish = true` + v0.1.0** once the specs align with the impl and at least one downstream consumer exists (archmage could adopt it internally as a migration path).

## What NOT to do next session

- **Don't publish or push to crates.io.** This is a draft; `Cargo.toml` has `publish = false`.
- **Don't post either GitHub issue draft.** The user has them in hand (`PARITY_HARNESS_ISSUE_DRAFT.md` and `ARCHMAGE-THREAD-LOCAL-ISSUE.md`); they decide when.
- **Don't consolidate `src/lib.rs` into modules.** The "single file" aspiration is core to the artisan ethos — if lib.rs grows past ~1000 LoC, consider splitting, but not before.
- **Don't add tier presets or a registry.** User-owned feature strings are a feature, not an oversight.

## Where this lives

- jj workspace: `/home/lilith/work/archmage-artisan/` (workspace added to the archmage repo; see `jj -R ... workspace list`)
- Primary checkout: `/home/lilith/work/archmage/` (unchanged; hasn't seen this work)
- Bookmark: **`feat/artisan-macros-draft`** — this explicitly does NOT merge to `main`. Pushes go to the bookmark, not the main branch. Promotion to `main` requires explicit review and decision.
- Commit hash at handoff: recorded in the final `jj log` output of the session transcript — check `jj log -r feat/artisan-macros-draft` or `git log`.

## Workongoing marker

`.workongoing` exists in both the primary checkout and the `../archmage-artisan` workspace. Before resuming, refresh the marker at whichever checkout you're operating in. Delete it when done.

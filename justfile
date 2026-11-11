# Archmage build and test commands

# Default: run all tests
default: test

# Run all tests (excludes sleef which requires nightly)
test:
    cargo test --features "std avx512"

# Exact float mask/negation bits, generated parity, and their documented contracts.
test-float-contracts:
    cargo test -p magetypes --features "std avx512" --test float_inequality --test float_negation --test scalar_parity --test doc_examples

# Run the same float contracts under ARM emulation and strict WASM SIMD.
test-float-contracts-arm:
    cross test -p magetypes --target aarch64-unknown-linux-gnu --features std --test float_inequality --test float_negation --test scalar_parity --test doc_examples

test-float-contracts-wasm:
    RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime cargo test -p magetypes --target wasm32-wasip1 --features std --test float_inequality --test float_negation --test scalar_parity --test doc_examples

# Emit the named neg8_codegen probe for instruction inspection.
float-contract-codegen:
    cargo rustc -p magetypes --release --features "std avx512" --test float_negation -- --emit=asm

# Run tests with all features (requires nightly for sleef)
test-nightly:
    cargo +nightly test --all-features

# Run clippy (excludes sleef which requires nightly)
lint:
    cargo clippy --features "std avx512" -- -D warnings
    cargo clippy -p archmage-no-features-test -p archmage-clippy-consumer --all-targets -- -D warnings

# Clippy over every target/feature/CPU configuration (needs the rustup targets
# listed in .github/workflows/ci.yml's warning-sweep job). --pedantic adds
# clippy::pedantic and reports macro-expansion spans only.
warnings *ARGS:
    python3 -B xtask/warning_sweep.py {{ARGS}}

# Format code + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the workspace-excluded apidoc/ package, so it
# is never built or run by plain `cargo test`. Snapshots are generated per
# --target (x86_64 / aarch64 / wasm32 subdirectories, issue #75) so the
# output is byte-identical no matter the host arch; nightly + the target
# stdlibs are auto-installed via rustup.
fmt:
    cargo fmt
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only (per-target; see fmt)
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current (CI: the "Public API Check" job)
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# Check formatting
fmt-check:
    cargo fmt -- --check

# Run Miri tests (token logic only, no SIMD) - legacy alias
miri-tokens:
    rustup run "$(tr -d '[:space:]' < xtask/miri-nightly.txt)" cargo miri test --test miri_safe --all-features

# Run Miri on magetypes with full SIMD support (detects UB)
miri:
    cargo run -p xtask -- miri

# Static soundness verification (validates intrinsics against stdarch database)
soundness:
    cargo run -p xtask -- soundness

# Safety audit (scan for critical code, check intrinsics freshness)
audit:
    cargo run -p xtask -- audit

# Refresh intrinsics database from current Rust toolchain
intrinsics-refresh:
    cargo run -p xtask -- intrinsics-refresh

# Fuzz test for divergences between native and polyfill implementations
fuzz:
    cargo test -p magetypes --test fuzz_divergence --features avx512

# Regenerate all generated code (SIMD types, macro registry, docs)
generate:
    cargo run -p xtask -- generate

# Verify the working tree agrees with the generator: regeneration must be a
# no-op. Local mirror of CI's `generate-check` job, but hash-based rather than
# `git diff`, so it works on a dirty tree — the question is "does the generator
# emit what's on disk?", not "is the tree clean?".
#
# Run this before committing any change under a generated path (see CLAUDE.md,
# "FIX THE GENERATOR, NOT ITS OUTPUT"). A fix hand-applied to generated output
# passes tests locally and is then silently reverted by the next
# `just generate`; this is what catches that.
check-generated:
    #!/usr/bin/env bash
    set -uo pipefail
    before="$(mktemp)"; after="$(mktemp)"
    trap 'rm -f "$before" "$after"' EXIT
    git ls-files -z | xargs -0 shasum -a 256 2>/dev/null | sort > "$before"
    cargo run -p xtask -- generate || exit 1
    git ls-files -z | xargs -0 shasum -a 256 2>/dev/null | sort > "$after"
    if diff -q "$before" "$after" >/dev/null; then
        echo "OK: generated code matches the generator (regeneration is a no-op)."
        exit 0
    fi
    echo ""
    echo "ERROR: regeneration changed these files —"
    diff "$before" "$after" | awk '/^>/ {print "  " $3}' | sort -u
    echo ""
    echo "The tree disagrees with the generator. Either you edited generated"
    echo "output directly, or you changed a template without regenerating."
    echo "Generator templates live in xtask/src/simd_types/."
    exit 1

# Validate token-registry.toml (parse + structural checks)
validate-registry:
    cargo run -p xtask -- validate-registry

# Validate magetypes safety + try_new() feature checks against registry
validate-tokens:
    cargo run -p xtask -- validate

# Check API parity across x86/ARM/WASM architectures
parity:
    cargo run -p xtask -- parity

# Test no_std compilation and tests for all crates
test-nostd:
    cargo check -p archmage --no-default-features --features "avx512"
    cargo check -p magetypes --no-default-features
    cargo test -p archmage --no-default-features --features "avx512"
    cargo test -p magetypes --no-default-features
    cargo check -p magetypes --no-default-features --target aarch64-unknown-none
    cargo check -p magetypes --no-default-features --target thumbv7m-none-eabi

# Run ALL CI checks (MUST pass before push or publish)
ci:
    cargo run -p xtask -- ci

# Alias for ci (all-inclusive check)
all: ci

# Test testable_dispatch with CompileTimePolicy::Fail (must not panic)
test-dispatch:
    cargo test --features "std avx512 testable_dispatch" --test token_permutations -- --test-threads=1

# ============================================================================
# Parity tests (cross-architecture + polyfill vs native)
# ============================================================================

# Run parity tests on x86_64 (native)
test-parity-x86:
    cargo test -p magetypes --test cross_arch_parity --features "std avx512"

# Run parity tests on aarch64 (via QEMU/cross)
test-parity-arm:
    cross test -p magetypes --test cross_arch_parity --target aarch64-unknown-linux-gnu

# Run parity tests on WASM (via wasmtime)
test-parity-wasm:
    RUSTFLAGS="-C target-feature=+simd128" cargo test -p magetypes --test cross_arch_parity --target wasm32-wasip1

# Run all parity tests (x86 + ARM + WASM)
test-parity: test-parity-x86 test-parity-arm test-parity-wasm
    @echo "All parity tests passed!"

# ============================================================================
# Intel SDE testing (requires Intel SDE to be installed)
# Download from: https://www.intel.com/content/www/us/en/download/684897/intel-software-development-emulator.html
# ============================================================================

# Test as Pentium 4 (SSE2 only, no SSE3/SSSE3/SSE4)
test-p4:
    sde64 -p4 -- cargo test --all-features

# Test as Nehalem (SSE4.2, no AVX)
test-nehalem:
    sde64 -nhm -- cargo test --all-features

# Test as Haswell (AVX2 + FMA, no AVX-512)
test-haswell:
    sde64 -hsw -- cargo test --all-features

# Test as Skylake-X (AVX-512)
test-skylake:
    sde64 -skx -- cargo test --all-features

# Test as Ice Lake (AVX-512 + VBMI2)
test-icelake:
    sde64 -icl -- cargo test --all-features

# Run all SDE tests (requires Intel SDE)
test-all-cpus: test-p4 test-nehalem test-haswell test-skylake test-icelake

# ============================================================================
# Cross-compilation testing (requires cargo-cross)
# Install: cargo install cross --git https://github.com/cross-rs/cross
# ============================================================================

# Test on 32-bit x86 (via QEMU)
test-i686:
    cross test --all-features --target i686-unknown-linux-gnu
    cross test -p archmage-no-features-test --target i686-unknown-linux-gnu

# Test on aarch64 (via QEMU) - lib, cross-platform, and feature intrinsic tests
test-aarch64:
    cross test --lib --target aarch64-unknown-linux-gnu
    cross test --test miri_safe --test feature_consistency --test arm_safe_intrinsics --test arm_feature_intrinsics --target aarch64-unknown-linux-gnu

# Test on armv7 (via QEMU) - lib and cross-platform tests only
test-armv7:
    cross test --lib --target armv7-unknown-linux-gnueabihf
    cross test --test miri_safe --test feature_consistency --target armv7-unknown-linux-gnueabihf

# Build for all cross targets (faster than running tests)
build-cross:
    cross build --all-features --target i686-unknown-linux-gnu
    cross build --all-features --target aarch64-unknown-linux-gnu
    cross build --all-features --target armv7-unknown-linux-gnueabihf

# Run tests on all cross targets
test-cross: test-i686 test-aarch64 test-armv7
    @echo "All cross-compilation tests passed!"

# Clippy for x86_64
clippy-x86_64:
    cargo clippy --all-features --target x86_64-unknown-linux-gnu -- -D warnings

# Clippy for aarch64
clippy-aarch64:
    cargo clippy --all-features --target aarch64-unknown-linux-gnu -- -D warnings

# Clippy for i686
clippy-i686:
    cargo clippy --all-features --target i686-unknown-linux-gnu -- -D warnings

# Clippy for all targets
clippy-all: clippy-x86_64 clippy-aarch64 clippy-i686
    @echo "All clippy checks passed!"

# ============================================================================
# Per-token intrinsic exercise tests
# ============================================================================

# Test x86 crypto token intrinsics (PCLMULQDQ, AES-NI, VPCLMULQDQ, VAES)
test-crypto:
    cargo test --features "std" --test x86_crypto_intrinsics

# Test AVX-512 FP16 token (hierarchy only — intrinsics are nightly-only)
test-fp16:
    cargo test --features "std avx512" --test avx512fp16_intrinsics

# Test ARM feature-specific intrinsics (via QEMU/cross)
test-arm-features:
    cross test --test arm_feature_intrinsics --target aarch64-unknown-linux-gnu

# Test WASM SIMD128 intrinsics (via wasmtime)
test-wasm-intrinsics:
    RUSTFLAGS="-C target-feature=+simd128" cargo test --test wasm_intrinsics_exercise --target wasm32-wasip1

# Test all per-token intrinsics (x86 + ARM + WASM)
test-all-tokens: test-crypto test-fp16 test-arm-features test-wasm-intrinsics
    @echo "All per-token intrinsic tests passed!"

# ============================================================================
# CI-style comprehensive test
# ============================================================================

# Note: Main CI target is defined above (uses cargo xtask ci)
# These are extended validation targets for more thorough testing

# Full validation with SDE (for local development)
validate-sde: ci test-all-cpus
    @echo "Full SDE validation complete!"

# Full validation including cross-compilation
validate-cross: ci test-cross clippy-all
    @echo "Full cross-platform validation complete!"

# ============================================================================
# Benchmarking (requires -C target-cpu=native for accurate results)
# ============================================================================

# Run all benchmarks with native CPU optimizations
bench:
    RUSTFLAGS="-C target-cpu=native" cargo bench

# Run transcendental benchmarks
bench-transcendental:
    RUSTFLAGS="-C target-cpu=native" cargo bench --bench transcendental_accuracy

# Run edge case benchmarks
bench-edge-cases:
    RUSTFLAGS="-C target-cpu=native" cargo bench --bench edge_case_perf

# IMPORTANT: Without -C target-cpu=native, intrinsics won't inline properly
# and benchmarks will show archmage being 4-5x slower than wide.
# With native CPU targeting, archmage is 1.2-1.4x faster than wide.

# Time cold builds of the magetypes crate. ROOT holds before/ and after/ source
# trees, e.g. `git archive v0.9.29 | tar -x -C ROOT/before`.
magetypes-compile-perf ROOT PAIRS="6":
    python3 benchmarks/magetypes_compile_perf.py {{ROOT}} {{PAIRS}}

# ============================================================================
# ASM Verification (requires cargo-show-asm)
# Install: cargo install cargo-show-asm
# ============================================================================

# Verify documented ASM claims match actual compiler output
verify-asm:
    ./scripts/verify-asm.sh

# Update expected ASM baselines (run after intentional codegen changes)
verify-asm-update:
    ./scripts/verify-asm.sh --update

# ============================================================================
# Documentation
# ============================================================================

# Check rustdoc builds cleanly (catches broken doc links)
doc-check:
    RUSTDOCFLAGS="-Dwarnings" cargo doc --features "std avx512" --no-deps

# Compile and execute website and README Rust examples directly
docs-test:
    python3 xtask/check_docs.py
    python3 xtask/check_docs.py --features avx512

# Test Markdown examples using the selected target and configured Cargo runner.
docs-test-target target *pages:
    python3 xtask/check_docs.py --target {{target}} {{pages}}

# Build the documentation site (Zola)
docs:
    cd docs/site && zola build -o ../../target/site

# Requires Playwright + Chromium; omit site for the deployed Pages audit.
docs-browser-check output site="":
    node xtask/check_site.cjs {{quote(output)}} {{quote(site)}}

# Type-check and capture the displayed cargo-expand examples (pinned tools).
docs-expand:
    python3 xtask/site_expansions.py

docs-expand-check:
    python3 xtask/site_expansions.py --check

# Serve the documentation locally (with auto-reload, port 3100)
docs-serve:
    cd docs/site && zola serve --port 3100

# Clean the built documentation
docs-clean:
    rm -rf target/site

# Serve the intrinsics browser locally (port 3500)
intrinsics-serve:
    cd docs/intrinsics-browser && python3 -m http.server 3500
# Native ARM codegen comparisons; no target-cpu override.
bench-arm-codegen-macos:
    #!/usr/bin/env bash
    set -euo pipefail
    test "$(uname -s)" = Darwin
    mkdir -p "$HOME/tmp"
    audit_log="$HOME/tmp/archmage-arm-codegen-$(date -u +%Y%m%dT%H%M%SZ).log"
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 \
      nice -n 19 /usr/bin/time -l cargo bench --locked -p magetypes --bench generic_vs_concrete -- --format=llm \
      2>&1 | tee "$audit_log"

# Compare optimized consumer codegen between two revisions on x86_64, AArch64
# and WASM (CI runs this only for pull requests). Covers mul_add_portable in
# plain and V3 callers; run it after changing magetypes/src/soft_fma.rs.
codegen-diff base="origin/main" head="HEAD":
    python3 xtask/codegen.py {{base}} {{head}}

# Integer APIs, independent intrinsic references and ARM accumulation fusion.
integer-codegen:
    PYTHONDONTWRITEBYTECODE=1 python3 xtask/codegen.py --integer-ops

integer-tests:
    cargo test -p magetypes --test int_widen_narrow --features "std avx512"

# Explicit-token migration aliases and independent raw/context safety coverage.
token-migration-check:
    cargo test -p xtask token_aliases
    cargo test -p magetypes --test token_aliases --test raw_interop
    cargo test -p magetypes --test token_aliases --no-default-features
    cargo test -p magetypes --test token_aliases --features avx512
    cargo test -p archmage --test magetypes_scalar_dispatch --test soundness_exploits

token-migration-arm:
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu" cargo test -p magetypes --test token_aliases --test raw_interop --target aarch64-unknown-linux-gnu

# Refresh both expansion snapshot sets after a macro change: the cargo-expand
# ones (post-cfg, host arch) and the raw ones (macro output, every arch).
expand-snapshots:
    cargo run -p xtask -- gen-expand
    MACROTEST=overwrite cargo test -p archmage --test macro_expand -- --test-threads=1
    MACROTEST=overwrite cargo test -p magetypes --test macro_expand -- --test-threads=1
    ARCHMAGE_RAW_SNAPSHOTS=overwrite cargo test -p archmage-macros raw_snapshots

# Regenerate magetypes/tests/harvest_shapes.rs from snapshots of consumer
# crates: every macro signature shape they use, compiled as one test.
# ROOT holds the snapshots (e.g. `git archive HEAD | tar -x -C ROOT/zen/<crate>`).
harvest-shapes ROOT:
    python3 -I xtask/harvest_shapes.py {{ROOT}}
    cargo fmt --all

# Build the actual crate archives together without publishing.
check-packages *TARGETS:
    cargo run -p xtask -- check-packages {{TARGETS}}

# Fused arithmetic regression and interleaved software-path comparison
test-fused:
    cargo test -p magetypes --all-features --test fused_arithmetic

bench-fused:
    cargo bench -p magetypes --bench nostd_math_perf -- --group=fma --format=json

# Override runner to exercise the engine's non-fusing relaxed lowering on x86.
test-fused-wasm runner="wasmtime" flags="+simd128":
    RUSTFLAGS="-Ctarget-feature={{flags}}" CARGO_TARGET_WASM32_WASIP1_RUNNER="{{runner}}" cargo test -p magetypes --target wasm32-wasip1 --test fused_arithmetic --test doc_examples --test scalar_parity

# Exercise the same arithmetic with magetypes' std feature disabled.
test-fused-wasm-no-std runner="wasmtime" flags="+simd128":
    RUSTFLAGS="-Ctarget-feature={{flags}}" CARGO_TARGET_WASM32_WASIP1_RUNNER="{{runner}}" cargo test -p magetypes --no-default-features --features w512 --target wasm32-wasip1 --test fused_arithmetic --test doc_examples --test scalar_parity

# Review .rs changes since the last release tag, without generated code:
# generated/ directories, files with a generated-file header, and
# macro-expansion snapshots (*.expanded.rs). Scope: handwritten (default),
# lib (the three crates' src/), or all. Extra arguments go to git diff:
#   just diff-rs lib --stat
#   BASE=v0.9.28 just diff-rs
diff-rs scope="handwritten" *args:
    #!/usr/bin/env bash
    set -euo pipefail
    base="${BASE:-$(git describe --tags --abbrev=0 --match 'v[0-9]*' HEAD)}"
    files=$(git diff --name-only "$base"..HEAD -- '*.rs' | while read -r f; do
        if [ "{{scope}}" != all ]; then
            case "$f" in *.expanded.rs|*/generated/*) continue ;; esac
            if [ -f "$f" ] && head -15 "$f" | grep -qiE '@generated|auto-generated|generated by|do not edit'; then
                continue
            fi
        fi
        if [ "{{scope}}" = lib ]; then
            case "$f" in src/*|magetypes/src/*|archmage-macros/src/*) ;; *) continue ;; esac
        fi
        echo "$f"
    done)
    [ -n "$files" ] || { echo "no .rs changes since $base"; exit 0; }
    echo "base $base, $(echo "$files" | wc -l) files" >&2
    git -c delta.navigate=true diff {{args}} "$base"..HEAD -- $files

# Beta integration and unchanged legacy output contracts.
attune-beta-unit:
    cargo test -p archmage-macros

attune-expanded-raw out *args:
    python3 -B tests/attune_expansion/run.py --out {{out}} {{args}}

attune-expanded-generator-test:
    python3 -B -m unittest discover -s tests/attune_expansion -v

# Release validation runs serially, including generated sources and all targets.
attune-beta-check:
    just generate
    just validate-registry
    just validate-tokens
    just soundness
    cargo test --all-targets
    cargo test --doc

attune-beta-clippy:
    cargo clippy -p archmage-macros --all-targets --all-features -- -D warnings

attune-beta-packages:
    cargo test -p magetypes --all-targets --features "std avx512"
    cargo test -p magetypes --doc --features "std avx512"
    cargo test -p archmage-macros
    cargo publish --workspace --dry-run --allow-dirty

attune-beta-semver:
    python3 -B xtask/check_semver.py archmage --baseline-version 0.9.30
    python3 -B xtask/check_semver.py magetypes --baseline-version 0.9.30

# Published baseline selected by cargo-semver-checks; includes the precise guard.
semver package:
    python3 -B xtask/check_semver.py {{package}}

# Alternate pinned source archives; no CPU-native flags or reused cold targets.
attune-beta-compare baseline candidate out pairs="6":
    python3 -B benchmarks/attune_compare.py --baseline {{baseline}} --candidate {{candidate}} --out {{out}} --pairs {{pairs}} --refresh-lock

# Diagnose semver-checker self-comparisons without suppressing any lint.
semver-self rustdoc:
    cargo semver-checks check-release --current-rustdoc {{rustdoc}} --baseline-rustdoc {{rustdoc}} --release-type patch

# Compare CPU feature requirements without conflating token specializations.
check-target-features baseline current:
    python3 xtask/check_target_features.py {{baseline}} {{current}}

check-target-features-test:
    python3 -B -m unittest discover -s xtask -p test_target_features.py -v

# Cold checks/release builds of retained, fixed codec source captures.
consumer-compile out mode *args:
    python3 -B benchmarks/consumer_compile.py --out {{out}} {{mode}} {{args}}

# Check snapshots and positive compilation without refreshing expectations.
expand-check:
    cargo test -p archmage --test macro_expand -- --test-threads=1

# Check the independent, exact-rational FMA corpus without rewriting expectations.
fma-fixtures-check:
    python3 xtask/fma_vectors.py --check

# Arithmetic contracts, including unfused scalar and backend-native fast paths.
fma-contract:
    cargo test -p magetypes --features avx512 --test fma_contract --test fused_arithmetic
    cargo test -p magetypes --no-default-features --features w512 --test fma_contract --test fused_arithmetic

# Exercise the native software fallback on x86 without FMA.
fma-no-hardware:
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="qemu-x86_64 -cpu Nehalem" cargo test -p magetypes --target x86_64-unknown-linux-gnu --test fma_contract --test fused_arithmetic
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="qemu-x86_64 -cpu Nehalem" cargo test -p magetypes --no-default-features --features w512 --target x86_64-unknown-linux-gnu --test fma_contract --test fused_arithmetic

# Independently build/test the self-contained dispatch prototype (x86-64 host).
artisanal-check:
    bash experiments/artisanal-dispatch/tools/native-check.sh

# Validate an isolated stable candidate; run on an admitted worker under run-heavy.
release-compat source output phase:
    python3 xtask/release_compat/run.py --source {{source}} --output {{output}} --phase {{phase}}

release-compat-recheck source output inventory:
    python3 xtask/release_compat/recheck.py --source {{source}} --output {{output}} --inventory {{inventory}}

release-compat-audit directory source *options:
    python3 xtask/release_compat/audit.py {{directory}} --source {{source}} {{options}}

# Bounded, fixed-fused powf: independent graph fixtures and required backend.
powf-fixtures-check:
    python3 -B xtask/powf_vectors.py --check

powf-contract tier="scalar":
    ARCHMAGE_POWF_TIER={{tier}} cargo test --release -p magemath --features avx512 --lib powf::

powf-payloads tier="scalar" stride="65537":
    ARCHMAGE_POWF_TIER={{tier}} ARCHMAGE_NAN_STRIDE={{stride}} cargo test --release -p magemath --features avx512 --lib powf::tests::nan_sign_payload

# Build the frozen bounded-powf study (distribution target flags only).
powf-study-build target="x86_64-unknown-linux-gnu":
    cargo build --release --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --target {{target}}

# Strict zenbench admission; output must not already exist.
powf-study result filter="":
    POWF_RESULT="{{result}}" POWF_FILTER="{{filter}}" cargo run --release --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml

powf-accuracy output:
    cargo build --release --locked --manifest-path benchmarks/powf_validation/Cargo.toml --bin powf_contract_dump
    python3 -B benchmarks/powf_module_2026-11-07/accuracy.py benchmarks/powf_validation/target/release/powf_contract_dump "{{output}}"

powf-study-matrix directory native="v3":
    python3 -B benchmarks/powf_module_2026-11-07/run_matrix.py benchmarks/powf_module_2026-11-07/target/release/archmage-powf-study "{{directory}}" --native {{native}}

powf-study-analyze *files:
    cargo run --release --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --bin analyze -- {{files}}

powf-contract-target target:
    cargo test --release -p magemath --target {{target}} --lib powf::

powf-wasm-build:
    RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_DIR=target/powf-wasm-strict cargo test --release -p magemath --target wasm32-wasip1 --lib powf:: --no-run
    RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" CARGO_TARGET_DIR=target/powf-wasm-relaxed cargo test --release -p magemath --target wasm32-wasip1 --lib powf:: --no-run

powf-wasm-contract:
    bash benchmarks/powf_module_2026-11-07/verify_wasm.sh

powf-study-cold directory base candidate:
    python3 -B benchmarks/powf_module_2026-11-07/cold_build.py "{{directory}}" --base {{base}} --candidate {{candidate}}

powf-study-wasm-build directory mode:
    python3 -B benchmarks/powf_module_2026-11-07/build_guest.py "{{directory}}" {{mode}}

powf-study-wasm-host:
    cargo build --release --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --features wasm-host --bin wasm_host

powf-study-lint:
    cargo clippy --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --all-targets --all-features -- -D warnings

powf-study-wasm-verify wasm mode:
    POWF_WASM="{{wasm}}" POWF_WASM_MODE={{mode}} POWF_VERIFY_ONLY=1 benchmarks/powf_module_2026-11-07/target/release/wasm_host

powf-sde-contract sde cpu="-nhm" expected="software":
    python3 -B benchmarks/powf_module_2026-11-07/sde_contract.py "{{sde}}" --cpu={{cpu}} --expected={{expected}}

# Actual legacy/std controls, const options, and explicitly labeled diagnostics.
powf-replacement-build:
    cargo build --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --bin replacement --bin replacement_eval --bin replacement_analyze

# Controlled benchmark-only rounding study; public fused defaults unchanged.
powf-rounding-build:
    cargo build --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --bin rounding_bench --bin rounding_eval --bin rounding_analyze --bin rounding_codegen --bin rounding_widths --lib

powf-rounding-check:
    cargo test --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --lib
    POWF_VERIFY_ONLY=1 benchmarks/powf_module_2026-11-07/target/release/rounding_bench

powf-policy-check:
    cargo fmt -p magetypes --check
    cargo test --release -p magemath --all-features --lib powf::
    cargo test --release -p magemath --no-default-features --features w512 --lib powf::
    cargo test --doc -p magetypes --all-features
    cargo clippy -p magetypes --all-features --all-targets -- -D warnings

powf-policy-build:
    cargo build --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --bin policy_widths --bin policy_bench --bin policy_codegen --bin policy_analyze --bin highp_eval

powf-policy-widths fixtures:
    benchmarks/powf_module_2026-11-07/target/release/policy_widths scalar < "{{fixtures}}"
    benchmarks/powf_module_2026-11-07/target/release/policy_widths native < "{{fixtures}}"

powf-rounding-matrix directory *options:
    python3 -B benchmarks/powf_module_2026-11-07/replacement_matrix.py benchmarks/powf_module_2026-11-07/target/release/rounding_bench "{{directory}}" --rounding {{options}}

powf-rounding-accuracy inputs output *options:
    python3 -B benchmarks/powf_module_2026-11-07/rounding_accuracy.py benchmarks/powf_module_2026-11-07/target/release/rounding_eval "{{inputs}}" "{{output}}" {{options}}

powf-rounding-codegen:
    cargo rustc --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --bin rounding_codegen -- --emit=asm

powf-rounding-codegen-audit assembly output *options:
    python3 -B benchmarks/powf_module_2026-11-07/rounding_codegen_audit.py "{{assembly}}" "{{output}}" {{options}}

powf-rounding-matrix-analyze directory *runs:
    python3 -B benchmarks/powf_module_2026-11-07/analyze_replacement_matrix.py benchmarks/powf_module_2026-11-07/target/release/rounding_analyze "{{directory}}" {{runs}} --require-cells 96

powf-rounding-search-build source directory:
    python3 -B benchmarks/powf_module_2026-11-07/error_search.py build "{{source}}" benchmarks/powf_module_2026-11-07/target/release/libarchmage_powf_study.a "{{directory}}" --rounding

powf-rounding-search binary directory:
    python3 -B benchmarks/powf_module_2026-11-07/error_search.py run "{{binary}}" "{{directory}}" --rounding

powf-rounding-replay binary inputs reference output *options:
    python3 -B benchmarks/powf_module_2026-11-07/rounding_replay.py "{{binary}}" "{{inputs}}" "{{reference}}" "{{output}}" {{options}}

powf-rounding-widths fixtures:
    benchmarks/powf_module_2026-11-07/target/release/rounding_widths scalar < "{{fixtures}}"
    benchmarks/powf_module_2026-11-07/target/release/rounding_widths native < "{{fixtures}}"

powf-rounding-widths-i686 fixtures:
    cargo build --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --target i686-unknown-linux-gnu --bin rounding_widths
    benchmarks/powf_module_2026-11-07/target/i686-unknown-linux-gnu/release/rounding_widths scalar < "{{fixtures}}"

powf-rounding-wasm-build flags="+simd128":
    RUSTFLAGS="-Ctarget-feature={{flags}}" cargo build --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --target wasm32-wasip1 --bin rounding_eval

powf-rounding-wasm-codegen flags="+simd128":
    RUSTFLAGS="-Ctarget-feature={{flags}}" cargo rustc --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --target wasm32-wasip1 --bin rounding_codegen -- --emit=asm

powf-replacement-matrix directory *options:
    python3 -B benchmarks/powf_module_2026-11-07/replacement_matrix.py benchmarks/powf_module_2026-11-07/target/release/replacement "{{directory}}" {{options}}

powf-replacement-analyze *files:
    benchmarks/powf_module_2026-11-07/target/release/replacement_analyze {{files}}

powf-replacement-codegen:
    cargo rustc --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --bin replacement_codegen -- --emit=asm

powf-replacement-accuracy inputs output *options:
    python3 -B benchmarks/powf_module_2026-11-07/replacement_accuracy.py benchmarks/powf_module_2026-11-07/target/release/replacement_eval "{{inputs}}" "{{output}}" {{options}}

# Check a later build against preserved actual outputs without changing its oracle.
powf-replacement-replay inputs before output:
    #!/usr/bin/env bash
    set -euo pipefail
    for tier in scalar native; do
        test ! -e "{{output}}.$tier.actual"
        benchmarks/powf_module_2026-11-07/target/release/replacement_eval "$tier" < "{{inputs}}" > "{{output}}.$tier.actual"
        cmp "{{before}}.$tier.actual" "{{output}}.$tier.actual"
    done

powf-replacement-search-build source directory:
    python3 -B benchmarks/powf_module_2026-11-07/error_search.py build "{{source}}" benchmarks/powf_module_2026-11-07/target/release/libarchmage_powf_study.a "{{directory}}"

powf-replacement-search binary directory *options:
    python3 -B benchmarks/powf_module_2026-11-07/error_search.py run "{{binary}}" "{{directory}}" {{options}}

powf-replacement-wasm-build directory mode:
    python3 -B benchmarks/powf_module_2026-11-07/build_guest.py "{{directory}}" {{mode}} replacement_guest

powf-replacement-wasm-host:
    cargo build --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --features wasm-host --bin replacement_wasm_host

powf-replacement-matrix-analyze directory cells *runs:
    python3 -B benchmarks/powf_module_2026-11-07/analyze_replacement_matrix.py benchmarks/powf_module_2026-11-07/target/release/replacement_analyze "{{directory}}" {{runs}} --require-cells {{cells}}

powf-replacement-codegen-audit assembly output *options:
    python3 -B benchmarks/powf_module_2026-11-07/codegen_audit.py "{{assembly}}" "{{output}}" {{options}}

wasmtime-ci-installer-check:
    bash -n xtask/install-wasmtime-ci.sh
    bash -n xtask/install-semver-ci.sh
    python3 -B xtask/test_wasmtime_installer.py

powf-policy-codegen:
    cargo rustc --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --bin policy_codegen -- --emit=asm

powf-policy-matrix directory *options:
    python3 -B benchmarks/powf_module_2026-11-07/replacement_matrix.py benchmarks/powf_module_2026-11-07/target/release/policy_bench "{{directory}}" --policies {{options}}

powf-policy-matrix-analyze directory *runs:
    python3 -B benchmarks/powf_module_2026-11-07/analyze_replacement_matrix.py benchmarks/powf_module_2026-11-07/target/release/policy_analyze "{{directory}}" {{runs}} --require-cells 56

powf-policy-codegen-audit assembly output *options:
    python3 -B benchmarks/powf_module_2026-11-07/policy_codegen_audit.py "{{assembly}}" "{{output}}" {{options}}

powf-highp-accuracy fixtures output *options:
    python3 -B benchmarks/powf_module_2026-11-07/highp_accuracy.py benchmarks/powf_module_2026-11-07/target/release/highp_eval "{{fixtures}}" "{{output}}" {{options}}

powf-policy-cross directory fixtures sde wasm_runner="wasmtime":
    bash benchmarks/powf_module_2026-11-07/policy_cross.sh "{{directory}}" "{{fixtures}}" "{{sde}}" "{{wasm_runner}}"

powf-policy-avx512-build:
    cargo build --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --features avx512 --bin policy_widths

powf-policy-avx512-check fixtures sde:
    {{sde}} -icx -- benchmarks/powf_module_2026-11-07/target/release/policy_widths v4 < "{{fixtures}}"
    {{sde}} -icx -- benchmarks/powf_module_2026-11-07/target/release/policy_widths v4x < "{{fixtures}}"

powf-policy-miri-highp:
    cargo +$(cat xtask/miri-nightly.txt) miri test -p magemath --features avx512 --lib powf::tests::std_highp_full_domain_and_restricted_masks

powf-highp-cross directory fixtures sde wasm_runner="wasmtime":
    bash benchmarks/powf_module_2026-11-07/highp_cross.sh "{{directory}}" "{{fixtures}}" "{{sde}}" "{{wasm_runner}}"

# Preserve default linker/engine stack limits while exercising debug policy fanout.
powf-policy-debug-wasm:
    cargo test -p magetypes --features 'std avx512' --target wasm32-wasip1

powf-policy-codegen-identity before after output:
    python3 -B benchmarks/powf_module_2026-11-07/policy_identity.py "{{before}}" "{{after}}" "{{output}}"

powf-providers-build:
    cargo build --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --bin providers_bench --bin providers_eval --bin providers_analyze

powf-providers-matrix directory *options:
    python3 -B benchmarks/powf_module_2026-11-07/replacement_matrix.py benchmarks/powf_module_2026-11-07/target/release/providers_bench "{{directory}}" --providers {{options}}

powf-providers-analyze directory *runs:
    python3 -B benchmarks/powf_module_2026-11-07/analyze_replacement_matrix.py benchmarks/powf_module_2026-11-07/target/release/providers_analyze "{{directory}}" {{runs}} --require-cells 56

powf-providers-accuracy fixtures output tier *options:
    python3 -B benchmarks/powf_module_2026-11-07/providers_accuracy.py benchmarks/powf_module_2026-11-07/target/release/providers_eval "{{fixtures}}" "{{output}}" --tier {{tier}} {{options}}

powf-providers-compare output *actuals:
    python3 -B benchmarks/powf_module_2026-11-07/providers_compare.py "{{output}}" {{actuals}}

powf-providers-trace trace output:
    python3 -B benchmarks/powf_module_2026-11-07/providers_trace.py "{{trace}}" "{{output}}"

powf-external-build:
    cargo +nightly-2026-10-06 build --release --locked --manifest-path benchmarks/powf_providers_2026-11-09/Cargo.toml

powf-external-native-build output:
    c++ -std=c++17 -O3 -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off -pthread benchmarks/powf_providers_2026-11-09/native_worker.cpp -o "{{output}}" $(if test "$(uname -s)" = Linux; then echo -ldl; fi)

powf-providers-cross directory fixtures sde wasm_runner="wasmtime":
    bash benchmarks/powf_module_2026-11-07/providers_cross.sh "{{directory}}" "{{fixtures}}" "{{sde}}" "{{wasm_runner}}"

powf-external-discover rust native output *options:
    python3 -B benchmarks/powf_providers_2026-11-09/external.py discover "{{rust}}" "{{native}}" "{{output}}" {{options}}

powf-external-accuracy manifest fixtures directory *options:
    python3 -B benchmarks/powf_providers_2026-11-09/accuracy.py "{{manifest}}" "{{fixtures}}" "{{directory}}" {{options}}

powf-external-traces manifest directory:
    python3 -B benchmarks/powf_providers_2026-11-09/traces.py "{{manifest}}" "{{directory}}"

powf-external-matrix manifest directory *options:
    python3 -B benchmarks/powf_providers_2026-11-09/external.py run benchmarks/powf_providers_2026-11-09/target/release/external_providers_bench "{{manifest}}" "{{directory}}" {{options}}

powf-external-verify directory fixtures tier="baseline":
    bash benchmarks/powf_providers_2026-11-09/verify.sh "{{directory}}" "{{fixtures}}" "{{tier}}"

powf-external-format:
    cargo +nightly-2026-10-06 fmt --manifest-path benchmarks/powf_providers_2026-11-09/Cargo.toml

powf-external-analyze *runs:
    cargo +nightly-2026-10-06 run --release --locked --manifest-path benchmarks/powf_providers_2026-11-09/Cargo.toml --bin external_providers_analyze -- {{runs}}

powf-external-inventory directory *runs:
    python3 -B benchmarks/powf_providers_2026-11-09/inventory.py "{{directory}}" {{runs}}

powf-policy-cold-build base patched directory toolchain:
    python3 -B benchmarks/powf_providers_2026-11-09/cold_build.py "{{base}}" "{{patched}}" "{{directory}}" --toolchain "{{toolchain}}"

powf-core-math-prepare crate directory:
    python3 -B benchmarks/powf_providers_2026-11-09/core_math_prepare.py "{{crate}}" "{{directory}}"

powf-glibc-history repository directory:
    python3 -B benchmarks/powf_providers_2026-11-09/glibc_history.py "{{repository}}" "{{directory}}"

powf-core-math-manifest rust native output:
    python3 -B benchmarks/powf_providers_2026-11-09/core_math_manifest.py "{{rust}}" "{{native}}" "{{output}}"

powf-core-worst crate directory *options:
    python3 -B benchmarks/powf_providers_2026-11-09/core_worst.py "{{crate}}" "{{directory}}" {{options}}

powf-core-emulation manifest fixtures directory:
    python3 -B benchmarks/powf_providers_2026-11-09/core_emulation.py "{{manifest}}" "{{fixtures}}" "{{directory}}"

powf-glibc-binding output:
    cc -std=c11 -O2 -Wall -Wextra -Werror benchmarks/powf_providers_2026-11-09/glibc_binding.c -ldl -o "{{output}}"

powf-external-helpers native directory:
    python3 -B benchmarks/powf_providers_2026-11-09/verify_helpers.py "{{native}}" "{{directory}}"

powf-core-trace trace *options:
    python3 -B benchmarks/powf_providers_2026-11-09/core_trace.py "{{trace}}" {{options}}

powf-provider-artifacts mode directory manifest:
    python3 -B benchmarks/powf_providers_2026-11-09/artifact_manifest.py {{mode}} "{{directory}}" "{{manifest}}"

powf-arm-c-prepare repository directory *options:
    python3 -B benchmarks/powf_providers_2026-11-09/arm_powf_prepare.py "{{repository}}" "{{directory}}" {{options}}

powf-arm-c-manifest rust native build output *options:
    python3 -B benchmarks/powf_providers_2026-11-09/arm_c_manifest.py "{{rust}}" "{{native}}" "{{build}}" "{{output}}" {{options}}

powf-arm-c-flags manifest fixtures directory:
    python3 -B benchmarks/powf_providers_2026-11-09/arm_c_flags.py "{{manifest}}" "{{fixtures}}" "{{directory}}"

powf-arm-data build output:
    python3 -B benchmarks/powf_providers_2026-11-09/arm_data.py "{{build}}" "{{output}}"

powf-arm-port-manifest c_manifest rust output *options:
    python3 -B benchmarks/powf_providers_2026-11-09/arm_port_manifest.py "{{c_manifest}}" "{{rust}}" "{{output}}" {{options}}

powf-arm-port-asm function *options:
    cargo +nightly-2026-10-06 asm --manifest-path benchmarks/powf_providers_2026-11-09/Cargo.toml --bin arm_codegen --locked --no-color --asm {{options}} "{{function}}"

powf-arm-profile manifest directory *methods:
    python3 -B benchmarks/powf_providers_2026-11-09/arm_profile.py "{{manifest}}" "{{directory}}" {{methods}}

powf-arm-frozen-asm binary function:
    cargo +nightly-2026-10-06 asm --file "{{binary}}" --disasm --no-color "{{function}}"

powf-arm-contexts reference contexts directory:
    python3 -B benchmarks/powf_providers_2026-11-09/arm_context_manifest.py "{{reference}}" "{{contexts}}" "{{directory}}"

powf-arm-eval-test:
    cargo +nightly-2026-10-06 test --release --locked --manifest-path benchmarks/powf_providers_2026-11-09/Cargo.toml --bin arm_eval

# Canonical HIGHP graph oracle and physical/cross evaluator.
powf-highp-oracle *options:
    python3 -B xtask/highp_powf_vectors.py {{options}}

powf-highp-eval-build:
    cargo build --release --locked --manifest-path benchmarks/powf_validation/Cargo.toml --bin powf_highp_eval

powf-highp-test:
    cargo test -p magemath --test powf_highp
    cargo test -p magemath --no-default-features --test powf_highp

powf-highp-verify expected directory command *options:
    python3 -B benchmarks/powf_providers_2026-11-09/highp_verify.py "{{expected}}" "{{directory}}" --command '{{command}}' {{options}}

powf-highp-manifest worker output source:
    python3 -B benchmarks/powf_providers_2026-11-09/highp_manifest.py "{{worker}}" "{{output}}" --source "{{source}}"

powf-canonical-highp-cross directory expected wasmtime:
    bash benchmarks/powf_providers_2026-11-09/highp_cross.sh "{{directory}}" "{{expected}}" "{{wasmtime}}"

powf-canonical-highp-fusion-modes directory expected modules wasmtime:
    bash benchmarks/powf_providers_2026-11-09/highp_fusion_modes.sh "{{directory}}" "{{expected}}" "{{modules}}" "{{wasmtime}}"

powf-canonical-highp-fusion-probe directory wasmtime:
    bash benchmarks/powf_providers_2026-11-09/highp_fusion_probe.sh "{{directory}}" "{{wasmtime}}"

powf-canonical-highp-nan-build:
    cargo build --release --locked --manifest-path benchmarks/powf_validation/Cargo.toml --bin powf_highp_nan_scan

powf-canonical-highp-nan-scan binary context:
    "{{binary}}" "{{context}}"

powf-canonical-highp-bars directory output:
    python3 -B benchmarks/powf_providers_2026-11-09/highp_bars.py "{{directory}}" "{{output}}"

# Bounded replacement matrix with token-preserving tails and literal flags.
powf-midlow-build:
    cargo build --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --features avx512 --bin midlow_bench

powf-midlow-verify binary:
    MIDLOW_VERIFY=1 {{binary}}

powf-midlow-matrix binary directory *options:
    python3 -B benchmarks/powf_module_2026-11-07/midlow_matrix.py "{{binary}}" "{{directory}}" {{options}}

powf-midlow-analyze binary *files:
    {{binary}} {{files}}

powf-midlow-cross directory:
    bash benchmarks/powf_module_2026-11-07/midlow_cross.sh "{{directory}}"

powf-midlow-emulated directory:
    bash benchmarks/powf_module_2026-11-07/midlow_emulated.sh "{{directory}}"

powf-midlow-codegen binary directory:
    python3 -B benchmarks/powf_module_2026-11-07/midlow_codegen.py "{{binary}}" "{{directory}}"

powf-midlow-bars directory output:
    python3 -B benchmarks/powf_module_2026-11-07/midlow_bars.py "{{directory}}" "{{output}}"

powf-midlow-assembly-compare before after output *options:
    python3 -B benchmarks/powf_module_2026-11-07/midlow_assembly_compare.py "{{before}}" "{{after}}" "{{output}}" {{options}}

powf-remap-probes:
    cargo rustc --release --locked --manifest-path benchmarks/powf_module_2026-11-07/Cargo.toml --features avx512 --bin remap_codegen -- --emit=asm

powf-remap-codegen assembly output *options:
    python3 -B benchmarks/powf_module_2026-11-07/remap_codegen_audit.py "{{assembly}}" "{{output}}" {{options}}

# Full numerical replay is explicit; ordinary cargo test retains spot checks.
powf-test-exhaustive:
    cargo test -p magemath --features _exhaustive-tests,avx512 --lib powf::
    cargo test -p magemath --features _exhaustive-tests,avx512 --test powf_tails_portable_midp --test powf_tails_portable_lowp --test powf_tails_backend_midp --test powf_tails_backend_lowp

# Focused ordinary-test tier; no test-name filter leaks into integration tests.
powf-test-spot:
    cargo test -p magemath --lib powf::
    cargo test -p magemath --test powf_tails_portable_midp --test powf_tails_portable_lowp --test powf_tails_backend_midp --test powf_tails_backend_lowp

# Directed before/after exceptional-value ledger; no timing loop.
powf-legacy-edge-variance:
    cargo run --release --locked --manifest-path benchmarks/powf_validation/Cargo.toml --bin legacy_edge_variance

# Machine-facing powf validators are deliberately outside the examples/workspace.
powf-validation-build:
    cargo build --release --locked --manifest-path benchmarks/powf_validation/Cargo.toml --all-features --bins

powf-tutorial:
    cargo run -p magemath --features std --example bounded_powf

powf-validation-check:
    cargo clippy --locked --manifest-path benchmarks/powf_validation/Cargo.toml --all-features --all-targets -- -D warnings
    cargo clippy --locked --manifest-path benchmarks/powf_validation/Cargo.toml --no-default-features --features w512 --all-targets -- -D warnings

# Finite validation for the focused configurable-math migration.
magemath-check:
    cargo test -p magemath --features "std avx512"
    cargo test -p magemath --no-default-features --features w512
    cargo test -p magemath --release --features "std avx512" --test contracts
    cargo clippy -p magemath --all-targets --features "std avx512" -- -D warnings
    cargo test -p magemath --test expf_logf_contracts
    cargo fmt -p magemath -p magetypes -p xtask -- --check

magemath-check-wasm:
    RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime cargo test -p magemath --features "std w512" --target wasm32-wasip1
    RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime cargo test -p magemath --no-default-features --features w512 --target wasm32-wasip1


# Deterministic MPFR log1p fixture, including Miri's approximate std math shim.
magemath-miri-log1p:
    cargo +nightly-2026-10-06 miri test -p magemath --features avx512 --lib logf::tests::log1p_basics

magemath-compat-clippy:
    cargo clippy --locked --manifest-path benchmarks/powf_validation/Cargo.toml --all-features --all-targets -- -D warnings
    cargo clippy --locked --manifest-path benchmarks/powf_validation/Cargo.toml --no-default-features --features w512 --all-targets -- -D warnings

magemath-legacy-remap-wasm:
    RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime cargo test -p magetypes --features "std avx512" --target wasm32-wasip1 --test powf_engine_remap -- --nocapture

# Published method arithmetic stays in Magetypes; configured math belongs above it.
check-published-math:
    python3 -B xtask/check_published_math.py

# Verify the published implementation/API boundary and the remaining legacy calls.
magemath-public-boundary:
    python3 -B xtask/check_published_math.py
    cargo test -p magetypes --features "std avx512" --test powf_engine_remap --test generic_transcendentals
    cargo test -p magetypes --features "std avx512" --doc
    cargo clippy -p magetypes -p magemath --features "std avx512" --all-targets -- -D warnings

# Scalar sqrt routes and retained portable software controls.
sqrt-check:
    python3 -B xtask/sqrt_vectors.py --check
    cargo test -p magetypes --features "std avx512" --test sqrt_portability --test precise_reciprocals
    cargo test -p magetypes --no-default-features --features w512 --test sqrt_portability
    cargo clippy -p magetypes --all-targets --features "std avx512" -- -D warnings

sqrt-bench *ARGS:
    ARCHMAGE_SQRT_MEASURE=1 ZENBENCH_LAUNCHER_PIDS=$$ cargo run --release --manifest-path benchmarks/sqrt_2026-11-10/Cargo.toml -- {{ARGS}}

# Explicit opt-in, constant-memory comparison of all f32 input patterns.
sqrt-check-exhaustive:
    ARCHMAGE_SQRT_F32_STRIDE=1 cargo test --release -p magetypes --test sqrt_portability f32_stream_matches_hardware_and_retained_software -- --nocapture

sqrt-check-wasm:
    RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime cargo test -p magetypes --target wasm32-wasip1 --features "std w512" --test sqrt_portability
    RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime cargo test -p magetypes --target wasm32-wasip1 --no-default-features --features w512 --test sqrt_portability

sqrt-codegen:
    cargo rustc --release --manifest-path benchmarks/sqrt_2026-11-10/Cargo.toml --bin archmage-sqrt-study -- --emit=asm
    cargo rustc -p magetypes --release --no-default-features --features w512 --test sqrt_portability -- --emit=asm

sqrt-bench-check:
    cargo run --manifest-path benchmarks/sqrt_2026-11-10/Cargo.toml
    cargo clippy --manifest-path benchmarks/sqrt_2026-11-10/Cargo.toml --all-targets -- -D warnings

# Explicit token-disabling coverage; normal cargo test retains sqrt spot checks.
sqrt-token-testing:
    cargo test -p magetypes --no-default-features --features "w512 archmage/testable_dispatch" --test sqrt_token_testing

wasm-selection-check:
    bash xtask/test_wasm_selection.sh

# Ordinary no_std assembly must not contain token loads or software sqrt calls.
sqrt-selection-codegen target="x86_64-unknown-linux-gnu" flags="":
    RUSTFLAGS="{{flags}}" cargo rustc --manifest-path benchmarks/sqrt_2026-11-10/Cargo.toml --release --lib --no-default-features --features w512 --target {{target}} -- --emit=asm

sqrt-check-exhaustive-no-std:
    ARCHMAGE_SQRT_F32_STRIDE=1 cargo test --release -p magetypes --no-default-features --features w512 --test sqrt_portability f32_stream_matches_hardware_and_retained_software -- --nocapture

# Focused cross-target checks, without broadening into unrelated WASM test lints.
sqrt-selection-validate:
    cargo fmt -p xtask -p magetypes -p archmage --check
    just generate
    just validate-tokens
    just soundness
    cargo clippy -p magetypes --lib --test sqrt_portability --test sqrt_token_testing --no-default-features --features "w512 archmage/testable_dispatch" -- -D warnings
    RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd" cargo clippy --target wasm32-wasip1 -p magetypes --lib --test sqrt_portability --test sqrt_token_testing --test fused_arithmetic --no-default-features --features "w512 archmage/testable_dispatch" -- -D warnings
    RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd" cargo clippy --target wasm32-wasip1 -p archmage --test wasm_token_testing --features testable_dispatch -- -D warnings
    just api-doc
    just codegen-diff bd83f034 WORKTREE

workflow-contract-check:
    python3 -m unittest xtask/check_ci_gate.py


sqrt-token-miri:
    cargo +$(tr -d '[:space:]' < xtask/miri-nightly.txt) miri test -p magetypes --no-default-features --features "w512 archmage/testable_dispatch" --test sqrt_token_testing

primitive-generate:
    python3 -B xtask/primitive_api.py
    cargo fmt -p magemath

primitive-check:
    cargo test -p magemath --features "std avx512" --test primitive_contracts
    cargo test -p magemath --no-default-features --features w512 --test primitive_contracts

primitive-type-check:
    cargo check -p magemath --features "std avx512"
    RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd" cargo check -p magemath --target wasm32-wasip1 --features "std w512"

primitive-verify:
    python3 -B xtask/primitive_api.py --check
    python3 -B xtask/reciprocal_vectors.py --check
    cargo clippy -p magemath --lib --test primitive_contracts --features "std avx512" -- -D warnings
    just primitive-check

primitive-cross:
    RUSTFLAGS="-Ctarget-feature=+simd128" CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime cargo test -p magemath --target wasm32-wasip1 --features "std w512" --test primitive_contracts
    RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd" CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime cargo test -p magemath --target wasm32-wasip1 --features "std w512" --test primitive_contracts
    RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd" CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime cargo test -p magemath --target wasm32-wasip1 --no-default-features --features w512 --test primitive_contracts
    cargo check -p magemath --target aarch64-unknown-linux-gnu --features "std w512"

primitive-study-build:
    cargo build --release --manifest-path benchmarks/primitive_variants_2026-11-10/Cargo.toml --bin archmage-primitive-study

primitive-study-check:
    cargo run --manifest-path benchmarks/primitive_variants_2026-11-10/Cargo.toml --bin archmage-primitive-study
    cargo clippy --manifest-path benchmarks/primitive_variants_2026-11-10/Cargo.toml --all-targets -- -D warnings

primitive-linear-inventory:
    cargo run --release --manifest-path benchmarks/primitive_variants_2026-11-10/Cargo.toml --bin linear_inventory

primitive-linear-original checkout output:
    python3 -B benchmarks/primitive_variants_2026-11-10/linear_original.py "{{checkout}}" "{{output}}"

primitive-codegen target="x86_64-unknown-linux-gnu" flags="":
    RUSTFLAGS="{{flags}}" cargo rustc --release --lib --manifest-path benchmarks/primitive_variants_2026-11-10/Cargo.toml --target {{target}} -- --emit=asm

primitive-wasm-token-testing:
    RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd" CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --env ARCHMAGE_EXPECT_TESTABLE_DISPATCH=yes" cargo test -p magemath --features "std w512 archmage/testable_dispatch" --target wasm32-wasip1 --lib primitive_wasm::tests::

# Opt-in full mantissa coverage in two exponent parity classes; constant memory.
primitive-accuracy:
    cargo run --release --manifest-path benchmarks/primitive_variants_2026-11-10/Cargo.toml --bin accuracy -- --exhaustive

# Fusion provider selection: test both engine choices and disable the relaxed token.
primitive-recip-fusion-wasm:
    RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd" CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --env ARCHMAGE_EXPECT_RELAXED_FUSION=yes --env ARCHMAGE_EXPECT_TESTABLE_DISPATCH=yes" cargo test -p magemath --features "std w512 archmage/testable_dispatch" --target wasm32-wasip1 --lib recip::tests::
    RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd" CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime -W relaxed-simd-deterministic=n -C cranelift-has-fma=false --env ARCHMAGE_EXPECT_RELAXED_FUSION=no --env ARCHMAGE_EXPECT_TESTABLE_DISPATCH=yes" cargo test -p magemath --features "std w512 archmage/testable_dispatch" --target wasm32-wasip1 --lib recip::tests::
    RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd" CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime -W relaxed-simd-deterministic=n -C cranelift-has-fma=false" cargo test -p magemath --features "std w512" --target wasm32-wasip1 --test primitive_contracts

primitive-fusion-analyze results:
    python3 -B benchmarks/primitive_variants_2026-11-10/fusion_analysis.py "{{results}}"

# Benchmark-only correction graphs; no public profile is changed.
primitive-seed-study *ARGS:
    cargo run --release --manifest-path benchmarks/primitive_variants_2026-11-10/Cargo.toml --bin seed_study -- {{ARGS}}

primitive-correction-analyze results:
    python3 -B benchmarks/primitive_variants_2026-11-10/correction_analysis.py "{{results}}"

# Resource admission diagnostic only: no timings and no baseline/promotion.
primitive-gate-check:
    ZENBENCH_GATE_TRACE=1 cargo run --release --manifest-path benchmarks/primitive_variants_2026-11-10/Cargo.toml --bin archmage-primitive-study -- --gate-check

primitive-rcp14-reference output:
    mkdir -p "{{output}}"
    cc -O3 -std=c11 -fno-strict-aliasing -ffp-contract=off -fno-fast-math benchmarks/primitive_variants_2026-11-10/rcp14_reference.c -lm -o "{{output}}/rcp14-reference"
    "{{output}}/rcp14-reference"

primitive-recip-reference-data output:
    mkdir -p "{{output}}"
    cc -O3 -std=c11 -fno-strict-aliasing -ffp-contract=off -fno-fast-math benchmarks/primitive_variants_2026-11-10/rcp14_reference.c -lm -o "{{output}}/rcp14-reference"
    "{{output}}/rcp14-reference" --emit "{{output}}/recip-reference.u32le"
    sha256sum "{{output}}/recip-reference.u32le"

primitive-recip-qualify reference tier *ARGS:
    cargo run --release --manifest-path benchmarks/primitive_variants_2026-11-10/Cargo.toml --bin recip_qualification -- --reference "{{reference}}" --tier {{tier}} {{ARGS}}

primitive-recip-qualify-wasm reference density="--dense":
    bash benchmarks/primitive_variants_2026-11-10/qualify_wasm.sh "{{reference}}" "{{density}}"

primitive-recip-qualify-perf binary tier:
    python3 -B benchmarks/primitive_variants_2026-11-10/run_local.py "{{binary}}" --recip-qualification --qual-tier "{{tier}}"

primitive-recip-qualification-analyze results:
    python3 -B benchmarks/primitive_variants_2026-11-10/qualification_analysis.py "{{results}}"

primitive-recip-qualification-process-analyze results:
    python3 -B benchmarks/primitive_variants_2026-11-10/qualification_process_analysis.py "{{results}}"

# Preserve the upstream oracle byte-for-byte; lint authored whitespace separately.
primitive-study-source-check base:
    python3 -c 'import hashlib,pathlib; assert hashlib.sha256(pathlib.Path("benchmarks/primitive_variants_2026-11-10/reference/RECIP14.c").read_bytes()).hexdigest() == "4cb551b434f65928e97b062b1a4718809c955da6ab8d650f19593c72679b21e9"'
    git diff "{{base}}" --check -- . ':!benchmarks/primitive_variants_2026-11-10/reference/RECIP14.c'

primitive-recip-round-reference output:
    mkdir -p "{{output}}"
    cc -O3 -std=c11 -ffp-contract=off -fno-fast-math benchmarks/primitive_variants_2026-11-10/recip_round_reference.c -lm -o "{{output}}/recip-round-reference"
    "{{output}}/recip-round-reference" "{{output}}/recip-round-reference.u32le"
    sha256sum "{{output}}/recip-round-reference.u32le"

primitive-recip-round-check reference tier *ARGS:
    cargo run --release --manifest-path benchmarks/primitive_variants_2026-11-10/Cargo.toml --bin recip_round_check -- --reference "{{reference}}" --tier {{tier}} {{ARGS}}

primitive-recip-round-wasm reference density="--dense":
    bash benchmarks/primitive_variants_2026-11-10/qualify_wasm.sh "{{reference}}" "{{density}}" recip_round_check

primitive-recip-guest output mode:
    python3 -B benchmarks/primitive_variants_2026-11-10/build_guest.py "{{output}}" {{mode}}

primitive-recip-wasm-host-build:
    cargo build --release --manifest-path benchmarks/primitive_variants_2026-11-10/Cargo.toml --features wasm-host --bin recip_wasm_host

primitive-recip-round-audit reference output:
    cc -O3 -std=c11 -ffp-contract=off -fno-fast-math benchmarks/primitive_variants_2026-11-10/recip_round_reference.c -lm -o "{{output}}/recip-round-audit"
    "{{output}}/recip-round-audit" --audit
    python3 -B benchmarks/primitive_variants_2026-11-10/oracle_spotcheck.py "{{reference}}"

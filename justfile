# Archmage build and test commands

# Default: run all tests
default: test

# Run all tests (excludes sleef which requires nightly)
test:
    cargo test --features "std avx512"

# Run tests with all features (requires nightly for sleef)
test-nightly:
    cargo +nightly test --all-features

# Run clippy (excludes sleef which requires nightly)
lint:
    cargo clippy --features "std avx512" -- -D warnings

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

# Build the documentation site (Zola)
docs:
    cd docs/site && zola build -o ../../target/site

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

# Compare cold downstream compilation; output directories must be new.
attune-compile out:
    python3 benchmarks/attune_compile.py --out {{out}}

# Unified macro definition/call contracts.
attune-test:
    cargo test --test attune --test attune_selection --test attune_attributes

# Compile the same unified contracts for every portable backend.
attune-cross-check:
    cargo check --target aarch64-unknown-linux-gnu --test attune --test attune_selection --test attune_attributes
    cargo check --target wasm32-unknown-unknown --test attune --test attune_selection --test attune_attributes
    cargo check --target i686-unknown-linux-gnu --test attune --test attune_selection --test attune_attributes

# Compare pinned source archives, alternating cold and consumer-only checks.
attune-compare baseline candidate out pairs="6" *args:
    python3 benchmarks/attune_compare.py --baseline {{baseline}} --candidate {{candidate}} --out {{out}} --pairs {{pairs}} {{args}}

# Include legacy expansion and proof-boundary regressions after engine changes.
attune-compat:
    cargo test -p archmage --test attune --test attune_selection --test attune_attributes --test attune_inline --test attune_conventions --test arcane_sibling_resolution --test soundness_exploits --test macro_expand

attune-unit:
    cargo test -p archmage-macros

# Package suites and doctests after shared macro-engine changes.
attune-packages:
    cargo test -p archmage -p magetypes --features "std avx512"

attune-clippy:
    cargo clippy -p archmage-macros --all-targets --all-features -- -D warnings

# Standalone allocation instrumentation; confirm speed in real consumers.
attune-profile:
    cargo test -p archmage-macros --lib profile_allocations -- --ignored --nocapture

# Isolated full dependency-stack comparison; prepare first, then run serially.
consumer-compile out mode *args:
    python3 benchmarks/consumer_compile.py --out {{out}} {{mode}} {{args}}

# Convention inference and dispatcher additions with the optional tier gate off/on.
attune-conventions:
    cargo test --test attune_conventions --no-default-features
    cargo test --test attune_conventions --features avx512

# Structured definition grammar and selector-local policies, gate off/on.
attune-parser:
    cargo test -p archmage-macros
    cargo test --test attune_structured_syntax --no-default-features
    cargo test --test attune_structured_syntax --features avx512

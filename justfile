# Archmage build and test commands

# Default: run all tests
default: test

# Run all tests (excludes sleef which requires nightly)
test:
    cargo test --features "std macros avx512"

# Run tests with all features (requires nightly for sleef)
test-nightly:
    cargo +nightly test --all-features

# Run clippy (excludes sleef which requires nightly)
lint:
    cargo clippy --features "std macros avx512" -- -D warnings

# Format code
fmt:
    cargo fmt

# Check formatting
fmt-check:
    cargo fmt -- --check

# Run Miri tests (token logic only, no SIMD) - legacy alias
miri-tokens:
    rustup run nightly cargo miri test --test miri_safe --all-features

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

# Validate token-registry.toml (parse + structural checks)
validate-registry:
    cargo run -p xtask -- validate-registry

# Validate magetypes safety + try_new() feature checks against registry
validate-tokens:
    cargo run -p xtask -- validate

# Check API parity across x86/ARM/WASM architectures
parity:
    cargo run -p xtask -- parity

# Run ALL CI checks (MUST pass before push or publish)
ci:
    cargo run -p xtask -- ci

# Alias for ci (all-inclusive check)
all: ci

# ============================================================================
# Parity tests (cross-architecture + polyfill vs native)
# ============================================================================

# Run parity tests on x86_64 (native)
test-parity-x86:
    cargo test -p magetypes --test cross_arch_parity --test polyfill_parity --features "std avx512"

# Run parity tests on aarch64 (via QEMU/cross)
test-parity-arm:
    cross test -p magetypes --test cross_arch_parity --target aarch64-unknown-linux-gnu

# Run parity tests on WASM (via wasmtime)
test-parity-wasm:
    RUSTFLAGS="-C target-feature=+simd128" cargo test -p magetypes --test cross_arch_parity --target wasm32-wasip1

# Run polyfill parity tests (x86 only, compares polyfill vs native)
test-parity-polyfill:
    cargo test -p magetypes --test polyfill_parity --features "std avx512"

# Run all parity tests (x86 + ARM + WASM + polyfill)
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

# Test on aarch64 (via QEMU) - lib and cross-platform tests only
test-aarch64:
    cross test --lib --target aarch64-unknown-linux-gnu
    cross test --test miri_safe --test feature_consistency --target aarch64-unknown-linux-gnu

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

# Build the documentation book
docs:
    cd docs/book && mdbook build

# Serve the documentation locally (with auto-reload)
docs-serve:
    cd docs/book && mdbook serve

# Clean the built documentation
docs-clean:
    rm -rf target/book target/simd_reference

# Build the SIMD reference book
simd-ref:
    cd docs/simd_reference && mdbook build

# Serve the SIMD reference locally (with auto-reload)
simd-ref-serve:
    cd docs/simd_reference && mdbook serve

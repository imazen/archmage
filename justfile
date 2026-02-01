# Archmage build and test commands

# Default: run all tests
default: test

# Run all tests (excludes sleef which requires nightly)
test:
    cargo test --features "std macros bytemuck wide __composite avx512"

# Run tests with all features (requires nightly for sleef)
test-nightly:
    cargo +nightly test --all-features

# Run clippy (excludes sleef which requires nightly)
lint:
    cargo clippy --features "std macros bytemuck wide __composite avx512" -- -D warnings

# Format code
fmt:
    cargo fmt

# Check formatting
fmt-check:
    cargo fmt -- --check

# Run Miri tests (token logic only, no SIMD)
miri:
    rustup run nightly cargo miri test --test miri_safe --all-features

# Regenerate all generated code (SIMD types, macro registry, docs)
generate:
    cargo run -p xtask -- generate

# Validate token-registry.toml (parse + structural checks)
validate-registry:
    cargo run -p xtask -- validate-registry

# Validate magetypes safety + try_new() feature checks against registry
validate-tokens:
    cargo run -p xtask -- validate

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

# Full CI check (no SDE)
ci: fmt-check lint test miri
    @echo "All CI checks passed!"

# Full validation with SDE (for local development)
validate: ci test-all-cpus
    @echo "Full validation complete!"

# Full validation including cross-compilation
validate-all: ci test-cross clippy-all
    @echo "Full cross-platform validation complete!"

# ============================================================================
# Benchmarking (requires -C target-cpu=native for accurate results)
# ============================================================================

# Run all benchmarks with native CPU optimizations
bench:
    RUSTFLAGS="-C target-cpu=native" cargo bench --features wide

# Run wide comparison benchmark (quick mode)
bench-wide:
    RUSTFLAGS="-C target-cpu=native" cargo bench --bench wide_comparison --features wide -- --quick

# Run wide comparison benchmark (full)
bench-wide-full:
    RUSTFLAGS="-C target-cpu=native" cargo bench --bench wide_comparison --features wide

# Run transcendental benchmarks
bench-transcendental:
    RUSTFLAGS="-C target-cpu=native" cargo bench --bench transcendental_accuracy

# Run edge case benchmarks
bench-edge-cases:
    RUSTFLAGS="-C target-cpu=native" cargo bench --bench edge_case_perf

# IMPORTANT: Without -C target-cpu=native, intrinsics won't inline properly
# and benchmarks will show archmage being 4-5x slower than wide.
# With native CPU targeting, archmage is 1.2-1.4x faster than wide.

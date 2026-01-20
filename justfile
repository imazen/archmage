# Archmage build and test commands

# Default: run all tests
default: test

# Run all tests
test:
    cargo test --all-features

# Run clippy
lint:
    cargo clippy --all-features -- -D warnings

# Format code
fmt:
    cargo fmt

# Check formatting
fmt-check:
    cargo fmt -- --check

# Run Miri tests (token logic only, no SIMD)
miri:
    rustup run nightly cargo miri test --test miri_safe --all-features

# Regenerate safe_unaligned_simd wrappers
generate:
    cargo run -p xtask -- generate

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
# CI-style comprehensive test
# ============================================================================

# Full CI check (no SDE)
ci: fmt-check lint test miri
    @echo "All CI checks passed!"

# Full validation with SDE (for local development)
validate: ci test-all-cpus
    @echo "Full validation complete!"

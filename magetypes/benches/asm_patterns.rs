//! ASM pattern verification for slice→SIMD load patterns.
//!
//! Each function is `#[unsafe(no_mangle)]` + `#[arcane]` so `cargo asm` can find it.
//! Verified by `scripts/verify-asm.sh`.
//!
//! Run: cargo asm -p archmage --bench asm_patterns --features "std avx512"

// x86-only bench: stub main so the `harness = false` target still links on
// other architectures (a crate-level `#![cfg]` would leave the bench with no
// `main` at all).
#[cfg(not(target_arch = "x86_64"))]
fn main() {}

#[cfg(target_arch = "x86_64")]
mod x86_impl {

    use archmage::{Desktop64, SimdToken, arcane};
    use std::arch::x86_64::*;

    // ============================================================================
    // 256-bit float loads
    // ============================================================================

    /// Baseline: load from array reference → vmovups
    #[unsafe(no_mangle)]
    #[arcane(import_intrinsics)]
    fn load_array_ref(_t: Desktop64, data: &[f32; 8]) -> __m256 {
        _mm256_loadu_ps(data)
    }

    /// Slice via .first_chunk() → should produce same vmovups
    #[unsafe(no_mangle)]
    #[arcane(import_intrinsics)]
    fn load_first_chunk_256(_t: Desktop64, data: &[f32]) -> __m256 {
        let arr: &[f32; 8] = data.first_chunk().unwrap();
        _mm256_loadu_ps(arr)
    }

    /// Slice via try_into → should produce same vmovups
    #[unsafe(no_mangle)]
    #[arcane(import_intrinsics)]
    fn load_try_into(_t: Desktop64, data: &[f32]) -> __m256 {
        let arr: &[f32; 8] = data[..8].try_into().unwrap();
        _mm256_loadu_ps(arr)
    }

    // ============================================================================
    // 256-bit integer load
    // ============================================================================

    /// Integer load via first_chunk → vmovdqu
    #[unsafe(no_mangle)]
    #[arcane(import_intrinsics)]
    fn load_first_chunk_i(_t: Desktop64, data: &[u8]) -> __m256i {
        let arr: &[u8; 32] = data.first_chunk().unwrap();
        _mm256_loadu_si256(arr)
    }

    // ============================================================================
    // 128-bit float load
    // ============================================================================

    /// 128-bit first_chunk → vmovups (128-bit)
    #[unsafe(no_mangle)]
    #[arcane(import_intrinsics)]
    fn load_first_chunk_128(_t: Desktop64, data: &[f32]) -> __m128 {
        let arr: &[f32; 4] = data.first_chunk().unwrap();
        _mm_loadu_ps(arr)
    }

    // ============================================================================
    // Store patterns
    // ============================================================================

    /// Store via first_chunk_mut → vmovups (store)
    #[unsafe(no_mangle)]
    #[arcane(import_intrinsics)]
    fn store_first_chunk_mut(_t: Desktop64, v: __m256, out: &mut [f32]) {
        let arr: &mut [f32; 8] = out.first_chunk_mut().unwrap();
        _mm256_storeu_ps(arr, v);
    }

    // ============================================================================
    // Magetypes patterns
    // ============================================================================

    /// magetypes from_slice → should produce vmovups
    #[unsafe(no_mangle)]
    #[arcane(import_intrinsics)]
    fn load_f32x8_from_slice(_t: Desktop64, data: &[f32]) -> __m256 {
        use magetypes::simd::f32x8;
        let v = f32x8::from_slice(_t, data);
        v.raw()
    }

    /// magetypes load via first_chunk → should produce vmovups
    #[unsafe(no_mangle)]
    #[arcane(import_intrinsics)]
    fn load_f32x8_first_chunk(_t: Desktop64, data: &[f32]) -> __m256 {
        use magetypes::simd::f32x8;
        let arr: &[f32; 8] = data.first_chunk().unwrap();
        let v = f32x8::load(_t, arr);
        v.raw()
    }

    // ============================================================================
    // Criterion benchmark (required for cargo asm --bench to work)
    // ============================================================================

    use zenbench::criterion_compat::*;
    use zenbench::{criterion_group, criterion_main};

    fn bench_load_patterns(c: &mut Criterion) {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let slice: &[f32] = &data;
        let bytes = [0u8; 32];
        let byte_slice: &[u8] = &bytes;

        if let Some(token) = Desktop64::summon() {
            c.bench_function("load_array_ref", |b| {
                b.iter(|| load_array_ref(token, black_box(&data)))
            });

            c.bench_function("load_first_chunk_256", |b| {
                b.iter(|| load_first_chunk_256(token, black_box(slice)))
            });

            c.bench_function("load_try_into", |b| {
                b.iter(|| load_try_into(token, black_box(slice)))
            });

            c.bench_function("load_first_chunk_i", |b| {
                b.iter(|| load_first_chunk_i(token, black_box(byte_slice)))
            });

            c.bench_function("load_first_chunk_128", |b| {
                b.iter(|| load_first_chunk_128(token, black_box(slice)))
            });

            c.bench_function("load_f32x8_from_slice", |b| {
                b.iter(|| load_f32x8_from_slice(token, black_box(slice)))
            });

            c.bench_function("load_f32x8_first_chunk", |b| {
                b.iter(|| load_f32x8_first_chunk(token, black_box(slice)))
            });
        } else {
            eprintln!("Desktop64 not available, skipping benchmarks");
        }
    }

    // ========================================================================
    // concat_shift: the funnel shift must reach the native instruction
    // ========================================================================
    //
    // The backend trait carries a portable lane-gather default and every ISA
    // with a funnel shift overrides it. If an override is ever deleted, moved,
    // or shadowed by a delegation that forgets to forward it, the code stays
    // CORRECT and silently loses 5-6 instructions per call. Nothing else in the
    // suite would notice — which is what these two exist to catch.
    //
    // Measured 2026-09-08: the default body is 6 ops (f32x8) / 7 ops (f32x16)
    // of scalar element moves; the native forms below are 2 and 1.

    /// AVX2 f32x8: expect `vperm2f128`/`vperm2i128` + `vpalignr`.
    #[unsafe(no_mangle)]
    #[arcane(import_intrinsics)]
    fn concat_shift_f32x8_v3(t: archmage::X64V3Token, lo: __m256, hi: __m256) -> __m256 {
        use magetypes::simd::backends::F32x8Backend;
        <archmage::X64V3Token as F32x8Backend>::concat_shift::<1>(t, lo, hi)
    }

    /// AVX-512 f32x16: expect a single `valignd`.
    #[cfg(feature = "avx512")]
    #[unsafe(no_mangle)]
    #[arcane(import_intrinsics)]
    fn concat_shift_f32x16_v4x(t: archmage::X64V4xToken, lo: __m512, hi: __m512) -> __m512 {
        use magetypes::simd::backends::F32x16Backend;
        <archmage::X64V4xToken as F32x16Backend>::concat_shift::<1>(t, lo, hi)
    }

    /// SSSE3 u8x16: a byte-granular shift, where the immediate IS the lane
    /// count. The integer types are the ones that most often get hand-rolled
    /// per-arch, so they are worth a gate of their own.
    #[unsafe(no_mangle)]
    #[arcane(import_intrinsics)]
    fn concat_shift_u8x16_v3(t: archmage::X64V3Token, lo: __m128i, hi: __m128i) -> __m128i {
        use magetypes::simd::backends::U8x16Backend;
        <archmage::X64V3Token as U8x16Backend>::concat_shift::<3>(t, lo, hi)
    }

    /// AVX-512 i16x32 at a shift that crosses a 128-bit lane. This is the one
    /// case with no single full-width instruction — `vpalignr` is per-lane at
    /// every width — so it is built from two `valignd` windows plus a per-lane
    /// `vpalignr`. If that decomposition ever regresses to the gather, this is
    /// what notices.
    #[cfg(feature = "avx512")]
    #[unsafe(no_mangle)]
    #[arcane(import_intrinsics)]
    fn concat_shift_i16x32_v4x(t: archmage::X64V4xToken, lo: __m512i, hi: __m512i) -> __m512i {
        use magetypes::simd::backends::I16x32Backend;
        <archmage::X64V4xToken as I16x32Backend>::concat_shift::<9>(t, lo, hi)
    }

    criterion_group!(benches, bench_load_patterns);
    criterion_main!(benches);

    /// Entry point for the crate-level `main` below.
    pub fn run() {
        main()
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    x86_impl::run()
}

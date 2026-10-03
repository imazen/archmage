//! Differential benchmark for the x86 pixel-pack paths: `to_u8` and the
//! RGBA stores on `f32x4` / `f32x8`. Results and the decisions they drove:
//! `benchmarks/pixel_pack_zen5-9950x3d_2026-10-03.md`.
//!
//! The 0.9.29 AVX2 form (`cvtps` then saturating packs) returned 0 for +inf
//! and values at or above 2^31, because `cvtps` turns them into `i32::MIN`.
//! What ships now:
//!
//! - **AVX2 (V3):** `min(255.0, x)` before `cvtps`. Compared with the
//!   replaced, wrong form ("unclamped"), a clamp-both + `pshufb` variant and
//!   the scalar default.
//! - **AVX-512 (V4):** `max(x, 0)`, `cvtps`, then `vpmovusdb`, whose unsigned
//!   saturation turns `cvtps`'s overflow value into 255. Used for `to_u8` at
//!   both widths and for `f32x4::store_4_rgba_u8` (two planes per 256-bit
//!   register). Compared with the V3 form they replaced. `f32x8`'s RGBA store
//!   stays on the V3 form: its only faster variant needs 512-bit registers,
//!   which an `f32x8` operation does not otherwise touch.
//!
//! Every kernel streams an L1-resident buffer inside one `#[arcane]` region,
//! so each form compiles the way real callers get it, and zenbench runs the
//! forms of one group interleaved, in shuffled order every round. Before
//! measuring, the setup checks that every correct form matches the scalar
//! reference on the edge values and the benchmark data.
//!
//! Run on x86_64 (no `-Ctarget-cpu=native`; bench what users get):
//!   cargo bench -p magetypes --bench pixel_pack --features avx512
//! Redraw code layout once (Zen 5 placement effects) and compare:
//!   RUSTFLAGS="--cfg pixel_pack_redraw" cargo bench -p magetypes --bench pixel_pack --features avx512

#[cfg(target_arch = "x86_64")]
mod kernels {
    use archmage::{ScalarToken, X64V3Token, arcane};
    use magetypes::simd::generic::f32x8;

    /// Shifts the layout of everything after it when built with
    /// `--cfg pixel_pack_redraw`, so a second run draws different code placement.
    #[cfg(pixel_pack_redraw)]
    #[inline(never)]
    pub fn layout_pad(x: u64) -> u64 {
        let mut h = x;
        for i in 0..64u64 {
            h = h.rotate_left(7) ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
        h
    }

    // ---------------- to_u8 ----------------

    #[arcane]
    pub fn to_u8_shipped(t: X64V3Token, src: &[f32], dst: &mut [u8]) {
        for (i, o) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
            o.copy_from_slice(&f32x8::<X64V3Token>::from_array_t(t, i.try_into().unwrap()).to_u8());
        }
    }

    #[arcane(import_intrinsics)]
    pub fn to_u8_unclamped(_t: X64V3Token, src: &[f32], dst: &mut [u8]) {
        for (i, o) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
            let a = _mm256_loadu_ps(<&[f32; 8]>::try_from(i).unwrap());
            let i32s = _mm256_cvtps_epi32(a);
            let lo = _mm256_castsi256_si128(i32s);
            let hi = _mm256_extracti128_si256::<1>(i32s);
            let u8s = _mm_packus_epi16(_mm_packs_epi32(lo, hi), _mm_setzero_si128());
            o.copy_from_slice(&(_mm_cvtsi128_si64(u8s) as u64).to_ne_bytes());
        }
    }

    #[arcane(import_intrinsics)]
    pub fn to_u8_clamp_both_pshufb(_t: X64V3Token, src: &[f32], dst: &mut [u8]) {
        let zero = _mm256_setzero_ps();
        let max = _mm256_set1_ps(255.0);
        // Byte 0 of each dword, per 128-bit lane, into that lane's low 4 bytes.
        let pick = _mm256_setr_epi8(
            0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, //
            0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        for (i, o) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
            let a = _mm256_loadu_ps(<&[f32; 8]>::try_from(i).unwrap());
            // max(NaN, 0) returns 0 (maxps returns its second operand on NaN).
            let c = _mm256_min_ps(_mm256_max_ps(a, zero), max);
            let b = _mm256_shuffle_epi8(_mm256_cvtps_epi32(c), pick);
            let packed =
                _mm_unpacklo_epi32(_mm256_castsi256_si128(b), _mm256_extracti128_si256::<1>(b));
            o.copy_from_slice(&(_mm_cvtsi128_si64(packed) as u64).to_ne_bytes());
        }
    }

    /// The generic per-lane default, compiled inside an AVX2 region.
    #[arcane]
    pub fn to_u8_scalar_default(_t: X64V3Token, src: &[f32], dst: &mut [u8]) {
        for (i, o) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
            o.copy_from_slice(
                &f32x8::<ScalarToken>::from_array_t(ScalarToken, i.try_into().unwrap()).to_u8(),
            );
        }
    }

    // ---------------- f32x4 (128-bit) ----------------

    use magetypes::simd::generic::f32x4;

    #[arcane]
    pub fn to_u8_x4_shipped(t: X64V3Token, src: &[f32], dst: &mut [u8]) {
        for (i, o) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
            o.copy_from_slice(&f32x4::<X64V3Token>::from_array_t(t, i.try_into().unwrap()).to_u8());
        }
    }

    #[arcane(import_intrinsics)]
    pub fn to_u8_x4_unclamped(_t: X64V3Token, src: &[f32], dst: &mut [u8]) {
        for (i, o) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
            let i32s = _mm_cvtps_epi32(_mm_loadu_ps(<&[f32; 4]>::try_from(i).unwrap()));
            let i16s = _mm_packs_epi32(i32s, i32s);
            let u8s = _mm_packus_epi16(i16s, i16s);
            o.copy_from_slice(&(_mm_cvtsi128_si32(u8s) as u32).to_ne_bytes());
        }
    }

    #[arcane]
    pub fn to_u8_x4_scalar_default(_t: X64V3Token, src: &[f32], dst: &mut [u8]) {
        for (i, o) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
            o.copy_from_slice(
                &f32x4::<ScalarToken>::from_array_t(ScalarToken, i.try_into().unwrap()).to_u8(),
            );
        }
    }

    fn planes4(src: &[f32]) -> impl Iterator<Item = [[f32; 4]; 4]> + '_ {
        src.chunks_exact(16)
            .map(|c| core::array::from_fn(|p| <[f32; 4]>::try_from(&c[p * 4..p * 4 + 4]).unwrap()))
    }

    #[arcane]
    pub fn rgba4_shipped(t: X64V3Token, src: &[f32], dst: &mut [u8]) {
        for (p, o) in planes4(src).zip(dst.chunks_exact_mut(16)) {
            let v = |i: usize| f32x4::<X64V3Token>::from_array_t(t, p[i]);
            o.copy_from_slice(&f32x4::<X64V3Token>::store_4_rgba_u8(
                v(0),
                v(1),
                v(2),
                v(3),
            ));
        }
    }

    #[arcane(import_intrinsics)]
    pub fn rgba4_unclamped(_t: X64V3Token, src: &[f32], dst: &mut [u8]) {
        let shuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        for (p, o) in planes4(src).zip(dst.chunks_exact_mut(16)) {
            let cvt = |i: usize| _mm_cvtps_epi32(_mm_loadu_ps(&p[i]));
            let rg = _mm_packs_epi32(cvt(0), cvt(1));
            let ba = _mm_packs_epi32(cvt(2), cvt(3));
            let out = _mm_shuffle_epi8(_mm_packus_epi16(rg, ba), shuf);
            _mm_storeu_si128(<&mut [u8; 16]>::try_from(o).unwrap(), out);
        }
    }

    #[arcane]
    pub fn rgba4_scalar_default(_t: X64V3Token, src: &[f32], dst: &mut [u8]) {
        for (p, o) in planes4(src).zip(dst.chunks_exact_mut(16)) {
            let v = |i: usize| f32x4::<ScalarToken>::from_array_t(ScalarToken, p[i]);
            o.copy_from_slice(&f32x4::<ScalarToken>::store_4_rgba_u8(
                v(0),
                v(1),
                v(2),
                v(3),
            ));
        }
    }

    // ---------------- store_8_rgba_u8 ----------------

    fn planes(src: &[f32]) -> impl Iterator<Item = [[f32; 8]; 4]> + '_ {
        src.chunks_exact(32)
            .map(|c| core::array::from_fn(|p| <[f32; 8]>::try_from(&c[p * 8..p * 8 + 8]).unwrap()))
    }

    #[arcane]
    pub fn rgba_shipped(t: X64V3Token, src: &[f32], dst: &mut [u8]) {
        for (p, o) in planes(src).zip(dst.chunks_exact_mut(32)) {
            let v = |i: usize| f32x8::<X64V3Token>::from_array_t(t, p[i]);
            o.copy_from_slice(&f32x8::<X64V3Token>::store_8_rgba_u8(
                v(0),
                v(1),
                v(2),
                v(3),
            ));
        }
    }

    #[arcane(import_intrinsics)]
    pub fn rgba_unclamped(_t: X64V3Token, src: &[f32], dst: &mut [u8]) {
        let shuf = _mm256_setr_epi8(
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, //
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        );
        for (p, o) in planes(src).zip(dst.chunks_exact_mut(32)) {
            let cvt = |i: usize| _mm256_cvtps_epi32(_mm256_loadu_ps(&p[i]));
            let rg = _mm256_packs_epi32(cvt(0), cvt(1));
            let ba = _mm256_packs_epi32(cvt(2), cvt(3));
            let out = _mm256_shuffle_epi8(_mm256_packus_epi16(rg, ba), shuf);
            _mm256_storeu_si256(<&mut [u8; 32]>::try_from(o).unwrap(), out);
        }
    }

    #[arcane]
    pub fn rgba_scalar_default(_t: X64V3Token, src: &[f32], dst: &mut [u8]) {
        for (p, o) in planes(src).zip(dst.chunks_exact_mut(32)) {
            let v = |i: usize| f32x8::<ScalarToken>::from_array_t(ScalarToken, p[i]);
            o.copy_from_slice(&f32x8::<ScalarToken>::store_8_rgba_u8(
                v(0),
                v(1),
                v(2),
                v(3),
            ));
        }
    }

    #[cfg(feature = "avx512")]
    pub mod v4 {
        use super::planes;
        use archmage::{ScalarToken, X64V3Token, X64V4Token, arcane};
        use magetypes::simd::generic::f32x8;

        #[arcane]
        pub fn to_u8_shipped(t: X64V4Token, src: &[f32], dst: &mut [u8]) {
            for (i, o) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
                o.copy_from_slice(
                    &f32x8::<X64V4Token>::from_array_t(t, i.try_into().unwrap()).to_u8(),
                );
            }
        }

        /// AVX-512VL: clamp to [0, 255], convert, truncate each dword to a byte.
        #[arcane(import_intrinsics)]
        pub fn to_u8_clamp_both_vpmovdb(_t: X64V4Token, src: &[f32], dst: &mut [u8]) {
            let zero = _mm256_setzero_ps();
            let max = _mm256_set1_ps(255.0);
            for (i, o) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
                let a = _mm256_loadu_ps(<&[f32; 8]>::try_from(i).unwrap());
                let c = _mm256_min_ps(_mm256_max_ps(a, zero), max);
                let b = _mm256_cvtepi32_epi8(_mm256_cvtps_epi32(c));
                o.copy_from_slice(&(_mm_cvtsi128_si64(b) as u64).to_ne_bytes());
            }
        }

        /// 256-bit: `max(x, 0)` + `cvtps` + `vpmovusdb` per plane, then
        /// byte/word unpacks into RGBA order.
        #[arcane(import_intrinsics)]
        pub fn rgba_max_vpmovusdb_256(_t: X64V4Token, src: &[f32], dst: &mut [u8]) {
            let zero = _mm256_setzero_ps();
            for (p, o) in planes(src).zip(dst.chunks_exact_mut(32)) {
                let n = |i: usize| {
                    _mm256_cvtusepi32_epi8(_mm256_cvtps_epi32(_mm256_max_ps(
                        _mm256_loadu_ps(&p[i]),
                        zero,
                    )))
                };
                let rg = _mm_unpacklo_epi8(n(0), n(1));
                let ba = _mm_unpacklo_epi8(n(2), n(3));
                let out = _mm256_set_m128i(_mm_unpackhi_epi16(rg, ba), _mm_unpacklo_epi16(rg, ba));
                _mm256_storeu_si256(<&mut [u8; 32]>::try_from(o).unwrap(), out);
            }
        }

        /// 512-bit: two planes per register, so one `max` and one `cvtps` cover
        /// two planes; `vpmovusdb` narrows 16 lanes, `pshufb` pairs the planes.
        #[arcane(import_intrinsics)]
        pub fn rgba_max_vpmovusdb_512(_t: X64V4Token, src: &[f32], dst: &mut [u8]) {
            let zero = _mm512_setzero_ps();
            // [x0..x7, y0..y7] -> [x0 y0 x1 y1 ... x7 y7]
            let pair = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
            for (p, o) in planes(src).zip(dst.chunks_exact_mut(32)) {
                let two = |i: usize, j: usize| {
                    let v = _mm512_insertf32x8::<1>(
                        _mm512_castps256_ps512(_mm256_loadu_ps(&p[i])),
                        _mm256_loadu_ps(&p[j]),
                    );
                    let b = _mm512_cvtusepi32_epi8(_mm512_cvtps_epi32(_mm512_max_ps(v, zero)));
                    _mm_shuffle_epi8(b, pair)
                };
                let rg = two(0, 1);
                let ba = two(2, 3);
                let out = _mm256_set_m128i(_mm_unpackhi_epi16(rg, ba), _mm_unpacklo_epi16(rg, ba));
                _mm256_storeu_si256(<&mut [u8; 32]>::try_from(o).unwrap(), out);
            }
        }

        use magetypes::simd::generic::f32x4;

        #[arcane]
        pub fn to_u8_x4_shipped(t: X64V4Token, src: &[f32], dst: &mut [u8]) {
            for (i, o) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
                o.copy_from_slice(
                    &f32x4::<X64V4Token>::from_array_t(t, i.try_into().unwrap()).to_u8(),
                );
            }
        }

        #[arcane]
        pub fn rgba4_shipped(t: X64V4Token, src: &[f32], dst: &mut [u8]) {
            for (p, o) in super::planes4(src).zip(dst.chunks_exact_mut(16)) {
                let v = |i: usize| f32x4::<X64V4Token>::from_array_t(t, p[i]);
                o.copy_from_slice(&f32x4::<X64V4Token>::store_4_rgba_u8(
                    v(0),
                    v(1),
                    v(2),
                    v(3),
                ));
            }
        }

        // The V3 forms these overrides replaced, run inside the same V4 region.

        #[arcane]
        pub fn to_u8_v3_form(t: X64V4Token, src: &[f32], dst: &mut [u8]) {
            let t = t.v3();
            for (i, o) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
                o.copy_from_slice(
                    &f32x8::<X64V3Token>::from_array_t(t, i.try_into().unwrap()).to_u8(),
                );
            }
        }

        #[arcane]
        pub fn to_u8_x4_v3_form(t: X64V4Token, src: &[f32], dst: &mut [u8]) {
            let t = t.v3();
            for (i, o) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
                o.copy_from_slice(
                    &f32x4::<X64V3Token>::from_array_t(t, i.try_into().unwrap()).to_u8(),
                );
            }
        }

        #[arcane]
        pub fn rgba4_v3_form(t: X64V4Token, src: &[f32], dst: &mut [u8]) {
            let t = t.v3();
            for (p, o) in super::planes4(src).zip(dst.chunks_exact_mut(16)) {
                let v = |i: usize| f32x4::<X64V3Token>::from_array_t(t, p[i]);
                o.copy_from_slice(&f32x4::<X64V3Token>::store_4_rgba_u8(
                    v(0),
                    v(1),
                    v(2),
                    v(3),
                ));
            }
        }

        #[arcane]
        pub fn to_u8_scalar_default(_t: X64V4Token, src: &[f32], dst: &mut [u8]) {
            for (i, o) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
                o.copy_from_slice(
                    &f32x8::<ScalarToken>::from_array_t(ScalarToken, i.try_into().unwrap()).to_u8(),
                );
            }
        }

        #[arcane]
        pub fn rgba_shipped(t: X64V4Token, src: &[f32], dst: &mut [u8]) {
            for (p, o) in planes(src).zip(dst.chunks_exact_mut(32)) {
                let v = |i: usize| f32x8::<X64V4Token>::from_array_t(t, p[i]);
                o.copy_from_slice(&f32x8::<X64V4Token>::store_8_rgba_u8(
                    v(0),
                    v(1),
                    v(2),
                    v(3),
                ));
            }
        }

        #[arcane]
        pub fn rgba_scalar_default(_t: X64V4Token, src: &[f32], dst: &mut [u8]) {
            for (p, o) in planes(src).zip(dst.chunks_exact_mut(32)) {
                let v = |i: usize| f32x8::<ScalarToken>::from_array_t(ScalarToken, p[i]);
                o.copy_from_slice(&f32x8::<ScalarToken>::store_8_rgba_u8(
                    v(0),
                    v(1),
                    v(2),
                    v(3),
                ));
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn bench_pixel_pack(suite: &mut zenbench::prelude::Suite) {
    use archmage::{SimdToken, X64V3Token};
    use zenbench::prelude::*;

    /// f32 values per pass: 4096 for `to_u8` (4 KiB out), and 1024 RGBA pixels.
    const N: usize = 4096;

    let Some(v3) = X64V3Token::summon() else {
        eprintln!("skipped: this CPU lacks x86-64-v3");
        return;
    };
    #[cfg(pixel_pack_redraw)]
    black_box(kernels::layout_pad(black_box(N as u64)));

    // Pixel-like data with some out-of-range values, then the edge cases.
    let mut src: Vec<f32> = (0..N)
        .map(|i| ((i * 37) % 300) as f32 - 20.0 + (i % 7) as f32 * 0.25)
        .collect();
    let edges = [
        3.0e9,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        2_147_483_648.0,
        f32::MAX,
        -f32::MAX,
        255.5,
        254.5,
        -0.5,
        0.5,
        1.5,
    ];
    src[..edges.len()].copy_from_slice(&edges);

    // Correct forms must match the scalar reference byte for byte.
    let run = |f: fn(X64V3Token, &[f32], &mut [u8]), len: usize| {
        let mut out = vec![0u8; len];
        f(v3, &src, &mut out);
        out
    };
    let want = run(kernels::to_u8_scalar_default, N);
    assert_eq!(run(kernels::to_u8_shipped, N), want, "shipped to_u8");
    assert_eq!(
        run(kernels::to_u8_clamp_both_pshufb, N),
        want,
        "clamp-both to_u8"
    );
    assert_ne!(
        run(kernels::to_u8_unclamped, N),
        want,
        "unclamped differs on the edges"
    );
    let want_rgba = run(kernels::rgba_scalar_default, N);
    let want_x4 = run(kernels::to_u8_x4_scalar_default, N);
    assert_eq!(want_x4, want, "f32x4 and f32x8 references agree");
    assert_eq!(
        run(kernels::to_u8_x4_shipped, N),
        want,
        "shipped f32x4 to_u8"
    );
    let want_rgba4 = run(kernels::rgba4_scalar_default, N);
    assert_eq!(
        run(kernels::rgba4_shipped, N),
        want_rgba4,
        "shipped f32x4 rgba"
    );
    assert_eq!(run(kernels::rgba_shipped, N), want_rgba, "shipped rgba");

    macro_rules! case {
        ($g:expr, $name:expr, $tok:expr, $kernel:path, $len:expr) => {{
            let src = src.clone();
            let mut dst = vec![0u8; $len];
            let tok = $tok;
            $g.bench($name, move |b| {
                b.iter(|| {
                    $kernel(tok, black_box(&src), &mut dst);
                    black_box(dst[0])
                })
            });
        }};
    }

    suite.group("to_u8 f32x8, AVX2 (V3)", |g| {
        g.throughput(Throughput::Elements(N as u64));
        case!(
            g,
            "shipped: min(255) then cvtps",
            v3,
            kernels::to_u8_shipped,
            N
        );
        case!(
            g,
            "unclamped (replaced; wrong past 2^31)",
            v3,
            kernels::to_u8_unclamped,
            N
        );
        case!(
            g,
            "clamp both + pshufb",
            v3,
            kernels::to_u8_clamp_both_pshufb,
            N
        );
        case!(g, "scalar default", v3, kernels::to_u8_scalar_default, N);
        g.baseline("shipped: min(255) then cvtps");
    });

    suite.group("store_8_rgba_u8, AVX2 (V3)", |g| {
        g.throughput(Throughput::Elements((N / 4) as u64));
        case!(
            g,
            "shipped: min(255) then cvtps",
            v3,
            kernels::rgba_shipped,
            N
        );
        case!(
            g,
            "unclamped (replaced; wrong past 2^31)",
            v3,
            kernels::rgba_unclamped,
            N
        );
        case!(g, "scalar default", v3, kernels::rgba_scalar_default, N);
        g.baseline("shipped: min(255) then cvtps");
    });

    suite.group("to_u8 f32x4, AVX2 (V3)", |g| {
        g.throughput(Throughput::Elements(N as u64));
        case!(
            g,
            "shipped: min(255) then cvtps",
            v3,
            kernels::to_u8_x4_shipped,
            N
        );
        case!(
            g,
            "unclamped (replaced; wrong past 2^31)",
            v3,
            kernels::to_u8_x4_unclamped,
            N
        );
        g.baseline("shipped: min(255) then cvtps");
    });

    suite.group("store_4_rgba_u8, AVX2 (V3)", |g| {
        g.throughput(Throughput::Elements((N / 4) as u64));
        case!(
            g,
            "shipped: min(255) then cvtps",
            v3,
            kernels::rgba4_shipped,
            N
        );
        case!(
            g,
            "unclamped (replaced; wrong past 2^31)",
            v3,
            kernels::rgba4_unclamped,
            N
        );
        g.baseline("shipped: min(255) then cvtps");
    });

    #[cfg(feature = "avx512")]
    if let Some(v4) = archmage::X64V4Token::summon() {
        use kernels::v4;
        let run4 = |f: fn(archmage::X64V4Token, &[f32], &mut [u8]), len: usize| {
            let mut out = vec![0u8; len];
            f(v4, &src, &mut out);
            out
        };
        let checks: [(fn(archmage::X64V4Token, &[f32], &mut [u8]), &Vec<u8>, &str); 11] = [
            (v4::to_u8_shipped, &want, "shipped V4 f32x8 to_u8"),
            (v4::to_u8_v3_form, &want, "V3-form f32x8 to_u8"),
            (v4::to_u8_clamp_both_vpmovdb, &want, "vpmovdb f32x8 to_u8"),
            (v4::to_u8_scalar_default, &want, "scalar f32x8 to_u8"),
            (v4::to_u8_x4_shipped, &want, "shipped V4 f32x4 to_u8"),
            (v4::to_u8_x4_v3_form, &want, "V3-form f32x4 to_u8"),
            (v4::rgba4_shipped, &want_rgba4, "shipped V4 f32x4 rgba"),
            (v4::rgba4_v3_form, &want_rgba4, "V3-form f32x4 rgba"),
            (v4::rgba_shipped, &want_rgba, "shipped V4 f32x8 rgba"),
            (
                v4::rgba_max_vpmovusdb_256,
                &want_rgba,
                "vpmovusdb 256 f32x8 rgba",
            ),
            (
                v4::rgba_max_vpmovusdb_512,
                &want_rgba,
                "vpmovusdb 512 f32x8 rgba",
            ),
        ];
        for (f, expected, name) in checks {
            assert_eq!(&run4(f, N), expected, "{name}");
        }

        suite.group("to_u8 f32x8, AVX-512 (V4)", |g| {
            g.throughput(Throughput::Elements(N as u64));
            case!(g, "shipped: max(0) + vpmovusdb", v4, v4::to_u8_shipped, N);
            case!(
                g,
                "V3 form (delegated until this change)",
                v4,
                v4::to_u8_v3_form,
                N
            );
            case!(
                g,
                "clamp both + vpmovdb",
                v4,
                v4::to_u8_clamp_both_vpmovdb,
                N
            );
            case!(
                g,
                "scalar default (0.9.29 V4)",
                v4,
                v4::to_u8_scalar_default,
                N
            );
            g.baseline("shipped: max(0) + vpmovusdb");
        });

        suite.group("to_u8 f32x4, AVX-512 (V4)", |g| {
            g.throughput(Throughput::Elements(N as u64));
            case!(
                g,
                "shipped: max(0) + vpmovusdb",
                v4,
                v4::to_u8_x4_shipped,
                N
            );
            case!(
                g,
                "V3 form (delegated until this change)",
                v4,
                v4::to_u8_x4_v3_form,
                N
            );
            g.baseline("shipped: max(0) + vpmovusdb");
        });

        suite.group("store_4_rgba_u8, AVX-512 (V4)", |g| {
            g.throughput(Throughput::Elements((N / 4) as u64));
            case!(
                g,
                "shipped: plane pairs + vpmovusdb",
                v4,
                v4::rgba4_shipped,
                N
            );
            case!(
                g,
                "V3 form (delegated until this change)",
                v4,
                v4::rgba4_v3_form,
                N
            );
            g.baseline("shipped: plane pairs + vpmovusdb");
        });

        suite.group("store_8_rgba_u8, AVX-512 (V4)", |g| {
            g.throughput(Throughput::Elements((N / 4) as u64));
            case!(g, "shipped: V3 form, delegated", v4, v4::rgba_shipped, N);
            case!(
                g,
                "max(0) + vpmovusdb, 256-bit",
                v4,
                v4::rgba_max_vpmovusdb_256,
                N
            );
            case!(
                g,
                "512-bit plane pairs (not adopted: zmm)",
                v4,
                v4::rgba_max_vpmovusdb_512,
                N
            );
            case!(
                g,
                "scalar default (0.9.29 V4)",
                v4,
                v4::rgba_scalar_default,
                N
            );
            g.baseline("shipped: V3 form, delegated");
        });
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn bench_pixel_pack(_suite: &mut zenbench::prelude::Suite) {}

zenbench::main!(bench_pixel_pack);

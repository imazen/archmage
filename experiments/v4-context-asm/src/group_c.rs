//! Group C: hand-written 256-bit AVX2 intrinsics bodies, compiled in both
//! the V3 and the V4 feature context. Each body loops over slices.
use archmage::prelude::*;

// C1: _mm256_blendv_ps(a, b, _mm256_cmp_ps(x, y, LT_OQ))
#[cfg_attr(feature = "avx512", rite(v3, v4, import_intrinsics))]
#[cfg_attr(not(feature = "avx512"), rite(v3, neon, import_intrinsics))]
pub fn c1_body(a: &[f32], b: &[f32], x: &[f32], y: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(x.len()).min(y.len()).min(out.len());
    let (a, b, x, y, out) = (&a[..n], &b[..n], &x[..n], &y[..n], &mut out[..n]);
    let (ac, _) = a.as_chunks::<8>();
    let (bc, _) = b.as_chunks::<8>();
    let (xc, _) = x.as_chunks::<8>();
    let (yc, _) = y.as_chunks::<8>();
    let (oc, _) = out.as_chunks_mut::<8>();
    for ((((pa, pb), px), py), po) in ac.iter().zip(bc).zip(xc).zip(yc).zip(oc) {
        let m = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_loadu_ps(px), _mm256_loadu_ps(py));
        let r = _mm256_blendv_ps(_mm256_loadu_ps(pa), _mm256_loadu_ps(pb), m);
        _mm256_storeu_ps(po, r);
    }
}

// C2: or(and(a, m), andnot(m, b)) on __m256i
#[cfg_attr(feature = "avx512", rite(v3, v4, import_intrinsics))]
#[cfg_attr(not(feature = "avx512"), rite(v3, neon, import_intrinsics))]
pub fn c2_body(a: &[i32], b: &[i32], m: &[i32], out: &mut [i32]) {
    let n = a.len().min(b.len()).min(m.len()).min(out.len());
    let (a, b, m, out) = (&a[..n], &b[..n], &m[..n], &mut out[..n]);
    let (ac, _) = a.as_chunks::<8>();
    let (bc, _) = b.as_chunks::<8>();
    let (mc, _) = m.as_chunks::<8>();
    let (oc, _) = out.as_chunks_mut::<8>();
    for (((pa, pb), pm), po) in ac.iter().zip(bc).zip(mc).zip(oc) {
        let vm = _mm256_loadu_si256(pm);
        let r = _mm256_or_si256(
            _mm256_and_si256(_mm256_loadu_si256(pa), vm),
            _mm256_andnot_si256(vm, _mm256_loadu_si256(pb)),
        );
        _mm256_storeu_si256(po, r);
    }
}

// C3: signed 64-bit min as cmpgt_epi64 + blendv_epi8
#[cfg_attr(feature = "avx512", rite(v3, v4, import_intrinsics))]
#[cfg_attr(not(feature = "avx512"), rite(v3, neon, import_intrinsics))]
pub fn c3_body(a: &[i64], b: &[i64], out: &mut [i64]) {
    let n = a.len().min(b.len()).min(out.len());
    let (a, b, out) = (&a[..n], &b[..n], &mut out[..n]);
    let (ac, _) = a.as_chunks::<4>();
    let (bc, _) = b.as_chunks::<4>();
    let (oc, _) = out.as_chunks_mut::<4>();
    for ((pa, pb), po) in ac.iter().zip(bc).zip(oc) {
        let va = _mm256_loadu_si256(pa);
        let vb = _mm256_loadu_si256(pb);
        let gt = _mm256_cmpgt_epi64(va, vb);
        _mm256_storeu_si256(po, _mm256_blendv_epi8(va, vb, gt));
    }
}

// C4: 32-bit rotate-left by 7 as or(slli, srli)
#[cfg_attr(feature = "avx512", rite(v3, v4, import_intrinsics))]
#[cfg_attr(not(feature = "avx512"), rite(v3, neon, import_intrinsics))]
pub fn c4_body(a: &[u32], out: &mut [u32]) {
    let n = a.len().min(out.len());
    let (a, out) = (&a[..n], &mut out[..n]);
    let (ac, _) = a.as_chunks::<8>();
    let (oc, _) = out.as_chunks_mut::<8>();
    for (pa, po) in ac.iter().zip(oc) {
        let v = _mm256_loadu_si256(pa);
        _mm256_storeu_si256(
            po,
            _mm256_or_si256(_mm256_slli_epi32::<7>(v), _mm256_srli_epi32::<25>(v)),
        );
    }
}

// C5: 64-bit absolute value, AVX2 way: (x ^ s) - s with s = (0 > x)
#[cfg_attr(feature = "avx512", rite(v3, v4, import_intrinsics))]
#[cfg_attr(not(feature = "avx512"), rite(v3, neon, import_intrinsics))]
pub fn c5_body(a: &[i64], out: &mut [i64]) {
    let n = a.len().min(out.len());
    let (a, out) = (&a[..n], &mut out[..n]);
    let (ac, _) = a.as_chunks::<4>();
    let (oc, _) = out.as_chunks_mut::<4>();
    for (pa, po) in ac.iter().zip(oc) {
        let v = _mm256_loadu_si256(pa);
        let s = _mm256_cmpgt_epi64(_mm256_setzero_si256(), v);
        _mm256_storeu_si256(po, _mm256_sub_epi64(_mm256_xor_si256(v, s), s));
    }
}

// C6: unsigned a > b via sign-bias xor + cmpgt_epi32, then movemask_ps
#[cfg_attr(feature = "avx512", rite(v3, v4, import_intrinsics))]
#[cfg_attr(not(feature = "avx512"), rite(v3, neon, import_intrinsics))]
pub fn c6_body(a: &[u32], b: &[u32], out: &mut [u8]) {
    let n = a.len().min(b.len()).min(out.len() * 8);
    let (a, b) = (&a[..n], &b[..n]);
    let (ac, _) = a.as_chunks::<8>();
    let (bc, _) = b.as_chunks::<8>();
    let bias = _mm256_set1_epi32(i32::MIN);
    for ((pa, pb), po) in ac.iter().zip(bc).zip(out.iter_mut()) {
        let va = _mm256_xor_si256(_mm256_loadu_si256(pa), bias);
        let vb = _mm256_xor_si256(_mm256_loadu_si256(pb), bias);
        let gt = _mm256_cmpgt_epi32(va, vb);
        *po = _mm256_movemask_ps(_mm256_castsi256_ps(gt)) as u8;
    }
}

macro_rules! centry {
    ($n3:ident, $n4:ident, $b3:ident, $b4:ident, ($($p:ident: $t:ty),*)) => {
        #[inline(never)]
        pub fn $n3(token: X64V3Token, $($p: $t),*) {
            #[arcane(import_intrinsics)]
            fn inner(_token: X64V3Token, $($p: $t),*) {
                $b3($($p),*)
            }
            inner(token, $($p),*)
        }
        #[cfg(feature = "avx512")]
        #[inline(never)]
        pub fn $n4(token: X64V4Token, $($p: $t),*) {
            #[arcane(import_intrinsics)]
            fn inner(_token: X64V4Token, $($p: $t),*) {
                $b4($($p),*)
            }
            inner(token, $($p),*)
        }
    };
}

centry!(c1_v3, c1_v4, c1_body_v3, c1_body_v4, (a: &[f32], b: &[f32], x: &[f32], y: &[f32], out: &mut [f32]));
centry!(c2_v3, c2_v4, c2_body_v3, c2_body_v4, (a: &[i32], b: &[i32], m: &[i32], out: &mut [i32]));
centry!(c3_v3, c3_v4, c3_body_v3, c3_body_v4, (a: &[i64], b: &[i64], out: &mut [i64]));
centry!(c4_v3, c4_v4, c4_body_v3, c4_body_v4, (a: &[u32], out: &mut [u32]));
centry!(c5_v3, c5_v4, c5_body_v3, c5_body_v4, (a: &[i64], out: &mut [i64]));
centry!(c6_v3, c6_v4, c6_body_v3, c6_body_v4, (a: &[u32], b: &[u32], out: &mut [u8]));

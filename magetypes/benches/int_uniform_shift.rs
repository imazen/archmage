//! What the uniform (runtime-count) shift costs against the shift-by-constant
//! it replaces, and what saturating add/sub costs against wrapping add/sub.
//!
//! The motivating case is a CDEF-shaped kernel whose shift amount is
//! `max(damping - msb(strength), 0)` — a runtime scalar. Written against
//! `shl_const::<N>` that kernel has to be monomorphised over every reachable
//! `N` (81 instantiations in the SVT-AV1 port that prompted this work);
//! written against `shl_uniform(count)` it is one body. This bench measures
//! the price of that single body.
//!
//! `const/*`    — `shr_logical_const::<3>`, i.e. the monomorphised upper bound.
//! `uniform/*`  — `shr_logical_uniform(count)` with a `black_box`ed count, so
//!                the compiler cannot fold it back into an immediate.
//! `cdef/*`     — `saturating_sub(threshold, x >> shift)`, the whole clamp, in
//!                both forms.
//! `wrapping/*` vs `saturating/*` — the cost of the saturating opcode itself.
//!
//! Run: `cargo bench -p magetypes --bench int_uniform_shift`
//!
//! Sizes span tiny (call-overhead-dominated) through large
//! (memory-bandwidth-bound), because the per-call scalar work the uniform form
//! adds — one `min`, one lane splat — is loop-invariant and amortises away as
//! the slice grows. The tiny sizes are where it is visible at all.

use archmage::SimdToken;
use magetypes::simd::generic::u16x8;
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

/// u16 lane counts: tiny → large. 8 lanes per vector.
const SIZES: &[usize] = &[16, 256, 4096, 65536, 1_048_576];

fn input(n: usize) -> Vec<u16> {
    (0..n)
        .map(|i| (i.wrapping_mul(2_654_435_761) & 0xFFFF) as u16)
        .collect()
}

/// `shr_logical_const::<3>` over the slice — the monomorphised baseline.
fn shr_const_slice<T>(t: T, src: &[u16], dst: &mut [u16])
where
    T: magetypes::simd::backends::U16x8Backend,
{
    for (s, d) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
        let arr: [u16; 8] = s.try_into().unwrap();
        let v = u16x8::<T>::from_array(t, arr);
        d.copy_from_slice(&v.shr_logical_const::<3>().to_array());
    }
}

/// `shr_logical_uniform(count)` over the slice — one body, runtime count.
fn shr_uniform_slice<T>(t: T, src: &[u16], dst: &mut [u16], count: u32)
where
    T: magetypes::simd::backends::U16x8Backend,
{
    for (s, d) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
        let arr: [u16; 8] = s.try_into().unwrap();
        let v = u16x8::<T>::from_array(t, arr);
        d.copy_from_slice(&v.shr_logical_uniform(count).to_array());
    }
}

/// The CDEF clamp with a constant shift: `max(thr - (x >> 3), 0)`.
fn cdef_const_slice<T>(t: T, src: &[u16], dst: &mut [u16], thr: u16)
where
    T: magetypes::simd::backends::U16x8Backend,
{
    let vthr = u16x8::<T>::splat(t, thr);
    for (s, d) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
        let arr: [u16; 8] = s.try_into().unwrap();
        let v = u16x8::<T>::from_array(t, arr);
        d.copy_from_slice(&vthr.saturating_sub(v.shr_logical_const::<3>()).to_array());
    }
}

/// The same clamp with a runtime shift — the one-body form.
fn cdef_uniform_slice<T>(t: T, src: &[u16], dst: &mut [u16], thr: u16, count: u32)
where
    T: magetypes::simd::backends::U16x8Backend,
{
    let vthr = u16x8::<T>::splat(t, thr);
    for (s, d) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
        let arr: [u16; 8] = s.try_into().unwrap();
        let v = u16x8::<T>::from_array(t, arr);
        d.copy_from_slice(&vthr.saturating_sub(v.shr_logical_uniform(count)).to_array());
    }
}

fn wrapping_sub_slice<T>(t: T, src: &[u16], dst: &mut [u16], thr: u16)
where
    T: magetypes::simd::backends::U16x8Backend,
{
    let vthr = u16x8::<T>::splat(t, thr);
    for (s, d) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
        let arr: [u16; 8] = s.try_into().unwrap();
        let v = u16x8::<T>::from_array(t, arr);
        d.copy_from_slice(&(vthr - v).to_array());
    }
}

fn saturating_sub_slice<T>(t: T, src: &[u16], dst: &mut [u16], thr: u16)
where
    T: magetypes::simd::backends::U16x8Backend,
{
    let vthr = u16x8::<T>::splat(t, thr);
    for (s, d) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
        let arr: [u16; 8] = s.try_into().unwrap();
        let v = u16x8::<T>::from_array(t, arr);
        d.copy_from_slice(&vthr.saturating_sub(v).to_array());
    }
}

macro_rules! bench_backend {
    ($c:expr, $label:literal, $Tok:ty, $t:expr) => {{
        let t = $t;
        for &n in SIZES {
            let src = input(n);
            let mut dst = vec![0u16; n];
            $c.bench_function(&format!("const/{}/{n}", $label), |b| {
                b.iter(|| shr_const_slice::<$Tok>(t, black_box(&src), black_box(&mut dst)))
            });
            $c.bench_function(&format!("uniform/{}/{n}", $label), |b| {
                b.iter(|| {
                    shr_uniform_slice::<$Tok>(
                        t,
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(3u32),
                    )
                })
            });
            $c.bench_function(&format!("cdef_const/{}/{n}", $label), |b| {
                b.iter(|| {
                    cdef_const_slice::<$Tok>(t, black_box(&src), black_box(&mut dst), black_box(40))
                })
            });
            $c.bench_function(&format!("cdef_uniform/{}/{n}", $label), |b| {
                b.iter(|| {
                    cdef_uniform_slice::<$Tok>(
                        t,
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(40),
                        black_box(3u32),
                    )
                })
            });
            $c.bench_function(&format!("wrapping_sub/{}/{n}", $label), |b| {
                b.iter(|| {
                    wrapping_sub_slice::<$Tok>(
                        t,
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(40),
                    )
                })
            });
            $c.bench_function(&format!("saturating_sub/{}/{n}", $label), |b| {
                b.iter(|| {
                    saturating_sub_slice::<$Tok>(
                        t,
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(40),
                    )
                })
            });
        }
    }};
}

fn bench_all(c: &mut Criterion) {
    bench_backend!(
        c,
        "scalar",
        archmage::ScalarToken,
        archmage::ScalarToken::summon().unwrap()
    );

    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        bench_backend!(c, "neon", archmage::NeonToken, t);
    }

    #[cfg(target_arch = "x86_64")]
    if let Some(t) = archmage::X64V3Token::summon() {
        bench_backend!(c, "v3", archmage::X64V3Token, t);
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);

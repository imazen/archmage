//! Whole-slice cost of three ways to apply a magemath function:
//! - `array8`: `try_*_array_with::<_, 8>` per chunk (Automatic execution per call)
//! - `attune`: a caller-written `#[attune]` kernel using the vector entries
//! - `slice`: the `try_*_slice_with` entries
//!
//! Run: `cargo bench --bench slices` (no `-C target-cpu`).
use archmage::prelude::*;
use magemath::{expf as e, logf as l, powf as p};
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

const PA: u32 = p::PowfOptions::PORTABLE_GAMMA_MIDP.bits();
const PC: u32 = p::PowfOptions::PORTABLE_GAMMA_MIDP.with_execution(p::PowfExecution::Caller).bits();
const EA: u32 = e::ExpfOptions::PORTABLE_MIDP.bits();
const EC: u32 = e::ExpfOptions::PORTABLE_MIDP.with_execution(e::ExpfExecution::Caller).bits();
const LA: u32 = l::LogfOptions::PORTABLE_MIDP.bits();
const LC: u32 = l::LogfOptions::PORTABLE_MIDP.with_execution(l::LogfExecution::Caller).bits();

fn pow_array8(src: &[f32], dst: &mut [f32], n: f32) {
    let (sc, st) = src.as_chunks::<8>();
    let (dc, dt) = dst.as_chunks_mut::<8>();
    for (s, d) in sc.iter().zip(dc) {
        *d = p::try_powf_array_with::<PA, 8>(*s, n).unwrap();
    }
    for (s, d) in st.iter().zip(dt) {
        *d = p::try_powf_with::<PA>(*s, n).unwrap();
    }
}
fn exp2_array8(src: &[f32], dst: &mut [f32]) {
    let (sc, st) = src.as_chunks::<8>();
    let (dc, dt) = dst.as_chunks_mut::<8>();
    for (s, d) in sc.iter().zip(dc) {
        *d = e::try_exp2_array_with::<EA, 8>(*s).unwrap();
    }
    for (s, d) in st.iter().zip(dt) {
        *d = e::try_exp2_with::<EA>(*s).unwrap();
    }
}
fn log2_array8(src: &[f32], dst: &mut [f32]) {
    let (sc, st) = src.as_chunks::<8>();
    let (dc, dt) = dst.as_chunks_mut::<8>();
    for (s, d) in sc.iter().zip(dc) {
        *d = l::try_log2_array_with::<LA, 8>(*s).unwrap();
    }
    for (s, d) in st.iter().zip(dt) {
        *d = l::try_log2_with::<LA>(*s).unwrap();
    }
}

#[attune(_*_t, define(f32x8))]
fn pow_attune_impl(token: Token, src: &[f32], dst: &mut [f32], n: f32) {
    let (sc, st) = f32x8::partition_slice_t(token, src);
    let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        p::try_powf_f32x8_with::<PC, _>(token, f32x8::load_t(token, s), n).unwrap().store(d);
    }
    for (&x, y) in st.iter().zip(dt.iter_mut()) {
        *y = p::try_powf_with_token::<PC, _>(token, x, n).unwrap();
    }
}
#[attune(_*_t, define(f32x8))]
fn exp2_attune_impl(token: Token, src: &[f32], dst: &mut [f32]) {
    let (sc, st) = f32x8::partition_slice_t(token, src);
    let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        e::try_exp2_f32x8_with::<EC, _>(token, f32x8::load_t(token, s)).unwrap().store(d);
    }
    for (&x, y) in st.iter().zip(dt.iter_mut()) {
        *y = e::try_exp2_with_token::<EC, _>(token, x).unwrap();
    }
}
#[attune(_*_t, define(f32x8))]
fn log2_attune_impl(token: Token, src: &[f32], dst: &mut [f32]) {
    let (sc, st) = f32x8::partition_slice_t(token, src);
    let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        l::try_log2_f32x8_with::<LC, _>(token, f32x8::load_t(token, s)).unwrap().store(d);
    }
    for (&x, y) in st.iter().zip(dt.iter_mut()) {
        *y = l::try_log2_with_token::<LC, _>(token, x).unwrap();
    }
}

fn bench(c: &mut Criterion) {
    for len in [64usize, 4096] {
        let src: Vec<f32> = (0..len).map(|i| 0.01 + 3.9 * i as f32 / len as f32).collect();
        let mut dst = vec![0.0f32; len];
        let mut g = c.benchmark_group(format!("len{len}"));
        g.bench_function("pow_array8", |b| b.iter(|| pow_array8(black_box(&src), &mut dst, black_box(2.4))));
        g.bench_function("pow_attune", |b| b.iter(|| attuned!(pow_attune_impl(black_box(&src), &mut dst, black_box(2.4)))));
        g.bench_function("pow_slice", |b| b.iter(|| p::try_powf_slice_with::<PA>(black_box(&src), &mut dst, black_box(2.4)).unwrap()));
        g.bench_function("exp2_array8", |b| b.iter(|| exp2_array8(black_box(&src), &mut dst)));
        g.bench_function("exp2_attune", |b| b.iter(|| attuned!(exp2_attune_impl(black_box(&src), &mut dst))));
        g.bench_function("exp2_slice", |b| b.iter(|| e::try_exp2_slice_with::<EA>(black_box(&src), &mut dst).unwrap()));
        g.bench_function("log2_array8", |b| b.iter(|| log2_array8(black_box(&src), &mut dst)));
        g.bench_function("log2_attune", |b| b.iter(|| attuned!(log2_attune_impl(black_box(&src), &mut dst))));
        g.bench_function("log2_slice", |b| b.iter(|| l::try_log2_slice_with::<LA>(black_box(&src), &mut dst).unwrap()));
        g.finish();
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);

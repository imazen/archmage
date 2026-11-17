//! Whole-slice cost of three ways to apply a magemath function:
//! - `array8`: `Powf::<CFG>::array::<8>` per chunk (`Execution::Upgrade` per call)
//! - `attune`: a caller-written `#[attune]` kernel using `SimdMath` + `scalar_with_token`
//! - `slice`: the `Powf::<CFG>::slice` / `Expf::<CFG>::exp2_slice` / `Logf::<CFG>::log2_slice` entries
//!
//! Run: `cargo bench --bench slices` (no `-C target-cpu`).
use archmage::prelude::*;
use magemath::{
    SimdMath,
    expf::{Expf, ExpfOptions},
    logf::{Logf, LogfOptions},
    powf::{Powf, PowfOptions},
};
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

const PA: u128 = PowfOptions::new().gamma().midp().bits();
const PC: u128 = PowfOptions::new().gamma().midp().caller().bits();
const EA: u128 = ExpfOptions::new().midp().bits();
const EC: u128 = ExpfOptions::new().midp().caller().bits();
const LA: u128 = LogfOptions::new().midp().bits();
const LC: u128 = LogfOptions::new().midp().caller().bits();

fn pow_array8(src: &[f32], dst: &mut [f32], n: f32) {
    let (sc, st) = src.as_chunks::<8>();
    let (dc, dt) = dst.as_chunks_mut::<8>();
    for (s, d) in sc.iter().zip(dc) {
        *d = Powf::<PA>::array::<8>(*s, n).unwrap();
    }
    for (s, d) in st.iter().zip(dt) {
        *d = Powf::<PA>::scalar(*s, n).unwrap();
    }
}
fn exp2_array8(src: &[f32], dst: &mut [f32]) {
    let (sc, st) = src.as_chunks::<8>();
    let (dc, dt) = dst.as_chunks_mut::<8>();
    for (s, d) in sc.iter().zip(dc) {
        *d = Expf::<EA>::exp2_array::<8>(*s).unwrap();
    }
    for (s, d) in st.iter().zip(dt) {
        *d = Expf::<EA>::exp2(*s).unwrap();
    }
}
fn log2_array8(src: &[f32], dst: &mut [f32]) {
    let (sc, st) = src.as_chunks::<8>();
    let (dc, dt) = dst.as_chunks_mut::<8>();
    for (s, d) in sc.iter().zip(dc) {
        *d = Logf::<LA>::log2_array::<8>(*s).unwrap();
    }
    for (s, d) in st.iter().zip(dt) {
        *d = Logf::<LA>::log2(*s).unwrap();
    }
}

#[attune(_*_t, define(f32x8))]
fn pow_attune_impl(token: Token, src: &[f32], dst: &mut [f32], n: f32) {
    let (sc, st) = f32x8::partition_slice_t(token, src);
    let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        f32x8::load_t(token, s)
            .try_powf_with::<PC>(n)
            .unwrap()
            .store(d);
    }
    for (&x, y) in st.iter().zip(dt.iter_mut()) {
        *y = Powf::<PC>::scalar_with_token(token, x, n).unwrap();
    }
}
#[attune(_*_t, define(f32x8))]
fn exp2_attune_impl(token: Token, src: &[f32], dst: &mut [f32]) {
    let (sc, st) = f32x8::partition_slice_t(token, src);
    let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        f32x8::load_t(token, s)
            .try_exp2_with::<EC>()
            .unwrap()
            .store(d);
    }
    for (&x, y) in st.iter().zip(dt.iter_mut()) {
        *y = Expf::<EC>::exp2_with_token(token, x).unwrap();
    }
}
#[attune(_*_t, define(f32x8))]
fn log2_attune_impl(token: Token, src: &[f32], dst: &mut [f32]) {
    let (sc, st) = f32x8::partition_slice_t(token, src);
    let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        f32x8::load_t(token, s)
            .try_log2_with::<LC>()
            .unwrap()
            .store(d);
    }
    for (&x, y) in st.iter().zip(dt.iter_mut()) {
        *y = Logf::<LC>::log2_with_token(token, x).unwrap();
    }
}

fn bench(c: &mut Criterion) {
    for len in [64usize, 4096] {
        let src: Vec<f32> = (0..len)
            .map(|i| 0.01 + 3.9 * i as f32 / len as f32)
            .collect();
        let mut dst = vec![0.0f32; len];
        let mut g = c.benchmark_group(format!("len{len}"));
        g.bench_function("pow_array8", |b| {
            b.iter(|| pow_array8(black_box(&src), &mut dst, black_box(2.4)))
        });
        g.bench_function("pow_attune", |b| {
            b.iter(|| attuned!(pow_attune_impl(black_box(&src), &mut dst, black_box(2.4))))
        });
        g.bench_function("pow_slice", |b| {
            b.iter(|| Powf::<PA>::slice(black_box(&src), &mut dst, black_box(2.4)).unwrap())
        });
        g.bench_function("exp2_array8", |b| {
            b.iter(|| exp2_array8(black_box(&src), &mut dst))
        });
        g.bench_function("exp2_attune", |b| {
            b.iter(|| attuned!(exp2_attune_impl(black_box(&src), &mut dst)))
        });
        g.bench_function("exp2_slice", |b| {
            b.iter(|| Expf::<EA>::exp2_slice(black_box(&src), &mut dst).unwrap())
        });
        g.bench_function("log2_array8", |b| {
            b.iter(|| log2_array8(black_box(&src), &mut dst))
        });
        g.bench_function("log2_attune", |b| {
            b.iter(|| attuned!(log2_attune_impl(black_box(&src), &mut dst)))
        });
        g.bench_function("log2_slice", |b| {
            b.iter(|| Logf::<LA>::log2_slice(black_box(&src), &mut dst).unwrap())
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);

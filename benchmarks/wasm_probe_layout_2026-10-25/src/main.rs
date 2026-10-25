//! Interleaved layout experiment for the relaxed-madd probe (f32x4).
use archmage::SimdToken;
use core::arch::wasm32::{self as w, v128};
use core::hint::black_box;
use core::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

const N: usize = 1024;
static F32: AtomicU8 = AtomicU8::new(0);

#[cold]
#[inline(never)]
fn probe() -> bool {
    let x = w::f32x4_splat(black_box(1.0 + 1.0 / 4096.0));
    let c = w::f32x4_splat(black_box(-(1.0 + 1.0 / 2048.0)));
    let f = w::f32x4_extract_lane::<0>(w::f32x4_relaxed_madd(x, x, c)) == 1.0 / 16_777_216.0;
    F32.store(if f { 2 } else { 1 }, Ordering::Relaxed);
    f
}
#[inline(always)]
fn fuses() -> bool {
    match F32.load(Ordering::Relaxed) {
        0 => probe(),
        s => s == 2,
    }
}

#[inline(always)]
fn wide_sum(a: v128, b: v128, c: v128) -> v128 {
    let product = w::f64x2_mul(w::f64x2_promote_low_f32x4(a), w::f64x2_promote_low_f32x4(b));
    let addend = w::f64x2_promote_low_f32x4(c);
    let sum = w::f64x2_add(product, addend);
    let va = w::f64x2_sub(sum, product);
    let error = w::f64x2_add(
        w::f64x2_sub(product, w::f64x2_sub(sum, va)),
        w::f64x2_sub(addend, va),
    );
    let zero = w::f64x2_splat(0.0);
    let one = w::i64x2_splat(1);
    let even = w::i64x2_eq(w::v128_and(sum, one), w::i64x2_splat(0));
    let finite = w::f64x2_lt(w::f64x2_abs(sum), w::f64x2_splat(f64::INFINITY));
    let adjust = w::v128_and(finite, w::v128_and(even, w::f64x2_ne(error, zero)));
    let same_sign = w::i64x2_eq(w::f64x2_gt(error, zero), w::f64x2_gt(sum, zero));
    let step = w::v128_bitselect(one, w::i64x2_splat(-1), same_sign);
    w::f32x4_demote_f64x2_zero(w::v128_bitselect(w::i64x2_add(sum, step), sum, adjust))
}
#[inline(always)]
fn soft(a: v128, b: v128, c: v128) -> v128 {
    let lo = wide_sum(a, b, c);
    let hi = wide_sum(
        w::i32x4_shuffle::<2, 3, 2, 3>(a, a),
        w::i32x4_shuffle::<2, 3, 2, 3>(b, b),
        w::i32x4_shuffle::<2, 3, 2, 3>(c, c),
    );
    w::i32x4_shuffle::<0, 1, 4, 5>(lo, hi)
}
#[inline(never)]
fn soft_outline(a: v128, b: v128, c: v128) -> v128 {
    soft(a, b, c)
}
#[cold]
#[inline(never)]
fn soft_cold(a: v128, b: v128, c: v128) -> v128 {
    soft(a, b, c)
}

#[inline(always)]
fn op_plain(a: v128, b: v128, c: v128) -> v128 {
    w::f32x4_add(w::f32x4_mul(a, b), c)
}
#[inline(always)]
fn op_madd(a: v128, b: v128, c: v128) -> v128 {
    w::f32x4_relaxed_madd(a, b, c)
}
#[inline(always)]
fn op_soft(a: v128, b: v128, c: v128) -> v128 {
    soft(a, b, c)
}
#[inline(always)]
fn op_probe_inline(a: v128, b: v128, c: v128) -> v128 {
    if fuses() {
        w::f32x4_relaxed_madd(a, b, c)
    } else {
        soft(a, b, c)
    }
}
#[inline(always)]
fn op_probe_outline(a: v128, b: v128, c: v128) -> v128 {
    if fuses() {
        w::f32x4_relaxed_madd(a, b, c)
    } else {
        soft_outline(a, b, c)
    }
}
#[inline(always)]
fn op_probe_cold(a: v128, b: v128, c: v128) -> v128 {
    if fuses() {
        w::f32x4_relaxed_madd(a, b, c)
    } else {
        soft_cold(a, b, c)
    }
}
/// Hot path: one load, one compare, madd; everything else in one cold call.
#[cold]
#[inline(never)]
fn slow(a: v128, b: v128, c: v128) -> v128 {
    if fuses() {
        w::f32x4_relaxed_madd(a, b, c)
    } else {
        soft(a, b, c)
    }
}
#[inline(always)]
fn op_one_cold(a: v128, b: v128, c: v128) -> v128 {
    if F32.load(Ordering::Relaxed) == 2 {
        w::f32x4_relaxed_madd(a, b, c)
    } else {
        slow(a, b, c)
    }
}
/// Pure check cost: no call on either side.
#[inline(always)]
fn op_check_nocall(a: v128, b: v128, c: v128) -> v128 {
    if F32.load(Ordering::Relaxed) == 2 {
        w::f32x4_relaxed_madd(a, b, c)
    } else {
        op_plain(a, b, c)
    }
}
/// No call site: the probe computes inline on first use.
#[inline(always)]
fn probe_inline_compute() -> bool {
    let x = w::f32x4_splat(black_box(1.0 + 1.0 / 4096.0));
    let c = w::f32x4_splat(black_box(-(1.0 + 1.0 / 2048.0)));
    let f = w::f32x4_extract_lane::<0>(w::f32x4_relaxed_madd(x, x, c)) == 1.0 / 16_777_216.0;
    F32.store(if f { 2 } else { 1 }, Ordering::Relaxed);
    f
}
#[inline(always)]
fn op_nocall_full(a: v128, b: v128, c: v128) -> v128 {
    let s = F32.load(Ordering::Relaxed);
    if s == 2 || (s == 0 && probe_inline_compute()) {
        w::f32x4_relaxed_madd(a, b, c)
    } else {
        soft(a, b, c)
    }
}
#[inline(always)]
fn op_magetypes(a: v128, b: v128, c: v128) -> v128 {
    use magetypes::simd::generic::f32x4;
    let t = archmage::Wasm128Token::summon().unwrap();
    let v = |x: v128| f32x4::from_raw_t(t, x);
    v(a).mul_add_fused(v(b), v(c)).raw()
}

fn t() -> archmage::Wasm128Token {
    archmage::Wasm128Token::summon().unwrap()
}
#[inline(always)]
fn f64_plain(a: v128, b: v128, c: v128) -> v128 {
    w::f64x2_add(w::f64x2_mul(a, b), c)
}
#[inline(always)]
fn f64_madd(a: v128, b: v128, c: v128) -> v128 {
    w::f64x2_relaxed_madd(a, b, c)
}
#[inline(always)]
fn f64_fused(a: v128, b: v128, c: v128) -> v128 {
    use magetypes::simd::generic::f64x2;
    let v = |x: v128| f64x2::from_raw_t(t(), x);
    v(a).mul_add_fused(v(b), v(c)).raw()
}
macro_rules! kernels {
    ($($name:ident => $op:ident),*) => {
        $(mod $name {
            use super::*;
            #[inline(never)]
            pub fn stream(a: &[v128], b: &[v128], c: &[v128], o: &mut [v128]) {
                for i in 0..a.len() { o[i] = $op(a[i], b[i], c[i]); }
            }
            #[inline(never)]
            pub fn chain(x: v128, c: v128) -> v128 {
                let mut acc = c;
                for _ in 0..N { acc = $op(acc, x, c); }
                acc
            }
        })*
        const NAMES: &[&str] = &[$(stringify!($name)),*];
        fn run(i: usize, chain: bool, a: &[v128], b: &[v128], c: &[v128], o: &mut [v128]) -> v128 {
            let mut k = 0;
            $(if i == k { return if chain { $name::chain(black_box(a[0]), black_box(c[0])) } else { $name::stream(a, b, c, o); o[N - 1] }; } k += 1;)*
            let _ = k; unreachable!()
        }
    };
}
kernels!(plain => op_plain, madd => op_madd, soft => op_soft, probe_inline => op_probe_inline,
         probe_outline => op_probe_outline, probe_cold => op_probe_cold, one_cold => op_one_cold, check_nocall => op_check_nocall, nocall_full => op_nocall_full, magetypes_fused => op_magetypes,
         f64_plain_k => f64_plain, f64_madd_k => f64_madd, f64_magetypes_fused => f64_fused);

fn main() {
    let mk = |s: f32| {
        (0..N)
            .map(|i| w::f32x4_splat(s + i as f32 * 1e-4))
            .collect::<Vec<_>>()
    };
    let (a, b, c) = (mk(1.0001), mk(0.9999), mk(-0.5));
    let mut o = vec![w::f32x4_splat(0.0); N];
    fuses();
    std::eprintln!("probe fuses: {}", fuses());
    for chain in [true, false] {
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); NAMES.len()];
        for round in 0..400 {
            for j in 0..NAMES.len() {
                let i = (j + round) % NAMES.len();
                let t = Instant::now();
                for _ in 0..8 {
                    black_box(run(i, chain, &a, &b, &c, &mut o));
                }
                times[i].push(t.elapsed().as_nanos() as f64 / 8.0);
            }
        }
        let med: Vec<f64> = times
            .iter_mut()
            .map(|v| {
                v.sort_by(|x, y| x.partial_cmp(y).unwrap());
                v[v.len() / 2]
            })
            .collect();
        let base = med[0];
        println!(
            "{}",
            if chain {
                "chain (1024 dependent steps)"
            } else {
                "stream (1024 vectors)"
            }
        );
        for (n, m) in NAMES.iter().zip(&med) {
            println!("  {n:16} {m:9.0} ns  {:5.2}x plain", m / base);
        }
    }
}

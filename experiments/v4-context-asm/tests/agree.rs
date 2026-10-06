//! Each kernel is run on a fixed input in the v3 tier, the v4 tier and the
//! scalar tier (or a plain-Rust reference for groups B and C). Exact equality
//! unless the test says otherwise: A10 and A11 compare against the scalar
//! tier within a stated ULP bound and v3 against v4 exactly.
use archmage::prelude::*;
use v4_context_asm::*;

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 32) as u32
}
fn floats(n: usize, lo: f32, hi: f32, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| lo + (hi - lo) * (lcg(&mut s) as f32 / u32::MAX as f32)).collect()
}
fn u32s(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s)).collect()
}
fn i64s(n: usize, seed: u64) -> Vec<i64> {
    let mut s = seed;
    (0..n).map(|_| ((lcg(&mut s) as u64) << 32 | lcg(&mut s) as u64) as i64).collect()
}
fn ulp(a: f32, b: f32) -> u32 {
    let k = |x: f32| {
        let i = x.to_bits() as i32;
        if i < 0 { i32::MIN.wrapping_sub(i) } else { i }
    };
    k(a).abs_diff(k(b))
}

const N: usize = 203; // not a multiple of 8, 16 or 32

fn tokens() -> (X64V3Token, X64V4Token) {
    (
        X64V3Token::summon().expect("host has AVX2+FMA"),
        X64V4Token::summon().expect("host has AVX-512"),
    )
}

macro_rules! agree_inplace {
    ($name:ident, $v3:ident, $v4:ident, $sc:ident, $input:expr $(, $arg:expr)*) => {
        #[test]
        fn $name() {
            let (t3, t4) = tokens();
            let input: Vec<f32> = $input;
            let (mut a, mut b, mut c) = (input.clone(), input.clone(), input.clone());
            $v3(t3, &mut a $(, $arg)*);
            $v4(t4, &mut b $(, $arg)*);
            $sc(ScalarToken, &mut c $(, $arg)*);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a), bits(&b), "v3 vs v4");
            assert_eq!(bits(&a), bits(&c), "v3 vs scalar");
        }
    };
}

agree_inplace!(a1, a1_v3, a1_v4, a1_impl_scalar, floats(N, -10.0, 10.0, 1), 0.37);
agree_inplace!(a2, a2_v3, a2_v4, a2_impl_scalar, floats(N, -10.0, 10.0, 2), 0.37);
agree_inplace!(a3, a3_v3, a3_v4, a3_impl_scalar, floats(N, -10.0, 10.0, 3), 0.37);
agree_inplace!(a4, a4_v3, a4_v4, a4_impl_scalar, floats(N, -10.0, 10.0, 4), 1.5, 2.0, -7.0);
// A6: fused mul_add on x86 vs multiply-then-add on the scalar tier (documented
// 0.9.29 contract), so the scalar comparison is a ULP bound; v3 vs v4 is exact.
#[test]
fn a6() {
    let (t3, t4) = tokens();
    let input = floats(N, -1.0, 1.0, 6);
    let (mut a, mut b, mut c) = (input.clone(), input.clone(), input.clone());
    a6_v3(t3, &mut a);
    a6_v4(t4, &mut b);
    a6_impl_scalar(ScalarToken, &mut c);
    assert_eq!(a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), b.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    for (x, y) in a.iter().zip(&c) {
        assert!(ulp(*x, *y) <= 8, "{x} vs {y}");
    }
}
#[test]
fn a8_round() {
    let (t3, t4) = tokens();
    let mut input = floats(N, -1000.0, 1000.0, 8);
    input[0] = 0.5;
    input[1] = 1.5;
    input[2] = -2.5;
    input[3] = 2.5;
    let (mut a, mut b, mut c) = (input.clone(), input.clone(), input.clone());
    a8_round_v3(t3, &mut a);
    a8_round_v4(t4, &mut b);
    a8_round_impl_scalar(ScalarToken, &mut c);
    assert_eq!(a, b);
    assert_eq!(a, c);
}
#[test]
fn a10_recip() {
    let (t3, t4) = tokens();
    let input = floats(N, 0.01, 100.0, 10);
    let (mut a, mut b, mut c) = (input.clone(), input.clone(), input.clone());
    a10_recip_v3(t3, &mut a);
    a10_recip_v4(t4, &mut b);
    a10_recip_impl_scalar(ScalarToken, &mut c);
    assert_eq!(a, b, "v3 vs v4 exact");
    for (x, y) in a.iter().zip(&c) {
        assert!(ulp(*x, *y) <= 4, "{x} vs {y}");
    }
}
#[test]
fn a10_rsqrt() {
    let (t3, t4) = tokens();
    let input = floats(N, 0.01, 100.0, 11);
    let (mut a, mut b, mut c) = (input.clone(), input.clone(), input.clone());
    a10_rsqrt_v3(t3, &mut a);
    a10_rsqrt_v4(t4, &mut b);
    a10_rsqrt_impl_scalar(ScalarToken, &mut c);
    assert_eq!(a, b, "v3 vs v4 exact");
    for (x, y) in a.iter().zip(&c) {
        assert!(ulp(*x, *y) <= 4, "{x} vs {y}");
    }
}
// A11: exp2_midp <= 1.9 ULP, ln_midp <= 4.1 ULP (doc comments). The scalar
// tier here is the scalar backend's own midp; the tail uses std (1 ULP), so
// the bound against it is the documented one plus one.
#[test]
fn a11_exp2() {
    let (t3, t4) = tokens();
    let input = floats(N, -20.0, 20.0, 12);
    let (mut a, mut b, mut c) = (input.clone(), input.clone(), input.clone());
    a11_exp2_v3(t3, &mut a);
    a11_exp2_via_v3(t4, &mut b);
    a11_exp2_impl_scalar(ScalarToken, &mut c);
    assert_eq!(a, b, "v3 vs v4-via-v3 exact");
    for (x, y) in a.iter().zip(&c) {
        assert!(ulp(*x, *y) <= 3, "{x} vs {y}");
    }
}
#[test]
fn a11_ln() {
    let (t3, t4) = tokens();
    let input = floats(N, 2.5, 10000.0, 13);
    let (mut a, mut b, mut c) = (input.clone(), input.clone(), input.clone());
    a11_ln_v3(t3, &mut a);
    a11_ln_via_v3(t4, &mut b);
    a11_ln_impl_scalar(ScalarToken, &mut c);
    assert_eq!(a, b, "v3 vs v4-via-v3 exact");
    for (x, y) in a.iter().zip(&c) {
        assert!(ulp(*x, *y) <= 6, "{x} vs {y}");
    }
}
#[test]
fn a12() {
    let (t3, t4) = tokens();
    let input = floats(N, -10.0, 10.0, 14);
    let (mut a, mut b, mut c) = (input.clone(), input.clone(), input.clone());
    a12_v3(t3, &mut a, 0.37);
    a12_v4(t4, &mut b, 0.37);
    a12_impl_scalar(ScalarToken, &mut c, 0.37);
    assert_eq!(a, b);
    assert_eq!(a, c);
}
#[test]
fn a5() {
    let (t3, t4) = tokens();
    let x = floats(N, -10.0, 10.0, 5);
    let m: Vec<f32> = u32s(N, 55).into_iter().map(f32::from_bits).collect();
    let (mut a, mut b, mut c) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
    a5_v3(t3, &x, &m, &mut a);
    a5_v4(t4, &x, &m, &mut b);
    a5_impl_scalar(ScalarToken, &x, &m, &mut c);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a), bits(&b));
    assert_eq!(bits(&a), bits(&c));
}
#[test]
fn a7() {
    let (t3, t4) = tokens();
    let input = floats(N + A7_TAPS, -1.0, 1.0, 7);
    let coef: [f32; A7_TAPS] = floats(A7_TAPS, -1.0, 1.0, 77).try_into().unwrap();
    let n = N;
    let (mut a, mut b, mut c) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    a7_v3(t3, &input, &coef, &mut a);
    a7_v4(t4, &input, &coef, &mut b);
    a7_impl_scalar(ScalarToken, &input, &coef, &mut c);
    assert_eq!(a, b, "v3 vs v4");
    for (x, y) in a.iter().zip(&c) {
        assert!(ulp(*x, *y) <= 64 || (x - y).abs() < 1e-5, "{x} vs {y}");
    }
}
#[test]
fn a8_sat() {
    let (t3, t4) = tokens();
    let mut x = floats(N, -1e10, 1e10, 80);
    x[0] = f32::NAN;
    x[1] = 3e9;
    x[2] = -3e9;
    x[3] = f32::INFINITY;
    let (mut a, mut b, mut c) = (vec![0; N], vec![0; N], vec![0; N]);
    a8_sat_v3(t3, &x, &mut a);
    a8_sat_via_v3(t4, &x, &mut b);
    a8_sat_impl_scalar(ScalarToken, &x, &mut c);
    let n8 = N / 8 * 8;
    assert_eq!(a[..n8], b[..n8]);
    assert_eq!(a[..n8], c[..n8]);
    assert_eq!(a[1], i32::MAX);
    assert_eq!(a[0], 0);
}
#[test]
fn a8_u8() {
    let (t3, t4) = tokens();
    let mut x = floats(N, -50.0, 300.0, 81);
    x[0] = f32::NAN;
    x[1] = f32::INFINITY;
    x[2] = 254.5;
    x[3] = 0.5;
    let (mut a, mut b, mut c) = (vec![0u8; N], vec![0u8; N], vec![0u8; N]);
    a8_u8_v3(t3, &x, &mut a);
    a8_u8_v4(t4, &x, &mut b);
    a8_u8_impl_scalar(ScalarToken, &x, &mut c);
    assert_eq!(a, b);
    assert_eq!(a, c);
}
#[test]
fn a9() {
    let (t3, t4) = tokens();
    let x = floats(N, -10.0, 10.0, 9);
    // Same lane-wise accumulation order in every tier: exact.
    assert_eq!(a9_sum_v3(t3, &x).to_bits(), a9_sum_v4(t4, &x).to_bits());
    assert_eq!(a9_sum_v3(t3, &x).to_bits(), a9_sum_impl_scalar(ScalarToken, &x).to_bits());
    assert_eq!(a9_max_v3(t3, &x), a9_max_v4(t4, &x));
    assert_eq!(a9_max_v3(t3, &x), a9_max_impl_scalar(ScalarToken, &x));
}

// ---- Group B ----
fn b1_ref(v: &mut [f32], g: f32) {
    for x in v {
        *x *= g;
    }
}
#[test]
fn b1() {
    let (t3, t4) = tokens();
    let input = floats(N, -10.0, 10.0, 21);
    let (mut a, mut b, mut c) = (input.clone(), input.clone(), input.clone());
    b1_v3(t3, &mut a, 0.37);
    b1_v4(t4, &mut b, 0.37);
    b1_ref(&mut c, 0.37);
    assert_eq!(a, b);
    assert_eq!(a, c);
}
#[test]
fn b2() {
    let (t3, t4) = tokens();
    let x = floats(N, -10.0, 10.0, 22);
    // LLVM does not reassociate float adds, so every tier equals the serial sum.
    let r: f32 = x.iter().fold(0.0, |s, v| s + v * v);
    assert_eq!(b2_v3(t3, &x).to_bits(), r.to_bits());
    assert_eq!(b2_v4(t4, &x).to_bits(), r.to_bits());
}

// ---- Group C ----
#[test]
fn c1() {
    let (t3, t4) = tokens();
    let (a, b) = (floats(N, -5.0, 5.0, 31), floats(N, -5.0, 5.0, 32));
    let (x, y) = (floats(N, -5.0, 5.0, 33), floats(N, -5.0, 5.0, 34));
    let (mut o3, mut o4) = (vec![0.0; N], vec![0.0; N]);
    c1_v3(t3, &a, &b, &x, &y, &mut o3);
    c1_v4(t4, &a, &b, &x, &y, &mut o4);
    for i in 0..N / 8 * 8 {
        let want = if x[i] < y[i] { b[i] } else { a[i] };
        assert_eq!(o3[i], want);
        assert_eq!(o4[i], want);
    }
}
#[test]
fn c2() {
    let (t3, t4) = tokens();
    let cast = |v: Vec<u32>| v.into_iter().map(|x| x as i32).collect::<Vec<_>>();
    let (a, b, m) = (cast(u32s(N, 41)), cast(u32s(N, 42)), cast(u32s(N, 43)));
    let (mut o3, mut o4) = (vec![0; N], vec![0; N]);
    c2_v3(t3, &a, &b, &m, &mut o3);
    c2_v4(t4, &a, &b, &m, &mut o4);
    for i in 0..N / 8 * 8 {
        let want = (a[i] & m[i]) | (!m[i] & b[i]);
        assert_eq!(o3[i], want);
        assert_eq!(o4[i], want);
    }
}
#[test]
fn c3() {
    let (t3, t4) = tokens();
    let (a, b) = (i64s(N, 51), i64s(N, 52));
    let (mut o3, mut o4) = (vec![0; N], vec![0; N]);
    c3_v3(t3, &a, &b, &mut o3);
    c3_v4(t4, &a, &b, &mut o4);
    for i in 0..N / 4 * 4 {
        assert_eq!(o3[i], a[i].min(b[i]));
        assert_eq!(o4[i], a[i].min(b[i]));
    }
}
#[test]
fn c4() {
    let (t3, t4) = tokens();
    let a = u32s(N, 61);
    let (mut o3, mut o4) = (vec![0; N], vec![0; N]);
    c4_v3(t3, &a, &mut o3);
    c4_v4(t4, &a, &mut o4);
    for i in 0..N / 8 * 8 {
        assert_eq!(o3[i], a[i].rotate_left(7));
        assert_eq!(o4[i], a[i].rotate_left(7));
    }
}
#[test]
fn c5() {
    let (t3, t4) = tokens();
    let mut a = i64s(N, 71);
    a[0] = i64::MIN;
    let (mut o3, mut o4) = (vec![0; N], vec![0; N]);
    c5_v3(t3, &a, &mut o3);
    c5_v4(t4, &a, &mut o4);
    for i in 0..N / 4 * 4 {
        assert_eq!(o3[i], a[i].wrapping_abs());
        assert_eq!(o4[i], a[i].wrapping_abs());
    }
}
#[test]
fn c6() {
    let (t3, t4) = tokens();
    let (a, b) = (u32s(N, 91), u32s(N, 92));
    let (mut o3, mut o4) = (vec![0u8; N / 8], vec![0u8; N / 8]);
    c6_v3(t3, &a, &b, &mut o3);
    c6_v4(t4, &a, &b, &mut o4);
    for (k, (&m3, &m4)) in o3.iter().zip(&o4).enumerate() {
        let want = (0..8).fold(0u8, |m, j| m | (((a[k * 8 + j] > b[k * 8 + j]) as u8) << j));
        assert_eq!(m3, want);
        assert_eq!(m4, want);
    }
}

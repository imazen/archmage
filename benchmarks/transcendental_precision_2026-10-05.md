# Generic transcendental precision, exhaustive (Zen 5, 2026-10-05)

Accuracy of the generic `f32` transcendentals (`exp2`, `exp`, `log2`, `ln`,
`log10`, `pow`, `cbrt`, midp and lowp) over every f32 input in each band. Two
backends that differ in one way: the scalar backend computes `mul_add` as a
multiply and then an add, and AVX2 (`X64V3Token`) as one FMA instruction. The
transcendentals evaluate their polynomials with `mul_add`, so the comparison
shows what two roundings per polynomial step cost. NEON uses FMA like AVX2;
strict WASM rounds twice like the scalar backend. Neither was run here.

Host: AMD Ryzen 9 9950X3D, rustc 1.99.0 (b940084d7 2026-09-28), release with
`codegen-units = 1`, no `-Ctarget-cpu`. Sources: main at 2f856d2c.

## Method

The probe below evaluates `f32x4<ScalarToken>` and `f32x4<X64V3Token>` on every
f32 in each band (negatives included where the band has them) and compares both
with the f64 result from `std` (`f64::exp2`, `exp`, `log2`, `ln`, `log10`,
`cbrt`, `powf`). Error in ULP is `|y - r| / ulp(r as f32)`, where `r` is the f64
reference. Inputs whose reference is zero, subnormal or infinite as an f32 are
skipped. "differ" counts inputs where the two backends return different bits;
"max diff" is the largest difference between them, in ULP of the reference.

Run: `~/work/zen/scripts/run-heavy --mem 8G -- ./target/release/precision-final`
(28 threads, 24 s).

## Results: midp

| Function | Band | Inputs | Scalar max / mean ULP | AVX2 max / mean ULP | Max rel. error | Differ | Max diff |
|---|---|---:|---|---|---:|---:|---:|
| `exp2_midp` | x in [-126, 127.5) | 2,247,819,264 | 1.858 / 0.0656 | 1.594 / 0.0647 | 1.57e-7 | 0.80% | 1 ULP |
| `exp2_midp` | x in [127.5, 128) | 65,536 | 134.077 / 32.16 | 133.844 / 32.16 | 7.99e-6 | 26.0% | 1 ULP |
| `exp_midp` | \|x\| <= 1 | 2,130,706,433 | 1.995 / 0.0546 | 1.714 / 0.0543 | 1.68e-7 | 0.34% | 1 ULP |
| `exp_midp` | \|x\| <= 10 | 2,185,232,385 | 8.181 / 0.0826 | 8.181 / 0.0822 | 5.84e-7 | 0.59% | 1 ULP |
| `exp_midp` | \|x\| <= 40 | 2,218,786,817 | 31.289 / 0.1744 | 31.289 / 0.1739 | 1.97e-6 | 0.72% | 1 ULP |
| `exp_midp` | x in [-87, 88.5] | 2,237,595,649 | 64.081 / 0.3129 | 64.081 / 0.3125 | 4.40e-6 | 0.80% | 1 ULP |
| `exp_midp` | x in (88.5, ln(MAX)] | 29,207 | 197.095 / 66.31 | 197.095 / 66.31 | 1.18e-5 | 28.6% | 1 ULP |
| `log2_midp` | positive normals | 2,130,706,431 | 4.404 / 0.2526 | 4.404 / 0.2524 | 3.21e-7 | 0.52% | 2 ULP |
| `ln_midp` | positive normals | 2,130,706,431 | 4.056 / 0.3452 | 3.732 / 0.3449 | 3.49e-7 | 0.44% | 2 ULP |
| `log10_midp` | positive normals | 2,130,706,431 | 4.426 / 0.6417 | 4.426 / 0.6416 | 4.07e-7 | 0.46% | 3 ULP |
| `cbrt_midp` | x in [2^-126, MAX/3] | 2,116,725,419 | 3.138 / 0.5289 | 3.138 / 0.5289 | 2.45e-7 | 0 | 0 |
| `cbrt_midp_precise` | positive normals | 2,130,706,432 | 3.138 / 0.5285 | 3.138 / 0.5285 | 2.45e-7 | 0 | 0 |
| `cbrt_midp_precise` | positive subnormals | 8,388,607 | 3.138 / 0.5666 | 3.138 / 0.5666 | 2.05e-7 | 0 | 0 |
| `pow_midp(2.4)` | x in [2^-8, 2^8] | 134,217,729 | 27.774 / 3.661 | 27.444 / 3.653 | 1.67e-6 | 13.0% | 23 ULP |
| `pow_midp(2.4)` | x in [2^-20, 2^20] | 335,544,321 | 64.715 / 8.958 | 64.730 / 8.954 | 3.94e-6 | 11.5% | 89 ULP |
| `pow_midp(2.4)` | x in [2^-50, 2^50] | 838,860,801 | 144.724 / 22.42 | 144.857 / 22.41 | 8.71e-6 | 10.7% | 178 ULP |

No band produced a non-finite result. "Max rel. error" is the larger of the two
backends' maxima. The `exp_midp` row above 88.5 comes from a second run of the
same probe with that band added; `ln(MAX)` is 88.72283.

## Results: lowp

ULP is not a useful unit for the lowp logarithms near x = 1, where the reference
approaches zero; their absolute error is the bound.

| Function | Band | Max rel. error | Max abs. error | Differ | Max diff |
|---|---|---:|---:|---:|---:|
| `exp2_lowp` | x in [-126, 127.99] | 5.561e-3 | | 3.62% | 1 ULP |
| `exp2_lowp` | x in (127.99, 128) | 1.224e-2 | | 0 | 0 |
| `exp_lowp` | x in [-87, 88.5] | 5.565e-3 | | 3.66% | 1 ULP |
| `exp_lowp` | x in (88.5, ln(MAX)] | 1.224e-2 | | 27.9% | 1 ULP |
| `pow_lowp(2.4)` | x in [2^-8, 2^8] | 5.566e-3 | | 19.0% | 23 ULP |
| `pow_lowp(2.4)` | x in [2^-20, 2^20] | 5.569e-3 | | 17.8% | 93,343 ULP |
| `log2_lowp` | positive normals | | 6.35e-6 | 0.44% | 6 ULP |
| `ln_lowp` | positive normals | | 8.43e-6 | 0.37% | 6 ULP |
| `log10_lowp` | positive normals | | 5.77e-6 | 0.39% | 7 ULP |
| `cbrt_lowp` | x in [2^-126, MAX/3] | 2.980e-5 | | 0 | 0 |

Each error column gives the larger of the two backends' maxima, rounded up; they
agree to within 0.1%. `exp2_lowp` clamps its input
to 127.99, so above that it returns about 2^127.99 and runs up to 1.23% low.
`exp2_lowp`'s cubic gives 1.98888 at the top of each unit interval and 2 at the
bottom of the next, a 0.56% step at every integer. A `pow_lowp` input whose
exponent `2.4 * log2(x)` lands next to an integer can step one way on one
backend and the other way on the other: that is the 93,343-ULP difference, both
results within the 0.56% bound.

## Two roundings against one

- **Accuracy against the true value barely moves.** Across the bands above,
  the largest change in a maximum is 0.33 ULP (`pow_midp` on [2^-8, 2^8]:
  27.774 scalar, 27.444 AVX2), and 0.43 ULP on the narrower `exp2_midp` band
  [127.5, 127.9] (63.248 against 62.822). Means move by at most 0.008 ULP. On
  `pow_midp`'s two wider bands the scalar backend has the smaller maximum.
- **Results differ between the backends.** 0.3–0.8% of exp/log results on
  their main bands differ by 1–3 ULP (about a quarter in the narrow bands at the
  top of `exp2`'s range), and 10.7–13% of `pow_midp` results by up to 178 ULP,
  because
  `pow` scales a one-ULP difference in `log2` by `2.4 * log2(x)` before `exp2`.
  `cbrt` uses no `mul_add` and never differs.
- **Fusion is the whole difference.** In the interim design (#116), when the
  scalar backend fused `mul_add` in software, the same comparison for
  `exp2/exp/log2/ln/log10/cbrt/pow_midp`, `exp2_lowp` and `log2_lowp` found the
  two backends bit-identical on every input. At 66986145 (the split, before the
  `exp2_lowp` fix), the scalar results for `exp2/exp/log2/ln/log10/pow_midp`,
  `exp2_lowp` and `log2_lowp` matched 0.9.29 again: identical maxima, means and
  difference counts.

## Fixes made, and those left documented

Timings are medians of 200 interleaved rounds over 1,024 L1-resident vectors
(`f32x8` on AVX2, `f32x4` on the scalar backend), three rounds per build, with
the timing probe below built against the tree before and after each change.

- **`exp2_lowp` clamp (64ae008f).** It clamped its input to 126.0, so every x in
  [126, 128) returned 2^126: 50% low at x = 127, 75% near 128; `exp_lowp` and
  `pow_lowp` inherited it. Clamping at 127.99 gives the table above. Timing:
  686–761 ns against 687–688 ns on AVX2, 8,760–9,009 against 8,799–8,811 ns on
  the scalar backend, the same within noise.
- **`exp2_midp` above 127.5 (documented, not fixed).** It clamps the rounded
  integer part to 127, so for x in [127.5, 128) the polynomial runs outside
  [-0.5, 0.5]: up to 134.1 ULP, and 197.1 ULP for `exp_midp` above 88.5.
  Folding in the missing factor of two brought [127.5, 127.9] from 63.2 to
  1.741 ULP but took
  7–12% longer with AVX2 (`exp2_midp` 1,510–1,550 ns against 1,387; `exp_midp`
  +8–11%; `pow_midp` +7–10%) and 4–8% longer on the scalar backend, on every
  call. The function docs state the band's error instead.
- **`cbrt` above `f32::MAX / 3`.** The Halley step forms `y³ + 2x`, which
  overflows from x = 1.1342859e38: `cbrt_midp`, `cbrt_lowp` and, through
  `cbrt_midp`, `cbrt_midp_precise` returned ±inf for those 13,980,901 positive
  normals on every backend, 0.9.29 included. Three fixes for the fast forms,
  timed against the unchanged code (AVX2 / scalar backend):

  | Fix | `cbrt_midp` | `cbrt_lowp` | `cbrt_midp_precise` |
  |---|---|---|---|
  | Halley step on the ratio y³/x (two more divisions) | +28–29% / +25% | +13–14% / +8% | +25–26% / +21–22% |
  | Rescale large inputs inside each iteration | +19–21% / +28–29% | +21% / +26–27% | +17–18% / +15–16% |
  | Rescale once around both iterations | +30–34% / +26–29% | +43–47% / +35–39% | +25–28% / +17–20% |

  The first also lowered `cbrt_midp`'s maximum to 2.499 ULP. None was applied to
  the fast forms; their docs state the limit. `cbrt_midp_precise`, already the
  edge-handling variant, now runs magnitudes from 1e36 up on x/8 and doubles
  the result, exact in binary (2f856d2c). The table's precise rows are that
  code. It takes 15.4–16.9% longer on AVX2 and 8.2–9.3% on the scalar backend
  than before the change; an earlier form with separate blends cost 23–24% and
  28–29%.

## Probe sources

Precision probe, `Cargo.toml`:

```toml
[package]
name = "precision-final"
version = "0.0.0"
edition = "2024"
publish = false

[[bin]]
name = "precision-final"
path = "src/main.rs"

[dependencies]
archmage = { path = "/home/lilith/work/archmage" }
magetypes = { path = "/home/lilith/work/archmage/magetypes" }

[profile.release]
codegen-units = 1

[workspace]
```

`src/main.rs`:

```rust
//! Per-band precision of magetypes f32x4 transcendentals on the scalar backend
//! (`mul_add` rounds twice) and AVX2 (`mul_add` is one FMA), against an f64
//! reference, over every f32 in each band. Reports max/mean ULP, max relative
//! and absolute error, and how many results differ between the two backends.
#![allow(deprecated)]
use archmage::{ScalarToken, SimdToken, X64V3Token, arcane};
use magetypes::simd::generic::f32x4;
use std::sync::atomic::{AtomicUsize, Ordering};

type Lanes = [f32; 4];
const BATCH: usize = 4096;

macro_rules! kernels {
    ($scalar:ident, $v3:ident, $($call:tt)+) => {
        fn $scalar(xs: &[Lanes], out: &mut [Lanes]) {
            for (x, o) in xs.iter().zip(out.iter_mut()) {
                *o = f32x4::<ScalarToken>::load(ScalarToken, x).$($call)+.to_array();
            }
        }
        #[arcane]
        fn $v3(t: X64V3Token, xs: &[Lanes], out: &mut [Lanes]) {
            for (x, o) in xs.iter().zip(out.iter_mut()) {
                *o = f32x4::<X64V3Token>::load(t, x).$($call)+.to_array();
            }
        }
    };
}
kernels!(exp2_s, exp2_v, exp2_midp());
kernels!(exp_s, exp_v, exp_midp());
kernels!(log2_s, log2_v, log2_midp());
kernels!(ln_s, ln_v, ln_midp());
kernels!(log10_s, log10_v, log10_midp());
kernels!(cbrt_s, cbrt_v, cbrt_midp());
kernels!(pow_s, pow_v, pow_midp(2.4));
kernels!(exp2l_s, exp2l_v, exp2_lowp());
kernels!(log2l_s, log2l_v, log2_lowp());
kernels!(expl_s, expl_v, exp_lowp());
kernels!(lnl_s, lnl_v, ln_lowp());
kernels!(powl_s, powl_v, pow_lowp(2.4));
kernels!(cbrtl_s, cbrtl_v, cbrt_lowp());
kernels!(cbrtp_s, cbrtp_v, cbrt_midp_precise());
kernels!(log10l_s, log10l_v, log10_lowp());

#[derive(Clone, Copy)]
struct Func {
    name: &'static str,
    scalar: fn(&[Lanes], &mut [Lanes]),
    v3: fn(X64V3Token, &[Lanes], &mut [Lanes]),
    reference: fn(f64) -> f64,
}

#[derive(Default, Clone, Copy)]
struct Side { max_ulp: f64, sum_ulp: f64, max_rel: f64, max_abs: f64, nonfinite: u64 }
impl Side {
    fn add(&mut self, y: f32, r: f64, ulp: f64) {
        if !y.is_finite() { self.nonfinite += 1; return; }
        let d = (y as f64 - r).abs();
        let u = d / ulp;
        self.sum_ulp += u;
        self.max_ulp = self.max_ulp.max(u);
        self.max_abs = self.max_abs.max(d);
        if r != 0.0 { self.max_rel = self.max_rel.max(d / r.abs()); }
    }
    fn merge(&mut self, o: &Side) {
        self.sum_ulp += o.sum_ulp;
        self.max_ulp = self.max_ulp.max(o.max_ulp);
        self.max_rel = self.max_rel.max(o.max_rel);
        self.max_abs = self.max_abs.max(o.max_abs);
        self.nonfinite += o.nonfinite;
    }
}

fn ranges(lo: f32, hi: f32) -> Vec<(u32, u32)> {
    let mut r = Vec::new();
    if lo < 0.0 { r.push((0x8000_0001, lo.to_bits())); }
    if hi >= 0.0 { r.push((lo.max(0.0).to_bits(), hi.to_bits())); }
    r
}

fn ulp_of(r: f64) -> Option<f64> {
    let e = ((r as f32).abs().to_bits() >> 23) & 0xFF;
    if e == 0 || e == 0xFF { return None; }
    Some(f64::from_bits(((e as u64 + 1023 - 127 - 23) & 0x7FF) << 52))
}

fn run(f: Func, lo: f32, hi: f32, t: X64V3Token, threads: usize) -> (u64, Side, Side, u64, f64) {
    let mut chunks = Vec::new();
    for (a, b) in ranges(lo, hi) {
        let mut s = a as u64;
        while s <= b as u64 {
            let e = (s + (1 << 22) - 1).min(b as u64);
            chunks.push((s as u32, e as u32));
            s = e + 1;
        }
    }
    let next = AtomicUsize::new(0);
    let parts: Vec<(u64, Side, Side, u64, f64)> = std::thread::scope(|scope| {
        let hs: Vec<_> = (0..threads).map(|_| scope.spawn(|| {
            let (mut n, mut ss, mut sv) = (0u64, Side::default(), Side::default());
            let (mut differ, mut max_diff) = (0u64, 0f64);
            let mut xs = vec![[0f32; 4]; BATCH];
            let (mut ys, mut yv) = (vec![[0f32; 4]; BATCH], vec![[0f32; 4]; BATCH]);
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(a, b)) = chunks.get(i) else { break };
                let mut bits = a as u64;
                while bits <= b as u64 {
                    let count = ((b as u64 - bits + 1) as usize).min(BATCH * 4);
                    let vecs = count.div_ceil(4);
                    for k in 0..vecs * 4 {
                        xs[k / 4][k % 4] = f32::from_bits((bits + k.min(count - 1) as u64) as u32);
                    }
                    (f.scalar)(&xs[..vecs], &mut ys[..vecs]);
                    (f.v3)(t, &xs[..vecs], &mut yv[..vecs]);
                    for k in 0..count {
                        let x = xs[k / 4][k % 4];
                        let r = (f.reference)(x as f64);
                        let Some(ulp) = ulp_of(r) else { continue };
                        n += 1;
                        let (a, b) = (ys[k / 4][k % 4], yv[k / 4][k % 4]);
                        ss.add(a, r, ulp);
                        sv.add(b, r, ulp);
                        if a.to_bits() != b.to_bits() {
                            differ += 1;
                            if a.is_finite() && b.is_finite() {
                                max_diff = max_diff.max((a as f64 - b as f64).abs() / ulp);
                            }
                        }
                    }
                    bits += count as u64;
                }
            }
            (n, ss, sv, differ, max_diff)
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (mut n, mut ss, mut sv) = (0u64, Side::default(), Side::default());
    let (mut differ, mut max_diff) = (0u64, 0f64);
    for (pn, ps, pv, pd, pm) in &parts {
        n += pn;
        ss.merge(ps);
        sv.merge(pv);
        differ += pd;
        max_diff = max_diff.max(*pm);
    }
    (n, ss, sv, differ, max_diff)
}

fn main() {
    let t = X64V3Token::summon().expect("x86-64-v3");
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(28);
    let below = |v: f32| f32::from_bits(v.to_bits() - 1);
    let p = |e: i32| 2f32.powi(e);
    let mn = f32::MIN_POSITIVE;
    let f = |name, scalar, v3, reference| Func { name, scalar, v3, reference };
    let exp2 = f("exp2_midp", exp2_s, exp2_v, f64::exp2 as fn(f64) -> f64);
    let exp = f("exp_midp", exp_s, exp_v, f64::exp);
    let log2 = f("log2_midp", log2_s, log2_v, f64::log2);
    let ln = f("ln_midp", ln_s, ln_v, f64::ln);
    let log10 = f("log10_midp", log10_s, log10_v, f64::log10);
    let cbrt = f("cbrt_midp", cbrt_s, cbrt_v, f64::cbrt);
    let pow = f("pow_midp(2.4)", pow_s, pow_v, |x| x.powf(2.4));
    let exp2l = f("exp2_lowp", exp2l_s, exp2l_v, f64::exp2);
    let log2l = f("log2_lowp", log2l_s, log2l_v, f64::log2);
    let expl = f("exp_lowp", expl_s, expl_v, f64::exp);
    let lnl = f("ln_lowp", lnl_s, lnl_v, f64::ln);
    let powl = f("pow_lowp(2.4)", powl_s, powl_v, |x| x.powf(2.4));
    let cbrtl = f("cbrt_lowp", cbrtl_s, cbrtl_v, f64::cbrt);
    let cbrtp = f("cbrt_midp_precise", cbrtp_s, cbrtp_v, f64::cbrt);
    let log10l = f("log10_lowp", log10l_s, log10l_v, f64::log10);
    let bands: Vec<(Func, &str, f32, f32)> = vec![
        (exp2, "x in [-126, 127.5)", -126.0, below(127.5)),
        (exp2, "x in [127.5, 128)", 127.5, below(128.0)),
        (exp, "|x| <= 1", -1.0, 1.0),
        (exp, "|x| <= 10", -10.0, 10.0),
        (exp, "|x| <= 40", -40.0, 40.0),
        (exp, "x in [-87, 88.5]", -87.0, 88.5),
        (log2, "all positive normals", mn, f32::MAX),
        (ln, "all positive normals", mn, f32::MAX),
        (log10, "all positive normals", mn, f32::MAX),
        (cbrt, "x in [2^-126, MAX/3]", mn, f32::MAX / 3.0),
        (cbrtp, "all positive normals", mn, f32::MAX),
        (cbrtp, "positive subnormals", f32::from_bits(1), f32::from_bits(0x007f_ffff)),
        (pow, "x in [2^-8, 2^8]", p(-8), p(8)),
        (pow, "x in [2^-20, 2^20]", p(-20), p(20)),
        (pow, "x in [2^-50, 2^50]", p(-50), p(50)),
        (exp2l, "x in [-126, 127.99]", -126.0, 127.99),
        (exp2l, "x in (127.99, 128)", f32::from_bits(127.99f32.to_bits() + 1), below(128.0)),
        (expl, "x in [-87, 88.5]", -87.0, 88.5),
        (expl, "x in (88.5, ln(MAX)]", f32::from_bits(88.5f32.to_bits() + 1), 88.722_83),
        (log2l, "all positive normals", mn, f32::MAX),
        (log2l, "x in [0.5, 2]", 0.5, 2.0),
        (lnl, "all positive normals", mn, f32::MAX),
        (log10l, "all positive normals", mn, f32::MAX),
        (powl, "x in [2^-8, 2^8]", p(-8), p(8)),
        (powl, "x in [2^-20, 2^20]", p(-20), p(20)),
        (cbrtl, "x in [2^-126, MAX/3]", mn, f32::MAX / 3.0),
    ];
    println!("threads: {threads}");
    for (func, label, lo, hi) in bands {
        let (n, s, v, differ, max_diff) = run(func, lo, hi, t, threads);
        let nf = n as f64;
        println!(
            "{:17} {:22} n={:>10} | scalar max {:>10.3} ulp, mean {:.4}, max rel {:.3e}, max abs {:.3e}, nonfinite {} | avx2 max {:>10.3} ulp, mean {:.4}, max rel {:.3e}, max abs {:.3e}, nonfinite {} | differ {:>10} ({:.3}%), max diff {:.3} ulp",
            func.name, label, n, s.max_ulp, s.sum_ulp / nf, s.max_rel, s.max_abs, s.nonfinite,
            v.max_ulp, v.sum_ulp / nf, v.max_rel, v.max_abs, v.nonfinite, differ, 100.0 * differ as f64 / nf, max_diff
        );
    }
}
```

Timing probe (`exp2` variant; the `cbrt` variant times `cbrt_midp`,
`cbrt_lowp`, `cbrt_midp_precise` and `exp2_lowp` in the same four slots). One
copy builds against the tree before a change, one against the tree after:

```rust
//! Times exp2_midp, exp_midp, pow_midp(2.4) and exp2_lowp on f32x8 (AVX2) and
//! f32x4 (scalar backend), 1,024 L1-resident vectors, 200 interleaved rounds.
use archmage::{ScalarToken, SimdToken, X64V3Token, arcane};
use magetypes::simd::generic::{f32x4, f32x8};
use std::hint::black_box;
use std::time::Instant;

const N: usize = 1024;
type V8 = [[f32; 8]];
type V4 = [[f32; 4]];

macro_rules! avx2 { ($name:ident, $($call:tt)+) => {
    #[arcane]
    fn $name(t: X64V3Token, s: &V8, o: &mut V8) {
        for (s, o) in s.iter().zip(o.iter_mut()) { *o = f32x8::<X64V3Token>::from_array_t(t, *s).$($call)+.to_array(); }
    }
}}
macro_rules! scalar { ($name:ident, $($call:tt)+) => {
    #[inline(never)]
    fn $name(s: &V4, o: &mut V4) {
        for (s, o) in s.iter().zip(o.iter_mut()) { *o = f32x4::<ScalarToken>::from_array_t(ScalarToken, *s).$($call)+.to_array(); }
    }
}}
avx2!(a_exp2, exp2_midp()); avx2!(a_exp, exp_midp()); avx2!(a_pow, pow_midp(2.4)); avx2!(a_exp2l, exp2_lowp());
scalar!(s_exp2, exp2_midp()); scalar!(s_exp, exp_midp()); scalar!(s_pow, pow_midp(2.4)); scalar!(s_exp2l, exp2_lowp());

fn main() {
    let t = X64V3Token::summon().expect("v3");
    let x8: Vec<[f32; 8]> = (0..N).map(|i| core::array::from_fn(|j| -20.0 + ((i * 8 + j) % 4000) as f32 * 0.01)).collect();
    let p8: Vec<[f32; 8]> = (0..N).map(|i| core::array::from_fn(|j| 0.01 + ((i * 8 + j) % 997) as f32 * 0.1)).collect();
    let x4: Vec<[f32; 4]> = x8.iter().map(|a| [a[0], a[1], a[2], a[3]]).collect();
    let p4: Vec<[f32; 4]> = p8.iter().map(|a| [a[0], a[1], a[2], a[3]]).collect();
    let (mut o8, mut o4) = (vec![[0f32; 8]; N], vec![[0f32; 4]; N]);
    let names = ["avx2 exp2_midp", "avx2 exp_midp", "avx2 pow_midp", "avx2 exp2_lowp", "scalar exp2_midp", "scalar exp_midp", "scalar pow_midp", "scalar exp2_lowp"];
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); 8];
    for round in 0..200 {
        for k in 0..8 {
            let v = (k + round) % 8;
            let reps = if v < 4 { 64 } else { 8 };
            let start = Instant::now();
            for _ in 0..reps {
                match v {
                    0 => a_exp2(t, black_box(&x8), black_box(&mut o8)),
                    1 => a_exp(t, black_box(&x8), black_box(&mut o8)),
                    2 => a_pow(t, black_box(&p8), black_box(&mut o8)),
                    3 => a_exp2l(t, black_box(&x8), black_box(&mut o8)),
                    4 => s_exp2(black_box(&x4), black_box(&mut o4)),
                    5 => s_exp(black_box(&x4), black_box(&mut o4)),
                    6 => s_pow(black_box(&p4), black_box(&mut o4)),
                    _ => s_exp2l(black_box(&x4), black_box(&mut o4)),
                }
            }
            times[v].push(start.elapsed().as_nanos() as f64 / reps as f64);
        }
    }
    black_box((&o8, &o4));
    for (v, n) in names.iter().enumerate() {
        let mut s = times[v].clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{n:18} median {:9.0} ns per 1,024 vectors", s[s.len() / 2]);
    }
}
```

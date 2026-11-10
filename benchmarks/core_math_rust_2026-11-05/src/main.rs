//! Streams of 1024 values. std is a performance comparison, not a bit oracle.
use std::{hint::black_box, time::Duration};
use zenbench::prelude::*;
use zenmath::scalar as z;
const N: usize = 1024;
macro_rules! group {
    ($s:expr, $name:literal, $ty:ty, $lo:expr, $hi:expr, $core:expr, $std:expr) => {
        $s.group($name, |g| {
            g.throughput(Throughput::Elements(N as u64));
            g.config()
                .min_rounds(30)
                .max_rounds(100)
                .max_wall_time(Duration::from_secs(3));
            let input: Vec<$ty> = (0..N)
                .map(|i| $lo + ($hi - $lo) * ((i * 719 % N) as $ty / N as $ty))
                .collect();
            for (name, f) in [
                ("core", $core as fn($ty) -> $ty),
                ("std", $std as fn($ty) -> $ty),
            ] {
                let input = input.clone();
                let mut out = vec![0.0; N];
                g.bench(name, move |b| {
                    b.iter(|| {
                        for (o, x) in out.iter_mut().zip(black_box(&input)) {
                            *o = f(*x);
                        }
                        black_box(&out);
                    })
                });
            }
            g.baseline("core");
        });
    };
}
fn benches(s: &mut Suite) {
    group!(s, "exp2f", f32, -8.0, 8.0, z::exp2f, f32::exp2);
    group!(s, "expf", f32, -8.0, 8.0, z::expf, f32::exp);
    group!(s, "log2f", f32, 0.25, 4.0, z::log2f, f32::log2);
    group!(s, "logf", f32, 0.25, 4.0, z::logf, f32::ln);
    group!(s, "log10f", f32, 0.25, 4.0, z::log10f, f32::log10);
    group!(s, "cbrtf", f32, -8.0, 8.0, z::cbrtf, f32::cbrt);
    group!(s, "powf", f32, 0.25, 4.0, |x| z::powf(x, 2.4), |x: f32| x
        .powf(2.4));
    group!(s, "sinf", f32, -8.0, 8.0, z::sinf, f32::sin);
    group!(s, "cosf", f32, -8.0, 8.0, z::cosf, f32::cos);
    group!(s, "tanf", f32, -8.0, 8.0, z::tanf, f32::tan);
    group!(s, "atanf", f32, -8.0, 8.0, z::atanf, f32::atan);
    group!(
        s,
        "atan2f",
        f32,
        -8.0,
        8.0,
        |x| z::atan2f(x, 2.4),
        |x: f32| x.atan2(2.4)
    );
    group!(s, "tanhf", f32, -8.0, 8.0, z::tanhf, f32::tanh);
    group!(s, "log1pf", f32, 0.25, 4.0, z::log1pf, f32::ln_1p);
    group!(s, "expm1f", f32, -8.0, 8.0, z::expm1f, f32::exp_m1);
    group!(s, "acosf", f32, -0.99, 0.99, z::acosf, f32::acos);
    group!(
        s,
        "hypotf",
        f32,
        0.25,
        4.0,
        |x| z::hypotf(x, 2.4),
        |x: f32| x.hypot(2.4)
    );
    group!(s, "cbrt", f64, -8.0, 8.0, z::cbrt, f64::cbrt);
    group!(s, "exp2", f64, -8.0, 8.0, z::exp2, f64::exp2);
    group!(s, "exp", f64, -8.0, 8.0, z::exp, f64::exp);
    group!(s, "log2", f64, 0.25, 4.0, z::log2, f64::log2);
    group!(s, "log", f64, 0.25, 4.0, z::log, f64::ln);
    group!(s, "log10", f64, 0.25, 4.0, z::log10, f64::log10);
    group!(s, "atan", f64, -8.0, 8.0, z::atan, f64::atan);
    group!(s, "log1p", f64, 0.25, 4.0, z::log1p, f64::ln_1p);
    group!(s, "sin", f64, -8.0, 8.0, z::sin, f64::sin);
    group!(s, "cos", f64, -8.0, 8.0, z::cos, f64::cos);
    group!(
        s,
        "atan2",
        f64,
        -8.0,
        8.0,
        |x| z::atan2(x, 2.4),
        |x: f64| x.atan2(2.4)
    );
    group!(s, "tan", f64, -8.0, 8.0, z::tan, f64::tan);
    group!(s, "pow", f64, 0.25, 4.0, |x| z::pow(x, 2.4), |x: f64| x
        .powf(2.4));
    group!(s, "atanh", f64, -0.99, 0.99, z::atanh, f64::atanh);
    group!(s, "tanh", f64, -8.0, 8.0, z::tanh, f64::tanh);
    group!(s, "asinh", f64, -8.0, 8.0, z::asinh, f64::asinh);
}
fn main() {
    let path = std::env::args().nth(1).expect("result JSON path required");
    let result = zenbench::run(benches);
    result.save(path).expect("save measurements");
}

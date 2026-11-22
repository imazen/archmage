//! Run with `cargo bench -p magemath --bench transcendentals --features "std avx512"`.
//! Inputs stay inside each operation's domain. Times include slice dispatch and
//! tails, with setup and allocation outside the timed region.

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use archmage::ScalarToken;
    use magemath::{
        cbrtf::{Cbrtf, CbrtfDomain, CbrtfOptions},
        expf::{Expf, ExpfDomain, ExpfOptions},
        log1pf::{Log1pf, Log1pfDomain, Log1pfOptions},
        logf::{Logf, LogfDomain, LogfOptions},
        powf::{Powf, PowfDomain, PowfOptions},
    };
    use std::hint::black_box;
    use zenbench::criterion_compat::{BenchmarkId, Criterion, Throughput};

    fn bench_slice(
        c: &mut Criterion,
        name: &str,
        input: impl Fn(f32) -> f32,
        scalar: impl Fn(&[f32], &mut [f32]),
        upgrade: impl Fn(&[f32], &mut [f32]),
    ) {
        let mut group = c.benchmark_group(name);
        for n in [4, 64, 4096] {
            let src: Vec<_> = (0..n)
                .map(|i| input(((i * 997) % n) as f32 / n as f32))
                .collect();
            let mut dst = vec![0.0; n];
            group.throughput(Throughput::Elements(n as u64));
            for (tier, run) in [
                ("scalar", &scalar as &dyn Fn(&[f32], &mut [f32])),
                ("upgrade", &upgrade as &dyn Fn(&[f32], &mut [f32])),
            ] {
                group.bench_function(BenchmarkId::new(tier, n), |b| {
                    b.iter(|| {
                        run(black_box(&src), black_box(&mut dst));
                        black_box(&dst);
                    });
                });
            }
        }
        group.finish();
    }

    macro_rules! benchmark {
        ($c:ident, $name:literal, $input:expr, $function:ident, $opts:expr, $method:ident $(, $arg:expr)?) => {{
            bench_slice(
                $c,
                concat!($name, "/bit_exact"),
                $input,
                |src, dst| {
                    $function::<{ ($opts).bit_exact().caller().bits() }>::$method(ScalarToken, src, dst $(, $arg)?).unwrap();
                },
                |src, dst| {
                    $function::<{ ($opts).bit_exact().upgrade().bits() }>::$method(ScalarToken, src, dst $(, $arg)?).unwrap();
                },
            );
            bench_slice(
                $c,
                concat!($name, "/backend_dependent"),
                $input,
                |src, dst| {
                    $function::<{ ($opts).backend_dependent().caller().bits() }>::$method(ScalarToken, src, dst $(, $arg)?).unwrap();
                },
                |src, dst| {
                    $function::<{ ($opts).backend_dependent().upgrade().bits() }>::$method(ScalarToken, src, dst $(, $arg)?).unwrap();
                },
            );
        }};
    }

    pub fn transcendentals(c: &mut Criterion) {
        const UNIT: PowfOptions = PowfOptions::new()
            .domain(PowfDomain::Unit)
            .const_exponent(1.0 / 2.4);
        const HDR: PowfOptions = PowfOptions::new().domain(PowfDomain::HdrGamma);
        benchmark!(
            c,
            "powf/unit_const_midp",
            |x| x,
            Powf,
            UNIT.midp(),
            slice_const_with_token
        );
        benchmark!(
            c,
            "powf/hdr_gamma_midp",
            |x| x * 4.0,
            Powf,
            HDR.midp(),
            slice_with_token,
            black_box(2.2)
        );
        benchmark!(
            c,
            "powf/hdr_gamma_highp",
            |x| x * 4.0,
            Powf,
            HDR.highp(),
            slice_with_token,
            black_box(2.2)
        );
        benchmark!(
            c,
            "exp2/finite_midp",
            |x| x * 16.0 - 8.0,
            Expf,
            ExpfOptions::new().domain(ExpfDomain::Finite).midp(),
            exp2_slice_with_token
        );
        benchmark!(
            c,
            "log2/positive_normals_midp",
            |x| x + 0.01,
            Logf,
            LogfOptions::new()
                .domain(LogfDomain::PositiveNormals)
                .midp(),
            log2_slice_with_token
        );
        benchmark!(
            c,
            "log1p/around_zero_midp",
            |x| x * 0.5 - 0.25,
            Log1pf,
            Log1pfOptions::new().domain(Log1pfDomain::AroundZero).midp(),
            slice_with_token
        );
        benchmark!(
            c,
            "cbrtf/bounded_midp",
            |x| x * 16.0 - 8.0,
            Cbrtf,
            CbrtfOptions::new().domain(CbrtfDomain::Bounded).midp(),
            slice_with_token
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
zenbench::criterion_group!(benches, native::transcendentals);
#[cfg(not(target_arch = "wasm32"))]
zenbench::criterion_main!(benches);

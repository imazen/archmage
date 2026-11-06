//! What each `magemath` surface costs, per tier, against the others and against
//! magetypes' methods.
//!
//! Each group streams 1,024 `[f32; 8]` values, L1-resident, through one form; the
//! forms are listed in `support/forms.rs`, and `fused` is the baseline. Scalar-tier
//! groups add the `zenmath` mirrors per lane (`mirror_fused`, `mirror_unfused`) and
//! `std`. `exp2_midp feedback` runs the kernels with each result feeding the next
//! call. `exp2_midp slice tail` compares the padded tail against a scalar one.
//! zenbench does not build for wasm32: see `examples/wasm_cost.rs`.
//!
//! Run (no `-Ctarget-cpu=native`, not on a machine doing other work):
//!   cargo bench -p magemath --bench portable_cost

#[cfg(not(target_arch = "wasm32"))]
#[path = "support/forms.rs"]
mod forms;

#[cfg(not(target_arch = "wasm32"))]
use archmage::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use forms::*;
#[cfg(not(target_arch = "wasm32"))]
use std::hint::black_box;
#[cfg(not(target_arch = "wasm32"))]
use zenbench::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
const N: usize = 1024;

#[cfg(not(target_arch = "wasm32"))]
type Form<T> = fn(T, &[[f32; 8]], &mut [[f32; 8]]);
#[cfg(not(target_arch = "wasm32"))]
type PerLane = fn(f32) -> f32;

/// Generates a `fn(T, input, out)` calling `$family` on exactly `T`'s tier.
#[cfg(not(target_arch = "wasm32"))]
macro_rules! on_tier {
    ($($family:ident)::+) => {
        |t: T, input: &[[f32; 8]], out: &mut [[f32; 8]]| {
            attuned!(
                $($family)::+(input, out),
                [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar],
                using(t)
            )
        }
    };
}

#[cfg(not(target_arch = "wasm32"))]
fn group<T: IntoConcreteToken + Copy + Send + Sync + 'static>(
    suite: &mut Suite,
    label: String,
    token: T,
    input: Vec<[f32; 8]>,
    forms: Vec<(&'static str, Form<T>)>,
    per_lane: Vec<(&'static str, PerLane)>,
) {
    suite.group(label, |g| {
        g.throughput(Throughput::Elements((N * 8) as u64));
        for (name, form) in forms {
            let input = input.clone();
            let mut out = vec![[0.0f32; 8]; N];
            g.bench(name, move |b| {
                b.iter(|| {
                    form(token, black_box(&input), &mut out);
                    black_box(out[0])
                })
            });
        }
        for (name, f) in per_lane {
            let input = input.clone();
            let mut out = vec![[0.0f32; 8]; N];
            g.bench(name, move |b| {
                b.iter(|| {
                    for (o, i) in out.iter_mut().zip(black_box(&input)) {
                        *o = core::array::from_fn(|j| f(i[j]));
                    }
                    black_box(out[0])
                })
            });
        }
        g.baseline("fused");
    });
}

#[cfg(not(target_arch = "wasm32"))]
macro_rules! op_group {
    ($suite:expr, $tier:expr, $token:expr, $with_per_lane:expr, $op:ident, $input:expr, $fused:expr, $unfused:expr, $std:expr) => {
        group(
            $suite,
            format!("{}, {}", stringify!($op), $tier),
            $token,
            $input,
            vec![
                ("fused", on_tier!($op::fused)),
                ("fused_unchecked", on_tier!($op::fused_unchecked)),
                ("unfused", on_tier!($op::unfused)),
                ("unfused_unchecked", on_tier!($op::unfused_unchecked)),
                ("relaxed", on_tier!($op::relaxed)),
                ("relaxed_unchecked", on_tier!($op::relaxed_unchecked)),
                ("method", on_tier!($op::method)),
            ],
            if $with_per_lane {
                let lanes: Vec<(&'static str, PerLane)> = vec![
                    ("mirror_fused", $fused),
                    ("mirror_unfused", $unfused),
                    ("std", $std),
                ];
                lanes
            } else {
                vec![]
            },
        );
    };
}

#[cfg(not(target_arch = "wasm32"))]
fn tier_groups<T: IntoConcreteToken + Copy + Send + Sync + 'static>(
    suite: &mut Suite,
    tier: &str,
    token: T,
    with_per_lane: bool,
) {
    use zenmath::scalar as zs;
    use zenmath::scalar::unfused as zu;
    op_group!(
        suite,
        tier,
        token,
        with_per_lane,
        exp2_midp,
        inputs(-20.0, 20.0, N),
        zs::exp2_midp_f32,
        zu::exp2_midp_f32,
        f32::exp2
    );
    op_group!(
        suite,
        tier,
        token,
        with_per_lane,
        log2_midp,
        inputs(0.001, 1000.0, N),
        zs::log2_midp_f32,
        zu::log2_midp_f32,
        f32::log2
    );
    op_group!(
        suite,
        tier,
        token,
        with_per_lane,
        pow_midp,
        inputs(0.001, 1.0, N),
        |x| zs::pow_midp_f32(x, 2.4),
        |x| zu::pow_midp_f32(x, 2.4),
        |x| x.powf(2.4)
    );
    op_group!(
        suite,
        tier,
        token,
        with_per_lane,
        exp2_lowp,
        inputs(-20.0, 20.0, N),
        zs::exp2_lowp_f32,
        zu::exp2_lowp_f32,
        f32::exp2
    );
    op_group!(
        suite,
        tier,
        token,
        with_per_lane,
        log2_lowp,
        inputs(0.001, 1000.0, N),
        zs::log2_lowp_f32,
        zu::log2_lowp_f32,
        f32::log2
    );
    group(
        suite,
        format!("exp2_midp feedback, {tier}"),
        token,
        inputs(0.5, 0.6, N),
        vec![
            ("fused", on_tier!(feedback_fused)),
            ("unfused", on_tier!(feedback_unfused)),
        ],
        vec![],
    );
}

/// The shipped slice body against the alternatives its shape was chosen over:
/// a separate loop and tail, and a tail through the scalar mirror.
#[cfg(not(target_arch = "wasm32"))]
mod tails {
    use archmage::prelude::*;
    use magemath::portable::kernels;

    #[attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
    pub fn loop_then_pad(token: Token, data: &mut [f32]) {
        let (chunks, tail) = f32x8::partition_slice_mut_t(token, data);
        for chunk in chunks {
            kernels::exp2_midp(token, f32x8::load_t(token, chunk)).store(chunk);
        }
        if !tail.is_empty() {
            let mut pad = [0.0f32; 8];
            pad[..tail.len()].copy_from_slice(tail);
            let out = kernels::exp2_midp(token, f32x8::load_t(token, &pad)).to_array();
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }

    #[attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
    pub fn loop_then_mirror(token: Token, data: &mut [f32]) {
        let (chunks, tail) = f32x8::partition_slice_mut_t(token, data);
        for chunk in chunks {
            kernels::exp2_midp(token, f32x8::load_t(token, chunk)).store(chunk);
        }
        for x in tail {
            *x = zenmath::scalar::exp2_midp_f32(*x);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn tail_groups(suite: &mut Suite) {
    use magemath::portable::slice::exp2_midp_f32;
    for len in [1usize, 7, 8, 9, 15, 1031] {
        let input: Vec<f32> = (0..len).map(|i| i as f32 * 0.01 - 5.0).collect();
        suite.group(format!("exp2_midp slice tail, len {len}"), |g| {
            g.throughput(Throughput::Elements(len as u64));
            type SliceFn = fn(&mut [f32]);
            let shapes: [(&str, SliceFn); 3] = [
                ("one loop, pad", exp2_midp_f32),
                ("loop then pad", |d| {
                    attuned!(
                        tails::loop_then_pad(d),
                        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
                    )
                }),
                ("loop then mirror", |d| {
                    attuned!(
                        tails::loop_then_mirror(d),
                        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
                    )
                }),
            ];
            for (name, f) in shapes {
                let mut data = input.clone();
                let input = input.clone();
                g.bench(name, move |b| {
                    b.iter(|| {
                        data.copy_from_slice(black_box(&input));
                        f(black_box(&mut data));
                        black_box(data[0])
                    })
                });
            }
            g.baseline("one loop, pad");
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_portable(suite: &mut Suite) {
    #[cfg(target_arch = "x86_64")]
    match X64V3Token::summon() {
        Some(v3) => tier_groups(suite, "V3", v3, false),
        None => eprintln!("skipped V3 groups: this CPU lacks x86-64-v3"),
    }
    #[cfg(target_arch = "aarch64")]
    match NeonToken::summon() {
        Some(neon) => tier_groups(suite, "NEON", neon, false),
        None => eprintln!("skipped NEON groups: NEON not detected"),
    }
    tier_groups(suite, "scalar tier", ScalarToken, true);
    tail_groups(suite);
}

#[cfg(not(target_arch = "wasm32"))]
zenbench::main!(bench_portable);

#[cfg(target_arch = "wasm32")]
fn main() {
    panic!("zenbench is native-only; run just magemath-bench-wasm for WASM measurements");
}

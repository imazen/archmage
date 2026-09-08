//! Reproducible baseline/fixup pairs and bit-exact observations for the docs.
//! Run in release mode; --samples-only also works under Wasmtime/QEMU.
//! Baselines deliberately lack the advertised exceptional-input contract.
//! OP and FIX are consts: selection is outside the measured vector loop.
use archmage::{ScalarToken, SimdToken};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

type Kernel = dyn Fn(&[u32], &mut [u32]);

fn elapsed(f: &Kernel, input: &[u32], output: &mut [u32], iterations: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        f(black_box(input), black_box(output));
        black_box(&*output);
    }
    start.elapsed().as_secs_f64() * 1e9 / iterations as f64
}

fn pair(
    isa: &str,
    method: &str,
    width: usize,
    input: &[u32],
    baseline: &Kernel,
    fixed: &Kernel,
    bench: bool,
) {
    let mut a = vec![0; input.len()];
    let mut b = a.clone();
    baseline(input, &mut a);
    fixed(input, &mut b);
    // Integer arrays are JSON-safe, including NaNs and negative zero as f32 bits.
    println!(
        "{{\"kind\":\"sample\",\"isa\":\"{isa}\",\"method\":\"{method}\",\"lanes\":{width},\"input\":{input:?},\"baseline\":{a:?},\"fixed\":{b:?}}}"
    );
    if !bench {
        return;
    }
    // Positive, normal, in-range values: compare the repair premium on ordinary
    // data. Exceptional-input behavior is sampled separately above, never timed.
    let data: Vec<u32> = (0..2048)
        .map(|i| (0.5 + (i % 997) as f32 / 16.0).to_bits())
        .collect();
    let mut output = vec![0; data.len()];
    let mut iterations = 1;
    loop {
        let start = Instant::now();
        elapsed(fixed, &data, &mut output, iterations);
        if start.elapsed() >= Duration::from_millis(15) {
            break;
        }
        iterations *= 2;
    }
    let vectors = (data.len() / width) as f64;
    let mut rounds = Vec::new();
    for round in 0..9 {
        // AB/BA ordering limits drift. Both arms use identical traffic and calls.
        let (a, b) = if round % 2 == 0 {
            let a = elapsed(baseline, &data, &mut output, iterations);
            (a, elapsed(fixed, &data, &mut output, iterations))
        } else {
            let b = elapsed(fixed, &data, &mut output, iterations);
            (elapsed(baseline, &data, &mut output, iterations), b)
        };
        rounds.push([a / vectors, b / vectors]);
    }
    println!(
        "{{\"kind\":\"timing\",\"isa\":\"{isa}\",\"method\":\"{method}\",\"lanes\":{width},\"iterations\":{iterations},\"ns_per_vector_pairs\":{rounds:?}}}"
    );
}

macro_rules! run_tier {
    ($module:ident, $token:expr, $label:expr, $width:expr, $samples:expr, $bench:expr) => {{
        let token = $token;
        macro_rules! one {
            ($op:expr, $name:expr, $timed:expr) => {
                pair(
                    $label,
                    $name,
                    $width,
                    &$samples,
                    &move |a, b| $module::run::<$op, false>(token, a, b),
                    &move |a, b| $module::run::<$op, true>(token, a, b),
                    $bench && $timed,
                );
            };
        }
        one!(0, "to_i32_saturating()", true);
        one!(1, "recip()", true);
        one!(2, "rsqrt()", true);
        one!(3, "min(splat(1.0))", false);
        one!(4, "max(splat(1.0))", false);
        one!(5, "simd_ne(splat(1.0))", false);
        one!(6, "-v", false);
        one!(7, "round()", false);
        one!(8, "splat(1.0).min(v)", false);
        one!(9, "splat(1.0).max(v)", false);
        one!(10, "recip_portable()", false);
        one!(11, "rsqrt_portable()", false);
    }};
}

#[cfg(target_arch = "x86_64")]
mod v3 {
    use archmage::{X64V3Token, arcane};
    use magetypes::simd::generic::f32x4;
    type V = f32x4<X64V3Token>;
    #[arcane(import_intrinsics)]
    pub fn run<const OP: u8, const FIX: bool>(
        token: X64V3Token,
        input: &[u32],
        output: &mut [u32],
    ) {
        for (a, b) in input
            .as_chunks::<4>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<4>().0.iter_mut())
        {
            let v = V::from_array(token, core::array::from_fn(|i| f32::from_bits(a[i])));
            let one = V::splat(token, 1.0);
            let result = match OP {
                0 => {
                    let i = if FIX {
                        v.to_i32_saturating()
                    } else {
                        v.to_i32()
                    };
                    i.to_array().map(|x| x as u32)
                }
                1 => {
                    let r = if FIX {
                        v.recip()
                    } else {
                        let a = _mm_loadu_ps(&v.to_array());
                        let r = _mm_rcp_ps(a);
                        let e = _mm_fnmadd_ps(a, r, _mm_set1_ps(1.0));
                        let refined = _mm_fmadd_ps(r, e, r);
                        let mut output = [0.0; 4];
                        _mm_storeu_ps(&mut output, refined);
                        V::from_array(token, output)
                    };
                    r.to_array().map(f32::to_bits)
                }
                2 => {
                    let r = if FIX {
                        v.rsqrt()
                    } else {
                        let a = _mm_loadu_ps(&v.to_array());
                        let y = _mm_rsqrt_ps(a);
                        let t = _mm_mul_ps(a, _mm_mul_ps(y, y));
                        let refined =
                            _mm_mul_ps(y, _mm_fnmadd_ps(_mm_set1_ps(0.5), t, _mm_set1_ps(1.5)));
                        let mut output = [0.0; 4];
                        _mm_storeu_ps(&mut output, refined);
                        V::from_array(token, output)
                    };
                    r.to_array().map(f32::to_bits)
                }
                3 => v.min(one).to_array().map(f32::to_bits),
                4 => v.max(one).to_array().map(f32::to_bits),
                5 => v.simd_ne(one).to_array().map(|x| x as u32),
                6 => (-v).to_array().map(f32::to_bits),
                7 => v.round().to_array().map(f32::to_bits),
                8 => one.min(v).to_array().map(f32::to_bits),
                9 => one.max(v).to_array().map(f32::to_bits),
                10 => v.recip_portable().to_array().map(f32::to_bits),
                11 => v.rsqrt_portable().to_array().map(f32::to_bits),
                _ => unreachable!(),
            };
            b.copy_from_slice(&result);
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod v4 {
    use archmage::{X64V4Token, arcane};
    use magetypes::simd::generic::f32x16;
    type V = f32x16<X64V4Token>;
    #[arcane(import_intrinsics)]
    pub fn run<const OP: u8, const FIX: bool>(
        token: X64V4Token,
        input: &[u32],
        output: &mut [u32],
    ) {
        for (a, b) in input
            .as_chunks::<16>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<16>().0.iter_mut())
        {
            let v = V::from_array(token, core::array::from_fn(|i| f32::from_bits(a[i])));
            let one = V::splat(token, 1.0);
            let result = match OP {
                0 => {
                    let i = if FIX {
                        v.to_i32_saturating()
                    } else {
                        v.to_i32()
                    };
                    i.to_array().map(|x| x as u32)
                }
                1 => {
                    let r = if FIX {
                        v.recip()
                    } else {
                        let a = _mm512_loadu_ps(&v.to_array());
                        let r = _mm512_rcp14_ps(a);
                        let e = _mm512_fnmadd_ps(a, r, _mm512_set1_ps(1.0));
                        let refined = _mm512_fmadd_ps(r, e, r);
                        let mut output = [0.0; 16];
                        _mm512_storeu_ps(&mut output, refined);
                        V::from_array(token, output)
                    };
                    r.to_array().map(f32::to_bits)
                }
                2 => {
                    let r = if FIX {
                        v.rsqrt()
                    } else {
                        let a = _mm512_loadu_ps(&v.to_array());
                        let y = _mm512_rsqrt14_ps(a);
                        let t = _mm512_mul_ps(a, _mm512_mul_ps(y, y));
                        let refined = _mm512_mul_ps(
                            y,
                            _mm512_fnmadd_ps(_mm512_set1_ps(0.5), t, _mm512_set1_ps(1.5)),
                        );
                        let mut output = [0.0; 16];
                        _mm512_storeu_ps(&mut output, refined);
                        V::from_array(token, output)
                    };
                    r.to_array().map(f32::to_bits)
                }
                3 => v.min(one).to_array().map(f32::to_bits),
                4 => v.max(one).to_array().map(f32::to_bits),
                5 => v.simd_ne(one).to_array().map(|x| x as u32),
                6 => (-v).to_array().map(f32::to_bits),
                7 => v.round().to_array().map(f32::to_bits),
                8 => one.min(v).to_array().map(f32::to_bits),
                9 => one.max(v).to_array().map(f32::to_bits),
                10 => v.recip_portable().to_array().map(f32::to_bits),
                11 => v.rsqrt_portable().to_array().map(f32::to_bits),
                _ => unreachable!(),
            };
            b.copy_from_slice(&result);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use archmage::{NeonToken, arcane};
    use magetypes::simd::generic::f32x4;
    type V = f32x4<NeonToken>;
    #[arcane]
    pub fn run<const OP: u8, const FIX: bool>(token: NeonToken, input: &[u32], output: &mut [u32]) {
        for (a, b) in input
            .as_chunks::<4>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<4>().0.iter_mut())
        {
            let v = V::from_array(token, core::array::from_fn(|i| f32::from_bits(a[i])));
            let one = V::splat(token, 1.0);
            let result = match OP {
                0 => {
                    let i = if FIX {
                        v.to_i32_saturating()
                    } else {
                        v.to_i32()
                    };
                    i.to_array().map(|x| x as u32)
                }
                1 => {
                    let r = v.recip();
                    r.to_array().map(f32::to_bits)
                }
                2 => {
                    let r = v.rsqrt();
                    r.to_array().map(f32::to_bits)
                }
                3 => v.min(one).to_array().map(f32::to_bits),
                4 => v.max(one).to_array().map(f32::to_bits),
                5 => v.simd_ne(one).to_array().map(|x| x as u32),
                6 => (-v).to_array().map(f32::to_bits),
                7 => v.round().to_array().map(f32::to_bits),
                8 => one.min(v).to_array().map(f32::to_bits),
                9 => one.max(v).to_array().map(f32::to_bits),
                10 => v.recip_portable().to_array().map(f32::to_bits),
                11 => v.rsqrt_portable().to_array().map(f32::to_bits),
                _ => unreachable!(),
            };
            b.copy_from_slice(&result);
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use archmage::{Wasm128Token, arcane};
    use magetypes::simd::generic::f32x4;
    type V = f32x4<Wasm128Token>;
    #[arcane]
    pub fn run<const OP: u8, const FIX: bool>(
        token: Wasm128Token,
        input: &[u32],
        output: &mut [u32],
    ) {
        for (a, b) in input
            .as_chunks::<4>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<4>().0.iter_mut())
        {
            let v = V::from_array(token, core::array::from_fn(|i| f32::from_bits(a[i])));
            let one = V::splat(token, 1.0);
            let result = match OP {
                0 => {
                    let i = if FIX {
                        v.to_i32_saturating()
                    } else {
                        v.to_i32()
                    };
                    i.to_array().map(|x| x as u32)
                }
                1 => {
                    let r = v.recip();
                    r.to_array().map(f32::to_bits)
                }
                2 => {
                    let r = v.rsqrt();
                    r.to_array().map(f32::to_bits)
                }
                3 => v.min(one).to_array().map(f32::to_bits),
                4 => v.max(one).to_array().map(f32::to_bits),
                5 => v.simd_ne(one).to_array().map(|x| x as u32),
                6 => (-v).to_array().map(f32::to_bits),
                7 => v.round().to_array().map(f32::to_bits),
                8 => one.min(v).to_array().map(f32::to_bits),
                9 => one.max(v).to_array().map(f32::to_bits),
                10 => v.recip_portable().to_array().map(f32::to_bits),
                11 => v.rsqrt_portable().to_array().map(f32::to_bits),
                _ => unreachable!(),
            };
            b.copy_from_slice(&result);
        }
    }
}

mod scalar {
    use archmage::ScalarToken;
    use magetypes::simd::generic::f32x4;
    type V = f32x4<ScalarToken>;
    #[inline(never)]
    pub fn run<const OP: u8, const FIX: bool>(
        token: ScalarToken,
        input: &[u32],
        output: &mut [u32],
    ) {
        for (a, b) in input
            .as_chunks::<4>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<4>().0.iter_mut())
        {
            let v = V::from_array(token, core::array::from_fn(|i| f32::from_bits(a[i])));
            let one = V::splat(token, 1.0);
            let result = match OP {
                0 => {
                    let i = if FIX {
                        v.to_i32_saturating()
                    } else {
                        v.to_i32()
                    };
                    i.to_array().map(|x| x as u32)
                }
                1 => {
                    let r = v.recip();
                    r.to_array().map(f32::to_bits)
                }
                2 => {
                    let r = v.rsqrt();
                    r.to_array().map(f32::to_bits)
                }
                3 => v.min(one).to_array().map(f32::to_bits),
                4 => v.max(one).to_array().map(f32::to_bits),
                5 => v.simd_ne(one).to_array().map(|x| x as u32),
                6 => (-v).to_array().map(f32::to_bits),
                7 => v.round().to_array().map(f32::to_bits),
                8 => one.min(v).to_array().map(f32::to_bits),
                9 => one.max(v).to_array().map(f32::to_bits),
                10 => v.recip_portable().to_array().map(f32::to_bits),
                11 => v.rsqrt_portable().to_array().map(f32::to_bits),
                _ => unreachable!(),
            };
            b.copy_from_slice(&result);
        }
    }
}

#[cfg(feature = "w512")]
mod scalar16 {
    use archmage::ScalarToken;
    use magetypes::simd::generic::f32x16;
    type V = f32x16<ScalarToken>;
    #[inline(never)]
    pub fn run<const OP: u8, const FIX: bool>(
        token: ScalarToken,
        input: &[u32],
        output: &mut [u32],
    ) {
        for (a, b) in input
            .as_chunks::<16>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<16>().0.iter_mut())
        {
            let v = V::from_array(token, core::array::from_fn(|i| f32::from_bits(a[i])));
            let one = V::splat(token, 1.0);
            let result = match OP {
                0 => {
                    let i = if FIX {
                        v.to_i32_saturating()
                    } else {
                        v.to_i32()
                    };
                    i.to_array().map(|x| x as u32)
                }
                1 => {
                    let r = v.recip();
                    r.to_array().map(f32::to_bits)
                }
                2 => {
                    let r = v.rsqrt();
                    r.to_array().map(f32::to_bits)
                }
                3 => v.min(one).to_array().map(f32::to_bits),
                4 => v.max(one).to_array().map(f32::to_bits),
                5 => v.simd_ne(one).to_array().map(|x| x as u32),
                6 => (-v).to_array().map(f32::to_bits),
                7 => v.round().to_array().map(f32::to_bits),
                8 => one.min(v).to_array().map(f32::to_bits),
                9 => one.max(v).to_array().map(f32::to_bits),
                10 => v.recip_portable().to_array().map(f32::to_bits),
                11 => v.rsqrt_portable().to_array().map(f32::to_bits),
                _ => unreachable!(),
            };
            b.copy_from_slice(&result);
        }
    }
}

fn main() {
    let bench = !std::env::args().any(|x| x == "--samples-only");
    assert!(
        !bench || !cfg!(debug_assertions),
        "timing requires --release"
    );
    // IEEE rails, payload-bearing NaNs, subnormal/normal boundaries, adjacent
    // conversion thresholds, signed ties, and ordinary values. Same bits per ISA.
    let mut samples = vec![
        0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345, 1, 0x80000001, 0x007fffff,
        0x00800000, 0x7f7fffff, 0xff7fffff, 0x4effffff, 0x4f000000, 0xcf000000, 0xcf000001,
    ];
    samples.extend(
        [
            1.0f32, -1.0, 1.9, -1.9, 2.5, 3.5, -2.5, -3.5, 0.5, 2.0, 4.0, 16.0, 0.125, 255.0,
            256.0, 65535.0,
        ]
        .map(f32::to_bits),
    );
    // Stratify all finite exponents with two signs and two mantissas. A compact
    // deterministic sample, not a claim of exhaustive floating-point coverage.
    for exponent in (1..255u32).step_by(17) {
        for sign in [0, 0x80000000] {
            for mantissa in [0, 0x00555555] {
                samples.push(sign | exponent << 23 | mantissa);
            }
        }
    }
    while samples.len() % 16 != 0 {
        samples.push(1.0f32.to_bits());
    }
    run_tier!(scalar, ScalarToken, "scalar/f32x4", 4, samples, bench);
    #[cfg(feature = "w512")]
    run_tier!(scalar16, ScalarToken, "scalar/f32x16", 16, samples, bench);
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = archmage::X64V3Token::summon() {
        run_tier!(v3, t, "x86-v3/f32x4", 4, samples, bench);
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = archmage::X64V4Token::summon() {
        run_tier!(v4, t, "x86-v4/f32x16", 16, samples, bench);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        run_tier!(neon, t, "neon/f32x4", 4, samples, bench);
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = archmage::Wasm128Token::summon() {
        run_tier!(wasm, t, "wasm/f32x4", 4, samples, bench);
    }
}

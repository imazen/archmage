//! Experimental range contracts. No public library API is added.
#![forbid(unsafe_code)]
use archmage::SimdToken;
#[path = "../frozen.rs"]
#[allow(dead_code)]
mod frozen;

const LABELS: [&str; 13] = [
    "main_exp2_midp",
    "main_exp2_lowp",
    "MIDP_finite_fused_assume_token",
    "MIDP_normal_fused_assume_token",
    "MIDP_normal_fused_checked_token",
    "MIDP_normal_relaxed_assume_token",
    "LOWP_finite_cubic_fused_assume_token",
    "LOWP_normal_cubic_fused_assume_token",
    "LOWP_normal_quadratic_fused_assume_token",
    "LOWP_normal_quadratic_fused_checked_token",
    "LOWP_normal_quadratic_relaxed_assume_token",
    "MIDP_public_BitExact_AssumeValid_Automatic",
    "LOWP_public_BitExact_AssumeValid_Automatic",
];

#[inline(always)]
#[allow(clippy::excessive_precision)] // Preserve the reviewed coefficient spellings.
fn kernel<
    T: magetypes::simd::backends::F32x8Convert,
    const LOW: bool,
    const MODE: u8,
    const FUSED: bool,
    const CHECKED: bool,
>(
    token: T,
    input: magetypes::simd::generic::f32x8<T>,
) -> Result<magetypes::simd::generic::f32x8<T>, u16> {
    use magetypes::simd::generic::{f32x8 as V, i32x8 as I};
    let sp = |x| V::splat_t(token, x);
    let ip = |x| I::splat_t(token, x);
    if CHECKED {
        let valid = (input.simd_ge(sp(-126.)) & input.simd_lt(sp(127.5))).bitcast_to_i32();
        if !valid.all_true() {
            let mask = valid
                .to_array()
                .iter()
                .enumerate()
                .fold(0u16, |m, (i, x)| m | (u16::from(*x == 0) << i));
            return Err(mask);
        }
    }
    let madd = |a: V<T>, b: V<T>, c: V<T>| {
        if FUSED {
            a.mul_add_fused(b, c)
        } else {
            a.mul_add_relaxed(b, c)
        }
    };
    let x = if MODE == 0 {
        input
            .max(sp(-126.))
            .min(sp(if LOW { 127.99 } else { 128. }))
    } else {
        input
    };
    let xi = if LOW && MODE != 2 {
        x.floor()
    } else if MODE == 0 {
        x.round().min(sp(127.))
    } else {
        x.round()
    };
    let xf = x - xi;
    let p = if LOW && MODE == 2 {
        let p = madd(sp(0.239625156), xf, sp(0.701271514));
        madd(p, xf, sp(1.))
    } else if LOW {
        let p = madd(sp(0.07191699702349297), xf, sp(0.23433638464483292));
        let p = madd(p, xf, sp(core::f32::consts::LN_2));
        madd(p, xf, sp(1.))
    } else {
        let p = madd(sp(0.00015252789649879), xf, sp(0.00133860472782621));
        let p = madd(p, xf, sp(0.00961871577914975));
        let p = madd(p, xf, sp(0.05550369554299311));
        let p = madd(p, xf, sp(0.240226463934714));
        let p = madd(p, xf, sp(core::f32::consts::LN_2));
        madd(p, xf, sp(1.))
    };
    let integer =
        if core::any::TypeId::of::<T>() == core::any::TypeId::of::<archmage::ScalarToken>() {
            xi.to_i32()
        } else {
            xi.to_i32_round()
        };
    let result = p * (integer + ip(127)).shl_const::<23>().bitcast_to_f32();
    Ok(if MODE == 0 && !LOW {
        let result = V::blend(input.simd_lt(sp(-126.)), sp(0.), result);
        V::blend(input.simd_ge(sp(128.)), sp(f32::INFINITY), result)
    } else {
        result
    })
}

#[archmage::magetypes(v3, neon, scalar)]
#[inline(never)]
fn loop8<const ARM: usize>(token: Token, input: &[f32], output: &mut [f32], chain: bool) {
    use frozen::x8::Frozen;
    use magetypes::simd::generic::f32x8 as V;
    let mut acc = V::splat_t(token, 0.75);
    for (xs, ys) in input
        .as_chunks::<8>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<8>().0)
    {
        let x = if chain {
            acc
        } else {
            V::from_array_t(token, *xs)
        };
        let y = match ARM {
            0 => x.old_exp2_midp(token),
            1 => x.old_exp2_lowp(token),
            2 => kernel::<_, false, 0, true, false>(token, x).unwrap(),
            3 => kernel::<_, false, 1, true, false>(token, x).unwrap(),
            4 => kernel::<_, false, 1, true, true>(token, x).unwrap(),
            5 => kernel::<_, false, 1, false, false>(token, x).unwrap(),
            6 => kernel::<_, true, 0, true, false>(token, x).unwrap(),
            7 => kernel::<_, true, 1, true, false>(token, x).unwrap(),
            8 => kernel::<_, true, 2, true, false>(token, x).unwrap(),
            9 => kernel::<_, true, 2, true, true>(token, x).unwrap(),
            10 => kernel::<_, true, 2, false, false>(token, x).unwrap(),
            11 => V::from_array_t(
                token,
                magetypes::expf::try_exp2_array_with::<32, 8>(x.to_array()).unwrap(),
            ),
            12 => V::from_array_t(
                token,
                magetypes::expf::try_exp2_array_with::<40, 8>(x.to_array()).unwrap(),
            ),
            _ => panic!("arm"),
        };
        *ys = y.to_array();
        if chain {
            acc = V::from_array_t(
                token,
                y.to_array()
                    .map(|v| f32::from_bits(0x3f000000 | (v.to_bits() & 0x007fffff))),
            );
        }
    }
}
fn execute<const ARM: usize>(context: &str, input: &[f32], output: &mut [f32], chain: bool) {
    match context {
        "scalar" => loop8_scalar::<ARM>(archmage::ScalarToken, input, output, chain),
        #[cfg(target_arch = "x86_64")]
        "native" => loop8_v3::<ARM>(
            archmage::X64V3Token::summon().expect("V3 required"),
            input,
            output,
            chain,
        ),
        #[cfg(target_arch = "aarch64")]
        "native" => loop8_neon::<ARM>(
            archmage::NeonToken::summon().expect("NEON required"),
            input,
            output,
            chain,
        ),
        _ => panic!("context"),
    }
}
fn main() {
    if std::env::var_os("NORMAL_EVAL").is_some() {
        evaluate();
        return;
    }
    let cell = std::env::var("AUDIT_CELL").expect("cell");
    let p: Vec<_> = cell.split('/').collect();
    assert_eq!(p.len(), 6);
    let context = p[3];
    let size: usize = p[4].parse().unwrap();
    let chain = p[5] == "chain";
    let output = std::env::var("AUDIT_RESULT").expect("output");
    assert!(!std::path::Path::new(&output).exists());
    zenbench::run_gated(zenbench::GateConfig::strict(), |s| {
        s.group(&cell, |g| {
            g.config()
                .min_rounds(50)
                .max_rounds(50)
                .auto_rounds(false)
                .warmup_time(std::time::Duration::from_millis(10))
                .max_time(std::time::Duration::from_millis(100))
                .max_wall_time(std::time::Duration::from_secs(180));
            macro_rules! arm {
                ($n:literal) => {{
                    let input: Vec<f32> =
                        (0..size).map(|i| 0.5 + (i % 256) as f32 / 512.).collect();
                    let context = context.to_owned();
                    g.bench(LABELS[$n], move |b| {
                        let mut output = vec![0.; size];
                        b.iter(|| {
                            execute::<$n>(
                                &context,
                                std::hint::black_box(&input),
                                std::hint::black_box(&mut output),
                                chain,
                            )
                        });
                        std::hint::black_box(output);
                    });
                }};
            }
            arm!(0);
            arm!(1);
            arm!(2);
            arm!(3);
            arm!(4);
            arm!(5);
            arm!(6);
            arm!(7);
            arm!(8);
            arm!(9);
            arm!(10);
            arm!(11);
            arm!(12);
        })
    })
    .save(std::path::Path::new(&output))
    .unwrap();
}
fn evaluate() {
    use std::io::Read;
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let context = std::env::var("NORMAL_CONTEXT").unwrap_or_else(|_| "native".into());
    for row in text.lines() {
        let bits = u32::from_str_radix(row, 16).unwrap();
        let x = f32::from_bits(bits);
        let mut values = Vec::new();
        macro_rules! value {
            ($arm:literal) => {{
                let mut out = [0.; 8];
                execute::<$arm>(&context, &[x; 8], &mut out, false);
                assert!(out.iter().all(|v| v.to_bits() == out[0].to_bits()));
                values.push(out[0].to_bits());
            }};
        }
        value!(2);
        value!(3);
        value!(4);
        value!(5);
        value!(6);
        value!(7);
        value!(8);
        value!(9);
        value!(10);
        print!("{bits:08x}");
        for v in values {
            print!(" {v:08x}");
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_range_rejects_every_unsupported_lane() {
        for x in [
            f32::from_bits(0xc2fc0001),
            127.5,
            128.,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0x7f800001),
        ] {
            for lane in 0..8 {
                let mut xs = [0.; 8];
                xs[lane] = x;
                let v = magetypes::simd::generic::f32x8::from_array_t(archmage::ScalarToken, xs);
                assert_eq!(
                    kernel::<_, false, 1, true, true>(archmage::ScalarToken, v).unwrap_err(),
                    1 << lane
                );
                assert_eq!(
                    kernel::<_, true, 2, true, true>(archmage::ScalarToken, v).unwrap_err(),
                    1 << lane
                );
            }
        }
        let xs = [
            -126.,
            -0.,
            0.,
            1.,
            127.,
            f32::from_bits(0x42feffff),
            f32::from_bits(1),
            -f32::from_bits(1),
        ];
        let v = magetypes::simd::generic::f32x8::from_array_t(archmage::ScalarToken, xs);
        assert!(kernel::<_, false, 1, true, true>(archmage::ScalarToken, v).is_ok());
        assert!(kernel::<_, true, 2, true, true>(archmage::ScalarToken, v).is_ok());
    }
}

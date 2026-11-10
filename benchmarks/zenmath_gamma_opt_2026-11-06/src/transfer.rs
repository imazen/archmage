//! Adapted scalar C0 composition from linear-srgb8900ffc87cd50b249bae19c99d874431e055d888.
use crate::{generated as g, kernel, piecewise};
#[inline(always)]
fn power<const K: usize, const METHOD: usize>(x: f32) -> f32 {
    let p = kernel::exponent::<K>();
    match METHOD {
        0 => x.powf(p),
        1 => zenmath::scalar::reference::powf(x, p),
        2 => zenmath::scalar::powf_with::<{ zenmath::PowBehavior::MIDP.to_u64() }>(x, p),
        3 => zenmath::scalar::powf_with::<
            {
                zenmath::PowBehavior::MIDP
                    .range(zenmath::PowRange::PositiveNormalPrecondition)
                    .to_u64()
            },
        >(x, p),
        4 => piecewise::evaluate::<K, 4, 1>(x).value,
        5 => piecewise::evaluate::<K, 5, 1>(x).value,
        6 => {
            if K == 0 {
                kernel::evaluate::<K, 13, 1>(x).value
            } else {
                kernel::evaluate::<K, 15, 1>(x).value
            }
        }
        _ => panic!("transfer method"),
    }
}
#[inline(always)]
pub(crate) fn evaluate<const K: usize, const METHOD: usize>(x: f32) -> f32 {
    if K == 0 {
        if x < 0.0 {
            0.0
        } else if x < f32::from_bits(g::SRGB_CUTOFF_BITS) {
            x * f32::from_bits(g::LINEAR_SCALE_BITS)
        } else if x < 1.0 {
            power::<K, METHOD>((x + f32::from_bits(g::A_BITS)) / f32::from_bits(g::AP1_BITS))
        } else {
            1.0
        }
    } else {
        if x < 0.0 {
            0.0
        } else if x < f32::from_bits(g::LINEAR_CUTOFF_BITS) {
            x * 12.92f32
        } else if x < 1.0 {
            f32::from_bits(g::AP1_BITS) * power::<K, METHOD>(x) - f32::from_bits(g::A_BITS)
        } else {
            1.0
        }
    }
}
/// Distinct f64 constant/exponent curve control; not the exact-f32-power oracle.
pub(crate) fn f64_curve<const K: usize>(x: f32) -> f64 {
    let x = f64::from(x);
    let a = f64::from_bits(g::A_F64_BITS);
    let ap = f64::from_bits(g::AP1_F64_BITS);
    if K == 0 {
        if x < 0.0 {
            0.0
        } else if x < f64::from_bits(g::SRGB_CUTOFF_F64_BITS) {
            x * (1.0f64 / 12.92)
        } else if x < 1.0 {
            ((x + a) / ap).powf(2.4f64)
        } else {
            1.0
        }
    } else {
        if x < 0.0 {
            0.0
        } else if x < f64::from_bits(g::LINEAR_CUTOFF_F64_BITS) {
            x * 12.92f64
        } else if x < 1.0 {
            ap * x.powf(1.0f64 / 2.4) - a
        } else {
            1.0
        }
    }
}

pub(crate) fn oracle(
    input: &std::path::Path,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::{
        fs::File,
        io::{BufRead, BufReader, Write},
    };
    let mut counts = [0usize; 2];
    let mut maximum = [0.0f64; 2];
    let mut curve_std_maximum = [0.0f64; 2];
    for (i, line) in BufReader::new(File::open(input)?).lines().enumerate() {
        let line = line?;
        if i == 0 {
            if line != "kind,x_bits,expected_f32_bits,curve_lo_f64_bits,curve_hi_f64_bits,label" {
                return Err("transfer header".into());
            }
            continue;
        }
        let f: Vec<_> = line.split(',').collect();
        if f.len() != 6 {
            return Err("transfer fields".into());
        }
        let k: usize = f[0].parse()?;
        if k > 1 {
            return Err("kind".into());
        }
        let xb = u32::from_str_radix(f[1], 16)?;
        let x = f32::from_bits(xb);
        let expected = u32::from_str_radix(f[2], 16)?;
        let lo = f64::from_bits(u64::from_str_radix(f[3], 16)?);
        let hi = f64::from_bits(u64::from_str_radix(f[4], 16)?);
        let mut check = |value: f32| {
            assert_eq!(
                value.to_bits(),
                expected,
                "f32 C0 composition k={k} x={xb:08x}"
            );
            maximum[k] = maximum[k].max(
                (f64::from(value) - lo)
                    .abs()
                    .max((f64::from(value) - hi).abs()),
            );
        };
        if k == 0 {
            check(evaluate::<0, 1>(x));
            check(evaluate::<0, 4>(x));
            check(evaluate::<0, 5>(x));
            check(evaluate::<0, 6>(x));
        } else {
            check(evaluate::<1, 1>(x));
            check(evaluate::<1, 4>(x));
            check(evaluate::<1, 5>(x));
            check(evaluate::<1, 6>(x));
        }
        let curve = if k == 0 {
            f64_curve::<0>(x)
        } else {
            f64_curve::<1>(x)
        };
        curve_std_maximum[k] = curve_std_maximum[k].max((curve - lo).abs().max((curve - hi).abs()));
        counts[k] += 1;
    }
    let mut out = File::create_new(output)?;
    writeln!(
        out,
        "kind,records,f32_composition_max_abs_error_vs_MPFR_f64_constant_curve,f64_std_curve_max_abs_error_vs_MPFR_curve"
    )?;
    for k in 0..2 {
        if counts[k] == 0 {
            return Err("empty transfer oracle".into());
        }
        writeln!(
            out,
            "{k},{},{},{}",
            counts[k], maximum[k], curve_std_maximum[k]
        )?;
    }
    println!(
        "C0 f32 compositions exact MPFR stage-rounded oracle PASS; f64-constant curve error is a separate sampled metric"
    );
    Ok(())
}

#![forbid(unsafe_code)]
mod generated;
mod kernel;
#[cfg(target_arch = "x86_64")]
mod timing;
use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader, Write},
    path::Path,
};

fn facts() {
    for (name, bits) in [
        ("linear_cutoff", u64::from(generated::LINEAR_CUTOFF_BITS)),
        ("srgb_cutoff", u64::from(generated::SRGB_CUTOFF_BITS)),
        ("A", u64::from(generated::A_BITS)),
        ("Aplus1", u64::from(generated::AP1_BITS)),
        ("linear_scale", u64::from(generated::LINEAR_SCALE_BITS)),
        ("decode_min", u64::from(generated::DECODE_MIN_BITS)),
        ("encode_min", u64::from(generated::ENCODE_MIN_BITS)),
        ("decode_p", u64::from(generated::DECODE_P_BITS)),
        ("encode_p", u64::from(generated::ENCODE_P_BITS)),
        ("linear_cutoff_f64", generated::LINEAR_CUTOFF_F64_BITS),
        ("srgb_cutoff_f64", generated::SRGB_CUTOFF_F64_BITS),
        ("A_f64", generated::A_F64_BITS),
        ("Aplus1_f64", generated::AP1_F64_BITS),
    ] {
        println!("{name}={bits:016x}");
    }
    for (name, start) in [
        ("decode", generated::DECODE_MIN_BITS),
        ("encode", generated::ENCODE_MIN_BITS),
    ] {
        println!("{name}_count={}", u64::from(1.0f32.to_bits() - start) + 1);
    }
}
fn oracle(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let mut out = File::create_new(output)?;
    writeln!(out, "kind,degree,layout,records,domain_records,rescues")?;
    let mut counts = [[[0u64; 3]; 26]; 2];
    for (line_number, line) in BufReader::new(File::open(input)?).lines().enumerate() {
        let line = line?;
        if line_number == 0 {
            if line != "kind,x_bits,p_bits,expected_bits,lo_f64_bits,hi_f64_bits,label" {
                return Err("bad oracle header".into());
            }
            continue;
        }
        let fields: Vec<_> = line.split(',').collect();
        if fields.len() != 7 {
            return Err("bad oracle row".into());
        }
        let k = fields[0].parse::<usize>()?;
        if k > 1 {
            return Err("bad oracle kind".into());
        }
        let xb = u32::from_str_radix(fields[1], 16)?;
        let pb = u32::from_str_radix(fields[2], 16)?;
        let expected = u32::from_str_radix(fields[3], 16)?;
        let lo = f64::from_bits(u64::from_str_radix(fields[4], 16)?);
        let hi = f64::from_bits(u64::from_str_radix(fields[5], 16)?);
        let expected_p = if k == 0 {
            generated::DECODE_P_BITS
        } else {
            generated::ENCODE_P_BITS
        };
        if pb != expected_p {
            return Err("oracle exponent mismatch".into());
        }
        let x = f32::from_bits(xb);
        let mut failure = None;
        let mut visit = |n: usize, l: usize, value: kernel::Evaluation| {
            let row = &mut counts[k][(n - 8) * 2 + l];
            row[0] += 1;
            let inside =
                xb >= (if k == 0 {
                    generated::DECODE_MIN_BITS
                } else {
                    generated::ENCODE_MIN_BITS
                }) && xb <= 1.0f32.to_bits();
            if inside {
                row[1] += 1;
                if value.rescued {
                    row[2] += 1;
                }
                if value.lower > lo
                    || value.upper < hi
                    || !value.lower.is_finite()
                    || !value.upper.is_finite()
                {
                    failure = Some(format!(
                        "ENCLOSURE FAILURE K={k} N={n} L={l} x={xb:08x} candidate=[{:e},{:e}] MPFR=[{:e},{:e}]",
                        value.lower, value.upper, lo, hi
                    ));
                }
            }
            if value.value.to_bits() != expected
                && !(value.value.is_nan() && f32::from_bits(expected).is_nan())
            {
                failure = Some(format!(
                    "BIT FAILURE K={k} N={n} L={l} x={xb:08x} got={:08x} expected={expected:08x}",
                    value.value.to_bits()
                ));
            }
        };
        if k == 0 {
            kernel::each::<0>(x, &mut visit);
        } else {
            kernel::each::<1>(x, &mut visit);
        }
        if let Some(failure) = failure {
            return Err(failure.into());
        }
    }
    for (k, rows) in counts.iter().enumerate() {
        for (i, row) in rows.iter().enumerate() {
            if row[0] == 0 {
                return Err("empty oracle cell".into());
            }
            writeln!(
                out,
                "{k},{},{},{},{},{}",
                i / 2 + 8,
                i % 2,
                row[0],
                row[1],
                row[2]
            )?;
        }
    }
    println!("MPFR all52 combinations:bit equality AND interval containment passed");
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("facts") => facts(),
        Some("oracle") if args.len() == 4 => oracle(Path::new(&args[2]), Path::new(&args[3]))?,
        #[cfg(target_arch = "x86_64")]
        Some("bench") if args.len() == 3 => timing::run(Path::new(&args[2]))?,
        _ => return Err("usage:facts | oracle INPUT.csv OUTPUT.csv | bench OUTPUT.json".into()),
    }
    Ok(())
}

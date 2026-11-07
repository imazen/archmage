use crate::{
    kernel::{self, Evaluation},
    vector,
};
use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader, Write},
    path::Path,
};
#[derive(Clone, Copy)]
struct Row {
    x: f32,
    expected: u32,
    lower: f64,
    upper: f64,
}

pub(crate) fn run(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    use archmage::SimdToken;
    archmage::X64V3Token::summon().ok_or("vector oracle requires V3")?;
    let mut rows = [vec![], vec![]];
    for (i, line) in BufReader::new(File::open(input)?).lines().enumerate() {
        let line = line?;
        if i == 0 {
            if line != "kind,x_bits,p_bits,expected_bits,lo_f64_bits,hi_f64_bits,label" {
                return Err("oracle header".into());
            }
            continue;
        }
        let f: Vec<_> = line.split(',').collect();
        if f.len() != 7 {
            return Err("oracle fields".into());
        }
        let k: usize = f[0].parse()?;
        if k > 1 {
            return Err("oracle kind".into());
        }
        let p = u32::from_str_radix(f[2], 16)?;
        let expected_p = if k == 0 {
            kernel::exponent::<0>()
        } else {
            kernel::exponent::<1>()
        };
        if p != expected_p.to_bits() {
            return Err("oracle p".into());
        }
        rows[k].push(Row {
            x: f32::from_bits(u32::from_str_radix(f[1], 16)?),
            expected: u32::from_str_radix(f[3], 16)?,
            lower: f64::from_bits(u64::from_str_radix(f[4], 16)?),
            upper: f64::from_bits(u64::from_str_radix(f[5], 16)?),
        });
    }
    let mut out = File::create_new(output)?;
    writeln!(
        out,
        "kind,family,degree,layout,records,domain_records,rescues"
    )?;
    fn kind<const K: usize>(rows: &mut [Row], out: &mut File) -> Result<(), Box<dyn Error>> {
        if rows.is_empty() {
            return Err("empty vector oracle".into());
        }
        let mut state = 0x57454354u32;
        for i in (1..rows.len()).rev() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            rows.swap(i, state as usize % (i + 1));
        }
        fn cell<const K: usize, const N: usize, const PIECE: bool>(
            rows: &[Row],
            out: &mut File,
        ) -> Result<(), Box<dyn Error>> {
            let mut count = 0;
            let mut domain = 0;
            let mut rescues = 0;
            let (start, end) = kernel::domain::<K>();
            for chunk in rows.chunks(4) {
                let input = std::array::from_fn(|i| chunk[i.min(chunk.len() - 1)].x);
                let got = vector::evaluate::<K, N, 1, PIECE>(&input);
                for (
                    &r,
                    Evaluation {
                        value,
                        rescued,
                        lower,
                        upper,
                    },
                ) in chunk.iter().zip(got)
                {
                    let bits = r.x.to_bits();
                    count += 1;
                    if bits >= start && bits <= end {
                        domain += 1;
                        rescues += usize::from(rescued);
                        if !lower.is_finite()
                            || !upper.is_finite()
                            || lower > r.lower
                            || upper < r.upper
                        {
                            return Err(format!(
                                "VECTOR ENCLOSURE FAILURE K={K} N={N} PIECE={PIECE} x={bits:08x}"
                            )
                            .into());
                        }
                    }
                    if value.to_bits() != r.expected
                        && !(value.is_nan() && f32::from_bits(r.expected).is_nan())
                    {
                        return Err(format!(
                            "VECTOR BIT FAILURE K={K} N={N} PIECE={PIECE} x={bits:08x}"
                        )
                        .into());
                    }
                }
            }
            // Both buffer widths and a non-multiple tail are checked on the same independent rows.
            let input: Vec<_> = rows.iter().map(|r| r.x).collect();
            let mut output = vec![0.0; input.len()];
            fn stream<const K: usize, const N: usize, const PIECE: bool, const W: usize>(
                input: &[f32],
                output: &mut [f32],
                rows: &[Row],
            ) {
                vector::stream::<K, N, 1, PIECE, W>(input, output);
                for (y, r) in output.iter().zip(rows) {
                    assert!(
                        y.to_bits() == r.expected
                            || y.is_nan() && f32::from_bits(r.expected).is_nan()
                    );
                }
                vector::stream::<K, N, 1, PIECE, W>(
                    &input[..input.len() - 1],
                    &mut output[..input.len() - 1],
                );
                for (y, r) in output[..input.len() - 1].iter().zip(rows) {
                    assert!(
                        y.to_bits() == r.expected
                            || y.is_nan() && f32::from_bits(r.expected).is_nan()
                    );
                }
            }
            stream::<K, N, PIECE, 4>(&input, &mut output, rows);
            stream::<K, N, PIECE, 8>(&input, &mut output, rows);
            writeln!(
                out,
                "{K},{},{N},1,{count},{domain},{rescues}",
                if PIECE { "piece" } else { "global" }
            )?;
            Ok(())
        }
        if K == 0 {
            cell::<K, 13, false>(rows, out)?;
        } else {
            cell::<K, 15, false>(rows, out)?;
        }
        cell::<K, 4, true>(rows, out)?;
        cell::<K, 5, true>(rows, out)?;
        Ok(())
    }
    kind::<0>(&mut rows[0], &mut out)?;
    kind::<1>(&mut rows[1], &mut out)?;
    println!(
        "VECTOR MPFR containment+bits: six cells, mixed lanes; array4/8 stream bits and tails PASS"
    );
    Ok(())
}

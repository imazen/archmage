//! Experimental AVX2 four-f64-lane polynomial; scalar rounding certificates per lane.
use crate::{
    generated as g,
    kernel::{self, Evaluation},
    piecewise_generated as pg,
};
use archmage::intrinsics::x86_64::*;
use archmage::{SimdToken, X64V3Token};

#[cold]
#[inline(never)]
fn rescue<const K: usize>(x: f32) -> f32 {
    zenmath::scalar::reference::powf(x, kernel::exponent::<K>())
}

// Expansion in the enclosing proof context prevents an array-return helper call.
macro_rules! block_body {
    ($kind:ident,$degree:ident,$layout:ident,$piece:ident,$input:expr) => {{
        let input = $input;
        const {
            assert!(
                $kind < 2
                    && $layout < 2
                    && if $piece {
                        $degree >= 3 && $degree <= 5
                    } else {
                        $degree >= 8 && $degree <= 20
                    }
            );
        }
        let (minimum, maximum) = kernel::domain::<$kind>();
        let mut centers = [0usize; 4];
        let mut valid = [false; 4];
        let mut v = [0.0; 4];
        let mut factors = [0.0; 4];
        let mut deltas = [0.0; 4];
        for lane in 0..4 {
            let bits = input[lane].to_bits();
            valid[lane] = bits >= minimum && bits <= maximum;
            let safe = if valid[lane] { bits } else { 1.0f32.to_bits() };
            let e = (safe >> 23) as usize - 118;
            let j = ((safe >> 19) & 15) as usize;
            centers[lane] = j;
            let m = f64::from(f32::from_bits((safe & 0x007f_ffff) | 0x3f80_0000));
            v[lane] = m - if $piece {
                (33 + 2 * j) as f64 * (1.0 / 32.0)
            } else {
                1.5
            };
            factors[lane] = if $piece {
                pg::FACTOR[$kind][e][j]
            } else {
                g::FACTOR[$kind][e]
            };
            deltas[lane] = if $piece {
                pg::ERROR[$kind][$layout][$degree.saturating_sub(3)][e][j]
            } else {
                g::ERROR[$kind][$layout][$degree.saturating_sub(8)][e]
            };
        }
        let v = _mm256_loadu_pd(&v);
        let mut coeff = [_mm256_setzero_pd(); 21];
        for (k, item) in coeff.iter_mut().enumerate().take($degree + 1) {
            *item = if $piece {
                _mm256_loadu_pd(&[
                    pg::COEFF[$kind][centers[0]][k],
                    pg::COEFF[$kind][centers[1]][k],
                    pg::COEFF[$kind][centers[2]][k],
                    pg::COEFF[$kind][centers[3]][k],
                ])
            } else {
                _mm256_set1_pd(g::COEFF[$kind][k])
            };
        }
        let q = if $layout == 0 {
            let mut q = coeff[$degree];
            let mut k = $degree;
            while k > 0 {
                k -= 1;
                q = _mm256_add_pd(_mm256_mul_pd(q, v), coeff[k]);
            }
            q
        } else {
            let mut count = $degree + 1;
            let mut power = v;
            while count > 1 {
                let mut i = 0;
                while i < count / 2 {
                    coeff[i] = _mm256_add_pd(coeff[i * 2], _mm256_mul_pd(coeff[i * 2 + 1], power));
                    i += 1;
                }
                if count % 2 == 1 {
                    coeff[i] = coeff[count - 1];
                }
                count = count.div_ceil(2);
                if count > 1 {
                    power = _mm256_mul_pd(power, power);
                }
            }
            coeff[0]
        };
        let y = _mm256_mul_pd(q, _mm256_loadu_pd(&factors));
        let delta = _mm256_loadu_pd(&deltas);
        let mut lower = [0.0; 4];
        let mut upper = [0.0; 4];
        _mm256_storeu_pd(&mut lower, _mm256_sub_pd(y, delta));
        _mm256_storeu_pd(&mut upper, _mm256_add_pd(y, delta));
        std::array::from_fn(|i| {
            if !valid[i] {
                return Evaluation {
                    value: rescue::<$kind>(input[i]),
                    rescued: true,
                    lower: f64::NAN,
                    upper: f64::NAN,
                };
            }
            let lower = f64::from_bits(lower[i].to_bits() - 1);
            let upper = f64::from_bits(upper[i].to_bits() + 1);
            let lo = lower as f32;
            let hi = upper as f32;
            if lo.to_bits() == hi.to_bits() {
                Evaluation {
                    value: lo,
                    rescued: false,
                    lower,
                    upper,
                }
            } else {
                Evaluation {
                    value: rescue::<$kind>(input[i]),
                    rescued: true,
                    lower,
                    upper,
                }
            }
        })
    }};
}

#[archmage::rite(v3, import_intrinsics)]
fn block<const K: usize, const N: usize, const L: usize, const PIECE: bool>(
    input: &[f32; 4],
) -> [Evaluation; 4] {
    block_body!(K, N, L, PIECE, input)
}

#[archmage::arcane]
fn entry<const K: usize, const N: usize, const L: usize, const PIECE: bool>(
    token: X64V3Token,
    input: &[f32; 4],
) -> [Evaluation; 4] {
    let _ = token;
    block::<K, N, L, PIECE>(input)
}

pub(crate) fn evaluate<const K: usize, const N: usize, const L: usize, const PIECE: bool>(
    input: &[f32; 4],
) -> [Evaluation; 4] {
    entry::<K, N, L, PIECE>(X64V3Token::summon().expect("V3 required"), input)
}

#[archmage::arcane]
fn stream_context<
    const K: usize,
    const N: usize,
    const L: usize,
    const PIECE: bool,
    const WIDTH: usize,
>(
    token: X64V3Token,
    input: &[f32],
    output: &mut [f32],
) {
    const {
        assert!(WIDTH == 4 || WIDTH == 8);
    }
    let _ = token;
    assert_eq!(input.len(), output.len());
    let complete = input.len() / WIDTH * WIDTH;
    for (xs, ys) in input[..complete]
        .as_chunks::<WIDTH>()
        .0
        .iter()
        .zip(output[..complete].as_chunks_mut::<WIDTH>().0.iter_mut())
    {
        for (x, y) in xs
            .as_chunks::<4>()
            .0
            .iter()
            .zip(ys.as_chunks_mut::<4>().0.iter_mut())
        {
            let result: [Evaluation; 4] = block_body!(K, N, L, PIECE, x);
            for i in 0..4 {
                y[i] = result[i].value;
            }
        }
    }
    for (y, &x) in output[complete..].iter_mut().zip(&input[complete..]) {
        *y = zenmath::scalar::reference::powf(x, kernel::exponent::<K>());
    }
}

pub(crate) fn stream<
    const K: usize,
    const N: usize,
    const L: usize,
    const PIECE: bool,
    const WIDTH: usize,
>(
    input: &[f32],
    output: &mut [f32],
) {
    stream_context::<K, N, L, PIECE, WIDTH>(
        X64V3Token::summon().expect("V3 required"),
        input,
        output,
    )
}

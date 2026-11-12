//! Shared preflight outside timing; avoids repeating verifier syntax per cell.
use crate::workload::{CheckScalar, check_mm, check_rc};
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify<F: CheckScalar, const N: usize, const W: usize>(
    flags: u32,
    legacy: bool,
    recip: bool,
    op: bool,
    dependent: bool,
    data: &([F; N], [F; N]),
    stream: impl Fn(&[F; N], &[F; N], &mut [F; N]),
    step: impl Fn(&[F; W], &[F; W], &mut [F; W]),
    chain: impl Fn(F) -> [F; W],
) -> f64 {
    let check = |a: F, b: F, y: F| {
        if recip {
            check_rc(flags, legacy, op, a.narrow(), y.narrow())
        } else {
            if op {
                check_mm::<true>(flags, legacy, a.widen(), b.widen(), y.widen());
            } else {
                check_mm::<false>(flags, legacy, a.widen(), b.widen(), y.widen());
            }
            0.0
        }
    };
    let mut out = [F::from_f32(0.0); N];
    stream(&data.0, &data.1, &mut out);
    let mut error = 0.0_f64;
    for (i, y) in out.into_iter().enumerate() {
        error = error.max(check(data.0[i], data.1[i], y));
    }
    if dependent {
        let mut a = [F::from_f32(0.75); W];
        let b = [F::from_f32(2.0); W];
        let mut out = [F::from_f32(0.0); W];
        for _ in 0..64 {
            step(&a, &b, &mut out);
            for i in 0..W {
                error = error.max(check(a[i], b[i], out[i]));
                a[i] = out[i] + F::from_f32(if recip { 0.5 } else { 0.03125 });
            }
        }
        assert_eq!(chain(F::from_f32(0.75)).map(F::bits), a.map(F::bits));
    }
    error
}

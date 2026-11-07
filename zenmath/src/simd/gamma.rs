//! Intrinsic-only packed global Estrin-13/15 candidates, with per-lane rescue.
use super::packed::{Backend, V};
use crate::{gamma, gamma_tables as g};

#[inline(always)]
pub(super) fn candidate<T: Backend<N>, const N: usize>(
    t: T,
    x: [f32; N],
    y: [f32; N],
) -> Option<[f32; N]> {
    // Uniform exponent guard is also required for dynamic-y callers.
    let bits = y[0].to_bits();
    let kind = if bits == gamma::EXPONENT_BITS[0] {
        0
    } else if bits == gamma::EXPONENT_BITS[1] {
        1
    } else {
        return None;
    };
    if y.iter().any(|v| v.to_bits() != bits)
        || !x
            .iter()
            .any(|v| (gamma::MINIMUM[kind]..=gamma::MAXIMUM).contains(&v.to_bits()))
    {
        return None;
    }
    Some(if kind == 0 {
        global::<_, N, 0, 14>(t, x, g::GLOBAL_DECODE_COEFF)
    } else {
        global::<_, N, 1, 16>(t, x, g::GLOBAL_ENCODE_COEFF)
    })
}

#[inline(always)]
fn global<T: Backend<N>, const N: usize, const K: usize, const C: usize>(
    t: T,
    x: [f32; N],
    coefficients: [f64; C],
) -> [f32; N] {
    let valid = x.map(|v| (gamma::MINIMUM[K]..=gamma::MAXIMUM).contains(&v.to_bits()));
    let safe: [u32; N] = core::array::from_fn(|i| {
        if valid[i] {
            x[i].to_bits()
        } else {
            gamma::MAXIMUM
        }
    });
    let e = safe.map(|b| (b >> 23) as usize - 118);
    let v = V::from_array_t(
        t,
        safe.map(|b| f64::from(f32::from_bits((b & 0x007f_ffff) | 0x3f80_0000)) - 1.5),
    );
    let mut coefficients = coefficients.map(|b| V::splat_t(t, b));
    let mut count = C;
    let mut power = v;
    // Same pair/carry/square graph as the certified experimental vector body.
    while count > 1 {
        for i in 0..count / 2 {
            coefficients[i] = coefficients[i * 2] + coefficients[i * 2 + 1] * power;
        }
        if count % 2 == 1 {
            coefficients[count / 2] = coefficients[count - 1];
        }
        count = count.div_ceil(2);
        if count > 1 {
            power = power * power;
        }
    }
    let y = (coefficients[0] * V::from_array_t(t, e.map(|i| g::GLOBAL_FACTOR[K][i]))).to_array();
    core::array::from_fn(|i| {
        if valid[i]
            && let Some(value) = gamma::certificate(y[i], g::GLOBAL_ERROR[K][e[i]])
        {
            return value;
        }
        crate::reference::powf(x[i], f32::from_bits(gamma::EXPONENT_BITS[K]))
    })
}

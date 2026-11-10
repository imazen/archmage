use archmage::prelude::*;
use zenmath::{
    scalar::reference as r,
    simd::{array as a, scalar as s},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Exp,
    Ln,
    Log2,
    Pow(u32),
    Sqrt,
    Floor,
    Ceil,
    Trunc,
    Round,
    Roundeven,
}
impl Op {
    pub fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "exp" => Self::Exp,
            "ln" => Self::Ln,
            "log2" => Self::Log2,
            "sqrt" => Self::Sqrt,
            "floor" => Self::Floor,
            "ceil" => Self::Ceil,
            "trunc" => Self::Trunc,
            "round" => Self::Round,
            "roundeven" => Self::Roundeven,
            _ if s.starts_with("pow:") => {
                Self::Pow(u32::from_str_radix(&s[4..], 16).map_err(|e| e.to_string())?)
            }
            _ => {
                return Err(format!(
                    "unknown operation {s}; pow uses exact exponent hex bits"
                ));
            }
        })
    }
    pub fn reference(self, x: f32) -> f32 {
        match self {
            Self::Exp => r::expf(x),
            Self::Ln => r::logf(x),
            Self::Log2 => r::log2f(x),
            Self::Pow(y) => r::powf(x, f32::from_bits(y)),
            Self::Sqrt => r::sqrtf(x),
            Self::Floor => r::floorf(x),
            Self::Ceil => r::ceilf(x),
            Self::Trunc => r::floorf(x.abs()).copysign(x),
            Self::Round => r::roundf(x),
            Self::Roundeven => r::roundevenf(x),
        }
    }
    pub fn packed(self) -> bool {
        matches!(self, Self::Exp | Self::Ln | Self::Log2 | Self::Pow(_))
    }
}

pub type Kernel = Box<dyn Fn(&mut [f32]) + Send + Sync>;
pub fn select(backend: &str, op: Op) -> Result<Kernel, String> {
    match backend {
        "reference" => Ok(Box::new(move |xs| {
            xs.iter_mut().for_each(|x| *x = op.reference(*x))
        })),
        "packed-scalar" if op.packed() => {
            Ok(Box::new(move |xs| packed_scalar(ScalarToken, op, xs)))
        }
        "scalar-context" if op.packed() => {
            Ok(Box::new(move |xs| scalar_context(ScalarToken, op, xs)))
        }
        #[cfg(target_arch = "x86_64")]
        "v3" if op.packed() => {
            let t = X64V3Token::summon().ok_or("unavailable v3")?;
            Ok(Box::new(move |xs| packed_v3(t, op, xs)))
        }
        #[cfg(target_arch = "x86_64")]
        "scalar-v3" if op.packed() => {
            let t = X64V3Token::summon().ok_or("unavailable v3")?;
            Ok(Box::new(move |xs| scalar_v3(t, op, xs)))
        }
        #[cfg(target_arch = "aarch64")]
        "neon" if op.packed() => {
            let t = NeonToken::summon().ok_or("unavailable neon")?;
            Ok(Box::new(move |xs| packed_neon(t, op, xs)))
        }
        #[cfg(target_arch = "aarch64")]
        "scalar-neon" if op.packed() => {
            let t = NeonToken::summon().ok_or("unavailable neon")?;
            Ok(Box::new(move |xs| scalar_neon(t, op, xs)))
        }
        #[cfg(target_arch = "wasm32")]
        "wasm128" if op.packed() => {
            let t = Wasm128Token::summon().ok_or("unavailable wasm128")?;
            Ok(Box::new(move |xs| packed_wasm128(t, op, xs)))
        }
        #[cfg(target_arch = "wasm32")]
        "scalar-wasm128" if op.packed() => {
            let t = Wasm128Token::summon().ok_or("unavailable wasm128")?;
            Ok(Box::new(move |xs| scalar_wasm128(t, op, xs)))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4x" if op.packed() => {
            let t = X64V4xToken::summon().ok_or("unavailable v4x")?;
            Ok(Box::new(move |xs| packed_v4x(t, op, xs)))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "scalar-v4x" if op.packed() => {
            let t = X64V4xToken::summon().ok_or("unavailable v4x")?;
            Ok(Box::new(move |xs| scalar_v4x(t, op, xs)))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4x8" if op.packed() => {
            let t = X64V4xToken::summon().ok_or("unavailable v4x8")?;
            Ok(Box::new(move |xs| wide(t, op, xs)))
        }
        #[cfg(target_arch = "x86_64")]
        "v1" if op == Op::Sqrt => {
            let t = X64V1Token::summon().ok_or("unavailable v1")?;
            Ok(Box::new(move |xs| primitive_v1(t, xs)))
        }
        #[cfg(target_arch = "x86_64")]
        "v2" if !op.packed() && op != Op::Sqrt => {
            let t = X64V2Token::summon().ok_or("unavailable v2")?;
            Ok(Box::new(move |xs| primitive_v2(t, op, xs)))
        }
        #[cfg(target_arch = "aarch64")]
        "neon" if !op.packed() => {
            let t = NeonToken::summon().ok_or("unavailable neon")?;
            Ok(Box::new(move |xs| primitive_neon(t, op, xs)))
        }
        #[cfg(target_arch = "wasm32")]
        "wasm128" if !op.packed() => {
            let t = Wasm128Token::summon().ok_or("unavailable wasm128")?;
            Ok(Box::new(move |xs| primitive_wasm128(t, op, xs)))
        }
        _ => Err(format!(
            "unsupported operation/backend combination {op:?}/{backend} on {}",
            std::env::consts::ARCH
        )),
    }
}
#[archmage::arcane]
fn packed_scalar(token: ScalarToken, op: Op, xs: &mut [f32]) {
    for chunk in xs.chunks_mut(4) {
        let mut input = [1.0; 4];
        input[..chunk.len()].copy_from_slice(chunk);
        let out = match op {
            Op::Exp => a::exp_f32x4_scalar(input),
            Op::Ln => a::ln_f32x4_scalar(input),
            Op::Log2 => a::log2_f32x4_scalar(input),
            Op::Pow(y) => a::pow_f32x4_scalar(input, [f32::from_bits(y); 4]),
            _ => unreachable!(),
        };
        chunk.copy_from_slice(&out[..chunk.len()]);
    }
    let _ = token;
}
#[archmage::arcane]
fn scalar_context(token: ScalarToken, op: Op, xs: &mut [f32]) {
    for x in xs {
        *x = match op {
            Op::Exp => s::expf_scalar(*x),
            Op::Ln => s::logf_scalar(*x),
            Op::Log2 => s::log2f_scalar(*x),
            Op::Pow(y) => s::powf_scalar(*x, f32::from_bits(y)),
            _ => unreachable!(),
        };
    }
    let _ = token;
}
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn packed_v3(token: X64V3Token, op: Op, xs: &mut [f32]) {
    for chunk in xs.chunks_mut(4) {
        let mut input = [1.0; 4];
        input[..chunk.len()].copy_from_slice(chunk);
        let out = match op {
            Op::Exp => a::exp_f32x4_v3(input),
            Op::Ln => a::ln_f32x4_v3(input),
            Op::Log2 => a::log2_f32x4_v3(input),
            Op::Pow(y) => a::pow_f32x4_v3(input, [f32::from_bits(y); 4]),
            _ => unreachable!(),
        };
        chunk.copy_from_slice(&out[..chunk.len()]);
    }
    let _ = token;
}
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn scalar_v3(token: X64V3Token, op: Op, xs: &mut [f32]) {
    for x in xs {
        *x = match op {
            Op::Exp => s::expf_v3(*x),
            Op::Ln => s::logf_v3(*x),
            Op::Log2 => s::log2f_v3(*x),
            Op::Pow(y) => s::powf_v3(*x, f32::from_bits(y)),
            _ => unreachable!(),
        };
    }
    let _ = token;
}
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn packed_neon(token: NeonToken, op: Op, xs: &mut [f32]) {
    for chunk in xs.chunks_mut(4) {
        let mut input = [1.0; 4];
        input[..chunk.len()].copy_from_slice(chunk);
        let out = match op {
            Op::Exp => a::exp_f32x4_neon(input),
            Op::Ln => a::ln_f32x4_neon(input),
            Op::Log2 => a::log2_f32x4_neon(input),
            Op::Pow(y) => a::pow_f32x4_neon(input, [f32::from_bits(y); 4]),
            _ => unreachable!(),
        };
        chunk.copy_from_slice(&out[..chunk.len()]);
    }
    let _ = token;
}
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn scalar_neon(token: NeonToken, op: Op, xs: &mut [f32]) {
    for x in xs {
        *x = match op {
            Op::Exp => s::expf_neon(*x),
            Op::Ln => s::logf_neon(*x),
            Op::Log2 => s::log2f_neon(*x),
            Op::Pow(y) => s::powf_neon(*x, f32::from_bits(y)),
            _ => unreachable!(),
        };
    }
    let _ = token;
}
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn packed_wasm128(token: Wasm128Token, op: Op, xs: &mut [f32]) {
    for chunk in xs.chunks_mut(4) {
        let mut input = [1.0; 4];
        input[..chunk.len()].copy_from_slice(chunk);
        let out = match op {
            Op::Exp => a::exp_f32x4_wasm128(input),
            Op::Ln => a::ln_f32x4_wasm128(input),
            Op::Log2 => a::log2_f32x4_wasm128(input),
            Op::Pow(y) => a::pow_f32x4_wasm128(input, [f32::from_bits(y); 4]),
            _ => unreachable!(),
        };
        chunk.copy_from_slice(&out[..chunk.len()]);
    }
    let _ = token;
}
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn scalar_wasm128(token: Wasm128Token, op: Op, xs: &mut [f32]) {
    for x in xs {
        *x = match op {
            Op::Exp => s::expf_wasm128(*x),
            Op::Ln => s::logf_wasm128(*x),
            Op::Log2 => s::log2f_wasm128(*x),
            Op::Pow(y) => s::powf_wasm128(*x, f32::from_bits(y)),
            _ => unreachable!(),
        };
    }
    let _ = token;
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn packed_v4x(token: X64V4xToken, op: Op, xs: &mut [f32]) {
    for chunk in xs.chunks_mut(4) {
        let mut input = [1.0; 4];
        input[..chunk.len()].copy_from_slice(chunk);
        let out = match op {
            Op::Exp => a::exp_f32x4_v4x(input),
            Op::Ln => a::ln_f32x4_v4x(input),
            Op::Log2 => a::log2_f32x4_v4x(input),
            Op::Pow(y) => a::pow_f32x4_v4x(input, [f32::from_bits(y); 4]),
            _ => unreachable!(),
        };
        chunk.copy_from_slice(&out[..chunk.len()]);
    }
    let _ = token;
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn scalar_v4x(token: X64V4xToken, op: Op, xs: &mut [f32]) {
    for x in xs {
        *x = match op {
            Op::Exp => s::expf_v4x(*x),
            Op::Ln => s::logf_v4x(*x),
            Op::Log2 => s::log2f_v4x(*x),
            Op::Pow(y) => s::powf_v4x(*x, f32::from_bits(y)),
            _ => unreachable!(),
        };
    }
    let _ = token;
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn wide(token: X64V4xToken, op: Op, xs: &mut [f32]) {
    for chunk in xs.chunks_mut(8) {
        let mut input = [1.0; 8];
        input[..chunk.len()].copy_from_slice(chunk);
        let out = match op {
            Op::Exp => a::exp_f32x8_v4x(input),
            Op::Ln => a::ln_f32x8_v4x(input),
            Op::Log2 => a::log2_f32x8_v4x(input),
            Op::Pow(y) => a::pow_f32x8_v4x(input, [f32::from_bits(y); 8]),
            _ => unreachable!(),
        };
        chunk.copy_from_slice(&out[..chunk.len()]);
    }
    let _ = token;
}
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn primitive_v1(token: X64V1Token, xs: &mut [f32]) {
    for x in xs {
        *x = s::sqrtf_v1(*x);
    }
    let _ = token;
}
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn primitive_v2(token: X64V2Token, op: Op, xs: &mut [f32]) {
    for x in xs {
        *x = match op {
            Op::Floor => s::floorf_v2(*x),
            Op::Ceil => s::ceilf_v2(*x),
            Op::Trunc => s::truncf_v2(*x),
            Op::Round => s::roundf_v2(*x),
            Op::Roundeven => s::roundevenf_v2(*x),
            _ => unreachable!(),
        };
    }
    let _ = token;
}
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn primitive_neon(token: NeonToken, op: Op, xs: &mut [f32]) {
    for x in xs {
        *x = match op {
            Op::Sqrt => s::sqrtf_neon(*x),
            Op::Floor => s::floorf_neon(*x),
            Op::Ceil => s::ceilf_neon(*x),
            Op::Trunc => s::truncf_neon(*x),
            Op::Round => s::roundf_neon(*x),
            Op::Roundeven => s::roundevenf_neon(*x),
            _ => unreachable!(),
        };
    }
    let _ = token;
}
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn primitive_wasm128(token: Wasm128Token, op: Op, xs: &mut [f32]) {
    for x in xs {
        *x = match op {
            Op::Sqrt => s::sqrtf_wasm128(*x),
            Op::Floor => s::floorf_wasm128(*x),
            Op::Ceil => s::ceilf_wasm128(*x),
            Op::Trunc => s::truncf_wasm128(*x),
            Op::Round => s::roundf_wasm128(*x),
            Op::Roundeven => s::roundevenf_wasm128(*x),
            _ => unreachable!(),
        };
    }
    let _ = token;
}

//! `#[arcane]`, `#[rite]`, `#[autoversion]`, `#[magetypes]` and `incant!`.

use archmage::prelude::*;

/// Sums eight lanes with AVX2.
#[arcane(import_intrinsics)]
#[must_use]
pub fn sum8(_token: X64V3Token, data: &[f32; 8]) -> f32 {
    let doubled = double(_mm256_loadu_ps(data));
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, doubled);
    out.iter().sum::<f32>() * 0.5
}

#[rite(v3, import_intrinsics)]
fn double(v: __m256) -> __m256 {
    _mm256_add_ps(v, v)
}

#[rite(v3, neon, wasm128)]
fn halve(x: f32) -> f32 {
    x * 0.5
}

#[rite(v3, neon, wasm128)]
fn quarter(x: f32) -> f32 {
    incant!(halve(x) without token) * 0.5
}

/// Quarters a value through tokenless multi-tier helpers.
#[must_use]
pub fn quarter_dispatch(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = X64V3Token::summon() {
        return quarter_entry(token, x);
    }
    x * 0.25
}

#[arcane]
fn quarter_entry(_token: X64V3Token, x: f32) -> f32 {
    quarter_v3(x)
}

/// Sums squares, auto-vectorized per tier.
#[autoversion]
#[must_use]
pub fn sum_squares(data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum()
}

/// Sums, with an explicit tier list.
#[autoversion(v3, neon)]
#[must_use]
pub fn sum(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn gain_impl(token: Token, plane: &mut [f32], gain: f32) {
    let factor = f32x8::splat_t(token, gain);
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, plane);
    for chunk in chunks {
        (f32x8::load_t(token, chunk) * factor).store(chunk);
    }
    for value in tail {
        *value *= gain;
    }
}

/// Applies a gain through `incant!` with an explicit tier list.
pub fn apply_gain(plane: &mut [f32], gain: f32) {
    incant!(gain_impl(plane, gain), [v3, neon, wasm128, scalar]);
}

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn total_impl(token: Token, plane: &[f32]) -> f32 {
    let (chunks, tail) = f32x8::partition_slice_t(token, plane);
    let mut acc = f32x8::zero_t(token);
    for chunk in chunks {
        acc += f32x8::load_t(token, chunk);
    }
    acc.reduce_add() + tail.iter().sum::<f32>()
}

/// Totals a plane, in `let` position.
#[must_use]
pub fn total(plane: &[f32]) -> f32 {
    let sum = incant!(total_impl(plane), [v3, neon, wasm128, scalar]);
    sum * 1.0
}

/// A buffer with methods.
#[derive(Debug, Default)]
pub struct Buffer {
    data: [f32; 16],
}

impl Buffer {
    /// Totals the buffer.
    #[autoversion]
    #[must_use]
    pub fn total(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Doubles the buffer with AVX2.
    #[arcane(import_intrinsics)]
    pub fn double(&mut self, _token: X64V3Token) {
        for chunk in self.data.as_chunks_mut::<8>().0 {
            let v = _mm256_loadu_ps(chunk);
            _mm256_storeu_ps(chunk, _mm256_add_ps(v, v));
        }
    }
}

/// A trait implemented with a feature boundary.
pub trait Kernel {
    /// Runs the kernel.
    fn run(&self, token: X64V3Token, data: &[f32; 8]) -> f32;
}

/// A kernel with a bias.
#[derive(Debug)]
pub struct Summer {
    /// Added to the sum.
    pub bias: f32,
}

// `#[arcane]` omits wrong-architecture definitions, so the impl is gated too.
#[cfg(target_arch = "x86_64")]
impl Kernel for Summer {
    #[arcane(_self = Summer, import_intrinsics)]
    fn run(&self, _token: X64V3Token, data: &[f32; 8]) -> f32 {
        let v = _mm256_loadu_ps(data);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(&mut out, v);
        out.iter().sum::<f32>() + _self.bias
    }
}

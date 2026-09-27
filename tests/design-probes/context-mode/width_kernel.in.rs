//! Expansion-shape probe: the width selector is modeled by replacing @...@.
//! This is not a public `use(f32x)` implementation.
#![forbid(unsafe_code)]
use archmage::{arcane, incant, ScalarToken};

macro_rules! kernel {
    ($name:ident, $tier:ident, $token:ty, $vector:ident) => {
        #[archmage::rite($tier)]
        fn $name(values: &mut [f32]) {
            type V = magetypes::simd::generic::local::$vector<$token>;
            let (chunks, tail) = V::partition_slice_mut(values);
            for chunk in chunks {
                (V::load(chunk) + V::splat(2.0)).store(chunk);
            }
            for value in tail { *value += 2.0; }
        }
    };
}
kernel!(kernel_v3, v3, archmage::X64V3Token, @V3@);
kernel!(kernel_neon, neon, archmage::NeonToken, @NEON@);
kernel!(kernel_wasm128, wasm128, archmage::Wasm128Token, @WASM@);
kernel!(kernel_scalar, scalar, ScalarToken, @SCALAR@);
@V4_KERNEL@
#[arcane]
fn apply_v3(_: archmage::X64V3Token, values: &mut [f32]) { kernel_v3(values); }
#[arcane]
fn apply_neon(_: archmage::NeonToken, values: &mut [f32]) { kernel_neon(values); }
#[arcane]
fn apply_wasm128(_: archmage::Wasm128Token, values: &mut [f32]) { kernel_wasm128(values); }
fn apply_scalar(_: ScalarToken, values: &mut [f32]) { kernel_scalar(values); }
@V4_ENTRY@
pub fn run(values: &mut [f32]) { incant!(apply(values), [@TIERS@neon, wasm128, scalar]); }

#[cfg(test)]
mod tests {
    use super::*;
    use archmage::SimdToken;
    // Every remainder 0..15, short slices, offset starts, and padded rows.
    fn exercise(f: impl Fn(&mut [f32])) {
        for width in 0..=65 {
            for offset in 0..4 {
                let stride = width + 3;
                let mut data = vec![-999.0; offset + 3 * stride];
                for row in 0..3 {
                    let start = offset + row * stride;
                    for (i, x) in data[start..start+width].iter_mut().enumerate() {
                        *x = (i as f32) - 20.0;
                    }
                }
                let mut expected = data.clone();
                for row in 0..3 {
                    let start = offset + row * stride;
                    for x in &mut expected[start..start+width] { *x += 2.0; }
                    f(&mut data[start..start+width]);
                }
                assert_eq!(data, expected, "width={width} offset={offset}");
            }
        }
    }
    #[test] fn scalar_tails_and_rows() { exercise(|v| apply_scalar(ScalarToken, v)); }
    #[test] fn dispatch_tails_and_rows() { exercise(run); }
    #[cfg(target_arch="x86_64")]
    #[test] fn v3_tails_and_rows() {
        let token = archmage::X64V3Token::summon().expect("caller must provide a V3 CPU");
        exercise(|v| apply_v3(token, v));
    }
    #[cfg(target_arch="aarch64")]
    #[test] fn neon_tails_and_rows() {
        let token = archmage::NeonToken::summon().expect("caller must provide NEON");
        exercise(|v| apply_neon(token, v));
    }
    @V4_TEST@
}

//! Packed software FMA, entered after whole-operation selection.
use archmage::Wasm128Token;

#[archmage::rite]
pub(super) fn software_lanes<const N: usize>(
    token: Wasm128Token,
    a: [f32; N],
    b: [f32; N],
    c: [f32; N],
) -> [f32; N] {
    crate::wasm_fma::software_lanes(token, a, b, c)
}

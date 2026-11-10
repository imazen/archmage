//! Packed software FMA, entered after whole-operation selection.
use archmage::Wasm128Token;
use core::arch::wasm32 as w;

#[archmage::rite]
pub(super) fn software_lanes<const N: usize>(
    token: Wasm128Token,
    a: [f32; N],
    b: [f32; N],
    c: [f32; N],
) -> [f32; N] {
    assert_eq!(
        N % 4,
        0,
        "private packed software adapter requires whole vectors"
    );
    let mut output = [0.0; N];
    for (((a, b), c), out) in a
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_chunks::<4>().0)
        .zip(c.as_chunks::<4>().0)
        .zip(output.as_chunks_mut::<4>().0)
    {
        let a = w::f32x4(a[0], a[1], a[2], a[3]);
        let b = w::f32x4(b[0], b[1], b[2], b[3]);
        let c = w::f32x4(c[0], c[1], c[2], c[3]);
        let r = crate::wasm_fma::powf_soft_f32x4(token, a, b, c);
        *out = [
            w::f32x4_extract_lane::<0>(r),
            w::f32x4_extract_lane::<1>(r),
            w::f32x4_extract_lane::<2>(r),
            w::f32x4_extract_lane::<3>(r),
        ];
    }
    output
}

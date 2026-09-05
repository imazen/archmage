//! Checked copies and views for backend storage, inspired by fearless_simd's transmute module:
//! https://github.com/linebender/fearless_simd/blob/main/fearless_simd/src/transmute.rs
//!
//! Only raw scalar/vector storage implements Pod. Never implement it for tokens
//! or token-bearing SIMD wrappers: arbitrary bytes must not manufacture proofs.

use core::mem::{align_of, size_of};

/// Plain storage with no padding, pointers, or invalid bit patterns.
///
/// # Safety
/// Every byte must be initialized, every bit pattern valid, and the type must
/// contain no pointers or interior mutability. Copying its bits must be sound.
pub(crate) unsafe trait Pod: Copy {}

macro_rules! impl_pod {
    ($($ty:ty),+ $(,)?) => {$(
        // SAFETY: numeric scalars and stdarch vectors have no padding or pointers,
        // and accept every bit pattern. Vector layout matches the lane array,
        // with potentially stronger alignment; the helpers handle that difference.
        unsafe impl Pod for $ty {}
    )+};
}

impl_pod!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);
// SAFETY: arrays add no padding between elements and preserve their validity.
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

#[cfg(target_arch = "x86_64")]
const _: () = {
    use core::arch::x86_64::*;
    impl_pod!(__m128, __m128d, __m128i, __m256, __m256d, __m256i);
    #[cfg(feature = "avx512")]
    impl_pod!(__m512, __m512d, __m512i);
};

#[cfg(target_arch = "aarch64")]
const _: () = {
    use core::arch::aarch64::*;
    impl_pod!(
        float32x2_t,
        float32x4_t,
        float64x1_t,
        float64x2_t,
        int8x8_t,
        int8x16_t,
        uint8x8_t,
        uint8x16_t,
        int16x4_t,
        int16x8_t,
        uint16x4_t,
        uint16x8_t,
        int32x2_t,
        int32x4_t,
        uint32x2_t,
        uint32x4_t,
        int64x1_t,
        int64x2_t,
        uint64x1_t,
        uint64x2_t,
    );
};

#[cfg(target_arch = "wasm32")]
const _: () = {
    // v128 is exactly 16 initialized bytes, accepts every bit pattern, and
    // contains no pointers. Arrays of v128 have no inter-element padding.
    impl_pod!(core::arch::wasm32::v128);
};

/// Storage whose arbitrary bits are valid only in the presence of a token.
/// This is deliberately separate from Pod: wrappers must never manufacture tokens.
///
/// # Safety
/// Self must be a repr(C) pair of a Pod representation followed by Token, a
/// zero-sized, alignment-one token. Given a valid Token, every representation
/// bit pattern must be valid as Self. Self must have no additional invariants,
/// padding, pointers, interior mutability, or drop behavior. Mutable writes to
/// Self must preserve arbitrary-bit validity of its storage. Implement only for
/// the generated SIMD wrappers over sealed backend implementations.
pub(crate) unsafe trait TokenStorage: Copy {
    type Token: archmage::SimdToken;
}

#[inline(always)]
fn check_token_layout<Dst: TokenStorage>() {
    const {
        assert!(size_of::<Dst::Token>() == 0);
        assert!(align_of::<Dst::Token>() == 1);
    }
}

/// Borrow raw storage as a vector, carrying an existing feature proof.
#[inline(always)]
pub(crate) fn vector_view<Src: Pod, Dst: TokenStorage>(_: Dst::Token, src: &Src) -> &Dst {
    check_token_layout::<Dst>();
    const {
        assert!(size_of::<Src>() == size_of::<Dst>());
        assert!(align_of::<Src>() >= align_of::<Dst>());
    }
    // SAFETY: checked layout, initialized Pod bytes, and the supplied token
    // satisfy TokenStorage's validity contract. The borrow retains its lifetime.
    unsafe { &*core::ptr::from_ref(src).cast::<Dst>() }
}

/// Exclusively borrow raw storage as a vector with an existing feature proof.
#[inline(always)]
pub(crate) fn vector_view_mut<Src: Pod, Dst: TokenStorage>(
    _: Dst::Token,
    src: &mut Src,
) -> &mut Dst {
    check_token_layout::<Dst>();
    const {
        assert!(size_of::<Src>() == size_of::<Dst>());
        assert!(align_of::<Src>() >= align_of::<Dst>());
    }
    // SAFETY: as vector_view, with exclusive access. TokenStorage writes leave
    // initialized storage, and Pod accepts all resulting bits when reborrow ends.
    unsafe { &mut *core::ptr::from_mut(src).cast::<Dst>() }
}

/// Borrow whole vectors from scalar storage; retain the API's length/alignment checks.
#[inline(always)]
pub(crate) fn vector_slice<Src: Pod, Dst: TokenStorage, const N: usize>(
    _: Dst::Token,
    slice: &[Src],
) -> Option<&[Dst]> {
    check_token_layout::<Dst>();
    const {
        assert!(N > 0 && size_of::<Dst>() == size_of::<[Src; N]>());
    }
    if !slice.len().is_multiple_of(N) {
        return None;
    }
    let ptr = slice.as_ptr();
    if ptr.align_offset(align_of::<Dst>()) != 0 {
        return None;
    }
    // SAFETY: same byte extent, checked alignment, and TokenStorage validity
    // provided by initialized Pod elements and the supplied token. Same lifetime.
    Some(unsafe { core::slice::from_raw_parts(ptr.cast::<Dst>(), slice.len() / N) })
}

/// Mutable counterpart of vector_slice, preserving exclusive access.
#[inline(always)]
pub(crate) fn vector_slice_mut<Src: Pod, Dst: TokenStorage, const N: usize>(
    _: Dst::Token,
    slice: &mut [Src],
) -> Option<&mut [Dst]> {
    check_token_layout::<Dst>();
    const {
        assert!(N > 0 && size_of::<Dst>() == size_of::<[Src; N]>());
    }
    if !slice.len().is_multiple_of(N) {
        return None;
    }
    let ptr = slice.as_mut_ptr();
    if ptr.align_offset(align_of::<Dst>()) != 0 {
        return None;
    }
    // SAFETY: as vector_slice, with exclusive access. Every vector write leaves
    // initialized bytes valid as the original Pod elements.
    Some(unsafe { core::slice::from_raw_parts_mut(ptr.cast::<Dst>(), slice.len() / N) })
}

/// Copy same-sized storage without requiring the source's alignment to match Dst.
#[inline(always)]
pub(crate) fn copy<Src: Pod, Dst: Pod>(src: &Src) -> Dst {
    const { assert!(size_of::<Src>() == size_of::<Dst>()) };
    // SAFETY: Pod guarantees initialized bytes and validity for both types;
    // the const check guarantees equal sizes. transmute_copy handles alignment.
    unsafe { core::mem::transmute_copy(src) }
}

/// Borrow plain storage as an equally sized, no-more-aligned plain type.
#[inline(always)]
pub(crate) fn view<Src: Pod, Dst: Pod>(src: &Src) -> &Dst {
    const {
        assert!(size_of::<Src>() == size_of::<Dst>());
        assert!(align_of::<Src>() >= align_of::<Dst>());
    }
    // SAFETY: size and alignment are checked at compile time. Pod guarantees
    // initialized bytes and validity; the returned borrow retains src's lifetime.
    unsafe { &*core::ptr::from_ref(src).cast::<Dst>() }
}

/// Exclusively borrow plain storage, permitting every possible replacement bit pattern.
#[inline(always)]
pub(crate) fn view_mut<Src: Pod, Dst: Pod>(src: &mut Src) -> &mut Dst {
    const {
        assert!(size_of::<Src>() == size_of::<Dst>());
        assert!(align_of::<Src>() >= align_of::<Dst>());
    }
    // SAFETY: as in view, with exclusive access for the returned lifetime.
    // Both types are Pod, so writes through Dst leave a valid Src.
    unsafe { &mut *core::ptr::from_mut(src).cast::<Dst>() }
}

/// By-value convenience for existing backend bit reinterpretations.
#[inline(always)]
pub(crate) fn cast<Src: Pod, Dst: Pod>(src: Src) -> Dst {
    copy(&src)
}

/// Write exactly one destination, even if it is less aligned than Src.
#[inline(always)]
pub(crate) fn store<Src: Pod, Dst: Pod>(src: Src, dest: &mut Dst) {
    const { assert!(size_of::<Src>() == size_of::<Dst>()) };
    // SAFETY: dest is exclusively borrowed and valid for exactly this many
    // bytes. Pod permits every source bit pattern as Dst; unaligned write
    // imposes no extra alignment and constructs no misaligned reference.
    unsafe { core::ptr::write_unaligned((dest as *mut Dst).cast::<Src>(), src) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_all_bits_and_weak_alignment() {
        // The offset-one array forces a destination unaligned for u64.
        #[repr(align(8))]
        struct Bytes([u8; 17]);
        let mut bytes = Bytes([0xa5; 17]);
        let dest: &mut [u8; 8] = (&mut bytes.0[1..9]).try_into().unwrap();
        let bits = 0xfff8_1234_5678_9abc_u64;
        store(bits, dest);
        assert_eq!(copy::<_, u64>(dest), bits);
        assert_eq!(cast::<_, u64>(cast::<_, f64>(bits)), bits);
        assert_eq!(bytes.0[0], 0xa5);
        assert_eq!(&bytes.0[9..], &[0xa5; 8]);
    }

    #[test]
    fn views_preserve_bits_and_mutations() {
        let mut bits = [0x7fc0_1234_u32, 0x8000_0000];
        let floats: &[f32; 2] = view(&bits);
        assert_eq!(floats[0].to_bits(), bits[0]);
        assert_eq!(floats[1].to_bits(), bits[1]);
        view_mut::<_, [f32; 2]>(&mut bits)[1] = f32::from_bits(0xffff_ffff);
        assert_eq!(bits[1], 0xffff_ffff);
        view_mut::<_, [u8; 8]>(&mut bits).fill(0xa5);
        assert_eq!(bits, [0xa5a5_a5a5; 2]);
    }

    #[test]
    fn token_views_and_slices_preserve_proofs_bits_and_exclusivity() {
        use crate::simd::generic::{f32x4, u32x4, u64x2};
        use archmage::ScalarToken;
        let token = ScalarToken;
        let mut source = u32x4::from_array(token, [0x7fc0_1234, 0, u32::MAX, 1]);
        assert_eq!(source.bitcast_ref_f32x4()[0].to_bits(), 0x7fc0_1234);
        source.bitcast_mut_f32x4()[1] = -0.0;
        assert_eq!(source[1], 0x8000_0000);
        let bytes = source.as_bytes();
        assert_eq!(f32x4::from_bytes(token, bytes).as_bytes(), bytes);
        assert_eq!(f32x4::from_bytes_owned(token, *bytes).as_bytes(), bytes);

        #[repr(align(8))]
        struct Bytes([u8; 33]);
        let mut bytes = Bytes([0xa5; 33]);
        assert!(vector_slice::<_, u64x2<ScalarToken>, 16>(token, &bytes.0[1..33]).is_none());
        assert!(vector_slice::<_, u64x2<ScalarToken>, 16>(token, &bytes.0[..31]).is_none());
        assert!(
            vector_slice_mut::<_, u64x2<ScalarToken>, 16>(token, &mut bytes.0[1..33]).is_none()
        );
        assert!(vector_slice_mut::<_, u64x2<ScalarToken>, 16>(token, &mut bytes.0[..31]).is_none());
        let vectors =
            vector_slice_mut::<_, u64x2<ScalarToken>, 16>(token, &mut bytes.0[..32]).unwrap();
        vectors[0][1] = 0;
        vectors[1] = u64x2::from_array(token, [u64::MAX; 2]);
        assert_eq!(&bytes.0[8..16], &[0; 8]);
        assert_eq!(&bytes.0[16..32], &[255; 16]);
        assert_eq!(bytes.0[32], 0xa5);
        assert_eq!(
            vector_slice::<_, u64x2<ScalarToken>, 16>(token, &bytes.0[..32])
                .unwrap()
                .len(),
            2
        );
        assert!(
            vector_slice_mut::<_, u64x2<ScalarToken>, 16>(token, &mut bytes.0[..0])
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn array_cast_preserves_order() {
        let lanes = [0_u32, 1, u32::MAX, 0x8000_0000];
        let halves: [[u32; 2]; 2] = cast(lanes);
        assert_eq!(halves, [[0, 1], [u32::MAX, 0x8000_0000]]);
        assert_eq!(cast::<_, [u32; 4]>(halves), lanes);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn wasm_vectors_preserve_bits_order_and_unaligned_destinations() {
        use core::arch::wasm32::v128;

        #[repr(align(16))]
        struct Bytes([u8; 66]);
        let input: [u8; 64] = core::array::from_fn(|i| (i * 37 + 11) as u8);
        let vectors: [v128; 4] = cast(input);
        assert_eq!(cast::<_, [u8; 16]>(vectors[0]), input[..16]);
        assert_eq!(cast::<_, [u8; 16]>(vectors[3]), input[48..]);
        let mut bytes = Bytes([0xa5; 66]);
        let destination: &mut [u8; 64] = (&mut bytes.0[1..65]).try_into().unwrap();
        store(vectors, destination);
        let reloaded: [v128; 4] = copy(destination);
        assert_eq!(cast::<_, [u8; 64]>(reloaded), input);
        assert_eq!(bytes.0[0], 0xa5);
        assert_eq!(bytes.0[65], 0xa5);
    }
}

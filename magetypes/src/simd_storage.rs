//! Checked bit copies for backend storage, inspired by fearless_simd's transmute module:
//! https://github.com/linebender/fearless_simd/blob/main/fearless_simd/src/transmute.rs
//!
//! Only raw scalar/vector storage implements Pod. Never implement it for tokens
//! or token-bearing SIMD wrappers: arbitrary bytes must not manufacture proofs.

use core::mem::size_of;

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

/// Copy same-sized storage without requiring the source's alignment to match Dst.
#[inline(always)]
pub(crate) fn copy<Src: Pod, Dst: Pod>(src: &Src) -> Dst {
    const { assert!(size_of::<Src>() == size_of::<Dst>()) };
    // SAFETY: Pod guarantees initialized bytes and validity for both types;
    // the const check guarantees equal sizes. transmute_copy handles alignment.
    unsafe { core::mem::transmute_copy(src) }
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
    fn array_cast_preserves_order() {
        let lanes = [0_u32, 1, u32::MAX, 0x8000_0000];
        let halves: [[u32; 2]; 2] = cast(lanes);
        assert_eq!(halves, [[0, 1], [u32::MAX, 0x8000_0000]]);
        assert_eq!(cast::<_, [u32; 4]>(halves), lanes);
    }
}

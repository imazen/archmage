//! Checked copies and views for backend storage, inspired by fearless_simd's transmute module:
//! <https://github.com/linebender/fearless_simd/blob/main/fearless_simd/src/transmute.rs>
//!
//! Only raw scalar/vector storage implements Pod. Never implement it for tokens
//! or token-bearing SIMD wrappers: arbitrary bytes must not manufacture proofs.
//!
//! This is the only module in magetypes that may contain `unsafe`. The crate
//! root denies `unsafe_code` and allows it for this module alone, and
//! `cargo xtask soundness` rejects the `unsafe` keyword and any other
//! `allow(unsafe_code)` elsewhere in `magetypes/src`.

use core::mem::{align_of, size_of};

/// Plain storage with no padding, pointers, or invalid bit patterns.
///
/// # Safety
/// Every byte must be initialized, every bit pattern valid, and the type must
/// contain no pointers or interior mutability. Copying its bits must be sound.
pub(crate) unsafe trait Pod: Copy {}

/// Register a type as [`Pod`], stating its layout in bytes.
///
/// The number is **the sum of the type's field sizes**, not `size_of`, and the
/// two are asserted equal. For every type registered here they coincide, which
/// is the point: a type with padding has `size_of` strictly greater than the sum
/// of its fields, so writing the honest field total makes the assert fire.
///
/// That matters because padding is the one `Pod` violation with no other
/// backstop. `copy` reaches `transmute_copy`, which reads `size_of::<Dst>()`
/// bytes; over a padded type those bytes are uninitialized and reading them is
/// undefined behavior — Miri reports `constructing invalid value: encountered
/// uninitialized memory`. Nothing else in this module can catch it: the size and
/// alignment asserts at every call site pass, because a padded type has a
/// perfectly ordinary size and alignment.
///
/// So: adding a type here means asserting, by hand and on the record, that its
/// bytes are all real. The other `Pod` obligations — no pointers, no validity
/// invariant (no `bool`, `char`, `NonZero*`, enums), no interior mutability —
/// stay the author's to check; only the byte total is mechanical.
macro_rules! impl_pod {
    ($($ty:ty => $field_bytes:expr),+ $(,)?) => {$(
        // SAFETY: numeric scalars and stdarch vectors have no padding or pointers,
        // and accept every bit pattern. Vector layout matches the lane array,
        // with potentially stronger alignment; the helpers handle that difference.
        // The padding half of that claim is checked below.
        unsafe impl Pod for $ty {}
        const _: () = assert!(
            size_of::<$ty>() == $field_bytes,
            concat!(
                "Pod registration for `", stringify!($ty), "` declares a field-byte \
                 total that does not equal its size_of. Either the declared total is \
                 wrong, or the type has padding — and a padded type is not Pod, \
                 because `copy`/`cast` would read its uninitialized padding bytes."
            )
        );
    )+};
}

impl_pod!(
    u8 => 1, i8 => 1,
    u16 => 2, i16 => 2,
    u32 => 4, i32 => 4,
    u64 => 8, i64 => 8,
    f32 => 4, f64 => 8,
);
// SAFETY: arrays add no padding between elements and preserve their validity.
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

#[cfg(target_arch = "x86_64")]
const _: () = {
    use core::arch::x86_64::*;
    impl_pod!(
        __m128 => 16, __m128d => 16, __m128i => 16,
        __m256 => 32, __m256d => 32, __m256i => 32,
    );
    #[cfg(feature = "avx512")]
    impl_pod!(__m512 => 64, __m512d => 64, __m512i => 64);
};

#[cfg(target_arch = "aarch64")]
const _: () = {
    use core::arch::aarch64::*;
    impl_pod!(
        float32x2_t => 8,   float32x4_t => 16,
        float64x1_t => 8,   float64x2_t => 16,
        int8x8_t => 8,      int8x16_t => 16,
        uint8x8_t => 8,     uint8x16_t => 16,
        int16x4_t => 8,     int16x8_t => 16,
        uint16x4_t => 8,    uint16x8_t => 16,
        int32x2_t => 8,     int32x4_t => 16,
        uint32x2_t => 8,    uint32x4_t => 16,
        int64x1_t => 8,     int64x2_t => 16,
        uint64x1_t => 8,    uint64x2_t => 16,
    );
};

#[cfg(target_arch = "wasm32")]
const _: () = {
    // v128 is exactly 16 initialized bytes, accepts every bit pattern, and
    // contains no pointers. Arrays of v128 have no inter-element padding.
    impl_pod!(core::arch::wasm32::v128 => 16);
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
/// the generated SIMD wrappers over sealed backend implementations, and only
/// through [`impl_token_storage!`], which checks the layout.
pub(crate) unsafe trait TokenStorage: Copy {
    type Token: archmage::SimdToken;

    /// Layout assertions for Self; every helper below evaluates them.
    const LAYOUT: ();
}

#[inline(always)]
fn check_token_layout<Dst: TokenStorage>() {
    const {
        assert!(size_of::<Dst::Token>() == 0);
        assert!(align_of::<Dst::Token>() == 1);
        Dst::LAYOUT
    }
}

/// Implement [`TokenStorage`] for a generated vector wrapper `$ty<T>`.
///
/// Every obligation of the impl that the compiler can check, it checks:
///
/// - `$ty(repr, token)` must build the wrapper from exactly a `T::Repr` and a
///   `T`, in that order, so the wrapper has those two fields and no others.
///   This fails at expansion.
/// - `T::Repr` must sit at offset 0, and the wrapper must be exactly as large
///   as `T::Repr` (`LAYOUT`). With the token zero-sized and alignment 1
///   (`check_token_layout`), that leaves no padding. These fail when a helper
///   instantiates the impl.
///
/// What remains is what the backend traits and `archmage` already guarantee:
/// `T::Repr: Pod` is a bound on every backend's `Repr`, and the token types
/// are sealed zero-sized proofs. The invocations sit next to each struct in
/// `simd/generic/generated/`; the `allow(unsafe_code)` below is what lets the
/// expansion compile there, so this macro is the only way to write the impl.
macro_rules! impl_token_storage {
    ($ty:ident, $backend:ident) => {
        // SAFETY: the checks in this expansion pin the layout TokenStorage
        // requires: exactly a Pod `T::Repr` at offset 0 followed by the sealed
        // zero-sized token `T`. A supplied `T` proves CPU support, and the
        // wrapper adds no invariants of its own to the representation's bits.
        #[allow(unsafe_code)]
        unsafe impl<T: crate::simd::backends::$backend> crate::simd_storage::TokenStorage
            for $ty<T>
        {
            type Token = T;
            const LAYOUT: () = {
                assert!(core::mem::offset_of!($ty<T>, 0) == 0);
                assert!(core::mem::size_of::<$ty<T>>() == core::mem::size_of::<T::Repr>());
            };
        }
        const _: () = {
            // Compiles only if the wrapper is exactly `(T::Repr, T)`.
            #[allow(dead_code)]
            fn fields_are_repr_then_token<T: crate::simd::backends::$backend>(
                repr: T::Repr,
                token: T,
            ) -> $ty<T> {
                $ty(repr, token)
            }
        };
    };
}
pub(crate) use impl_token_storage;

/// Marker trait for types that can be upcast with proof of context.
///
/// Upcasting requires being in the appropriate context (inside `#[arcane]`
/// with the right token).
///
/// Public as `magetypes::cast::Upcast`. It is defined here only because it
/// declares an `unsafe fn`, and this module is the one place in magetypes
/// allowed to; nothing in magetypes implements it.
pub trait Upcast<T> {
    /// Upcast to a wider context type.
    ///
    /// # Safety
    ///
    /// Caller must ensure they are in an appropriate SIMD context
    /// (inside `#[arcane]` function with matching token).
    unsafe fn upcast(self) -> T;
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

/// AVX-512 gather and scatter: the pointer-taking half of the `u32x16`,
/// `i32x16` and `f32x16` methods in `simd/generic/gather.rs`.
///
/// The intrinsics access `base + 4 * offset` for per-lane signed 32-bit
/// offsets, so the borrow alone proves nothing about the addresses. Every
/// helper bounds the offsets against the borrow first, so each lane the
/// instruction accesses has `0 <= offset < len`:
///
/// - Wrapping gathers mask with `N - 1`, where `N` is a power of two no larger
///   than 2^31 (const-asserted). Every offset is in `0..N`.
/// - Slice gathers and scatters enable only lanes whose unsigned index is below
///   `min(len, 2^31)`. Masked-off lanes access no memory, so an empty slice is
///   fine, and enabled offsets stay non-negative after sign extension.
/// - Elements are 4-byte `Pod`: reads see only initialized bytes, and scatters
///   (through `&mut`) may write any bit pattern.
/// - Each helper is an `#[arcane]` region for the `X64V4Token` it takes.
///
/// `cargo xtask soundness` rejects gather and scatter intrinsics anywhere else
/// in magetypes.
///
/// # The intrinsics, as Intel specifies them
///
/// Each entry quotes the Intel Intrinsics Guide (data version 3.6.9, 2024-07-12,
/// the copy Rust's stdarch vendors as `library/stdarch/intrinsics_data/x86-intel.xml`)
/// and links to the live guide. Reading the pseudocode:
///
/// - `MEM` is addressed in bits: `MEM[addr+31:addr]` is the 32 bits starting at
///   `addr`, and an offset from an address is written in bits. That is what the
///   `* 8` in the gather and scatter `addr` lines does (the guide's compress-store
///   entries likewise advance their address by `size := 32` per 32-bit element).
///   In bytes, lane `j` accesses the 4 bytes at
///   `base_addr + SignExtend64(vindex[j]) * scale`.
/// - In the masked forms `MEM` appears only inside `IF k[j]`, so a lane whose
///   mask bit is clear reads or writes nothing.
/// - The loops run `j` from 0 to 15. When scatter lanes share an index, the
///   highest lane's value is the one left in memory, which `scatter_select`
///   documents and `tests/gather_scatter_v4.rs` checks.
/// - Every call here passes scale 4, the element size, so a lane with a
///   non-negative index `v` accesses element `v` of the slice.
/// - Rust takes `scale` as the const parameter `SCALE` and renames the arguments:
///   `slice` is `base_addr`, `offsets` is `vindex`, `mask` is `k`, and a scatter's
///   `src` is `a`. The Rust signatures are Rust 1.99's `core::arch::x86_64`.
///
/// ## `_mm512_set1_epi32`
///
/// [Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_set1_epi32)
///
/// - Synopsis: `__m512i _mm512_set1_epi32(int a)`
/// - Instruction: `VPBROADCASTD zmm, r32`
/// - CPUID flags: AVX512F
/// - Rust: `pub const fn _mm512_set1_epi32(a: i32) -> __m512i`
///
/// Description:
///
/// > Broadcast 32-bit integer "a" to all elements of "dst".
///
/// Operation:
///
/// ```text
/// FOR j := 0 to 15
///     i := j*32
///     dst[i+31:i] := a[31:0]
/// ENDFOR
/// dst[MAX:512] := 0
/// ```
///
/// ## `_mm512_and_si512`
///
/// [Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_and_si512)
///
/// - Synopsis: `__m512i _mm512_and_si512(__m512i a, __m512i b)`
/// - Instruction: `VPANDD zmm, zmm, zmm`
/// - CPUID flags: AVX512F
/// - Rust: `pub const fn _mm512_and_si512(a: __m512i, b: __m512i) -> __m512i`
///
/// Description:
///
/// > Compute the bitwise AND of 512 bits (representing integer data) in "a" and
/// > "b", and store the result in "dst".
///
/// Operation:
///
/// ```text
/// dst[511:0] := (a[511:0] AND b[511:0])
/// dst[MAX:512] := 0
/// ```
///
/// ## `_mm512_cmplt_epu32_mask`
///
/// [Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_cmplt_epu32_mask)
///
/// - Synopsis: `__mmask16 _mm512_cmplt_epu32_mask(__m512i a, __m512i b)`
/// - Instruction: `VPCMPUD k, zmm, zmm, imm8`
/// - CPUID flags: AVX512F
/// - Rust: `pub const fn _mm512_cmplt_epu32_mask(a: __m512i, b: __m512i) -> __mmask16`
///
/// Description:
///
/// > Compare packed unsigned 32-bit integers in "a" and "b" for less-than, and
/// > store the results in mask vector "k".
///
/// Operation:
///
/// ```text
/// FOR j := 0 to 15
///     i := j*32
///     k[j] := ( a[i+31:i] < b[i+31:i] ) ? 1 : 0
/// ENDFOR
/// k[MAX:16] := 0
/// ```
///
/// ## `_mm512_i32gather_epi32`
///
/// [Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_i32gather_epi32)
///
/// - Synopsis: `__m512i _mm512_i32gather_epi32(__m512i vindex, void const* base_addr, int scale)`
/// - Instruction: `VPGATHERDD zmm, vm32z`
/// - CPUID flags: AVX512F
/// - Rust: `pub unsafe fn _mm512_i32gather_epi32<const SCALE: i32>(offsets: __m512i, slice: *const i32) -> __m512i`
///
/// Description:
///
/// > Gather 32-bit integers from memory using 32-bit indices. 32-bit elements are
/// > loaded from addresses starting at "base_addr" and offset by each 32-bit
/// > element in "vindex" (each index is scaled by the factor in "scale"). Gathered
/// > elements are merged into "dst". "scale" should be 1, 2, 4 or 8.
///
/// Operation:
///
/// ```text
/// FOR j := 0 to 15
///     i := j*32
///     m := j*32
///     addr := base_addr + SignExtend64(vindex[m+31:m]) * ZeroExtend64(scale) * 8
///     dst[i+31:i] := MEM[addr+31:addr]
/// ENDFOR
/// dst[MAX:512] := 0
/// ```
///
/// ## `_mm512_i32gather_ps`
///
/// [Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_i32gather_ps)
///
/// - Synopsis: `__m512 _mm512_i32gather_ps(__m512i vindex, void const* base_addr, int scale)`
/// - Instruction: `VGATHERDPS zmm, vm32z`
/// - CPUID flags: AVX512F
/// - Rust: `pub unsafe fn _mm512_i32gather_ps<const SCALE: i32>(offsets: __m512i, slice: *const f32) -> __m512`
///
/// Description:
///
/// > Gather single-precision (32-bit) floating-point elements from memory using
/// > 32-bit indices. 32-bit elements are loaded from addresses starting at
/// > "base_addr" and offset by each 32-bit element in "vindex" (each index is
/// > scaled by the factor in "scale"). Gathered elements are merged into "dst".
/// > "scale" should be 1, 2, 4 or 8.
///
/// Operation:
///
/// ```text
/// FOR j := 0 to 15
///     i := j*32
///     m := j*32
///     addr := base_addr + SignExtend64(vindex[m+31:m]) * ZeroExtend64(scale) * 8
///     dst[i+31:i] := MEM[addr+31:addr]
/// ENDFOR
/// dst[MAX:512] := 0
/// ```
///
/// ## `_mm512_mask_i32gather_epi32`
///
/// [Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_i32gather_epi32)
///
/// - Synopsis: `__m512i _mm512_mask_i32gather_epi32(__m512i src, __mmask16 k, __m512i vindex, void const* base_addr, int scale)`
/// - Instruction: `VPGATHERDD zmm {k}, vm32z`
/// - CPUID flags: AVX512F
/// - Rust: `pub unsafe fn _mm512_mask_i32gather_epi32<const SCALE: i32>(src: __m512i, mask: __mmask16, offsets: __m512i, slice: *const i32) -> __m512i`
///
/// Description:
///
/// > Gather 32-bit integers from memory using 32-bit indices. 32-bit elements are
/// > loaded from addresses starting at "base_addr" and offset by each 32-bit
/// > element in "vindex" (each index is scaled by the factor in "scale"). Gathered
/// > elements are merged into "dst" using writemask "k" (elements are copied from
/// > "src" when the corresponding mask bit is not set). "scale" should be 1, 2, 4
/// > or 8.
///
/// Operation:
///
/// ```text
/// FOR j := 0 to 15
///     i := j*32
///     m := j*32
///     IF k[j]
///         addr := base_addr + SignExtend64(vindex[m+31:m]) * ZeroExtend64(scale) * 8
///         dst[i+31:i] := MEM[addr+31:addr]
///     ELSE
///         dst[i+31:i] := src[i+31:i]
///     FI
/// ENDFOR
/// dst[MAX:512] := 0
/// ```
///
/// ## `_mm512_mask_i32gather_ps`
///
/// [Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_i32gather_ps)
///
/// - Synopsis: `__m512 _mm512_mask_i32gather_ps(__m512 src, __mmask16 k, __m512i vindex, void const* base_addr, int scale)`
/// - Instruction: `VGATHERDPS zmm {k}, vm32z`
/// - CPUID flags: AVX512F
/// - Rust: `pub unsafe fn _mm512_mask_i32gather_ps<const SCALE: i32>(src: __m512, mask: __mmask16, offsets: __m512i, slice: *const f32) -> __m512`
///
/// Description:
///
/// > Gather single-precision (32-bit) floating-point elements from memory using
/// > 32-bit indices. 32-bit elements are loaded from addresses starting at
/// > "base_addr" and offset by each 32-bit element in "vindex" (each index is
/// > scaled by the factor in "scale"). Gathered elements are merged into "dst"
/// > using writemask "k" (elements are copied from "src" when the corresponding
/// > mask bit is not set). "scale" should be 1, 2, 4 or 8.
///
/// Operation:
///
/// ```text
/// FOR j := 0 to 15
///     i := j*32
///     m := j*32
///     IF k[j]
///         addr := base_addr + SignExtend64(vindex[m+31:m]) * ZeroExtend64(scale) * 8
///         dst[i+31:i] := MEM[addr+31:addr]
///     ELSE
///         dst[i+31:i] := src[i+31:i]
///     FI
/// ENDFOR
/// dst[MAX:512] := 0
/// ```
///
/// ## `_mm512_mask_i32scatter_epi32`
///
/// [Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_i32scatter_epi32)
///
/// - Synopsis: `void _mm512_mask_i32scatter_epi32(void* base_addr, __mmask16 k, __m512i vindex, __m512i a, int scale)`
/// - Instruction: `VPSCATTERDD vm32z {k}, zmm`
/// - CPUID flags: AVX512F
/// - Rust: `pub unsafe fn _mm512_mask_i32scatter_epi32<const SCALE: i32>(slice: *mut i32, mask: __mmask16, offsets: __m512i, src: __m512i)`
///
/// Description:
///
/// > Scatter 32-bit integers from "a" into memory using 32-bit indices. 32-bit
/// > elements are stored at addresses starting at "base_addr" and offset by each
/// > 32-bit element in "vindex" (each index is scaled by the factor in "scale")
/// > subject to mask "k" (elements are not stored when the corresponding mask bit
/// > is not set). "scale" should be 1, 2, 4 or 8.
///
/// Operation:
///
/// ```text
/// FOR j := 0 to 15
///     i := j*32
///     m := j*32
///     IF k[j]
///         addr := base_addr + SignExtend64(vindex[m+31:m]) * ZeroExtend64(scale) * 8
///         MEM[addr+31:addr] := a[i+31:i]
///     FI
/// ENDFOR
/// ```
///
/// ## `_mm512_mask_i32scatter_ps`
///
/// [Intel's entry](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_i32scatter_ps)
///
/// - Synopsis: `void _mm512_mask_i32scatter_ps(void* base_addr, __mmask16 k, __m512i vindex, __m512 a, int scale)`
/// - Instruction: `VSCATTERDPS vm32z {k}, zmm`
/// - CPUID flags: AVX512F
/// - Rust: `pub unsafe fn _mm512_mask_i32scatter_ps<const SCALE: i32>(slice: *mut f32, mask: __mmask16, offsets: __m512i, src: __m512)`
///
/// Description:
///
/// > Scatter single-precision (32-bit) floating-point elements from "a" into memory
/// > using 32-bit indices. 32-bit elements are stored at addresses starting at
/// > "base_addr" and offset by each 32-bit element in "vindex" (each index is
/// > scaled by the factor in "scale") subject to mask "k" (elements are not stored
/// > when the corresponding mask bit is not set). "scale" should be 1, 2, 4 or 8.
///
/// Operation:
///
/// ```text
/// FOR j := 0 to 15
///     i := j*32
///     m := j*32
///     IF k[j]
///         addr := base_addr + SignExtend64(vindex[m+31:m]) * ZeroExtend64(scale) * 8
///         MEM[addr+31:addr] := a[i+31:i]
///     FI
/// ENDFOR
/// ```
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod gather {
    use super::Pod;
    use archmage::X64V4Token;
    use core::arch::x86_64::{__m512, __m512i};

    const fn assert_wrapping_table<const N: usize>() {
        assert!(
            N.is_power_of_two() && N <= 1 << 31,
            "gather_wrapping: the table length must be a power of two no larger than 2^31"
        );
    }

    const fn assert_lane<E>() {
        assert!(size_of::<E>() == 4, "gather/scatter elements are 4 bytes");
    }

    /// Exclusive bound on enabled indices for a slice of `len` elements:
    /// `min(len, 2^31)`, as the bit pattern of the unsigned compare operand.
    #[inline(always)]
    fn lane_bound(len: usize) -> i32 {
        len.min(1 << 31) as u32 as i32
    }

    #[archmage::arcane(import_intrinsics)]
    pub(crate) fn gather_wrapping_epi32<E: Pod, const N: usize>(
        _token: X64V4Token,
        table: &[E; N],
        idx: __m512i,
    ) -> __m512i {
        const { assert_wrapping_table::<N>() };
        const { assert_lane::<E>() };
        let off = _mm512_and_si512(idx, _mm512_set1_epi32((N - 1) as i32));
        // SAFETY: per the `_mm512_i32gather_epi32` Operation above, lane `j` reads
        // 4 bytes at `table + SignExtend64(off[j]) * 4`. `off[j] = idx[j] & (N - 1)`
        // is in `0..N` (`N` a power of two <= 2^31), so sign extension keeps it
        // and the read is `table[off[j]]`.
        unsafe { _mm512_i32gather_epi32::<4>(off, table.as_ptr().cast()) }
    }

    #[archmage::arcane(import_intrinsics)]
    pub(crate) fn gather_wrapping_ps<const N: usize>(
        _token: X64V4Token,
        table: &[f32; N],
        idx: __m512i,
    ) -> __m512 {
        const { assert_wrapping_table::<N>() };
        let off = _mm512_and_si512(idx, _mm512_set1_epi32((N - 1) as i32));
        // SAFETY: per the `_mm512_i32gather_ps` Operation above, lane `j` reads
        // 4 bytes at `table + SignExtend64(off[j]) * 4`. `off[j] = idx[j] & (N - 1)`
        // is in `0..N` (`N` a power of two <= 2^31), so sign extension keeps it
        // and the read is `table[off[j]]`.
        unsafe { _mm512_i32gather_ps::<4>(off, table.as_ptr()) }
    }

    #[archmage::arcane(import_intrinsics)]
    pub(crate) fn gather_or_epi32<E: Pod>(
        _token: X64V4Token,
        table: &[E],
        idx: __m512i,
        or: __m512i,
    ) -> __m512i {
        const { assert_lane::<E>() };
        let live = _mm512_cmplt_epu32_mask(idx, _mm512_set1_epi32(lane_bound(table.len())));
        // SAFETY: per the `_mm512_mask_i32gather_epi32` Operation above, lanes with
        // `live[j]` clear read nothing and copy `or[j]`; the rest read 4 bytes at
        // `table + SignExtend64(idx[j]) * 4`. `live[j]` means `idx[j] < min(len, 2^31)`
        // unsigned, so sign extension keeps `idx[j]` and the read is `table[idx[j]]`.
        unsafe { _mm512_mask_i32gather_epi32::<4>(or, live, idx, table.as_ptr().cast()) }
    }

    #[archmage::arcane(import_intrinsics)]
    pub(crate) fn gather_or_ps(
        _token: X64V4Token,
        table: &[f32],
        idx: __m512i,
        or: __m512,
    ) -> __m512 {
        let live = _mm512_cmplt_epu32_mask(idx, _mm512_set1_epi32(lane_bound(table.len())));
        // SAFETY: per the `_mm512_mask_i32gather_ps` Operation above, lanes with
        // `live[j]` clear read nothing and copy `or[j]`; the rest read 4 bytes at
        // `table + SignExtend64(idx[j]) * 4`. `live[j]` means `idx[j] < min(len, 2^31)`
        // unsigned, so sign extension keeps `idx[j]` and the read is `table[idx[j]]`.
        unsafe { _mm512_mask_i32gather_ps::<4>(or, live, idx, table.as_ptr()) }
    }

    #[archmage::arcane(import_intrinsics)]
    pub(crate) fn scatter_select_epi32<E: Pod>(
        _token: X64V4Token,
        dst: &mut [E],
        enable: u16,
        idx: __m512i,
        v: __m512i,
    ) {
        const { assert_lane::<E>() };
        let live = enable & _mm512_cmplt_epu32_mask(idx, _mm512_set1_epi32(lane_bound(dst.len())));
        // SAFETY: per the `_mm512_mask_i32scatter_epi32` Operation above, only lanes
        // with `live[j]` set write: 4 bytes at `dst + SignExtend64(idx[j]) * 4`.
        // `live[j]` implies unsigned `idx[j] < min(len, 2^31)`, so that is `dst[idx[j]]`,
        // exclusively borrowed here. `E` is Pod, so any written bit pattern is valid.
        unsafe { _mm512_mask_i32scatter_epi32::<4>(dst.as_mut_ptr().cast(), live, idx, v) }
    }

    #[archmage::arcane(import_intrinsics)]
    pub(crate) fn scatter_select_ps(
        _token: X64V4Token,
        dst: &mut [f32],
        enable: u16,
        idx: __m512i,
        v: __m512,
    ) {
        let live = enable & _mm512_cmplt_epu32_mask(idx, _mm512_set1_epi32(lane_bound(dst.len())));
        // SAFETY: per the `_mm512_mask_i32scatter_ps` Operation above, only lanes
        // with `live[j]` set write: 4 bytes at `dst + SignExtend64(idx[j]) * 4`.
        // `live[j]` implies unsigned `idx[j] < min(len, 2^31)`, so that is `dst[idx[j]]`,
        // exclusively borrowed here. Any written bit pattern is a valid `f32`.
        unsafe { _mm512_mask_i32scatter_ps::<4>(dst.as_mut_ptr(), live, idx, v) }
    }
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
        let mut source = u32x4::from_array_t(token, [0x7fc0_1234, 0, u32::MAX, 1]);
        assert_eq!(source.bitcast_ref_f32x4()[0].to_bits(), 0x7fc0_1234);
        source.bitcast_mut_f32x4()[1] = -0.0;
        assert_eq!(source[1], 0x8000_0000);
        let bytes = source.as_bytes();
        assert_eq!(f32x4::from_bytes_t(token, bytes).as_bytes(), bytes);
        assert_eq!(f32x4::from_bytes_owned_t(token, *bytes).as_bytes(), bytes);

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
        vectors[1] = u64x2::from_array_t(token, [u64::MAX; 2]);
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

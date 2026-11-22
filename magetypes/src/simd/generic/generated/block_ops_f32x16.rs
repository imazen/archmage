//! Block layout operations for `f32x16<T>`.

use crate::simd::backends::F32x16Backend;
use crate::simd::generic::f32x16;

/// ### Interleave and matrix blocks
impl<T: F32x16Backend> f32x16<T> {
    // Each index selects from the two input vectors concatenated in order.
    #[inline(always)]
    pub(crate) fn permute2(self, other: Self, indices: [i32; 16]) -> Self {
        let token = self.1;
        Self(T::permute2_wide(token, [self.0, other.0], indices), token)
    }

    /// Interleave the low two elements of each four-lane group.
    ///
    /// Each group `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]` yields
    /// `[a0, b0, a1, b1]`.
    #[inline(always)]
    pub fn interleave_lo(self, other: Self) -> Self {
        self.permute2(
            other,
            [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29],
        )
    }

    /// Interleave the high two elements of each four-lane group.
    ///
    /// Each group `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]` yields
    /// `[a2, b2, a3, b3]`.
    #[inline(always)]
    pub fn interleave_hi(self, other: Self) -> Self {
        self.permute2(
            other,
            [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31],
        )
    }

    /// Interleave within four-lane groups, returning the low and high results.
    #[inline(always)]
    pub fn interleave(self, other: Self) -> (Self, Self) {
        (self.interleave_lo(other), self.interleave_hi(other))
    }

    /// Transpose two adjacent row-major 8x8 blocks in place.
    ///
    /// Lanes `0..8` and `8..16` of each row belong to independent blocks.
    #[inline(always)]
    pub fn transpose_dual_8x8(rows: &mut [Self; 8]) {
        *rows = Self::transpose_dual_8x8_copy(*rows);
    }

    /// Transpose two adjacent 8x8 blocks, returning the transposed rows.
    #[inline(always)]
    pub fn transpose_dual_8x8_copy(rows: [Self; 8]) -> [Self; 8] {
        // Transpose four rows independently in each four-lane group.
        #[inline(always)]
        fn transpose_four<T: F32x16Backend>([a, b, c, d]: [f32x16<T>; 4]) -> [f32x16<T>; 4] {
            const LO: [i32; 16] = [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29];
            const HI: [i32; 16] = [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31];
            let (ab0, ab1) = a.interleave(b);
            let (cd0, cd1) = c.interleave(d);
            [
                ab0.permute2(cd0, LO),
                ab0.permute2(cd0, HI),
                ab1.permute2(cd1, LO),
                ab1.permute2(cd1, HI),
            ]
        }

        let [a, b, c, d, e, f, g, h] = rows;
        let top = transpose_four([a, b, c, d]);
        let bottom = transpose_four([e, f, g, h]);
        // Join four-lane groups from the two sets of rows within each half.
        const LO: [i32; 16] = [0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27];
        const HI: [i32; 16] = [4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31];
        [
            top[0].permute2(bottom[0], LO),
            top[1].permute2(bottom[1], LO),
            top[2].permute2(bottom[2], LO),
            top[3].permute2(bottom[3], LO),
            top[0].permute2(bottom[0], HI),
            top[1].permute2(bottom[1], HI),
            top[2].permute2(bottom[2], HI),
            top[3].permute2(bottom[3], HI),
        ]
    }

    /// Transpose a row-major 16x16 block in place.
    #[inline(always)]
    pub fn transpose_16x16(rows: &mut [Self; 16]) {
        *rows = Self::transpose_16x16_copy(*rows);
    }

    /// Transpose a 16x16 block, returning the transposed rows.
    #[inline(always)]
    pub fn transpose_16x16_copy(rows: [Self; 16]) -> [Self; 16] {
        let (halves, _) = rows.as_chunks::<8>();
        let top = Self::transpose_dual_8x8_copy(halves[0]);
        let bottom = Self::transpose_dual_8x8_copy(halves[1]);
        const LO: [i32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23];
        const HI: [i32; 16] = [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31];
        [
            top[0].permute2(bottom[0], LO),
            top[1].permute2(bottom[1], LO),
            top[2].permute2(bottom[2], LO),
            top[3].permute2(bottom[3], LO),
            top[4].permute2(bottom[4], LO),
            top[5].permute2(bottom[5], LO),
            top[6].permute2(bottom[6], LO),
            top[7].permute2(bottom[7], LO),
            top[0].permute2(bottom[0], HI),
            top[1].permute2(bottom[1], HI),
            top[2].permute2(bottom[2], HI),
            top[3].permute2(bottom[3], HI),
            top[4].permute2(bottom[4], HI),
            top[5].permute2(bottom[5], HI),
            top[6].permute2(bottom[6], HI),
            top[7].permute2(bottom[7], HI),
        ]
    }

    /// Load eight contiguous rows of sixteen floats as two adjacent 8x8 blocks.
    #[inline(always)]
    pub fn load_dual_8x8_t(token: T, block: &[f32; 128]) -> [Self; 8] {
        core::array::from_fn(|i| Self::load_t(token, &block.as_chunks::<16>().0[i]))
    }

    /// Store two adjacent 8x8 blocks as eight contiguous rows of sixteen floats.
    #[inline(always)]
    pub fn store_dual_8x8(rows: &[Self; 8], block: &mut [f32; 128]) {
        for (row, dst) in rows.iter().zip(block.as_chunks_mut::<16>().0) {
            row.store(dst);
        }
    }

    /// Load sixteen contiguous rows of sixteen floats from a row-major block.
    #[inline(always)]
    pub fn load_16x16_t(token: T, block: &[f32; 256]) -> [Self; 16] {
        core::array::from_fn(|i| Self::load_t(token, &block.as_chunks::<16>().0[i]))
    }

    /// Store sixteen rows of sixteen floats as a contiguous row-major block.
    #[inline(always)]
    pub fn store_16x16(rows: &[Self; 16], block: &mut [f32; 256]) {
        for (row, dst) in rows.iter().zip(block.as_chunks_mut::<16>().0) {
            row.store(dst);
        }
    }
}

// ============================================================================
// Cross-type bitcast references (require F32x16Convert)
// ============================================================================

impl<T: crate::simd::backends::F32x16Convert> f32x16<T> {
    /// Reinterpret bits as `&i32x16<T>` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i32(&self) -> &super::i32x16<T> {
        crate::simd_storage::vector_view(self.1, &self.0)
    }

    /// Reinterpret bits as `&mut i32x16<T>` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i32(&mut self) -> &mut super::i32x16<T> {
        crate::simd_storage::vector_view_mut(self.1, &mut self.0)
    }
}

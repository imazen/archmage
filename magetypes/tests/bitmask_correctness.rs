//! Tests for i16x16/u16x16 bitmask correctness.
//!
//! These tests verify that bitmask() correctly extracts the sign/high bit
//! from ALL 16 lanes of 256-bit i16/u16 vectors. Regression tests for
//! a bug where the pack+extract logic produced incorrect results for
//! the upper 8 lanes.

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use archmage::{SimdToken, X64V3Token};

    // ==================== i16x16 bitmask ====================

    /// Direct bitmask test: each individual lane with sign bit set.
    /// This does NOT use simd_eq — it tests bitmask() in isolation.
    #[test]
    fn i16x16_bitmask_individual_lanes() {
        use magetypes::simd::i16x16;

        let Some(token) = X64V3Token::summon() else {
            return;
        };

        for lane in 0..16u32 {
            let mut arr = [0i16; 16];
            arr[lane as usize] = -1; // sign bit set
            let v = i16x16::from_array(token, arr);
            let mask = v.bitmask();
            assert_eq!(
                mask,
                1 << lane,
                "i16x16: single negative in lane {lane} should give {:#06x}, got {mask:#06x}",
                1u32 << lane
            );
        }
    }

    /// Direct bitmask test: alternating sign pattern across all 16 lanes.
    #[test]
    fn i16x16_bitmask_alternating_pattern() {
        use magetypes::simd::i16x16;

        let Some(token) = X64V3Token::summon() else {
            return;
        };

        // Alternating: negative in even lanes
        let v = i16x16::from_array(
            token,
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0],
        );
        let mask = v.bitmask();
        assert_eq!(
            mask, 0x5555,
            "i16x16: alternating negative-even should give 0x5555, got {mask:#06x}"
        );

        // Opposite: negative in odd lanes
        let v = i16x16::from_array(
            token,
            [0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
        );
        let mask = v.bitmask();
        assert_eq!(
            mask, 0xAAAA,
            "i16x16: alternating negative-odd should give 0xAAAA, got {mask:#06x}"
        );
    }

    /// Direct bitmask test: cross-lane pattern (low half vs high half differ).
    #[test]
    fn i16x16_bitmask_cross_lane() {
        use magetypes::simd::i16x16;

        let Some(token) = X64V3Token::summon() else {
            return;
        };

        // Low lanes negative, high lanes positive
        let v = i16x16::from_array(
            token,
            [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        );
        let mask = v.bitmask();
        assert_eq!(
            mask, 0x00FF,
            "i16x16: low-negative, high-zero should give 0x00FF, got {mask:#06x}"
        );

        // High lanes negative, low lanes positive
        let v = i16x16::from_array(
            token,
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
        );
        let mask = v.bitmask();
        assert_eq!(
            mask, 0xFF00,
            "i16x16: high-negative, low-zero should give 0xFF00, got {mask:#06x}"
        );

        // Mixed: specific pattern crossing the 128-bit boundary
        let v = i16x16::from_array(
            token,
            [-1, 0, -1, 0, -1, 0, -1, 0, 0, -1, 0, -1, 0, -1, 0, -1],
        );
        let mask = v.bitmask();
        assert_eq!(
            mask, 0xAA55,
            "i16x16: cross-lane mixed should give 0xAA55, got {mask:#06x}"
        );
    }

    /// bitmask via simd_eq: verifies the full pipeline (compare + extract).
    #[test]
    fn i16x16_bitmask_via_simd_eq() {
        use magetypes::simd::i16x16;

        let Some(token) = X64V3Token::summon() else {
            return;
        };
        let zero = i16x16::zero(token);

        // Only lanes 8-15 nonzero -> eq mask should have bits 0-7 set
        let v = i16x16::from_array(token, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]);
        let eq = v.simd_eq(zero);
        let mask = eq.bitmask();
        assert_eq!(
            mask, 0x00FF,
            "i16x16: lanes 0-7 zero eq zero should give 0x00FF, got {mask:#06x}"
        );

        // Only lanes 0-7 nonzero -> eq mask should have bits 8-15 set
        let v = i16x16::from_array(token, [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        let eq = v.simd_eq(zero);
        let mask = eq.bitmask();
        assert_eq!(
            mask, 0xFF00,
            "i16x16: lanes 8-15 zero eq zero should give 0xFF00, got {mask:#06x}"
        );
    }

    // ==================== u16x16 bitmask ====================

    /// Direct bitmask test: each individual lane with MSB set.
    #[test]
    fn u16x16_bitmask_individual_lanes() {
        use magetypes::simd::u16x16;

        let Some(token) = X64V3Token::summon() else {
            return;
        };

        for lane in 0..16u32 {
            let mut arr = [0u16; 16];
            arr[lane as usize] = 0x8000; // MSB set
            let v = u16x16::from_array(token, arr);
            let mask = v.bitmask();
            assert_eq!(
                mask,
                1 << lane,
                "u16x16: single MSB in lane {lane} should give {:#06x}, got {mask:#06x}",
                1u32 << lane
            );
        }
    }

    /// Direct bitmask test: cross-lane MSB pattern.
    #[test]
    fn u16x16_bitmask_cross_lane() {
        use magetypes::simd::u16x16;

        let Some(token) = X64V3Token::summon() else {
            return;
        };

        let v = u16x16::from_array(
            token,
            [
                0x8000, 0, 0x8000, 0, 0x8000, 0, 0x8000, 0, 0, 0x8000, 0, 0x8000, 0, 0x8000, 0,
                0x8000,
            ],
        );
        let mask = v.bitmask();
        assert_eq!(
            mask, 0xAA55,
            "u16x16: cross-lane MSB pattern should give 0xAA55, got {mask:#06x}"
        );
    }
}

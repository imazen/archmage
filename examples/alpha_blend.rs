//! Alpha blending with raw intrinsics and #[arcane]
//!
//! Demonstrates premultiply, unpremultiply, and Porter-Duff "over" compositing
//! using raw AVX2 intrinsics inside `#[arcane]` functions. No magetypes dependency.
//!
//! Run: `cargo run --example alpha_blend --release`
//!
//! For a version using magetypes' ergonomic f32x8 API, see:
//! <https://github.com/imazen/archmage/blob/main/magetypes/examples/alpha_blend.rs>

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    println!("This example requires x86_64 with AVX2");
}

#[cfg(target_arch = "x86_64")]
fn main() {
    x86_impl::run();
}

#[cfg(target_arch = "x86_64")]
mod x86_impl {
    use archmage::{SimdToken, X64V3Token, arcane};

    /// Premultiply alpha for 2 RGBA pixels packed as [R0,G0,B0,A0, R1,G1,B1,A1].
    ///
    /// Converts straight alpha to premultiplied: R' = R * A, G' = G * A, B' = B * A
    #[arcane(import_intrinsics)]
    fn premultiply_2px(_token: X64V3Token, pixels: &[f32; 8]) -> [f32; 8] {
        let v = _mm256_loadu_ps(pixels);

        // Broadcast alpha: [A0,A0,A0,A0, A1,A1,A1,A1]
        let alpha = _mm256_permutevar8x32_ps(v, _mm256_set_epi32(7, 7, 7, 7, 3, 3, 3, 3));

        // Multiply all channels by alpha
        let premul = _mm256_mul_ps(v, alpha);

        // Restore original alpha (don't multiply alpha by itself)
        // Mask: blend alpha lanes from original, RGB from premultiplied
        let mask = _mm256_set_epi32(-1, 0, 0, 0, -1, 0, 0, 0); // 1 = keep original
        let result = _mm256_blendv_ps(premul, v, _mm256_castsi256_ps(mask));

        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(&mut out, result);
        out
    }

    /// Unpremultiply alpha for 2 RGBA pixels.
    ///
    /// Converts premultiplied back to straight: R = R' / A, G = G' / A, B = B' / A
    #[arcane(import_intrinsics)]
    fn unpremultiply_2px(_token: X64V3Token, pixels: &[f32; 8]) -> [f32; 8] {
        let v = _mm256_loadu_ps(pixels);

        // Broadcast alpha
        let alpha = _mm256_permutevar8x32_ps(v, _mm256_set_epi32(7, 7, 7, 7, 3, 3, 3, 3));

        // Avoid division by zero: if alpha == 0, use 1.0
        let epsilon = _mm256_set1_ps(1e-10);
        let safe_alpha = _mm256_max_ps(alpha, epsilon);

        // Divide RGB by alpha
        let divided = _mm256_div_ps(v, safe_alpha);

        // Restore original alpha
        let mask = _mm256_set_epi32(-1, 0, 0, 0, -1, 0, 0, 0);
        let result = _mm256_blendv_ps(divided, v, _mm256_castsi256_ps(mask));

        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(&mut out, result);
        out
    }

    /// Porter-Duff "over" compositing for 2 pixel pairs.
    ///
    /// out = src + dst * (1 - src_alpha)
    #[arcane(import_intrinsics)]
    fn composite_over_2px(_token: X64V3Token, src: &[f32; 8], dst: &[f32; 8]) -> [f32; 8] {
        let s = _mm256_loadu_ps(src);
        let d = _mm256_loadu_ps(dst);

        // Broadcast src alpha
        let src_alpha = _mm256_permutevar8x32_ps(s, _mm256_set_epi32(7, 7, 7, 7, 3, 3, 3, 3));

        // (1 - src_alpha) * dst + src
        let one = _mm256_set1_ps(1.0);
        let inv_alpha = _mm256_sub_ps(one, src_alpha);
        let result = _mm256_fmadd_ps(inv_alpha, d, s);

        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(&mut out, result);
        out
    }

    pub fn run() {
        let Some(token) = X64V3Token::summon() else {
            println!("AVX2+FMA not available");
            return;
        };

        // Two RGBA pixels: semi-transparent red and green
        let pixels = [1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.8];
        println!("Input:        {pixels:?}");

        let premul = premultiply_2px(token, &pixels);
        println!("Premultiplied: {premul:?}");
        // Expected: [0.5, 0.0, 0.0, 0.5, 0.0, 0.8, 0.0, 0.8]

        let restored = unpremultiply_2px(token, &premul);
        println!("Restored:     {restored:?}");

        // Composite: semi-transparent blue over white
        let src = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5]; // premultiplied blue @ 50%
        let dst = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // opaque white
        let over = composite_over_2px(token, &src, &dst);
        println!("Over:         {over:?}");
        // Expected: [0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0]
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_premultiply() {
            let Some(token) = X64V3Token::summon() else {
                return;
            };
            let pixels = [1.0, 0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.8];
            let result = premultiply_2px(token, &pixels);
            assert_eq!(result, [0.5, 0.25, 0.0, 0.5, 0.0, 0.8, 0.0, 0.8]);
        }

        #[test]
        fn test_roundtrip() {
            let Some(token) = X64V3Token::summon() else {
                return;
            };
            let pixels = [0.8, 0.4, 0.2, 0.6, 0.1, 0.9, 0.5, 1.0];
            let premul = premultiply_2px(token, &pixels);
            let restored = unpremultiply_2px(token, &premul);
            for (a, b) in pixels.iter().zip(restored.iter()) {
                assert!((a - b).abs() < 1e-5, "{a} != {b}");
            }
        }

        #[test]
        fn test_composite_over() {
            let Some(token) = X64V3Token::summon() else {
                return;
            };
            // Opaque source overwrites destination
            let src = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
            let dst = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0];
            let result = composite_over_2px(token, &src, &dst);
            assert_eq!(result, [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        }
    }
}

//! Tests zero-KLOC sub-tier `#[attune(..., define(...))]` namespace aliasing
//! and token-gated sub-tier operations on base representation vectors.

#![forbid(unsafe_code)]
#![deny(deprecated)]

#[test]
fn foreign_subtier_tokens_never_summon() {
    use archmage::SimdToken;
    #[cfg(not(target_arch = "x86_64"))]
    assert!(archmage::X64V3VnniToken::summon().is_none());
    #[cfg(not(target_arch = "aarch64"))]
    assert!(archmage::Arm64V2Token::summon().is_none());
    #[cfg(not(target_arch = "wasm32"))]
    {
        assert!(archmage::Wasm128RelaxedToken::summon().is_none());
        assert!(archmage::Wasm128RelaxedFmaToken::summon().is_none());
        assert_eq!(
            archmage::Wasm128RelaxedFmaToken::compiled_with(),
            Some(false)
        );
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
mod x86_v4x_w512_delegation_tests {
    use archmage::{SimdToken, X64V4xToken, attune};

    #[attune(_v4x_t, define(f32x8, f32x16, f64x8, i32x16, u64x8))]
    fn check_v4x_w512(token: Token) {
        let e32 = 1.0f32 / 4096.0;
        let a16 = f32x16::splat_t(token, core::hint::black_box(1.0 + e32));
        let c16 = f32x16::splat_t(token, core::hint::black_box(-(1.0 + 2.0 * e32)));
        assert_eq!(a16.mul_add_portable(a16, c16).to_array(), [e32 * e32; 16]);

        let rounded =
            f32x16::from_array_t(token, core::array::from_fn(|i| (i as f32) * 1.5 - 10.25))
                .to_i32_round()
                .to_array();
        assert_eq!(rounded[0], -10);
        assert_eq!(rounded[15], 12);

        let lo = f32x8::splat_t(token, 3.0);
        let hi = f32x8::splat_t(token, 7.0);
        let wide = f32x16::from_halves_t(token, lo, hi);
        assert_eq!(wide.low().to_array(), [3.0; 8]);
        assert_eq!(wide.high().to_array(), [7.0; 8]);

        let e64 = 1.0f64 / 134_217_728.0;
        let a8 = f64x8::splat_t(token, core::hint::black_box(1.0 + e64));
        let c8 = f64x8::splat_t(token, core::hint::black_box(-(1.0 + 2.0 * e64)));
        assert_eq!(a8.mul_add_portable(a8, c8).to_array(), [e64 * e64; 8]);

        let vi = i32x16::from_array_t(token, core::array::from_fn(|i| (1i32 << i) | 1));
        let pop_i = vi.popcnt().to_array();
        assert_eq!(pop_i[0], 1);
        assert_eq!(pop_i[5], 2);

        let vu = u64x8::from_array_t(token, core::array::from_fn(|i| (1u64 << (i * 8)) - 1));
        let pop_u = vu.popcnt().to_array();
        for (i, &got) in pop_u.iter().enumerate() {
            assert_eq!(got, (i * 8) as u64);
        }
    }

    #[test]
    fn v4x_w512_delegation_and_popcnt_work() {
        if let Some(token) = X64V4xToken::summon() {
            check_v4x_w512_v4x_t(token);
            check_v4x_vnni_v4x_t(token);
        }
    }

    #[attune(
        _v4x_t,
        define(
            i32x4, u8x16, i8x16, i16x8, i32x8, u8x32, i8x32, i16x16, i32x16, u8x64, i8x64, i16x32
        )
    )]
    fn check_v4x_vnni(token: Token) {
        let u16_in: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(17).wrapping_add(5));
        let s16_in: [i8; 16] = core::array::from_fn(|i| if i % 2 == 0 { 40 } else { -35 });
        let acc4 = [i32::MAX - 20, i32::MIN + 20, 0, -42];
        let v_acc4 = i32x4::from_array_t(token, acc4);
        let dp4 = v_acc4
            .mul_add_u8_i8_vnni(
                token,
                u8x16::from_array_t(token, u16_in),
                i8x16::from_array_t(token, s16_in),
            )
            .to_array();
        let dps4 = v_acc4
            .mul_add_u8_i8_sat_vnni(
                token,
                u8x16::from_array_t(token, u16_in),
                i8x16::from_array_t(token, s16_in),
            )
            .to_array();
        for k in 0..4 {
            let prod: i32 = (0..4)
                .map(|j| (u16_in[4 * k + j] as i32) * (s16_in[4 * k + j] as i32))
                .sum();
            assert_eq!(dp4[k], acc4[k].wrapping_add(prod));
            assert_eq!(dps4[k], acc4[k].saturating_add(prod));
        }

        let w8_a: [i16; 8] = [
            30000,
            20000,
            -30000,
            -20000,
            1234,
            -5678,
            i16::MIN,
            i16::MAX,
        ];
        let w8_b: [i16; 8] = [2, 3, 2, 3, -7, 11, -1, 1];
        let dw4 = v_acc4
            .mul_add_i16_vnni(
                token,
                i16x8::from_array_t(token, w8_a),
                i16x8::from_array_t(token, w8_b),
            )
            .to_array();
        let dws4 = v_acc4
            .mul_add_i16_sat_vnni(
                token,
                i16x8::from_array_t(token, w8_a),
                i16x8::from_array_t(token, w8_b),
            )
            .to_array();
        for k in 0..4 {
            let prod = (w8_a[2 * k] as i32) * (w8_b[2 * k] as i32)
                + (w8_a[2 * k + 1] as i32) * (w8_b[2 * k + 1] as i32);
            assert_eq!(dw4[k], acc4[k].wrapping_add(prod));
            assert_eq!(dws4[k], acc4[k].saturating_add(prod));
        }

        let acc8: [i32; 8] = core::array::from_fn(|i| (i as i32 - 4) * 1000);
        let u32_in: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(11).wrapping_add(3));
        let s32_in: [i8; 32] = core::array::from_fn(|i| (i as i8).wrapping_mul(5).wrapping_sub(60));
        let dp8 = i32x8::from_array_t(token, acc8)
            .mul_add_u8_i8_vnni(
                token,
                u8x32::from_array_t(token, u32_in),
                i8x32::from_array_t(token, s32_in),
            )
            .to_array();
        for k in 0..8 {
            let prod: i32 = (0..4)
                .map(|j| (u32_in[4 * k + j] as i32) * (s32_in[4 * k + j] as i32))
                .sum();
            assert_eq!(dp8[k], acc8[k].wrapping_add(prod));
        }

        let acc16: [i32; 16] = core::array::from_fn(|i| {
            if i == 0 {
                i32::MAX - 5
            } else if i == 1 {
                i32::MIN + 5
            } else {
                (i as i32) * 100
            }
        });
        let u64_in: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(7).wrapping_add(19));
        let s64_in: [i8; 64] = core::array::from_fn(|i| if i % 2 == 0 { 25 } else { -30 });
        let v_acc16 = i32x16::from_array_t(token, acc16);
        let dp16 = v_acc16
            .mul_add_u8_i8_vnni(
                token,
                u8x64::from_array_t(token, u64_in),
                i8x64::from_array_t(token, s64_in),
            )
            .to_array();
        let dps16 = v_acc16
            .mul_add_u8_i8_sat_vnni(
                token,
                u8x64::from_array_t(token, u64_in),
                i8x64::from_array_t(token, s64_in),
            )
            .to_array();
        for k in 0..16 {
            let prod: i32 = (0..4)
                .map(|j| (u64_in[4 * k + j] as i32) * (s64_in[4 * k + j] as i32))
                .sum();
            assert_eq!(dp16[k], acc16[k].wrapping_add(prod));
            assert_eq!(dps16[k], acc16[k].saturating_add(prod));
        }

        let w32_a: [i16; 32] = core::array::from_fn(|i| ((i as i16) - 16) * 1500);
        let w32_b: [i16; 32] = core::array::from_fn(|i| if i % 3 == 0 { -9 } else { 11 });
        let dw16 = v_acc16
            .mul_add_i16_vnni(
                token,
                i16x32::from_array_t(token, w32_a),
                i16x32::from_array_t(token, w32_b),
            )
            .to_array();
        let dws16 = v_acc16
            .mul_add_i16_sat_vnni(
                token,
                i16x32::from_array_t(token, w32_a),
                i16x32::from_array_t(token, w32_b),
            )
            .to_array();
        for k in 0..16 {
            let prod = (w32_a[2 * k] as i32) * (w32_b[2 * k] as i32)
                + (w32_a[2 * k + 1] as i32) * (w32_b[2 * k + 1] as i32);
            assert_eq!(dw16[k], acc16[k].wrapping_add(prod));
            assert_eq!(dws16[k], acc16[k].saturating_add(prod));
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_vnni_tests {
    use archmage::{SimdToken, X64V3Token, X64V3VnniToken, attune};

    #[attune(v3_vnni, define(f32x8))]
    fn axpy_v3_vnni_body(x: &[f32; 8], a: f32, b: f32) -> [f32; 8] {
        let base = X64V3Token::from_context();
        let vx = f32x8::load_t(base, x);
        let va = f32x8::splat_t(base, a);
        let vb = f32x8::splat_t(base, b);
        vx.mul_add_relaxed(va, vb).to_array()
    }

    #[attune(wrap)]
    fn call_axpy_v3_vnni(_token: X64V3VnniToken, x: &[f32; 8], a: f32, b: f32) -> [f32; 8] {
        axpy_v3_vnni_body(x, a, b)
    }

    fn ref_dpbusd(acc: i32, u: [u8; 4], s: [i8; 4], sat: bool) -> i32 {
        let prod: i32 = u
            .into_iter()
            .zip(s)
            .map(|(a, b)| (a as i32) * (b as i32))
            .sum();
        if sat {
            acc.saturating_add(prod)
        } else {
            acc.wrapping_add(prod)
        }
    }

    fn ref_dpwssd(acc: i32, a: [i16; 2], b: [i16; 2], sat: bool) -> i32 {
        let prod = (a[0] as i32) * (b[0] as i32) + (a[1] as i32) * (b[1] as i32);
        if sat {
            acc.saturating_add(prod)
        } else {
            acc.wrapping_add(prod)
        }
    }

    #[attune(_v3_vnni_t, define(i32x4, u8x16, i8x16, i16x8))]
    fn check_vnni_128(token: Token) {
        let acc4 = [i32::MAX - 20, i32::MIN + 20, 0, -42];
        let u16_in: [u8; 16] = core::array::from_fn(|i| (i as u8).wrapping_mul(17).wrapping_add(5));
        let s16_in: [i8; 16] = core::array::from_fn(|i| if i % 2 == 0 { 40 } else { -35 });
        let w8_a: [i16; 8] = [
            30000,
            20000,
            -30000,
            -20000,
            1234,
            -5678,
            i16::MIN,
            i16::MAX,
        ];
        let w8_b: [i16; 8] = [2, 3, 2, 3, -7, 11, -1, 1];

        let base = token.v3();
        let v_acc4 = i32x4::from_array_t(base, acc4);
        let v_u16 = u8x16::from_array_t(base, u16_in);
        let v_s16 = i8x16::from_array_t(base, s16_in);
        let v_w8a = i16x8::from_array_t(base, w8_a);
        let v_w8b = i16x8::from_array_t(base, w8_b);

        let dp4 = v_acc4.mul_add_u8_i8_vnni(token, v_u16, v_s16).to_array();
        let dps4 = v_acc4
            .mul_add_u8_i8_sat_vnni(token, v_u16, v_s16)
            .to_array();
        let dw4 = v_acc4.mul_add_i16_vnni(token, v_w8a, v_w8b).to_array();
        let dws4 = v_acc4.mul_add_i16_sat_vnni(token, v_w8a, v_w8b).to_array();

        for k in 0..4 {
            let u = [
                u16_in[4 * k],
                u16_in[4 * k + 1],
                u16_in[4 * k + 2],
                u16_in[4 * k + 3],
            ];
            let s = [
                s16_in[4 * k],
                s16_in[4 * k + 1],
                s16_in[4 * k + 2],
                s16_in[4 * k + 3],
            ];
            assert_eq!(dp4[k], ref_dpbusd(acc4[k], u, s, false));
            assert_eq!(dps4[k], ref_dpbusd(acc4[k], u, s, true));

            let wa = [w8_a[2 * k], w8_a[2 * k + 1]];
            let wb = [w8_b[2 * k], w8_b[2 * k + 1]];
            assert_eq!(dw4[k], ref_dpwssd(acc4[k], wa, wb, false));
            assert_eq!(dws4[k], ref_dpwssd(acc4[k], wa, wb, true));
        }
    }

    #[attune(_v3_vnni_t, define(i32x8, u8x32, i8x32, i16x16))]
    fn check_vnni_256(token: Token) {
        let acc8 = [
            i32::MAX - 10,
            i32::MIN + 10,
            100,
            -100,
            0,
            777,
            -888,
            123456,
        ];
        let u32_in: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(13).wrapping_add(9));
        let s32_in: [i8; 32] = core::array::from_fn(|i| ((i as i8) * 3).wrapping_sub(40));
        let w16_a: [i16; 16] = core::array::from_fn(|i| (i as i16 - 8) * 4000);
        let w16_b: [i16; 16] = core::array::from_fn(|i| if i % 3 == 0 { -5 } else { 7 });

        let base = token.v3();
        let v_acc8 = i32x8::from_array_t(base, acc8);
        let v_u32 = u8x32::from_array_t(base, u32_in);
        let v_s32 = i8x32::from_array_t(base, s32_in);
        let v_w16a = i16x16::from_array_t(base, w16_a);
        let v_w16b = i16x16::from_array_t(base, w16_b);

        let dp8 = v_acc8.mul_add_u8_i8_vnni(token, v_u32, v_s32).to_array();
        let dps8 = v_acc8
            .mul_add_u8_i8_sat_vnni(token, v_u32, v_s32)
            .to_array();
        let dw8 = v_acc8.mul_add_i16_vnni(token, v_w16a, v_w16b).to_array();
        let dws8 = v_acc8
            .mul_add_i16_sat_vnni(token, v_w16a, v_w16b)
            .to_array();

        for k in 0..8 {
            let u = [
                u32_in[4 * k],
                u32_in[4 * k + 1],
                u32_in[4 * k + 2],
                u32_in[4 * k + 3],
            ];
            let s = [
                s32_in[4 * k],
                s32_in[4 * k + 1],
                s32_in[4 * k + 2],
                s32_in[4 * k + 3],
            ];
            assert_eq!(dp8[k], ref_dpbusd(acc8[k], u, s, false));
            assert_eq!(dps8[k], ref_dpbusd(acc8[k], u, s, true));

            let wa = [w16_a[2 * k], w16_a[2 * k + 1]];
            let wb = [w16_b[2 * k], w16_b[2 * k + 1]];
            assert_eq!(dw8[k], ref_dpwssd(acc8[k], wa, wb, false));
            assert_eq!(dws8[k], ref_dpwssd(acc8[k], wa, wb, true));
        }
    }

    #[test]
    fn vnni_dot_products_match_scalar_reference() {
        #[cfg(target_feature = "avxvnni")]
        assert!(
            X64V3VnniToken::summon().is_some(),
            "avxvnni enabled at compile time must summon X64V3VnniToken"
        );
        let Some(token) = X64V3VnniToken::summon() else {
            return;
        };
        let x = [1.0, -2.0, 3.5, 4.0, -0.5, 6.0, 7.25, -8.0];
        let got = call_axpy_v3_vnni(token, &x, 2.0, 1.0);
        for (g, e) in got.into_iter().zip(x.map(|v| v * 2.0 + 1.0)) {
            let diff = if g >= e { g - e } else { e - g };
            assert!(diff <= 1e-6);
        }
        check_vnni_128_v3_vnni_t(token);
        check_vnni_256_v3_vnni_t(token);
    }
}

#[cfg(target_arch = "aarch64")]
mod arm_v2_tests {
    use archmage::{Arm64V2Token, Arm64V3Token, NeonToken, SimdToken, attune};

    #[attune(arm_v3, define(f32x4))]
    fn axpy_arm_v3_body(x: &[f32; 4], a: f32, b: f32) -> [f32; 4] {
        let base = NeonToken::from_context();
        f32x4::load_t(base, x)
            .mul_add_relaxed(f32x4::splat_t(base, a), f32x4::splat_t(base, b))
            .to_array()
    }

    #[attune(wrap)]
    fn call_axpy_arm_v3(_token: Arm64V3Token, x: &[f32; 4], a: f32, b: f32) -> [f32; 4] {
        axpy_arm_v3_body(x, a, b)
    }

    fn ref_qrdml(acc: i16, a: i16, b: i16, sub: bool) -> i16 {
        let prod = 2i64 * (a as i64) * (b as i64);
        let base = (acc as i64) << 16;
        let combined = if sub { base - prod } else { base + prod };
        let rounded = (combined + (1i64 << 15)) >> 16;
        rounded.clamp(i16::MIN as i64, i16::MAX as i64) as i16
    }

    #[attune(_arm_v2_t, define(i32x4, u32x4, i8x16, u8x16, i16x8))]
    fn check_arm_v2(token: Token) {
        let base = token.neon();
        let s_acc = i32x4::from_array_t(base, [10, -20, 30, -40]);
        let sa: [i8; 16] = core::array::from_fn(|i| (i as i8) * 5 - 30);
        let sb: [i8; 16] = core::array::from_fn(|i| if i % 2 == 0 { 7 } else { -9 });
        let s_got = s_acc
            .dot_i8(
                token,
                i8x16::from_array_t(base, sa),
                i8x16::from_array_t(base, sb),
            )
            .to_array();
        for k in 0..4 {
            let dot: i32 = (0..4)
                .map(|j| (sa[4 * k + j] as i32) * (sb[4 * k + j] as i32))
                .sum();
            assert_eq!(s_got[k], [10, -20, 30, -40][k] + dot);
        }

        let u_acc = u32x4::from_array_t(base, [100, 200, 300, 400]);
        let ua: [u8; 16] = core::array::from_fn(|i| (i as u8) * 15 + 3);
        let ub: [u8; 16] = core::array::from_fn(|i| 255 - (i as u8) * 11);
        let u_got = u_acc
            .dot_u8(
                token,
                u8x16::from_array_t(base, ua),
                u8x16::from_array_t(base, ub),
            )
            .to_array();
        for k in 0..4 {
            let dot: u32 = (0..4)
                .map(|j| (ua[4 * k + j] as u32) * (ub[4 * k + j] as u32))
                .sum();
            assert_eq!(u_got[k], [100, 200, 300, 400][k] + dot);
        }

        let q_acc_arr: [i16; 8] = [
            0,
            1000,
            -1000,
            i16::MAX - 5,
            i16::MIN + 5,
            16384,
            -16384,
            42,
        ];
        let qa_arr: [i16; 8] = [16384, -16384, 8192, 30000, 30000, i16::MIN, i16::MIN, 12345];
        let qb_arr: [i16; 8] = [16384, 8192, -8192, 30000, -30000, i16::MIN, i16::MAX, -6789];
        let q_acc = i16x8::from_array_t(base, q_acc_arr);
        let qa = i16x8::from_array_t(base, qa_arr);
        let qb = i16x8::from_array_t(base, qb_arr);
        let got_add = q_acc.qrdmlah(token, qa, qb).to_array();
        let got_sub = q_acc.qrdmlsh(token, qa, qb).to_array();
        for k in 0..8 {
            assert_eq!(
                got_add[k],
                ref_qrdml(q_acc_arr[k], qa_arr[k], qb_arr[k], false)
            );
            assert_eq!(
                got_sub[k],
                ref_qrdml(q_acc_arr[k], qa_arr[k], qb_arr[k], true)
            );
        }
    }

    #[attune(_arm_v2_t, define(i32x8, u32x8, i8x32, u8x32, i16x16))]
    fn check_arm_v2_256(token: Token) {
        use magetypes::simd::MagetypesToken;
        let base = token.magetypes();
        let acc8: [i32; 8] = [10, -20, 30, -40, 50, -60, 70, -80];
        let sa: [i8; 32] = core::array::from_fn(|i| (i as i8) * 3 - 45);
        let sb: [i8; 32] = core::array::from_fn(|i| if i % 2 == 0 { 11 } else { -7 });
        let got_s8 = i32x8::from_array_t(base, acc8)
            .dot_i8(
                token,
                i8x32::from_array_t(base, sa),
                i8x32::from_array_t(base, sb),
            )
            .to_array();
        for k in 0..8 {
            let dot: i32 = (0..4)
                .map(|j| (sa[4 * k + j] as i32) * (sb[4 * k + j] as i32))
                .sum();
            assert_eq!(got_s8[k], acc8[k] + dot);
        }

        let uacc8: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let ua: [u8; 32] = core::array::from_fn(|i| (i as u8) * 7 + 5);
        let ub: [u8; 32] = core::array::from_fn(|i| 240 - (i as u8) * 5);
        let got_u8 = u32x8::from_array_t(base, uacc8)
            .dot_u8(
                token,
                u8x32::from_array_t(base, ua),
                u8x32::from_array_t(base, ub),
            )
            .to_array();
        for k in 0..8 {
            let dot: u32 = (0..4)
                .map(|j| (ua[4 * k + j] as u32) * (ub[4 * k + j] as u32))
                .sum();
            assert_eq!(got_u8[k], uacc8[k] + dot);
        }

        let q_acc16: [i16; 16] = core::array::from_fn(|i| ((i as i16) - 8) * 1000);
        let qa16: [i16; 16] = core::array::from_fn(|i| ((i as i16) - 7) * 4000);
        let qb16: [i16; 16] = core::array::from_fn(|i| if i % 2 == 0 { 8192 } else { -8192 });
        let v_qacc16 = i16x16::from_array_t(base, q_acc16);
        let v_qa16 = i16x16::from_array_t(base, qa16);
        let v_qb16 = i16x16::from_array_t(base, qb16);
        let add16 = v_qacc16.qrdmlah(token, v_qa16, v_qb16).to_array();
        let sub16 = v_qacc16.qrdmlsh(token, v_qa16, v_qb16).to_array();
        for k in 0..16 {
            assert_eq!(add16[k], ref_qrdml(q_acc16[k], qa16[k], qb16[k], false));
            assert_eq!(sub16[k], ref_qrdml(q_acc16[k], qa16[k], qb16[k], true));
        }
    }

    #[attune(_arm_v3_t, define(f32x4, f32x8, i32x4, u16x8))]
    fn check_arm_v3_fhm(token: Token) {
        use magetypes::simd::MagetypesToken;
        let base = token.magetypes();
        let af32: [f32; 8] = [1.5, -2.25, 0.5, 3.75, -0.125, 4.0, -1.0, 2.5];
        let bf32: [f32; 8] = [2.0, 0.5, -4.0, 1.25, 8.0, -0.75, 3.5, -2.0];
        let acc_lo = [10.0f32, -5.0, 1.25, 0.0];
        let acc_hi = [-3.0f32, 7.5, 2.0, -1.5];

        let encode8 = |vals: [f32; 8]| -> [u16; 8] {
            let lo = f32x4::from_array_t(base, [vals[0], vals[1], vals[2], vals[3]])
                .to_f16()
                .to_array();
            let hi = f32x4::from_array_t(base, [vals[4], vals[5], vals[6], vals[7]])
                .to_f16()
                .to_array();
            [
                lo[0] as u16,
                lo[1] as u16,
                lo[2] as u16,
                lo[3] as u16,
                hi[0] as u16,
                hi[1] as u16,
                hi[2] as u16,
                hi[3] as u16,
            ]
        };

        let ah = u16x8::from_array_t(base, encode8(af32));
        let bh = u16x8::from_array_t(base, encode8(bf32));
        let v_lo = f32x4::from_array_t(base, acc_lo);
        let v_hi = f32x4::from_array_t(base, acc_hi);

        let add_lo = v_lo.fmlal_low_f16(token, ah, bh).to_array();
        let add_hi = v_hi.fmlal_high_f16(token, ah, bh).to_array();
        let sub_lo = v_lo.fmlsl_low_f16(token, ah, bh).to_array();
        let sub_hi = v_hi.fmlsl_high_f16(token, ah, bh).to_array();

        for i in 0..4 {
            assert_eq!(add_lo[i], acc_lo[i] + af32[i] * bf32[i]);
            assert_eq!(add_hi[i], acc_hi[i] + af32[i + 4] * bf32[i + 4]);
            assert_eq!(sub_lo[i], acc_lo[i] - af32[i] * bf32[i]);
            assert_eq!(sub_hi[i], acc_hi[i] - af32[i + 4] * bf32[i + 4]);
        }

        let acc8 = [
            acc_lo[0], acc_lo[1], acc_lo[2], acc_lo[3], acc_hi[0], acc_hi[1], acc_hi[2], acc_hi[3],
        ];
        let v_acc8 = f32x8::from_array_t(base, acc8);
        let add8 = v_acc8.fmlal_f16(token, ah, bh).to_array();
        let sub8 = v_acc8.fmlsl_f16(token, ah, bh).to_array();
        for i in 0..8 {
            assert_eq!(add8[i], acc8[i] + af32[i] * bf32[i]);
            assert_eq!(sub8[i], acc8[i] - af32[i] * bf32[i]);
        }

        let _ = i32x4::splat_t(base, 0);
    }

    #[test]
    fn arm_v2_dotprod_and_rdm_match_scalar_reference() {
        #[cfg(all(target_feature = "dotprod", target_feature = "rdm"))]
        assert!(
            Arm64V2Token::summon().is_some(),
            "dotprod+rdm enabled at compile time must summon Arm64V2Token"
        );
        if let Some(token) = Arm64V2Token::summon() {
            check_arm_v2_arm_v2_t(token);
            check_arm_v2_256_arm_v2_t(token);
        }
        if let Some(v3) = Arm64V3Token::summon() {
            let got = call_axpy_arm_v3(v3, &[1.0, 2.0, -3.0, 4.0], 2.0, 0.5);
            assert_eq!(got, [2.5, 4.5, -5.5, 8.5]);
            check_arm_v2_arm_v2_t(v3.arm_v2());
            check_arm_v2_256_arm_v2_t(v3.arm_v2());
            check_arm_v3_fhm_arm_v3_t(v3);
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm_relaxed_tests {
    use archmage::{SimdToken, Wasm128RelaxedToken, attune};

    #[attune(wasm128_relaxed, define(f32x4))]
    fn axpy_wasm_relaxed_body(x: &[f32; 4], a: f32, b: f32) -> [f32; 4] {
        let base = archmage::Wasm128Token::from_context();
        f32x4::load_t(base, x)
            .mul_add_relaxed(f32x4::splat_t(base, a), f32x4::splat_t(base, b))
            .to_array()
    }

    #[attune(wrap)]
    fn call_axpy_wasm_relaxed(
        _token: Wasm128RelaxedToken,
        x: &[f32; 4],
        a: f32,
        b: f32,
    ) -> [f32; 4] {
        axpy_wasm_relaxed_body(x, a, b)
    }

    #[attune(
        _wasm128_relaxed_t,
        define(
            f32x4, f32x8, f64x2, f64x4, i32x4, i32x8, u32x4, u32x8, i16x8, i16x16, i8x16, i8x32
        )
    )]
    fn check_wasm_relaxed(token: Token) {
        use magetypes::simd::MagetypesToken;
        let base = token.magetypes();
        let a = f32x4::from_array_t(base, [1.5, -2.5, 3.0, 10.0]);
        let b = f32x4::from_array_t(base, [-1.0, -3.0, 4.0, 10.0]);
        assert_eq!(a.min_relaxed(token, b).to_array(), [-1.0, -3.0, 3.0, 10.0]);
        assert_eq!(a.max_relaxed(token, b).to_array(), [1.5, -2.5, 4.0, 10.0]);
        assert_eq!(
            f32x4::from_array_t(base, [1.9, -2.9, 100.25, 0.0])
                .to_i32_relaxed(token)
                .to_array(),
            [1, -2, 100, 0]
        );
        assert_eq!(
            f32x4::from_array_t(base, [0.0, 3.75, 2_500_000_000.0, 4_294_967_040.0])
                .to_u32_relaxed(token)
                .to_array(),
            [0, 3, 2_500_000_000, 4_294_967_040]
        );

        let a8 = f32x8::from_array_t(base, [1.5, -2.5, 3.0, 10.0, -5.0, 6.0, 0.25, -0.5]);
        let b8 = f32x8::from_array_t(base, [-1.0, -3.0, 4.0, 10.0, -4.0, 2.0, 0.5, -1.5]);
        assert_eq!(
            a8.min_relaxed(token, b8).to_array(),
            [-1.0, -3.0, 3.0, 10.0, -5.0, 2.0, 0.25, -1.5]
        );
        assert_eq!(
            a8.max_relaxed(token, b8).to_array(),
            [1.5, -2.5, 4.0, 10.0, -4.0, 6.0, 0.5, -0.5]
        );
        assert_eq!(
            a8.to_i32_relaxed(token).to_array(),
            [1, -2, 3, 10, -5, 6, 0, 0]
        );
        assert_eq!(
            f32x8::from_array_t(
                base,
                [
                    0.0,
                    1.5,
                    2.5,
                    3.5,
                    100.0,
                    2_147_483_648.0,
                    3_000_000_000.0,
                    4_294_967_040.0
                ]
            )
            .to_u32_relaxed(token)
            .to_array(),
            [0, 1, 2, 3, 100, 2_147_483_648, 3_000_000_000, 4_294_967_040]
        );

        let da = f64x2::from_array_t(base, [2.25, -9.5]);
        let db = f64x2::from_array_t(base, [-4.0, -3.5]);
        assert_eq!(da.min_relaxed(token, db).to_array(), [-4.0, -9.5]);
        assert_eq!(da.max_relaxed(token, db).to_array(), [2.25, -3.5]);

        let da4 = f64x4::from_array_t(base, [2.25, -9.5, 1.0, -2.0]);
        let db4 = f64x4::from_array_t(base, [-4.0, -3.5, 5.0, -8.0]);
        assert_eq!(
            da4.min_relaxed(token, db4).to_array(),
            [-4.0, -9.5, 1.0, -8.0]
        );
        assert_eq!(
            da4.max_relaxed(token, db4).to_array(),
            [2.25, -3.5, 5.0, -2.0]
        );

        let qa = i16x8::from_array_t(base, [16384, -16384, 8192, -8192, 0, 32767, -32767, 4096]);
        let qb = i16x8::from_array_t(
            base,
            [16384, 16384, -8192, -8192, 12345, 16384, 16384, 8192],
        );
        let qgot = qa.mul_hrs_q15_relaxed(token, qb).to_array();
        assert_eq!(qgot, [8192, -8192, -2048, 2048, 0, 16384, -16383, 1024]);

        let qa16 = i16x16::from_array_t(base, core::array::from_fn(|i| ((i as i16) - 8) * 2048));
        let qb16 = i16x16::splat_t(base, 16384);
        let qgot16 = qa16.mul_hrs_q15_relaxed(token, qb16).to_array();
        for (i, &got) in qgot16.iter().enumerate() {
            let a_i = ((i as i32) - 8) * 2048;
            assert_eq!(got, ((a_i * 16384 + 0x4000) >> 15) as i16);
        }

        let acc = i32x4::from_array_t(base, [10, 20, 30, 40]);
        let ia: [i8; 16] = core::array::from_fn(|i| (i as i8) * 3 - 20);
        let ib: [i8; 16] = core::array::from_fn(|i| ((i as i8) * 7) & 0x7f);
        let got16 = i16x8::dot_i8_i7_relaxed(
            token,
            i8x16::from_array_t(base, ia),
            i8x16::from_array_t(base, ib),
        )
        .to_array();
        for k in 0..8 {
            let pair = (ia[2 * k] as i16) * (ib[2 * k] as i16)
                + (ia[2 * k + 1] as i16) * (ib[2 * k + 1] as i16);
            assert_eq!(got16[k], pair);
        }
        let got = acc
            .dot_i8_i7_add_relaxed(
                token,
                i8x16::from_array_t(base, ia),
                i8x16::from_array_t(base, ib),
            )
            .to_array();
        for k in 0..4 {
            let dot: i32 = (0..4)
                .map(|j| (ia[4 * k + j] as i32) * (ib[4 * k + j] as i32))
                .sum();
            assert_eq!(got[k], [10, 20, 30, 40][k] + dot);
        }

        let acc8 = i32x8::from_array_t(base, [1, 2, 3, 4, 5, 6, 7, 8]);
        let ia32: [i8; 32] = core::array::from_fn(|i| (i as i8).wrapping_mul(2).wrapping_sub(30));
        let ib32: [i8; 32] = core::array::from_fn(|i| ((i as i8).wrapping_mul(5)) & 0x7f);
        let got8 = acc8
            .dot_i8_i7_add_relaxed(
                token,
                i8x32::from_array_t(base, ia32),
                i8x32::from_array_t(base, ib32),
            )
            .to_array();
        for k in 0..8 {
            let dot: i32 = (0..4)
                .map(|j| (ia32[4 * k + j] as i32) * (ib32[4 * k + j] as i32))
                .sum();
            assert_eq!(got8[k], (k as i32 + 1) + dot);
        }

        let _ = (u32x4::splat_t(base, 0), u32x8::splat_t(base, 0));
    }

    #[test]
    fn wasm_relaxed_ops_match_scalar_reference() {
        #[cfg(all(target_feature = "simd128", target_feature = "relaxed-simd"))]
        {
            assert_ne!(Wasm128RelaxedToken::compiled_with(), Some(false));
            let token = Wasm128RelaxedToken::summon().expect(
                "simd128+relaxed-simd enabled at compile time must summon Wasm128RelaxedToken",
            );
            let got = call_axpy_wasm_relaxed(token, &[1.0, -2.0, 3.0, 4.0], 2.0, 0.5);
            assert_eq!(got, [2.5, -3.5, 6.5, 8.5]);
            check_wasm_relaxed_wasm128_relaxed_t(token);
        }
        #[cfg(not(all(target_feature = "simd128", target_feature = "relaxed-simd")))]
        {
            assert_eq!(Wasm128RelaxedToken::compiled_with(), Some(false));
            assert!(
                Wasm128RelaxedToken::summon().is_none(),
                "WASM without +simd128,+relaxed-simd must not summon Wasm128RelaxedToken"
            );
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm_relaxed_fma_tests {
    use archmage::{SimdToken, Wasm128RelaxedFmaToken, attune};

    /// Single-rounding witness across 128-bit and 256-bit float/int/convert/cross-width ops.
    /// `(1 + 2^-12)^2 - (1 + 2^-11)` is `2^-24` with one rounding and `0.0` with two;
    /// `(1 + 2^-27)^2 - (1 + 2^-26)` is `2^-54` with one rounding and `0.0` with two.
    #[attune(
        _*_t,
        +wasm128_relaxed_fma,
        define(f32x4, f32x8, f64x2, f64x4, i32x4)
    )]
    fn verify_single_rounding_narrow(token: Token, e32: f32, e64: f64) -> (f32, f64, i32) {
        let x4 = f32x4::from_array_t(token, [1.0 + e32, -(1.0 + e32), 1.0 + e32, -(1.0 + e32)]);
        let b4 = f32x4::from_array_t(token, [1.0 + e32, -(1.0 + e32), -(1.0 + e32), 1.0 + e32]);
        let pos_c32 = 1.0 + 2.0 * e32;
        let c4_add = f32x4::from_array_t(token, [-pos_c32, -pos_c32, pos_c32, pos_c32]);
        let c4_sub = f32x4::from_array_t(token, [pos_c32, pos_c32, -pos_c32, -pos_c32]);
        let expect32 = [e32 * e32, e32 * e32, -(e32 * e32), -(e32 * e32)];
        assert_eq!(x4.mul_add_portable(b4, c4_add).to_array(), expect32);
        assert_eq!(x4.mul_sub_portable(b4, c4_sub).to_array(), expect32);

        let x8 = f32x8::from_halves_t(token, x4, x4);
        let b8 = f32x8::from_halves_t(token, b4, b4);
        let c8_add = f32x8::from_halves_t(token, c4_add, c4_add);
        let c8_sub = f32x8::from_halves_t(token, c4_sub, c4_sub);
        let expect32_8 = [
            e32 * e32,
            e32 * e32,
            -(e32 * e32),
            -(e32 * e32),
            e32 * e32,
            e32 * e32,
            -(e32 * e32),
            -(e32 * e32),
        ];
        assert_eq!(x8.mul_add_portable(b8, c8_add).to_array(), expect32_8);
        assert_eq!(x8.mul_sub_portable(b8, c8_sub).to_array(), expect32_8);

        let x2 = f64x2::from_array_t(token, [1.0 + e64, -(1.0 + e64)]);
        let b2 = f64x2::from_array_t(token, [1.0 + e64, 1.0 + e64]);
        let pos_c64 = 1.0 + 2.0 * e64;
        let c2_add = f64x2::from_array_t(token, [-pos_c64, pos_c64]);
        let c2_sub = f64x2::from_array_t(token, [pos_c64, -pos_c64]);
        let expect64 = [e64 * e64, -(e64 * e64)];
        assert_eq!(x2.mul_add_portable(b2, c2_add).to_array(), expect64);
        assert_eq!(x2.mul_sub_portable(b2, c2_sub).to_array(), expect64);

        let x4d = f64x4::from_array_t(token, [1.0 + e64, -(1.0 + e64), -(1.0 + e64), 1.0 + e64]);
        let b4d = f64x4::from_array_t(token, [1.0 + e64, -(1.0 + e64), 1.0 + e64, -(1.0 + e64)]);
        let c4d_add = f64x4::from_array_t(token, [-pos_c64, -pos_c64, pos_c64, pos_c64]);
        let c4d_sub = f64x4::from_array_t(token, [pos_c64, pos_c64, -pos_c64, -pos_c64]);
        let expect64_4 = [e64 * e64, e64 * e64, -(e64 * e64), -(e64 * e64)];
        assert_eq!(x4d.mul_add_portable(b4d, c4d_add).to_array(), expect64_4);
        assert_eq!(x4d.mul_sub_portable(b4d, c4d_sub).to_array(), expect64_4);

        let conv = f32x4::from_array_t(token, [1.25, 2.75, -1.25, -2.75]).to_i32_round();
        let sum = (conv + i32x4::splat_t(token, 10)).reduce_add();
        (
            x4.mul_add_portable(b4, c4_add).to_array()[0],
            x2.mul_add_portable(b2, c2_add).to_array()[0],
            sum,
        )
    }

    #[cfg(feature = "w512")]
    #[archmage::magetypes(
        wasm128_relaxed_fma,
        wasm128,
        scalar,
        define(f32x8, f32x16, f64x8, i32x16)
    )]
    fn verify_single_rounding_w512(token: Token, e32: f32, e64: f64) {
        let pos_c32 = 1.0 + 2.0 * e32;
        let x16 = f32x16::splat_t(token, 1.0 + e32);
        let c16_add = f32x16::splat_t(token, -pos_c32);
        let c16_sub = f32x16::splat_t(token, pos_c32);
        assert_eq!(
            x16.mul_add_portable(x16, c16_add).to_array(),
            [e32 * e32; 16]
        );
        assert_eq!(
            x16.mul_sub_portable(x16, c16_sub).to_array(),
            [e32 * e32; 16]
        );
        assert_eq!(x16.mul_add_relaxed(x16, c16_add).to_array().len(), 16);

        let halves = f32x16::from_halves_t(
            token,
            f32x8::splat_t(token, 2.0),
            f32x8::splat_t(token, -2.0),
        );
        let rounded: i32 = (halves.to_i32_round() + i32x16::splat_t(token, 1)).reduce_add();
        assert_eq!(rounded, 16);

        let pos_c64 = 1.0 + 2.0 * e64;
        let x8d = f64x8::splat_t(token, 1.0 + e64);
        let c8d_add = f64x8::splat_t(token, -pos_c64);
        let c8d_sub = f64x8::splat_t(token, pos_c64);
        assert_eq!(
            x8d.mul_add_portable(x8d, c8d_add).to_array(),
            [e64 * e64; 8]
        );
        assert_eq!(
            x8d.mul_sub_portable(x8d, c8d_sub).to_array(),
            [e64 * e64; 8]
        );
    }

    #[attune(_wasm128_relaxed_fma_t, define(f32x4, f64x2))]
    fn check_extractions_and_fast_fma(token: Token, e32: f32, e64: f64) {
        let relaxed: archmage::Wasm128RelaxedToken = token.wasm128_relaxed();
        let base: archmage::Wasm128Token = token.wasm128();
        let _ = (relaxed, base);

        // On `Wasm128RelaxedFmaToken`, even `.mul_add_relaxed()` / `.mul_sub_relaxed()`
        // run on a runtime-verified fusing engine and round once on all lanes.
        let x4 = f32x4::splat_t(token, 1.0 + e32);
        let c4 = f32x4::splat_t(token, -(1.0 + 2.0 * e32));
        assert_eq!(x4.mul_add_relaxed(x4, c4).to_array(), [e32 * e32; 4]);
        assert_eq!(
            x4.mul_sub_relaxed(x4, f32x4::splat_t(token, 1.0 + 2.0 * e32))
                .to_array(),
            [e32 * e32; 4]
        );

        let x2 = f64x2::splat_t(token, 1.0 + e64);
        let c2 = f64x2::splat_t(token, -(1.0 + 2.0 * e64));
        assert_eq!(x2.mul_add_relaxed(x2, c2).to_array(), [e64 * e64; 2]);
        assert_eq!(
            x2.mul_sub_relaxed(x2, f64x2::splat_t(token, 1.0 + 2.0 * e64))
                .to_array(),
            [e64 * e64; 2]
        );
    }

    #[test]
    fn wasm_relaxed_fma_backend_and_dispatch() {
        #[cfg(all(target_feature = "simd128", target_feature = "relaxed-simd"))]
        assert_eq!(Wasm128RelaxedFmaToken::compiled_with(), None);
        #[cfg(not(all(target_feature = "simd128", target_feature = "relaxed-simd")))]
        assert_eq!(Wasm128RelaxedFmaToken::compiled_with(), Some(false));

        #[cfg(not(all(target_feature = "simd128", target_feature = "relaxed-simd")))]
        assert!(
            Wasm128RelaxedFmaToken::summon().is_none(),
            "WASM without +simd128,+relaxed-simd must not summon Wasm128RelaxedFmaToken"
        );

        let e32 = core::hint::black_box(1.0f32 / 4096.0);
        let e64 = core::hint::black_box(1.0f64 / 134_217_728.0);

        let (r32, r64, sum) = archmage::attuned!(
            verify_single_rounding_narrow(e32, e64),
            [_wasm128_relaxed_fma_t, _wasm128_t, _scalar_t]
        );
        assert_eq!(r32, e32 * e32);
        assert_eq!(r64, e64 * e64);
        assert_eq!(sum, 40);

        #[cfg(feature = "w512")]
        archmage::incant!(
            verify_single_rounding_w512(e32, e64),
            [wasm128_relaxed_fma, wasm128, scalar]
        );

        #[cfg(all(target_feature = "simd128", target_feature = "relaxed-simd"))]
        if let Some(token) = Wasm128RelaxedFmaToken::summon() {
            check_extractions_and_fast_fma_wasm128_relaxed_fma_t(token, e32, e64);
        }

        #[cfg(feature = "std")]
        {
            if let Ok(expect) = std::env::var("ARCHMAGE_EXPECT_RELAXED_FUSION") {
                let expected = match expect.as_str() {
                    "yes" => true,
                    "no" => false,
                    other => panic!("unexpected ARCHMAGE_EXPECT_RELAXED_FUSION={other:?}"),
                };
                assert_eq!(
                    Wasm128RelaxedFmaToken::summon().is_some(),
                    expected,
                    "Wasm128RelaxedFmaToken::summon() disagreed with ARCHMAGE_EXPECT_RELAXED_FUSION={expect}"
                );
            }

            // Exercise every reachable token permutation (including disabling
            // `Wasm128RelaxedFmaToken` on fusing engines so dispatch falls back to
            // `Wasm128Token` while preserving bit-exact single-rounding results).
            let had_fma = Wasm128RelaxedFmaToken::summon().is_some();
            let mut saw_enabled = false;
            let mut saw_disabled = false;
            let report = archmage::testing::for_each_token_permutation(
                archmage::testing::CompileTimePolicy::Warn,
                |_| {
                    if Wasm128RelaxedFmaToken::summon().is_some() {
                        saw_enabled = true;
                    } else {
                        saw_disabled = true;
                    }
                    let (p32, p64, psum) = archmage::attuned!(
                        verify_single_rounding_narrow(e32, e64),
                        [_wasm128_relaxed_fma_t, _wasm128_t, _scalar_t]
                    );
                    assert_eq!(p32, e32 * e32);
                    assert_eq!(p64, e64 * e64);
                    assert_eq!(psum, 40);
                    #[cfg(feature = "w512")]
                    archmage::incant!(
                        verify_single_rounding_w512(e32, e64),
                        [wasm128_relaxed_fma, wasm128, scalar]
                    );
                },
            );
            if had_fma {
                assert!(report.permutations_run >= 2);
                assert!(saw_enabled && saw_disabled);
            }
        }
    }
}

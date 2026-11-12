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
    assert!(archmage::Wasm128RelaxedToken::summon().is_none());
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

    #[test]
    fn arm_v2_dotprod_and_rdm_match_scalar_reference() {
        #[cfg(all(target_feature = "dotprod", target_feature = "rdm"))]
        assert!(
            Arm64V2Token::summon().is_some(),
            "dotprod+rdm enabled at compile time must summon Arm64V2Token"
        );
        if let Some(token) = Arm64V2Token::summon() {
            check_arm_v2_arm_v2_t(token);
        }
        if let Some(v3) = Arm64V3Token::summon() {
            let got = call_axpy_arm_v3(v3, &[1.0, 2.0, -3.0, 4.0], 2.0, 0.5);
            assert_eq!(got, [2.5, 4.5, -5.5, 8.5]);
            check_arm_v2_arm_v2_t(v3.arm_v2());
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm_relaxed_tests {
    use archmage::{SimdToken, Wasm128RelaxedToken, Wasm128Token, attune};

    #[attune(wasm128_relaxed, define(f32x4))]
    fn axpy_wasm_relaxed_body(x: &[f32; 4], a: f32, b: f32) -> [f32; 4] {
        let base = Wasm128Token::from_context();
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

    #[attune(_wasm128_relaxed_t, define(f32x4, f64x2, i32x4, i8x16))]
    fn check_wasm_relaxed(token: Token) {
        let base = token.wasm128();
        let a = f32x4::from_array_t(base, [1.5, -2.5, 3.0, 10.0]);
        let b = f32x4::from_array_t(base, [-1.0, -3.0, 4.0, 10.0]);
        assert_eq!(a.min_relaxed(token, b).to_array(), [-1.0, -3.0, 3.0, 10.0]);
        assert_eq!(a.max_relaxed(token, b).to_array(), [1.5, -2.5, 4.0, 10.0]);

        let da = f64x2::from_array_t(base, [2.25, -9.5]);
        let db = f64x2::from_array_t(base, [-4.0, -3.5]);
        assert_eq!(da.min_relaxed(token, db).to_array(), [-4.0, -9.5]);
        assert_eq!(da.max_relaxed(token, db).to_array(), [2.25, -3.5]);

        let acc = i32x4::from_array_t(base, [10, 20, 30, 40]);
        let ia: [i8; 16] = core::array::from_fn(|i| (i as i8) * 3 - 20);
        let ib: [i8; 16] = core::array::from_fn(|i| ((i as i8) * 7) & 0x7f);
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
    }

    #[test]
    fn wasm_relaxed_ops_match_scalar_reference() {
        #[cfg(target_feature = "relaxed-simd")]
        assert!(
            Wasm128RelaxedToken::summon().is_some(),
            "relaxed-simd enabled at compile time must summon Wasm128RelaxedToken"
        );
        #[cfg(not(target_feature = "relaxed-simd"))]
        assert!(
            Wasm128RelaxedToken::summon().is_none(),
            "strict WASM without +relaxed-simd must not summon Wasm128RelaxedToken"
        );
        if let Some(token) = Wasm128RelaxedToken::summon() {
            let got = call_axpy_wasm_relaxed(token, &[1.0, -2.0, 3.0, 4.0], 2.0, 0.5);
            assert_eq!(got, [2.5, -3.5, 6.5, 8.5]);
            check_wasm_relaxed_wasm128_relaxed_t(token);
        }
    }
}

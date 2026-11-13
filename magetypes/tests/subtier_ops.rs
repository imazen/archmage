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

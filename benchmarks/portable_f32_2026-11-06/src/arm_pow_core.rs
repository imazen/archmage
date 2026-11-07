//! Experimental ARM fast pow arithmetic with CORE-MATH Rust edge fallback.
//! Derived from zenmetrics-math glibc.rs at097b5e4de9f8e5322b0f0abd4efb2b7055e05931.
//! Rust adaptation copyright Imazen, licensed MIT OR Apache-2.0 here.
//! Algorithm/tables copyright (c)2017-2026 Arm Limited; MIT license.
//! Sources: https://github.com/ARM-software/optimized-routines/tree/master/math
//! No host-libm call remains; this does not promise glibc compatibility.
//! Requires nearest-even and gradual underflow. Separate mul/add is deliberate.

const OFF: u32 = 0x3f330000;

/// `__powf_log2_data.tab` — (invc, logc) per interval center; POWF_SCALE=1
/// (TOINT_INTRINSICS is 0 on this target).
const LOG2_TAB: [(f64, f64); 16] = [
    (
        f64::from_bits(0x3ff661ec79f8f3be),
        f64::from_bits(0xbfdefec65b963019),
    ),
    (
        f64::from_bits(0x3ff571ed4aaf883d),
        f64::from_bits(0xbfdb0b6832d4fca4),
    ),
    (
        f64::from_bits(0x3ff49539f0f010b0),
        f64::from_bits(0xbfd7418b0a1fb77b),
    ),
    (
        f64::from_bits(0x3ff3c995b0b80385),
        f64::from_bits(0xbfd39de91a6dcf7b),
    ),
    (
        f64::from_bits(0x3ff30d190c8864a5),
        f64::from_bits(0xbfd01d9bf3f2b631),
    ),
    (
        f64::from_bits(0x3ff25e227b0b8ea0),
        f64::from_bits(0xbfc97c1d1b3b7af0),
    ),
    (
        f64::from_bits(0x3ff1bb4a4a1a343f),
        f64::from_bits(0xbfc2f9e393af3c9f),
    ),
    (
        f64::from_bits(0x3ff12358f08ae5ba),
        f64::from_bits(0xbfb960cbbf788d5c),
    ),
    (
        f64::from_bits(0x3ff0953f419900a7),
        f64::from_bits(0xbfaa6f9db6475fce),
    ),
    (
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0x0000000000000000),
    ),
    (
        f64::from_bits(0x3fee608cfd9a47ac),
        f64::from_bits(0x3fb338ca9f24f53d),
    ),
    (
        f64::from_bits(0x3feca4b31f026aa0),
        f64::from_bits(0x3fc476a9543891ba),
    ),
    (
        f64::from_bits(0x3feb2036576afce6),
        f64::from_bits(0x3fce840b4ac4e4d2),
    ),
    (
        f64::from_bits(0x3fe9c2d163a1aa2d),
        f64::from_bits(0x3fd40645f0c6651c),
    ),
    (
        f64::from_bits(0x3fe886e6037841ed),
        f64::from_bits(0x3fd88e9c2c1b9ff8),
    ),
    (
        f64::from_bits(0x3fe767dcf5534862),
        f64::from_bits(0x3fdce0a44eb17bcc),
    ),
];
const LOG2_POLY: [f64; 5] = [
    f64::from_bits(0x3fd27616c9496e0b),
    f64::from_bits(0xbfd71969a075c67a),
    f64::from_bits(0x3fdec70a6ca7badd),
    f64::from_bits(0xbfe7154748bef6c8),
    f64::from_bits(0x3ff71547652ab82b),
];

/// `__exp2f_data.tab` — `uint(2^(i/32)) - (i << 47)`.
const EXP2_TAB: [u64; 32] = [
    0x3ff0000000000000,
    0x3fefd9b0d3158574,
    0x3fefb5586cf9890f,
    0x3fef9301d0125b51,
    0x3fef72b83c7d517b,
    0x3fef54873168b9aa,
    0x3fef387a6e756238,
    0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715,
    0x3feef1a7373aa9cb,
    0x3feedea64c123422,
    0x3feece086061892d,
    0x3feebfdad5362a27,
    0x3feeb42b569d4f82,
    0x3feeab07dd485429,
    0x3feea47eb03a5585,
    0x3feea09e667f3bcd,
    0x3fee9f75e8ec5f74,
    0x3feea11473eb0187,
    0x3feea589994cce13,
    0x3feeace5422aa0db,
    0x3feeb737b0cdc5e5,
    0x3feec49182a3f090,
    0x3feed503b23e255d,
    0x3feee89f995ad3ad,
    0x3feeff76f2fb5e47,
    0x3fef199bdd85529c,
    0x3fef3720dcef9069,
    0x3fef5818dcfba487,
    0x3fef7c97337b9b5f,
    0x3fefa4afa2a490da,
    0x3fefd0765b6e4540,
];
const EXP2_POLY: [f64; 3] = [
    f64::from_bits(0x3fac6af84b912394),
    f64::from_bits(0x3fcebfce50fac4f3),
    f64::from_bits(0x3fe62e42ff0c52d6),
];
/// `__exp2f_data.shift_scaled` = f64::from_bits(0x4338000000000000) / 32 — rounds to the 1/32 grid.
const SHIFT_SCALED: f64 = f64::from_bits(0x42e8000000000000);

/// e_powf's `log2_inline` (POWF_SCALE_BITS = 0 → no output scaling).
// `x = a * b + x` lines are transcribed verbatim from e_powf.c.
#[allow(clippy::assign_op_pattern)]
fn log2_inline(ix: u32) -> f64 {
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> 19) % 16) as usize;
    let top = tmp & 0xff800000;
    let iz = ix.wrapping_sub(top);
    let k = (top as i32) >> 23;
    let (invc, logc) = LOG2_TAB[i];
    let z = f64::from(f32::from_bits(iz));
    let r = z * invc - 1.0;
    let y0 = logc + f64::from(k);
    let r2 = r * r;
    let mut y = LOG2_POLY[0] * r + LOG2_POLY[1];
    let p = LOG2_POLY[2] * r + LOG2_POLY[3];
    let r4 = r2 * r2;
    let mut q = LOG2_POLY[4] * r + y0;
    q = p * r2 + q;
    y = y * r4 + q;
    y
}

/// e_powf's `exp2_inline` (non-TOINT_INTRINSICS branch; sign_bias != 0 only
/// on slow paths this module delegates, kept for transcription fidelity).
// `x = a * b + x` lines are transcribed verbatim from e_powf.c.
#[allow(clippy::assign_op_pattern)]
fn exp2_inline(xd: f64, sign_bias: u32) -> f64 {
    let kd = xd + SHIFT_SCALED;
    let ki = kd.to_bits();
    let kd = kd - SHIFT_SCALED;
    let r = xd - kd;
    let t = EXP2_TAB[(ki % 32) as usize];
    let ski = ki.wrapping_add(u64::from(sign_bias));
    let t = t.wrapping_add(ski << 47);
    let s = f64::from_bits(t);
    let z = EXP2_POLY[0] * r + EXP2_POLY[1];
    let r2 = r * r;
    let mut y = EXP2_POLY[2] * r + 1.0;
    y = z * r2 + y;
    y * s
}

fn zeroinfnan(ix: u32) -> bool {
    ix.wrapping_mul(2).wrapping_sub(1) >= 2 * 0x7f80_0000u32 - 1
}

/// ARM fast pow arithmetic with CORE-MATH Rust fallback on all edge paths.
#[inline]
pub(crate) fn powf_arm_core(x: f32, y: f32) -> f32 {
    let ix = x.to_bits();
    let iy = y.to_bits();
    if ix.wrapping_sub(0x0080_0000) >= 0x7f80_0000 - 0x0080_0000 || zeroinfnan(iy) {
        return zenmath::scalar::powf(x, y);
    }
    // Positive normal finite x, nonzero finite y: the fast path.
    if ix & 0x8000_0000 != 0 {
        return zenmath::scalar::powf(x, y); // negative x: checkint/invalid branches
    }
    let logx = log2_inline(ix);
    let ylogx = f64::from(y) * logx;
    if (ylogx.to_bits() >> 47 & 0xffff) >= (126.0f64.to_bits() >> 47) {
        // |y*log(x)| >= 126 — oflow/uflow/may_uflow boundary lanes.
        return zenmath::scalar::powf(x, y);
    }
    exp2_inline(ylogx, 0) as f32
}

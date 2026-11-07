//! Shared safe-integer range reduction from CORE-MATH sinf/cosf, 660067df.
//! See their retained copyright and MIT license notices.
#[inline(never)]
pub(super) fn big(u: u32) -> (f64, i32) {
    big_scaled(u, 147)
}
#[inline(never)]
pub(super) fn big_scaled(u: u32, offset: u32) -> (f64, i32) {
    const IPI: [u64; 4] = [
        0xfe5163abdebbc562,
        0xdb6295993c439041,
        0xfc2757d1f534ddc0,
        0xa2f9836e4e441529,
    ];
    let e = (u >> 23) & 255;
    let m = ((u & 0x7fffff) | (1 << 23)) as u128;
    let p0 = m * IPI[0] as u128;
    let p1 = m * IPI[1] as u128 + (p0 >> 64);
    let p2 = m * IPI[2] as u128 + (p1 >> 64);
    let p3 = m * IPI[3] as u128 + (p2 >> 64);
    let p3h = (p3 >> 64) as u64;
    let p3l = p3 as u64;
    let p2l = p2 as u64;
    let p1l = p1 as u64;
    let s = e - offset;
    let (i, a) = if s < 64 {
        (
            (p3h << s) | (p3l >> (64 - s)),
            (p3l << s) | (p2l >> (64 - s)),
        )
    } else if s == 64 {
        (p3l, p2l)
    } else {
        (
            (p3l << (s - 64)) | (p2l >> (128 - s)),
            (p2l << (s - 64)) | (p1l >> (128 - s)),
        )
    };
    let a = a as i64;
    let sign = (u as i32) >> 31;
    let sm = (a >> 63) as i32;
    let i = (i as i32).wrapping_sub(sm);
    let z = (a ^ (sign as i64)) as f64 * f64::from_bits(0x3bf0000000000000);
    (z, (i ^ sign).wrapping_sub(sign))
}
pub(super) fn small(x: f64, short: bool) -> (f64, i32) {
    let (hi, lo) = if short {
        (f64::from_bits(0x40145f306dc9c883) * x, 0.0)
    } else {
        (
            f64::from_bits(0x40145f306e000000) * x,
            -f64::from_bits(0x3e2b1bbead603d8b) * x,
        )
    };
    let id = crate::reference::roundeven(hi);
    let q = (f64::from_bits(0x4338000000000000) + id).to_bits() as i32;
    ((hi - id) + lo, q)
}

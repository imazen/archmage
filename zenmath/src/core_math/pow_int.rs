/* Correctly-rounded power function for two binary64 values.

Copyright (c) 2022, 2023 CERN and Inria
Authors: Tom Hubrecht and Paul Zimmermann

This file is part of the CORE-MATH project
(https://core-math.gitlabpages.inria.fr/).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#[derive(Clone, Copy)]
pub(super) struct D {
    pub hi: u64,
    pub lo: u64,
    pub ex: i64,
    pub sgn: u64,
}
const ZERO: D = D {
    hi: 0,
    lo: 0,
    ex: -1076,
    sgn: 0,
};
impl D {
    pub fn bits(self) -> u128 {
        ((self.hi as u128) << 64) | self.lo as u128
    }
    pub fn with_bits(self, b: u128) -> Self {
        Self {
            hi: (b >> 64) as u64,
            lo: b as u64,
            ..self
        }
    }
    pub fn neg(mut self) -> Self {
        self.sgn ^= 1;
        self
    }
    pub fn add(self, b: Self) -> Self {
        if self.hi == 0 {
            return b;
        }
        if b.hi == 0 {
            return self;
        }
        let cmp = (self.ex, self.hi, self.lo).cmp(&(b.ex, b.hi, b.lo));
        if cmp.is_lt() {
            return b.add(self);
        }
        if cmp.is_eq() {
            return if self.sgn != b.sgn {
                ZERO
            } else {
                Self {
                    ex: self.ex + 1,
                    ..self
                }
            };
        }
        let a = self.bits();
        let k = self.ex - b.ex;
        let bb = if k < 128 { b.bits() >> k } else { 0 };
        let mut ex = self.ex;
        let c = if self.sgn != b.sgn {
            let mut c = a.wrapping_sub(bb);
            let sh = c.leading_zeros();
            if sh > 0 {
                c = if k == 1 {
                    (a << sh).wrapping_sub(b.bits() << (sh - 1))
                } else {
                    (a << sh).wrapping_sub(bb << sh)
                };
                ex -= i64::from(sh);
            }
            let sh = (c >> 64).leading_zeros() - 64;
            ex -= i64::from(sh);
            c << sh
        } else {
            let (c, carry) = a.overflowing_add(bb);
            if carry {
                ex += 1;
                (1 << 127) | (c >> 1)
            } else {
                c
            }
        };
        // The normalization above is exact when cancellation exposes discarded bits.

        Self { ex, ..self }.with_bits(c)
    }
    pub fn mul(self, b: Self) -> Self {
        let v = (self.hi as u128 * b.hi as u128)
            .wrapping_add((self.hi as u128 * b.lo as u128) >> 64)
            .wrapping_add((self.lo as u128 * b.hi as u128) >> 64);
        let sh = u32::from(v >> 127 == 0);
        Self {
            ex: self.ex + b.ex + 1 - i64::from(sh),
            sgn: self.sgn ^ b.sgn,
            ..self
        }
        .with_bits(v << sh)
    }
    pub fn from_double(x: f64) -> Self {
        let b = x.to_bits();
        let e = (b >> 52) & 2047;
        let m = (b & 0xfffffffffffff) + if e != 0 { 1 << 52 } else { 0 };
        let sh = m.leading_zeros();
        Self {
            hi: m << sh,
            lo: 0,
            ex: e as i64 - 1023 - if sh > 11 { i64::from(sh) - 12 } else { 0 },
            sgn: u64::from(x < 0.0),
        }
    }
    pub fn to_double(self) -> f64 {
        let self_ = Self {
            ex: self.ex + 1,
            ..self
        };
        self_.round_bits()
    }
    pub fn round_bits(self) -> f64 {
        let sign = self.sgn << 63;
        if self.hi == 0 || self.ex < -1074 {
            return f64::from_bits(sign);
        }
        if self.ex > 1024 {
            return f64::from_bits(sign | 0x7ff0_0000_0000_0000);
        }
        // Round the retained 128-bit significand directly, including ties and
        // subnormal results; this is the nearest-even form of dint_tod.
        let shift = if self.ex >= -1021 {
            75
        } else {
            (-946 - self.ex) as u32
        };
        let bits = self.bits();
        let rounded = if shift == 128 {
            u64::from(bits > (1u128 << 127))
        } else {
            let hi = (bits >> shift) as u64;
            let low = bits & ((1u128 << shift) - 1);
            let mid = 1u128 << (shift - 1);
            hi + u64::from(low > mid || (low == mid && hi & 1 != 0))
        };
        let value = if self.ex >= -1021 {
            (((self.ex + 1022) as u64) << 52) + rounded - (1 << 52)
        } else {
            rounded
        };
        f64::from_bits(sign | value)
    }
}

impl D {
    pub fn mul11(self, b: Self) -> Self {
        let v = self.hi as u128 * b.hi as u128;
        let sh = u32::from(v >> 127 == 0);
        Self {
            ex: self.ex + b.ex + 1 - i64::from(sh),
            sgn: self.sgn ^ b.sgn,
            ..self
        }
        .with_bits(v << sh)
    }
    pub fn mul21(self, b: Self) -> Self {
        self.mul(Self { lo: 0, ..b })
    }
    pub fn add11(self, b: Self) -> Self {
        if self.hi == 0 {
            return b;
        }
        if b.hi == 0 {
            return self;
        }
        let cmp = (self.ex, self.hi).cmp(&(b.ex, b.hi));
        if cmp.is_eq() {
            return if self.sgn != b.sgn {
                ZERO
            } else {
                Self {
                    ex: self.ex + 1,
                    ..self
                }
            };
        }
        let (a, bb) = if cmp.is_lt() { (b, self) } else { (self, b) };
        let k = a.ex - bb.ex;
        let bh = if k < 64 { bb.hi >> k } else { 0 };
        let mut ex = a.ex;
        let h = if a.sgn != bb.sgn {
            let c = a.hi.wrapping_sub(bh);
            let sh = c.leading_zeros();
            ex -= i64::from(sh);
            c << sh
        } else {
            let (c, carry) = a.hi.overflowing_add(bh);
            if carry {
                ex += 1;
                (1 << 63) | (c >> 1)
            } else {
                c
            }
        };
        Self {
            hi: h,
            lo: b.lo,
            ex,
            sgn: a.sgn,
        }
    }
    pub fn muli(self, n: i64) -> Self {
        if n == 0 {
            return ZERO;
        }
        let c = n.unsigned_abs();
        let mut v = self.hi as u128 * c as u128;
        let sh = if v >> 64 != 0 {
            (v >> 64).leading_zeros() - 64
        } else {
            64
        };
        v <<= sh;
        let low = ((self.lo as u128 * c as u128) << (sh - 1)) >> 63;
        let (v, carry) = v.overflowing_add(low);
        let mut r = Self {
            ex: self.ex + 64 - i64::from(sh),
            sgn: self.sgn ^ u64::from(n < 0),
            ..self
        }
        .with_bits(v);
        if carry {
            r = r.with_bits((1 << 127) | (v >> 1));
            r.ex += 1;
        }
        r
    }
    pub fn to_int(self) -> i64 {
        if self.ex < 0 {
            return 0;
        }
        let r = (self.hi >> (63 - self.ex)) as i64;
        if self.sgn != 0 { -r } else { r }
    }
}
// 256-bit arithmetic from CORE-MATH pow/qint.h. Truncated products retain
// the same limbs as each upstream precision-specialized operation.
#[derive(Clone, Copy)]
pub(super) struct Q {
    pub hh: u64,
    pub hl: u64,
    pub lh: u64,
    pub ll: u64,
    pub ex: i64,
    pub sgn: u64,
}
impl Q {
    fn hi(self) -> u128 {
        ((self.hh as u128) << 64) | self.hl as u128
    }
    fn lo(self) -> u128 {
        ((self.lh as u128) << 64) | self.ll as u128
    }
    fn with(self, h: u128, l: u128) -> Self {
        Self {
            hh: (h >> 64) as u64,
            hl: h as u64,
            lh: (l >> 64) as u64,
            ll: l as u64,
            ..self
        }
    }
    pub fn neg(mut self) -> Self {
        self.sgn ^= 1;
        self
    }
    fn shift(self, k: i64) -> Self {
        let (h, l) = (self.hi(), self.lo());
        let (h, l) = match k {
            0 => (h, l),
            1..=127 => ((h << k) | (l >> (128 - k)), l << k),
            128..=255 => (l << (k - 128), 0),
            -127..=-1 => {
                let j = -k;
                (h >> j, (h << (128 - j)) | (l >> j))
            }
            -255..=-128 => (0, h >> (-k - 128)),
            _ => (0, 0),
        };
        self.with(h, l)
    }
    fn leading(self) -> i64 {
        if self.hi() != 0 {
            i64::from(self.hi().leading_zeros())
        } else {
            128 + i64::from(self.lo().leading_zeros())
        }
    }
    fn sub(self, b: Self) -> Self {
        let (l, c) = self.lo().overflowing_sub(b.lo());
        self.with(
            self.hi().wrapping_sub(b.hi()).wrapping_sub(u128::from(c)),
            l,
        )
    }
    pub fn add(self, b: Self) -> Self {
        if self.hi() == 0 && self.lo() == 0 {
            return b;
        }
        if b.hi() == 0 && b.lo() == 0 {
            return self;
        }
        let cmp = (self.ex, self.hi(), self.lo()).cmp(&(b.ex, b.hi(), b.lo()));
        if cmp.is_lt() {
            return b.add(self);
        }
        if cmp.is_eq() {
            return if self.sgn != b.sgn {
                Self {
                    ex: -1076,
                    ..self.with(0, 0)
                }
            } else {
                Self {
                    ex: self.ex + 1,
                    ..self
                }
            };
        }
        let k = self.ex - b.ex;
        let t = b.shift(-k);
        if self.sgn != b.sgn {
            let mut r = self.sub(t);
            let e = r.leading();
            if e != 0 {
                r = self.shift(e).sub(b.shift(e - k));
                r.ex -= e;
            }
            let e = r.leading();
            r = r.shift(e);
            r.ex -= e;
            r
        } else {
            let (h, c1) = self.hi().overflowing_add(t.hi());
            let (l, c2) = self.lo().overflowing_add(t.lo());
            let (h, c3) = h.overflowing_add(u128::from(c2));
            let mut r = self.with(h, l);
            if c1 || c3 {
                r = r.shift(-1);
                r.hh |= 1 << 63;
                r.ex += 1;
            }
            r
        }
    }
    pub fn add22(self, b: Self) -> Self {
        if self.hi() == 0 {
            return b;
        }
        if b.hi() == 0 {
            return self;
        }
        let cmp = (self.ex, self.hi()).cmp(&(b.ex, b.hi()));
        if cmp.is_eq() {
            return if self.sgn != b.sgn {
                Self {
                    ex: 0,
                    ..self.with(0, 0)
                }
            } else {
                Self {
                    ex: self.ex + 1,
                    ..self
                }
            };
        }
        let mut r = self.with(self.hi(), 0).add(b.with(b.hi(), 0));
        r.lh = 0;
        r.ll = 0;
        r
    }
    pub fn mul(self, b: Self, na: usize, nb: usize) -> Self {
        let aa = [self.hh, self.hl, self.lh, self.ll];
        let bb = [b.hh, b.hl, b.lh, b.ll];
        // Accumulate retained terms into four little-endian limbs. Products on
        // the discarded diagonal contribute their high half only, as upstream.
        let mut out = [0u64; 4];
        for (i, &a) in aa[..na].iter().enumerate() {
            for (j, &b) in bb[..nb].iter().enumerate() {
                if i + j > 3 {
                    continue;
                }
                let p = a as u128 * b as u128;
                let high = 3 - i - j;
                let (v, c) = out[high].overflowing_add((p >> 64) as u64);
                out[high] = v;
                let mut carry = c;
                for v in &mut out[high + 1..] {
                    let (n, c) = v.overflowing_add(u64::from(carry));
                    *v = n;
                    carry = c;
                }
                if high > 0 {
                    let (v, c) = out[high - 1].overflowing_add(p as u64);
                    out[high - 1] = v;
                    let mut carry = c;
                    for v in &mut out[high..] {
                        let (n, c) = v.overflowing_add(u64::from(carry));
                        *v = n;
                        carry = c;
                    }
                }
            }
        }
        let mut r = Self {
            hh: out[3],
            hl: out[2],
            lh: out[1],
            ll: out[0],
            ex: self.ex + b.ex + 1,
            sgn: self.sgn ^ b.sgn,
        };
        if r.hh >> 63 == 0 {
            r = r.shift(1);
            r.ex -= 1;
        }
        r
    }
    pub fn from_double(x: f64) -> Self {
        let d = D::from_double(x);
        Self {
            hh: d.hi,
            hl: 0,
            lh: 0,
            ll: 0,
            ex: d.ex,
            sgn: d.sgn,
        }
    }
    pub fn muli(self, n: i64) -> Self {
        if n == 0 {
            return self.with(0, 0);
        }
        let c = n.unsigned_abs();
        let z = c.leading_zeros();
        self.mul(
            Self {
                hh: c << z,
                hl: 0,
                lh: 0,
                ll: 0,
                ex: 63 - i64::from(z),
                sgn: u64::from(n < 0),
            },
            4,
            1,
        )
    }
    pub fn to_int(self) -> i64 {
        if self.ex < 0 {
            return 0;
        }
        let r = (self.hh >> (63 - self.ex)) as i64;
        if self.sgn != 0 { -r } else { r }
    }
    pub fn to_double(self) -> f64 {
        D {
            hi: self.hh,
            lo: self.hl | u64::from(self.lh != 0 || self.ll != 0),
            ex: self.ex,
            sgn: self.sgn,
        }
        .to_double()
    }
}

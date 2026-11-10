/* Fast 192-bit arithmetic routines.

Copyright (c) 2023-2024 Paul Zimmermann and Alexei Sibidanov

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
use super::fma;
#[derive(Clone, Copy)]
pub(super) struct Tint {
    pub h: u64,
    pub m: u64,
    pub l: u64,
    pub ex: i64,
    pub sgn: u64,
}
pub(super) const ZERO: Tint = Tint {
    h: 0,
    m: 0,
    l: 0,
    ex: -1076,
    sgn: 0,
};
pub(super) const ONE: Tint = Tint {
    h: 1 << 63,
    m: 0,
    l: 0,
    ex: 1,
    sgn: 0,
};
pub(super) const PI: Tint = Tint {
    h: 0xc90fdaa22168c234,
    m: 0xc4c6628b80dc1cd1,
    l: 0x29024e088a67cc74,
    ex: 2,
    sgn: 0,
};
pub(super) const PI2: Tint = Tint { ex: 1, ..PI };
impl Tint {
    fn hi(self) -> u128 {
        (u128::from(self.h) << 64) | u128::from(self.m)
    }
    fn set_hi(&mut self, h: u128) {
        self.h = (h >> 64) as u64;
        self.m = h as u64;
    }
    pub fn neg(mut self) -> Self {
        self.sgn ^= 1;
        self
    }
    fn right(mut self, k: i64) -> Self {
        let h = self.hi();
        let l = self.l;
        let (h, l) = match k {
            0 => (h, l),
            1..=63 => (h >> k, ((h << (64 - k)) as u64) | (l >> k)),
            64 => (h >> 64, h as u64),
            65..=127 => (h >> k, (h >> (k - 64)) as u64),
            128..=191 => (0, (h >> (k - 64)) as u64),
            _ => (0, 0),
        };
        self.set_hi(h);
        self.l = l;
        self
    }
    fn left(mut self, k: i64) -> Self {
        let h = self.hi();
        let l = self.l;
        let (h, l) = match k {
            0 => (h, l),
            1..=63 => ((h << k) | u128::from(l >> (64 - k)), l << k),
            64 => ((h << 64) | u128::from(l), 0),
            65..=127 => ((h << k) | (u128::from(l) << (k - 64)), 0),
            128..=191 => (u128::from(l) << (k - 64), 0),
            _ => (0, 0),
        };
        self.set_hi(h);
        self.l = l;
        self
    }
    fn leading(self) -> i64 {
        if self.hi() != 0 {
            i64::from(self.hi().leading_zeros())
        } else {
            128 + i64::from(self.l.leading_zeros())
        }
    }
    fn sub_sig(mut self, b: Self) -> Self {
        let (l, c) = self.l.overflowing_sub(b.l);
        self.set_hi(self.hi().wrapping_sub(b.hi()).wrapping_sub(u128::from(c)));
        self.l = l;
        self
    }
    pub fn add(mut self, mut b: Self) -> Self {
        let cmp = match (self.h == 0, b.h == 0) {
            (true, true) => core::cmp::Ordering::Equal,
            (true, false) => core::cmp::Ordering::Less,
            (false, true) => core::cmp::Ordering::Greater,
            _ => (self.ex, self.h, self.m, self.l).cmp(&(b.ex, b.h, b.m, b.l)),
        };
        if cmp.is_eq() {
            if self.sgn != b.sgn {
                return ZERO;
            }
            self.ex += 1;
            return self;
        }
        if cmp.is_lt() {
            core::mem::swap(&mut self, &mut b);
        }
        let sh = self.ex - b.ex;
        let t = b.right(sh);
        if self.sgn != b.sgn {
            let t = self.sub_sig(t);
            let e = t.leading();
            let mut r = if e <= 1 || sh == 0 {
                let mut r = t.left(e);
                r.ex = self.ex - e;
                r
            } else {
                let t = self.left(e).sub_sig(b.left(e - sh));
                let e1 = t.leading();
                let mut r = t.left(e1);
                r.ex = self.ex - e - e1;
                r
            };
            r.sgn = self.sgn;
            r
        } else {
            let (l, cl) = self.l.overflowing_add(t.l);
            let (h, c1) = self.hi().overflowing_add(t.hi());
            let (h, c2) = h.overflowing_add(u128::from(cl));
            let mut r = self;
            r.l = l;
            r.set_hi(h);
            if c1 || c2 {
                r = r.right(1);
                r.h |= 1 << 63;
                r.ex += 1;
            }
            r
        }
    }
    pub fn mul(self, b: Self) -> Self {
        let ah = u128::from(self.h);
        let am = u128::from(self.m);
        let al = u128::from(self.l);
        let bh = u128::from(b.h);
        let bm = u128::from(b.m);
        let bl = u128::from(b.l);
        let rh = ah * bh;
        let rm1 = ah * bm;
        let rm2 = am * bh;
        let rl = ((ah * bl) >> 64) + ((am * bm) >> 64) + ((al * bh) >> 64);
        let mut r = Self {
            h: (rh >> 64) as u64,
            m: rh as u64,
            l: rm1 as u64,
            ex: self.ex + b.ex,
            sgn: self.sgn ^ b.sgn,
        };
        let (m, c) = r.m.overflowing_add((rm1 >> 64) as u64);
        r.m = m;
        r.h += u64::from(c);
        let (l, c) = r.l.overflowing_add(rm2 as u64);
        r.l = l;
        let mut cm = u64::from(c);
        let (m, c) = r.m.overflowing_add((rm2 >> 64) as u64);
        r.m = m;
        r.h += u64::from(c);
        let (l, c) = r.l.overflowing_add(rl as u64);
        r.l = l;
        cm += (rl >> 64) as u64 + u64::from(c);
        let (m, c) = r.m.overflowing_add(cm);
        r.m = m;
        r.h += u64::from(c);
        if r.h >> 63 == 0 {
            r = r.left(1);
            r.ex -= 1;
        }
        r
    }
    pub fn from_double(x: f64) -> Self {
        let bits = x.to_bits();
        let ax = bits & 0x7fff_ffff_ffff_ffff;
        let e = (ax >> 52) as i64;
        let (h, ex) = if e != 0 {
            ((1 << 63) | (ax << 11), e - 1022)
        } else {
            let z = ax.leading_zeros();
            (ax << z, -1010 - i64::from(z))
        };
        Self {
            h,
            m: 0,
            l: 0,
            ex,
            sgn: bits >> 63,
        }
    }
    pub fn to_double(self, err: u64) -> f64 {
        let sign = self.sgn << 63;
        if self.ex >= 1025 {
            return f64::from_bits(sign | 0x7ff0_0000_0000_0000);
        }
        if self.ex <= -1074 {
            return f64::from_bits(
                sign | u64::from(
                    self.ex == -1074 && !(self.h == 1 << 63 && self.m == 0 && self.l == 0),
                ),
            );
        }
        let (mut hh, mut mm, mut ll, mut ex) = (self.h, self.m, self.l, self.ex);
        let mut low = hh & 0x7ff;
        assert!(
            !((mm == 0 && (low == 0 || low == 0x400) && ll < err)
                || (mm == u64::MAX && (low == 0x3ff || low == 0x7ff) && !ll < err)),
            "CORE-MATH atan2 unresolved rounding"
        );
        if ex <= -1022 {
            let sh = -1021 - ex;
            ll = (mm << (64 - sh)) | (ll >> sh) | u64::from(ll > 0);
            mm = (hh << (64 - sh)) | (mm >> sh);
            hh >>= sh;
            low = hh & 0x7ff;
            ex += sh;
        }
        let l = if err == 0 {
            0.0
        } else if low < 0x400 {
            0.25
        } else if low == 0x400 && mm == 0 && ll == 0 {
            0.5
        } else {
            0.75
        };
        let s = if self.sgn == 0 { 1.0 } else { -1.0 };
        let h = fma(l, s, s * ((hh >> 11) as f64)) * f64::from_bits(0x3cb0000000000000);
        h * f64::from_bits(((ex + 1022) as u64) << 52)
    }
    fn inv(self) -> Self {
        let mut a = self.to_double(0);
        let sub = a.abs() < f64::MIN_POSITIVE;
        if sub {
            a *= 9007199254740992.0;
        }
        let mut r = Self::from_double(1.0 / a);
        if sub {
            r.ex += 53;
        }
        r = r.add(r.mul(ONE.add(self.mul(r).neg())));
        r
    }
    pub fn div(self, a: Self) -> Self {
        let y = a.inv();
        let r = y.mul(self);
        r.add(y.mul(self.add(a.mul(r).neg())))
    }
}

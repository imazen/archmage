/* Correctly-rounded power function for two binary64 values.

Copyright (c) 2022 CERN.
Author: Tom Hubrecht

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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, exp_dRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// Specialized private 128-bit floating arithmetic from CORE-MATH log/dint.h.
#[derive(Clone, Copy)]
struct D {
    hi: u64,
    lo: u64,
    ex: i64,
    sgn: u64,
}
impl D {
    fn bits(self) -> u128 {
        ((self.hi as u128) << 64) | self.lo as u128
    }
    fn with_bits(self, v: u128) -> Self {
        Self {
            hi: (v >> 64) as u64,
            lo: v as u64,
            ..self
        }
    }
    fn add(self, b: Self) -> Self {
        if self.bits() == 0 {
            return b;
        }
        if b.bits() == 0 {
            return self;
        }
        let order = (self.ex, self.hi, self.lo).cmp(&(b.ex, b.hi, b.lo));
        if order.is_lt() {
            return b.add(self);
        }
        if order.is_eq() {
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
        let mut bb = b.bits();
        let mut ex = self.ex;
        if self.ex > b.ex {
            let sh = (self.ex - b.ex) as u32;
            if sh <= 128 {
                bb = bb.wrapping_add((bb >> (sh - 1)) & 1);
            }
            bb = if sh < 128 { bb >> sh } else { 0 };
        }
        let mut v = if self.sgn != b.sgn {
            a.wrapping_sub(bb)
        } else {
            let (mut v, carry) = a.overflowing_add(bb);
            if carry {
                v = v.wrapping_add(v & 1);
                v = (1 << 127) | (v >> 1);
                ex += 1;
            }
            v
        };
        if v == 0 {
            return ZERO;
        }
        let shift = v.leading_zeros();
        v <<= shift;
        Self {
            ex: ex - shift as i64,
            ..self
        }
        .with_bits(v)
    }
    fn mul(self, b: Self) -> Self {
        let mut t = self.hi as u128 * b.hi as u128;
        let (m, carry) =
            (self.hi as u128 * b.lo as u128).overflowing_add(self.lo as u128 * b.hi as u128);
        t = t.wrapping_add((carry as u128) << 64).wrapping_add(m >> 64);
        let shift = u32::from(t >> 127 == 0);
        t <<= shift;
        t = t.wrapping_add((m as u64 >> 63) as u128);
        Self {
            ex: self.ex + b.ex - shift as i64 + 1,
            sgn: self.sgn ^ b.sgn,
            ..self
        }
        .with_bits(t)
    }
    fn muli(self, b: i64) -> Self {
        if b == 0 {
            return ZERO;
        }
        let c = b.unsigned_abs();
        let mut t = self.hi as u128 * c as u128;
        let mut m = if t >> 64 != 0 {
            (t >> 64).leading_zeros() - 64
        } else {
            64
        };
        t <<= m;
        let l = ((self.lo as u128 * c as u128) << (m - 1)) >> 63;
        let (v, carry) = t.overflowing_add(l);
        t = v;
        if carry {
            t = t.wrapping_add(t & 1);
            t = (1 << 127) | (t >> 1);
            m -= 1;
        }
        Self {
            ex: self.ex + 64 - m as i64,
            sgn: self.sgn ^ u64::from(b < 0),
            ..self
        }
        .with_bits(t)
    }
}
const M_ONE: D = D {
    hi: 0x8000000000000000,
    lo: 0x0,
    ex: 0,
    sgn: 0x1,
};
const LOG2: D = D {
    hi: 0xb17217f7d1cf79ab,
    lo: 0xc9e3b39803f2f6af,
    ex: -1,
    sgn: 0x0,
};
const ZERO: D = D {
    hi: 0x0,
    lo: 0x0,
    ex: 0,
    sgn: 0x0,
};
const _INVERSE_2: [D; 240] = [
    D {
        hi: 0x8000000000000000,
        lo: 0x0,
        ex: 1,
        sgn: 0x0,
    },
    D {
        hi: 0xfe03f80fe03f80ff,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xfc0fc0fc0fc0fc10,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xfa232cf252138ac0,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xf83e0f83e0f83e10,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xf6603d980f6603da,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xf4898d5f85bb3951,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xf2b9d6480f2b9d65,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xf0f0f0f0f0f0f0f1,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xef2eb71fc4345239,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xed7303b5cc0ed731,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xebbdb2a5c1619c8c,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xea0ea0ea0ea0ea0f,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xe865ac7b7603a197,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xe6c2b4481cd8568a,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xe525982af70c880f,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xe38e38e38e38e38f,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xe1fc780e1fc780e2,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xe070381c0e070382,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xdee95c4ca037ba58,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xdd67c8a60dd67c8b,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xdbeb61eed19c5958,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xda740da740da740e,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xd901b2036406c80e,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xd79435e50d79435f,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xd62b80d62b80d62c,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xd4c77b03531dec0e,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xd3680d3680d3680e,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xd20d20d20d20d20e,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xd0b69fcbd2580d0c,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xcf6474a8819ec8ea,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xce168a7725080ce2,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xcccccccccccccccd,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xcb8727c065c393e1,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xca4587e6b74f032a,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xc907da4e871146ad,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xc7ce0c7ce0c7ce0d,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xc6980c6980c6980d,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xc565c87b5f9d4d1c,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xc4372f855d824ca6,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xc30c30c30c30c30d,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xc1e4bbd595f6e948,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xc0c0c0c0c0c0c0c1,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xbfa02fe80bfa02ff,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xbe82fa0be82fa0bf,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xbd69104707661aa3,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xbc52640bc52640bd,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xbb3ee721a54d880c,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xba2e8ba2e8ba2e8c,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xb92143fa36f5e02f,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xb81702e05c0b8171,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xb70fbb5a19be3659,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xb60b60b60b60b60c,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xb509e68a9b948220,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xb40b40b40b40b40c,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xb30f63528917c80c,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xb21642c8590b2165,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xb11fd3b80b11fd3c,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xb02c0b02c0b02c0c,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xaf3addc680af3ade,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xae4c415c9882b932,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xad602b580ad602b6,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xac7691840ac76919,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xab8f69e28359cd12,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xaaaaaaaaaaaaaaab,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xa9c84a47a07f5638,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xa8e83f5717c0a8e9,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xa80a80a80a80a80b,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xa72f05397829cbc2,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xa655c4392d7b73a8,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xa57eb50295fad40b,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xa4a9cf1d96833752,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xa3d70a3d70a3d70b,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xa3065e3fae7cd0e1,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xa237c32b16cfd773,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xa16b312ea8fc377d,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xa0a0a0a0a0a0a0a1,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x9fd809fd809fd80a,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x9f1165e7254813e3,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x9e4cad23dd5f3a21,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x9d89d89d89d89d8a,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x9cc8e160c3fb19b9,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x9c09c09c09c09c0a,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x9b4c6f9ef03a3caa,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x9a90e7d95bc609aa,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x99d722dabde58f07,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x991f1a515885fb38,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x9868c809868c8099,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x97b425ed097b425f,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x97012e025c04b80a,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x964fda6c0964fda7,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x95a02568095a0257,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x94f2094f2094f20a,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x9445809445809446,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x939a85c40939a85d,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x92f113840497889d,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x924924924924924a,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x91a2b3c4d5e6f80a,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x90fdbc090fdbc091,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x905a38633e06c43b,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8fb823ee08fb823f,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8f1779d9fdc3a219,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8e78356d1408e784,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8dda520237694809,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8d3dcb08d3dcb08e,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8ca29c046514e024,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8c08c08c08c08c09,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8b70344a139bc75b,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8ad8f2fba9386823,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8a42f8705669db47,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x89ae4089ae4089af,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x891ac73ae9819b51,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8888888888888889,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x87f78087f78087f8,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8767ab5f34e47ef2,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x86d905447a34acc7,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x864b8a7de6d1d609,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x85bf37612cee3c9b,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8534085340853409,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x84a9f9c8084a9f9d,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8421084210842109,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x839930523fbe3368,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x83126e978d4fdf3c,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x828cbfbeb9a020a4,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8208208208208209,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x81848da8faf0d278,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8102040810204082,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8000000000000000,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0x8000000000000000,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
    D {
        hi: 0xff00ff00ff00ff02,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xfe03f80fe03f80ff,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xfd08e5500fd08e56,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xfc0fc0fc0fc0fc11,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xfb18856506ddaba7,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xfa232cf252138ac1,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xf92fb2211855a866,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xf83e0f83e0f83e11,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xf74e3fc22c700f76,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xf6603d980f6603db,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xf57403d5d00f5741,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xf4898d5f85bb3951,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xf3a0d52cba872337,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xf2b9d6480f2b9d66,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xf1d48bcee0d399fb,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xf0f0f0f0f0f0f0f2,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xf00f00f00f00f010,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xef2eb71fc4345239,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xee500ee500ee5010,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xed7303b5cc0ed731,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xec979118f3fc4da3,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xebbdb2a5c1619c8d,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xeae56403ab959010,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xea0ea0ea0ea0ea10,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xe939651fe2d8d35d,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xe865ac7b7603a198,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xe79372e225fe30da,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xe6c2b4481cd8568a,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xe5f36cb00e5f36cc,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xe525982af70c880f,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xe45932d7dc52100f,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xe38e38e38e38e38f,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xe2c4a6886a4c2e11,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xe1fc780e1fc780e3,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xe135a9c97500e137,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xe070381c0e070383,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xdfac1f74346c5760,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xdee95c4ca037ba58,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xde27eb2c41f3d9d2,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xdd67c8a60dd67c8b,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xdca8f158c7f91ab9,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xdbeb61eed19c5959,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xdb2f171df770291a,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xda740da740da740f,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xd9ba4256c0366e92,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xd901b2036406c80f,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xd84a598ec9151f44,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xd79435e50d79435f,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xd6df43fca482f00e,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xd62b80d62b80d62d,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xd578e97c3f5fe552,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xd4c77b03531dec0e,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xd4173289870ac52f,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xd3680d3680d3680e,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xd2ba083b445250ac,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xd20d20d20d20d20e,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xd161543e28e50275,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xd0b69fcbd2580d0c,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xd00d00d00d00d00e,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xcf6474a8819ec8ea,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xcebcf8bb5b4169cc,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xce168a7725080ce2,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xcd712752a886d243,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xccccccccccccccce,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xcc29786c7607f9a0,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xcb8727c065c393e1,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xcae5d85f1bbd6c96,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xca4587e6b74f032a,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc9a633fcd967300e,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc907da4e871146ae,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc86a78900c86a78a,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc7ce0c7ce0c7ce0d,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc73293d789b9f839,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc6980c6980c6980d,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc5fe740317f9d00d,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc565c87b5f9d4d1d,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc4ce07b00c4ce07c,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc4372f855d824ca7,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc3a13de60495c774,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc30c30c30c30c30d,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc2780613c0309e03,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc1e4bbd595f6e948,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc152500c152500c2,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc0c0c0c0c0c0c0c2,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xc0300c0300c0300d,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xbfa02fe80bfa0300,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xbf112a8ad278e8de,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xbe82fa0be82fa0c0,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xbdf59c91700bdf5b,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xbd69104707661aa4,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xbcdd535db1cc5b7c,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xbc52640bc52640bd,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xbbc8408cd63069a2,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xbb3ee721a54d880d,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xbab656100bab6562,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xba2e8ba2e8ba2e8d,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xb9a7862a0ff46589,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xb92143fa36f5e02f,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xb89bc36ce3e0453b,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xb81702e05c0b8171,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xb79300b79300b794,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xb70fbb5a19be365a,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xb68d31340e4307d9,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xb60b60b60b60b60c,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xb58a485518d1e7e5,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xb509e68a9b948220,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xb48a39d44685fe98,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xb40b40b40b40b40c,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xb38cf9b00b38cf9c,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xb30f63528917c80c,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
    D {
        hi: 0xb2927c29da5519d0,
        lo: 0x0,
        ex: -1,
        sgn: 0x0,
    },
];
const _LOG_INV_2: [D; 240] = [
    D {
        hi: 0xb17217f7d1cf79ab,
        lo: 0xc9e3b39803f2f6af,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0xaf74155120c9011d,
        lo: 0x46d235ee63073dc,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0xad7a02e1b24efd32,
        lo: 0x160864fd949b4bd3,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0xab83d135dc633301,
        lo: 0xffe6607ba902ef3b,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0xa991713433c2b999,
        lo: 0xba4aea614d05700,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0xa7a2d41ad270c9d7,
        lo: 0xcd362382a7688479,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0xa5b7eb7cb860fb89,
        lo: 0x7b6a62a0dec6e072,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0xa3d0a93f45169a4b,
        lo: 0x9594fab088c0d64,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0xa1ecff97c91e267b,
        lo: 0x1b7efae08e597e16,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0xa00ce1092e5498c4,
        lo: 0x69879c5a30cd1241,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x9e304061b5fda91a,
        lo: 0x4603d87b6df81ac,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x9c5710b8cbb73a42,
        lo: 0xaa554b2dd4619e63,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x9a81456cec642e10,
        lo: 0x4d49f9aaea3cb5e0,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x98aed221a03458b6,
        lo: 0x732f89321647b358,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x96dfaabd86fa1647,
        lo: 0xd61188fbc94e2f14,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x9513c36876083696,
        lo: 0xb5cbc416a2418011,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x934b1089a6dc93c2,
        lo: 0xbf5bb3b60554e151,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x918586c5f5e4bf01,
        lo: 0x9f92199ed1a4bab0,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x8fc31afe30b2c6de,
        lo: 0xe300bf167e95da66,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x8e03c24d7300395a,
        lo: 0xcddae1ccce247837,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x8c47720791e53314,
        lo: 0x762ad19415fe25a5,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x8a8e1fb794b09134,
        lo: 0x9eb628dba173c82d,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x88d7c11e3ad53cdc,
        lo: 0x8a3111a707b6de2c,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x87244c308e670a66,
        lo: 0x85e005d06dbfa8f7,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x8573b71682a7d21b,
        lo: 0xb21f9f89c1ab80b2,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x83c5f8299e2b4091,
        lo: 0xb8f6fafe8fbb68b8,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x821b05f3b01d6774,
        lo: 0xdb0d58c3f7e2ea1e,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0x8072d72d903d588c,
        lo: 0x7dd1b09c70c40109,
        ex: -1,
        sgn: 0x1,
    },
    D {
        hi: 0xfd9ac57bd2442180,
        lo: 0xaf05924d258c14c4,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xfa553f7018c966f4,
        lo: 0x2780a545a1b54dce,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xf7150ab5a09f27f6,
        lo: 0xa470250d40ebe8e,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xf3da161eed6b9ab1,
        lo: 0x248d42f78d3e65d2,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xf0a450d139366ca7,
        lo: 0x7c66eb6408ff6432,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xed73aa4264b0adeb,
        lo: 0x5391cf4b33e42996,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xea481236f7d35bb2,
        lo: 0x39a767a80d6d97e6,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xe72178c0323a1a0f,
        lo: 0xcc4e1653e71d9973,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xe3ffce3a2aa64923,
        lo: 0x8eadb651b49ac539,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xe0e30349fd1cec82,
        lo: 0x3e8e1802aba24d5,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xddcb08dc0717d85c,
        lo: 0x940a666c87842842,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xdab7d02231484a93,
        lo: 0xbec20cca6efe2ac4,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xd7a94a92466e833c,
        lo: 0xcd88bba7d0cee8df,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xd49f69e456cf1b7b,
        lo: 0x7f53bd2e406e66e6,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xd19a201127d3c646,
        lo: 0x279d79f51dcc7301,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xce995f50af69d863,
        lo: 0x432f3f4f861ad6a8,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xcb9d1a189ab56e77,
        lo: 0x7d7e9307c70c0667,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xc8a5431adfb44ca6,
        lo: 0x48ce7c1a75e341a,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xc5b1cd44596fa51f,
        lo: 0xf218fb8f9f9ef27f,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xc2c2abbb6e5fd570,
        lo: 0x3337789d592e296,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xbfd7d1dec0a8df70,
        lo: 0x37eda996244bccaf,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xbcf13343e7d9ec7f,
        lo: 0x2afd17781bb3afea,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xba0ec3b633dd8b0b,
        lo: 0x91dc60b2b059a609,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xb730773578cb90b3,
        lo: 0xaa1116c3466beb6c,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xb45641f4e350a0d4,
        lo: 0xe756eba00bc33976,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xb1801859d56249de,
        lo: 0x98ce51fff99479cb,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xaeadeefacaf97d37,
        lo: 0x9dd6e688ebb13b01,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xabdfba9e468fd6f9,
        lo: 0x472ea07749ce6bd1,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xa9157039c51ebe72,
        lo: 0xe164c759686a2207,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xa64f04f0b961df78,
        lo: 0x54f5275c2d15c21e,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xa38c6e138e20d834,
        lo: 0xd698298adddd7f30,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xa0cda11eaf46390e,
        lo: 0x632438273918db7d,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0x9e1293b9998c1dad,
        lo: 0x3b035eae273a855c,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0x9b5b3bb5f088b768,
        lo: 0x5078bbe3d392be24,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0x98a78f0e9ae71d87,
        lo: 0x64dec34784707838,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0x95f783e6e49a9cfc,
        lo: 0x25004f3ef063312,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0x934b1089a6dc93c2,
        lo: 0xdf5bb3b60554e151,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0x90a22b6875c6a1f8,
        lo: 0x8e91aeba609c8876,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0x8dfccb1ad35ca6ef,
        lo: 0x9947bdb6ddcaf59a,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0x8b5ae65d67db9acf,
        lo: 0x7ba5168126a58b99,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0x88bc74113f23def3,
        lo: 0xbc5a0fe396f40f1c,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0x86216b3b0b17188c,
        lo: 0x363ceae88f720f1d,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0x8389c3026ac3139d,
        lo: 0x6adda9d2270fa1f3,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0x80f572b1363487bc,
        lo: 0xedbd0b5b3479d5f2,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0xfcc8e3659d9bcbf1,
        lo: 0x8a0cdf301431b60b,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xf7ad6f26e7ff2efc,
        lo: 0x9cd2238f75f969ad,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xf29877ff38809097,
        lo: 0x2b020fa1820c948d,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xed89ed86a44a01ab,
        lo: 0x9d49f96cb88317a,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xe881bf932af3dac3,
        lo: 0x2524848e3443e03f,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xe37fde37807b84e3,
        lo: 0x5e9a750b6b68781c,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xde8439c1dec5687c,
        lo: 0x9d57da945b5d0aa6,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xd98ec2bade71e53e,
        lo: 0xd0a98f2ad65bee96,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xd49f69e456cf1b7a,
        lo: 0x5f53bd2e406e66e7,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xcfb6203844b3209b,
        lo: 0x18cb02f33f79c16b,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xcad2d6e7b80bf915,
        lo: 0xcc507fb7a3d0bf69,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xc5f57f59c7f46156,
        lo: 0x9a8b6997a402bf30,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xc11e0b2a8d1e0de1,
        lo: 0xda631e830fd308fe,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xbc4c6c2a226399f6,
        lo: 0x276ebcfb2016a433,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xb780945bab55dcea,
        lo: 0xb4c7bc3d32750fd9,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xb2ba75f46099cf8f,
        lo: 0x243c2e77904afa76,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xadfa035aa1ed8fdd,
        lo: 0x549767e410316d2b,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xa93f2f250dac67d5,
        lo: 0x9ad2fb8d48054add,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xa489ec199dab06f4,
        lo: 0x59fb6cf0ecb411b7,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0x9fda2d2cc9465c52,
        lo: 0x6b2b9565f5355180,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0x9b2fe580ac80b182,
        lo: 0x11a5b944aca8705,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0x968b08643409ceb9,
        lo: 0xd5c0da506a088482,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0x91eb89524e100d28,
        lo: 0xbfd3df5c52d67e77,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0x8d515bf11fb94f22,
        lo: 0xa0713268840cbcbb,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0x88bc74113f23def7,
        lo: 0x9c5a0fe396f40f19,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0x842cc5acf1d0344b,
        lo: 0x6fecdfa819b96092,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xff4489cedeab2ca6,
        lo: 0xe17bd40d8d9291ec,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0xf639cc185088fe62,
        lo: 0x5066e87f2c0f733d,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0xed393b1c22351281,
        lo: 0xff4e2e660317d55f,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0xe442c00de2591b4c,
        lo: 0xe96ab34ce0bccd10,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0xdb56446d6ad8df09,
        lo: 0x28112e35a60e636f,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0xd273b2058de1bd4b,
        lo: 0x36bbf837b4d320c6,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0xc99af2eaca4c457b,
        lo: 0xeaf51f66692844b2,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0xc0cbf17a071f80e9,
        lo: 0x396ffdf76a147cc2,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0xb8069857560707a7,
        lo: 0xa677b4c8bec22e0,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0xaf4ad26cbc8e5bef,
        lo: 0x9e8b8b88a14ff0c9,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0xa6988ae903f562f1,
        lo: 0x7e858f08597b3a68,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0x9defad3e8f732186,
        lo: 0x476d3b5b45f6ca02,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0x9550252238bd2468,
        lo: 0x658e5a0b811c596d,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0x8cb9de8a32ab3694,
        lo: 0x97c9859530a4514c,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0x842cc5acf1d0344c,
        lo: 0x1fecdfa819b96094,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0xf7518e0035c3dd92,
        lo: 0x606d89093278a931,
        ex: -5,
        sgn: 0x1,
    },
    D {
        hi: 0xe65b9e6eed965c4f,
        lo: 0x609f5fe2058d5ff2,
        ex: -5,
        sgn: 0x1,
    },
    D {
        hi: 0xd5779687d887e0ee,
        lo: 0x49dda17056e45ebb,
        ex: -5,
        sgn: 0x1,
    },
    D {
        hi: 0xc4a550a4fd9a19bb,
        lo: 0x3e97660a23cc5402,
        ex: -5,
        sgn: 0x1,
    },
    D {
        hi: 0xb3e4a796a5dac213,
        lo: 0x7cca0bcc06c2f8e,
        ex: -5,
        sgn: 0x1,
    },
    D {
        hi: 0xa33576a16f1f4c79,
        lo: 0x121016bd904dc95a,
        ex: -5,
        sgn: 0x1,
    },
    D {
        hi: 0x9297997c68c1f4e6,
        lo: 0x610db3d4dd423bc9,
        ex: -5,
        sgn: 0x1,
    },
    D {
        hi: 0x820aec4f3a222397,
        lo: 0xb9e3aea6c444eef6,
        ex: -5,
        sgn: 0x1,
    },
    D {
        hi: 0xe31e9760a5578c6d,
        lo: 0xf9eb2f284f31c35a,
        ex: -6,
        sgn: 0x1,
    },
    D {
        hi: 0xc24929464655f482,
        lo: 0xda5f3cc0b3251da6,
        ex: -6,
        sgn: 0x1,
    },
    D {
        hi: 0xa195492cc0660519,
        lo: 0x4a18dff7cdb4ae33,
        ex: -6,
        sgn: 0x1,
    },
    D {
        hi: 0x8102b2c49ac23a86,
        lo: 0x91d082dce3ddcd08,
        ex: -6,
        sgn: 0x1,
    },
    D {
        hi: 0xc122451c45155150,
        lo: 0xb16137f09a002b0e,
        ex: -7,
        sgn: 0x1,
    },
    D {
        hi: 0x8080abac46f389c4,
        lo: 0x662d417ced0079c9,
        ex: -7,
        sgn: 0x1,
    },
    D {
        hi: 0x0,
        lo: 0x0,
        ex: 127,
        sgn: 0x0,
    },
    D {
        hi: 0x0,
        lo: 0x0,
        ex: 127,
        sgn: 0x0,
    },
    D {
        hi: 0xff805515885e014e,
        lo: 0x435ab4da6a5bb50f,
        ex: -9,
        sgn: 0x0,
    },
    D {
        hi: 0xff015358833c4762,
        lo: 0xbb481c8ee1416999,
        ex: -8,
        sgn: 0x0,
    },
    D {
        hi: 0xbee23afc0853b6a8,
        lo: 0xa89782c20df350c2,
        ex: -7,
        sgn: 0x0,
    },
    D {
        hi: 0xfe054587e01f1e2b,
        lo: 0xf6d3a69bd5eab72f,
        ex: -7,
        sgn: 0x0,
    },
    D {
        hi: 0x9e75221a352ba751,
        lo: 0x452b7ea62f2198ea,
        ex: -6,
        sgn: 0x0,
    },
    D {
        hi: 0xbdc8d83ead88d518,
        lo: 0x7faa638b5e00ee90,
        ex: -6,
        sgn: 0x0,
    },
    D {
        hi: 0xdcfe013d7c8cbfc5,
        lo: 0x632dbac46f30d009,
        ex: -6,
        sgn: 0x0,
    },
    D {
        hi: 0xfc14d873c1980236,
        lo: 0xc7e09e3de453f5fc,
        ex: -6,
        sgn: 0x0,
    },
    D {
        hi: 0x8d86cc491ecbfe03,
        lo: 0xf1776453b7e82558,
        ex: -5,
        sgn: 0x0,
    },
    D {
        hi: 0x9cf43dcff5eafd2f,
        lo: 0x2ad90155c8a7236a,
        ex: -5,
        sgn: 0x0,
    },
    D {
        hi: 0xac52dd7e4726a456,
        lo: 0xa47a963a91bb3018,
        ex: -5,
        sgn: 0x0,
    },
    D {
        hi: 0xbba2c7b196e7e224,
        lo: 0xe7950f7252c163cf,
        ex: -5,
        sgn: 0x0,
    },
    D {
        hi: 0xcae41876471f5bde,
        lo: 0x91d00a417e330f8e,
        ex: -5,
        sgn: 0x0,
    },
    D {
        hi: 0xda16eb88cb8df5fb,
        lo: 0x28a63ecfb66e94c0,
        ex: -5,
        sgn: 0x0,
    },
    D {
        hi: 0xe93b5c56d85a9083,
        lo: 0xce2992bfea38e76b,
        ex: -5,
        sgn: 0x0,
    },
    D {
        hi: 0xf85186008b1532f9,
        lo: 0xe64b8b7759978998,
        ex: -5,
        sgn: 0x0,
    },
    D {
        hi: 0x83acc1acc7238978,
        lo: 0x5a5333c45b7f442e,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0x8b29b7751bd7073b,
        lo: 0x2e0b9ee992f2372,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0x929fb17850a0b7be,
        lo: 0x5b4d3807660516a4,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0x9a0ebcb0de8e848e,
        lo: 0x2c1bb082689ba814,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xa176e5f5323781d2,
        lo: 0xdcf935996c92e8d4,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xa8d839f830c1fb40,
        lo: 0x4c7343517c8ac264,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xb032c549ba861d83,
        lo: 0x774e27bc92ce3373,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xb78694572b5a5cd3,
        lo: 0x24cdcf68cdb2067c,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xbed3b36bd8966419,
        lo: 0x7c0644d7d9ed08b4,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xc61a2eb18cd907a1,
        lo: 0xe5a1532f6d5a1ac1,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xcd5a1231019d66d7,
        lo: 0x761e3e7b171e44b2,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xd49369d256ab1b1f,
        lo: 0x9e9154e1d5263cda,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xdbc6415d876d0839,
        lo: 0x3e33c0c9f8824f54,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xe2f2a47ade3a18a8,
        lo: 0xa0bf7c0b0d8bb4ef,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xea189eb3659aeaeb,
        lo: 0x93b2a3b21f448259,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xf1383b7157972f48,
        lo: 0x543fff0ff4f0aaf1,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xf85186008b153302,
        lo: 0x5e4b8b7759978993,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xff64898edf55d548,
        lo: 0x428ccfc99271dffa,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0x8338a89652cb714a,
        lo: 0xb247eb86498c2ce7,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0x86bbf3e68472cb2f,
        lo: 0xb8bd20615747126,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0x8a3c2c233a156341,
        lo: 0x9027c74fe0e6f64f,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0x8db956a97b3d0143,
        lo: 0xf023472cd739f9e1,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0x913378c852d65be6,
        lo: 0x977e3013d10f7525,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0x94aa97c0ffa91a5d,
        lo: 0x4ee3880fb7d34429,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0x981eb8c723fe97f2,
        lo: 0x1f1c134fb702d433,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0x9b8fe100f47ba1d8,
        lo: 0x4b62af189fcba0d,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0x9efe158766314e4f,
        lo: 0x4d71827efe892fc8,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xa2695b665be8f338,
        lo: 0x4eca87c3f0f06211,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xa5d1b79cd2af2aca,
        lo: 0x8837986ceabfbed6,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xa9372f1d0da1bd10,
        lo: 0x580eb71e58cd36e5,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xac99c6ccc1042e94,
        lo: 0x3dd557528315838d,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xaff983853c9e9e40,
        lo: 0x5f105039091dd7f5,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xb3566a13956a86f4,
        lo: 0x471b1e1574d9fd55,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xb6b07f38ce90e463,
        lo: 0x7bb2e265d0de37e1,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xba07c7aa01bd2648,
        lo: 0x43f9d57b324bd05f,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xbd5c481086c848db,
        lo: 0xbb596b5030403242,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xc0ae050a1abf56ad,
        lo: 0x2f7f8c5fa9c50d76,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xc3fd03290648847d,
        lo: 0x30480bee4cbbd698,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xc74946f4436a054e,
        lo: 0xf4f5cb531201c0d3,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xca92d4e7a2b5a3ad,
        lo: 0xc983a9c5c4b3b135,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xcdd9b173efdc1aaa,
        lo: 0x8863e007c184a1e7,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xd11de0ff15ab18c6,
        lo: 0xd88d83d4cc613f21,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xd45f67e44178c612,
        lo: 0x5486e73c615158b4,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xd79e4a7405ff96c3,
        lo: 0x1300c9be67ae5da0,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xdada8cf47dad236d,
        lo: 0xdffb833c3409ee7e,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xde1433a16c66b14c,
        lo: 0xde744870f54f0f18,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xe14b42ac60c60512,
        lo: 0x4e38eb8092a01f06,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xe47fbe3cd4d10d5b,
        lo: 0x2ec0f797fdcd125c,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xe7b1aa704e2ee240,
        lo: 0xb40faab6d2ad0841,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xeae10b5a7ddc8ad8,
        lo: 0x806b2fc9a8038790,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xee0de5055f63eb01,
        lo: 0x90a33316df83ba5a,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xf1383b7157972f4a,
        lo: 0xb43fff0ff4f0aaf1,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xf460129552d2ff41,
        lo: 0xe62e3201bb2bbdce,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xf7856e5ee2c9b28a,
        lo: 0x76f2a1b84190a7dc,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xfaa852b25bd9b833,
        lo: 0xa6dbfa03186e0666,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xfdc8c36af1f15468,
        lo: 0xa3361bca696504a,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0x8073622d6a80e631,
        lo: 0xe897009015316073,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x82012ca5a68206d5,
        lo: 0x8fde85afdd2bc88a,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x838dc2fe6ac868e7,
        lo: 0x1a3fcbdef40100cb,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x851927139c871af8,
        lo: 0x67bd00c38061c51f,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x86a35abcd5ba5901,
        lo: 0x5481c3cbd925ccd2,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x882c5fcd7256a8c1,
        lo: 0x39055a6598e7c29e,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x89b438149d4582f5,
        lo: 0x34531dba493eb5a6,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x8b3ae55d5d30701a,
        lo: 0xc63eab8837170480,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x8cc0696ea11b7b36,
        lo: 0x94361c9a28d38a6a,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x8e44c60b4ccfd7dc,
        lo: 0x1473aa01c7778679,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x8fc7fcf24517946a,
        lo: 0x380cbe769f2c6793,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x914a0fde7bcb2d0e,
        lo: 0xc429ed3aea197a60,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x92cb0086fbb1cf75,
        lo: 0xa29d47c50b1182d0,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x944ad09ef4351af1,
        lo: 0xa49827e081cb16ba,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x95c981d5c4e924ea,
        lo: 0x45404f5aa577d6b4,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x974715d708e984dd,
        lo: 0x6648d42840d9e6fb,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x98c38e4aa20c27d2,
        lo: 0x846767ec990d7333,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x9a3eecd4c3eaa6ae,
        lo: 0xdb3a7f6e6087b947,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x9bb93315fec2d790,
        lo: 0x7f589fba0865790f,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x9d3262ab4a2f4e37,
        lo: 0xa1ae6ba06846fae0,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0x9eaa7d2e0fb87c35,
        lo: 0xff472bc6ce648a7d,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xa0218434353f1de4,
        lo: 0xd493efa632530acc,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xa197795027409daa,
        lo: 0x1dd1d4a6df960357,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xa30c5e10e2f613e4,
        lo: 0x9bd9bd99e39a20b3,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xa4803402004e865c,
        lo: 0x31cbe0e8824116cd,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xa5f2fcabbbc506d8,
        lo: 0x68ca4fb7ec323d74,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xa764b99300134d79,
        lo: 0xd04d10474301862,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xa8d56c396fc1684c,
        lo: 0x1eb067d578c4756,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xaa45161d6e93167b,
        lo: 0x9b081cf72249f5b2,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xabb3b8ba2ad362a1,
        lo: 0x1db6506cc17a01f5,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xad215587a67f0cdf,
        lo: 0xe890422cb86b7cb1,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xae8dedfac04e5282,
        lo: 0xac707b8ffc22b3e8,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xaff983853c9e9e3f,
        lo: 0xc5105039091dd7f8,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xb1641795ce3ca978,
        lo: 0xfaf915300e517393,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xb2cdab981f0f940b,
        lo: 0xc857c77dc1df600f,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xb43640f4d8a5761f,
        lo: 0xf5f080a71c34b25d,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xb59dd911aca1ec48,
        lo: 0x1d2664cf09a0c1bf,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xb70475515d0f1c5e,
        lo: 0x4c98c6b8be17818d,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xb86a1713c491aeaa,
        lo: 0xd37ee2872a6f1cd6,
        ex: -2,
        sgn: 0x0,
    },
];
const P_2: [D; 13] = [
    D {
        hi: 0x99df88a0430813ca,
        lo: 0xa1cffb6e966a70f6,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xaaa02d43f696c3e4,
        lo: 0x4dbe754667b6bc48,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0xba2e7a1eaf856174,
        lo: 0x70e5c5a5ebbe0226,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xccccccb9ec017492,
        lo: 0xf934e28d924e76d4,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0xe38e38e3807cfa4b,
        lo: 0xc976e6cbd22e203f,
        ex: -4,
        sgn: 0x0,
    },
    D {
        hi: 0xfffffffffff924cc,
        lo: 0x5b308e39fa7dfb5,
        ex: -4,
        sgn: 0x1,
    },
    D {
        hi: 0x924924924924911d,
        lo: 0x862bc3d33abb3649,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xaaaaaaaaaaaaaaaa,
        lo: 0x6637fd4b19743eec,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xcccccccccccccccc,
        lo: 0xccc2ca18b08fe343,
        ex: -3,
        sgn: 0x0,
    },
    D {
        hi: 0xffffffffffffffff,
        lo: 0xffffff2245823ae0,
        ex: -3,
        sgn: 0x1,
    },
    D {
        hi: 0xaaaaaaaaaaaaaaaa,
        lo: 0xaaaaaaaaa5c48b54,
        ex: -2,
        sgn: 0x0,
    },
    D {
        hi: 0xffffffffffffffff,
        lo: 0xffffffffffffebd8,
        ex: -2,
        sgn: 0x1,
    },
    D {
        hi: 0x8000000000000000,
        lo: 0x0,
        ex: 0,
        sgn: 0x0,
    },
];

pub(super) fn accurate(x: f64) -> f64 {
    accurate_base(x, false)
}
pub(super) fn accurate10(x: f64) -> f64 {
    accurate_base(x, true)
}
fn accurate_base(x: f64, base10: bool) -> f64 {
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i64;
    let hi = (bits & 0xfffffffffffff) + if e != 0 { 1 << 52 } else { 0 };
    let t = hi.leading_zeros();
    let mut xx = D {
        hi: hi << t,
        lo: 0,
        ex: e - 1023 - if t > 11 { t as i64 - 12 } else { 0 },
        sgn: 0,
    };
    let mut e = xx.ex;
    let mut i = (xx.hi >> 55) as usize;
    if xx.hi > 0xb504f333f9de6484 {
        e += 1;
        i >>= 1;
    }
    xx.ex -= e;
    let z = M_ONE.add(xx.mul(_INVERSE_2[i - 128]));
    let mut p = P_2[0];
    for c in &P_2[1..] {
        p = c.add(z.mul(p));
    }
    p = z.mul(p);
    let result = _LOG_INV_2[i - 128].add(p).add(LOG2.muli(e));
    let result = if base10 {
        D {
            hi: 0xde5bd8a937287195,
            lo: 0x355baaafad33dc32,
            ex: -2,
            sgn: 0,
        }
        .mul(result)
    } else {
        result
    };
    let mut r = f64::from_bits((result.hi >> 11) | (0x3ff << 52) | (result.sgn << 63));
    let rd = f64::from_bits(0x3c90000000000000)
        + if (result.hi >> 10) & 1 != 0 {
            f64::from_bits(0x3ca0000000000000)
        } else {
            0.0
        };
    r += if result.sgn == 0 { rd } else { -rd };
    r * f64::from_bits((((result.ex + 1023) & 0x7ff) as u64) << 52)
}

#![forbid(unsafe_code)]
use archmage::prelude::*;
pub mod transfer;
use zenmath::{PowBehavior as B, PowConsistency as C, PowRange as R};
pub const CUTOFF: f32 = 0.003041282560127521_f64 as f32;
pub const DECODE_CUTOFF: f32 = (12.92_f64 * 0.003041282560127521_f64) as f32;
pub const A: f32 = 0.0550107189475866_f64 as f32;
pub const A1: f32 = 1.0550107189475866_f64 as f32;
pub const MID: u64 = B::MIDP.to_u64();
pub const LOW: u64 = B::LOWP.to_u64();
pub const MID_SEP: u64 = B::MIDP.consistency(C::PortableSeparate).to_u64();
pub const LOW_SEP: u64 = B::LOWP.consistency(C::PortableSeparate).to_u64();
pub const MID_PRE: u64 = B::MIDP.range(R::PositiveNormalPrecondition).to_u64();
pub const LOW_PRE: u64 = B::LOWP.range(R::PositiveNormalPrecondition).to_u64();
pub const MID_PRE_SEP: u64 = B::MIDP
    .consistency(C::PortableSeparate)
    .range(R::PositiveNormalPrecondition)
    .to_u64();
pub const LOW_PRE_SEP: u64 = B::LOWP
    .consistency(C::PortableSeparate)
    .range(R::PositiveNormalPrecondition)
    .to_u64();
pub type Kernel = Box<dyn Fn(&[f32], &[f32], &mut [f32]) + Send + Sync>;
pub struct Case {
    pub name: String,
    pub policy: u64,
    pub run: Kernel,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    Positive,
    Decode,
    Encode,
    Special,
}
impl Domain {
    pub fn name(self) -> &'static str {
        match self {
            Self::Positive => "positive",
            Self::Decode => "gamma-decode",
            Self::Encode => "gamma-encode",
            Self::Special => "special",
        }
    }
    pub fn parse(s: &str) -> Self {
        match s {
            "positive" => Self::Positive,
            "gamma-decode" => Self::Decode,
            "gamma-encode" => Self::Encode,
            "special" => Self::Special,
            _ => panic!("unknown domain {s}"),
        }
    }
}
pub const DOMAINS: [Domain; 4] = [
    Domain::Positive,
    Domain::Decode,
    Domain::Encode,
    Domain::Special,
];
/// The log corpus is uniform in positive f32 bit codes (exponent/mantissa),
/// giving logarithmically distributed magnitudes without a host libm generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Distribution {
    Uniform,
    Log,
    Boundary,
    Historical,
}
impl Distribution {
    pub fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Log => "log-bits",
            Self::Boundary => "boundary",
            Self::Historical => "historical",
        }
    }
}
pub fn distributions(d: Domain) -> &'static [Distribution] {
    match d {
        Domain::Decode | Domain::Encode => &[
            Distribution::Uniform,
            Distribution::Log,
            Distribution::Boundary,
        ],
        Domain::Positive => &[Distribution::Uniform, Distribution::Historical],
        Domain::Special => &[Distribution::Uniform],
    }
}
pub fn distributed_inputs(d: Domain, distribution: Distribution, n: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut xs, mut ys) = inputs(d, n);
    if distribution == Distribution::Uniform {
        return (xs, ys);
    }
    let low = match d {
        Domain::Decode => (DECODE_CUTOFF + A) / A1,
        Domain::Encode => CUTOFF,
        _ => 0.01,
    };
    let mut state = 0x4397d059u32;
    for (i, x) in xs.iter_mut().enumerate() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *x = match distribution {
            Distribution::Uniform => unreachable!(),
            Distribution::Log => {
                f32::from_bits(low.to_bits() + state % (1.0_f32.to_bits() - low.to_bits() + 1))
            }
            Distribution::Boundary => {
                let b = [
                    low.to_bits() - 1,
                    low.to_bits(),
                    low.to_bits() + 1,
                    0.5_f32.to_bits() - 1,
                    0.5_f32.to_bits(),
                    0.5_f32.to_bits() + 1,
                    1.0_f32.to_bits() - 1,
                    1.0_f32.to_bits(),
                    1.0_f32.to_bits() + 1,
                ];
                f32::from_bits(b[i % b.len()])
            }
            Distribution::Historical => 0.01 + 4.0 * ((state >> 8) as f32 / 16777216.0),
        };
    }
    if distribution == Distribution::Historical {
        ys.fill(2.4);
    }
    (xs, ys)
}
pub fn inputs(d: Domain, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut state = 0x4397d059u32;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let u = (state >> 8) as f32 / 16777216.0;
        let (x, y) = match d {
            Domain::Positive => (
                f32::from_bits(0x21800000 + (state % (0x5d800000 - 0x21800000))),
                -2.0 + 4.0 * u,
            ),
            Domain::Decode => {
                let c = (12.92_f64 * 0.003041282560127521) as f32
                    + (1.0 - (12.92_f64 * 0.003041282560127521) as f32) * u;
                ((c + A) / A1, 2.4)
            }
            Domain::Encode => (CUTOFF + (1.0 - CUTOFF) * u, (1.0_f64 / 2.4) as f32),
            Domain::Special => (
                [
                    -4.0,
                    -2.0,
                    -1.0,
                    -0.0,
                    0.0,
                    f32::from_bits(1),
                    f32::INFINITY,
                    f32::NAN,
                ][i % 8],
                [-3.0, 2.0, 0.5, 2.4, 0.0, 1.0, f32::INFINITY][(i / 8) % 7],
            ),
        };
        xs.push(x);
        ys.push(y);
    }
    (xs, ys)
}
// KIND: 0 ordinary policy, 1 independent non-dispatch policy, 2 std control,
// 3 ordinary correct powf. Y==0 means per-value exponents; otherwise exact bits.
#[inline(never)]
pub fn scalar<const P: u64, const KIND: u8, const Y: u32>(xs: &[f32], ys: &[f32], out: &mut [f32]) {
    assert_eq!(xs.len(), ys.len());
    assert_eq!(xs.len(), out.len());
    for i in 0..xs.len() {
        let y = if Y == 0 { ys[i] } else { f32::from_bits(Y) };
        out[i] = match KIND {
            0 => zenmath::scalar::powf_with::<P>(xs[i], y),
            1 => zenmath::scalar::reference::powf_with::<P>(xs[i], y),
            2 => xs[i].powf(y),
            3 => zenmath::scalar::powf(xs[i], y),
            4 => oldzenmath::scalar::powf_with::<P>(xs[i], y),
            5 => oldzenmath::scalar::powf(xs[i], y),
            _ => unreachable!(),
        };
    }
}
macro_rules! packed_body {
    ($name:ident,$tier:ident,$proof:ident,$lib:ident,$call:ident) => {
        #[archmage::attune($tier, $proof, inline(never))]
        fn $name<const P: u64, const Y: u32>(xs: &[f32], ys: &[f32], out: &mut [f32]) {
            assert_eq!(xs.len(), ys.len());
            assert_eq!(xs.len(), out.len());
            let mut i = 0;
            while i < xs.len() {
                let len = (xs.len() - i).min(8);
                let mut x = [1.0; 8];
                let mut y = [0.0; 8];
                x[..len].copy_from_slice(&xs[i..i + len]);
                if Y == 0 {
                    y[..len].copy_from_slice(&ys[i..i + len]);
                } else {
                    y[..len].fill(f32::from_bits(Y));
                }
                let r = $lib::portable::array::$call::<P>(x, y);
                out[i..i + len].copy_from_slice(&r[..len]);
                i += len;
            }
        }
    };
}
packed_body!(packed, _v3, _v3_t, magemath, pow_f32x8_with_v3);
packed_body!(packed_old, _v3, _v3_t, oldmagemath, pow_f32x8_with_v3);
packed_body!(packed, _neon, _neon_t, magemath, pow_f32x8_with_neon);
packed_body!(packed_old, _neon, _neon_t, oldmagemath, pow_f32x8_with_neon);
packed_body!(
    packed,
    _wasm128,
    _wasm128_t,
    magemath,
    pow_f32x8_with_wasm128
);
packed_body!(
    packed_old,
    _wasm128,
    _wasm128_t,
    oldmagemath,
    pow_f32x8_with_wasm128
);

macro_rules! context_body {
    ($tier:ident,$proof:ident,$func:ident) => {
        #[archmage::attune($tier, $proof, inline(never))]
        fn scalar_context<const KIND: u8, const Y: u32>(xs: &[f32], ys: &[f32], out: &mut [f32]) {
            assert_eq!(xs.len(), ys.len());
            assert_eq!(xs.len(), out.len());
            for i in 0..xs.len() {
                let y = if Y == 0 { ys[i] } else { f32::from_bits(Y) };
                out[i] = match KIND {
                    0 => zenmath::simd::scalar::$func(xs[i], y),
                    1 => oldzenmath::simd::scalar::$func(xs[i], y),
                    2 => xs[i].powf(y),
                    _ => unreachable!(),
                };
            }
        }
    };
}
context_body!(_v3, _v3_t, powf_v3);
context_body!(_neon, _neon_t, powf_neon);
context_body!(_wasm128, _wasm128_t, powf_wasm128);
fn add<const P: u64, const Y: u32>(v: &mut Vec<Case>, label: &str) {
    v.push(Case {
        name: format!("reference/{label}"),
        policy: P,
        run: Box::new(scalar::<P, 1, Y>),
    });
    v.push(Case {
        name: format!("old-scalar/{label}"),
        policy: P,
        run: Box::new(scalar::<P, 4, Y>),
    });
    v.push(Case {
        name: format!("current/{label}"),
        policy: P,
        run: Box::new(scalar::<P, 0, Y>),
    });
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            name: format!("old-packed-v3/{label}"),
            policy: P,
            run: Box::new(move |x, y, o| packed_old_v3_t::<P, Y>(t, x, y, o)),
        });
        v.push(Case {
            name: format!("packed-v3/{label}"),
            policy: P,
            run: Box::new(move |x, y, o| packed_v3_t::<P, Y>(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            name: format!("old-packed-neon/{label}"),
            policy: P,
            run: Box::new(move |x, y, o| packed_old_neon_t::<P, Y>(t, x, y, o)),
        });
        v.push(Case {
            name: format!("packed-neon/{label}"),
            policy: P,
            run: Box::new(move |x, y, o| packed_neon_t::<P, Y>(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            name: format!("old-packed-wasm128/{label}"),
            policy: P,
            run: Box::new(move |x, y, o| packed_old_wasm128_t::<P, Y>(t, x, y, o)),
        });
        v.push(Case {
            name: format!("packed-wasm128/{label}"),
            policy: P,
            run: Box::new(move |x, y, o| packed_wasm128_t::<P, Y>(t, x, y, o)),
        });
    }
}
pub fn cases<const Y: u32>(d: Domain) -> Vec<Case> {
    let mut v = Vec::new();
    add::<0, Y>(&mut v, "correct");
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            name: "direct-v3/correct".into(),
            policy: 0,
            run: Box::new(move |x, y, o| scalar_context_v3_t::<0, Y>(t, x, y, o)),
        });
        v.push(Case {
            name: "old-direct-v3/correct".into(),
            policy: 0,
            run: Box::new(move |x, y, o| scalar_context_v3_t::<1, Y>(t, x, y, o)),
        });
        v.push(Case {
            name: "std-context-v3/control".into(),
            policy: 0,
            run: Box::new(move |x, y, o| scalar_context_v3_t::<2, Y>(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            name: "direct-neon/correct".into(),
            policy: 0,
            run: Box::new(move |x, y, o| scalar_context_neon_t::<0, Y>(t, x, y, o)),
        });
        v.push(Case {
            name: "old-direct-neon/correct".into(),
            policy: 0,
            run: Box::new(move |x, y, o| scalar_context_neon_t::<1, Y>(t, x, y, o)),
        });
        v.push(Case {
            name: "std-context-neon/control".into(),
            policy: 0,
            run: Box::new(move |x, y, o| scalar_context_neon_t::<2, Y>(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            name: "direct-wasm128/correct".into(),
            policy: 0,
            run: Box::new(move |x, y, o| scalar_context_wasm128_t::<0, Y>(t, x, y, o)),
        });
        v.push(Case {
            name: "old-direct-wasm128/correct".into(),
            policy: 0,
            run: Box::new(move |x, y, o| scalar_context_wasm128_t::<1, Y>(t, x, y, o)),
        });
        v.push(Case {
            name: "std-context-wasm128/control".into(),
            policy: 0,
            run: Box::new(move |x, y, o| scalar_context_wasm128_t::<2, Y>(t, x, y, o)),
        });
    }
    v.push(Case {
        name: "old-ordinary/correct".into(),
        policy: 0,
        run: Box::new(scalar::<0, 5, Y>),
    });
    v.push(Case {
        name: "ordinary/correct".into(),
        policy: 0,
        run: Box::new(scalar::<0, 3, Y>),
    });
    v.push(Case {
        name: "std/control".into(),
        policy: 0,
        run: Box::new(scalar::<0, 2, Y>),
    });
    add::<MID, Y>(&mut v, "midp-full-fused");
    add::<LOW, Y>(&mut v, "lowp-full-fused");
    add::<MID_SEP, Y>(&mut v, "midp-full-separate");
    add::<LOW_SEP, Y>(&mut v, "lowp-full-separate");
    if d != Domain::Special {
        add::<MID_PRE, Y>(&mut v, "midp-pre-fused");
        add::<LOW_PRE, Y>(&mut v, "lowp-pre-fused");
        add::<MID_PRE_SEP, Y>(&mut v, "midp-pre-separate");
        add::<LOW_PRE_SEP, Y>(&mut v, "lowp-pre-separate");
    }
    v
}
pub fn domain_cases(d: Domain, uniform: bool) -> Vec<Case> {
    if uniform {
        match d {
            Domain::Decode => cases::<{ 2.4_f32.to_bits() }>(d),
            Domain::Encode => cases::<{ ((1.0_f64 / 2.4) as f32).to_bits() }>(d),
            _ => panic!("uniform mode requires gamma exponent"),
        }
    } else {
        cases::<0>(d)
    }
}
pub fn inventory() {
    let tiers = [
        ("scalar", true),
        ("v3", X64V3Token::summon().is_some()),
        ("neon", NeonToken::summon().is_some()),
        ("wasm128", Wasm128Token::summon().is_some()),
    ];
    for (name, available) in tiers {
        println!("TIER {name} supported={available}");
    }
    let required = std::env::var("POW_EXPECT_TIER").expect("caller must set POW_EXPECT_TIER");
    assert!(
        tiers.iter().any(|(name, ok)| *name == required && *ok),
        "required study tier unavailable: {required}"
    );
}
pub fn verify() {
    inventory();
    for d in DOMAINS {
        for &distribution in distributions(d) {
            for uniform in [false, true] {
                if uniform && !matches!(d, Domain::Decode | Domain::Encode) {
                    continue;
                }
                for n in [1, 7, 16, 1024] {
                    let (x, y) = distributed_inputs(d, distribution, n);
                    for c in domain_cases(d, uniform) {
                        if c.name == "std/control" || c.name.starts_with("std-context-") {
                            continue;
                        }
                        let mut got = vec![0.0; n];
                        (c.run)(&x, &y, &mut got);
                        for i in 0..n {
                            let want = reference(c.policy, x[i], y[i]);
                            assert!(
                                got[i].to_bits() == want.to_bits()
                                    || got[i].is_nan() && want.is_nan(),
                                "{} {:?} n={n} lane={i} {:08x}^{:08x}: got{:08x} expected{:08x}",
                                c.name,
                                d,
                                x[i].to_bits(),
                                y[i].to_bits(),
                                got[i].to_bits(),
                                want.to_bits()
                            );
                        }
                    }
                }
            }
        }
    }
}
fn reference(p: u64, x: f32, y: f32) -> f32 {
    match p {
        0 => oldzenmath::scalar::reference::powf_with::<0>(x, y),
        MID => oldzenmath::scalar::reference::powf_with::<MID>(x, y),
        LOW => oldzenmath::scalar::reference::powf_with::<LOW>(x, y),
        MID_SEP => oldzenmath::scalar::reference::powf_with::<MID_SEP>(x, y),
        LOW_SEP => oldzenmath::scalar::reference::powf_with::<LOW_SEP>(x, y),
        MID_PRE => oldzenmath::scalar::reference::powf_with::<MID_PRE>(x, y),
        LOW_PRE => oldzenmath::scalar::reference::powf_with::<LOW_PRE>(x, y),
        MID_PRE_SEP => oldzenmath::scalar::reference::powf_with::<MID_PRE_SEP>(x, y),
        LOW_PRE_SEP => oldzenmath::scalar::reference::powf_with::<LOW_PRE_SEP>(x, y),
        _ => unreachable!(),
    }
}
#[test]
fn adapters() {
    verify();
}

#[test]
fn common_entry_guard_edges_match_frozen_correct_kernel() {
    let bases = [
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x00800001, 0x3f7fffff, 0x3f800000,
        0x3f800001, 0xbf800000, 0xc0000000, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc00123,
        0x7f800001,
    ];
    let exponents = [
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x3effffff, 0x3f000000, 0x3f7fffff,
        0x3f800000, 0x3f800001, 0xbf800000, 0x40000000, 0x40400000, 0xc0400000, 0x7f7fffff,
        0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00123, 0x7f800001,
    ];
    for xb in bases {
        for yb in exponents {
            let x = f32::from_bits(xb);
            let y = f32::from_bits(yb);
            let want = oldzenmath::scalar::reference::powf(x, y);
            for c in cases::<0>(Domain::Special)
                .into_iter()
                .filter(|c| c.policy == 0 && !c.name.starts_with("std"))
            {
                let mut got = [0.0];
                (c.run)(&[x], &[y], &mut got);
                assert!(
                    got[0].to_bits() == want.to_bits() || got[0].is_nan() && want.is_nan(),
                    "{}: {xb:08x}^{yb:08x}: got{:08x} expected{:08x}",
                    c.name,
                    got[0].to_bits(),
                    want.to_bits()
                );
            }
        }
    }
}

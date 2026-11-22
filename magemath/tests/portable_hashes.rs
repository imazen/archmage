//! Cross-backend, cross-architecture pin of BitExact result bits.
use archmage::ScalarToken;
use magemath::{
    SimdMath,
    cbrtf::{Cbrtf, CbrtfOptions},
    expf::{Expf, ExpfOptions},
    logf::{Logf, LogfOptions},
    powf::{Powf, PowfOptions},
};
use magetypes::simd::generic::f32x8;

const GM: u128 = PowfOptions::new().gamma().midp().bits();
const GL: u128 = PowfOptions::new().gamma().lowp().bits();
const GH: u128 = PowfOptions::new().gamma().highp().bits();
const FH: u128 = PowfOptions::new().full_domain().highp().bits();
const NM: u128 = PowfOptions::new().midp().bits();
const EM: u128 = ExpfOptions::new().midp().bits();
const EL: u128 = ExpfOptions::new().lowp().bits();
const LM: u128 = LogfOptions::new().midp().bits();
const LL: u128 = LogfOptions::new().lowp().bits();
const CM: u128 = CbrtfOptions::new().full_domain().midp().bits();
const CL: u128 = CbrtfOptions::new().full_domain().lowp().bits();

static NONE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn mix(h: &mut u64, r: Option<[f32; 8]>) {
    match r {
        None => {
            *h = (*h ^ 0xdead).wrapping_mul(0x100000001b3);
            NONE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Some(a) => {
            for v in a {
                let b = if v.is_nan() { 0x7fc0_0000 } else { v.to_bits() };
                *h = (*h ^ b as u64).wrapping_mul(0x100000001b3);
            }
        }
    }
}
const NAMES: [&str; 11] = [
    "pow gamma midp",
    "pow gamma lowp",
    "pow gamma highp",
    "pow full highp",
    "pow normal midp",
    "exp2 midp",
    "exp2 lowp",
    "log2 midp",
    "log2 lowp",
    "cbrt midp full",
    "cbrt lowp full",
];

/// Result bits of every BitExact configuration over a fixed input stride, hashed.
/// A change here is a change to portable results: refits, reordered arithmetic
/// or a backend that rounds differently.
#[test]
fn portable_result_hashes() {
    let exps = [
        1.0 / 2.2f32,
        2.2,
        2.4,
        0.5,
        4.0,
        1.0,
        3.0,
        0.0,
        0.123,
        -1.0,
        -2.5,
        7.0,
        f32::INFINITY,
        f32::NAN,
        33.3,
        -0.0,
    ];
    let (mut ha, mut hs) = ([0xcbf29ce484222325u64; 11], [0xcbf29ce484222325u64; 11]);
    let (mut b, mut it) = (1u32, 0usize);
    loop {
        let xs: [f32; 8] = core::array::from_fn(|i| {
            if i < 6 {
                f32::from_bits(b.wrapping_add(i as u32 * 0x1000_0193))
            } else {
                (b as f32 / 4294967296.0) * 260.0 - 130.0 + i as f32
            }
        });
        let n = exps[it % exps.len()];
        it += 1;
        // In-domain inputs for the restricted presets: [2^-16, 2^16], exponent in [0, 4].
        let ys: [f32; 8] = core::array::from_fn(|i| {
            f32::from_bits(
                0x3780_0000
                    + (b.wrapping_mul(2654435761)
                        .wrapping_add(i as u32 * 0x0101_0193)
                        % 0x1000_0000),
            )
        });
        let g = [
            1.0 / 2.2f32,
            2.2,
            2.4,
            0.5,
            4.0,
            1.0,
            3.0,
            0.0,
            0.123,
            1.0 / 2.4,
        ][it % 10];
        let w = f32x8::from_array_t(ScalarToken, ys);
        let t = ScalarToken;
        let v = f32x8::from_array_t(t, xs);
        let a = |r: Option<f32x8<ScalarToken>>| r.map(|v| v.to_array());
        let auto = [
            Powf::<GM>::array::<8>(ys, g).ok(),
            Powf::<GL>::array::<8>(ys, g).ok(),
            Powf::<GH>::array::<8>(ys, g).ok(),
            Powf::<FH>::array::<8>(xs, n).ok(),
            Powf::<NM>::array::<8>(ys, g).ok(),
            Expf::<EM>::exp2_array::<8>(xs).ok(),
            Expf::<EL>::exp2_array::<8>(xs).ok(),
            Logf::<LM>::log2_array::<8>(ys).ok(),
            Logf::<LL>::log2_array::<8>(ys).ok(),
            Cbrtf::<CM>::array::<8>(xs).ok(),
            Cbrtf::<CL>::array::<8>(xs).ok(),
        ];
        let sc = [
            a(w.try_powf_with::<GM>(g).ok()),
            a(w.try_powf_with::<GL>(g).ok()),
            a(w.try_powf_with::<GH>(g).ok()),
            a(v.try_powf_with::<FH>(n).ok()),
            a(w.try_powf_with::<NM>(g).ok()),
            a(v.try_exp2_with::<EM>().ok()),
            a(v.try_exp2_with::<EL>().ok()),
            a(w.try_log2_with::<LM>().ok()),
            a(w.try_log2_with::<LL>().ok()),
            a(v.try_cbrt_with::<CM>().ok()),
            a(v.try_cbrt_with::<CL>().ok()),
        ];
        for k in 0..11 {
            mix(&mut ha[k], auto[k]);
            mix(&mut hs[k], sc[k]);
        }
        let nb = b.wrapping_add(40009);
        if nb < b {
            break;
        }
        b = nb;
    }
    // Pinned 2026-11-17: identical on x86-64 (AVX-512 automatic and ScalarToken)
    // and AArch64 Neoverse-N1 (NEON automatic and ScalarToken).
    // The final two entries pin the division-free cube-root graph.
    const EXPECTED: [u64; 11] = [
        0xce6e05b0a6121aea,
        0xa218fefe2cd0d6c4,
        0xe4d4d487161a9a50,
        0x9f15f2c1b6e538c4,
        0xce6e05b0a6121aea,
        0x422c3bd93ca2a283,
        0x4fe057004d949a1b,
        0x9b8c1c453d4dc044,
        0xde622032bc96abbb,
        0x76ff602673eecedd,
        0xb3b701b0234d1bc1,
    ];
    assert_eq!(
        NONE.load(std::sync::atomic::Ordering::Relaxed),
        10072,
        "rejected vector count"
    );
    for k in 0..11 {
        assert_eq!(ha[k], EXPECTED[k], "{} automatic: {:#x}", NAMES[k], ha[k]);
        assert_eq!(hs[k], EXPECTED[k], "{} ScalarToken: {:#x}", NAMES[k], hs[k]);
    }
}

//! Cross-backend, cross-architecture pin of BitExact result bits.
use archmage::ScalarToken;
use magemath::{
    SimdMath,
    cbrtf::{Cbrtf, CbrtfOptions},
    expf::{Expf, ExpfOptions},
    log1pf::{Log1pf, Log1pfOptions},
    logf::{Logf, LogfOptions},
    powf::{Powf, PowfOptions},
};
use magetypes::simd::generic::f32x8;

const GM: PowfOptions = PowfOptions::new().gamma().midp();
const GL: PowfOptions = PowfOptions::new().gamma().lowp();
const GH: PowfOptions = PowfOptions::new().gamma().highp();
const FH: PowfOptions = PowfOptions::new().full_domain().highp();
const NM: PowfOptions = PowfOptions::new().midp();
const EM: ExpfOptions = ExpfOptions::new().midp();
const EL: ExpfOptions = ExpfOptions::new().lowp();
const LM: LogfOptions = LogfOptions::new().midp();
const LL: LogfOptions = LogfOptions::new().lowp();
const L1M: Log1pfOptions = Log1pfOptions::new().admit_all_floats().midp();
const L1L: Log1pfOptions = Log1pfOptions::new().admit_all_floats().lowp();
const SM: Log1pfOptions = Log1pfOptions::new()
    .around_zero_range()
    .admit_subnormals()
    .midp();
const SL: Log1pfOptions = Log1pfOptions::new()
    .around_zero_range()
    .admit_subnormals()
    .lowp();
const CM: CbrtfOptions = CbrtfOptions::new().full_domain().midp();
const CL: CbrtfOptions = CbrtfOptions::new().full_domain().lowp();

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
const NAMES: [&str; 15] = [
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
    "log1p midp full",
    "log1p lowp full",
    "log1p midp around zero",
    "log1p lowp around zero",
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
    let (mut ha, mut hs) = (
        [0xcbf29ce484222325u64; NAMES.len()],
        [0xcbf29ce484222325u64; NAMES.len()],
    );
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
        let zs: [f32; 8] = core::array::from_fn(|i| {
            let sample = b
                .wrapping_mul(2654435761)
                .wrapping_add(i as u32 * 0x0101_0193);
            let magnitude = (sample >> 1) % (0.25_f32.to_bits() + 1);
            f32::from_bits(magnitude | (sample & 0x8000_0000))
        });
        let z = f32x8::from_array_t(t, zs);
        let a = |r: Option<f32x8<ScalarToken>>| r.map(|v| v.to_array());
        let auto = [
            Powf::<{ GM.bits() }>::array::<8>(ys, g).ok(),
            Powf::<{ GL.bits() }>::array::<8>(ys, g).ok(),
            Powf::<{ GH.bits() }>::array::<8>(ys, g).ok(),
            Powf::<{ FH.bits() }>::array::<8>(xs, n).ok(),
            Powf::<{ NM.bits() }>::array::<8>(ys, g).ok(),
            Expf::<{ EM.bits() }>::exp2_array::<8>(xs).ok(),
            Expf::<{ EL.bits() }>::exp2_array::<8>(xs).ok(),
            Logf::<{ LM.bits() }>::log2_array::<8>(ys).ok(),
            Logf::<{ LL.bits() }>::log2_array::<8>(ys).ok(),
            Cbrtf::<{ CM.bits() }>::array::<8>(xs).ok(),
            Cbrtf::<{ CL.bits() }>::array::<8>(xs).ok(),
            Log1pf::<{ L1M.bits() }>::array::<8>(xs).ok(),
            Log1pf::<{ L1L.bits() }>::array::<8>(xs).ok(),
            Log1pf::<{ SM.bits() }>::array::<8>(zs).ok(),
            Log1pf::<{ SL.bits() }>::array::<8>(zs).ok(),
        ];
        let sc = [
            a(w.try_powf_with::<{ GM.caller().bits() }>(g).ok()),
            a(w.try_powf_with::<{ GL.caller().bits() }>(g).ok()),
            a(w.try_powf_with::<{ GH.caller().bits() }>(g).ok()),
            a(v.try_powf_with::<{ FH.caller().bits() }>(n).ok()),
            a(w.try_powf_with::<{ NM.caller().bits() }>(g).ok()),
            a(v.try_exp2_with::<{ EM.caller().bits() }>().ok()),
            a(v.try_exp2_with::<{ EL.caller().bits() }>().ok()),
            a(w.try_log2_with::<{ LM.caller().bits() }>().ok()),
            a(w.try_log2_with::<{ LL.caller().bits() }>().ok()),
            a(v.try_cbrt_with::<{ CM.caller().bits() }>().ok()),
            a(v.try_cbrt_with::<{ CL.caller().bits() }>().ok()),
            a(v.try_log1p_with::<{ L1M.caller().bits() }>().ok()),
            a(v.try_log1p_with::<{ L1L.caller().bits() }>().ok()),
            a(z.try_log1p_with::<{ SM.caller().bits() }>().ok()),
            a(z.try_log1p_with::<{ SL.caller().bits() }>().ok()),
        ];
        for k in 0..NAMES.len() {
            mix(&mut ha[k], auto[k]);
            mix(&mut hs[k], sc[k]);
        }
        let nb = b.wrapping_add(40009);
        if nb < b {
            break;
        }
        b = nb;
    }
    const EXPECTED: [u64; NAMES.len()] = [
        0xce6e05b0a6121aea,
        0x0c46bff7aac262ab,
        0xe4d4d487161a9a50,
        0x9f15f2c1b6e538c4,
        0xce6e05b0a6121aea,
        0x422c3bd93ca2a283,
        0xc45fd4b726a49f48,
        0x9b8c1c453d4dc044,
        0xde622032bc96abbb,
        0x76ff602673eecedd,
        0xb3b701b0234d1bc1,
        0x0986ae2eb2104fd2,
        0xc54465e340d85b67,
        0x359a089c4d29fc0e,
        0xbd4e3c1318256aca,
    ];
    assert_eq!(
        NONE.load(std::sync::atomic::Ordering::Relaxed),
        10072,
        "rejected vector count"
    );
    for k in 0..NAMES.len() {
        assert_eq!(ha[k], EXPECTED[k], "{} automatic: {:#x}", NAMES[k], ha[k]);
        assert_eq!(hs[k], EXPECTED[k], "{} ScalarToken: {:#x}", NAMES[k], hs[k]);
    }
}

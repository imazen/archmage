//! WASI guest; linker exports compiler-assigned symbols, without unsafe attrs.
#![forbid(unsafe_code)]
#[cfg(target_arch = "wasm32")]
mod guest {
    use archmage::SimdToken;
    use magetypes::powf::{PowfDomain, PowfExecution, PowfOptions};
    use std::sync::{Mutex, OnceLock};
    #[allow(dead_code)]
    mod soft {
        include!(concat!(env!("OUT_DIR"), "/soft.rs"));
    }
    #[allow(dead_code, unused_variables)]
    mod prevalidated {
        include!(concat!(env!("OUT_DIR"), "/prevalidated.rs"));
    }
    include!(concat!(env!("OUT_DIR"), "/replacement_kernels.rs"));
    #[derive(Default)]
    struct State {
        input1: Vec<f32>,
        output1: Vec<f32>,
        input4: Vec<[f32; 4]>,
        output4: Vec<[f32; 4]>,
        input8: Vec<[f32; 8]>,
        output8: Vec<[f32; 8]>,
        input16: Vec<[f32; 16]>,
        output16: Vec<[f32; 16]>,
        options: PowfOptions,
        exponent: u32,
        dynamic: bool,
    }
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    pub extern "C" fn guest_prepare(
        size: i32,
        distribution: i32,
        domain: i32,
        exponent: i32,
        dynamic: i32,
    ) {
        let mut s = STATE
            .get_or_init(|| Mutex::new(State::default()))
            .lock()
            .unwrap();
        let value = |i: usize| match distribution {
            0 => ((i % 255) + 1) as f32 / 255.0,
            1 => f32::from_bits(0x37800000 + (i as u32).wrapping_mul(0x9e3779b9) % 0x10000001),
            2 => 0.5 + (i % 193) as f32 / 128.0,
            _ => panic!("distribution"),
        };
        s.exponent = exponent as u32;
        s.dynamic = dynamic != 0;
        let n = size as usize;
        s.input1 = (0..n).map(value).collect();
        s.output1 = vec![0.; n];
        s.input4 = (0..n / 4)
            .map(|i| core::array::from_fn(|j| value(i * 4 + j)))
            .collect();
        s.output4 = vec![[0.; 4]; n / 4];
        s.input8 = (0..n / 8)
            .map(|i| core::array::from_fn(|j| value(i * 8 + j)))
            .collect();
        s.output8 = vec![[0.; 8]; n / 8];
        s.input16 = (0..n / 16)
            .map(|i| core::array::from_fn(|j| value(i * 16 + j)))
            .collect();
        s.output16 = vec![[0.; 16]; n / 16];
        s.options = PowfOptions {
            domain: match domain {
                0 => PowfDomain::PositiveNormal,
                1 => PowfDomain::NonNegativeNormal,
                2 => PowfDomain::Gamma,
                _ => panic!("domain"),
            },
            execution: PowfExecution::Caller,
        };
    }
    pub extern "C" fn guest_probe() -> i32 {
        let token = archmage::Wasm128Token::summon().expect("SIMD module");
        let a =
            magetypes::simd::generic::f32x4::splat_t(token, std::hint::black_box(1. + 1. / 4096.));
        let c = magetypes::simd::generic::f32x4::splat_t(
            token,
            std::hint::black_box(-(1. + 1. / 2048.)),
        );
        let bits = a.mul_add_relaxed(a, c).to_array().map(f32::to_bits);
        assert!(bits == [0; 4] || bits == [0x3380_0000; 4]);
        i32::from(bits[0] != 0)
    }
    pub extern "C" fn guest_run(
        width: i32,
        arm: i32,
        chain: i32,
        repeats: i32,
        context: i32,
    ) -> i64 {
        let mut s = STATE.get().unwrap().lock().unwrap();
        let options = s.options;
        for _ in 0..repeats {
            let s = &mut *s;
            let context = if context == 0 { "scalar" } else { "wasm128" };
            let domain = match options.domain {
                PowfDomain::PositiveNormal => 0,
                PowfDomain::NonNegativeNormal => 1,
                PowfDomain::Gamma => 2,
            };
            match width {
                1 => replacement_guest1(
                    arm as usize,
                    context,
                    s.input1.as_chunks::<1>().0,
                    s.output1.as_chunks_mut::<1>().0,
                    chain != 0,
                    domain,
                    s.exponent,
                    s.dynamic,
                ),
                4 => replacement_guest4(
                    arm as usize,
                    context,
                    &s.input4,
                    &mut s.output4,
                    chain != 0,
                    domain,
                    s.exponent,
                    s.dynamic,
                ),
                8 => replacement_guest8(
                    arm as usize,
                    context,
                    &s.input8,
                    &mut s.output8,
                    chain != 0,
                    domain,
                    s.exponent,
                    s.dynamic,
                ),
                16 => replacement_guest16(
                    arm as usize,
                    context,
                    &s.input16,
                    &mut s.output16,
                    chain != 0,
                    domain,
                    s.exponent,
                    s.dynamic,
                ),
                _ => panic!("width"),
            }
        }
        match width {
            1 => s.output1[0],
            4 => s.output4[0][0],
            8 => s.output8[0][0],
            16 => s.output16[0][0],
            _ => unreachable!(),
        }
        .to_bits() as i64
    }
    fn result_bits(width: i32) -> Vec<u32> {
        let s = STATE.get().unwrap().lock().unwrap();
        match width {
            1 => s.output1.iter().map(|x| x.to_bits()).collect(),
            4 => s.output4.iter().flatten().map(|x| x.to_bits()).collect(),
            8 => s.output8.iter().flatten().map(|x| x.to_bits()).collect(),
            16 => s.output16.iter().flatten().map(|x| x.to_bits()).collect(),
            _ => panic!("width"),
        }
    }
    pub extern "C" fn guest_verify(width: i32, chain: i32, context: i32) {
        guest_run(width, 13, chain, 1, context);
        let expected = result_bits(width);
        let integer = {
            let s = STATE.get().unwrap().lock().unwrap();
            [0x40000000, 0x40800000].contains(&s.exponent)
        };
        let mut midp = None;
        let mut lowp = None;
        for arm in 0..if integer { 15 } else { 14 } {
            guest_run(width, arm, chain, 1, context);
            let bits = result_bits(width);
            assert!(bits.iter().all(|&b| f32::from_bits(b).is_finite()));
            if arm == 0 {
                midp = Some(bits.clone());
            }
            if arm == 2 {
                assert_eq!(Some(&bits), midp.as_ref());
            }
            if arm == 3 {
                lowp = Some(bits.clone());
            }
            if arm == 4 {
                assert_eq!(Some(&bits), lowp.as_ref());
            }
            if (7..=13).contains(&arm) {
                assert_eq!(bits, expected);
            }
        }
    }
    fn normalize(x: f32) -> f32 {
        f32::from_bits(0x3f000000 | (x.to_bits() & 0x007fffff))
    }

    pub fn entry() {
        std::hint::black_box((
            guest_prepare as extern "C" fn(i32, i32, i32, i32, i32),
            guest_probe as extern "C" fn() -> i32,
            guest_run as extern "C" fn(i32, i32, i32, i32, i32) -> i64,
            guest_verify as extern "C" fn(i32, i32, i32),
        ));
    }
}
fn main() {
    #[cfg(target_arch = "wasm32")]
    guest::entry();
    #[cfg(not(target_arch = "wasm32"))]
    panic!("build for wasm32-wasip1");
}

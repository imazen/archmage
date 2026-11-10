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
    include!(concat!(env!("OUT_DIR"), "/cost_kernels.rs"));
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
    }
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    pub extern "C" fn guest_prepare(size: i32, distribution: i32, domain: i32) {
        let mut s = STATE
            .get_or_init(|| Mutex::new(State::default()))
            .lock()
            .unwrap();
        let value = |i: usize| {
            if distribution == 0 {
                0.5 + (i % 193) as f32 / 128.0
            } else {
                f32::from_bits(0x3f00_0000 + ((i as u32).wrapping_mul(0x9e37_79b9) % 0x0100_0000))
            }
        };
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
            ..PowfOptions::new()
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
            match width {
                1 => scalar(arm as usize, &s.input1, &mut s.output1, chain != 0, options),
                4 => execute4(
                    arm as usize,
                    context,
                    &s.input4,
                    &mut s.output4,
                    chain != 0,
                    options,
                ),
                8 => execute8(
                    arm as usize,
                    context,
                    &s.input8,
                    &mut s.output8,
                    chain != 0,
                    options,
                ),
                16 => execute16(
                    arm as usize,
                    context,
                    &s.input16,
                    &mut s.output16,
                    chain != 0,
                    options,
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
        guest_run(width, 0, chain, 1, context);
        let expected = result_bits(width);
        for arm in 1..3 {
            guest_run(width, arm, chain, 1, context);
            assert_eq!(result_bits(width), expected);
        }
    }
    fn scalar(arm: usize, input: &[f32], output: &mut [f32], chain: bool, options: PowfOptions) {
        let n = std::hint::black_box(2.4);
        let mut acc = input[0];
        for (&base, out) in input.iter().zip(output) {
            let x = if chain { acc } else { base };
            let y = match arm {
                0 => prevalidated_call4(archmage::ScalarToken, [x; 4], n, options.domain)[0],
                1 => magetypes::powf::try_powf_runtime(
                    x,
                    n,
                    PowfOptions {
                        execution: PowfExecution::Caller,
                        ..options
                    },
                )
                .unwrap(),
                2 => magetypes::powf::try_powf_runtime(
                    x,
                    n,
                    PowfOptions {
                        execution: PowfExecution::Automatic,
                        ..options
                    },
                )
                .unwrap(),
                _ => unreachable!(),
            };
            acc = if chain { y.abs() * 0.125 + 0.5 } else { y };
            *out = acc;
        }
    }

    pub fn entry() {
        std::hint::black_box((
            guest_prepare as extern "C" fn(i32, i32, i32),
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

//! Timer-free WASI guest. Compiler-assigned symbols are exported by the builder.
#![forbid(unsafe_code)]
#[cfg(target_arch = "wasm32")]
#[path = "../rcp14.rs"]
mod rcp14;
#[cfg(target_arch = "wasm32")]
#[path = "../recip_research.rs"]
mod recip_research;
#[cfg(target_arch = "wasm32")]
#[path = "../recip_workload.rs"]
mod recip_workload;
#[cfg(target_arch = "wasm32")]
#[path = "../seed_graphs.rs"]
mod seed_graphs;
#[cfg(target_arch = "wasm32")]
mod guest {
    use crate::recip_workload::*;
    use archmage::SimdToken;
    use std::sync::{Mutex, OnceLock};
    struct State {
        input: [f32; 4096],
        output: [f32; 4096],
        n: usize,
    }
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    pub extern "C" fn guest_prepare(n: i32, dist: i32) {
        assert!(n == 64 || n == 4096);
        let mut s = STATE
            .get_or_init(|| {
                Mutex::new(State {
                    input: [0.; 4096],
                    output: [0.; 4096],
                    n: 0,
                })
            })
            .lock()
            .unwrap();
        s.n = n as usize;
        for i in 0..s.n {
            s.input[i] = input(i, dist as u8);
        }
    }
    pub extern "C" fn guest_probe() -> i32 {
        use magetypes::simd::generic::f32x4 as V;
        let t = archmage::Wasm128Token::summon().unwrap();
        let a = V::splat_t(t, std::hint::black_box(1.0 + 1.0 / 4096.0));
        let c = V::splat_t(t, std::hint::black_box(-(1.0 + 1.0 / 2048.0)));
        i32::from(a.mul_add_relaxed(a, c).to_array()[0] == 1.0 / 16777216.0)
    }
    macro_rules! kernel {
        ($name:ident,$stream:ident,$chain:ident,$w:literal) => {
            fn $name<const MODE: u8, const N: usize, const CHAIN: bool>(
                s: &mut State,
                reps: usize,
            ) {
                let t = archmage::Wasm128Token::summon().unwrap();
                let State { input, output, .. } = s;
                let a: &[f32; N] = input[..N].try_into().unwrap();
                let out: &mut [f32; N] = (&mut output[..N]).try_into().unwrap();
                for _ in 0..reps {
                    if CHAIN {
                        let result = $chain::<MODE>(t, std::hint::black_box(0.75));
                        out[..$w].copy_from_slice(&std::hint::black_box(result));
                    } else {
                        $stream::<MODE, N>(
                            t,
                            std::hint::black_box(a),
                            std::hint::black_box(&mut *out),
                        );
                        std::hint::black_box(&*out);
                    }
                }
            }
        };
    }
    kernel!(run4, stream4_wasm128, chain4_wasm128, 4);
    kernel!(run8, stream8_wasm128, chain8_wasm128, 8);
    kernel!(run16, stream16_wasm128, chain16_wasm128, 16);
    pub extern "C" fn guest_run(width: i32, mode: i32, chain: i32, reps: i32) -> i64 {
        assert!(reps > 0);
        let mut s = STATE.get().expect("prepare first").lock().unwrap();
        macro_rules! call {
            ($mode:literal) => {{
                macro_rules! size {
                    ($n:literal,$chain:literal) => {
                        match width {
                            4 => run4::<$mode, $n, $chain>(&mut s, reps as usize),
                            8 => run8::<$mode, $n, $chain>(&mut s, reps as usize),
                            16 => run16::<$mode, $n, $chain>(&mut s, reps as usize),
                            _ => panic!("width"),
                        }
                    };
                }
                match (s.n, chain) {
                    (64, 0) => size!(64, false),
                    (4096, 0) => size!(4096, false),
                    (64, 1) => size!(64, true),
                    _ => panic!("case"),
                }
            }};
        }
        match mode {
            0 => call!(0),
            2 => call!(2),
            3 => call!(3),
            10 => call!(10),
            11 => call!(11),
            12 => call!(12),
            13 => call!(13),
            14 => call!(14),
            15 => call!(15),
            _ => panic!("arm"),
        }
        i64::from(s.output[0].to_bits())
    }
    fn reference(x: f32, mode: i32) -> f32 {
        if mode == 0 || mode == 3 {
            return 1.0 / x;
        }
        if mode == 2 || mode == 10 {
            let mut y = f32::from_bits(0x7ef127ea_u32.wrapping_sub(x.to_bits()));
            for _ in 0..2 {
                y *= (2.0_f64 - f64::from(x) * f64::from(y)) as f32;
            }
            return y;
        }
        let b = x.to_bits();
        let e = (b >> 23) & 255;
        let m = u64::from((b & 0x7fffff) | 0x800000);
        let num = 1_u64 << 37;
        let mut k = num / m;
        let rem = num % m;
        if 2 * rem > m || (2 * rem == m && k & 1 == 1) {
            k += 1;
        }
        let normalized = 0x3f000000_u32 + ((k as u32 - 8192) << 10);
        f32::from_bits(
            (i64::from(normalized) - ((i64::from(e) - 127) << 23)) as u32 | (b & 0x80000000),
        )
    }
    pub extern "C" fn guest_verify(width: i32, mode: i32, chain: i32) {
        guest_run(width, mode, chain, 1);
        let s = STATE.get().unwrap().lock().unwrap();
        if chain == 1 {
            let mut x = 0.75;
            for _ in 0..64 {
                x = reference(x, mode) + 0.5;
            }
            for y in &s.output[..width as usize] {
                assert_eq!(y.to_bits(), x.to_bits());
            }
        } else {
            for i in 0..s.n {
                assert_eq!(s.output[i].to_bits(), reference(s.input[i], mode).to_bits());
            }
        }
    }
    pub fn entry() {
        std::hint::black_box((
            guest_prepare as extern "C" fn(i32, i32),
            guest_run as extern "C" fn(i32, i32, i32, i32) -> i64,
            guest_verify as extern "C" fn(i32, i32, i32),
            guest_probe as extern "C" fn() -> i32,
        ));
    }
}
fn main() {
    #[cfg(target_arch = "wasm32")]
    guest::entry();
    #[cfg(not(target_arch = "wasm32"))]
    panic!("build the WASI guest target");
}

use std::sync::{Mutex, OnceLock};
pub struct State {
    pub inputd2: Vec<[f64; 2]>,
    pub outputd2: Vec<[f64; 2]>,
    pub fusion: i32,
    pub input4: Vec<[f32; 4]>,
    pub output4: Vec<[f32; 4]>,
    pub input8: Vec<[f32; 8]>,
    pub output8: Vec<[f32; 8]>,
    pub input16: Vec<[f32; 16]>,
    pub output16: Vec<[f32; 16]>,
}
static STATE: OnceLock<Mutex<State>> = OnceLock::new();
// The build discovers compiler-assigned names, then explicitly exports them
// through the linker without unsafe attributes or hand-chosen global symbols.
pub extern "C" fn guest_prepare(size: i32, distribution: i32, op: i32) {
    let dist = ["ordinary", "bits", "sparse", "dense"][distribution as usize];
    let n = size as usize;
    let mut state = STATE
        .get_or_init(|| {
            Mutex::new(State {
                inputd2: Vec::new(),
                outputd2: Vec::new(),
                fusion: 0,
                input4: Vec::new(),
                output4: Vec::new(),
                input8: Vec::new(),
                output8: Vec::new(),
                input16: Vec::new(),
                output16: Vec::new(),
            })
        })
        .lock()
        .unwrap();
    state.fusion = guest_probe();
    state.inputd2 = (0..n / 2)
        .map(|i| core::array::from_fn(|j| super::fma_value_f64(i * 2 + j, dist)))
        .collect();
    state.outputd2 = vec![[0.; 2]; n / 2];
    state.input4 = (0..n / 4)
        .map(|i| {
            core::array::from_fn(|j| {
                if op < 0 {
                    super::fma_value_f32(i * 4 + j, dist)
                } else {
                    super::value(i * 4 + j, dist, op as usize)
                }
            })
        })
        .collect();
    state.input8 = (0..n / 8)
        .map(|i| core::array::from_fn(|j| super::value(i * 8 + j, dist, op as usize)))
        .collect();
    state.input16 = (0..n / 16)
        .map(|i| core::array::from_fn(|j| super::value(i * 16 + j, dist, op as usize)))
        .collect();
    state.output4 = vec![[0.; 4]; n / 4];
    state.output8 = vec![[0.; 8]; n / 8];
    state.output16 = vec![[0.; 16]; n / 16];
}
pub extern "C" fn guest_run(op: i32, arm: i32, width: i32, chain: i32, repeats: i32) -> i64 {
    let mut state = STATE.get().expect("prepare first").lock().unwrap();
    for _ in 0..repeats {
        super::kernels::guest_kernel(op, arm, width, &mut state, chain != 0);
    }
    (match width {
        4 => state.output4[0][0].to_bits(),
        8 => state.output8[0][0].to_bits(),
        16 => state.output16[0][0].to_bits(),
        _ => panic!("width"),
    }) as i64
}

pub extern "C" fn guest_probe() -> i32 {
    use archmage::SimdToken;
    let t = archmage::Wasm128Token::summon().expect("SIMD module");
    let x = magetypes::simd::generic::f32x4::splat_t(t, std::hint::black_box(1.0 + 1.0 / 4096.0));
    let c =
        magetypes::simd::generic::f32x4::splat_t(t, std::hint::black_box(-(1.0 + 1.0 / 2048.0)));
    let r32 = x.mul_add_relaxed(x, c).to_array()[0].to_bits();
    let x =
        magetypes::simd::generic::f64x2::splat_t(t, std::hint::black_box(1.0 + 1.0 / 134217728.0));
    let c = magetypes::simd::generic::f64x2::splat_t(
        t,
        std::hint::black_box(-(1.0 + 1.0 / 67108864.0)),
    );
    let r64 = x.mul_add_relaxed(x, c).to_array()[0].to_bits();
    assert!(r32 == 0 || r32 == 0x33800000);
    assert!(r64 == 0 || r64 == 969u64 << 52);
    i32::from(r32 != 0) | (i32::from(r64 != 0) << 1)
}

// Representative WASM shapes: scalar and one native packed register per type.
// The hardware control uses a per-buffer fusion decision; scalar controls
// broadcast into a packed register because WASM has no scalar FMA opcode.
pub extern "C" fn guest_fma(kind: i32, width: i32, arm: i32, chain: i32, repeats: i32) -> i64 {
    use archmage::SimdToken;
    let t = archmage::Wasm128Token::summon().expect("SIMD module");
    let mut state = STATE.get().expect("prepare first").lock().unwrap();
    match (kind, width, arm) {
        (32, 1, 0) => run_fma::<32, 1, 0, false>(t, &mut state, chain, repeats),
        (32, 1, 1) => run_fma::<32, 1, 1, false>(t, &mut state, chain, repeats),
        (32, 1, 2) => {
            if state.fusion & 1 != 0 {
                run_fma::<32, 1, 2, true>(t, &mut state, chain, repeats)
            } else {
                run_fma::<32, 1, 2, false>(t, &mut state, chain, repeats)
            }
        }
        (32, 4, 0) => run_fma::<32, 4, 0, false>(t, &mut state, chain, repeats),
        (32, 4, 1) => run_fma::<32, 4, 1, false>(t, &mut state, chain, repeats),
        (32, 4, 2) => {
            if state.fusion & 1 != 0 {
                run_fma::<32, 4, 2, true>(t, &mut state, chain, repeats)
            } else {
                run_fma::<32, 4, 2, false>(t, &mut state, chain, repeats)
            }
        }
        (64, 1, 0) => run_fma::<64, 1, 0, false>(t, &mut state, chain, repeats),
        (64, 1, 1) => run_fma::<64, 1, 1, false>(t, &mut state, chain, repeats),
        (64, 1, 2) => {
            if state.fusion & 2 != 0 {
                run_fma::<64, 1, 2, true>(t, &mut state, chain, repeats)
            } else {
                run_fma::<64, 1, 2, false>(t, &mut state, chain, repeats)
            }
        }
        (64, 2, 0) => run_fma::<64, 2, 0, false>(t, &mut state, chain, repeats),
        (64, 2, 1) => run_fma::<64, 2, 1, false>(t, &mut state, chain, repeats),
        (64, 2, 2) => {
            if state.fusion & 2 != 0 {
                run_fma::<64, 2, 2, true>(t, &mut state, chain, repeats)
            } else {
                run_fma::<64, 2, 2, false>(t, &mut state, chain, repeats)
            }
        }
        _ => panic!("unsupported FMA shape"),
    }
}

fn run_fma<const KIND: i32, const WIDTH: i32, const ARM: i32, const FUSED: bool>(
    t: archmage::Wasm128Token,
    state: &mut State,
    chain: i32,
    repeats: i32,
) -> i64 {
    use super::fma_kernels::soft;
    let dependent = chain != 0;
    if KIND == 32 {
        use magetypes::simd::generic::f32x4 as V;
        let b = [std::hint::black_box(0.99991f32); 4];
        let c = [std::hint::black_box(0.00013f32); 4];
        let bv = V::from_array_t(t, b);
        let cv = V::from_array_t(t, c);
        for _ in 0..repeats {
            let mut acc = state.input4[0];
            let mut scalar = acc[0];
            for index in 0..state.input4.len() {
                let a = state.input4[index];
                if WIDTH == 4 {
                    let x = if dependent { acc } else { a };
                    let y = match ARM {
                        0 => V::from_array_t(t, x).mul_add_fused(bv, cv).to_array(),
                        1 => soft::fmaf_soft_lanes(x, b, c),
                        2 => {
                            if FUSED {
                                V::from_array_t(t, x).mul_add_relaxed(bv, cv).to_array()
                            } else {
                                soft::fmaf_soft_lanes(x, b, c)
                            }
                        }
                        _ => panic!("arm"),
                    };
                    if dependent {
                        acc = y;
                    } else {
                        state.output4[index] = y;
                    }
                } else {
                    assert_eq!(WIDTH, 1);
                    for (lane, &lane_value) in a.iter().enumerate() {
                        let x = if dependent { scalar } else { lane_value };
                        let y = match ARM {
                            0 => {
                                magetypes::simd::f32x1(x)
                                    .mul_add_fused(
                                        magetypes::simd::f32x1(b[0]),
                                        magetypes::simd::f32x1(c[0]),
                                    )
                                    .0
                            }
                            1 => soft::fmaf_soft_lanes([x], [b[0]], [c[0]])[0],
                            2 => {
                                if FUSED {
                                    V::splat_t(t, x).mul_add_relaxed(bv, cv).to_array()[0]
                                } else {
                                    soft::fmaf_soft_lanes([x], [b[0]], [c[0]])[0]
                                }
                            }
                            _ => panic!("arm"),
                        };
                        if dependent {
                            scalar = y;
                        } else {
                            state.output4[index][lane] = y;
                        }
                    }
                }
            }
            if dependent {
                state.output4[0] = if WIDTH == 4 { acc } else { [scalar; 4] };
            }
        }
        state.output4[0][0].to_bits() as i64
    } else {
        assert_eq!(KIND, 64);
        use magetypes::simd::generic::f64x2 as V;
        let b = [std::hint::black_box(0.99991f64); 2];
        let c = [std::hint::black_box(0.00013f64); 2];
        let bv = V::from_array_t(t, b);
        let cv = V::from_array_t(t, c);
        for _ in 0..repeats {
            let mut acc = state.inputd2[0];
            let mut scalar = acc[0];
            for index in 0..state.inputd2.len() {
                let a = state.inputd2[index];
                if WIDTH == 2 {
                    let x = if dependent { acc } else { a };
                    let y = match ARM {
                        0 => V::from_array_t(t, x).mul_add_fused(bv, cv).to_array(),
                        1 => soft::software64(x, b, c),
                        2 => {
                            if FUSED {
                                V::from_array_t(t, x).mul_add_relaxed(bv, cv).to_array()
                            } else {
                                soft::software64(x, b, c)
                            }
                        }
                        _ => panic!("arm"),
                    };
                    if dependent {
                        acc = y;
                    } else {
                        state.outputd2[index] = y;
                    }
                } else {
                    assert_eq!(WIDTH, 1);
                    for (lane, &lane_value) in a.iter().enumerate() {
                        let x = if dependent { scalar } else { lane_value };
                        let y = match ARM {
                            0 => {
                                magetypes::simd::f64x1(x)
                                    .mul_add_fused(
                                        magetypes::simd::f64x1(b[0]),
                                        magetypes::simd::f64x1(c[0]),
                                    )
                                    .0
                            }
                            1 => soft::software64([x], [b[0]], [c[0]])[0],
                            2 => {
                                if FUSED {
                                    V::splat_t(t, x).mul_add_relaxed(bv, cv).to_array()[0]
                                } else {
                                    soft::software64([x], [b[0]], [c[0]])[0]
                                }
                            }
                            _ => panic!("arm"),
                        };
                        if dependent {
                            scalar = y;
                        } else {
                            state.outputd2[index][lane] = y;
                        }
                    }
                }
            }
            if dependent {
                state.outputd2[0] = if WIDTH == 2 { acc } else { [scalar; 2] };
            }
        }
        state.outputd2[0][0].to_bits() as i64
    }
}

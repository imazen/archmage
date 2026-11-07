//! Legacy compatibility: all35 names, all enabled widths, every available ISA.
#![allow(deprecated)] // The legacy inherent methods are the contract oracle.
use archmage::prelude::*;
use std::hint::black_box;
fn same<const N: usize>(
    tier: &str,
    op: &str,
    input: [f32; N],
    y: f32,
    got: [f32; N],
    want: [f32; N],
) {
    for i in 0..N {
        assert!(
            got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
            "{tier}/{op}/width{N}/lane{i} {:08x}^{:08x}: got{:08x} expected{:08x}",
            input[i].to_bits(),
            y.to_bits(),
            got[i].to_bits(),
            want[i].to_bits()
        );
    }
}
fn corpus() -> Vec<f32> {
    let mut bits = vec![
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x00800001, 0x3f7fffff, 0x3f800000,
        0x3f800001, 0xbf800000, 0xc0000000, 0x42feffff, 0x42ff0000, 0x42ffffff, 0x43000000,
        0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7f800001, 0xff800001, 0x7fc00123,
        0xffc00123,
    ];
    let mut state = 0x5a17b39du32;
    for _ in 0..4096 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        bits.push(state);
    }
    bits.into_iter().map(f32::from_bits).collect()
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x4)
)]
fn width4(tier: &str) {
    let t = Token::from_context();
    let data = corpus();
    for chunk in data.chunks(4) {
        let mut input = [1.0; 4];
        input[..chunk.len()].copy_from_slice(chunk);
        let v = f32x4::from_array_t(t, input);
        {
            let y = 0.0;
            let want = black_box(v).log2_lowp().to_array();
            let got = magemath::legacy::kernels::f32x4::log2_lowp(t, black_box(v)).to_array();
            same(tier, "log2_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log2_lowp_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log2_lowp", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log2_lowp_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log2_lowp/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log2_lowp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x4::log2_lowp_unchecked(t, black_box(v)).to_array();
            same(tier, "log2_lowp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log2_lowp_unchecked_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log2_lowp_unchecked", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log2_lowp_unchecked_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log2_lowp_unchecked/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp2_lowp().to_array();
            let got = magemath::legacy::kernels::f32x4::exp2_lowp(t, black_box(v)).to_array();
            same(tier, "exp2_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp2_lowp_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp2_lowp", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp2_lowp_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp2_lowp/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp2_lowp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x4::exp2_lowp_unchecked(t, black_box(v)).to_array();
            same(tier, "exp2_lowp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp2_lowp_unchecked_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp2_lowp_unchecked", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp2_lowp_unchecked_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp2_lowp_unchecked/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).ln_lowp().to_array();
            let got = magemath::legacy::kernels::f32x4::ln_lowp(t, black_box(v)).to_array();
            same(tier, "ln_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::ln_lowp_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/ln_lowp", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::ln_lowp_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/ln_lowp/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).ln_lowp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x4::ln_lowp_unchecked(t, black_box(v)).to_array();
            same(tier, "ln_lowp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::ln_lowp_unchecked_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/ln_lowp_unchecked", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::ln_lowp_unchecked_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/ln_lowp_unchecked/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp_lowp().to_array();
            let got = magemath::legacy::kernels::f32x4::exp_lowp(t, black_box(v)).to_array();
            same(tier, "exp_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp_lowp_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp_lowp", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp_lowp_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp_lowp/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp_lowp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x4::exp_lowp_unchecked(t, black_box(v)).to_array();
            same(tier, "exp_lowp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp_lowp_unchecked_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp_lowp_unchecked", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp_lowp_unchecked_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp_lowp_unchecked/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log10_lowp().to_array();
            let got = magemath::legacy::kernels::f32x4::log10_lowp(t, black_box(v)).to_array();
            same(tier, "log10_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log10_lowp_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log10_lowp", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log10_lowp_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log10_lowp/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log10_lowp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x4::log10_lowp_unchecked(t, black_box(v)).to_array();
            same(tier, "log10_lowp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log10_lowp_unchecked_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log10_lowp_unchecked", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log10_lowp_unchecked_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log10_lowp_unchecked/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            for y in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.4, f32::INFINITY, f32::NAN] {
                let want = black_box(v).pow_lowp(y).to_array();
                let got = magemath::legacy::kernels::f32x4::pow_lowp(t, black_box(v), y).to_array();
                same(tier, "pow_lowp", input, y, got, want);
                let got = attuned!(
                    magemath::legacy::array::pow_lowp_f32x4(black_box(input), y),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                same(tier, "array/pow_lowp", input, y, got, want);
                if chunk.len() == 4 {
                    for len in 0..=4 {
                        let mut tail = input[..len].to_vec();
                        attuned!(
                            magemath::legacy::slice::pow_lowp_f32x4(&mut tail, y),
                            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                            using(t)
                        );
                        for i in 0..len {
                            assert!(
                                tail[i].to_bits() == want[i].to_bits()
                                    || tail[i].is_nan() && want[i].is_nan(),
                                "{tier}/slice/pow_lowp/width4/tail{len}/lane{i} input{:08x}",
                                input[i].to_bits()
                            );
                        }
                    }
                }
            }
        }
        {
            for y in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.4, f32::INFINITY, f32::NAN] {
                let want = black_box(v).pow_lowp_unchecked(y).to_array();
                let got = magemath::legacy::kernels::f32x4::pow_lowp_unchecked(t, black_box(v), y)
                    .to_array();
                same(tier, "pow_lowp_unchecked", input, y, got, want);
                let got = attuned!(
                    magemath::legacy::array::pow_lowp_unchecked_f32x4(black_box(input), y),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                same(tier, "array/pow_lowp_unchecked", input, y, got, want);
                if chunk.len() == 4 {
                    for len in 0..=4 {
                        let mut tail = input[..len].to_vec();
                        attuned!(
                            magemath::legacy::slice::pow_lowp_unchecked_f32x4(&mut tail, y),
                            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                            using(t)
                        );
                        for i in 0..len {
                            assert!(
                                tail[i].to_bits() == want[i].to_bits()
                                    || tail[i].is_nan() && want[i].is_nan(),
                                "{tier}/slice/pow_lowp_unchecked/width4/tail{len}/lane{i} input{:08x}",
                                input[i].to_bits()
                            );
                        }
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log2_midp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x4::log2_midp_unchecked(t, black_box(v)).to_array();
            same(tier, "log2_midp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log2_midp_unchecked_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log2_midp_unchecked", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log2_midp_unchecked_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log2_midp_unchecked/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log2_midp().to_array();
            let got = magemath::legacy::kernels::f32x4::log2_midp(t, black_box(v)).to_array();
            same(tier, "log2_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log2_midp_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log2_midp", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log2_midp_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log2_midp/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log2_midp_precise().to_array();
            let got =
                magemath::legacy::kernels::f32x4::log2_midp_precise(t, black_box(v)).to_array();
            same(tier, "log2_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log2_midp_precise_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log2_midp_precise", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log2_midp_precise_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log2_midp_precise/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp2_midp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x4::exp2_midp_unchecked(t, black_box(v)).to_array();
            same(tier, "exp2_midp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp2_midp_unchecked_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp2_midp_unchecked", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp2_midp_unchecked_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp2_midp_unchecked/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp2_midp().to_array();
            let got = magemath::legacy::kernels::f32x4::exp2_midp(t, black_box(v)).to_array();
            same(tier, "exp2_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp2_midp_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp2_midp", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp2_midp_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp2_midp/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp2_midp_precise().to_array();
            let got =
                magemath::legacy::kernels::f32x4::exp2_midp_precise(t, black_box(v)).to_array();
            same(tier, "exp2_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp2_midp_precise_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp2_midp_precise", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp2_midp_precise_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp2_midp_precise/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).ln_midp().to_array();
            let got = magemath::legacy::kernels::f32x4::ln_midp(t, black_box(v)).to_array();
            same(tier, "ln_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::ln_midp_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/ln_midp", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::ln_midp_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/ln_midp/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).ln_midp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x4::ln_midp_unchecked(t, black_box(v)).to_array();
            same(tier, "ln_midp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::ln_midp_unchecked_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/ln_midp_unchecked", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::ln_midp_unchecked_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/ln_midp_unchecked/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).ln_midp_precise().to_array();
            let got = magemath::legacy::kernels::f32x4::ln_midp_precise(t, black_box(v)).to_array();
            same(tier, "ln_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::ln_midp_precise_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/ln_midp_precise", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::ln_midp_precise_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/ln_midp_precise/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp_midp().to_array();
            let got = magemath::legacy::kernels::f32x4::exp_midp(t, black_box(v)).to_array();
            same(tier, "exp_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp_midp_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp_midp", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp_midp_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp_midp/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp_midp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x4::exp_midp_unchecked(t, black_box(v)).to_array();
            same(tier, "exp_midp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp_midp_unchecked_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp_midp_unchecked", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp_midp_unchecked_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp_midp_unchecked/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp_midp_precise().to_array();
            let got =
                magemath::legacy::kernels::f32x4::exp_midp_precise(t, black_box(v)).to_array();
            same(tier, "exp_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp_midp_precise_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp_midp_precise", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp_midp_precise_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp_midp_precise/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).sigmoid_midp().to_array();
            let got = magemath::legacy::kernels::f32x4::sigmoid_midp(t, black_box(v)).to_array();
            same(tier, "sigmoid_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::sigmoid_midp_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/sigmoid_midp", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::sigmoid_midp_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/sigmoid_midp/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).silu_midp().to_array();
            let got = magemath::legacy::kernels::f32x4::silu_midp(t, black_box(v)).to_array();
            same(tier, "silu_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::silu_midp_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/silu_midp", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::silu_midp_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/silu_midp/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log10_midp().to_array();
            let got = magemath::legacy::kernels::f32x4::log10_midp(t, black_box(v)).to_array();
            same(tier, "log10_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log10_midp_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log10_midp", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log10_midp_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log10_midp/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log10_midp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x4::log10_midp_unchecked(t, black_box(v)).to_array();
            same(tier, "log10_midp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log10_midp_unchecked_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log10_midp_unchecked", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log10_midp_unchecked_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log10_midp_unchecked/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log10_midp_precise().to_array();
            let got =
                magemath::legacy::kernels::f32x4::log10_midp_precise(t, black_box(v)).to_array();
            same(tier, "log10_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log10_midp_precise_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log10_midp_precise", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log10_midp_precise_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log10_midp_precise/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            for y in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.4, f32::INFINITY, f32::NAN] {
                let want = black_box(v).pow_midp(y).to_array();
                let got = magemath::legacy::kernels::f32x4::pow_midp(t, black_box(v), y).to_array();
                same(tier, "pow_midp", input, y, got, want);
                let got = attuned!(
                    magemath::legacy::array::pow_midp_f32x4(black_box(input), y),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                same(tier, "array/pow_midp", input, y, got, want);
                if chunk.len() == 4 {
                    for len in 0..=4 {
                        let mut tail = input[..len].to_vec();
                        attuned!(
                            magemath::legacy::slice::pow_midp_f32x4(&mut tail, y),
                            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                            using(t)
                        );
                        for i in 0..len {
                            assert!(
                                tail[i].to_bits() == want[i].to_bits()
                                    || tail[i].is_nan() && want[i].is_nan(),
                                "{tier}/slice/pow_midp/width4/tail{len}/lane{i} input{:08x}",
                                input[i].to_bits()
                            );
                        }
                    }
                }
            }
        }
        {
            for y in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.4, f32::INFINITY, f32::NAN] {
                let want = black_box(v).pow_midp_unchecked(y).to_array();
                let got = magemath::legacy::kernels::f32x4::pow_midp_unchecked(t, black_box(v), y)
                    .to_array();
                same(tier, "pow_midp_unchecked", input, y, got, want);
                let got = attuned!(
                    magemath::legacy::array::pow_midp_unchecked_f32x4(black_box(input), y),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                same(tier, "array/pow_midp_unchecked", input, y, got, want);
                if chunk.len() == 4 {
                    for len in 0..=4 {
                        let mut tail = input[..len].to_vec();
                        attuned!(
                            magemath::legacy::slice::pow_midp_unchecked_f32x4(&mut tail, y),
                            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                            using(t)
                        );
                        for i in 0..len {
                            assert!(
                                tail[i].to_bits() == want[i].to_bits()
                                    || tail[i].is_nan() && want[i].is_nan(),
                                "{tier}/slice/pow_midp_unchecked/width4/tail{len}/lane{i} input{:08x}",
                                input[i].to_bits()
                            );
                        }
                    }
                }
            }
        }
        {
            for y in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.4, f32::INFINITY, f32::NAN] {
                let want = black_box(v).pow_midp_precise(y).to_array();
                let got = magemath::legacy::kernels::f32x4::pow_midp_precise(t, black_box(v), y)
                    .to_array();
                same(tier, "pow_midp_precise", input, y, got, want);
                let got = attuned!(
                    magemath::legacy::array::pow_midp_precise_f32x4(black_box(input), y),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                same(tier, "array/pow_midp_precise", input, y, got, want);
                if chunk.len() == 4 {
                    for len in 0..=4 {
                        let mut tail = input[..len].to_vec();
                        attuned!(
                            magemath::legacy::slice::pow_midp_precise_f32x4(&mut tail, y),
                            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                            using(t)
                        );
                        for i in 0..len {
                            assert!(
                                tail[i].to_bits() == want[i].to_bits()
                                    || tail[i].is_nan() && want[i].is_nan(),
                                "{tier}/slice/pow_midp_precise/width4/tail{len}/lane{i} input{:08x}",
                                input[i].to_bits()
                            );
                        }
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).cbrt_lowp().to_array();
            let got = magemath::legacy::kernels::f32x4::cbrt_lowp(t, black_box(v)).to_array();
            same(tier, "cbrt_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::cbrt_lowp_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/cbrt_lowp", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::cbrt_lowp_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/cbrt_lowp/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).cbrt_midp().to_array();
            let got = magemath::legacy::kernels::f32x4::cbrt_midp(t, black_box(v)).to_array();
            same(tier, "cbrt_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::cbrt_midp_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/cbrt_midp", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::cbrt_midp_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/cbrt_midp/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).cbrt_midp_precise().to_array();
            let got =
                magemath::legacy::kernels::f32x4::cbrt_midp_precise(t, black_box(v)).to_array();
            same(tier, "cbrt_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::cbrt_midp_precise_f32x4(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/cbrt_midp_precise", input, y, got, want);
            if chunk.len() == 4 {
                for len in 0..=4 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::cbrt_midp_precise_f32x4(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/cbrt_midp_precise/width4/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).log2_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log2_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log2_lowp/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).log2_lowp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log2_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log2_lowp_unchecked/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).exp2_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp2_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp2_lowp/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).exp2_lowp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp2_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp2_lowp_unchecked/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).ln_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::ln_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/ln_lowp/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).ln_lowp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::ln_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/ln_lowp_unchecked/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).exp_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp_lowp/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).exp_lowp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp_lowp_unchecked/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).log10_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log10_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log10_lowp/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).log10_lowp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log10_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log10_lowp_unchecked/width4/lane{i}"
            );
        }
    }
    {
        let y = 2.4;
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).pow_lowp(y).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::pow_lowp_f32x4(&mut got, y),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/pow_lowp/width4/lane{i}"
            );
        }
    }
    {
        let y = 2.4;
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).pow_lowp_unchecked(y).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::pow_lowp_unchecked_f32x4(&mut got, y),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/pow_lowp_unchecked/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).log2_midp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log2_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log2_midp_unchecked/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).log2_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log2_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log2_midp/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).log2_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log2_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log2_midp_precise/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).exp2_midp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp2_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp2_midp_unchecked/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).exp2_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp2_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp2_midp/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).exp2_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp2_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp2_midp_precise/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).ln_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::ln_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/ln_midp/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).ln_midp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::ln_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/ln_midp_unchecked/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).ln_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::ln_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/ln_midp_precise/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).exp_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp_midp/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).exp_midp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp_midp_unchecked/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).exp_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp_midp_precise/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).sigmoid_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::sigmoid_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/sigmoid_midp/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).silu_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::silu_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/silu_midp/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).log10_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log10_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log10_midp/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).log10_midp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log10_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log10_midp_unchecked/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).log10_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log10_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log10_midp_precise/width4/lane{i}"
            );
        }
    }
    {
        let y = 2.4;
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).pow_midp(y).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::pow_midp_f32x4(&mut got, y),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/pow_midp/width4/lane{i}"
            );
        }
    }
    {
        let y = 2.4;
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).pow_midp_unchecked(y).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::pow_midp_unchecked_f32x4(&mut got, y),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/pow_midp_unchecked/width4/lane{i}"
            );
        }
    }
    {
        let y = 2.4;
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).pow_midp_precise(y).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::pow_midp_precise_f32x4(&mut got, y),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/pow_midp_precise/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).cbrt_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::cbrt_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/cbrt_lowp/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).cbrt_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::cbrt_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/cbrt_midp/width4/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 4 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(4) {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, pad);
            let out = black_box(v).cbrt_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::cbrt_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/cbrt_midp_precise/width4/lane{i}"
            );
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8)
)]
fn width8(tier: &str) {
    let t = Token::from_context();
    let data = corpus();
    for chunk in data.chunks(8) {
        let mut input = [1.0; 8];
        input[..chunk.len()].copy_from_slice(chunk);
        let v = f32x8::from_array_t(t, input);
        {
            let y = 0.0;
            let want = black_box(v).log2_lowp().to_array();
            let got = magemath::legacy::kernels::log2_lowp(t, black_box(v)).to_array();
            same(tier, "log2_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log2_lowp_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log2_lowp", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log2_lowp_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log2_lowp/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log2_lowp_unchecked().to_array();
            let got = magemath::legacy::kernels::log2_lowp_unchecked(t, black_box(v)).to_array();
            same(tier, "log2_lowp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log2_lowp_unchecked_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log2_lowp_unchecked", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log2_lowp_unchecked_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log2_lowp_unchecked/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp2_lowp().to_array();
            let got = magemath::legacy::kernels::exp2_lowp(t, black_box(v)).to_array();
            same(tier, "exp2_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp2_lowp_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp2_lowp", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp2_lowp_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp2_lowp/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp2_lowp_unchecked().to_array();
            let got = magemath::legacy::kernels::exp2_lowp_unchecked(t, black_box(v)).to_array();
            same(tier, "exp2_lowp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp2_lowp_unchecked_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp2_lowp_unchecked", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp2_lowp_unchecked_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp2_lowp_unchecked/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).ln_lowp().to_array();
            let got = magemath::legacy::kernels::ln_lowp(t, black_box(v)).to_array();
            same(tier, "ln_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::ln_lowp_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/ln_lowp", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::ln_lowp_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/ln_lowp/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).ln_lowp_unchecked().to_array();
            let got = magemath::legacy::kernels::ln_lowp_unchecked(t, black_box(v)).to_array();
            same(tier, "ln_lowp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::ln_lowp_unchecked_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/ln_lowp_unchecked", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::ln_lowp_unchecked_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/ln_lowp_unchecked/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp_lowp().to_array();
            let got = magemath::legacy::kernels::exp_lowp(t, black_box(v)).to_array();
            same(tier, "exp_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp_lowp_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp_lowp", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp_lowp_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp_lowp/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp_lowp_unchecked().to_array();
            let got = magemath::legacy::kernels::exp_lowp_unchecked(t, black_box(v)).to_array();
            same(tier, "exp_lowp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp_lowp_unchecked_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp_lowp_unchecked", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp_lowp_unchecked_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp_lowp_unchecked/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log10_lowp().to_array();
            let got = magemath::legacy::kernels::log10_lowp(t, black_box(v)).to_array();
            same(tier, "log10_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log10_lowp_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log10_lowp", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log10_lowp_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log10_lowp/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log10_lowp_unchecked().to_array();
            let got = magemath::legacy::kernels::log10_lowp_unchecked(t, black_box(v)).to_array();
            same(tier, "log10_lowp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log10_lowp_unchecked_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log10_lowp_unchecked", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log10_lowp_unchecked_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log10_lowp_unchecked/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            for y in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.4, f32::INFINITY, f32::NAN] {
                let want = black_box(v).pow_lowp(y).to_array();
                let got = magemath::legacy::kernels::pow_lowp(t, black_box(v), y).to_array();
                same(tier, "pow_lowp", input, y, got, want);
                let got = attuned!(
                    magemath::legacy::array::pow_lowp_f32x8(black_box(input), y),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                same(tier, "array/pow_lowp", input, y, got, want);
                if chunk.len() == 8 {
                    for len in 0..=8 {
                        let mut tail = input[..len].to_vec();
                        attuned!(
                            magemath::legacy::slice::pow_lowp_f32x8(&mut tail, y),
                            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                            using(t)
                        );
                        for i in 0..len {
                            assert!(
                                tail[i].to_bits() == want[i].to_bits()
                                    || tail[i].is_nan() && want[i].is_nan(),
                                "{tier}/slice/pow_lowp/width8/tail{len}/lane{i} input{:08x}",
                                input[i].to_bits()
                            );
                        }
                    }
                }
            }
        }
        {
            for y in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.4, f32::INFINITY, f32::NAN] {
                let want = black_box(v).pow_lowp_unchecked(y).to_array();
                let got =
                    magemath::legacy::kernels::pow_lowp_unchecked(t, black_box(v), y).to_array();
                same(tier, "pow_lowp_unchecked", input, y, got, want);
                let got = attuned!(
                    magemath::legacy::array::pow_lowp_unchecked_f32x8(black_box(input), y),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                same(tier, "array/pow_lowp_unchecked", input, y, got, want);
                if chunk.len() == 8 {
                    for len in 0..=8 {
                        let mut tail = input[..len].to_vec();
                        attuned!(
                            magemath::legacy::slice::pow_lowp_unchecked_f32x8(&mut tail, y),
                            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                            using(t)
                        );
                        for i in 0..len {
                            assert!(
                                tail[i].to_bits() == want[i].to_bits()
                                    || tail[i].is_nan() && want[i].is_nan(),
                                "{tier}/slice/pow_lowp_unchecked/width8/tail{len}/lane{i} input{:08x}",
                                input[i].to_bits()
                            );
                        }
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log2_midp_unchecked().to_array();
            let got = magemath::legacy::kernels::log2_midp_unchecked(t, black_box(v)).to_array();
            same(tier, "log2_midp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log2_midp_unchecked_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log2_midp_unchecked", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log2_midp_unchecked_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log2_midp_unchecked/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log2_midp().to_array();
            let got = magemath::legacy::kernels::log2_midp(t, black_box(v)).to_array();
            same(tier, "log2_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log2_midp_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log2_midp", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log2_midp_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log2_midp/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log2_midp_precise().to_array();
            let got = magemath::legacy::kernels::log2_midp_precise(t, black_box(v)).to_array();
            same(tier, "log2_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log2_midp_precise_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log2_midp_precise", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log2_midp_precise_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log2_midp_precise/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp2_midp_unchecked().to_array();
            let got = magemath::legacy::kernels::exp2_midp_unchecked(t, black_box(v)).to_array();
            same(tier, "exp2_midp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp2_midp_unchecked_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp2_midp_unchecked", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp2_midp_unchecked_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp2_midp_unchecked/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp2_midp().to_array();
            let got = magemath::legacy::kernels::exp2_midp(t, black_box(v)).to_array();
            same(tier, "exp2_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp2_midp_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp2_midp", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp2_midp_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp2_midp/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp2_midp_precise().to_array();
            let got = magemath::legacy::kernels::exp2_midp_precise(t, black_box(v)).to_array();
            same(tier, "exp2_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp2_midp_precise_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp2_midp_precise", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp2_midp_precise_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp2_midp_precise/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).ln_midp().to_array();
            let got = magemath::legacy::kernels::ln_midp(t, black_box(v)).to_array();
            same(tier, "ln_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::ln_midp_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/ln_midp", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::ln_midp_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/ln_midp/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).ln_midp_unchecked().to_array();
            let got = magemath::legacy::kernels::ln_midp_unchecked(t, black_box(v)).to_array();
            same(tier, "ln_midp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::ln_midp_unchecked_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/ln_midp_unchecked", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::ln_midp_unchecked_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/ln_midp_unchecked/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).ln_midp_precise().to_array();
            let got = magemath::legacy::kernels::ln_midp_precise(t, black_box(v)).to_array();
            same(tier, "ln_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::ln_midp_precise_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/ln_midp_precise", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::ln_midp_precise_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/ln_midp_precise/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp_midp().to_array();
            let got = magemath::legacy::kernels::exp_midp(t, black_box(v)).to_array();
            same(tier, "exp_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp_midp_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp_midp", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp_midp_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp_midp/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp_midp_unchecked().to_array();
            let got = magemath::legacy::kernels::exp_midp_unchecked(t, black_box(v)).to_array();
            same(tier, "exp_midp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp_midp_unchecked_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp_midp_unchecked", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp_midp_unchecked_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp_midp_unchecked/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp_midp_precise().to_array();
            let got = magemath::legacy::kernels::exp_midp_precise(t, black_box(v)).to_array();
            same(tier, "exp_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp_midp_precise_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp_midp_precise", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp_midp_precise_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp_midp_precise/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).sigmoid_midp().to_array();
            let got = magemath::legacy::kernels::sigmoid_midp(t, black_box(v)).to_array();
            same(tier, "sigmoid_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::sigmoid_midp_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/sigmoid_midp", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::sigmoid_midp_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/sigmoid_midp/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).silu_midp().to_array();
            let got = magemath::legacy::kernels::silu_midp(t, black_box(v)).to_array();
            same(tier, "silu_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::silu_midp_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/silu_midp", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::silu_midp_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/silu_midp/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log10_midp().to_array();
            let got = magemath::legacy::kernels::log10_midp(t, black_box(v)).to_array();
            same(tier, "log10_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log10_midp_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log10_midp", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log10_midp_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log10_midp/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log10_midp_unchecked().to_array();
            let got = magemath::legacy::kernels::log10_midp_unchecked(t, black_box(v)).to_array();
            same(tier, "log10_midp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log10_midp_unchecked_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log10_midp_unchecked", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log10_midp_unchecked_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log10_midp_unchecked/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log10_midp_precise().to_array();
            let got = magemath::legacy::kernels::log10_midp_precise(t, black_box(v)).to_array();
            same(tier, "log10_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log10_midp_precise_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log10_midp_precise", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log10_midp_precise_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log10_midp_precise/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            for y in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.4, f32::INFINITY, f32::NAN] {
                let want = black_box(v).pow_midp(y).to_array();
                let got = magemath::legacy::kernels::pow_midp(t, black_box(v), y).to_array();
                same(tier, "pow_midp", input, y, got, want);
                let got = attuned!(
                    magemath::legacy::array::pow_midp_f32x8(black_box(input), y),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                same(tier, "array/pow_midp", input, y, got, want);
                if chunk.len() == 8 {
                    for len in 0..=8 {
                        let mut tail = input[..len].to_vec();
                        attuned!(
                            magemath::legacy::slice::pow_midp_f32x8(&mut tail, y),
                            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                            using(t)
                        );
                        for i in 0..len {
                            assert!(
                                tail[i].to_bits() == want[i].to_bits()
                                    || tail[i].is_nan() && want[i].is_nan(),
                                "{tier}/slice/pow_midp/width8/tail{len}/lane{i} input{:08x}",
                                input[i].to_bits()
                            );
                        }
                    }
                }
            }
        }
        {
            for y in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.4, f32::INFINITY, f32::NAN] {
                let want = black_box(v).pow_midp_unchecked(y).to_array();
                let got =
                    magemath::legacy::kernels::pow_midp_unchecked(t, black_box(v), y).to_array();
                same(tier, "pow_midp_unchecked", input, y, got, want);
                let got = attuned!(
                    magemath::legacy::array::pow_midp_unchecked_f32x8(black_box(input), y),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                same(tier, "array/pow_midp_unchecked", input, y, got, want);
                if chunk.len() == 8 {
                    for len in 0..=8 {
                        let mut tail = input[..len].to_vec();
                        attuned!(
                            magemath::legacy::slice::pow_midp_unchecked_f32x8(&mut tail, y),
                            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                            using(t)
                        );
                        for i in 0..len {
                            assert!(
                                tail[i].to_bits() == want[i].to_bits()
                                    || tail[i].is_nan() && want[i].is_nan(),
                                "{tier}/slice/pow_midp_unchecked/width8/tail{len}/lane{i} input{:08x}",
                                input[i].to_bits()
                            );
                        }
                    }
                }
            }
        }
        {
            for y in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.4, f32::INFINITY, f32::NAN] {
                let want = black_box(v).pow_midp_precise(y).to_array();
                let got =
                    magemath::legacy::kernels::pow_midp_precise(t, black_box(v), y).to_array();
                same(tier, "pow_midp_precise", input, y, got, want);
                let got = attuned!(
                    magemath::legacy::array::pow_midp_precise_f32x8(black_box(input), y),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                same(tier, "array/pow_midp_precise", input, y, got, want);
                if chunk.len() == 8 {
                    for len in 0..=8 {
                        let mut tail = input[..len].to_vec();
                        attuned!(
                            magemath::legacy::slice::pow_midp_precise_f32x8(&mut tail, y),
                            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                            using(t)
                        );
                        for i in 0..len {
                            assert!(
                                tail[i].to_bits() == want[i].to_bits()
                                    || tail[i].is_nan() && want[i].is_nan(),
                                "{tier}/slice/pow_midp_precise/width8/tail{len}/lane{i} input{:08x}",
                                input[i].to_bits()
                            );
                        }
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).cbrt_lowp().to_array();
            let got = magemath::legacy::kernels::cbrt_lowp(t, black_box(v)).to_array();
            same(tier, "cbrt_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::cbrt_lowp_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/cbrt_lowp", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::cbrt_lowp_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/cbrt_lowp/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).cbrt_midp().to_array();
            let got = magemath::legacy::kernels::cbrt_midp(t, black_box(v)).to_array();
            same(tier, "cbrt_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::cbrt_midp_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/cbrt_midp", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::cbrt_midp_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/cbrt_midp/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).cbrt_midp_precise().to_array();
            let got = magemath::legacy::kernels::cbrt_midp_precise(t, black_box(v)).to_array();
            same(tier, "cbrt_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::cbrt_midp_precise_f32x8(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/cbrt_midp_precise", input, y, got, want);
            if chunk.len() == 8 {
                for len in 0..=8 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::cbrt_midp_precise_f32x8(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/cbrt_midp_precise/width8/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).log2_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log2_lowp_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log2_lowp/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).log2_lowp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log2_lowp_unchecked_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log2_lowp_unchecked/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).exp2_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp2_lowp_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp2_lowp/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).exp2_lowp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp2_lowp_unchecked_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp2_lowp_unchecked/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).ln_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::ln_lowp_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/ln_lowp/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).ln_lowp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::ln_lowp_unchecked_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/ln_lowp_unchecked/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).exp_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp_lowp_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp_lowp/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).exp_lowp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp_lowp_unchecked_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp_lowp_unchecked/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).log10_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log10_lowp_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log10_lowp/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).log10_lowp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log10_lowp_unchecked_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log10_lowp_unchecked/width8/lane{i}"
            );
        }
    }
    {
        let y = 2.4;
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).pow_lowp(y).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::pow_lowp_f32x8(&mut got, y),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/pow_lowp/width8/lane{i}"
            );
        }
    }
    {
        let y = 2.4;
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).pow_lowp_unchecked(y).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::pow_lowp_unchecked_f32x8(&mut got, y),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/pow_lowp_unchecked/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).log2_midp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log2_midp_unchecked_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log2_midp_unchecked/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).log2_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log2_midp_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log2_midp/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).log2_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log2_midp_precise_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log2_midp_precise/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).exp2_midp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp2_midp_unchecked_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp2_midp_unchecked/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).exp2_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp2_midp_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp2_midp/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).exp2_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp2_midp_precise_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp2_midp_precise/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).ln_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::ln_midp_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/ln_midp/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).ln_midp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::ln_midp_unchecked_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/ln_midp_unchecked/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).ln_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::ln_midp_precise_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/ln_midp_precise/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).exp_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp_midp_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp_midp/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).exp_midp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp_midp_unchecked_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp_midp_unchecked/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).exp_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp_midp_precise_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp_midp_precise/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).sigmoid_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::sigmoid_midp_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/sigmoid_midp/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).silu_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::silu_midp_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/silu_midp/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).log10_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log10_midp_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log10_midp/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).log10_midp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log10_midp_unchecked_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log10_midp_unchecked/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).log10_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log10_midp_precise_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log10_midp_precise/width8/lane{i}"
            );
        }
    }
    {
        let y = 2.4;
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).pow_midp(y).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::pow_midp_f32x8(&mut got, y),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/pow_midp/width8/lane{i}"
            );
        }
    }
    {
        let y = 2.4;
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).pow_midp_unchecked(y).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::pow_midp_unchecked_f32x8(&mut got, y),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/pow_midp_unchecked/width8/lane{i}"
            );
        }
    }
    {
        let y = 2.4;
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).pow_midp_precise(y).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::pow_midp_precise_f32x8(&mut got, y),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/pow_midp_precise/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).cbrt_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::cbrt_lowp_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/cbrt_lowp/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).cbrt_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::cbrt_midp_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/cbrt_midp/width8/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 8 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, pad);
            let out = black_box(v).cbrt_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::cbrt_midp_precise_f32x8(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/cbrt_midp_precise/width8/lane{i}"
            );
        }
    }
}
#[cfg(feature = "w512")]
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x16)
)]
fn width16(tier: &str) {
    let t = Token::from_context();
    let data = corpus();
    for chunk in data.chunks(16) {
        let mut input = [1.0; 16];
        input[..chunk.len()].copy_from_slice(chunk);
        let v = f32x16::from_array_t(t, input);
        {
            let y = 0.0;
            let want = black_box(v).log2_lowp().to_array();
            let got = magemath::legacy::kernels::f32x16::log2_lowp(t, black_box(v)).to_array();
            same(tier, "log2_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log2_lowp_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log2_lowp", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log2_lowp_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log2_lowp/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log2_lowp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x16::log2_lowp_unchecked(t, black_box(v)).to_array();
            same(tier, "log2_lowp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log2_lowp_unchecked_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log2_lowp_unchecked", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log2_lowp_unchecked_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log2_lowp_unchecked/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp2_lowp().to_array();
            let got = magemath::legacy::kernels::f32x16::exp2_lowp(t, black_box(v)).to_array();
            same(tier, "exp2_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp2_lowp_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp2_lowp", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp2_lowp_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp2_lowp/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp2_lowp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x16::exp2_lowp_unchecked(t, black_box(v)).to_array();
            same(tier, "exp2_lowp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp2_lowp_unchecked_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp2_lowp_unchecked", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp2_lowp_unchecked_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp2_lowp_unchecked/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).ln_lowp().to_array();
            let got = magemath::legacy::kernels::f32x16::ln_lowp(t, black_box(v)).to_array();
            same(tier, "ln_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::ln_lowp_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/ln_lowp", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::ln_lowp_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/ln_lowp/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).ln_lowp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x16::ln_lowp_unchecked(t, black_box(v)).to_array();
            same(tier, "ln_lowp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::ln_lowp_unchecked_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/ln_lowp_unchecked", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::ln_lowp_unchecked_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/ln_lowp_unchecked/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp_lowp().to_array();
            let got = magemath::legacy::kernels::f32x16::exp_lowp(t, black_box(v)).to_array();
            same(tier, "exp_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp_lowp_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp_lowp", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp_lowp_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp_lowp/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp_lowp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x16::exp_lowp_unchecked(t, black_box(v)).to_array();
            same(tier, "exp_lowp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp_lowp_unchecked_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp_lowp_unchecked", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp_lowp_unchecked_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp_lowp_unchecked/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log10_lowp().to_array();
            let got = magemath::legacy::kernels::f32x16::log10_lowp(t, black_box(v)).to_array();
            same(tier, "log10_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log10_lowp_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log10_lowp", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log10_lowp_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log10_lowp/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log10_lowp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x16::log10_lowp_unchecked(t, black_box(v)).to_array();
            same(tier, "log10_lowp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log10_lowp_unchecked_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log10_lowp_unchecked", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log10_lowp_unchecked_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log10_lowp_unchecked/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            for y in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.4, f32::INFINITY, f32::NAN] {
                let want = black_box(v).pow_lowp(y).to_array();
                let got =
                    magemath::legacy::kernels::f32x16::pow_lowp(t, black_box(v), y).to_array();
                same(tier, "pow_lowp", input, y, got, want);
                let got = attuned!(
                    magemath::legacy::array::pow_lowp_f32x16(black_box(input), y),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                same(tier, "array/pow_lowp", input, y, got, want);
                if chunk.len() == 16 {
                    for len in 0..=16 {
                        let mut tail = input[..len].to_vec();
                        attuned!(
                            magemath::legacy::slice::pow_lowp_f32x16(&mut tail, y),
                            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                            using(t)
                        );
                        for i in 0..len {
                            assert!(
                                tail[i].to_bits() == want[i].to_bits()
                                    || tail[i].is_nan() && want[i].is_nan(),
                                "{tier}/slice/pow_lowp/width16/tail{len}/lane{i} input{:08x}",
                                input[i].to_bits()
                            );
                        }
                    }
                }
            }
        }
        {
            for y in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.4, f32::INFINITY, f32::NAN] {
                let want = black_box(v).pow_lowp_unchecked(y).to_array();
                let got = magemath::legacy::kernels::f32x16::pow_lowp_unchecked(t, black_box(v), y)
                    .to_array();
                same(tier, "pow_lowp_unchecked", input, y, got, want);
                let got = attuned!(
                    magemath::legacy::array::pow_lowp_unchecked_f32x16(black_box(input), y),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                same(tier, "array/pow_lowp_unchecked", input, y, got, want);
                if chunk.len() == 16 {
                    for len in 0..=16 {
                        let mut tail = input[..len].to_vec();
                        attuned!(
                            magemath::legacy::slice::pow_lowp_unchecked_f32x16(&mut tail, y),
                            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                            using(t)
                        );
                        for i in 0..len {
                            assert!(
                                tail[i].to_bits() == want[i].to_bits()
                                    || tail[i].is_nan() && want[i].is_nan(),
                                "{tier}/slice/pow_lowp_unchecked/width16/tail{len}/lane{i} input{:08x}",
                                input[i].to_bits()
                            );
                        }
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log2_midp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x16::log2_midp_unchecked(t, black_box(v)).to_array();
            same(tier, "log2_midp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log2_midp_unchecked_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log2_midp_unchecked", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log2_midp_unchecked_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log2_midp_unchecked/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log2_midp().to_array();
            let got = magemath::legacy::kernels::f32x16::log2_midp(t, black_box(v)).to_array();
            same(tier, "log2_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log2_midp_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log2_midp", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log2_midp_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log2_midp/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log2_midp_precise().to_array();
            let got =
                magemath::legacy::kernels::f32x16::log2_midp_precise(t, black_box(v)).to_array();
            same(tier, "log2_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log2_midp_precise_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log2_midp_precise", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log2_midp_precise_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log2_midp_precise/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp2_midp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x16::exp2_midp_unchecked(t, black_box(v)).to_array();
            same(tier, "exp2_midp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp2_midp_unchecked_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp2_midp_unchecked", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp2_midp_unchecked_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp2_midp_unchecked/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp2_midp().to_array();
            let got = magemath::legacy::kernels::f32x16::exp2_midp(t, black_box(v)).to_array();
            same(tier, "exp2_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp2_midp_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp2_midp", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp2_midp_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp2_midp/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp2_midp_precise().to_array();
            let got =
                magemath::legacy::kernels::f32x16::exp2_midp_precise(t, black_box(v)).to_array();
            same(tier, "exp2_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp2_midp_precise_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp2_midp_precise", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp2_midp_precise_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp2_midp_precise/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).ln_midp().to_array();
            let got = magemath::legacy::kernels::f32x16::ln_midp(t, black_box(v)).to_array();
            same(tier, "ln_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::ln_midp_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/ln_midp", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::ln_midp_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/ln_midp/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).ln_midp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x16::ln_midp_unchecked(t, black_box(v)).to_array();
            same(tier, "ln_midp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::ln_midp_unchecked_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/ln_midp_unchecked", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::ln_midp_unchecked_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/ln_midp_unchecked/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).ln_midp_precise().to_array();
            let got =
                magemath::legacy::kernels::f32x16::ln_midp_precise(t, black_box(v)).to_array();
            same(tier, "ln_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::ln_midp_precise_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/ln_midp_precise", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::ln_midp_precise_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/ln_midp_precise/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp_midp().to_array();
            let got = magemath::legacy::kernels::f32x16::exp_midp(t, black_box(v)).to_array();
            same(tier, "exp_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp_midp_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp_midp", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp_midp_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp_midp/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp_midp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x16::exp_midp_unchecked(t, black_box(v)).to_array();
            same(tier, "exp_midp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp_midp_unchecked_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp_midp_unchecked", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp_midp_unchecked_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp_midp_unchecked/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).exp_midp_precise().to_array();
            let got =
                magemath::legacy::kernels::f32x16::exp_midp_precise(t, black_box(v)).to_array();
            same(tier, "exp_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::exp_midp_precise_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/exp_midp_precise", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::exp_midp_precise_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/exp_midp_precise/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).sigmoid_midp().to_array();
            let got = magemath::legacy::kernels::f32x16::sigmoid_midp(t, black_box(v)).to_array();
            same(tier, "sigmoid_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::sigmoid_midp_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/sigmoid_midp", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::sigmoid_midp_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/sigmoid_midp/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).silu_midp().to_array();
            let got = magemath::legacy::kernels::f32x16::silu_midp(t, black_box(v)).to_array();
            same(tier, "silu_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::silu_midp_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/silu_midp", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::silu_midp_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/silu_midp/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log10_midp().to_array();
            let got = magemath::legacy::kernels::f32x16::log10_midp(t, black_box(v)).to_array();
            same(tier, "log10_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log10_midp_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log10_midp", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log10_midp_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log10_midp/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log10_midp_unchecked().to_array();
            let got =
                magemath::legacy::kernels::f32x16::log10_midp_unchecked(t, black_box(v)).to_array();
            same(tier, "log10_midp_unchecked", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log10_midp_unchecked_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log10_midp_unchecked", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log10_midp_unchecked_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log10_midp_unchecked/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).log10_midp_precise().to_array();
            let got =
                magemath::legacy::kernels::f32x16::log10_midp_precise(t, black_box(v)).to_array();
            same(tier, "log10_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::log10_midp_precise_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/log10_midp_precise", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::log10_midp_precise_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/log10_midp_precise/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            for y in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.4, f32::INFINITY, f32::NAN] {
                let want = black_box(v).pow_midp(y).to_array();
                let got =
                    magemath::legacy::kernels::f32x16::pow_midp(t, black_box(v), y).to_array();
                same(tier, "pow_midp", input, y, got, want);
                let got = attuned!(
                    magemath::legacy::array::pow_midp_f32x16(black_box(input), y),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                same(tier, "array/pow_midp", input, y, got, want);
                if chunk.len() == 16 {
                    for len in 0..=16 {
                        let mut tail = input[..len].to_vec();
                        attuned!(
                            magemath::legacy::slice::pow_midp_f32x16(&mut tail, y),
                            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                            using(t)
                        );
                        for i in 0..len {
                            assert!(
                                tail[i].to_bits() == want[i].to_bits()
                                    || tail[i].is_nan() && want[i].is_nan(),
                                "{tier}/slice/pow_midp/width16/tail{len}/lane{i} input{:08x}",
                                input[i].to_bits()
                            );
                        }
                    }
                }
            }
        }
        {
            for y in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.4, f32::INFINITY, f32::NAN] {
                let want = black_box(v).pow_midp_unchecked(y).to_array();
                let got = magemath::legacy::kernels::f32x16::pow_midp_unchecked(t, black_box(v), y)
                    .to_array();
                same(tier, "pow_midp_unchecked", input, y, got, want);
                let got = attuned!(
                    magemath::legacy::array::pow_midp_unchecked_f32x16(black_box(input), y),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                same(tier, "array/pow_midp_unchecked", input, y, got, want);
                if chunk.len() == 16 {
                    for len in 0..=16 {
                        let mut tail = input[..len].to_vec();
                        attuned!(
                            magemath::legacy::slice::pow_midp_unchecked_f32x16(&mut tail, y),
                            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                            using(t)
                        );
                        for i in 0..len {
                            assert!(
                                tail[i].to_bits() == want[i].to_bits()
                                    || tail[i].is_nan() && want[i].is_nan(),
                                "{tier}/slice/pow_midp_unchecked/width16/tail{len}/lane{i} input{:08x}",
                                input[i].to_bits()
                            );
                        }
                    }
                }
            }
        }
        {
            for y in [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.4, f32::INFINITY, f32::NAN] {
                let want = black_box(v).pow_midp_precise(y).to_array();
                let got = magemath::legacy::kernels::f32x16::pow_midp_precise(t, black_box(v), y)
                    .to_array();
                same(tier, "pow_midp_precise", input, y, got, want);
                let got = attuned!(
                    magemath::legacy::array::pow_midp_precise_f32x16(black_box(input), y),
                    [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                    using(t)
                );
                same(tier, "array/pow_midp_precise", input, y, got, want);
                if chunk.len() == 16 {
                    for len in 0..=16 {
                        let mut tail = input[..len].to_vec();
                        attuned!(
                            magemath::legacy::slice::pow_midp_precise_f32x16(&mut tail, y),
                            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                            using(t)
                        );
                        for i in 0..len {
                            assert!(
                                tail[i].to_bits() == want[i].to_bits()
                                    || tail[i].is_nan() && want[i].is_nan(),
                                "{tier}/slice/pow_midp_precise/width16/tail{len}/lane{i} input{:08x}",
                                input[i].to_bits()
                            );
                        }
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).cbrt_lowp().to_array();
            let got = magemath::legacy::kernels::f32x16::cbrt_lowp(t, black_box(v)).to_array();
            same(tier, "cbrt_lowp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::cbrt_lowp_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/cbrt_lowp", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::cbrt_lowp_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/cbrt_lowp/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).cbrt_midp().to_array();
            let got = magemath::legacy::kernels::f32x16::cbrt_midp(t, black_box(v)).to_array();
            same(tier, "cbrt_midp", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::cbrt_midp_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/cbrt_midp", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::cbrt_midp_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/cbrt_midp/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
        {
            let y = 0.0;
            let want = black_box(v).cbrt_midp_precise().to_array();
            let got =
                magemath::legacy::kernels::f32x16::cbrt_midp_precise(t, black_box(v)).to_array();
            same(tier, "cbrt_midp_precise", input, y, got, want);
            let got = attuned!(
                magemath::legacy::array::cbrt_midp_precise_f32x16(black_box(input)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            same(tier, "array/cbrt_midp_precise", input, y, got, want);
            if chunk.len() == 16 {
                for len in 0..=16 {
                    let mut tail = input[..len].to_vec();
                    attuned!(
                        magemath::legacy::slice::cbrt_midp_precise_f32x16(&mut tail),
                        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                        using(t)
                    );
                    for i in 0..len {
                        assert!(
                            tail[i].to_bits() == want[i].to_bits()
                                || tail[i].is_nan() && want[i].is_nan(),
                            "{tier}/slice/cbrt_midp_precise/width16/tail{len}/lane{i} input{:08x}",
                            input[i].to_bits()
                        );
                    }
                }
            }
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).log2_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log2_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log2_lowp/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).log2_lowp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log2_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log2_lowp_unchecked/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).exp2_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp2_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp2_lowp/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).exp2_lowp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp2_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp2_lowp_unchecked/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).ln_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::ln_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/ln_lowp/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).ln_lowp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::ln_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/ln_lowp_unchecked/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).exp_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp_lowp/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).exp_lowp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp_lowp_unchecked/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).log10_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log10_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log10_lowp/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).log10_lowp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log10_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log10_lowp_unchecked/width16/lane{i}"
            );
        }
    }
    {
        let y = 2.4;
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).pow_lowp(y).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::pow_lowp_f32x16(&mut got, y),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/pow_lowp/width16/lane{i}"
            );
        }
    }
    {
        let y = 2.4;
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).pow_lowp_unchecked(y).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::pow_lowp_unchecked_f32x16(&mut got, y),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/pow_lowp_unchecked/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).log2_midp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log2_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log2_midp_unchecked/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).log2_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log2_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log2_midp/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).log2_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log2_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log2_midp_precise/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).exp2_midp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp2_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp2_midp_unchecked/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).exp2_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp2_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp2_midp/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).exp2_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp2_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp2_midp_precise/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).ln_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::ln_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/ln_midp/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).ln_midp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::ln_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/ln_midp_unchecked/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).ln_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::ln_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/ln_midp_precise/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).exp_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp_midp/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).exp_midp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp_midp_unchecked/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).exp_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::exp_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/exp_midp_precise/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).sigmoid_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::sigmoid_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/sigmoid_midp/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).silu_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::silu_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/silu_midp/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).log10_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log10_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log10_midp/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).log10_midp_unchecked().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log10_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log10_midp_unchecked/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).log10_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::log10_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/log10_midp_precise/width16/lane{i}"
            );
        }
    }
    {
        let y = 2.4;
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).pow_midp(y).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::pow_midp_f32x16(&mut got, y),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/pow_midp/width16/lane{i}"
            );
        }
    }
    {
        let y = 2.4;
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).pow_midp_unchecked(y).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::pow_midp_unchecked_f32x16(&mut got, y),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/pow_midp_unchecked/width16/lane{i}"
            );
        }
    }
    {
        let y = 2.4;
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).pow_midp_precise(y).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::pow_midp_precise_f32x16(&mut got, y),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/pow_midp_precise/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).cbrt_lowp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::cbrt_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/cbrt_lowp/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).cbrt_midp().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::cbrt_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/cbrt_midp/width16/lane{i}"
            );
        }
    }
    {
        let mut got = data[..2 * 16 + 3].to_vec();
        let mut want = Vec::new();
        for chunk in got.chunks(16) {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, pad);
            let out = black_box(v).cbrt_midp_precise().to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        attuned!(
            magemath::legacy::slice::cbrt_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                got[i].to_bits() == want[i].to_bits() || got[i].is_nan() && want[i].is_nan(),
                "{tier}/multiblock/cbrt_midp_precise/width16/lane{i}"
            );
        }
    }
}

#[test]
fn all_legacy_names_widths_and_tiers() {
    #[allow(unused_mut)] // AVX-512 adds a tier only in that feature build.
    let mut tiers = vec![
        ("scalar", ScalarToken::summon().is_some()),
        ("v3", X64V3Token::summon().is_some()),
        ("neon", NeonToken::summon().is_some()),
        ("wasm128", Wasm128Token::summon().is_some()),
    ];
    #[cfg(feature = "avx512")]
    tiers.push(("v4x", X64V4xToken::summon().is_some()));
    for (name, available) in &tiers {
        println!("LEGACY_TIER {name} supported={available}");
    }
    if let Ok(required) = std::env::var("LEGACY_REQUIRE_TIERS") {
        for tier in required.split(',') {
            assert!(
                tiers.iter().any(|(name, ok)| *name == tier && *ok),
                "required legacy tier missing: {tier}"
            );
        }
    }
    println!("LEGACY_WIDTHS 4=true 8=true 16={}", cfg!(feature = "w512"));
    if let Some(t) = ScalarToken::summon() {
        width4_scalar_t(t, "scalar");
        width8_scalar_t(t, "scalar");
        #[cfg(feature = "w512")]
        width16_scalar_t(t, "scalar");
        println!(
            "PASS legacy scalar: {}/35x{} names/widths, kernel+array+alltails",
            if cfg!(feature = "w512") { 105 } else { 70 },
            if cfg!(feature = "w512") { 3 } else { 2 }
        );
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        width4_v3_t(t, "v3");
        width8_v3_t(t, "v3");
        #[cfg(feature = "w512")]
        width16_v3_t(t, "v3");
        println!(
            "PASS legacy v3: {}/35x{} names/widths, kernel+array+alltails",
            if cfg!(feature = "w512") { 105 } else { 70 },
            if cfg!(feature = "w512") { 3 } else { 2 }
        );
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        width4_neon_t(t, "neon");
        width8_neon_t(t, "neon");
        #[cfg(feature = "w512")]
        width16_neon_t(t, "neon");
        println!(
            "PASS legacy neon: {}/35x{} names/widths, kernel+array+alltails",
            if cfg!(feature = "w512") { 105 } else { 70 },
            if cfg!(feature = "w512") { 3 } else { 2 }
        );
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        width4_wasm128_t(t, "wasm128");
        width8_wasm128_t(t, "wasm128");
        #[cfg(feature = "w512")]
        width16_wasm128_t(t, "wasm128");
        println!(
            "PASS legacy wasm128: {}/35x{} names/widths, kernel+array+alltails",
            if cfg!(feature = "w512") { 105 } else { 70 },
            if cfg!(feature = "w512") { 3 } else { 2 }
        );
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        width4_v4x_t(t, "v4x");
        width8_v4x_t(t, "v4x");
        #[cfg(feature = "w512")]
        width16_v4x_t(t, "v4x");
        println!(
            "PASS legacy v4x: {}/35x{} names/widths, kernel+array+alltails",
            if cfg!(feature = "w512") { 105 } else { 70 },
            if cfg!(feature = "w512") { 3 } else { 2 }
        );
    }
}

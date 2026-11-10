//! Existing portable/relaxed/unfused contracts at all enabled widths.
use archmage::prelude::*;
use std::hint::black_box;
fn equal(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan()
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_log2_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log2_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log2_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log2_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::log2_lowp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log2_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log2_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_log2_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log2_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log2_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log2_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::log2_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log2_lowp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log2_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_exp2_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp2_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp2_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp2_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::exp2_lowp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp2_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp2_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_exp2_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp2_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp2_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp2_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::exp2_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp2_lowp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp2_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_ln_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::ln_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::ln_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/ln_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::ln_lowp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::ln_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/ln_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_ln_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::ln_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::ln_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/ln_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::ln_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::ln_lowp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/ln_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_exp_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::exp_lowp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_exp_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::exp_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp_lowp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_log10_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log10_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log10_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log10_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::log10_lowp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log10_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log10_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_log10_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log10_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log10_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log10_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::log10_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log10_lowp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log10_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_pow_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::pow_lowp(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::pow_lowp_f32x4(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/pow_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::pow_lowp(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::portable::array::pow_lowp_f32x4(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/pow_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_pow_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::pow_lowp_unchecked(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::pow_lowp_unchecked_f32x4(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/pow_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel =
                magemath::portable::kernels::f32x4::pow_lowp_unchecked(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::portable::array::pow_lowp_unchecked_f32x4(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/pow_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_log2_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log2_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log2_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log2_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::log2_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log2_midp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log2_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_log2_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log2_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log2_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log2_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::log2_midp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log2_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log2_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_log2_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log2_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log2_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log2_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::log2_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log2_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log2_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_exp2_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp2_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp2_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp2_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::exp2_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp2_midp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp2_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_exp2_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp2_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp2_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp2_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::exp2_midp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp2_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp2_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_exp2_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp2_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp2_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp2_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::exp2_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp2_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp2_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_ln_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::ln_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::ln_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/ln_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::ln_midp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::ln_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/ln_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_ln_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::ln_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::ln_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/ln_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::ln_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::ln_midp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/ln_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_ln_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::ln_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::ln_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/ln_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::ln_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::ln_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/ln_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_exp_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::exp_midp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_exp_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::exp_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp_midp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_exp_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::exp_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_sigmoid_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::sigmoid_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::sigmoid_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/sigmoid_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::sigmoid_midp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::sigmoid_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/sigmoid_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_silu_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::silu_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::silu_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/silu_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::silu_midp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::silu_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/silu_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_log10_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log10_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log10_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log10_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::log10_midp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log10_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log10_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_log10_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log10_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log10_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log10_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::log10_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log10_midp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log10_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_log10_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log10_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log10_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log10_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::log10_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log10_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log10_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_pow_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::pow_midp(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::pow_midp_f32x4(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/pow_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::pow_midp(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::portable::array::pow_midp_f32x4(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/pow_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_pow_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::pow_midp_unchecked(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::pow_midp_unchecked_f32x4(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/pow_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel =
                magemath::portable::kernels::f32x4::pow_midp_unchecked(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::portable::array::pow_midp_unchecked_f32x4(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/pow_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_pow_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::pow_midp_precise(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::pow_midp_precise_f32x4(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/pow_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::pow_midp_precise(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::portable::array::pow_midp_precise_f32x4(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/pow_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_cbrt_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::cbrt_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::cbrt_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/cbrt_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::cbrt_lowp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::cbrt_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/cbrt_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_cbrt_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::cbrt_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::cbrt_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/cbrt_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::cbrt_midp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::cbrt_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/cbrt_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn portable_cbrt_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::cbrt_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::cbrt_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/cbrt_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x4::cbrt_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::cbrt_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/cbrt_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_log2_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log2_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log2_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log2_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::log2_lowp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log2_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log2_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_log2_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log2_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log2_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log2_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::log2_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log2_lowp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log2_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_exp2_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp2_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp2_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp2_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::exp2_lowp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp2_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp2_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_exp2_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp2_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp2_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp2_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::exp2_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp2_lowp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp2_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_ln_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::ln_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::ln_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/ln_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::ln_lowp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::ln_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/ln_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_ln_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::ln_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::ln_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/ln_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::ln_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::ln_lowp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/ln_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_exp_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::exp_lowp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_exp_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::exp_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp_lowp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_log10_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log10_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log10_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log10_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::log10_lowp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log10_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log10_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_log10_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log10_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log10_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log10_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::log10_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log10_lowp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log10_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_pow_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::pow_lowp(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::pow_lowp_f32x4(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/pow_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::pow_lowp(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::relaxed::array::pow_lowp_f32x4(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/pow_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_pow_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::pow_lowp_unchecked(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::pow_lowp_unchecked_f32x4(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/pow_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel =
                magemath::relaxed::kernels::f32x4::pow_lowp_unchecked(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::relaxed::array::pow_lowp_unchecked_f32x4(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/pow_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_log2_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log2_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log2_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log2_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::log2_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log2_midp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log2_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_log2_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log2_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log2_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log2_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::log2_midp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log2_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log2_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_log2_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log2_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log2_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log2_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::log2_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log2_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log2_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_exp2_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp2_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp2_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp2_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::exp2_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp2_midp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp2_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_exp2_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp2_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp2_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp2_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::exp2_midp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp2_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp2_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_exp2_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp2_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp2_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp2_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::exp2_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp2_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp2_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_ln_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::ln_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::ln_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/ln_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::ln_midp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::ln_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/ln_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_ln_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::ln_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::ln_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/ln_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::ln_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::ln_midp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/ln_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_ln_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::ln_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::ln_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/ln_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::ln_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::ln_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/ln_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_exp_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::exp_midp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_exp_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::exp_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp_midp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_exp_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::exp_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_sigmoid_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::sigmoid_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::sigmoid_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/sigmoid_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::sigmoid_midp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::sigmoid_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/sigmoid_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_silu_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::silu_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::silu_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/silu_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::silu_midp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::silu_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/silu_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_log10_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log10_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log10_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log10_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::log10_midp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log10_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log10_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_log10_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log10_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log10_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log10_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::log10_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log10_midp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log10_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_log10_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log10_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log10_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log10_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::log10_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log10_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log10_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_pow_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::pow_midp(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::pow_midp_f32x4(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/pow_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::pow_midp(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::relaxed::array::pow_midp_f32x4(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/pow_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_pow_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::pow_midp_unchecked(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::pow_midp_unchecked_f32x4(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/pow_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel =
                magemath::relaxed::kernels::f32x4::pow_midp_unchecked(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::relaxed::array::pow_midp_unchecked_f32x4(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/pow_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_pow_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::pow_midp_precise(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::pow_midp_precise_f32x4(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/pow_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::pow_midp_precise(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::relaxed::array::pow_midp_precise_f32x4(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/pow_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_cbrt_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::cbrt_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::cbrt_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/cbrt_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::cbrt_lowp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::cbrt_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/cbrt_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_cbrt_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::cbrt_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::cbrt_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/cbrt_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::cbrt_midp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::cbrt_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/cbrt_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn relaxed_cbrt_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::cbrt_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::cbrt_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/cbrt_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x4::cbrt_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::cbrt_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/cbrt_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_log2_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log2_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log2_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log2_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::log2_lowp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log2_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log2_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_log2_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log2_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log2_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log2_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::log2_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log2_lowp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log2_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_exp2_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp2_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp2_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp2_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::exp2_lowp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp2_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp2_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_exp2_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp2_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp2_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp2_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::exp2_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp2_lowp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp2_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_ln_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::ln_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::ln_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/ln_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::ln_lowp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::ln_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/ln_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_ln_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::ln_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::ln_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/ln_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::ln_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::ln_lowp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/ln_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_exp_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::exp_lowp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_exp_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::exp_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp_lowp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_log10_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log10_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log10_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log10_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::log10_lowp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log10_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log10_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_log10_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log10_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log10_lowp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log10_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::log10_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log10_lowp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log10_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_pow_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::pow_lowp(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::pow_lowp_f32x4(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/pow_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::pow_lowp(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::unfused::array::pow_lowp_f32x4(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/pow_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_pow_lowp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::pow_lowp_unchecked(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::pow_lowp_unchecked_f32x4(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/pow_lowp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel =
                magemath::unfused::kernels::f32x4::pow_lowp_unchecked(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::unfused::array::pow_lowp_unchecked_f32x4(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/pow_lowp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_log2_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log2_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log2_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log2_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::log2_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log2_midp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log2_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_log2_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log2_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log2_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log2_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::log2_midp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log2_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log2_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_log2_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log2_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log2_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log2_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::log2_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log2_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log2_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_exp2_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp2_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp2_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp2_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::exp2_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp2_midp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp2_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_exp2_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp2_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp2_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp2_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::exp2_midp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp2_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp2_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_exp2_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp2_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp2_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp2_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::exp2_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp2_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp2_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_ln_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::ln_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::ln_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/ln_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::ln_midp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::ln_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/ln_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_ln_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::ln_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::ln_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/ln_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::ln_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::ln_midp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/ln_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_ln_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::ln_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::ln_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/ln_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::ln_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::ln_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/ln_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_exp_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::exp_midp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_exp_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::exp_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp_midp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_exp_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::exp_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_sigmoid_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::sigmoid_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::sigmoid_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/sigmoid_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::sigmoid_midp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::sigmoid_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/sigmoid_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_silu_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::silu_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::silu_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/silu_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::silu_midp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::silu_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/silu_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_log10_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log10_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log10_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log10_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::log10_midp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log10_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log10_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_log10_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log10_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log10_midp_unchecked_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log10_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::log10_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log10_midp_unchecked_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log10_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_log10_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log10_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log10_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log10_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::log10_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log10_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log10_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_pow_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::pow_midp(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::pow_midp_f32x4(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/pow_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::pow_midp(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::unfused::array::pow_midp_f32x4(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/pow_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_pow_midp_unchecked_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::pow_midp_unchecked(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::pow_midp_unchecked_f32x4(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/pow_midp_unchecked/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel =
                magemath::unfused::kernels::f32x4::pow_midp_unchecked(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::unfused::array::pow_midp_unchecked_f32x4(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/pow_midp_unchecked/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_pow_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::pow_midp_precise(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::pow_midp_precise_f32x4(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/pow_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::pow_midp_precise(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::unfused::array::pow_midp_precise_f32x4(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/pow_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_cbrt_lowp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::cbrt_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::cbrt_lowp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/cbrt_lowp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::cbrt_lowp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::cbrt_lowp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/cbrt_lowp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_cbrt_midp_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::cbrt_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::cbrt_midp_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/cbrt_midp/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::cbrt_midp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::cbrt_midp_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/cbrt_midp/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4),
    inline(never)
)]
fn unfused_cbrt_midp_precise_width4(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 4 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::cbrt_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::cbrt_midp_precise_f32x4(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/cbrt_midp_precise/width4/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(4).enumerate() {
            let mut pad = [1.0; 4];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x4::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x4::cbrt_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::cbrt_midp_precise_f32x4(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 4 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/cbrt_midp_precise/width4/kernel-array/{i}"
                );
            }
        }
    }
}
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x4)
)]
fn width4(tier: &str) {
    let t = Token::from_context();
    attuned!(
        portable_log2_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log2_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp2_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp2_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_ln_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_ln_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log10_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log10_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_pow_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_pow_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log2_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log2_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log2_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp2_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp2_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp2_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_ln_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_ln_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_ln_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_sigmoid_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_silu_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log10_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log10_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log10_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_pow_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_pow_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_pow_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_cbrt_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_cbrt_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_cbrt_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log2_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log2_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp2_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp2_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_ln_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_ln_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log10_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log10_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_pow_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_pow_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log2_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log2_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log2_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp2_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp2_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp2_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_ln_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_ln_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_ln_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_sigmoid_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_silu_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log10_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log10_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log10_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_pow_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_pow_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_pow_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_cbrt_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_cbrt_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_cbrt_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log2_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log2_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp2_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp2_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_ln_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_ln_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log10_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log10_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_pow_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_pow_lowp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log2_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log2_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log2_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp2_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp2_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp2_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_ln_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_ln_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_ln_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_sigmoid_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_silu_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log10_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log10_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log10_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_pow_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_pow_midp_unchecked_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_pow_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_cbrt_lowp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_cbrt_midp_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_cbrt_midp_precise_width4(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
}
#[cfg(feature = "w512")]
#[attune(
    _scalar_t,
    _v3_t,
    _neon_t,
    _wasm128_t,
    _v4x_t(cfg(avx512)),
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_log2_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log2_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log2_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log2_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::log2_lowp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log2_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log2_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_log2_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log2_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log2_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log2_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::log2_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log2_lowp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log2_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_exp2_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp2_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp2_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp2_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::exp2_lowp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp2_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp2_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_exp2_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp2_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp2_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp2_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::exp2_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp2_lowp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp2_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_ln_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::ln_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::ln_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/ln_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::ln_lowp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::ln_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/ln_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_ln_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::ln_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::ln_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/ln_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::ln_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::ln_lowp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/ln_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_exp_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::exp_lowp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_exp_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::exp_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp_lowp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_log10_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log10_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log10_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log10_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::log10_lowp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log10_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log10_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_log10_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log10_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log10_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log10_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::log10_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log10_lowp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log10_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_pow_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::pow_lowp(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::pow_lowp_f32x16(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/pow_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::pow_lowp(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::portable::array::pow_lowp_f32x16(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/pow_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_pow_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::pow_lowp_unchecked(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::pow_lowp_unchecked_f32x16(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/pow_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel =
                magemath::portable::kernels::f32x16::pow_lowp_unchecked(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::portable::array::pow_lowp_unchecked_f32x16(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/pow_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_log2_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log2_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log2_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log2_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::log2_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log2_midp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log2_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_log2_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log2_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log2_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log2_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::log2_midp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log2_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log2_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_log2_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log2_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log2_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log2_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::log2_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log2_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log2_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_exp2_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp2_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp2_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp2_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::exp2_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp2_midp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp2_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_exp2_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp2_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp2_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp2_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::exp2_midp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp2_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp2_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_exp2_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp2_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp2_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp2_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::exp2_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp2_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp2_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_ln_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::ln_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::ln_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/ln_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::ln_midp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::ln_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/ln_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_ln_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::ln_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::ln_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/ln_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::ln_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::ln_midp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/ln_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_ln_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::ln_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::ln_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/ln_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::ln_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::ln_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/ln_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_exp_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::exp_midp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_exp_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::exp_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp_midp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_exp_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::exp_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::exp_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/exp_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::exp_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::exp_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/exp_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_sigmoid_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::sigmoid_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::sigmoid_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/sigmoid_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::sigmoid_midp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::sigmoid_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/sigmoid_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_silu_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::silu_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::silu_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/silu_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::silu_midp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::silu_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/silu_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_log10_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log10_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log10_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log10_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::log10_midp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log10_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log10_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_log10_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log10_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log10_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log10_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::log10_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log10_midp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log10_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_log10_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::log10_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::log10_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/log10_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::log10_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::log10_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/log10_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_pow_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::pow_midp(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::pow_midp_f32x16(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/pow_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::pow_midp(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::portable::array::pow_midp_f32x16(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/pow_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_pow_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::pow_midp_unchecked(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::pow_midp_unchecked_f32x16(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/pow_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel =
                magemath::portable::kernels::f32x16::pow_midp_unchecked(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::portable::array::pow_midp_unchecked_f32x16(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/pow_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_pow_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::pow_midp_precise(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::pow_midp_precise_f32x16(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/pow_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel =
                magemath::portable::kernels::f32x16::pow_midp_precise(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::portable::array::pow_midp_precise_f32x16(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/pow_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_cbrt_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::cbrt_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::cbrt_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/cbrt_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::cbrt_lowp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::cbrt_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/cbrt_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_cbrt_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::cbrt_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::cbrt_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/cbrt_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::cbrt_midp(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::cbrt_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/cbrt_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn portable_cbrt_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::portable::kernels::cbrt_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::portable::slice::cbrt_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/portable/cbrt_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::portable::kernels::f32x16::cbrt_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::portable::array::cbrt_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/portable/cbrt_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_log2_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log2_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log2_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log2_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::log2_lowp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log2_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log2_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_log2_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log2_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log2_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log2_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::log2_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log2_lowp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log2_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_exp2_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp2_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp2_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp2_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::exp2_lowp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp2_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp2_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_exp2_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp2_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp2_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp2_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::exp2_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp2_lowp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp2_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_ln_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::ln_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::ln_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/ln_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::ln_lowp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::ln_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/ln_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_ln_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::ln_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::ln_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/ln_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::ln_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::ln_lowp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/ln_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_exp_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::exp_lowp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_exp_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::exp_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp_lowp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_log10_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log10_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log10_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log10_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::log10_lowp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log10_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log10_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_log10_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log10_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log10_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log10_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::log10_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log10_lowp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log10_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_pow_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::pow_lowp(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::pow_lowp_f32x16(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/pow_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::pow_lowp(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::relaxed::array::pow_lowp_f32x16(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/pow_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_pow_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::pow_lowp_unchecked(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::pow_lowp_unchecked_f32x16(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/pow_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel =
                magemath::relaxed::kernels::f32x16::pow_lowp_unchecked(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::relaxed::array::pow_lowp_unchecked_f32x16(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/pow_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_log2_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log2_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log2_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log2_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::log2_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log2_midp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log2_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_log2_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log2_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log2_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log2_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::log2_midp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log2_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log2_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_log2_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log2_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log2_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log2_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::log2_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log2_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log2_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_exp2_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp2_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp2_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp2_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::exp2_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp2_midp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp2_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_exp2_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp2_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp2_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp2_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::exp2_midp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp2_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp2_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_exp2_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp2_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp2_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp2_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::exp2_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp2_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp2_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_ln_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::ln_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::ln_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/ln_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::ln_midp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::ln_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/ln_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_ln_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::ln_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::ln_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/ln_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::ln_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::ln_midp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/ln_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_ln_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::ln_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::ln_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/ln_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::ln_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::ln_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/ln_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_exp_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::exp_midp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_exp_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::exp_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp_midp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_exp_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::exp_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::exp_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/exp_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::exp_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::exp_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/exp_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_sigmoid_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::sigmoid_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::sigmoid_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/sigmoid_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::sigmoid_midp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::sigmoid_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/sigmoid_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_silu_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::silu_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::silu_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/silu_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::silu_midp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::silu_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/silu_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_log10_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log10_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log10_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log10_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::log10_midp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log10_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log10_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_log10_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log10_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log10_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log10_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::log10_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log10_midp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log10_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_log10_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::log10_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::log10_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/log10_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::log10_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::log10_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/log10_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_pow_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::pow_midp(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::pow_midp_f32x16(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/pow_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::pow_midp(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::relaxed::array::pow_midp_f32x16(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/pow_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_pow_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::pow_midp_unchecked(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::pow_midp_unchecked_f32x16(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/pow_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel =
                magemath::relaxed::kernels::f32x16::pow_midp_unchecked(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::relaxed::array::pow_midp_unchecked_f32x16(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/pow_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_pow_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::pow_midp_precise(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::pow_midp_precise_f32x16(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/pow_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::pow_midp_precise(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::relaxed::array::pow_midp_precise_f32x16(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/pow_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_cbrt_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::cbrt_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::cbrt_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/cbrt_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::cbrt_lowp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::cbrt_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/cbrt_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_cbrt_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::cbrt_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::cbrt_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/cbrt_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::cbrt_midp(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::cbrt_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/cbrt_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn relaxed_cbrt_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::relaxed::kernels::cbrt_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::relaxed::slice::cbrt_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/relaxed/cbrt_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::relaxed::kernels::f32x16::cbrt_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::relaxed::array::cbrt_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/relaxed/cbrt_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_log2_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log2_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log2_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log2_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::log2_lowp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log2_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log2_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_log2_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log2_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log2_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log2_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::log2_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log2_lowp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log2_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_exp2_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp2_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp2_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp2_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::exp2_lowp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp2_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp2_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_exp2_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp2_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp2_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp2_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::exp2_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp2_lowp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp2_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_ln_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::ln_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::ln_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/ln_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::ln_lowp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::ln_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/ln_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_ln_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::ln_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::ln_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/ln_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::ln_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::ln_lowp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/ln_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_exp_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::exp_lowp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_exp_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::exp_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp_lowp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_log10_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log10_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log10_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log10_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::log10_lowp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log10_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log10_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_log10_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log10_lowp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log10_lowp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log10_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::log10_lowp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log10_lowp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log10_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_pow_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::pow_lowp(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::pow_lowp_f32x16(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/pow_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::pow_lowp(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::unfused::array::pow_lowp_f32x16(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/pow_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_pow_lowp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::pow_lowp_unchecked(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::pow_lowp_unchecked_f32x16(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/pow_lowp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel =
                magemath::unfused::kernels::f32x16::pow_lowp_unchecked(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::unfused::array::pow_lowp_unchecked_f32x16(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/pow_lowp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_log2_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log2_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log2_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log2_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::log2_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log2_midp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log2_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_log2_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log2_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log2_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log2_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::log2_midp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log2_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log2_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_log2_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log2_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log2_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log2_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::log2_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log2_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log2_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_exp2_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp2_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp2_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp2_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::exp2_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp2_midp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp2_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_exp2_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp2_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp2_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp2_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::exp2_midp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp2_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp2_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_exp2_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp2_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp2_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp2_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::exp2_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp2_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp2_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_ln_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::ln_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::ln_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/ln_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::ln_midp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::ln_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/ln_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_ln_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::ln_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::ln_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/ln_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::ln_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::ln_midp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/ln_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_ln_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::ln_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::ln_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/ln_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::ln_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::ln_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/ln_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_exp_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::exp_midp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_exp_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::exp_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp_midp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_exp_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::exp_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::exp_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/exp_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::exp_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::exp_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/exp_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_sigmoid_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::sigmoid_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::sigmoid_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/sigmoid_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::sigmoid_midp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::sigmoid_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/sigmoid_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_silu_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::silu_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::silu_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/silu_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::silu_midp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::silu_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/silu_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_log10_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log10_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log10_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log10_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::log10_midp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log10_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log10_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_log10_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log10_midp_unchecked(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log10_midp_unchecked_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log10_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::log10_midp_unchecked(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log10_midp_unchecked_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log10_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_log10_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::log10_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::log10_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/log10_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::log10_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::log10_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/log10_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_pow_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::pow_midp(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::pow_midp_f32x16(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/pow_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::pow_midp(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::unfused::array::pow_midp_f32x16(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/pow_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_pow_midp_unchecked_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::pow_midp_unchecked(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::pow_midp_unchecked_f32x16(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/pow_midp_unchecked/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel =
                magemath::unfused::kernels::f32x16::pow_midp_unchecked(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::unfused::array::pow_midp_unchecked_f32x16(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/pow_midp_unchecked/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_pow_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::pow_midp_precise(t, v, 2.4).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::pow_midp_precise_f32x16(&mut got, 2.4),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/pow_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::pow_midp_precise(t, v, 2.4).to_array();
            let array = attuned!(
                magemath::unfused::array::pow_midp_precise_f32x16(black_box(pad), 2.4),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/pow_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_cbrt_lowp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::cbrt_lowp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::cbrt_lowp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/cbrt_lowp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::cbrt_lowp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::cbrt_lowp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/cbrt_lowp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_cbrt_midp_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::cbrt_midp(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::cbrt_midp_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/cbrt_midp/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::cbrt_midp(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::cbrt_midp_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/cbrt_midp/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16),
    inline(never)
)]
fn unfused_cbrt_midp_precise_width16(tier: &str) {
    let t = Token::from_context();
    {
        let data = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
            0.75,
            2.0,
            4.0,
            32.0,
            -32.0,
        ];
        let input: Vec<f32> = (0..2 * 16 + 3).map(|i| data[i % data.len()]).collect();
        let mut want = Vec::new();
        for chunk in input.chunks(8) {
            let mut pad = [1.0; 8];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x8::from_array_t(t, black_box(pad));
            let out = magemath::unfused::kernels::cbrt_midp_precise(t, v).to_array();
            want.extend_from_slice(&out[..chunk.len()]);
        }
        let mut got = input.clone();
        attuned!(
            magemath::unfused::slice::cbrt_midp_precise_f32x16(&mut got),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        );
        for i in 0..got.len() {
            assert!(
                equal(got[i], want[i]),
                "{tier}/unfused/cbrt_midp_precise/width16/slice/{i} input{:08x} got{:08x} want{:08x}",
                input[i].to_bits(),
                got[i].to_bits(),
                want[i].to_bits()
            );
        }
        for (index, chunk) in input.chunks(16).enumerate() {
            let mut pad = [1.0; 16];
            pad[..chunk.len()].copy_from_slice(chunk);
            let v = f32x16::from_array_t(t, black_box(pad));
            let kernel = magemath::unfused::kernels::f32x16::cbrt_midp_precise(t, v).to_array();
            let array = attuned!(
                magemath::unfused::array::cbrt_midp_precise_f32x16(black_box(pad)),
                [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
                using(t)
            );
            for i in 0..chunk.len() {
                let expected = want[index * 16 + i];
                assert!(
                    equal(kernel[i], expected) && equal(array[i], expected),
                    "{tier}/unfused/cbrt_midp_precise/width16/kernel-array/{i}"
                );
            }
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
    define(f32x8, f32x16)
)]
fn width16(tier: &str) {
    let t = Token::from_context();
    attuned!(
        portable_log2_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log2_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp2_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp2_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_ln_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_ln_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log10_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log10_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_pow_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_pow_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log2_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log2_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log2_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp2_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp2_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp2_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_ln_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_ln_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_ln_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_exp_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_sigmoid_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_silu_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log10_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log10_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_log10_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_pow_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_pow_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_pow_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_cbrt_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_cbrt_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        portable_cbrt_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log2_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log2_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp2_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp2_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_ln_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_ln_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log10_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log10_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_pow_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_pow_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log2_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log2_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log2_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp2_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp2_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp2_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_ln_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_ln_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_ln_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_exp_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_sigmoid_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_silu_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log10_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log10_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_log10_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_pow_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_pow_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_pow_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_cbrt_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_cbrt_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        relaxed_cbrt_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log2_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log2_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp2_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp2_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_ln_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_ln_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log10_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log10_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_pow_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_pow_lowp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log2_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log2_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log2_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp2_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp2_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp2_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_ln_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_ln_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_ln_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_exp_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_sigmoid_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_silu_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log10_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log10_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_log10_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_pow_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_pow_midp_unchecked_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_pow_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_cbrt_lowp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_cbrt_midp_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
    attuned!(
        unfused_cbrt_midp_precise_width16(tier),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    );
}
#[test]
fn same_mode_widths() {
    let mut supported = vec!["scalar"];

    if let Some(t) = ScalarToken::summon() {
        supported.push("scalar");
        width4_scalar_t(t, "scalar");
        #[cfg(feature = "w512")]
        width16_scalar_t(t, "scalar");
        println!("PASS existing families width parity scalar");
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        supported.push("v3");
        width4_v3_t(t, "v3");
        #[cfg(feature = "w512")]
        width16_v3_t(t, "v3");
        println!("PASS existing families width parity v3");
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        supported.push("neon");
        width4_neon_t(t, "neon");
        #[cfg(feature = "w512")]
        width16_neon_t(t, "neon");
        println!("PASS existing families width parity neon");
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        supported.push("wasm128");
        width4_wasm128_t(t, "wasm128");
        #[cfg(feature = "w512")]
        width16_wasm128_t(t, "wasm128");
        println!("PASS existing families width parity wasm128");
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        supported.push("v4x");
        width4_v4x_t(t, "v4x");
        #[cfg(feature = "w512")]
        width16_v4x_t(t, "v4x");
        println!("PASS existing families width parity v4x");
    }
    if let Ok(required) = std::env::var("LEGACY_REQUIRE_TIERS") {
        for tier in required.split(',') {
            assert!(
                supported.contains(&tier),
                "required width tier missing: {tier}"
            );
        }
    }
}

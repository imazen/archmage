use crate::Case;
use archmage::prelude::*;
fn k0(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.powf(b);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k1(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.powf(b);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k2(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::powf(a, b);
    }
}
fn k3(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::powf(a, b);
    }
}
fn k4(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = libm::powf(a, b);
    }
}
fn k5(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = fpmath::pow(a, b);
    }
}
fn k6(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = micromath::F32(a).powf(micromath::F32(b)).0;
    }
}
fn k7(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = metallic::powf(a, b);
    }
}
fn k8(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::pow_midp_f32(a, b);
    }
}
fn k9(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::unfused::pow_midp_f32(a, b);
    }
}
fn k10(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::pow_lowp_f32(a, b);
    }
}
fn k11(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::unfused::pow_lowp_f32(a, b);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k12(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    out.copy_from_slice(x);
    archmage::attuned!(magemath::portable::slice::pow_f32(out, y));
}
fn k13(x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = y;
    out.copy_from_slice(x);
    magemath::portable::slice::pow_f32(out, y);
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k14(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    out.copy_from_slice(x);
    archmage::attuned!(magemath::portable::slice::pow_f32_with::<
        { magemath::PowBehavior::MIDP.to_u64() },
    >(out, y));
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k15(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    out.copy_from_slice(x);
    archmage::attuned!(magemath::portable::slice::pow_f32_with::<
        { magemath::PowBehavior::LOWP.to_u64() },
    >(out, y));
}
fn k16(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.powf(2.4);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k17(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.powf(2.4);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k18(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::powf(a, 2.4);
    }
}
fn k19(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::powf(a, 2.4);
    }
}
fn k20(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = libm::powf(a, 2.4);
    }
}
fn k21(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = fpmath::pow(a, 2.4);
    }
}
fn k22(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = micromath::F32(a).powf(micromath::F32(2.4)).0;
    }
}
fn k23(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = metallic::powf(a, 2.4);
    }
}
fn k24(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::pow_midp_f32(a, 2.4);
    }
}
fn k25(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::unfused::pow_midp_f32(a, 2.4);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k26(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::portable::kernels::pow_midp(t, a, 2.4);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k27(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::unfused::kernels::pow_midp(t, a, 2.4);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k28(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::relaxed::kernels::pow_midp(t, a, 2.4);
        r.store(o);
    }
}
fn k29(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::pow_lowp_f32(a, 2.4);
    }
}
fn k30(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::unfused::pow_lowp_f32(a, 2.4);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k31(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::portable::kernels::pow_lowp(t, a, 2.4);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::unfused::kernels::pow_lowp(t, a, 2.4);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k33(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::relaxed::kernels::pow_lowp(t, a, 2.4);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k34(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    out.copy_from_slice(x);
    archmage::attuned!(magemath::portable::slice::pow_f32(out, y));
}
fn k35(x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = y;
    out.copy_from_slice(x);
    magemath::portable::slice::pow_f32(out, y);
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k36(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.pow_midp(2.4);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k37(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.pow_lowp(2.4);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k38(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.pow_midp(2.4);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k39(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.pow_lowp(2.4);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k40(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.pow_midp(2.4);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k41(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.pow_lowp(2.4);
        r.store(o);
    }
}
fn k42(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.exp();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k43(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.exp();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k44(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::expf(a);
    }
}
fn k45(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::expf(a);
    }
}
fn k46(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = libm::expf(a);
    }
}
fn k47(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = fpmath::exp(a);
    }
}
fn k48(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = micromath::F32(a).exp().0;
    }
}
fn k49(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = metallic::expf(a);
    }
}
fn k50(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::exp_midp_f32(a);
    }
}
fn k51(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::unfused::exp_midp_f32(a);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k52(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::portable::kernels::exp_midp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k53(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::unfused::kernels::exp_midp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k54(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::relaxed::kernels::exp_midp(t, a);
        r.store(o);
    }
}
fn k55(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::exp_lowp_f32(a);
    }
}
fn k56(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::unfused::exp_lowp_f32(a);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k57(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::portable::kernels::exp_lowp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k58(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::unfused::kernels::exp_lowp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k59(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::relaxed::kernels::exp_lowp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k60(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    out.copy_from_slice(x);
    archmage::attuned!(magemath::portable::slice::exp_f32(out));
}
fn k61(x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = y;
    out.copy_from_slice(x);
    magemath::portable::slice::exp_f32(out);
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k62(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.exp_midp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k63(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.exp_lowp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k64(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.exp_midp_precise();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k65(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.exp_midp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k66(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.exp_lowp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k67(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.exp_midp_precise();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k68(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.exp_midp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k69(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.exp_lowp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k70(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.exp_midp_precise();
        r.store(o);
    }
}
fn k71(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.ln();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k72(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.ln();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k73(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::logf(a);
    }
}
fn k74(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::logf(a);
    }
}
fn k75(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = libm::logf(a);
    }
}
fn k76(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = fpmath::ln(a);
    }
}
fn k77(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = micromath::F32(a).ln().0;
    }
}
fn k78(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = metallic::logf(a);
    }
}
fn k79(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::ln_midp_f32(a);
    }
}
fn k80(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::unfused::ln_midp_f32(a);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k81(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::portable::kernels::ln_midp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k82(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::unfused::kernels::ln_midp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k83(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::relaxed::kernels::ln_midp(t, a);
        r.store(o);
    }
}
fn k84(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::ln_lowp_f32(a);
    }
}
fn k85(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::unfused::ln_lowp_f32(a);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k86(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::portable::kernels::ln_lowp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k87(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::unfused::kernels::ln_lowp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k88(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::relaxed::kernels::ln_lowp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k89(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    out.copy_from_slice(x);
    archmage::attuned!(magemath::portable::slice::ln_f32(out));
}
fn k90(x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = y;
    out.copy_from_slice(x);
    magemath::portable::slice::ln_f32(out);
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k91(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.ln_midp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k92(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.ln_lowp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k93(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.ln_midp_precise();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k94(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.ln_midp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k95(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.ln_lowp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k96(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.ln_midp_precise();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k97(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.ln_midp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k98(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.ln_lowp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k99(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.ln_midp_precise();
        r.store(o);
    }
}
fn k100(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.log2();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k101(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.log2();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k102(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::log2f(a);
    }
}
fn k103(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::log2f(a);
    }
}
fn k104(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = libm::log2f(a);
    }
}
fn k105(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = fpmath::log2(a);
    }
}
fn k106(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = micromath::F32(a).log2().0;
    }
}
fn k107(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = metallic::log2f(a);
    }
}
fn k108(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::log2_midp_f32(a);
    }
}
fn k109(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::unfused::log2_midp_f32(a);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k110(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::portable::kernels::log2_midp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k111(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::unfused::kernels::log2_midp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k112(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::relaxed::kernels::log2_midp(t, a);
        r.store(o);
    }
}
fn k113(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::log2_lowp_f32(a);
    }
}
fn k114(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::unfused::log2_lowp_f32(a);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k115(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::portable::kernels::log2_lowp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k116(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::unfused::kernels::log2_lowp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k117(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::relaxed::kernels::log2_lowp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k118(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    out.copy_from_slice(x);
    archmage::attuned!(magemath::portable::slice::log2_f32(out));
}
fn k119(x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = y;
    out.copy_from_slice(x);
    magemath::portable::slice::log2_f32(out);
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k120(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.log2_midp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k121(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.log2_lowp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k122(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.log2_midp_precise();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k123(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.log2_midp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k124(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.log2_lowp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k125(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.log2_midp_precise();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k126(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.log2_midp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k127(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.log2_lowp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k128(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.log2_midp_precise();
        r.store(o);
    }
}
fn k129(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.log10();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k130(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.log10();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k131(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::log10f(a);
    }
}
fn k132(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::log10f(a);
    }
}
fn k133(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = libm::log10f(a);
    }
}
fn k134(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = fpmath::log10(a);
    }
}
fn k135(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = micromath::F32(a).log10().0;
    }
}
fn k136(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = metallic::log10f(a);
    }
}
fn k137(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::log10_midp_f32(a);
    }
}
fn k138(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::unfused::log10_midp_f32(a);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k139(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::portable::kernels::log10_midp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k140(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::unfused::kernels::log10_midp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k141(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::relaxed::kernels::log10_midp(t, a);
        r.store(o);
    }
}
fn k142(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::log10_lowp_f32(a);
    }
}
fn k143(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::unfused::log10_lowp_f32(a);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k144(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::portable::kernels::log10_lowp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k145(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::unfused::kernels::log10_lowp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k146(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::relaxed::kernels::log10_lowp(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k147(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.log10_midp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k148(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.log10_lowp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k149(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.log10_midp_precise();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k150(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.log10_midp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k151(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.log10_lowp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k152(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.log10_midp_precise();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k153(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.log10_midp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k154(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.log10_lowp();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k155(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.log10_midp_precise();
        r.store(o);
    }
}
fn k156(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.sqrt();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k157(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.sqrt();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k158(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::sqrtf(a);
    }
}
fn k159(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::sqrtf(a);
    }
}
fn k160(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = libm::sqrtf(a);
    }
}
fn k161(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = fpmath::sqrt(a);
    }
}
fn k162(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = micromath::F32(a).sqrt().0;
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k163(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::portable::kernels::sqrt(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k164(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::unfused::kernels::sqrt(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k165(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::relaxed::kernels::sqrt(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k166(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.sqrt();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k167(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.sqrt();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k168(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.sqrt();
        r.store(o);
    }
}
fn k169(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.floor();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k170(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.floor();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k171(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::floorf(a);
    }
}
fn k172(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::floorf(a);
    }
}
fn k173(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = libm::floorf(a);
    }
}
fn k174(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = fpmath::floor(a);
    }
}
fn k175(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = micromath::F32(a).floor().0;
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k176(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::portable::kernels::floor(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k177(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::unfused::kernels::floor(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k178(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::relaxed::kernels::floor(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k179(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.floor();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k180(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.floor();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k181(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.floor();
        r.store(o);
    }
}
fn k182(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.ceil();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k183(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.ceil();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k184(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::ceilf(a);
    }
}
fn k185(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::ceilf(a);
    }
}
fn k186(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = libm::ceilf(a);
    }
}
fn k187(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = fpmath::ceil(a);
    }
}
fn k188(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = micromath::F32(a).ceil().0;
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k189(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::portable::kernels::ceil(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k190(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::unfused::kernels::ceil(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k191(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::relaxed::kernels::ceil(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k192(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.ceil();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k193(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.ceil();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k194(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.ceil();
        r.store(o);
    }
}
fn k195(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.round();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k196(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.round();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k197(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::roundf(a);
    }
}
fn k198(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::roundf(a);
    }
}
fn k199(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = libm::roundf(a);
    }
}
fn k200(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = fpmath::round(a);
    }
}
fn k201(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = micromath::F32(a).round().0;
    }
}
fn k202(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = metallic::roundf(a);
    }
}
fn k203(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.round_ties_even();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k204(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.round_ties_even();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k205(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::roundevenf(a);
    }
}
fn k206(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::roundevenf(a);
    }
}
fn k207(x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = libm::rintf(a);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k208(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::portable::kernels::roundeven(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k209(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::unfused::kernels::roundeven(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k210(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = magemath::relaxed::kernels::roundeven(t, a);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x4))]
fn k211(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
        .zip(y.as_chunks::<4>().0.iter())
    {
        let a = f32x4::load_t(t, a);
        let b = f32x4::load_t(t, b);
        let _ = b;
        let r = a.round();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn k212(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
        .zip(y.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = a.round();
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x16))]
fn k213(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    let _ = y;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 16, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0.iter())
        .zip(y.as_chunks::<16>().0.iter())
    {
        let a = f32x16::load_t(t, a);
        let b = f32x16::load_t(t, b);
        let _ = b;
        let r = a.round();
        r.store(o);
    }
}
fn k214(x: &[f64], y: &[f64], out: &mut [f64]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.log2();
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k215(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = a.log2();
    }
}
fn k216(x: &[f64], y: &[f64], out: &mut [f64]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::log2(a);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)))]
fn k217(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    let _ = y;
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::log2(a);
    }
}
fn k218(x: &[f64], y: &[f64], out: &mut [f64]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = libm::log2(a);
    }
}
fn k219(x: &[f64], y: &[f64], out: &mut [f64]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = fpmath::log2(a);
    }
}
fn k220(x: &[f64], y: &[f64], out: &mut [f64]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = metallic::log2(a);
    }
}
fn k221(x: &[f64], y: &[f64], out: &mut [f64]) {
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        let _ = b;
        *o = zenmath::scalar::log2_lowp_f64(a);
    }
}
pub fn cases32() -> Vec<Case<f32>> {
    let mut v = Vec::new();
    v.push(Case {
        op: "powf",
        label: "std/baseline",
        run: Box::new(k0),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "powf",
            label: "std_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k1_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "powf",
            label: "std_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k1_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "powf",
            label: "std_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k1_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "powf",
            label: "std_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k1_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "powf",
            label: "std_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k1_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "powf",
            label: "zenmath_correct_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k2_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "powf",
            label: "zenmath_correct_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k2_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "powf",
            label: "zenmath_correct_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k2_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "powf",
            label: "zenmath_correct_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k2_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "powf",
            label: "zenmath_correct_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k2_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "powf",
        label: "zenmath_correct/baseline",
        run: Box::new(k3),
    });
    v.push(Case {
        op: "powf",
        label: "libm/baseline",
        run: Box::new(k4),
    });
    v.push(Case {
        op: "powf",
        label: "fpmath/baseline",
        run: Box::new(k5),
    });
    v.push(Case {
        op: "powf",
        label: "micromath/baseline",
        run: Box::new(k6),
    });
    v.push(Case {
        op: "powf",
        label: "metallic/baseline",
        run: Box::new(k7),
    });
    v.push(Case {
        op: "powf",
        label: "zenmath_portable_midp/baseline",
        run: Box::new(k8),
    });
    v.push(Case {
        op: "powf",
        label: "zenmath_unfused_midp/baseline",
        run: Box::new(k9),
    });
    v.push(Case {
        op: "powf",
        label: "zenmath_portable_lowp/baseline",
        run: Box::new(k10),
    });
    v.push(Case {
        op: "powf",
        label: "zenmath_unfused_lowp/baseline",
        run: Box::new(k11),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "powf",
            label: "magemath_correct_proof/scalar/x8",
            run: Box::new(move |x, y, o| k12_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "powf",
            label: "magemath_correct_proof/v3/x8",
            run: Box::new(move |x, y, o| k12_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "powf",
            label: "magemath_correct_proof/v4x/x8",
            run: Box::new(move |x, y, o| k12_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "powf",
            label: "magemath_correct_proof/neon/x8",
            run: Box::new(move |x, y, o| k12_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "powf",
            label: "magemath_correct_proof/wasm128/x8",
            run: Box::new(move |x, y, o| k12_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "powf",
        label: "magemath_correct_dispatch/runtime/x8",
        run: Box::new(k13),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "powf",
            label: "magemath_policy_midp/scalar/x8",
            run: Box::new(move |x, y, o| k14_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "powf",
            label: "magemath_policy_midp/v3/x8",
            run: Box::new(move |x, y, o| k14_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "powf",
            label: "magemath_policy_midp/v4x/x8",
            run: Box::new(move |x, y, o| k14_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "powf",
            label: "magemath_policy_midp/neon/x8",
            run: Box::new(move |x, y, o| k14_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "powf",
            label: "magemath_policy_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k14_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "powf",
            label: "magemath_policy_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k15_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "powf",
            label: "magemath_policy_lowp/v3/x8",
            run: Box::new(move |x, y, o| k15_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "powf",
            label: "magemath_policy_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k15_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "powf",
            label: "magemath_policy_lowp/neon/x8",
            run: Box::new(move |x, y, o| k15_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "powf",
            label: "magemath_policy_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k15_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "pow_fixed",
        label: "std/baseline",
        run: Box::new(k16),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "std_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k17_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "std_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k17_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "std_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k17_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "std_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k17_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "std_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k17_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "zenmath_correct_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k18_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "zenmath_correct_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k18_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "zenmath_correct_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k18_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "zenmath_correct_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k18_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "zenmath_correct_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k18_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "pow_fixed",
        label: "zenmath_correct/baseline",
        run: Box::new(k19),
    });
    v.push(Case {
        op: "pow_fixed",
        label: "libm/baseline",
        run: Box::new(k20),
    });
    v.push(Case {
        op: "pow_fixed",
        label: "fpmath/baseline",
        run: Box::new(k21),
    });
    v.push(Case {
        op: "pow_fixed",
        label: "micromath/baseline",
        run: Box::new(k22),
    });
    v.push(Case {
        op: "pow_fixed",
        label: "metallic/baseline",
        run: Box::new(k23),
    });
    v.push(Case {
        op: "pow_fixed",
        label: "zenmath_portable_midp/baseline",
        run: Box::new(k24),
    });
    v.push(Case {
        op: "pow_fixed",
        label: "zenmath_unfused_midp/baseline",
        run: Box::new(k25),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_portable_midp/scalar/x8",
            run: Box::new(move |x, y, o| k26_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_portable_midp/v3/x8",
            run: Box::new(move |x, y, o| k26_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_portable_midp/v4x/x8",
            run: Box::new(move |x, y, o| k26_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_portable_midp/neon/x8",
            run: Box::new(move |x, y, o| k26_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_portable_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k26_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_unfused_midp/scalar/x8",
            run: Box::new(move |x, y, o| k27_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_unfused_midp/v3/x8",
            run: Box::new(move |x, y, o| k27_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_unfused_midp/v4x/x8",
            run: Box::new(move |x, y, o| k27_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_unfused_midp/neon/x8",
            run: Box::new(move |x, y, o| k27_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_unfused_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k27_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_relaxed_midp/scalar/x8",
            run: Box::new(move |x, y, o| k28_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_relaxed_midp/v3/x8",
            run: Box::new(move |x, y, o| k28_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_relaxed_midp/v4x/x8",
            run: Box::new(move |x, y, o| k28_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_relaxed_midp/neon/x8",
            run: Box::new(move |x, y, o| k28_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_relaxed_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k28_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "pow_fixed",
        label: "zenmath_portable_lowp/baseline",
        run: Box::new(k29),
    });
    v.push(Case {
        op: "pow_fixed",
        label: "zenmath_unfused_lowp/baseline",
        run: Box::new(k30),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_portable_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k31_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_portable_lowp/v3/x8",
            run: Box::new(move |x, y, o| k31_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_portable_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k31_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_portable_lowp/neon/x8",
            run: Box::new(move |x, y, o| k31_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_portable_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k31_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_unfused_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_unfused_lowp/v3/x8",
            run: Box::new(move |x, y, o| k32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_unfused_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_unfused_lowp/neon/x8",
            run: Box::new(move |x, y, o| k32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_unfused_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_relaxed_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k33_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_relaxed_lowp/v3/x8",
            run: Box::new(move |x, y, o| k33_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_relaxed_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k33_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_relaxed_lowp/neon/x8",
            run: Box::new(move |x, y, o| k33_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_relaxed_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k33_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_correct_proof/scalar/x8",
            run: Box::new(move |x, y, o| k34_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_correct_proof/v3/x8",
            run: Box::new(move |x, y, o| k34_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_correct_proof/v4x/x8",
            run: Box::new(move |x, y, o| k34_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_correct_proof/neon/x8",
            run: Box::new(move |x, y, o| k34_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magemath_correct_proof/wasm128/x8",
            run: Box::new(move |x, y, o| k34_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "pow_fixed",
        label: "magemath_correct_dispatch/runtime/x8",
        run: Box::new(k35),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_midp/scalar/x4",
            run: Box::new(move |x, y, o| k36_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_midp/v3/x4",
            run: Box::new(move |x, y, o| k36_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_midp/v4x/x4",
            run: Box::new(move |x, y, o| k36_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_midp/neon/x4",
            run: Box::new(move |x, y, o| k36_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_midp/wasm128/x4",
            run: Box::new(move |x, y, o| k36_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_lowp/scalar/x4",
            run: Box::new(move |x, y, o| k37_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_lowp/v3/x4",
            run: Box::new(move |x, y, o| k37_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_lowp/v4x/x4",
            run: Box::new(move |x, y, o| k37_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_lowp/neon/x4",
            run: Box::new(move |x, y, o| k37_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_lowp/wasm128/x4",
            run: Box::new(move |x, y, o| k37_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_midp/scalar/x8",
            run: Box::new(move |x, y, o| k38_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_midp/v3/x8",
            run: Box::new(move |x, y, o| k38_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_midp/v4x/x8",
            run: Box::new(move |x, y, o| k38_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_midp/neon/x8",
            run: Box::new(move |x, y, o| k38_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k38_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k39_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_lowp/v3/x8",
            run: Box::new(move |x, y, o| k39_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k39_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_lowp/neon/x8",
            run: Box::new(move |x, y, o| k39_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k39_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_midp/scalar/x16",
            run: Box::new(move |x, y, o| k40_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_midp/v3/x16",
            run: Box::new(move |x, y, o| k40_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_midp/v4x/x16",
            run: Box::new(move |x, y, o| k40_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_midp/neon/x16",
            run: Box::new(move |x, y, o| k40_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_midp/wasm128/x16",
            run: Box::new(move |x, y, o| k40_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_lowp/scalar/x16",
            run: Box::new(move |x, y, o| k41_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_lowp/v3/x16",
            run: Box::new(move |x, y, o| k41_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_lowp/v4x/x16",
            run: Box::new(move |x, y, o| k41_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_lowp/neon/x16",
            run: Box::new(move |x, y, o| k41_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "pow_fixed",
            label: "magetypes_lowp/wasm128/x16",
            run: Box::new(move |x, y, o| k41_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "exp",
        label: "std/baseline",
        run: Box::new(k42),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "std_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k43_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "std_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k43_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "std_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k43_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "std_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k43_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "std_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k43_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "zenmath_correct_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k44_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "zenmath_correct_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k44_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "zenmath_correct_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k44_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "zenmath_correct_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k44_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "zenmath_correct_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k44_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "exp",
        label: "zenmath_correct/baseline",
        run: Box::new(k45),
    });
    v.push(Case {
        op: "exp",
        label: "libm/baseline",
        run: Box::new(k46),
    });
    v.push(Case {
        op: "exp",
        label: "fpmath/baseline",
        run: Box::new(k47),
    });
    v.push(Case {
        op: "exp",
        label: "micromath/baseline",
        run: Box::new(k48),
    });
    v.push(Case {
        op: "exp",
        label: "metallic/baseline",
        run: Box::new(k49),
    });
    v.push(Case {
        op: "exp",
        label: "zenmath_portable_midp/baseline",
        run: Box::new(k50),
    });
    v.push(Case {
        op: "exp",
        label: "zenmath_unfused_midp/baseline",
        run: Box::new(k51),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_midp/scalar/x8",
            run: Box::new(move |x, y, o| k52_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_midp/v3/x8",
            run: Box::new(move |x, y, o| k52_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_midp/v4x/x8",
            run: Box::new(move |x, y, o| k52_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_midp/neon/x8",
            run: Box::new(move |x, y, o| k52_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k52_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_midp/scalar/x8",
            run: Box::new(move |x, y, o| k53_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_midp/v3/x8",
            run: Box::new(move |x, y, o| k53_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_midp/v4x/x8",
            run: Box::new(move |x, y, o| k53_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_midp/neon/x8",
            run: Box::new(move |x, y, o| k53_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k53_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_midp/scalar/x8",
            run: Box::new(move |x, y, o| k54_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_midp/v3/x8",
            run: Box::new(move |x, y, o| k54_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_midp/v4x/x8",
            run: Box::new(move |x, y, o| k54_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_midp/neon/x8",
            run: Box::new(move |x, y, o| k54_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k54_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "exp",
        label: "zenmath_portable_lowp/baseline",
        run: Box::new(k55),
    });
    v.push(Case {
        op: "exp",
        label: "zenmath_unfused_lowp/baseline",
        run: Box::new(k56),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k57_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_lowp/v3/x8",
            run: Box::new(move |x, y, o| k57_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k57_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_lowp/neon/x8",
            run: Box::new(move |x, y, o| k57_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k57_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k58_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_lowp/v3/x8",
            run: Box::new(move |x, y, o| k58_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k58_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_lowp/neon/x8",
            run: Box::new(move |x, y, o| k58_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k58_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k59_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_lowp/v3/x8",
            run: Box::new(move |x, y, o| k59_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k59_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_lowp/neon/x8",
            run: Box::new(move |x, y, o| k59_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k59_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_correct_proof/scalar/x8",
            run: Box::new(move |x, y, o| k60_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_correct_proof/v3/x8",
            run: Box::new(move |x, y, o| k60_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_correct_proof/v4x/x8",
            run: Box::new(move |x, y, o| k60_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_correct_proof/neon/x8",
            run: Box::new(move |x, y, o| k60_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_correct_proof/wasm128/x8",
            run: Box::new(move |x, y, o| k60_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "exp",
        label: "magemath_correct_dispatch/runtime/x8",
        run: Box::new(k61),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/scalar/x4",
            run: Box::new(move |x, y, o| k62_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/v3/x4",
            run: Box::new(move |x, y, o| k62_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/v4x/x4",
            run: Box::new(move |x, y, o| k62_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/neon/x4",
            run: Box::new(move |x, y, o| k62_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/wasm128/x4",
            run: Box::new(move |x, y, o| k62_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/scalar/x4",
            run: Box::new(move |x, y, o| k63_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/v3/x4",
            run: Box::new(move |x, y, o| k63_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/v4x/x4",
            run: Box::new(move |x, y, o| k63_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/neon/x4",
            run: Box::new(move |x, y, o| k63_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/wasm128/x4",
            run: Box::new(move |x, y, o| k63_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/scalar/x4",
            run: Box::new(move |x, y, o| k64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/v3/x4",
            run: Box::new(move |x, y, o| k64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/v4x/x4",
            run: Box::new(move |x, y, o| k64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/neon/x4",
            run: Box::new(move |x, y, o| k64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/wasm128/x4",
            run: Box::new(move |x, y, o| k64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/scalar/x8",
            run: Box::new(move |x, y, o| k65_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/v3/x8",
            run: Box::new(move |x, y, o| k65_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/v4x/x8",
            run: Box::new(move |x, y, o| k65_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/neon/x8",
            run: Box::new(move |x, y, o| k65_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k65_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k66_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/v3/x8",
            run: Box::new(move |x, y, o| k66_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k66_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/neon/x8",
            run: Box::new(move |x, y, o| k66_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k66_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/scalar/x8",
            run: Box::new(move |x, y, o| k67_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/v3/x8",
            run: Box::new(move |x, y, o| k67_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/v4x/x8",
            run: Box::new(move |x, y, o| k67_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/neon/x8",
            run: Box::new(move |x, y, o| k67_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/wasm128/x8",
            run: Box::new(move |x, y, o| k67_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/scalar/x16",
            run: Box::new(move |x, y, o| k68_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/v3/x16",
            run: Box::new(move |x, y, o| k68_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/v4x/x16",
            run: Box::new(move |x, y, o| k68_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/neon/x16",
            run: Box::new(move |x, y, o| k68_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/wasm128/x16",
            run: Box::new(move |x, y, o| k68_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/scalar/x16",
            run: Box::new(move |x, y, o| k69_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/v3/x16",
            run: Box::new(move |x, y, o| k69_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/v4x/x16",
            run: Box::new(move |x, y, o| k69_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/neon/x16",
            run: Box::new(move |x, y, o| k69_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/wasm128/x16",
            run: Box::new(move |x, y, o| k69_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/scalar/x16",
            run: Box::new(move |x, y, o| k70_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/v3/x16",
            run: Box::new(move |x, y, o| k70_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/v4x/x16",
            run: Box::new(move |x, y, o| k70_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/neon/x16",
            run: Box::new(move |x, y, o| k70_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/wasm128/x16",
            run: Box::new(move |x, y, o| k70_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "ln",
        label: "std/baseline",
        run: Box::new(k71),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "std_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k72_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "std_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k72_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "std_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k72_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "std_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k72_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "std_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k72_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "zenmath_correct_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k73_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "zenmath_correct_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k73_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "zenmath_correct_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k73_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "zenmath_correct_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k73_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "zenmath_correct_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k73_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "ln",
        label: "zenmath_correct/baseline",
        run: Box::new(k74),
    });
    v.push(Case {
        op: "ln",
        label: "libm/baseline",
        run: Box::new(k75),
    });
    v.push(Case {
        op: "ln",
        label: "fpmath/baseline",
        run: Box::new(k76),
    });
    v.push(Case {
        op: "ln",
        label: "micromath/baseline",
        run: Box::new(k77),
    });
    v.push(Case {
        op: "ln",
        label: "metallic/baseline",
        run: Box::new(k78),
    });
    v.push(Case {
        op: "ln",
        label: "zenmath_portable_midp/baseline",
        run: Box::new(k79),
    });
    v.push(Case {
        op: "ln",
        label: "zenmath_unfused_midp/baseline",
        run: Box::new(k80),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_midp/scalar/x8",
            run: Box::new(move |x, y, o| k81_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_midp/v3/x8",
            run: Box::new(move |x, y, o| k81_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_midp/v4x/x8",
            run: Box::new(move |x, y, o| k81_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_midp/neon/x8",
            run: Box::new(move |x, y, o| k81_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k81_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_midp/scalar/x8",
            run: Box::new(move |x, y, o| k82_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_midp/v3/x8",
            run: Box::new(move |x, y, o| k82_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_midp/v4x/x8",
            run: Box::new(move |x, y, o| k82_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_midp/neon/x8",
            run: Box::new(move |x, y, o| k82_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k82_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_midp/scalar/x8",
            run: Box::new(move |x, y, o| k83_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_midp/v3/x8",
            run: Box::new(move |x, y, o| k83_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_midp/v4x/x8",
            run: Box::new(move |x, y, o| k83_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_midp/neon/x8",
            run: Box::new(move |x, y, o| k83_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k83_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "ln",
        label: "zenmath_portable_lowp/baseline",
        run: Box::new(k84),
    });
    v.push(Case {
        op: "ln",
        label: "zenmath_unfused_lowp/baseline",
        run: Box::new(k85),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k86_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_lowp/v3/x8",
            run: Box::new(move |x, y, o| k86_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k86_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_lowp/neon/x8",
            run: Box::new(move |x, y, o| k86_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k86_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k87_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_lowp/v3/x8",
            run: Box::new(move |x, y, o| k87_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k87_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_lowp/neon/x8",
            run: Box::new(move |x, y, o| k87_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k87_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k88_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_lowp/v3/x8",
            run: Box::new(move |x, y, o| k88_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k88_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_lowp/neon/x8",
            run: Box::new(move |x, y, o| k88_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k88_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_correct_proof/scalar/x8",
            run: Box::new(move |x, y, o| k89_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_correct_proof/v3/x8",
            run: Box::new(move |x, y, o| k89_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_correct_proof/v4x/x8",
            run: Box::new(move |x, y, o| k89_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_correct_proof/neon/x8",
            run: Box::new(move |x, y, o| k89_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_correct_proof/wasm128/x8",
            run: Box::new(move |x, y, o| k89_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "ln",
        label: "magemath_correct_dispatch/runtime/x8",
        run: Box::new(k90),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/scalar/x4",
            run: Box::new(move |x, y, o| k91_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/v3/x4",
            run: Box::new(move |x, y, o| k91_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/v4x/x4",
            run: Box::new(move |x, y, o| k91_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/neon/x4",
            run: Box::new(move |x, y, o| k91_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/wasm128/x4",
            run: Box::new(move |x, y, o| k91_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/scalar/x4",
            run: Box::new(move |x, y, o| k92_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/v3/x4",
            run: Box::new(move |x, y, o| k92_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/v4x/x4",
            run: Box::new(move |x, y, o| k92_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/neon/x4",
            run: Box::new(move |x, y, o| k92_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/wasm128/x4",
            run: Box::new(move |x, y, o| k92_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/scalar/x4",
            run: Box::new(move |x, y, o| k93_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/v3/x4",
            run: Box::new(move |x, y, o| k93_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/v4x/x4",
            run: Box::new(move |x, y, o| k93_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/neon/x4",
            run: Box::new(move |x, y, o| k93_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/wasm128/x4",
            run: Box::new(move |x, y, o| k93_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/scalar/x8",
            run: Box::new(move |x, y, o| k94_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/v3/x8",
            run: Box::new(move |x, y, o| k94_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/v4x/x8",
            run: Box::new(move |x, y, o| k94_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/neon/x8",
            run: Box::new(move |x, y, o| k94_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k94_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k95_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/v3/x8",
            run: Box::new(move |x, y, o| k95_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k95_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/neon/x8",
            run: Box::new(move |x, y, o| k95_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k95_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/scalar/x8",
            run: Box::new(move |x, y, o| k96_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/v3/x8",
            run: Box::new(move |x, y, o| k96_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/v4x/x8",
            run: Box::new(move |x, y, o| k96_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/neon/x8",
            run: Box::new(move |x, y, o| k96_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/wasm128/x8",
            run: Box::new(move |x, y, o| k96_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/scalar/x16",
            run: Box::new(move |x, y, o| k97_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/v3/x16",
            run: Box::new(move |x, y, o| k97_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/v4x/x16",
            run: Box::new(move |x, y, o| k97_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/neon/x16",
            run: Box::new(move |x, y, o| k97_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/wasm128/x16",
            run: Box::new(move |x, y, o| k97_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/scalar/x16",
            run: Box::new(move |x, y, o| k98_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/v3/x16",
            run: Box::new(move |x, y, o| k98_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/v4x/x16",
            run: Box::new(move |x, y, o| k98_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/neon/x16",
            run: Box::new(move |x, y, o| k98_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/wasm128/x16",
            run: Box::new(move |x, y, o| k98_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/scalar/x16",
            run: Box::new(move |x, y, o| k99_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/v3/x16",
            run: Box::new(move |x, y, o| k99_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/v4x/x16",
            run: Box::new(move |x, y, o| k99_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/neon/x16",
            run: Box::new(move |x, y, o| k99_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/wasm128/x16",
            run: Box::new(move |x, y, o| k99_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "log2",
        label: "std/baseline",
        run: Box::new(k100),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "std_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k101_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "std_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k101_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "std_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k101_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "std_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k101_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "std_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k101_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "zenmath_correct_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k102_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "zenmath_correct_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k102_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "zenmath_correct_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k102_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "zenmath_correct_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k102_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "zenmath_correct_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k102_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "log2",
        label: "zenmath_correct/baseline",
        run: Box::new(k103),
    });
    v.push(Case {
        op: "log2",
        label: "libm/baseline",
        run: Box::new(k104),
    });
    v.push(Case {
        op: "log2",
        label: "fpmath/baseline",
        run: Box::new(k105),
    });
    v.push(Case {
        op: "log2",
        label: "micromath/baseline",
        run: Box::new(k106),
    });
    v.push(Case {
        op: "log2",
        label: "metallic/baseline",
        run: Box::new(k107),
    });
    v.push(Case {
        op: "log2",
        label: "zenmath_portable_midp/baseline",
        run: Box::new(k108),
    });
    v.push(Case {
        op: "log2",
        label: "zenmath_unfused_midp/baseline",
        run: Box::new(k109),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_midp/scalar/x8",
            run: Box::new(move |x, y, o| k110_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_midp/v3/x8",
            run: Box::new(move |x, y, o| k110_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_midp/v4x/x8",
            run: Box::new(move |x, y, o| k110_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_midp/neon/x8",
            run: Box::new(move |x, y, o| k110_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k110_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_midp/scalar/x8",
            run: Box::new(move |x, y, o| k111_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_midp/v3/x8",
            run: Box::new(move |x, y, o| k111_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_midp/v4x/x8",
            run: Box::new(move |x, y, o| k111_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_midp/neon/x8",
            run: Box::new(move |x, y, o| k111_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k111_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_midp/scalar/x8",
            run: Box::new(move |x, y, o| k112_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_midp/v3/x8",
            run: Box::new(move |x, y, o| k112_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_midp/v4x/x8",
            run: Box::new(move |x, y, o| k112_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_midp/neon/x8",
            run: Box::new(move |x, y, o| k112_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k112_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "log2",
        label: "zenmath_portable_lowp/baseline",
        run: Box::new(k113),
    });
    v.push(Case {
        op: "log2",
        label: "zenmath_unfused_lowp/baseline",
        run: Box::new(k114),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k115_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_lowp/v3/x8",
            run: Box::new(move |x, y, o| k115_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k115_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_lowp/neon/x8",
            run: Box::new(move |x, y, o| k115_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k115_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k116_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_lowp/v3/x8",
            run: Box::new(move |x, y, o| k116_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k116_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_lowp/neon/x8",
            run: Box::new(move |x, y, o| k116_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k116_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k117_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_lowp/v3/x8",
            run: Box::new(move |x, y, o| k117_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k117_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_lowp/neon/x8",
            run: Box::new(move |x, y, o| k117_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k117_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_correct_proof/scalar/x8",
            run: Box::new(move |x, y, o| k118_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_correct_proof/v3/x8",
            run: Box::new(move |x, y, o| k118_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_correct_proof/v4x/x8",
            run: Box::new(move |x, y, o| k118_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_correct_proof/neon/x8",
            run: Box::new(move |x, y, o| k118_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_correct_proof/wasm128/x8",
            run: Box::new(move |x, y, o| k118_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "log2",
        label: "magemath_correct_dispatch/runtime/x8",
        run: Box::new(k119),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/scalar/x4",
            run: Box::new(move |x, y, o| k120_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/v3/x4",
            run: Box::new(move |x, y, o| k120_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/v4x/x4",
            run: Box::new(move |x, y, o| k120_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/neon/x4",
            run: Box::new(move |x, y, o| k120_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/wasm128/x4",
            run: Box::new(move |x, y, o| k120_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/scalar/x4",
            run: Box::new(move |x, y, o| k121_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/v3/x4",
            run: Box::new(move |x, y, o| k121_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/v4x/x4",
            run: Box::new(move |x, y, o| k121_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/neon/x4",
            run: Box::new(move |x, y, o| k121_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/wasm128/x4",
            run: Box::new(move |x, y, o| k121_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/scalar/x4",
            run: Box::new(move |x, y, o| k122_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/v3/x4",
            run: Box::new(move |x, y, o| k122_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/v4x/x4",
            run: Box::new(move |x, y, o| k122_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/neon/x4",
            run: Box::new(move |x, y, o| k122_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/wasm128/x4",
            run: Box::new(move |x, y, o| k122_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/scalar/x8",
            run: Box::new(move |x, y, o| k123_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/v3/x8",
            run: Box::new(move |x, y, o| k123_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/v4x/x8",
            run: Box::new(move |x, y, o| k123_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/neon/x8",
            run: Box::new(move |x, y, o| k123_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k123_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k124_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/v3/x8",
            run: Box::new(move |x, y, o| k124_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k124_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/neon/x8",
            run: Box::new(move |x, y, o| k124_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k124_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/scalar/x8",
            run: Box::new(move |x, y, o| k125_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/v3/x8",
            run: Box::new(move |x, y, o| k125_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/v4x/x8",
            run: Box::new(move |x, y, o| k125_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/neon/x8",
            run: Box::new(move |x, y, o| k125_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/wasm128/x8",
            run: Box::new(move |x, y, o| k125_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/scalar/x16",
            run: Box::new(move |x, y, o| k126_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/v3/x16",
            run: Box::new(move |x, y, o| k126_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/v4x/x16",
            run: Box::new(move |x, y, o| k126_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/neon/x16",
            run: Box::new(move |x, y, o| k126_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/wasm128/x16",
            run: Box::new(move |x, y, o| k126_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/scalar/x16",
            run: Box::new(move |x, y, o| k127_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/v3/x16",
            run: Box::new(move |x, y, o| k127_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/v4x/x16",
            run: Box::new(move |x, y, o| k127_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/neon/x16",
            run: Box::new(move |x, y, o| k127_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/wasm128/x16",
            run: Box::new(move |x, y, o| k127_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/scalar/x16",
            run: Box::new(move |x, y, o| k128_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/v3/x16",
            run: Box::new(move |x, y, o| k128_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/v4x/x16",
            run: Box::new(move |x, y, o| k128_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/neon/x16",
            run: Box::new(move |x, y, o| k128_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/wasm128/x16",
            run: Box::new(move |x, y, o| k128_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "log10",
        label: "std/baseline",
        run: Box::new(k129),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "std_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k130_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "std_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k130_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "std_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k130_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "std_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k130_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "std_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k130_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "zenmath_correct_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k131_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "zenmath_correct_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k131_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "zenmath_correct_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k131_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "zenmath_correct_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k131_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "zenmath_correct_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k131_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "log10",
        label: "zenmath_correct/baseline",
        run: Box::new(k132),
    });
    v.push(Case {
        op: "log10",
        label: "libm/baseline",
        run: Box::new(k133),
    });
    v.push(Case {
        op: "log10",
        label: "fpmath/baseline",
        run: Box::new(k134),
    });
    v.push(Case {
        op: "log10",
        label: "micromath/baseline",
        run: Box::new(k135),
    });
    v.push(Case {
        op: "log10",
        label: "metallic/baseline",
        run: Box::new(k136),
    });
    v.push(Case {
        op: "log10",
        label: "zenmath_portable_midp/baseline",
        run: Box::new(k137),
    });
    v.push(Case {
        op: "log10",
        label: "zenmath_unfused_midp/baseline",
        run: Box::new(k138),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_portable_midp/scalar/x8",
            run: Box::new(move |x, y, o| k139_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_portable_midp/v3/x8",
            run: Box::new(move |x, y, o| k139_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_portable_midp/v4x/x8",
            run: Box::new(move |x, y, o| k139_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_portable_midp/neon/x8",
            run: Box::new(move |x, y, o| k139_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_portable_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k139_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_unfused_midp/scalar/x8",
            run: Box::new(move |x, y, o| k140_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_unfused_midp/v3/x8",
            run: Box::new(move |x, y, o| k140_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_unfused_midp/v4x/x8",
            run: Box::new(move |x, y, o| k140_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_unfused_midp/neon/x8",
            run: Box::new(move |x, y, o| k140_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_unfused_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k140_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_relaxed_midp/scalar/x8",
            run: Box::new(move |x, y, o| k141_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_relaxed_midp/v3/x8",
            run: Box::new(move |x, y, o| k141_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_relaxed_midp/v4x/x8",
            run: Box::new(move |x, y, o| k141_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_relaxed_midp/neon/x8",
            run: Box::new(move |x, y, o| k141_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_relaxed_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k141_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "log10",
        label: "zenmath_portable_lowp/baseline",
        run: Box::new(k142),
    });
    v.push(Case {
        op: "log10",
        label: "zenmath_unfused_lowp/baseline",
        run: Box::new(k143),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_portable_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k144_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_portable_lowp/v3/x8",
            run: Box::new(move |x, y, o| k144_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_portable_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k144_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_portable_lowp/neon/x8",
            run: Box::new(move |x, y, o| k144_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_portable_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k144_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_unfused_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k145_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_unfused_lowp/v3/x8",
            run: Box::new(move |x, y, o| k145_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_unfused_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k145_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_unfused_lowp/neon/x8",
            run: Box::new(move |x, y, o| k145_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_unfused_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k145_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_relaxed_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k146_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_relaxed_lowp/v3/x8",
            run: Box::new(move |x, y, o| k146_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_relaxed_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k146_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_relaxed_lowp/neon/x8",
            run: Box::new(move |x, y, o| k146_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magemath_relaxed_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k146_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp/scalar/x4",
            run: Box::new(move |x, y, o| k147_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp/v3/x4",
            run: Box::new(move |x, y, o| k147_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp/v4x/x4",
            run: Box::new(move |x, y, o| k147_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp/neon/x4",
            run: Box::new(move |x, y, o| k147_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp/wasm128/x4",
            run: Box::new(move |x, y, o| k147_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_lowp/scalar/x4",
            run: Box::new(move |x, y, o| k148_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_lowp/v3/x4",
            run: Box::new(move |x, y, o| k148_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_lowp/v4x/x4",
            run: Box::new(move |x, y, o| k148_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_lowp/neon/x4",
            run: Box::new(move |x, y, o| k148_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_lowp/wasm128/x4",
            run: Box::new(move |x, y, o| k148_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp_precise/scalar/x4",
            run: Box::new(move |x, y, o| k149_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp_precise/v3/x4",
            run: Box::new(move |x, y, o| k149_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp_precise/v4x/x4",
            run: Box::new(move |x, y, o| k149_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp_precise/neon/x4",
            run: Box::new(move |x, y, o| k149_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp_precise/wasm128/x4",
            run: Box::new(move |x, y, o| k149_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp/scalar/x8",
            run: Box::new(move |x, y, o| k150_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp/v3/x8",
            run: Box::new(move |x, y, o| k150_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp/v4x/x8",
            run: Box::new(move |x, y, o| k150_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp/neon/x8",
            run: Box::new(move |x, y, o| k150_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp/wasm128/x8",
            run: Box::new(move |x, y, o| k150_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_lowp/scalar/x8",
            run: Box::new(move |x, y, o| k151_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_lowp/v3/x8",
            run: Box::new(move |x, y, o| k151_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_lowp/v4x/x8",
            run: Box::new(move |x, y, o| k151_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_lowp/neon/x8",
            run: Box::new(move |x, y, o| k151_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| k151_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp_precise/scalar/x8",
            run: Box::new(move |x, y, o| k152_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp_precise/v3/x8",
            run: Box::new(move |x, y, o| k152_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp_precise/v4x/x8",
            run: Box::new(move |x, y, o| k152_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp_precise/neon/x8",
            run: Box::new(move |x, y, o| k152_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp_precise/wasm128/x8",
            run: Box::new(move |x, y, o| k152_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp/scalar/x16",
            run: Box::new(move |x, y, o| k153_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp/v3/x16",
            run: Box::new(move |x, y, o| k153_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp/v4x/x16",
            run: Box::new(move |x, y, o| k153_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp/neon/x16",
            run: Box::new(move |x, y, o| k153_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp/wasm128/x16",
            run: Box::new(move |x, y, o| k153_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_lowp/scalar/x16",
            run: Box::new(move |x, y, o| k154_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_lowp/v3/x16",
            run: Box::new(move |x, y, o| k154_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_lowp/v4x/x16",
            run: Box::new(move |x, y, o| k154_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_lowp/neon/x16",
            run: Box::new(move |x, y, o| k154_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_lowp/wasm128/x16",
            run: Box::new(move |x, y, o| k154_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp_precise/scalar/x16",
            run: Box::new(move |x, y, o| k155_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp_precise/v3/x16",
            run: Box::new(move |x, y, o| k155_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp_precise/v4x/x16",
            run: Box::new(move |x, y, o| k155_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp_precise/neon/x16",
            run: Box::new(move |x, y, o| k155_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log10",
            label: "magetypes_midp_precise/wasm128/x16",
            run: Box::new(move |x, y, o| k155_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "sqrt",
        label: "std/baseline",
        run: Box::new(k156),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "std_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k157_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "std_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k157_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "std_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k157_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "std_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k157_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "std_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k157_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "zenmath_correct_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k158_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "zenmath_correct_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k158_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "zenmath_correct_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k158_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "zenmath_correct_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k158_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "zenmath_correct_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k158_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "sqrt",
        label: "zenmath_correct/baseline",
        run: Box::new(k159),
    });
    v.push(Case {
        op: "sqrt",
        label: "libm/baseline",
        run: Box::new(k160),
    });
    v.push(Case {
        op: "sqrt",
        label: "fpmath/baseline",
        run: Box::new(k161),
    });
    v.push(Case {
        op: "sqrt",
        label: "micromath/baseline",
        run: Box::new(k162),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_portable/scalar/x8",
            run: Box::new(move |x, y, o| k163_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_portable/v3/x8",
            run: Box::new(move |x, y, o| k163_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_portable/v4x/x8",
            run: Box::new(move |x, y, o| k163_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_portable/neon/x8",
            run: Box::new(move |x, y, o| k163_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_portable/wasm128/x8",
            run: Box::new(move |x, y, o| k163_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_unfused/scalar/x8",
            run: Box::new(move |x, y, o| k164_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_unfused/v3/x8",
            run: Box::new(move |x, y, o| k164_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_unfused/v4x/x8",
            run: Box::new(move |x, y, o| k164_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_unfused/neon/x8",
            run: Box::new(move |x, y, o| k164_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_unfused/wasm128/x8",
            run: Box::new(move |x, y, o| k164_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_relaxed/scalar/x8",
            run: Box::new(move |x, y, o| k165_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_relaxed/v3/x8",
            run: Box::new(move |x, y, o| k165_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_relaxed/v4x/x8",
            run: Box::new(move |x, y, o| k165_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_relaxed/neon/x8",
            run: Box::new(move |x, y, o| k165_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_relaxed/wasm128/x8",
            run: Box::new(move |x, y, o| k165_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/scalar/x4",
            run: Box::new(move |x, y, o| k166_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/v3/x4",
            run: Box::new(move |x, y, o| k166_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/v4x/x4",
            run: Box::new(move |x, y, o| k166_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/neon/x4",
            run: Box::new(move |x, y, o| k166_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/wasm128/x4",
            run: Box::new(move |x, y, o| k166_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/scalar/x8",
            run: Box::new(move |x, y, o| k167_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/v3/x8",
            run: Box::new(move |x, y, o| k167_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/v4x/x8",
            run: Box::new(move |x, y, o| k167_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/neon/x8",
            run: Box::new(move |x, y, o| k167_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/wasm128/x8",
            run: Box::new(move |x, y, o| k167_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/scalar/x16",
            run: Box::new(move |x, y, o| k168_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/v3/x16",
            run: Box::new(move |x, y, o| k168_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/v4x/x16",
            run: Box::new(move |x, y, o| k168_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/neon/x16",
            run: Box::new(move |x, y, o| k168_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/wasm128/x16",
            run: Box::new(move |x, y, o| k168_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "floor",
        label: "std/baseline",
        run: Box::new(k169),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "floor",
            label: "std_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k170_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "floor",
            label: "std_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k170_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "floor",
            label: "std_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k170_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "floor",
            label: "std_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k170_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "floor",
            label: "std_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k170_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "floor",
            label: "zenmath_correct_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k171_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "floor",
            label: "zenmath_correct_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k171_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "floor",
            label: "zenmath_correct_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k171_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "floor",
            label: "zenmath_correct_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k171_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "floor",
            label: "zenmath_correct_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k171_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "floor",
        label: "zenmath_correct/baseline",
        run: Box::new(k172),
    });
    v.push(Case {
        op: "floor",
        label: "libm/baseline",
        run: Box::new(k173),
    });
    v.push(Case {
        op: "floor",
        label: "fpmath/baseline",
        run: Box::new(k174),
    });
    v.push(Case {
        op: "floor",
        label: "micromath/baseline",
        run: Box::new(k175),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magemath_portable/scalar/x8",
            run: Box::new(move |x, y, o| k176_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "floor",
            label: "magemath_portable/v3/x8",
            run: Box::new(move |x, y, o| k176_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magemath_portable/v4x/x8",
            run: Box::new(move |x, y, o| k176_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magemath_portable/neon/x8",
            run: Box::new(move |x, y, o| k176_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "floor",
            label: "magemath_portable/wasm128/x8",
            run: Box::new(move |x, y, o| k176_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magemath_unfused/scalar/x8",
            run: Box::new(move |x, y, o| k177_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "floor",
            label: "magemath_unfused/v3/x8",
            run: Box::new(move |x, y, o| k177_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magemath_unfused/v4x/x8",
            run: Box::new(move |x, y, o| k177_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magemath_unfused/neon/x8",
            run: Box::new(move |x, y, o| k177_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "floor",
            label: "magemath_unfused/wasm128/x8",
            run: Box::new(move |x, y, o| k177_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magemath_relaxed/scalar/x8",
            run: Box::new(move |x, y, o| k178_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "floor",
            label: "magemath_relaxed/v3/x8",
            run: Box::new(move |x, y, o| k178_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magemath_relaxed/v4x/x8",
            run: Box::new(move |x, y, o| k178_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magemath_relaxed/neon/x8",
            run: Box::new(move |x, y, o| k178_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "floor",
            label: "magemath_relaxed/wasm128/x8",
            run: Box::new(move |x, y, o| k178_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magetypes_default/scalar/x4",
            run: Box::new(move |x, y, o| k179_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "floor",
            label: "magetypes_default/v3/x4",
            run: Box::new(move |x, y, o| k179_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magetypes_default/v4x/x4",
            run: Box::new(move |x, y, o| k179_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magetypes_default/neon/x4",
            run: Box::new(move |x, y, o| k179_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "floor",
            label: "magetypes_default/wasm128/x4",
            run: Box::new(move |x, y, o| k179_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magetypes_default/scalar/x8",
            run: Box::new(move |x, y, o| k180_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "floor",
            label: "magetypes_default/v3/x8",
            run: Box::new(move |x, y, o| k180_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magetypes_default/v4x/x8",
            run: Box::new(move |x, y, o| k180_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magetypes_default/neon/x8",
            run: Box::new(move |x, y, o| k180_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "floor",
            label: "magetypes_default/wasm128/x8",
            run: Box::new(move |x, y, o| k180_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magetypes_default/scalar/x16",
            run: Box::new(move |x, y, o| k181_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "floor",
            label: "magetypes_default/v3/x16",
            run: Box::new(move |x, y, o| k181_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magetypes_default/v4x/x16",
            run: Box::new(move |x, y, o| k181_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "floor",
            label: "magetypes_default/neon/x16",
            run: Box::new(move |x, y, o| k181_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "floor",
            label: "magetypes_default/wasm128/x16",
            run: Box::new(move |x, y, o| k181_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "ceil",
        label: "std/baseline",
        run: Box::new(k182),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "std_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k183_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ceil",
            label: "std_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k183_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "std_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k183_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "std_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k183_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ceil",
            label: "std_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k183_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "zenmath_correct_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k184_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ceil",
            label: "zenmath_correct_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k184_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "zenmath_correct_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k184_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "zenmath_correct_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k184_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ceil",
            label: "zenmath_correct_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k184_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "ceil",
        label: "zenmath_correct/baseline",
        run: Box::new(k185),
    });
    v.push(Case {
        op: "ceil",
        label: "libm/baseline",
        run: Box::new(k186),
    });
    v.push(Case {
        op: "ceil",
        label: "fpmath/baseline",
        run: Box::new(k187),
    });
    v.push(Case {
        op: "ceil",
        label: "micromath/baseline",
        run: Box::new(k188),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magemath_portable/scalar/x8",
            run: Box::new(move |x, y, o| k189_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ceil",
            label: "magemath_portable/v3/x8",
            run: Box::new(move |x, y, o| k189_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magemath_portable/v4x/x8",
            run: Box::new(move |x, y, o| k189_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magemath_portable/neon/x8",
            run: Box::new(move |x, y, o| k189_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ceil",
            label: "magemath_portable/wasm128/x8",
            run: Box::new(move |x, y, o| k189_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magemath_unfused/scalar/x8",
            run: Box::new(move |x, y, o| k190_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ceil",
            label: "magemath_unfused/v3/x8",
            run: Box::new(move |x, y, o| k190_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magemath_unfused/v4x/x8",
            run: Box::new(move |x, y, o| k190_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magemath_unfused/neon/x8",
            run: Box::new(move |x, y, o| k190_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ceil",
            label: "magemath_unfused/wasm128/x8",
            run: Box::new(move |x, y, o| k190_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magemath_relaxed/scalar/x8",
            run: Box::new(move |x, y, o| k191_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ceil",
            label: "magemath_relaxed/v3/x8",
            run: Box::new(move |x, y, o| k191_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magemath_relaxed/v4x/x8",
            run: Box::new(move |x, y, o| k191_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magemath_relaxed/neon/x8",
            run: Box::new(move |x, y, o| k191_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ceil",
            label: "magemath_relaxed/wasm128/x8",
            run: Box::new(move |x, y, o| k191_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magetypes_default/scalar/x4",
            run: Box::new(move |x, y, o| k192_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ceil",
            label: "magetypes_default/v3/x4",
            run: Box::new(move |x, y, o| k192_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magetypes_default/v4x/x4",
            run: Box::new(move |x, y, o| k192_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magetypes_default/neon/x4",
            run: Box::new(move |x, y, o| k192_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ceil",
            label: "magetypes_default/wasm128/x4",
            run: Box::new(move |x, y, o| k192_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magetypes_default/scalar/x8",
            run: Box::new(move |x, y, o| k193_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ceil",
            label: "magetypes_default/v3/x8",
            run: Box::new(move |x, y, o| k193_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magetypes_default/v4x/x8",
            run: Box::new(move |x, y, o| k193_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magetypes_default/neon/x8",
            run: Box::new(move |x, y, o| k193_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ceil",
            label: "magetypes_default/wasm128/x8",
            run: Box::new(move |x, y, o| k193_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magetypes_default/scalar/x16",
            run: Box::new(move |x, y, o| k194_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ceil",
            label: "magetypes_default/v3/x16",
            run: Box::new(move |x, y, o| k194_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magetypes_default/v4x/x16",
            run: Box::new(move |x, y, o| k194_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ceil",
            label: "magetypes_default/neon/x16",
            run: Box::new(move |x, y, o| k194_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ceil",
            label: "magetypes_default/wasm128/x16",
            run: Box::new(move |x, y, o| k194_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "round",
        label: "std/baseline",
        run: Box::new(k195),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "round",
            label: "std_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k196_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "round",
            label: "std_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k196_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "round",
            label: "std_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k196_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "round",
            label: "std_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k196_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "round",
            label: "std_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k196_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "round",
            label: "zenmath_correct_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k197_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "round",
            label: "zenmath_correct_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k197_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "round",
            label: "zenmath_correct_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k197_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "round",
            label: "zenmath_correct_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k197_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "round",
            label: "zenmath_correct_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k197_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "round",
        label: "zenmath_correct/baseline",
        run: Box::new(k198),
    });
    v.push(Case {
        op: "round",
        label: "libm/baseline",
        run: Box::new(k199),
    });
    v.push(Case {
        op: "round",
        label: "fpmath/baseline",
        run: Box::new(k200),
    });
    v.push(Case {
        op: "round",
        label: "micromath/baseline",
        run: Box::new(k201),
    });
    v.push(Case {
        op: "round",
        label: "metallic/baseline",
        run: Box::new(k202),
    });
    v.push(Case {
        op: "roundeven",
        label: "std/baseline",
        run: Box::new(k203),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "std_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k204_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "roundeven",
            label: "std_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k204_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "std_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k204_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "std_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k204_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "roundeven",
            label: "std_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k204_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "zenmath_correct_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k205_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "roundeven",
            label: "zenmath_correct_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k205_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "zenmath_correct_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k205_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "zenmath_correct_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k205_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "roundeven",
            label: "zenmath_correct_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k205_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "roundeven",
        label: "zenmath_correct/baseline",
        run: Box::new(k206),
    });
    v.push(Case {
        op: "roundeven",
        label: "libm/baseline",
        run: Box::new(k207),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magemath_portable/scalar/x8",
            run: Box::new(move |x, y, o| k208_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magemath_portable/v3/x8",
            run: Box::new(move |x, y, o| k208_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magemath_portable/v4x/x8",
            run: Box::new(move |x, y, o| k208_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magemath_portable/neon/x8",
            run: Box::new(move |x, y, o| k208_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magemath_portable/wasm128/x8",
            run: Box::new(move |x, y, o| k208_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magemath_unfused/scalar/x8",
            run: Box::new(move |x, y, o| k209_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magemath_unfused/v3/x8",
            run: Box::new(move |x, y, o| k209_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magemath_unfused/v4x/x8",
            run: Box::new(move |x, y, o| k209_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magemath_unfused/neon/x8",
            run: Box::new(move |x, y, o| k209_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magemath_unfused/wasm128/x8",
            run: Box::new(move |x, y, o| k209_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magemath_relaxed/scalar/x8",
            run: Box::new(move |x, y, o| k210_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magemath_relaxed/v3/x8",
            run: Box::new(move |x, y, o| k210_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magemath_relaxed/v4x/x8",
            run: Box::new(move |x, y, o| k210_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magemath_relaxed/neon/x8",
            run: Box::new(move |x, y, o| k210_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magemath_relaxed/wasm128/x8",
            run: Box::new(move |x, y, o| k210_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magetypes_default/scalar/x4",
            run: Box::new(move |x, y, o| k211_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magetypes_default/v3/x4",
            run: Box::new(move |x, y, o| k211_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magetypes_default/v4x/x4",
            run: Box::new(move |x, y, o| k211_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magetypes_default/neon/x4",
            run: Box::new(move |x, y, o| k211_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magetypes_default/wasm128/x4",
            run: Box::new(move |x, y, o| k211_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magetypes_default/scalar/x8",
            run: Box::new(move |x, y, o| k212_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magetypes_default/v3/x8",
            run: Box::new(move |x, y, o| k212_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magetypes_default/v4x/x8",
            run: Box::new(move |x, y, o| k212_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magetypes_default/neon/x8",
            run: Box::new(move |x, y, o| k212_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magetypes_default/wasm128/x8",
            run: Box::new(move |x, y, o| k212_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magetypes_default/scalar/x16",
            run: Box::new(move |x, y, o| k213_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magetypes_default/v3/x16",
            run: Box::new(move |x, y, o| k213_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magetypes_default/v4x/x16",
            run: Box::new(move |x, y, o| k213_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magetypes_default/neon/x16",
            run: Box::new(move |x, y, o| k213_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "roundeven",
            label: "magetypes_default/wasm128/x16",
            run: Box::new(move |x, y, o| k213_wasm128_t(t, x, y, o)),
        });
    }
    v
}
pub fn cases64() -> Vec<Case<f64>> {
    let mut v = Vec::new();
    v.push(Case {
        op: "log2_f64",
        label: "std/baseline",
        run: Box::new(k214),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2_f64",
            label: "std_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k215_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2_f64",
            label: "std_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k215_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2_f64",
            label: "std_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k215_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2_f64",
            label: "std_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k215_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2_f64",
            label: "std_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k215_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "log2_f64",
        label: "zenmath_correct/baseline",
        run: Box::new(k216),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2_f64",
            label: "zenmath_correct_loop_context/scalar/x1",
            run: Box::new(move |x, y, o| k217_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2_f64",
            label: "zenmath_correct_loop_context/v3/x1",
            run: Box::new(move |x, y, o| k217_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2_f64",
            label: "zenmath_correct_loop_context/v4x/x1",
            run: Box::new(move |x, y, o| k217_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2_f64",
            label: "zenmath_correct_loop_context/neon/x1",
            run: Box::new(move |x, y, o| k217_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2_f64",
            label: "zenmath_correct_loop_context/wasm128/x1",
            run: Box::new(move |x, y, o| k217_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "log2_f64",
        label: "libm/baseline",
        run: Box::new(k218),
    });
    v.push(Case {
        op: "log2_f64",
        label: "fpmath/baseline",
        run: Box::new(k219),
    });
    v.push(Case {
        op: "log2_f64",
        label: "metallic/baseline",
        run: Box::new(k220),
    });
    v.push(Case {
        op: "log2_f64",
        label: "zenmath_lowp/baseline",
        run: Box::new(k221),
    });
    v
}

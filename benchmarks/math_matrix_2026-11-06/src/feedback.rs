use crate::Case;
use archmage::prelude::*;
fn fb0(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = -a.abs();
        let b = 0.;
        let _ = b;
        a = a.exp();
        *z = a;
    }
}
fn fb1(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = -a.abs();
        let b = 0.;
        let _ = b;
        a = zenmath::scalar::expf(a);
        *z = a;
    }
}
fn fb2(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = -a.abs();
        let b = 0.;
        let _ = b;
        a = libm::expf(a);
        *z = a;
    }
}
fn fb3(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = -a.abs();
        let b = 0.;
        let _ = b;
        a = fpmath::exp(a);
        *z = a;
    }
}
fn fb4(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = -a.abs();
        let b = 0.;
        let _ = b;
        a = micromath::F32(a).exp().0;
        *z = a;
    }
}
fn fb5(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = -a.abs();
        let b = 0.;
        let _ = b;
        a = metallic::expf(a);
        *z = a;
    }
}
fn fb6(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = -a.abs();
        let b = 0.;
        let _ = b;
        a = zenmath::scalar::exp_midp_f32(a);
        *z = a;
    }
}
fn fb7(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = -a.abs();
        let b = 0.;
        let _ = b;
        a = zenmath::scalar::unfused::exp_midp_f32(a);
        *z = a;
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb8(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = -a.abs();
        a = magemath::portable::kernels::exp_midp(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb13(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = -a.abs();
        a = magemath::unfused::kernels::exp_midp(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb18(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = -a.abs();
        a = magemath::relaxed::kernels::exp_midp(t, a);
        a.store(z);
    }
}
fn fb23(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = -a.abs();
        let b = 0.;
        let _ = b;
        a = zenmath::scalar::exp_lowp_f32(a);
        *z = a;
    }
}
fn fb24(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = -a.abs();
        let b = 0.;
        let _ = b;
        a = zenmath::scalar::unfused::exp_lowp_f32(a);
        *z = a;
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb25(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = -a.abs();
        a = magemath::portable::kernels::exp_lowp(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb30(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = -a.abs();
        a = magemath::unfused::kernels::exp_lowp(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb35(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = -a.abs();
        a = magemath::relaxed::kernels::exp_lowp(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb40(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = -a.abs();
        a = f32x8::from_array_t(
            t,
            archmage::attuned!(magemath::portable::array::exp_f32x8(a.to_array())),
        );
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb45(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = -a.abs();
        a = a.exp_midp();
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb50(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = -a.abs();
        a = a.exp_lowp();
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb55(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = -a.abs();
        a = a.exp_midp_precise();
        a.store(z);
    }
}
fn fb60(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = a.ln();
        *z = a;
    }
}
fn fb61(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = zenmath::scalar::logf(a);
        *z = a;
    }
}
fn fb62(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = libm::logf(a);
        *z = a;
    }
}
fn fb63(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = fpmath::ln(a);
        *z = a;
    }
}
fn fb64(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = micromath::F32(a).ln().0;
        *z = a;
    }
}
fn fb65(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = metallic::logf(a);
        *z = a;
    }
}
fn fb66(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = zenmath::scalar::ln_midp_f32(a);
        *z = a;
    }
}
fn fb67(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = zenmath::scalar::unfused::ln_midp_f32(a);
        *z = a;
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb68(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = magemath::portable::kernels::ln_midp(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb73(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = magemath::unfused::kernels::ln_midp(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb78(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = magemath::relaxed::kernels::ln_midp(t, a);
        a.store(z);
    }
}
fn fb83(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = zenmath::scalar::ln_lowp_f32(a);
        *z = a;
    }
}
fn fb84(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = zenmath::scalar::unfused::ln_lowp_f32(a);
        *z = a;
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb85(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = magemath::portable::kernels::ln_lowp(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb90(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = magemath::unfused::kernels::ln_lowp(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb95(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = magemath::relaxed::kernels::ln_lowp(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb100(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = f32x8::from_array_t(
            t,
            archmage::attuned!(magemath::portable::array::ln_f32x8(a.to_array())),
        );
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb105(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = a.ln_midp();
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb110(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = a.ln_lowp();
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb115(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = a.ln_midp_precise();
        a.store(z);
    }
}
fn fb120(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = a.log2();
        *z = a;
    }
}
fn fb121(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = zenmath::scalar::log2f(a);
        *z = a;
    }
}
fn fb122(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = libm::log2f(a);
        *z = a;
    }
}
fn fb123(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = fpmath::log2(a);
        *z = a;
    }
}
fn fb124(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = micromath::F32(a).log2().0;
        *z = a;
    }
}
fn fb125(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = metallic::log2f(a);
        *z = a;
    }
}
fn fb126(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = zenmath::scalar::log2_midp_f32(a);
        *z = a;
    }
}
fn fb127(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = zenmath::scalar::unfused::log2_midp_f32(a);
        *z = a;
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb128(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = magemath::portable::kernels::log2_midp(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb133(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = magemath::unfused::kernels::log2_midp(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb138(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = magemath::relaxed::kernels::log2_midp(t, a);
        a.store(z);
    }
}
fn fb143(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = zenmath::scalar::log2_lowp_f32(a);
        *z = a;
    }
}
fn fb144(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 1.1;
        let b = 0.;
        let _ = b;
        a = zenmath::scalar::unfused::log2_lowp_f32(a);
        *z = a;
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb145(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = magemath::portable::kernels::log2_lowp(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb150(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = magemath::unfused::kernels::log2_lowp(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb155(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = magemath::relaxed::kernels::log2_lowp(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb160(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = f32x8::from_array_t(
            t,
            archmage::attuned!(magemath::portable::array::log2_f32x8(a.to_array())),
        );
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb165(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = a.log2_midp();
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb170(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = a.log2_lowp();
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb175(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 1.1);
        a = a.log2_midp_precise();
        a.store(z);
    }
}
fn fb180(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 0.1;
        let b = 0.;
        let _ = b;
        a = a.sqrt();
        *z = a;
    }
}
fn fb181(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 0.1;
        let b = 0.;
        let _ = b;
        a = zenmath::scalar::sqrtf(a);
        *z = a;
    }
}
fn fb182(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 0.1;
        let b = 0.;
        let _ = b;
        a = libm::sqrtf(a);
        *z = a;
    }
}
fn fb183(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 0.1;
        let b = 0.;
        let _ = b;
        a = fpmath::sqrt(a);
        *z = a;
    }
}
fn fb184(x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = x[0];
    for z in o {
        a = a.abs() + 0.1;
        let b = 0.;
        let _ = b;
        a = micromath::F32(a).sqrt().0;
        *z = a;
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb185(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 0.1);
        a = magemath::portable::kernels::sqrt(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb190(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 0.1);
        a = magemath::unfused::kernels::sqrt(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb195(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 0.1);
        a = magemath::relaxed::kernels::sqrt(t, a);
        a.store(z);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
fn fb200(t: Token, x: &[f32], _y: &[f32], o: &mut [f32]) {
    assert_eq!(o.len(), 1024);
    let mut a = f32x8::load_t(t, x[..8].try_into().unwrap());
    for z in o.as_chunks_mut::<8>().0 {
        a = a.abs() + f32x8::splat_t(t, 0.1);
        a = a.sqrt();
        a.store(z);
    }
}
pub fn cases() -> Vec<Case<f32>> {
    let mut v = Vec::with_capacity(128);
    v.push(Case {
        op: "exp",
        label: "std/baseline/x1",
        run: Box::new(fb0),
    });
    v.push(Case {
        op: "exp",
        label: "zenmath_correct/baseline/x1",
        run: Box::new(fb1),
    });
    v.push(Case {
        op: "exp",
        label: "libm/baseline/x1",
        run: Box::new(fb2),
    });
    v.push(Case {
        op: "exp",
        label: "fpmath/baseline/x1",
        run: Box::new(fb3),
    });
    v.push(Case {
        op: "exp",
        label: "micromath/baseline/x1",
        run: Box::new(fb4),
    });
    v.push(Case {
        op: "exp",
        label: "metallic/baseline/x1",
        run: Box::new(fb5),
    });
    v.push(Case {
        op: "exp",
        label: "zenmath_portable_midp/baseline/x1",
        run: Box::new(fb6),
    });
    v.push(Case {
        op: "exp",
        label: "zenmath_unfused_midp/baseline/x1",
        run: Box::new(fb7),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_midp/scalar/x8",
            run: Box::new(move |x, y, o| fb8_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_midp/v3/x8",
            run: Box::new(move |x, y, o| fb8_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_midp/v4x/x8",
            run: Box::new(move |x, y, o| fb8_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_midp/neon/x8",
            run: Box::new(move |x, y, o| fb8_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_midp/wasm128/x8",
            run: Box::new(move |x, y, o| fb8_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_midp/scalar/x8",
            run: Box::new(move |x, y, o| fb13_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_midp/v3/x8",
            run: Box::new(move |x, y, o| fb13_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_midp/v4x/x8",
            run: Box::new(move |x, y, o| fb13_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_midp/neon/x8",
            run: Box::new(move |x, y, o| fb13_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_midp/wasm128/x8",
            run: Box::new(move |x, y, o| fb13_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_midp/scalar/x8",
            run: Box::new(move |x, y, o| fb18_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_midp/v3/x8",
            run: Box::new(move |x, y, o| fb18_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_midp/v4x/x8",
            run: Box::new(move |x, y, o| fb18_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_midp/neon/x8",
            run: Box::new(move |x, y, o| fb18_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_midp/wasm128/x8",
            run: Box::new(move |x, y, o| fb18_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "exp",
        label: "zenmath_portable_lowp/baseline/x1",
        run: Box::new(fb23),
    });
    v.push(Case {
        op: "exp",
        label: "zenmath_unfused_lowp/baseline/x1",
        run: Box::new(fb24),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_lowp/scalar/x8",
            run: Box::new(move |x, y, o| fb25_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_lowp/v3/x8",
            run: Box::new(move |x, y, o| fb25_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_lowp/v4x/x8",
            run: Box::new(move |x, y, o| fb25_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_lowp/neon/x8",
            run: Box::new(move |x, y, o| fb25_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_portable_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| fb25_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_lowp/scalar/x8",
            run: Box::new(move |x, y, o| fb30_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_lowp/v3/x8",
            run: Box::new(move |x, y, o| fb30_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_lowp/v4x/x8",
            run: Box::new(move |x, y, o| fb30_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_lowp/neon/x8",
            run: Box::new(move |x, y, o| fb30_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_unfused_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| fb30_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_lowp/scalar/x8",
            run: Box::new(move |x, y, o| fb35_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_lowp/v3/x8",
            run: Box::new(move |x, y, o| fb35_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_lowp/v4x/x8",
            run: Box::new(move |x, y, o| fb35_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_lowp/neon/x8",
            run: Box::new(move |x, y, o| fb35_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_relaxed_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| fb35_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_correct_proof/scalar/x8",
            run: Box::new(move |x, y, o| fb40_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_correct_proof/v3/x8",
            run: Box::new(move |x, y, o| fb40_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_correct_proof/v4x/x8",
            run: Box::new(move |x, y, o| fb40_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_correct_proof/neon/x8",
            run: Box::new(move |x, y, o| fb40_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magemath_correct_proof/wasm128/x8",
            run: Box::new(move |x, y, o| fb40_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/scalar/x8",
            run: Box::new(move |x, y, o| fb45_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/v3/x8",
            run: Box::new(move |x, y, o| fb45_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/v4x/x8",
            run: Box::new(move |x, y, o| fb45_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/neon/x8",
            run: Box::new(move |x, y, o| fb45_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp/wasm128/x8",
            run: Box::new(move |x, y, o| fb45_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/scalar/x8",
            run: Box::new(move |x, y, o| fb50_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/v3/x8",
            run: Box::new(move |x, y, o| fb50_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/v4x/x8",
            run: Box::new(move |x, y, o| fb50_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/neon/x8",
            run: Box::new(move |x, y, o| fb50_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| fb50_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/scalar/x8",
            run: Box::new(move |x, y, o| fb55_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/v3/x8",
            run: Box::new(move |x, y, o| fb55_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/v4x/x8",
            run: Box::new(move |x, y, o| fb55_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/neon/x8",
            run: Box::new(move |x, y, o| fb55_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "exp",
            label: "magetypes_midp_precise/wasm128/x8",
            run: Box::new(move |x, y, o| fb55_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "ln",
        label: "std/baseline/x1",
        run: Box::new(fb60),
    });
    v.push(Case {
        op: "ln",
        label: "zenmath_correct/baseline/x1",
        run: Box::new(fb61),
    });
    v.push(Case {
        op: "ln",
        label: "libm/baseline/x1",
        run: Box::new(fb62),
    });
    v.push(Case {
        op: "ln",
        label: "fpmath/baseline/x1",
        run: Box::new(fb63),
    });
    v.push(Case {
        op: "ln",
        label: "micromath/baseline/x1",
        run: Box::new(fb64),
    });
    v.push(Case {
        op: "ln",
        label: "metallic/baseline/x1",
        run: Box::new(fb65),
    });
    v.push(Case {
        op: "ln",
        label: "zenmath_portable_midp/baseline/x1",
        run: Box::new(fb66),
    });
    v.push(Case {
        op: "ln",
        label: "zenmath_unfused_midp/baseline/x1",
        run: Box::new(fb67),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_midp/scalar/x8",
            run: Box::new(move |x, y, o| fb68_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_midp/v3/x8",
            run: Box::new(move |x, y, o| fb68_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_midp/v4x/x8",
            run: Box::new(move |x, y, o| fb68_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_midp/neon/x8",
            run: Box::new(move |x, y, o| fb68_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_midp/wasm128/x8",
            run: Box::new(move |x, y, o| fb68_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_midp/scalar/x8",
            run: Box::new(move |x, y, o| fb73_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_midp/v3/x8",
            run: Box::new(move |x, y, o| fb73_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_midp/v4x/x8",
            run: Box::new(move |x, y, o| fb73_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_midp/neon/x8",
            run: Box::new(move |x, y, o| fb73_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_midp/wasm128/x8",
            run: Box::new(move |x, y, o| fb73_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_midp/scalar/x8",
            run: Box::new(move |x, y, o| fb78_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_midp/v3/x8",
            run: Box::new(move |x, y, o| fb78_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_midp/v4x/x8",
            run: Box::new(move |x, y, o| fb78_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_midp/neon/x8",
            run: Box::new(move |x, y, o| fb78_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_midp/wasm128/x8",
            run: Box::new(move |x, y, o| fb78_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "ln",
        label: "zenmath_portable_lowp/baseline/x1",
        run: Box::new(fb83),
    });
    v.push(Case {
        op: "ln",
        label: "zenmath_unfused_lowp/baseline/x1",
        run: Box::new(fb84),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_lowp/scalar/x8",
            run: Box::new(move |x, y, o| fb85_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_lowp/v3/x8",
            run: Box::new(move |x, y, o| fb85_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_lowp/v4x/x8",
            run: Box::new(move |x, y, o| fb85_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_lowp/neon/x8",
            run: Box::new(move |x, y, o| fb85_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_portable_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| fb85_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_lowp/scalar/x8",
            run: Box::new(move |x, y, o| fb90_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_lowp/v3/x8",
            run: Box::new(move |x, y, o| fb90_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_lowp/v4x/x8",
            run: Box::new(move |x, y, o| fb90_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_lowp/neon/x8",
            run: Box::new(move |x, y, o| fb90_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_unfused_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| fb90_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_lowp/scalar/x8",
            run: Box::new(move |x, y, o| fb95_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_lowp/v3/x8",
            run: Box::new(move |x, y, o| fb95_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_lowp/v4x/x8",
            run: Box::new(move |x, y, o| fb95_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_lowp/neon/x8",
            run: Box::new(move |x, y, o| fb95_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_relaxed_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| fb95_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_correct_proof/scalar/x8",
            run: Box::new(move |x, y, o| fb100_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_correct_proof/v3/x8",
            run: Box::new(move |x, y, o| fb100_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_correct_proof/v4x/x8",
            run: Box::new(move |x, y, o| fb100_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_correct_proof/neon/x8",
            run: Box::new(move |x, y, o| fb100_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magemath_correct_proof/wasm128/x8",
            run: Box::new(move |x, y, o| fb100_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/scalar/x8",
            run: Box::new(move |x, y, o| fb105_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/v3/x8",
            run: Box::new(move |x, y, o| fb105_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/v4x/x8",
            run: Box::new(move |x, y, o| fb105_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/neon/x8",
            run: Box::new(move |x, y, o| fb105_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp/wasm128/x8",
            run: Box::new(move |x, y, o| fb105_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/scalar/x8",
            run: Box::new(move |x, y, o| fb110_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/v3/x8",
            run: Box::new(move |x, y, o| fb110_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/v4x/x8",
            run: Box::new(move |x, y, o| fb110_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/neon/x8",
            run: Box::new(move |x, y, o| fb110_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| fb110_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/scalar/x8",
            run: Box::new(move |x, y, o| fb115_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/v3/x8",
            run: Box::new(move |x, y, o| fb115_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/v4x/x8",
            run: Box::new(move |x, y, o| fb115_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/neon/x8",
            run: Box::new(move |x, y, o| fb115_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "ln",
            label: "magetypes_midp_precise/wasm128/x8",
            run: Box::new(move |x, y, o| fb115_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "log2",
        label: "std/baseline/x1",
        run: Box::new(fb120),
    });
    v.push(Case {
        op: "log2",
        label: "zenmath_correct/baseline/x1",
        run: Box::new(fb121),
    });
    v.push(Case {
        op: "log2",
        label: "libm/baseline/x1",
        run: Box::new(fb122),
    });
    v.push(Case {
        op: "log2",
        label: "fpmath/baseline/x1",
        run: Box::new(fb123),
    });
    v.push(Case {
        op: "log2",
        label: "micromath/baseline/x1",
        run: Box::new(fb124),
    });
    v.push(Case {
        op: "log2",
        label: "metallic/baseline/x1",
        run: Box::new(fb125),
    });
    v.push(Case {
        op: "log2",
        label: "zenmath_portable_midp/baseline/x1",
        run: Box::new(fb126),
    });
    v.push(Case {
        op: "log2",
        label: "zenmath_unfused_midp/baseline/x1",
        run: Box::new(fb127),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_midp/scalar/x8",
            run: Box::new(move |x, y, o| fb128_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_midp/v3/x8",
            run: Box::new(move |x, y, o| fb128_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_midp/v4x/x8",
            run: Box::new(move |x, y, o| fb128_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_midp/neon/x8",
            run: Box::new(move |x, y, o| fb128_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_midp/wasm128/x8",
            run: Box::new(move |x, y, o| fb128_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_midp/scalar/x8",
            run: Box::new(move |x, y, o| fb133_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_midp/v3/x8",
            run: Box::new(move |x, y, o| fb133_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_midp/v4x/x8",
            run: Box::new(move |x, y, o| fb133_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_midp/neon/x8",
            run: Box::new(move |x, y, o| fb133_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_midp/wasm128/x8",
            run: Box::new(move |x, y, o| fb133_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_midp/scalar/x8",
            run: Box::new(move |x, y, o| fb138_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_midp/v3/x8",
            run: Box::new(move |x, y, o| fb138_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_midp/v4x/x8",
            run: Box::new(move |x, y, o| fb138_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_midp/neon/x8",
            run: Box::new(move |x, y, o| fb138_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_midp/wasm128/x8",
            run: Box::new(move |x, y, o| fb138_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "log2",
        label: "zenmath_portable_lowp/baseline/x1",
        run: Box::new(fb143),
    });
    v.push(Case {
        op: "log2",
        label: "zenmath_unfused_lowp/baseline/x1",
        run: Box::new(fb144),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_lowp/scalar/x8",
            run: Box::new(move |x, y, o| fb145_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_lowp/v3/x8",
            run: Box::new(move |x, y, o| fb145_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_lowp/v4x/x8",
            run: Box::new(move |x, y, o| fb145_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_lowp/neon/x8",
            run: Box::new(move |x, y, o| fb145_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_portable_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| fb145_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_lowp/scalar/x8",
            run: Box::new(move |x, y, o| fb150_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_lowp/v3/x8",
            run: Box::new(move |x, y, o| fb150_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_lowp/v4x/x8",
            run: Box::new(move |x, y, o| fb150_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_lowp/neon/x8",
            run: Box::new(move |x, y, o| fb150_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_unfused_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| fb150_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_lowp/scalar/x8",
            run: Box::new(move |x, y, o| fb155_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_lowp/v3/x8",
            run: Box::new(move |x, y, o| fb155_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_lowp/v4x/x8",
            run: Box::new(move |x, y, o| fb155_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_lowp/neon/x8",
            run: Box::new(move |x, y, o| fb155_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_relaxed_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| fb155_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_correct_proof/scalar/x8",
            run: Box::new(move |x, y, o| fb160_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_correct_proof/v3/x8",
            run: Box::new(move |x, y, o| fb160_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_correct_proof/v4x/x8",
            run: Box::new(move |x, y, o| fb160_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_correct_proof/neon/x8",
            run: Box::new(move |x, y, o| fb160_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magemath_correct_proof/wasm128/x8",
            run: Box::new(move |x, y, o| fb160_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/scalar/x8",
            run: Box::new(move |x, y, o| fb165_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/v3/x8",
            run: Box::new(move |x, y, o| fb165_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/v4x/x8",
            run: Box::new(move |x, y, o| fb165_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/neon/x8",
            run: Box::new(move |x, y, o| fb165_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp/wasm128/x8",
            run: Box::new(move |x, y, o| fb165_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/scalar/x8",
            run: Box::new(move |x, y, o| fb170_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/v3/x8",
            run: Box::new(move |x, y, o| fb170_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/v4x/x8",
            run: Box::new(move |x, y, o| fb170_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/neon/x8",
            run: Box::new(move |x, y, o| fb170_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_lowp/wasm128/x8",
            run: Box::new(move |x, y, o| fb170_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/scalar/x8",
            run: Box::new(move |x, y, o| fb175_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/v3/x8",
            run: Box::new(move |x, y, o| fb175_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/v4x/x8",
            run: Box::new(move |x, y, o| fb175_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/neon/x8",
            run: Box::new(move |x, y, o| fb175_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "log2",
            label: "magetypes_midp_precise/wasm128/x8",
            run: Box::new(move |x, y, o| fb175_wasm128_t(t, x, y, o)),
        });
    }
    v.push(Case {
        op: "sqrt",
        label: "std/baseline/x1",
        run: Box::new(fb180),
    });
    v.push(Case {
        op: "sqrt",
        label: "zenmath_correct/baseline/x1",
        run: Box::new(fb181),
    });
    v.push(Case {
        op: "sqrt",
        label: "libm/baseline/x1",
        run: Box::new(fb182),
    });
    v.push(Case {
        op: "sqrt",
        label: "fpmath/baseline/x1",
        run: Box::new(fb183),
    });
    v.push(Case {
        op: "sqrt",
        label: "micromath/baseline/x1",
        run: Box::new(fb184),
    });
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_portable/scalar/x8",
            run: Box::new(move |x, y, o| fb185_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_portable/v3/x8",
            run: Box::new(move |x, y, o| fb185_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_portable/v4x/x8",
            run: Box::new(move |x, y, o| fb185_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_portable/neon/x8",
            run: Box::new(move |x, y, o| fb185_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_portable/wasm128/x8",
            run: Box::new(move |x, y, o| fb185_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_unfused/scalar/x8",
            run: Box::new(move |x, y, o| fb190_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_unfused/v3/x8",
            run: Box::new(move |x, y, o| fb190_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_unfused/v4x/x8",
            run: Box::new(move |x, y, o| fb190_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_unfused/neon/x8",
            run: Box::new(move |x, y, o| fb190_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_unfused/wasm128/x8",
            run: Box::new(move |x, y, o| fb190_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_relaxed/scalar/x8",
            run: Box::new(move |x, y, o| fb195_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_relaxed/v3/x8",
            run: Box::new(move |x, y, o| fb195_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_relaxed/v4x/x8",
            run: Box::new(move |x, y, o| fb195_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_relaxed/neon/x8",
            run: Box::new(move |x, y, o| fb195_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magemath_relaxed/wasm128/x8",
            run: Box::new(move |x, y, o| fb195_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/scalar/x8",
            run: Box::new(move |x, y, o| fb200_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/v3/x8",
            run: Box::new(move |x, y, o| fb200_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/v4x/x8",
            run: Box::new(move |x, y, o| fb200_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/neon/x8",
            run: Box::new(move |x, y, o| fb200_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "sqrt",
            label: "magetypes_default/wasm128/x8",
            run: Box::new(move |x, y, o| fb200_wasm128_t(t, x, y, o)),
        });
    }
    v
}

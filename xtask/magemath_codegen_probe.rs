// Standalone no_std consumer compiled by magemath_codegen.py.
#![no_std]

use archmage::ScalarToken;
use magemath::{
    cbrtf::{Cbrtf, CbrtfDomain, CbrtfOptions},
    expf::{Expf, ExpfDomain, ExpfOptions},
    log1pf::{Log1pf, Log1pfDomain, Log1pfOptions},
    logf::{Logf, LogfDomain, LogfOptions},
    powf::{Powf, PowfDomain, PowfOptions},
};
use magetypes::{
    numerics::RecipOptions,
    simd::generic::{f32x8, f32x16, u8x32},
};

#[cfg(target_arch = "aarch64")]
type NativeToken = archmage::NeonToken;
#[cfg(target_arch = "wasm32")]
type NativeToken = archmage::Wasm128Token;
#[cfg(target_arch = "x86_64")]
type NativeToken = archmage::X64V3Token;

// These probes allocate nothing. Any accidental allocation fails.
struct NoAlloc;

unsafe impl core::alloc::GlobalAlloc for NoAlloc {
    unsafe fn alloc(&self, _: core::alloc::Layout) -> *mut u8 {
        core::ptr::null_mut()
    }

    unsafe fn dealloc(&self, _: *mut u8, _: core::alloc::Layout) {}
}

#[global_allocator]
static ALLOCATOR: NoAlloc = NoAlloc;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

// The destination is sliced to the source length at the call sites by the
// consumer's contract. Keep both lengths opaque to model the public slice API.
macro_rules! slice_probe {
    (native $name:ident, |$token:ident, $src:ident, $dst:ident| $body:expr) => {
        #[cfg_attr(target_arch = "aarch64", archmage::attune(neon))]
        #[cfg_attr(target_arch = "wasm32", archmage::attune(wasm128))]
        #[cfg_attr(target_arch = "x86_64", archmage::attune(v3))]
        #[unsafe(no_mangle)]
        #[inline(never)]
        pub fn $name($src: &[f32], $dst: &mut [f32]) -> bool {
            let $token = NativeToken::from_context();
            $body.is_ok()
        }
    };
    (scalar $name:ident, |$token:ident, $src:ident, $dst:ident| $body:expr) => {
        #[unsafe(no_mangle)]
        #[inline(never)]
        pub fn $name($src: &[f32], $dst: &mut [f32]) -> bool {
            let $token = ScalarToken;
            $body.is_ok()
        }
    };
}

const UNIT: PowfOptions = PowfOptions::new()
    .domain(PowfDomain::Unit)
    .midp()
    .caller()
    .const_exponent(1.0 / 2.4);
const HDR: PowfOptions = PowfOptions::new()
    .gamma()
    .midp()
    .caller()
    .assume_in_bounds();

slice_probe!(native powf_unit_checked, |t, src, dst|
    Powf::<{ UNIT.bits() }>::slice_const_with_token(t, src, dst)
);
slice_probe!(native powf_unit_unchecked, |t, src, dst|
    Powf::<{ UNIT.assume_in_bounds().bits() }>::slice_const_with_token(t, src, dst)
);
slice_probe!(native powf_midp_bit_exact, |t, src, dst|
    Powf::<{ HDR.bit_exact().bits() }>::slice_with_token(t, src, dst, 2.2)
);
slice_probe!(native powf_midp_backend, |t, src, dst|
    Powf::<{ HDR.backend_dependent().bits() }>::slice_with_token(t, src, dst, 2.2)
);

#[unsafe(no_mangle)]
#[inline(never)]
pub fn powf_slice_scalar(src: &[f32], dst: &mut [f32], exponent: f32) -> bool {
    Powf::<{ PowfOptions::new().caller().bits() }>::slice_with_token(
        ScalarToken,
        src,
        dst,
        exponent,
    )
    .is_ok()
}

const HIGHP_GAMMA: PowfOptions = PowfOptions::new()
    .gamma()
    .assume_in_bounds()
    .caller()
    .highp();

macro_rules! gamma_probe {
    ($name:ident, $exponent:expr) => {
        // One input/output length models a caller with paired buffers. Keep
        // the repeated slice-length assertion out of the sharing measurement.
        #[unsafe(no_mangle)]
        #[inline(never)]
        pub unsafe extern "C" fn $name(src: *const f32, dst: *mut f32, len: usize) -> bool {
            // SAFETY: The caller supplies non-overlapping live buffers of len elements.
            let src = unsafe { core::slice::from_raw_parts(src, len) };
            let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
            Powf::<{ HIGHP_GAMMA.const_exponent($exponent).bits() }>::slice_const_with_token(
                ScalarToken,
                src,
                dst,
            )
            .is_ok()
        }
    };
}

gamma_probe!(powf_highp_gamma_0, 2.4);
#[cfg(feature = "gamma-four")]
gamma_probe!(powf_highp_gamma_1, 1.0 / 2.4);
#[cfg(feature = "gamma-four")]
gamma_probe!(powf_highp_gamma_2, 2.2);
#[cfg(feature = "gamma-four")]
gamma_probe!(powf_highp_gamma_3, 1.0 / 2.2);

// FullDomain with an opaque exponent is the packed-loop throughput witness.
#[cfg(target_arch = "x86_64")]
const HIGHP_FULL: u128 = PowfOptions::new().full_domain().highp().caller().bits();

#[cfg(target_arch = "x86_64")]
#[archmage::attune(v3)]
#[unsafe(no_mangle)]
#[inline(never)]
pub fn powf_highp_v3(src: &[f32], dst: &mut [f32], exponent: f32) -> bool {
    Powf::<HIGHP_FULL>::slice_with_token(archmage::X64V3Token::from_context(), src, dst, exponent)
        .is_ok()
}

#[cfg(target_arch = "x86_64")]
#[archmage::attune(v4x)]
#[unsafe(no_mangle)]
#[inline(never)]
pub fn powf_highp_v4x(src: &[f32], dst: &mut [f32], exponent: f32) -> bool {
    Powf::<HIGHP_FULL>::slice_with_token(archmage::X64V4xToken::from_context(), src, dst, exponent)
        .is_ok()
}

macro_rules! unary_probe {
    ($tier:ident $name:ident, $family:ident, $options:expr, $method:ident) => {
        slice_probe!($tier $name, |t, src, dst|
            $family::<{ $options.bits() }>::$method(t, src, dst)
        );
    };
}

const EXP: ExpfOptions = ExpfOptions::new()
    .domain(ExpfDomain::Finite)
    .caller()
    .assume_in_bounds();
unary_probe!(native exp2_midp_bit_exact, Expf, EXP.midp().bit_exact(), exp2_slice_with_token);
unary_probe!(native exp2_midp_backend, Expf, EXP.midp().backend_dependent(), exp2_slice_with_token);
unary_probe!(native exp2_lowp_bit_exact, Expf, EXP.lowp().bit_exact(), exp2_slice_with_token);
unary_probe!(native exp2_lowp_backend, Expf, EXP.lowp().backend_dependent(), exp2_slice_with_token);

const LOG: LogfOptions = LogfOptions::new()
    .domain(LogfDomain::PositiveNormals)
    .midp()
    .caller()
    .assume_in_bounds();
unary_probe!(native log2_midp_bit_exact, Logf, LOG.bit_exact(), log2_slice_with_token);
unary_probe!(native log2_midp_backend, Logf, LOG.backend_dependent(), log2_slice_with_token);

const LOG1P: Log1pfOptions = Log1pfOptions::new().midp().caller().assume_in_bounds();
const LOG1P_AROUND: Log1pfOptions = LOG1P.domain(Log1pfDomain::AroundZero);
const LOG1P_GREATER: Log1pfOptions = LOG1P.domain(Log1pfDomain::GreaterThanNegOne);
unary_probe!(native log1p_around, Log1pf, LOG1P_AROUND, slice_with_token);
unary_probe!(native log1p_greater, Log1pf, LOG1P_GREATER, slice_with_token);
unary_probe!(scalar log1p_around_scalar, Log1pf, LOG1P_AROUND, slice_with_token);
unary_probe!(scalar log1p_greater_scalar, Log1pf, LOG1P_GREATER, slice_with_token);

const CBRT: CbrtfOptions = CbrtfOptions::new()
    .domain(CbrtfDomain::Bounded)
    .caller()
    .assume_in_bounds();
unary_probe!(native cbrtf_midp_bit_exact, Cbrtf, CBRT.midp().bit_exact(), slice_with_token);
unary_probe!(native cbrtf_midp_backend, Cbrtf, CBRT.midp().backend_dependent(), slice_with_token);
unary_probe!(native cbrtf_lowp_bit_exact, Cbrtf, CBRT.lowp().bit_exact(), slice_with_token);
unary_probe!(native cbrtf_lowp_backend, Cbrtf, CBRT.lowp().backend_dependent(), slice_with_token);
unary_probe!(scalar cbrtf_midp_bit_exact_scalar, Cbrtf, CBRT.midp().bit_exact(), slice_with_token);
unary_probe!(scalar cbrtf_midp_backend_scalar, Cbrtf, CBRT.midp().backend_dependent(), slice_with_token);
unary_probe!(scalar cbrtf_lowp_bit_exact_scalar, Cbrtf, CBRT.lowp().bit_exact(), slice_with_token);
unary_probe!(scalar cbrtf_lowp_backend_scalar, Cbrtf, CBRT.lowp().backend_dependent(), slice_with_token);

#[unsafe(no_mangle)]
#[inline(never)]
pub fn scalar_transpose(rows: &mut [f32x16<ScalarToken>; 16]) {
    f32x16::transpose_16x16(rows);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn scalar_deinterleave(src: &[f32x16<ScalarToken>; 3], dst: &mut [f32x16<ScalarToken>; 3]) {
    *dst = f32x16::deinterleave_3ch(*src);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn scalar_rgb(src: &[u8; 24], dst: &mut [u8; 24], factor: f32) {
    let (r, g, b) = f32x8::load_rgb_u8_t(ScalarToken, src);
    let factor = f32x8::splat_t(ScalarToken, factor);
    f32x8::store_rgb_u8(r * factor, g * factor, b * factor, dst);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn scalar_rsqrt(src: &[f32; 8], dst: &mut [f32; 8]) {
    f32x8::load_t(ScalarToken, src)
        .rsqrt_with::<{ RecipOptions::new().bits() }>()
        .store(dst);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn scalar_round(src: &[f32; 8], dst: &mut [f32; 8]) {
    f32x8::load_t(ScalarToken, src).round().store(dst);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn scalar_bitmask(src: &[u8; 32]) -> u32 {
    u8x32::load_t(ScalarToken, src).bitmask()
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn scalar_sqrt_f32(x: f32) -> f32 {
    magetypes::scalar::sqrt_f32(x)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn scalar_sqrt_f64(x: f64) -> f64 {
    magetypes::scalar::sqrt_f64(x)
}

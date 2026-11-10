//! Fixed 64-value/step kernels. No corpus materialization; timing is opt-in.
#![forbid(unsafe_code)]
use archmage::{ScalarToken, SimdToken};
use magemath::{minmax::*, recip::*};
use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::{f32x8 as V, i32x8 as I};
use zenbench::{GateConfig, Suite, black_box};
const N: usize = 64;
const NUMBER: u32 = MinMaxOptions::PORTABLE_NUMBER.bits();
const PROP: u32 = MinMaxOptions::PORTABLE_PROPAGATE.bits();
const RELAXED: u32 = MinMaxOptions::RELAXED.bits();
const NORMAL: u32 = MinMaxOptions::NORMAL.bits();
const UNORMAL: u32 = MinMaxOptions::UNCHECKED_NORMAL.bits();
const HIGH: u32 = RecipOptions::HIGHP.bits();
const MID: u32 = RecipOptions::PORTABLE_MIDP
    .with_validation(RecipValidation::AssumeValid)
    .bits();
const LOW: u32 = RecipOptions::PORTABLE_LOWP
    .with_validation(RecipValidation::AssumeValid)
    .bits();
const NMID: u32 = RecipOptions::BACKEND_MIDP
    .with_validation(RecipValidation::AssumeValid)
    .bits();
const NLOW: u32 = RecipOptions::BACKEND_LOWP
    .with_validation(RecipValidation::AssumeValid)
    .bits();
const CHECKED: u32 = RecipOptions::BACKEND_MIDP.bits();
const LOG: u32 = magemath::logf::LogfOptions::PORTABLE_LOWP
    .with_portability(magemath::logf::LogfPortability::BackendDependent)
    .with_validation(magemath::logf::LogfValidation::AssumeValid)
    .with_execution(magemath::logf::LogfExecution::Caller)
    .bits();

// Mirror of linear-srgb 8900ffc tf/fast_math.rs, with its exact f32 constants.
// This is a graph mirror on current Magetypes, not a timing of that crate binary.
#[inline(always)]
fn linear_log2<T: F32x8Convert>(t: T, x: V<T>) -> V<T> {
    let offset = I::splat_t(t, 0x3f2aaaab);
    let shifted = x.bitcast_to_i32() - offset;
    let e = shifted.shr_arithmetic_const::<23>();
    let m = (x.bitcast_to_i32() - e.shl_const::<23>()).bitcast_to_f32() - V::splat_t(t, 1.0);
    let p = V::splat_t(t, f32::from_bits(0x3f3e11c7)).mul_add_relaxed(m, V::splat_t(t, 1.4287161));
    let p = p.mul_add_relaxed(m, V::splat_t(t, -1.8503833e-6));
    let q = V::splat_t(t, 0.17409343).mul_add_relaxed(m, V::splat_t(t, 1.0096718));
    let q = q.mul_add_relaxed(m, V::splat_t(t, 0.99032814));
    p / q + e.to_f32()
}
#[inline(always)]
fn operation<T: F32x8Convert, const OP: u8, const MODE: u8>(t: T, x: V<T>) -> V<T> {
    if OP == 0 {
        let y = V::splat_t(t, 2.0);
        return match MODE {
            0 => x.min(y),
            1 => try_min_f32x8_with::<NUMBER, _>(t, x, y).unwrap(),
            2 => try_min_f32x8_with::<PROP, _>(t, x, y).unwrap(),
            3 => try_min_f32x8_with::<RELAXED, _>(t, x, y).unwrap(),
            4 => try_min_f32x8_with::<NORMAL, _>(t, x, y).unwrap(),
            5 => try_min_f32x8_with::<UNORMAL, _>(t, x, y).unwrap(),
            _ => unreachable!(),
        };
    }
    if OP == 3 {
        return if MODE == 0 {
            linear_log2(t, x)
        } else {
            magemath::logf::try_log2_f32x8_with::<LOG, _>(t, x).unwrap()
        };
    }
    macro_rules! run {
        ($rc:ident,$rs:ident,$flags:ident) => {
            if OP == 1 {
                $rc::<$flags, _>(t, x).unwrap()
            } else {
                $rs::<$flags, _>(t, x).unwrap()
            }
        };
    }
    match MODE {
        0 => {
            if OP == 1 {
                x.recip()
            } else {
                x.rsqrt()
            }
        }
        1 => run!(try_recip_f32x8_with, try_rsqrt_f32x8_with, HIGH),
        2 => run!(try_recip_f32x8_with, try_rsqrt_f32x8_with, MID),
        3 => run!(try_recip_f32x8_with, try_rsqrt_f32x8_with, LOW),
        4 => run!(try_recip_f32x8_with, try_rsqrt_f32x8_with, NMID),
        5 => run!(try_recip_f32x8_with, try_rsqrt_f32x8_with, NLOW),
        6 => run!(try_recip_f32x8_with, try_rsqrt_f32x8_with, CHECKED),
        _ => unreachable!(),
    }
}
#[archmage::magetypes(v3, neon, wasm128, scalar)]
fn stream<const OP: u8, const MODE: u8>(t: Token, input: &[f32; N], out: &mut [f32; N]) {
    for (x, y) in input
        .as_chunks::<8>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<8>().0.iter_mut())
    {
        *y = operation::<_, OP, MODE>(t, V::from_array_t(t, *x)).to_array();
    }
}
#[archmage::magetypes(v3, neon, wasm128, scalar)]
fn chain<const OP: u8, const MODE: u8>(t: Token, initial: f32) -> [f32; 8] {
    let mut x = V::splat_t(t, initial);
    for _ in 0..N {
        x = operation::<_, OP, MODE>(t, x) + V::splat_t(t, if OP == 0 { 0.03125 } else { 0.5 });
    }
    x.to_array()
}
macro_rules! group{($suite:ident,$t:ident,$stream:ident,$chain:ident,$name:literal,$op:literal,$chain_mode:literal,$($label:literal=>$mode:literal),+)=>{
 $suite.group($name,|g|{
  g.config().min_rounds(50).max_rounds(50).auto_rounds(false);
  let input:[f32;N]=core::array::from_fn(|i|0.25+(i+1)as f32*0.03125);
  $(g.bench($label,move|b|{let mut out=[0.0;N];b.iter(||{if $chain_mode {black_box( $chain::<$op,$mode>($t,black_box(0.75)));}else{$stream::<$op,$mode>($t,black_box(&input),black_box(&mut out));black_box(&out);}});});)+
 });};}
macro_rules! register{
 ($suite:ident,$t:expr,$stream:ident,$chain:ident)=>{{
 let t=$t;

 group!($suite,t,$stream,$chain,"min/f32x8/stream64",0,false,"legacy native min"=>0,"portable number ordered"=>1,"portable propagate ordered"=>2,"backend NaN and zero"=>3,"normal checked"=>4,"normal assume valid"=>5);
 group!($suite,t,$stream,$chain,"min/f32x8/chain64",0,true,"legacy native min"=>0,"portable number ordered"=>1,"portable propagate ordered"=>2,"backend NaN and zero"=>3,"normal checked"=>4,"normal assume valid"=>5);
 group!($suite,t,$stream,$chain,"recip/f32x8/stream64",1,false,"legacy working recip"=>0,"highp division"=>1,"portable midp unchecked"=>2,"portable lowp unchecked"=>3,"native midp unchecked"=>4,"native lowp unchecked"=>5,"native midp checked"=>6);
 group!($suite,t,$stream,$chain,"recip/f32x8/chain64",1,true,"legacy working recip"=>0,"highp division"=>1,"portable midp unchecked"=>2,"portable lowp unchecked"=>3,"native midp unchecked"=>4,"native lowp unchecked"=>5,"native midp checked"=>6);
 group!($suite,t,$stream,$chain,"rsqrt/f32x8/stream64",2,false,"legacy working rsqrt"=>0,"highp sqrt then divide"=>1,"portable midp unchecked"=>2,"portable lowp unchecked"=>3,"native midp unchecked"=>4,"native lowp unchecked"=>5,"native midp checked"=>6);
 group!($suite,t,$stream,$chain,"rsqrt/f32x8/chain64",2,true,"legacy working rsqrt"=>0,"highp sqrt then divide"=>1,"portable midp unchecked"=>2,"portable lowp unchecked"=>3,"native midp unchecked"=>4,"native lowp unchecked"=>5,"native midp checked"=>6);
 group!($suite,t,$stream,$chain,"log2/f32x8/stream64",3,false,"linear-srgb graph mirror"=>0,"configured lowp unchecked"=>1);
 }};
}
fn register(suite: &mut Suite) {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = archmage::X64V3Token::summon() {
        register!(suite, t, stream_v3, chain_v3);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        register!(suite, t, stream_neon, chain_neon);
        return;
    }
    register!(suite, ScalarToken, stream_scalar, chain_scalar);
}
fn main() {
    if std::env::args().any(|a| a == "--cpu-set") {
        let topology = zenbench::platform::detect_cpu_topology();
        let mut classes = topology.classes;
        classes.sort_by(|a, b| a.name.cmp(&b.name));
        let selected = classes.first().expect("known CPU class");
        println!(
            "{}",
            selected
                .cpus
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        return;
    }
    if !std::env::args().any(|a| a == "--measure") {
        let mut out = [0.; N];
        stream_scalar::<1, 1>(ScalarToken, &[1.; N], &mut out);
        assert_eq!(out, [1.; N]);
        println!("primitive benchmark kernels compile; timing requires --measure");
        return;
    }
    let r = zenbench::run_gated(GateConfig::strict(), register);
    let path = std::env::var("ARCHMAGE_PRIMITIVE_RESULTS").expect("durable result path");
    r.save(path).unwrap();
    r.print_report();
    assert!(!r.unreliable, "retained timings failed admission");
    for c in &r.comparisons {
        assert_eq!(c.samples.len(), 50);
        assert!(c.samples.iter().all(|s| s.gate_clean == Some(true)));
    }
}

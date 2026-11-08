use std::sync::{Mutex, OnceLock};
pub struct State {
    pub input4:Vec<[f32;4]>, pub output4:Vec<[f32;4]>,
    pub input8:Vec<[f32;8]>, pub output8:Vec<[f32;8]>,
    pub input16:Vec<[f32;16]>, pub output16:Vec<[f32;16]>,
}
static STATE:OnceLock<Mutex<State>>=OnceLock::new();
// --export-all exposes the compiler's unique symbols. The host discovers the
// names; no unsafe export attribute or hand-chosen global symbol is required.
pub extern "C" fn guest_prepare(size:i32,distribution:i32,op:i32) {
    let dist=["ordinary","bits","sparse","dense"][distribution as usize];
    let n=size as usize;
    let mut state=STATE.get_or_init(||Mutex::new(State{input4:Vec::new(),output4:Vec::new(),input8:Vec::new(),output8:Vec::new(),input16:Vec::new(),output16:Vec::new()})).lock().unwrap();
    state.input4=(0..n/4).map(|i|core::array::from_fn(|j|super::value(i*4+j,dist,op as usize))).collect();
    state.input8=(0..n/8).map(|i|core::array::from_fn(|j|super::value(i*8+j,dist,op as usize))).collect();
    state.input16=(0..n/16).map(|i|core::array::from_fn(|j|super::value(i*16+j,dist,op as usize))).collect();
    state.output4=vec![[0.;4];n/4];state.output8=vec![[0.;8];n/8];state.output16=vec![[0.;16];n/16];
}
pub extern "C" fn guest_run(op:i32,arm:i32,width:i32,chain:i32,repeats:i32)->i64 {
    let mut state=STATE.get().expect("prepare first").lock().unwrap();
    for _ in 0..repeats {super::kernels::guest_kernel(op,arm,width,&mut state,chain!=0);}
    (match width {4=>state.output4[0][0].to_bits(),8=>state.output8[0][0].to_bits(),16=>state.output16[0][0].to_bits(),_=>panic!("width") }) as i64
}

pub extern "C" fn guest_probe()->i32 {
    use archmage::SimdToken;
    let t=archmage::Wasm128Token::summon().expect("SIMD module");
    let x=magetypes::simd::generic::f32x4::splat_t(t,std::hint::black_box(1.0+1.0/4096.0));
    let c=magetypes::simd::generic::f32x4::splat_t(t,std::hint::black_box(-(1.0+1.0/2048.0)));
    let r32=x.mul_add_relaxed(x,c).to_array()[0].to_bits();
    let x=magetypes::simd::generic::f64x2::splat_t(t,std::hint::black_box(1.0+1.0/134217728.0));
    let c=magetypes::simd::generic::f64x2::splat_t(t,std::hint::black_box(-(1.0+1.0/67108864.0)));
    let r64=x.mul_add_relaxed(x,c).to_array()[0].to_bits();
    assert!(r32==0 || r32==0x33800000);
    assert!(r64==0 || r64==969u64<<52);
    i32::from(r32!=0) | (i32::from(r64!=0)<<1)
}

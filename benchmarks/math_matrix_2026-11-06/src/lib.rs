#![forbid(unsafe_code)]
pub type Kernel<T> = Box<dyn Fn(&[T], &[T], &mut [T])>;
pub struct Case<T> { pub op: &'static str, pub label: &'static str, pub run: Kernel<T> }
mod generated;
pub use generated::{cases32,cases64};
pub fn inputs(op:&str, band:&str,n:usize)->(Vec<f32>,Vec<f32>){
    let special=[0.,-0.,f32::INFINITY,f32::NEG_INFINITY,f32::NAN,f32::from_bits(1),f32::MIN_POSITIVE,-1.];
    let edge=[f32::from_bits(1),f32::MIN_POSITIVE,f32::MAX,1.,f32::from_bits(0x3f7fffff),f32::from_bits(0x3f800001),-0.5,0.5,-1.5,1.5,-0.,0.,-100.,88.,-126.,127.];
    let x=(0..n).map(|i|{let t=((i*719)%1024) as f32/1024.;
        match band {"special"=>special[i%special.len()],"edge"=>edge[i%edge.len()],_=>match op {"exp"=>-8.+16.*t,"floor"|"ceil"|"round"|"roundeven"=>-64.+128.*t,_=>0.01+4.*t}}}).collect();
    let y=(0..n).map(|i| if op=="pow_fixed"{2.4}else if band=="special"{[0.,1.,2.,-3.,0.5,f32::NAN,f32::INFINITY,-0.][i%8]}else{-4.+8.*((i*313)%1024)as f32/1024.}).collect();(x,y)
}
pub fn inputs64(band:&str,n:usize)->(Vec<f64>,Vec<f64>){
    let e=[f64::from_bits(1),f64::MIN_POSITIVE,f64::MAX,1.,f64::from_bits(1f64.to_bits()-1),f64::from_bits(1f64.to_bits()+1),0.,-0.];
    let s=[f64::NAN,f64::INFINITY,f64::NEG_INFINITY,-1.,0.,-0.,f64::from_bits(1),1.];
    ((0..n).map(|i| match band {"edge"=>e[i%8],"special"=>s[i%8],_=>0.01+4.*((i*719)%1024)as f64/1024.}).collect(),vec![1.;n])
}

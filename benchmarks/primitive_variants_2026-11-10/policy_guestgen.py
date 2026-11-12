#!/usr/bin/env python3
"""Generate timer-free WASI dispatch for the same canonical policy kernels."""
import argparse
from pathlib import Path
import subprocess
from policy_tables import read

GUEST = r'''//! Timer-free guest; symbol discovery/export is handled by build_guest.py.
#![forbid(unsafe_code)]
#[cfg(target_arch="wasm32")]
#[path="../policy_workload.rs"]
mod workload;
#[cfg(target_arch="wasm32")]
#[path="../policy_check.rs"]
mod check;
#[cfg(target_arch="wasm32")]
mod guest {
use crate::workload::*;
use archmage::SimdToken;
use std::sync::{Mutex,OnceLock};
struct State {
    a32:[f32;4096],b32:[f32;4096],o32:[f32;4096],
    a64:[f64;4096],b64:[f64;4096],o64:[f64;4096],
}
static STATE:OnceLock<Mutex<State>>=OnceLock::new();
pub extern "C" fn guest_prepare(op:i32,kind:i32) {
    assert!((0..6).contains(&op) && (0..4).contains(&kind));
    assert!(kind!=3 || op<4,"zero clamps are min/max only");
    let mut s=STATE.get_or_init(||Mutex::new(State {
        a32:[0.;4096],b32:[0.;4096],o32:[0.;4096],a64:[0.;4096],b64:[0.;4096],o64:[0.;4096]
    })).lock().unwrap();
    let n=if kind==1 {4096}else{64};
    for i in 0..n {
        if op==2 || op==3 {let(a,b)=input64(i,kind==3);s.a64[i]=a;s.b64[i]=b;}
        else {let(a,b)=input(i,kind==3,op>=4,op==5);s.a32[i]=a;s.b32[i]=b;}
    }
}
pub extern "C" fn guest_probe()->i32 {
    use magetypes::simd::generic::f32x4 as V;
    let t=archmage::Wasm128Token::summon().unwrap();
    let a=V::splat_t(t,std::hint::black_box(1.0+1.0/4096.0));
    let c=V::splat_t(t,std::hint::black_box(-(1.0+1.0/2048.0)));
    i32::from(a.mul_add_relaxed(a,c).to_array()[0]==1.0/16777216.0)
}
macro_rules! functions {
    ($exec:ident,$verify:ident,$select:ident,$dispatch:ident,$stream:ident,$chain:ident,$f:ty,$w:literal,$a:ident,$b:ident,$out:ident,$rc:literal) => {
        fn $exec<const O:u32,const OP:bool,const L:bool,const N:usize,const CHAIN:bool>(s:&mut State,reps:i32) {
            let t=archmage::Wasm128Token::summon().unwrap();
            let State {$a:a,$b:b,$out:out,..}=s;
            let a:&[$f;N]=a[..N].try_into().unwrap();
            let b:&[$f;N]=b[..N].try_into().unwrap();
            let out:&mut[$f;N]=(&mut out[..N]).try_into().unwrap();
            for _ in 0..reps {
                if CHAIN {let y=$chain::<O,OP,L>(t,std::hint::black_box(0.75));out[..$w].copy_from_slice(&std::hint::black_box(y));}
                else {$stream::<O,OP,L,N>(t,std::hint::black_box(a),std::hint::black_box(b),std::hint::black_box(&mut *out));std::hint::black_box(&*out);}
            }
        }
        fn $verify<const O:u32,const OP:bool,const L:bool,const N:usize,const CHAIN:bool>(s:&State)->f64 {
            let t=archmage::Wasm128Token::summon().unwrap();
            let a:[$f;N]=s.$a[..N].try_into().unwrap();
            let b:[$f;N]=s.$b[..N].try_into().unwrap();
            crate::check::verify::<$f,N,$w>(O,L,$rc,OP,CHAIN,&(a,b),
                |a,b,out|$stream::<O,OP,L,N>(t,a,b,out),
                |a,b,out|$stream::<O,OP,L,$w>(t,a,b,out),
                |x|$chain::<O,OP,L>(t,x))
        }
        fn $select<const O:u32,const L:bool>(s:&mut State,op:bool,kind:i32,reps:i32,verify:bool)->f64 {
            macro_rules! call {($op:literal,$n:literal,$dep:literal)=>{{
                if verify {$verify::<O,$op,L,$n,$dep>(s)}
                else {$exec::<O,$op,L,$n,$dep>(s,reps);0.0}
            }};}
            match(op,kind) {
                (false,0|3)=>call!(false,64,false),(true,0|3)=>call!(true,64,false),
                (false,1)=>call!(false,4096,false),(true,1)=>call!(true,4096,false),
                (false,2)=>call!(false,64,true),(true,2)=>call!(true,64,true),
                _=>panic!("workload"),
            }
        }
        // Dispatcher emitted below, separately per width to bound type checking.
    };
}
// FUNCTIONS
// DISPATCH
fn invoke(width:i32,flags:i32,op:i32,kind:i32,reps:i32,verify:bool)->f64 {
    assert!((0..6).contains(&op) && (0..4).contains(&kind) && reps>0);
    assert!(kind!=3 || (op<4 && (flags<0 || flags&3!=2)),"unsupported zero-clamp input domain");
    let mut s=STATE.get().expect("prepare first").lock().unwrap();
    match(op,width) {
        // ROUTES
        _=>panic!("operation/width"),
    }
}
pub extern "C" fn guest_run(width:i32,flags:i32,op:i32,kind:i32,reps:i32)->i64 {
    invoke(width,flags,op,kind,reps,false);
    let s=STATE.get().unwrap().lock().unwrap();
    if op==2||op==3 {s.o64[0].to_bits() as i64}else{i64::from(s.o32[0].to_bits())}
}
pub extern "C" fn guest_verify(width:i32,flags:i32,op:i32,kind:i32)->f64 {
    invoke(width,flags,op,kind,1,true)
}
pub fn entry() {
    std::hint::black_box((guest_prepare as extern "C" fn(i32,i32),
        guest_probe as extern "C" fn()->i32,
        guest_run as extern "C" fn(i32,i32,i32,i32,i32)->i64,
        guest_verify as extern "C" fn(i32,i32,i32,i32)->f64));
}
}
fn main() {
    #[cfg(target_arch="wasm32")]
    guest::entry();
    #[cfg(not(target_arch="wasm32"))]
    panic!("compile the WASI guest target");
}
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('inventory',type=Path);p.add_argument('--check',action='store_true');args=p.parse_args()
    rows=read(args.inventory/'std.tsv');functions=[];dispatch=[];routes=[]
    for family,ty,widths in [('mm','f32',[4,8,16]),('mm','f64',[2,4,8]),('rc','f32',[4,8,16])]:
        flags=sorted({r['bits'] for r in rows if r['family']==('minmax' if family=='mm' else 'recip') and r['supported']})
        for w in widths:
            tag=str(w) if ty=='f32' else f'64x{w}'
            name=family+tag
            fields='a32,b32,o32' if ty=='f32' else 'a64,b64,o64'
            functions.append(f'functions!(exec_{name},verify_{name},select_{name},dispatch_{name},stream_{name}_wasm128,chain_{name}_wasm128,{ty},{w},{fields},{str(family=="rc").lower()});')
            arms=[f'{o}=>select_{name}::<{o},false>(s,op,kind,reps,verify),' for o in flags]
            dispatch.append(f'fn dispatch_{name}(s:&mut State,flags:i32,op:bool,kind:i32,reps:i32,verify:bool)->f64 {{match flags {{-1=>select_{name}::<0,true>(s,op,kind,reps,verify),'+''.join(arms)+'_=>panic!("unsupported flags"),}}')
            op0=4 if family=='rc' else 2 if ty=='f64' else 0
            routes += [f'({op0},{w})=>dispatch_{name}(&mut s,flags,false,kind,reps,verify),',f'({op0+1},{w})=>dispatch_{name}(&mut s,flags,true,kind,reps,verify),']
    text=GUEST.replace('// FUNCTIONS','\n'.join(functions)).replace('// DISPATCH','\n'.join(dispatch)).replace('// ROUTES','\n'.join(routes))
    text=subprocess.check_output(['rustfmt','--edition','2024','--emit','stdout'],input=text,text=True)
    path=Path(__file__).resolve().parent/'src/bin/policy_guest.rs'
    if args.check:assert path.read_text()==text,path
    else:path.write_text(text)
    print(path,len(text.encode()),'bytes')


if __name__=='__main__':main()

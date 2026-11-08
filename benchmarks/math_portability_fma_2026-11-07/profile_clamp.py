"""Inspect and count instructions for scalar clamp candidates; no speed gate."""
import json
import os
from pathlib import Path
import subprocess
import sys
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
source='''#![forbid(unsafe_code)]
#[inline(never)]
fn evaluate(x:[f32;4])->[f32;4] {magetypes::simd::generic::f32x4::from_array_t(archmage::ScalarToken,x).exp_midp().to_array()}
fn main(){let count:usize=std::env::args().nth(1).unwrap().parse().unwrap();let mut checksum=0u32;for i in 0..count {let v=std::hint::black_box([0.5+(i%128) as f32/256.0;4]);checksum^=std::hint::black_box(evaluate(v))[0].to_bits();}println!("{checksum:08x}");}
'''
for label,root in [('before',Path(sys.argv[2]).resolve()),('after',Path(sys.argv[3]).resolve())]:
    project=out/label;(project/'src').mkdir(parents=True)
    (project/'Cargo.toml').write_text(f'''[package]
name="clamp-profile"
version="0.0.0"
edition="2024"
[workspace]
[dependencies]
archmage={{path="{root}"}}
magetypes={{path="{root}/magetypes"}}
[profile.release]
codegen-units=1
lto="thin"
''')
    (project/'src/main.rs').write_text(source)
    with (project/'build.log').open('w') as log:
        subprocess.run(['cargo','build','--release','--manifest-path',str(project/'Cargo.toml')],stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=project/'target/release/clamp-profile'
    with (project/'disassembly.txt').open('w') as f:subprocess.run(['objdump','-Cd',str(binary)],stdout=f,check=True)
    with (project/'result.txt').open('w') as f:
        subprocess.run(['perf','stat','-x,','-e','instructions:u,cycles:u,branches:u','-o',str(project/'perf.csv'),str(binary),'1000000'],stdout=f,check=True)

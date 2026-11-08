"""Three alternating cold linked-consumer builds; every target is fresh.

Build timings are diagnostic. The host may be busy; no runtime promotion gate
or statistical speed claim is inferred from these six builds.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
revisions=[('baseline','6bfd83d744e41259ec758054631dcb0928fc4ea1'),('candidate',sys.argv[2])]
for label,rev in revisions:
    root=out/label;source=root/'source';source.mkdir(parents=True)
    with (root/'source.tar').open('wb') as f:subprocess.run(['git','archive',rev],stdout=f,check=True)
    subprocess.run(['tar','-xmf',str(root/'source.tar'),'-C',str(source)],check=True)
    consumer=root/'consumer';(consumer/'src').mkdir(parents=True)
    (consumer/'Cargo.toml').write_text(f'''[package]
name="portability-build-consumer"
version="0.0.0"
edition="2024"
[workspace]
[dependencies]
archmage={{path="{source}"}}
magetypes={{path="{source}/magetypes"}}
[profile.release]
codegen-units=1
lto="thin"
''')
    (consumer/'src/main.rs').write_text('''#![forbid(unsafe_code)]
use archmage::prelude::*;
#[magetypes(v3,neon,scalar)]
fn work(t:Token,x:[f32;8])->[f32;8] {
 let v=magetypes::simd::generic::f32x8::from_array_t(t,x);
 (v.exp2_lowp()+v.exp2_midp()+v.exp_lowp()+v.exp_midp()+v.pow_lowp(2.4)+v.pow_midp(2.4)+v.sigmoid_midp()+v.silu_midp()+v.log2_lowp()+v.log2_midp()+v.ln_lowp()+v.ln_midp()+v.log10_lowp()+v.log10_midp()).to_array()
}
fn main(){let x=std::env::args().nth(1).unwrap_or_else(||"0.5".into()).parse().unwrap();println!("{:?}",incant!(work(std::hint::black_box([x;8])),[v3,neon,scalar]).map(f32::to_bits));}
''')
subprocess.run(['cargo','generate-lockfile','--manifest-path',str(out/'baseline/consumer/Cargo.toml')],check=True)
(out/'candidate/consumer/Cargo.lock').write_bytes((out/'baseline/consumer/Cargo.lock').read_bytes())
records=[]
for pair in range(3):
    for label,rev in (revisions if pair%2==0 else list(reversed(revisions))):
        root=out/label;target=root/f'target-{pair}'
        assert not target.exists()
        cmd=['cargo','build','--locked','--release','--manifest-path',str(root/'consumer/Cargo.toml')]
        flags=[f'--remap-path-prefix={root/source.name}=source',f'--remap-path-prefix={root/consumer.name}=consumer']
        with (root/f'build-{pair}.log').open('x') as f:
            subprocess.run(['/usr/bin/time','-v','-o',str(root/f'build-{pair}.time'),*cmd],env=dict(os.environ,CARGO_TARGET_DIR=str(target),CARGO_ENCODED_RUSTFLAGS="\x1f".join(flags)),stdout=f,stderr=subprocess.STDOUT,check=True)
        binary=target/'release/portability-build-consumer'
        record=dict(label=label,revision=rev,pair=pair,command=cmd,rustflags=flags,binary_bytes=binary.stat().st_size,time_file=str(root/f'build-{pair}.time'))
        records.append(record);(out/'records.json').write_text(json.dumps(records,indent=2)+'\n')
subprocess.run(['rustc','-Vv'],stdout=(out/'compiler.txt').open('w'),check=True)

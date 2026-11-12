#!/usr/bin/env python3
"""Run a tiny real-crate chunk/tail reproducer from a clean local checkout.

The caller supplies checkout and durable output paths. This script reads the
checkout, archives committed source, and builds only the isolated archived copy.
The generated program intentionally fails if vector and scalar tail bits differ.
"""
from pathlib import Path
import argparse,io,json,subprocess,tarfile
p=argparse.ArgumentParser();p.add_argument('checkout',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
assert not subprocess.check_output(['git','-C',str(a.checkout),'status','--porcelain'],text=True).strip(),'checkout must be clean'
sha=subprocess.check_output(['git','-C',str(a.checkout),'rev-parse','HEAD'],text=True).strip()
a.output.mkdir(parents=True)
source=a.output/'source';source.mkdir()
blob=subprocess.check_output(['git','-C',str(a.checkout),'archive',sha])
with tarfile.open(fileobj=io.BytesIO(blob),mode='r:') as archive:archive.extractall(source,filter='data')
probe=a.output/'probe';(probe/'src').mkdir(parents=True)
(probe/'Cargo.toml').write_text('[package]\nname="linear-srgb-tail-reproducer"\nversion="0.0.0"\nedition="2024"\n[workspace]\n[dependencies]\nlinear-srgb={path="../source",features=["std","avx512"]}\n')
(probe/'src/main.rs').write_text('''fn main() {
    let mut v = [0.125_f32; 17];
    linear_srgb::default::srgb_to_linear_slice(&mut v);
    println!("vector={:08x} tail={:08x}", v[0].to_bits(), v[16].to_bits());
    assert_eq!(v[0].to_bits(), v[16].to_bits(), "slice chunk/tail mismatch at input 0x3e000000");
}
''')
(a.output/'SOURCE.json').write_text(json.dumps({'linear_srgb_source':sha,'command':'cargo run --release --manifest-path probe/Cargo.toml','original_checkout_mutated':False,'input_bits':'3e000000','vector_width':16,'length':17},indent=2)+'\n')
subprocess.run(['cargo','run','--release','--manifest-path',str(probe/'Cargo.toml')],check=True)

#!/usr/bin/env python3
"""Three cold-target build diagnostics; no runtime performance inference."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
p.add_argument('--base',required=True)
p.add_argument('--candidate',required=True)
a=p.parse_args();a.directory=a.directory.resolve();a.directory.mkdir(parents=True,exist_ok=False)
manifest={'base':a.base,'candidate':a.candidate,'compiler':subprocess.check_output(['rustc','-vV'],text=True),'builds':[]}
for name,revision in [('base',a.base),('candidate',a.candidate)]:
    archive=a.directory/f'{name}.tar'
    subprocess.run(['git','archive','--format=tar',f'--output={archive}',revision],check=True)
    source=a.directory/name;source.mkdir()
    with tarfile.open(archive) as t:t.extractall(source,filter='data')
legacy='''use archmage::ScalarToken;
use magetypes::simd::generic::f32x4;
fn main(){let x=f32x4::from_array_t(ScalarToken,std::hint::black_box([0.5,0.75,1.0,1.5]));std::hint::black_box(x.pow_midp(std::hint::black_box(2.4)).to_array());}
'''
for case,source,code in [('base_legacy','base',legacy),('candidate_legacy','candidate',legacy),('candidate_powf','candidate',legacy.replace('.pow_midp(std::hint::black_box(2.4))','.try_powf(std::hint::black_box(2.4)).unwrap()'))]:
    consumer=a.directory/case;(consumer/'src').mkdir(parents=True)
    root=a.directory/source
    (consumer/'Cargo.toml').write_text(f'''[package]
name="powf-cold-consumer"
version="0.0.0"
edition="2024"
[workspace]
[dependencies]
archmage={{path="{root}",features=["macros"]}}
magetypes={{path="{root}/magetypes"}}
[profile.release]
codegen-units=1
lto="thin"
''')
    (consumer/'src/main.rs').write_text(code)
    subprocess.run(['cargo','generate-lockfile','--offline','--manifest-path',str(consumer/'Cargo.toml')],check=True)
for rep in range(3):
    order=['base_legacy','candidate_legacy'] if rep%2==0 else ['candidate_legacy','base_legacy']
    for case in order+['candidate_powf']:
        consumer=a.directory/case
        source=a.directory/('base' if case=='base_legacy' else 'candidate')
        target=a.directory/f'target-{case}-{rep}'
        flags=f'--remap-path-prefix={source}=/archmage-source --remap-path-prefix={consumer}=/consumer'
        env=dict(os.environ,CARGO_TARGET_DIR=str(target),RUSTFLAGS=flags)
        env.pop('RUSTC_WRAPPER',None)
        cmd=['cargo','build','--release','--locked','--offline','--manifest-path',str(consumer/'Cargo.toml')]
        rss=a.directory/f'{case}-{rep}.time'
        with (a.directory/f'{case}-{rep}.log').open('w') as log:
            start=time.monotonic_ns()
            run=subprocess.run(['/usr/bin/time','-v','-o',str(rss),*cmd],env=env,stdout=log,stderr=subprocess.STDOUT)
            elapsed=time.monotonic_ns()-start
        assert run.returncode==0,(case,rep)
        binary=target/'release/powf-cold-consumer'
        manifest['builds'].append({'case':case,'rep':rep,'elapsed_ns':elapsed,'binary_bytes':binary.stat().st_size,'RUSTFLAGS':flags,'command':cmd,'rss_log':str(rss),'binary_sha256':subprocess.check_output(['sha256sum',str(binary)],text=True).split()[0]})
        (a.directory/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(manifest['builds'][-1]),flush=True)

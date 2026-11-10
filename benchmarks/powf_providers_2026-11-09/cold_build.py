#!/usr/bin/env python3
"""Three fresh-target compile/size diagnostics; every build uses run-heavy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('base',type=Path);p.add_argument('patched',type=Path);p.add_argument('directory',type=Path)
p.add_argument('--toolchain',required=True)
a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=False)
for name,archive in [('base',a.base),('patched',a.patched)]:
    source=a.directory/name;source.mkdir()
    with tarfile.open(archive)as f:f.extractall(source,filter='data')
manifest={'archives':{'base':sha(a.base),'patched':sha(a.patched)},'toolchain':a.toolchain,
          'profile':'release, thin LTO, 1 codegen unit, default distribution target',
          'memory_cap':'6G','jobs':2,'download_excluded':True,'repetitions':3,
          'scope':'legacy-only on both trees; patched plus const Gamma MIDP or opaque runtime options are separate inclusion costs',
          'results':[]}
for executable in ['rustc','cargo']:
    manifest[executable]=subprocess.check_output([executable,'+'+a.toolchain,'-Vv'],text=True)
save(a.directory/'manifest.json',manifest)
variants=[('base_legacy','base',None),('patched_legacy','patched',None),
          ('patched_const_Gamma_MIDP','patched','const_policy'),('patched_runtime_options','patched','runtime_policy')]
wrapper=Path.home()/'work/claudehints/scripts/run-heavy'
client=Path(__file__).with_name('cold_client.rs').read_text()
lock=None
for name,tree,feature in variants:
    directory=a.directory/name;(directory/'src').mkdir(parents=True)
    source=(a.directory/tree).resolve()
    (directory/'Cargo.toml').write_text(f'''[package]
name="powf-cold-client"
version="0.0.0"
edition="2024"
publish=false
[workspace]
[features]
const_policy=[]
runtime_policy=[]
[dependencies]
archmage={{path={json.dumps(str(source))}}}
magetypes={{path={json.dumps(str(source/'magetypes'))},features=["std"]}}
[profile.release]
lto="thin"
codegen-units=1
''')
    (directory/'src/main.rs').write_text(client)
    with (directory/'prepare.log').open('w')as out:
        subprocess.run([str(wrapper),'--mem','6G','--jobs','2','--','cargo','+'+a.toolchain,'generate-lockfile','--manifest-path',str(directory/'Cargo.toml')],stdout=out,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(wrapper),'--mem','6G','--jobs','2','--','cargo','+'+a.toolchain,'fetch','--locked','--manifest-path',str(directory/'Cargo.toml')],stdout=out,stderr=subprocess.STDOUT,check=True)
    current=(directory/'Cargo.lock').read_bytes()
    if lock is None:lock=current
    assert current==lock,'dependency locks differ; stop before comparing builds'
# Freeze order before timing; rotate to distribute temporal drift. The target
# directories are always new. Dependency downloads above are excluded.
order=[variants,variants[1:]+variants[:1],variants[2:]+variants[:2]]
for repetition,ordered in enumerate(order):
    for name,tree,feature in ordered:
        directory=a.directory/name;target=a.directory/f'cold-target-{repetition}-{name}'
        assert not target.exists()
        env={k:v for k,v in os.environ.items()if k not in ['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_TARGET_DIR']}
        env['CARGO_TARGET_DIR']=str(target.resolve())
        command=['cargo','+'+a.toolchain,'build','--release','--locked','--manifest-path',str(directory/'Cargo.toml')]
        if feature:command+=['--features',feature]
        stem=a.directory/f'{repetition}-{name}';timefile=stem.with_suffix('.time.txt')
        save(a.directory/'progress.json',{'stage':'building','repetition':repetition,'variant':name,'command':command})
        with stem.with_suffix('.log').open('w')as log:
            run=subprocess.run([str(wrapper),'--mem','6G','--jobs','2','--','/usr/bin/time','-v','-o',str(timefile),*command],env=env,stdout=log,stderr=subprocess.STDOUT)
        if run.returncode:
            save(a.directory/'progress.json',{'stage':'failed','variant':name,'repetition':repetition,'returncode':run.returncode})
            raise SystemExit('build failed; original logs and target retained')
        binary=target/'release/powf-cold-client';frozen=stem.with_suffix('.binary');shutil.copy2(binary,frozen)
        subprocess.run([str(frozen.resolve())],check=True)
        size=subprocess.check_output(['size',str(frozen)],text=True)
        result={'variant':name,'repetition':repetition,'binary_bytes':frozen.stat().st_size,
                'binary_sha256':sha(frozen),'size':size,'GNU_time':timefile.read_text(),'command':command}
        manifest['results'].append(result);save(a.directory/'manifest.json',manifest)
        # Only this completed build's Cargo output is reclaimed; all sources,
        # binary references, resource logs and measured data remain outside it.
        assert target.name.startswith('cold-target-')and(target/'.rustc_info.json').exists()
        shutil.rmtree(target)
        print(json.dumps({k:v for k,v in result.items()if k not in ['GNU_time','size','command']}),flush=True)
save(a.directory/'progress.json',{'stage':'complete','completed_builds':len(manifest['results']),'required':12})

#!/usr/bin/env python3
"""Ablate emitted inline attributes in pinned real consumers; never edit checkouts.
Run prepare/build/measure through run-heavy, serially. All outputs are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
POLICIES = ('baseline', 'body_none', 'body_never', 'proof_none', 'proof_hint', 'dispatcher_always')
PROFILES = ('release', 'ship')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def replace(path, before, after, expected=1):
    s = path.read_text()
    assert s.count(before) == expected, (path, before, s.count(before), expected)
    path.write_text(s.replace(before, after))

def policy_patch(stack, policy):
    arc = stack/'archmage-macros/src/arcane.rs'
    rite = stack/'archmage-macros/src/rite.rs'
    auto = stack/'archmage-macros/src/autoversion.rs'
    if policy.startswith('body_'):
        body = 'None' if policy == 'body_none' else 'Some(parse_quote!(#[inline(never)]))'
        replace(arc, '''let inline_attr: Attribute = if args.inline_always {
        parse_quote!(#[inline(always)])
    } else {
        parse_quote!(#[inline])
    };''', f'''let inline_attr: Option<Attribute> = if args.inline_always {{
        Some(parse_quote!(#[inline(always)]))
    }} else {{
        {body}
    }};''')
        replace(arc, 'inline_attr: Attribute,', 'inline_attr: Option<Attribute>,', 2)
        replace(arc, 'new_attrs.push(inline_attr);', 'new_attrs.extend(inline_attr);')
        new = '// Experimental absence of the implicit inline attribute.' if policy == 'body_none' else 'new_attrs.push(parse_quote!(#[inline(never)]));'
        replace(rite, 'new_attrs.push(parse_quote!(#[inline]));', new)
    elif policy.startswith('proof_'):
        new = '' if policy == 'proof_none' else '#[inline]'
        replace(arc, '        #[inline(always)]\n        #vis #wrapper_sig', f'        {new}\n        #vis #wrapper_sig', 2)
    elif policy == 'dispatcher_always':
        replace(auto, 'let fn_attrs: Vec<Attribute> = expect_as_allow(&input_fn.attrs);', '''let mut fn_attrs: Vec<Attribute> = expect_as_allow(&input_fn.attrs);
    fn_attrs.retain(|a| !a.path().is_ident("inline"));
    fn_attrs.push(syn::parse_quote!(#[inline(always)]));''')
    elif policy != 'baseline':
        raise ValueError(policy)


def clean_env():
    env = dict(os.environ, CARGO_INCREMENTAL='0', RUSTFLAGS='', TMPDIR=str(Path.home()/'tmp'))
    for key in ('RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','CARGO_ENCODED_RUSTFLAGS','CARGO_TARGET_DIR','CARGO_PROFILE_RELEASE_LTO','CARGO_PROFILE_RELEASE_CODEGEN_UNITS'):
        env.pop(key, None)
    return env


def prepare(out, sources, revision, image):
    out.mkdir(parents=True, exist_ok=False)
    rev = subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    archive = out/'archmage.tar'
    with archive.open('wb') as f:
        subprocess.run(['git','archive',rev],cwd=ROOT,stdout=f,check=True)
    plan = {'archmage':rev, 'consumer_sources':json.loads((sources/'provenance.json').read_text()),
            'rustc':subprocess.check_output(['rustc','-Vv'],text=True),
            'image':{'path':str(image),'sha256':sha(image)}, 'policies':{},
            'profiles':{'release':{'lto':'off','codegen_units':16}, 'ship':{'lto':'fat','codegen_units':1}},
            'driver_sha256':sha(HERE/'driver.rs'), 'cpu':2,
            'note':'Real consumer implementations unchanged. Macro emission ablations only. In-memory API timing; setup/drop/I/O outside clock. No target-cpu=native.'}
    for size in (256,512):
        cmd=['ffmpeg','-v','error','-threads','1','-i',str(image),'-vf',f'crop={size}:{size}',
             '-frames:v','1','-pix_fmt','yuv420p','-threads','1','-f','rawvideo',str(out/f'photo-{size}.yuv')]
        subprocess.run(cmd,check=True)
    plan['inputs'] = {str(s):sha(out/f'photo-{s}.yuv') for s in (256,512)}
    for policy in POLICIES:
        case=out/policy
        stack=case/'archmage'
        with tarfile.open(archive) as f: f.extractall(stack,filter='data')
        policy_patch(stack,policy)
        for consumer in ('zenav1-svt','rav1d-safe'):
            shutil.copytree(sources/consumer,case/consumer)
        manifest=case/'rav1d-safe/Cargo.toml'
        old='git = "https://github.com/imazen/archmage", rev = "b8d6c0777e55e4ea8d26cde0879faae858cba662"'
        replace(manifest,old,'version = "0.9.30"',2)
        driver=case/'driver'; (driver/'src').mkdir(parents=True)
        shutil.copyfile(HERE/'driver.rs',driver/'src/main.rs')
        (driver/'Cargo.toml').write_text(f'''[package]
name="inline-consumer-probe"
version="0.0.0"
edition="2024"
[workspace]
[dependencies]
zenav1-svt-encoder={{path="../zenav1-svt/rust/crates/svtav1-encoder"}}
rav1d-safe={{path="../rav1d-safe"}}
zenbench={{version="=0.1.10",default-features=false}}
[patch.crates-io]
archmage={{path="../archmage"}}
archmage-macros={{path="../archmage/archmage-macros"}}
magetypes={{path="../archmage/magetypes"}}
[profile.release]
opt-level=3
lto="off"
codegen-units=16
incremental=false
[profile.ship]
inherits="release"
lto="fat"
codegen-units=1
''')
        # Equal compiler support dependencies across every policy; no new public API.
        with (case/'prepare.log').open('w') as log:
            meta=subprocess.check_output(['cargo','metadata','--format-version','1'],cwd=driver,env=clean_env(),stderr=log)
        metadata=json.loads(meta)
        selected={p['name']:{'version':p['version'],'manifest':p['manifest_path']} for p in metadata['packages'] if p['name'] in ('archmage','archmage-macros','magetypes','zenbench')}
        assert all(selected[n]['version']=='0.9.30' for n in ('archmage','archmage-macros','magetypes')),selected
        plan['policies'][policy]={'driver':str(driver),'selected':selected,'lock_sha256':sha(driver/'Cargo.lock')}
        save(out/'plan.json',plan)
        print('Prepared',policy,flush=True)


def build(out, profiles, policies):
    plan=json.loads((out/'plan.json').read_text())
    dest=out/'builds'; dest.mkdir(exist_ok=True)
    for profile in profiles:
        for policy in policies:
            key=f'{profile}-{policy}'
            log_path=dest/f'{key}.log'
            assert not log_path.exists(),log_path
            cwd=Path(plan['policies'][policy]['driver'])
            cmd=['cargo','build','--profile',profile,'--locked']
            start=time.monotonic()
            with log_path.open('w') as log:
                rc=subprocess.run(['/usr/bin/time','-v','-o',str(dest/f'{key}.time')]+cmd,cwd=cwd,env=clean_env(),stdout=log,stderr=subprocess.STDOUT).returncode
            elapsed=time.monotonic()-start
            if rc:raise RuntimeError(f'{key} failed ({rc}); see {log_path}')
            binary=cwd/'target'/profile/'inline-consumer-probe'
            size=subprocess.check_output(['size',str(binary)],text=True)
            record={'command':cmd,'wall_seconds':elapsed,'binary':str(binary),'sha256':sha(binary),'size':size,'status':rc}
            save(dest/f'{key}.json',record)
            print('Built',key,round(elapsed,3),size.splitlines()[-1],flush=True)


def measure(out, profiles, policies, passes):
    plan=json.loads((out/'plan.json').read_text())
    root=out/'runs'; root.mkdir(exist_ok=True)
    all_rows=[]
    for profile in profiles:
        for repetition in range(passes):
            order=list(policies); random.Random(713+repetition).shuffle(order)
            for policy in order:
                binary=json.loads((out/'builds'/f'{profile}-{policy}.json').read_text())['binary']
                for size,qp,preset in [(256,32,8),(512,20,6)]:
                    key=f'{profile}-{policy}-{size}-{repetition}'
                    dest=root/key
                    log_path=root/f'{key}.log'
                    assert not log_path.exists()
                    cmd=['taskset','-c',str(plan['cpu']),binary,str(out/f'photo-{size}.yuv'),str(size),str(qp),str(preset),str(dest)]
                    with log_path.open('w') as log:
                        rc=subprocess.run(cmd,cwd=out,env=clean_env(),stdout=log,stderr=subprocess.STDOUT).returncode
                    if rc:raise RuntimeError(f'{key} failed ({rc}); see {log_path}')
                    text=log_path.read_text()
                    checks={name:sha(dest/name) for name in ('encoded.obu','decoded.yuv')}
                    for line in text.splitlines():
                        if line.startswith('SUMMARY\t'):
                            _,workload,n,mean,median,mad=line.split('\t')
                            all_rows.append({'profile':profile,'policy':policy,'size':size,'qp':qp,'preset':preset,'pass':repetition,
                                             'workload':workload,'n':int(n),'mean_ns':float(mean),'median_ns':float(median),'mad_ns':float(mad),
                                             'checks':checks,'reliable':'RELIABLE\ttrue\t' in text,'log':str(log_path)})
                    assert sum(r['log']==str(log_path) for r in all_rows)==2
                    for previous in all_rows:
                        if previous['size']==size:assert previous['checks']==checks,(key,previous['log'],'OUTPUT MISMATCH')
                    save(out/'measurements.json',all_rows)
                    print('Measured',key,checks,flush=True)
    summarize(out)


def summarize(out):
    rows=json.loads((out/'measurements.json').read_text())
    summary=[]
    for profile in sorted({r['profile'] for r in rows}):
        for size in sorted({r['size'] for r in rows}):
            for workload in ('encode','decode'):
                base=[r['median_ns'] for r in rows if (r['profile'],r['size'],r['workload'],r['policy'])==(profile,size,workload,'baseline')]
                if not base:continue
                for policy in POLICIES:
                    selected=[r for r in rows if (r['profile'],r['size'],r['workload'],r['policy'])==(profile,size,workload,policy)]
                    if not selected:continue
                    values=[r['median_ns'] for r in selected]
                    summary.append({'profile':profile,'size':size,'workload':workload,'policy':policy,'passes':len(values),
                                    'median_ns':statistics.median(values),'min_ns':min(values),'max_ns':max(values),
                                    'vs_baseline_pct':100*(statistics.median(values)/statistics.median(base)-1),
                                    'all_reliable':all(r['reliable'] for r in selected)})
    save(out/'summary.json',summary)
    for r in summary:print(r)


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','build','measure','summarize']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sources',type=Path);p.add_argument('--revision',default='origin/main');p.add_argument('--image',type=Path)
    p.add_argument('--profiles',nargs='+',choices=PROFILES,default=list(PROFILES));p.add_argument('--policies',nargs='+',choices=POLICIES,default=list(POLICIES));p.add_argument('--passes',type=int,default=5)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.out,a.sources,a.revision,a.image)
    elif a.action=='build':build(a.out,a.profiles,a.policies)
    elif a.action=='measure':measure(a.out,a.profiles,a.policies,a.passes)
    else:summarize(a.out)
if __name__=='__main__':main()

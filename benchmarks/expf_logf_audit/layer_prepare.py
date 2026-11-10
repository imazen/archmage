#!/usr/bin/env python3
"""Three matched crate topologies; preserve legacy implementations without cycles."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('revision');p.add_argument('output',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
revision=subprocess.check_output(['git','rev-parse',a.revision],text=True).strip()
for mode in ['direct','facade','owned-compat']:
    root=a.output/mode;source=root/'archmage';source.mkdir(parents=True)
    process=subprocess.Popen(['git','archive',revision],stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout,mode='r|')as archive:archive.extractall(source,filter='data')
    assert process.wait()==0
    vectors=source/'magetypes'
    # All three use the same public-primitive adapter, isolating ownership from
    # the separate question of which internal helpers a math crate can access.
    shutil.copy2(HERE/'layer_adapter.rs',vectors/'src/math_primitive_adapter.rs')
    lib=vectors/'src/lib.rs';lib.write_text(lib.read_text()+'\nmod math_primitive_adapter;\n')
    for f in (vectors/'src/powf').glob('*.rs'):
        s=f.read_text().replace('crate::soft_fma::fmaf_soft_lanes','crate::math_primitive_adapter::fmaf_soft_lanes').replace('crate::soft_fma::fma_lanes','crate::math_primitive_adapter::fma_lanes');f.write_text(s)
    # Keep std code unchanged; repair the subject's known no_std-only compile
    # error equally in all controls so default-no_std can be checked separately.
    f=vectors/'src/logf/const_api.rs';s=f.read_text()
    changes={
        'v.to_array()[0].mul_add(core::f32::consts::LN_2, c)':'log1p_finish(v.to_array()[0], c)',
        'o[i].mul_add(core::f32::consts::LN_2, c[i])':'log1p_finish(o[i], c[i])',
    }
    for old,new in changes.items():assert old in s;s=s.replace(old,new)
    s+='''\n#[inline(always)]\nfn log1p_finish(x:f32,c:f32)->f32{\n#[cfg(feature="std")] {x.mul_add(core::f32::consts::LN_2,c)}\n#[cfg(not(feature="std"))] {crate::math_primitive_adapter::fmaf_soft_lanes([x],[core::f32::consts::LN_2],[c])[0]}\n}\n'''
    f.write_text(s)
    if mode!='direct':
        math=root/'magemath';(math/'src').mkdir(parents=True)
        (math/'Cargo.toml').write_text('''[package]
name="magemath"
version="0.0.0"
edition="2024"
publish=false
license="MIT OR Apache-2.0"
[features]
default=[]
std=["magetypes/std","archmage/std"]
w512=["magetypes/w512"]
avx512=["w512","magetypes/avx512","archmage/avx512"]
_exhaustive-tests=[]
[dependencies]
magetypes={path="../archmage/magetypes",default-features=false}
archmage={path="../archmage",default-features=false}
''')
        if mode=='facade':
            (math/'src/lib.rs').write_text('#![no_std]\n#![forbid(unsafe_code)]\npub use magetypes::{powf,expf,logf};\n')
        else:
            for family in ['powf','expf','logf']:
                shutil.copytree(vectors/'src'/family,math/'src'/family)
                f=math/'src'/family/'const_api.rs';s=f.read_text();marker='// Expand the scalar composition'if family=='powf'else'// Existing infallible methods use'
                if marker in s:
                    start=s.index(marker)
                    if family=='powf':
                        end=s.index('// The runtime adapter shares literal const instantiations',start)
                        s=s[:start]+s[end:]
                    else:s=s[:start]
                if family=='logf':
                    # log1p_finish was appended after the compatibility helpers.
                    full=(vectors/'src/logf/const_api.rs').read_text();s+=full[full.index('\n#[inline(always)]\nfn log1p_finish'):]
                f.write_text(s)
            shutil.copy2(HERE/'layer_adapter.rs',math/'src/math_primitive_adapter.rs')
            shutil.copy2(vectors/'src/wasm_fma.rs',math/'src/wasm_fma.rs')
            shutil.copy2(vectors/'LICENSE-ARM',math/'LICENSE-ARM')
            (math/'src/lib.rs').write_text('''#![no_std]
#![forbid(unsafe_code)]
#[cfg(feature="std")] extern crate std;
extern crate alloc;
mod simd{pub use magetypes::simd::*;}
mod math_primitive_adapter;
#[cfg(target_arch="wasm32")]
mod soft_fma{pub(crate) use crate::math_primitive_adapter::{fma_lanes,fmaf_soft_lanes};}
#[cfg(target_arch="wasm32")] mod wasm_fma;
#[cfg(test)] mod nostd_math{pub fn fmaf_soft(a:f32,b:f32,c:f32)->f32{crate::math_primitive_adapter::fmaf_soft_lanes([a],[b],[c])[0]}}
pub mod powf;
pub mod expf;
pub mod logf;
''')
    consumer=root/'consumer';(consumer/'src').mkdir(parents=True)
    extra = mode != 'direct'
    std_features='["magetypes/std","archmage/std"'+(',"magemath/std"'if extra else'')+']'
    width_features='["magetypes/w512"'+(',"magemath/w512"'if extra else'')+']'
    extra_dependency='magemath={path="../magemath",default-features=false}\n'if extra else''
    (consumer/'Cargo.toml').write_text(f'''[package]
name="magemath-layer-consumer"
version="0.0.0"
edition="2024"
publish=false
[workspace]
resolver="3"
[features]
default=["std","w512"]
std={std_features}
w512={width_features}
[dependencies]
archmage={{path="../archmage",default-features=false}}
magetypes={{path="../archmage/magetypes",default-features=false}}
{extra_dependency}
[profile.release]
lto=false
codegen-units=16
[profile.thin]
inherits="release"
lto="thin"
codegen-units=1
''')
    consumer_source=(HERE/'layer_consumer.rs').read_text().replace('use archmage::SimdToken;', 'use archmage::SimdToken;\nuse '+('magemath'if extra else'magetypes')+' as math;')
    (consumer/'src/main.rs').write_text(consumer_source)
manifest=dict(revision=revision,modes=['direct','facade','owned-compat'],families=['powf','expf','logf'],
    scope='retains legacy/public implementations in Magetypes; owner copies only configurable math; no dependency cycle; no deprecation applied',
    common_changes='public-vector FMA adapter for private powf helpers; no_std-only log1p compilation repair, same in all three modes')
(a.output/'PREPARATION.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))

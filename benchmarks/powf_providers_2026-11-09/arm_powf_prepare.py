#!/usr/bin/env python3
"""Compile pinned Arm powf alone with explicit architecture and errno profiles."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
from external import native_identity

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('repository',type=Path);p.add_argument('directory',type=Path)
    p.add_argument('--snapshot-manifest',type=Path,help='rebuild a frozen source snapshot after verifying every source hash')
    a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=False)
    frozen=json.loads(a.snapshot_manifest.read_text())if a.snapshot_manifest else None
    revision=frozen['source_revision']if frozen else subprocess.check_output(['git','-C',str(a.repository),'rev-parse','HEAD'],text=True).strip()
    assert revision=='4be260a5117480382690c6d8c300bc784e927d76'
    # Ignore the caller's local claim; source changes would invalidate this pin.
    if frozen:
        actual={str(path.relative_to(a.repository)):digest(path)for path in a.repository.rglob('*')if path.is_file()}
        assert actual==frozen['source_files'],'frozen source differs'
    else:subprocess.run(['git','-C',str(a.repository),'diff','--quiet','HEAD','--','math'],check=True)
    snapshot=a.directory/'source';snapshot.mkdir()
    names=['powf.c','powf_log2_data.c','exp2f_data.c','math_errf.c','math_config.h','powf_common.h']
    source_root=a.repository if frozen else a.repository/'math'
    for name in names:shutil.copy2(source_root/name,snapshot/name)
    # Rename after math.h's vector declarations have been parsed.
    (snapshot/'powf_entry.c').write_text('#include <math.h>\n#define powf arm_powf_reference\n#include "powf.c"\n')
    shutil.copytree(source_root/'include',snapshot/'include')
    for name in ['LICENSE','LICENSE-MIT','LICENSE-Apache','LICENSE-LLVM','README.md']:
        source=a.repository/name
        if source.is_file():shutil.copy2(source,snapshot/name)
    darwin=platform.system()=='Darwin'
    common=['-std=c11','-O3','-g','-fPIC','-dynamiclib'if darwin else'-shared','-fno-fast-math',
            '-frounding-math','-fexcess-precision=standard','-DUSE_GLIBC_ABI=0',
            '-DWANT_ROUNDING=1','-I',str((snapshot/'include').resolve())]
    if platform.machine()=='x86_64':
        common+=['-march=x86-64']
        variants=[('x86_64_separate_errno1',['-mno-avx','-ffp-contract=off','-DWANT_ERRNO=1']),
                  ('AVX2_FMA_errno1',['-mavx2','-mfma','-ffp-contract=fast','-DWANT_ERRNO=1']),
                  ('AVX_FMA_no_AVX2_errno1',['-mavx','-mfma','-mno-avx2','-ffp-contract=fast','-DWANT_ERRNO=1']),
                  ('AVX2_FMA_errno0',['-mavx2','-mfma','-ffp-contract=fast','-DWANT_ERRNO=0'])]
    elif platform.machine() in ['aarch64','arm64']:
        common+=['-march=armv8-a']
        variants=[('Armv8_FMA_errno1',['-ffp-contract=fast','-DWANT_ERRNO=1']),
                  ('Armv8_separate_errno1',['-ffp-contract=off','-DWANT_ERRNO=1']),
                  ('Armv8_FMA_TOINT0_errno1',['-ffp-contract=fast','-DWANT_ERRNO=1','-DHAVE_FAST_ROUND=0','-DHAVE_FAST_LROUND=0'])]
    else:raise RuntimeError('unregistered platform')
    native_flag='-mcpu=native'if darwin and platform.machine()=='arm64'else'-march=native'
    variants.append(('native_errno1',[native_flag,'-ffp-contract=fast','-DWANT_ERRNO=1']))
    report={'source_revision':revision,'compiler':subprocess.check_output(['cc','--version'],text=True),
            'native_identity':native_identity(),
            'source_files':{str(path.relative_to(snapshot)):digest(path)for path in sorted(snapshot.rglob('*'))if path.is_file()},
            'builds':[],'claim':'isolated scalar C reference; no Rust port'}
    for name,features in variants:
        library=(a.directory/f'arm-powf-{name}{".dylib"if darwin else".so"}').resolve()
        sources=[str((snapshot/name).resolve())for name in ['powf_entry.c',*names[1:4]]]
        command=['cc',*common,*features,*sources,'-Wl,-undefined,error'if darwin else'-Wl,-z,defs','-lm','-o',str(library)]
        with (a.directory/f'{name}.log').open('w')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        disassembler=['xcrun','llvm-objdump']if darwin else['objdump']
        assembly=subprocess.check_output([*disassembler,'-d',str(library)],text=True)
        (a.directory/f'{name}.asm').write_text(assembly)
        if platform.machine()=='x86_64':
            assert ('vfmadd' in assembly or 'vfmsub' in assembly)==('separate'not in name)
        report['builds'].append({'name':name,'method':'Arm_powf_C_'+name,'environment':'POWF_ARM_C_'+name.upper(),
                                 'command':command,'binary':str(library),'sha256':digest(library)})
    (a.directory/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

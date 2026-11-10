#!/usr/bin/env python3
"""Frozen provider discovery and supervised zenbench batch matrix."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

RUST_METHODS = ['std', 'Rust_libm_0_2_16', 'SLEEF_0_3_3_scalar_u10',
                'SLEEF_0_3_3_scalar_fast', 'SLEEF_0_3_3_vector_u10',
                'SLEEF_0_3_3_vector_fast']
ARM_C_ENV = {'POWF_ARM_C_'+name.upper() for name in [
    'x86_64_separate_errno1','AVX2_FMA_errno1','AVX_FMA_no_AVX2_errno1',
    'AVX2_FMA_errno0','native_errno1','Armv8_FMA_errno1','Armv8_separate_errno1','Armv8_FMA_TOINT0_errno1']}

def native_identity():
    if sys.platform=='linux':
        fields=dict(line.split(':',1)for line in Path('/proc/cpuinfo').read_text().split('\n\n',1)[0].splitlines()if ':'in line)
        fields={k.strip():v.strip()for k,v in fields.items()}
        return {'machine':platform.machine(),'model':fields.get('model name',fields.get('CPU implementer')),
                'features':sorted(fields.get('flags',fields.get('Features','')).split())}
    if sys.platform=='darwin':
        return {'machine':platform.machine(),'model':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                'OS_build':subprocess.check_output(['sw_vers','-buildVersion'],text=True).strip()}
    raise RuntimeError('unregistered native diagnostic host')

def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_manifest(path):
    """Restore the frozen DSO selection before launching any worker."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get('native_identity') is not None:
        assert manifest['native_identity']==native_identity(),'host-specific native build moved to another CPU/context'
    allowed = {'POWF_ARM_DSO', 'POWF_AOCL_DSO', 'POWF_AOCL_FAST_DSO',
               'POWF_CORE_MATH_DSO', 'POWF_CORE_MATH_FMA_DSO'} | ARM_C_ENV
    assert set(manifest['dsos']) <= allowed
    for name in allowed:
        os.environ.pop(name, None)
    for name, identity in manifest['dsos'].items():
        assert digest(identity['path']) == identity['sha256'], 'frozen DSO changed'
        os.environ[name] = identity['path']
    for resolved, identity in manifest.get('resolved_native_dsos', {}).items():
        if 'sha256' in identity:
            assert digest(resolved) == identity['sha256'], 'resolved native DSO changed'
    if manifest.get('os_build') is not None:
        assert subprocess.check_output(['sw_vers', '-buildVersion'], text=True).strip() == manifest['os_build'], 'macOS shared-cache build changed'
    for worker in manifest['workers']:
        assert digest(worker['binary']) == worker['sha256'], 'frozen worker changed'
    for filename,identity in manifest.get('referenced_files',{}).items():
        assert digest(filename)==identity['sha256'],'frozen execution input changed'
    if 'runtime_env' in manifest:
        assert set(manifest['runtime_env']) <= {'GLIBC_TUNABLES'}
        for name,value in manifest['runtime_env'].items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value
    return manifest

def discover(rust, native, v3=None):
    workers = [{'name': 'Rust_baseline', 'binary': str(rust.resolve()), 'sha256': digest(rust)},
               {'name': 'native_C_ABI', 'binary': str(native.resolve()), 'sha256': digest(native)}]
    providers = [{'label': m+'_Rust_baseline', 'method': m, 'worker': 0, 'lanes': 1}
                 for m in RUST_METHODS]
    result = subprocess.run([str(native.resolve()), '--list'], capture_output=True, text=True)
    result.check_returncode()
    resolved={}
    for line in result.stderr.splitlines():
        if ' DSO ' not in line:continue
        path=line.split(' DSO ',1)[1]
        if Path(path).is_file():resolved[path]={'sha256':digest(path)}
        else:
            assert sys.platform=='darwin' and path.startswith('/usr/lib/'), 'unresolved native DSO identity'
            resolved[path]={'identity':'macOS shared-cache mapping; retain sw_vers OS build identity'}
    for line in result.stdout.splitlines():
        method, lanes = line.split()
        providers.append({'label': method, 'method': method, 'worker': 1, 'lanes': int(lanes)})
    if v3:
        workers.append({'name': 'Rust_AVX2_FMA', 'binary': str(v3.resolve()), 'sha256': digest(v3)})
        providers.extend({'label': m+'_Rust_AVX2_FMA', 'method': m, 'worker': 2, 'lanes': 1}
                         for m in RUST_METHODS[2:])
    return {'workers': workers, 'providers': providers, 'native_diagnostics': result.stderr,
            'os_build': subprocess.check_output(['sw_vers', '-buildVersion'], text=True).strip() if sys.platform == 'darwin' else None,
            'resolved_native_dsos':resolved,
            'dsos': {name: {'path': value, 'sha256': digest(value)} for name, value in os.environ.items()
                     if name in {'POWF_ARM_DSO', 'POWF_AOCL_DSO', 'POWF_AOCL_FAST_DSO', 'POWF_CORE_MATH_DSO', 'POWF_CORE_MATH_FMA_DSO'} | ARM_C_ENV},
            'timing_unit': 'ns per 1,048,576 actual evaluations including one socket request',
            'compiler_flags': {'Rust_baseline': 'build-arm label; verify actual flags from retained Cargo fingerprint; nightly-2026-10-06',
                               'Rust_AVX2_FMA': '-C target-feature=+avx2,+fma; physical support required'},
            'claim': 'benchmark-only; no production admission or universal accuracy guarantee'}

def cells():
    return [{'width': w, 'size': size, 'chain': chain, 'exponent': exponent,
             'group': f'external/x{w}/{size}/{"chain" if chain else "stream"}/{exponent}/1048576-evaluations'}
            for w in [1, 4, 8, 16] for size in [64, 4096]
            for chain in [False, True] for exponent in ['4019999a', '3f000000']]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('discover')
    d.add_argument('rust', type=Path); d.add_argument('native', type=Path)
    d.add_argument('output', type=Path); d.add_argument('--v3', type=Path)
    r = sub.add_parser('run')
    r.add_argument('bench', type=Path); r.add_argument('manifest', type=Path)
    r.add_argument('directory', type=Path)
    r.add_argument('--exploratory', action='store_true')
    r.add_argument('--pilot', action='store_true')
    a = p.parse_args()
    if a.command == 'discover':
        assert not a.output.exists()
        save(a.output, discover(a.rust, a.native, a.v3)); return
    manifest = load_manifest(a.manifest)
    assert a.exploratory == (os.environ.get('ZENBENCH_CPU_POLICY') == 'unrestricted')
    a.directory.mkdir(exist_ok=False, parents=True)
    save(a.directory/'frozen.json', {'providers': manifest, 'cells': cells(), 'bench': str(a.bench.resolve()),
                                   'bench_sha256': digest(a.bench), 'pairs': 50, 'pilot': a.pilot,
                                   'exploratory': a.exploratory, 'supervisor_pid': os.getpid()})
    processes, logs, sockets = [], [], []
    try:
        for index, worker in enumerate(manifest['workers']):
            assert digest(worker['binary']) == worker['sha256']
            path = (a.directory/f's{index}').resolve()
            assert not path.exists()
            log = (a.directory/f'worker-{index}.log').open('w'); logs.append(log)
            child = subprocess.Popen([worker['binary'], str(path)], stdout=log, stderr=subprocess.STDOUT)
            processes.append(child); sockets.append(path)
        deadline = time.monotonic()+30
        while not all(path.exists() for path in sockets):
            assert all(child.poll() is None for child in processes), 'provider worker failed; logs preserved'
            assert time.monotonic() < deadline, 'worker startup timeout'
            time.sleep(.02)
        child_env = dict(os.environ, ZENBENCH_LAUNCHER_PIDS=','.join(map(str, [os.getpid()]+[c.pid for c in processes])))
        for index, cell in enumerate(cells()):
            if a.pilot and not (cell['width']==8 and cell['size']==4096 and not cell['chain'] and cell['exponent']=='4019999a'):
                continue
            selected = [dict(p, socket=str(sockets[p['worker']])) for p in manifest['providers'] if p['lanes']<=cell['width']]
            selected += [{'label': 'EMPTY_'+w['name'], 'method': 'EMPTY', 'socket': str(sockets[i])}
                         for i, w in enumerate(manifest['workers'])]
            config = a.directory/f'{index:03d}.manifest.json'; save(config, dict(cell, providers=selected))
            output = (a.directory/f'{index:03d}.json').resolve()
            child_env.update(POWF_EXTERNAL_MANIFEST=str(config.resolve()), POWF_RESULT=str(output))
            progress = dict(cell, index=index, required=1 if a.pilot else 32, stage='running',
                            supervisor_pid=os.getpid(), worker_pids=[c.pid for c in processes],
                            last_check=datetime.now(timezone.utc).isoformat())
            save(a.directory/'progress.json', progress)
            with (a.directory/f'{index:03d}.log').open('w') as log:
                run = subprocess.run([str(a.bench.resolve())], env=child_env, stdout=log, stderr=subprocess.STDOUT)
            if run.returncode:
                progress.update(stage='failed', returncode=run.returncode)
            else:
                result = json.loads(output.read_text())
                assert len(result['comparisons']) == 1
                samples = result['comparisons'][0]['samples']
                clean = not result['unreliable'] and len(samples)==50 and all(s['gate_clean'] is True for s in samples)
                progress.update(stage='complete' if clean else 'inadmissible', pairs=len(samples), unreliable=result['unreliable'])
            progress['last_check'] = datetime.now(timezone.utc).isoformat()
            save(a.directory/'progress.json', progress); print(json.dumps(progress), flush=True)
            if progress['stage']=='failed' or (progress['stage']=='inadmissible' and not a.exploratory):
                raise SystemExit('stopped: all evidence preserved; no favorable rerun')
    finally:
        for child in processes:
            if child.poll() is None: child.terminate()
        for child in processes: child.wait()
        for log in logs: log.close()
        for path in sockets:
            if path.exists(): path.unlink()

if __name__ == '__main__': main()

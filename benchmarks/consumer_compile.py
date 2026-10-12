#!/usr/bin/env python3
"""Cold library checks/builds of fixed consumers against two pinned archmage revisions.

Prepare outside timed runs, then use run-heavy for each run invocation. Never
reuses a target directory or overwrites logs. Full Cargo output, timings and
/usr/bin/time -v reports stay beside compact machine-readable measurements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tarfile
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ('baseline', 'candidate')
WORKLOADS = ('magetypes', 'zenav1-svt', 'rav1d-safe')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out, sources, candidate, baseline):
    out.mkdir(parents=True, exist_ok=False)
    stacks = {}
    provenance = {}
    revisions = [('baseline', baseline), ('candidate', candidate)]
    versions = VERSIONS
    normalization = {}
    for label, revision in revisions:
        revision = subprocess.check_output(['git', 'rev-parse', revision], cwd=ROOT, text=True).strip()
        provenance[label] = revision
        archive = out / f'{label}.tar'
        with archive.open('wb') as f:
            subprocess.run(['git', 'archive', revision], cwd=ROOT, stdout=f, check=True)
        destination = out / label
        with tarfile.open(archive) as f:
            f.extractall(destination, filter='data')
        # Match the fixed consumer manifests without prerelease-resolution changes.
        # Only archive copies change; both code revisions use the same versions.
        version = tomllib.loads((destination/'Cargo.toml').read_text())['workspace']['package']['version']
        normalization[label] = {'original': version, 'normalized': '0.9.29', 'manifests': {}}
        for relative in ('Cargo.toml', 'magetypes/Cargo.toml'):
            manifest = destination/relative
            before = manifest.read_bytes()
            manifest.write_bytes(before.replace(version.encode(), b'0.9.29'))
            normalization[label]['manifests'][relative] = {
                'original_sha256': hashlib.sha256(before).hexdigest(),
                'normalized_sha256': digest(manifest)}
        stacks[label] = destination
    source_provenance = json.loads((sources/'provenance.json').read_text())
    plan = {'sources': {name: source_provenance[name] for name in WORKLOADS if name != 'magetypes'}, 'stacks': provenance, 'cases': {}, 'environment': {
        'rustc': subprocess.check_output(['rustc', '-Vv'], text=True),
        'cargo': subprocess.check_output(['cargo', '-V'], text=True).strip(),
        'host': subprocess.check_output(['hostname'], text=True).strip(),
        'incremental': False, 'rustflags': '', 'compiler_cache': False,
        'note': 'Fresh Cargo outputs, warmed dependency downloads and OS caches; unchanged consumer Rust sources.'}}
    plan['variants'] = versions
    plan['normalization'] = normalization
    for workload in WORKLOADS:
        for variant in versions:
            key = f'{workload}-{variant}'
            case = out / key
            if workload == 'magetypes':
                (case / 'src').mkdir(parents=True)
                (case / 'src/lib.rs').write_text('// Dependency-only build driver.\n')
                (case / 'Cargo.toml').write_text('[package]\nname="magetypes-build-driver"\nversion="0.0.0"\nedition="2024"\n[workspace]\n[dependencies]\nmagetypes="=0.9.29"\n')
                cwd = case
            else:
                shutil.copytree(sources / workload, case)
                cwd = case / 'rust' if workload == 'zenav1-svt' else case
                if workload == 'rav1d-safe':
                    manifest = cwd / 'Cargo.toml'
                    original = manifest.read_text()
                    changed, n = re.subn(r'git = "https://github.com/imazen/archmage", rev = "[0-9a-f]+"', 'version = "=0.9.29"', original)
                    if n != 2:
                        raise ValueError(f'Expected two archmage source overrides, got {n}')
                    manifest.write_text(changed)
            config = case / 'compile-comparison.toml'
            contents = '[patch.crates-io]\n' + '\n'.join(
                    f'{name} = {{ path = {json.dumps(str(stacks[variant] / path))} }}'
                    for name, path in [('archmage','.'),('archmage-macros','archmage-macros'),('magetypes','magetypes')]) + '\n'
            config.write_text(contents)
            lock = cwd / 'Cargo.lock'
            if lock.exists():
                for package in tomllib.loads(lock.read_text())['package']:
                    if package['name'] == 'syn' and package['version'].startswith('3.') and package['version'] != '3.0.7':
                        with (case / 'lock-normalization.log').open('w') as log:
                            subprocess.run(['cargo', '--config', str(config), 'update', '-p',
                                            'syn@' + package['version'], '--precise', '3.0.7'],
                                           cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
            command = ['cargo', '--config', str(config), 'metadata', '--format-version', '1']
            with (case / 'prepare.log').open('w') as errors:
                metadata = subprocess.check_output(command, cwd=cwd, stderr=errors)
            (case / 'metadata.json').write_bytes(metadata)
            decoded = json.loads(metadata)
            selected = {p['name']: {'version':p['version'], 'source':p['source']} for p in decoded['packages'] if p['name'] in ('archmage','archmage-macros','magetypes')}
            for name in selected:
                packages = [p for p in decoded['packages'] if p['name'] == name]
                if len(packages) != 1 or not Path(packages[0]['manifest_path']).is_relative_to(stacks[variant]):
                    raise ValueError(f'Wrong or duplicate source for {name} in {key}')
            if selected['archmage']['version'] != '0.9.29' or selected['archmage-macros']['version'] != '0.9.29':
                raise ValueError(f'Unexpected dependency resolution in {key}: {selected}')
            with (case / 'fetch.log').open('w') as log:
                subprocess.run(['cargo','--config',str(config),'fetch','--locked'],cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
            plan['cases'][key] = {'cwd':str(cwd), 'config':str(config), 'package':workload,
                                  'selected':selected, 'lock_sha256':digest(cwd/'Cargo.lock'),
                                  'manifest_sha256':digest(cwd/'Cargo.toml')}
            (out / 'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
            print('Prepared',key,selected,flush=True)
    for workload in WORKLOADS:
        first, second = [plan['cases'][f'{workload}-{variant}'] for variant in versions]
        for field in ('lock_sha256', 'manifest_sha256'):
            if first[field] != second[field]:
                raise ValueError(f'Unmatched {field} for {workload}')


def run(out, mode, repetitions, workloads):
    plan = json.loads((out/'plan.json').read_text())
    env = dict(os.environ,CARGO_INCREMENTAL='0',RUSTFLAGS='',RUSTC_WRAPPER='',RUSTC_WORKSPACE_WRAPPER='')
    for name in ('CARGO_ENCODED_RUSTFLAGS',):
        env.pop(name,None)
    versions = tuple(plan.get('variants', VERSIONS))
    for repetition in range(repetitions):
        offset = repetition % len(versions)
        order = versions[offset:] + versions[:offset]
        for workload in workloads:
            for variant in order:
                key = f'{workload}-{variant}'
                case = plan['cases'][key]
                label = f'{mode}-{repetition}-{key}'
                directory = out/'runs'/label
                directory.mkdir(parents=True,exist_ok=False)
                target = directory/'target'
                env['CARGO_TARGET_DIR']=str(target)
                command = ['cargo','--config',case['config'], 'check' if mode=='check' else 'build',
                           '--frozen','--lib','-p',workload,'--timings']
                if mode=='release': command.append('--release')
                rss = directory/'time.txt'
                start = time.perf_counter()
                with (directory/'cargo.log').open('w') as log:
                    process = subprocess.run(['/usr/bin/time','-v','-o',str(rss),*command],cwd=case['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
                elapsed = time.perf_counter()-start
                maximum = next(line.split(':',1)[1].strip() for line in rss.read_text().splitlines() if 'Maximum resident set size' in line)
                record = {'workload':workload,'variant':variant,'mode':mode,'repetition':repetition,
                          'seconds':elapsed,'max_rss_kib':int(maximum),'exit_code':process.returncode,
                          'command':command,'directory':str(directory)}
                (directory/'result.json').write_text(json.dumps(record,indent=2)+'\n')
                with (out/'results.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
                if process.returncode:raise SystemExit(process.returncode)



def report(out, destination):
    plan = json.loads((out/'plan.json').read_text())
    rows = [json.loads(line) for line in (out/'results.jsonl').read_text().splitlines()]
    summary = []
    records = []
    columns = ('workload','variant','mode','repetition','seconds','max_rss_kib','exit_code')
    for row in rows:
        timing = (Path(row['directory'])/'target/cargo-timings/cargo-timing.html').read_text()
        units = json.JSONDecoder().raw_decode(timing.split('const UNIT_DATA = ', 1)[1])[0]
        phases = {unit['name']:unit['duration'] for unit in units
                  if unit['name'].startswith(('archmage', 'magetypes', 'zenav1-svt', 'rav1d-safe'))
                  and 'build-script' not in unit['target']}
        records.append([row[k] for k in columns] + [phases])
    for mode in ('check','release'):
        for workload in WORKLOADS:
            groups = {v:[r['seconds'] for r in rows if (r['workload'],r['mode'],r['variant'])==(workload,mode,v) and r['exit_code']==0] for v in plan.get('variants', VERSIONS)}
            if any(len(g)!=3 for g in groups.values()):
                raise ValueError(f'Expected three successful repetitions per variant: {mode} {workload}')
            medians = {v:statistics.median(g) for v,g in groups.items()}
            summary.append({'workload':workload,'mode':mode,'medians':medians,
                            'range_seconds':{v:[min(g),max(g)] for v,g in groups.items()},
                            'candidate_vs_baseline_percent':100*(medians['candidate']/medians['baseline']-1)})
    result = {k:plan[k] for k in ('sources','stacks','environment','normalization')}
    result['sources'] = {name: value for name, value in result['sources'].items()
                         if name in WORKLOADS and name != 'magetypes'}
    result['commands'] = {'check':'cargo --config CASE_CONFIG check --frozen --lib -p PACKAGE --timings',
                          'release':'cargo --config CASE_CONFIG build --frozen --lib -p PACKAGE --timings --release'}
    result['logs'] = str(out)
    result['summary'] = summary
    result['columns'] = list(columns) + ['cargo_unit_seconds']
    result['runs'] = records
    result['run_heavy'] = {mode: next(line for line in (out.parent/(mode+'.log')).read_text().splitlines() if line.startswith('run-heavy: done')) for mode in ('check','release')}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    for row in summary:
        print(row)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    sub=p.add_subparsers(dest='mode',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--sources',type=Path,required=True)
    prep.add_argument('--candidate',required=True)
    prep.add_argument('--baseline',required=True)
    reporting=sub.add_parser('report')
    reporting.add_argument('--destination',type=Path,required=True)
    for mode in ('check','release'):
        s=sub.add_parser(mode)
        s.add_argument('--repetitions',type=int,default=3)
        s.add_argument('--workloads',nargs='+',choices=WORKLOADS,default=WORKLOADS)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.out.resolve(),a.sources.resolve(),a.candidate,a.baseline)
    elif a.mode=='report':report(a.out.resolve(),a.destination)
    else:run(a.out.resolve(),a.mode,a.repetitions,a.workloads)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Strict inventory, clean replacement, bit-gate and immutable snapshot audit."""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import subprocess
import tarfile

REPO=Path(__file__).resolve().parents[2]
ROOT=Path(sys.argv[1]).resolve()
SIZES=[1,16,1024,65536]
POLICIES=['correct','midp-full-fused','lowp-full-fused','midp-full-separate','lowp-full-separate','midp-pre-fused','lowp-pre-fused','midp-pre-separate','lowp-pre-separate']

def load(path):
    data=json.loads((ROOT/path).read_text())
    assert data['unreliable'] is False and data['incomplete'] is False,path
    groups={}
    for g in data['comparisons']:
        name=g['group_name'];assert name not in groups,(path,'duplicate group',name)
        names=[b['name']for b in g['benchmarks']];assert len(set(names))==len(names),(path,name,'duplicate case')
        assert len(g['samples'])>=30 and all(s['iterations']>0 for s in g['samples']),(path,name)
        groups[name]=g
    return groups

def dirty(groups):
    return {n for n,g in groups.items()if any(s['gate_clean']is not True for s in g['samples'])}

def replace(groups,path):
    new=load(path);assert set(new)==dirty(groups),(path,'replacement coverage',set(new),dirty(groups))
    for n,g in new.items():
        assert {b['name']for b in g['benchmarks']}=={b['name']for b in groups[n]['benchmarks']}
        groups[n]=g
    return groups

def cases(tier,special=False):
    p=POLICIES[:5]if special else POLICIES
    result={f'{prefix}/{policy}'for prefix in ['reference','old-scalar','current',f'old-packed-{tier}',f'packed-{tier}']for policy in p}
    result|={f'direct-{tier}/correct',f'old-direct-{tier}/correct',f'std-context-{tier}/control','old-ordinary/correct','ordinary/correct','std/control'}
    return result

pow_names={f'{d}{suffix}/{mode}/stream{n}'for d,variants,modes in [('positive',['','@historical'],['per-value']),('gamma-decode',['','@log-bits','@boundary'],['per-value','uniform']),('gamma-encode',['','@log-bits','@boundary'],['per-value','uniform']),('special',[''],['per-value'])]for suffix in variants for mode in modes for n in SIZES}
native=load('sources/native/results-native-v1/pow.json');assert set(native)==pow_names
native=replace(native,'sources/native/results-native-repeat-v1/pow.json');assert not dirty(native)
for name,g in native.items():assert {b['name']for b in g['benchmarks']}==cases('v3',name.startswith('special/')),name
transfer_names={f'transfer/{d}/{dist}/{comp}/stream{n}'for d in ['gamma-decode','gamma-encode']for dist in ['uniform','log-bits','boundary']for comp in ['separate','fused']for n in SIZES}
transfer=load('sources/native/results-native-v1/transfer.json');assert set(transfer)==transfer_names
transfer=replace(transfer,'sources/native/results-native-repeat-v1/transfer.json')
transfer=replace(transfer,'sources/native/timing-transfer-repeat-v2.json');assert not dirty(transfer)
for g in transfer.values():assert {b['name']for b in g['benchmarks']}==cases('v3')|{'linear-srgb-0.6.12/scalar-rational','linear-srgb-0.6.12/slice-rational'}
feedback=load('sources/native/results-native-v1/feedback.json');assert set(feedback)=={f'feedback/{d}/chains{n}'for d in ['gamma-decode','gamma-encode']for n in [1,8]};assert not dirty(feedback)
for g in feedback.values():assert {b['name']for b in g['benchmarks']}==cases('v3')
wasm=load('sources/wasm/timing-wasm-v3.json');assert set(wasm)==pow_names and not dirty(wasm)
for name,g in wasm.items():assert {b['name']for b in g['benchmarks']}==cases('wasm128',name.startswith('special/'))
assert sum(len(x['benchmarks'])for x in list(native.values())+list(transfer.values())+list(feedback.values()))==5728
print('PASS original native112/5728 and WASM60/2980 exact inventories; all effective groups clean')

gamma_names={f'{mode}/{kind}/{dist}/{n}'for mode in ['dynamic','fixed','C0']for kind in ['decode','encode']for dist in ['bits','values']for n in [64,4096]}|{f'dynamic/{d}/{n}'for d in ['general','historical']for n in [16,65536]}
common={'std','before/ordinary','current/ordinary','before/policy-correct','current/policy-correct'}
extra={'before/context','current/context','before/packed4','current/packed4','std/context','before/slice-context','current/slice-context','std/slice-context'}
for platform,path,replacement in [('native','sources/native/timing-gamma-native-v6.json','sources/native/timing-gamma-native-repeat-v6.json'),('NEON','sources/arm-timing/timing-gamma-arm-v6.json',None)]:
    groups=load(path);assert set(groups)==gamma_names
    if replacement:groups=replace(groups,replacement)
    assert not dirty(groups),(platform,dirty(groups))
    for name,g in groups.items():assert {b['name']for b in g['benchmarks']}==common|(extra if name.startswith('dynamic/')else set()),name
    assert sum(len(g['benchmarks'])for g in groups.values())==236
final_wasm=load('sources/native/timing-gamma-wasm-r7900x-v5.json')
final_wasm=replace(final_wasm,'sources/native/timing-gamma-wasm-r7900x-repeat-v5.json')
assert set(final_wasm)=={'positive/per-value/stream16','gamma-decode/uniform/stream1024','gamma-encode/uniform/stream1024'}and not dirty(final_wasm)
for g in final_wasm.values():assert {b['name']for b in g['benchmarks']}==cases('wasm128')
print('PASS selected gamma native28/236, NEON28/236, targetedWASM3/153; exact dirty replacement and ordinary inventory')

reference_chunks={};reference_aggregate={};cells=0;comparisons=0
for folder,paths in [('sources/native/gamma-full-native-v4',['scalar-array4','v3-array4','v3-ordinary','v4x-array8']),('sources/arm-build/gamma-full-arm-v4',['neon-array4','neon-array8','neon-ordinary']),('sources/wasm/gamma-full-wasm-v4',['wasm128-array4','wasm128-array8','wasm128-ordinary']),('sources/wasm/gamma-full-nosimd-v4',['scalar-ordinary'])]:
    files=sorted((ROOT/folder).glob('*.tsv'))
    assert {p.name for p in files}=={f'{p}-{k}.tsv'for p in paths for k in [0,1]},(folder,files)
    for path in files:
        kind=int(path.stem.rsplit('-',1)[1]);start=[0x3db71074,0x3b475041][kind];count=0;chunks=[];complete=None
        for line in path.read_text().splitlines():
            if line.startswith('#'):continue
            fields=line.split('\t')
            if fields[0]=='COMPLETE':assert complete is None;complete=fields;continue
            assert complete is None and len(fields)==4
            index,offset,n=map(int,fields[:3]);assert index==len(chunks)and offset==start+count and n==min(1<<20,0x3f800001-start-count)
            assert re.fullmatch('[0-9a-f]{32}',fields[3]);chunks.append(fields);count+=n
        assert complete is not None and int(complete[1])==count==0x3f800001-start
        expected_hash=['1ac8deaf0e5dfa2d0d75a859383ce568','1c229cfcfabcb6d54aa44974481b4db6'][kind]
        assert int(complete[2],16)==int(expected_hash,16)
        if kind in reference_chunks:assert chunks==reference_chunks[kind]and complete[3]==reference_aggregate[kind]
        else:reference_chunks[kind]=chunks;reference_aggregate[kind]=complete[3]
        comparisons+=count;cells+=1
assert cells==22 and comparisons==1108465743
print('PASS22 full admitted-domain cells /1,108,465,743 direct comparisons, every fixed chunk and raw/ordered digest crosspath')

snapshots=0;binaries=0
for identity in ROOT.rglob('identity.json'):
    meta=json.loads(identity.read_text())
    if 'binaries'not in meta:continue
    folder=identity.parent
    manifest=(folder/'source.sha256').read_bytes();assert hashlib.sha256(manifest).hexdigest()==meta['source_manifest_sha256'],identity
    with tarfile.open(folder/'source.tar.gz')as tar:
        for line in manifest.decode().splitlines():
            digest,name=line.split('  ',1);assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==digest,(identity,name)
    for row in meta['binaries']:
        data=(folder/row['archived_as']).read_bytes();assert len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'],(identity,row)
        binaries+=1
    snapshots+=1
# Final production sources are byte-exact to the sources used by selected timings.
for relative in ['sources/native/frozen-final-gamma-native-v6','sources/arm-build/frozen-final-gamma-arm-v6','sources/wasm/frozen-final-gamma-wasm-timing-v4','sources/wasm/frozen-final-gamma-nosimd-v4','sources/wasm/frozen-family-final-v4']:
    with tarfile.open(ROOT/relative/'source.tar.gz')as tar:
        files=[m for m in tar.getmembers()if m.name.startswith(('zenmath/src/','magemath/src/'))and m.name.endswith('.rs')]
        assert {m.name for m in files}=={str(p.relative_to(REPO))for folder in ['zenmath/src','magemath/src']for p in (REPO/folder).rglob('*.rs')}
        assert len(files)==91
        for m in files:assert hashlib.sha256(tar.extractfile(m).read()).digest()==hashlib.sha256((REPO/m.name).read_bytes()).digest(),(relative,m.name)
print(f'PASS {snapshots} immutable source snapshots /{binaries} binaries; selected sources equal all91 final production Rust files')

for relative in ['sources/native/feature-matrix-final-native-v1','sources/arm-build/feature-matrix-final-arm-v1']:
    data=json.loads((ROOT/relative/'inventory.json').read_text());assert len(data['positive'])==48 and len(data['negative'])==3
cost=ROOT/'sources/wasm/family-build-cost-final-v4'
rows=list(csv.DictReader((cost/'build-cost.csv').open()));assert len(rows)==54
assert len({(r['configuration'],r['replicate'])for r in rows})==54
for row in rows:
    file=cost/'binaries'/f"{row['configuration']}-{row['replicate']}-{row['binary_sha256'][:16]}"
    assert file.stat().st_size==int(row['binary_bytes'])and hashlib.sha256(file.read_bytes()).hexdigest()==row['binary_sha256']
assert len(json.loads((cost/'overlap.json').read_text()))==9
bits=json.loads((REPO/'zenmath/tools/gamma-bits.json').read_text())
assert len(bits)==902 and bits==re.findall(r'f64::from_bits\(0x([0-9a-f]{16})\)',(REPO/'zenmath/src/gamma_tables.rs').read_text())
print('PASS two48-consumer feature matrices,54 cold builds/9 Cargo timelines,902 literal bits')
# Preservation regression: an existing immutable snapshot must fail before writes.
existing=ROOT/'sources/native/frozen-final-gamma-native-v6'
names=['identity.json','source.sha256','source.tar.gz']
before={n:hashlib.sha256((existing/n).read_bytes()).digest()for n in names}
refusal=subprocess.run([sys.executable,str(REPO/'benchmarks/zenmath_pow_opt_2026-11-06/freeze.py'),str(existing)],cwd=REPO,capture_output=True,text=True)
assert refusal.returncode!=0 and 'FileExistsError' in refusal.stderr
assert before=={n:hashlib.sha256((existing/n).read_bytes()).digest()for n in names}
print('PASS existing snapshot refusal; original identity, manifest and source bytes unchanged')
print('PASS final audit; no push or release')

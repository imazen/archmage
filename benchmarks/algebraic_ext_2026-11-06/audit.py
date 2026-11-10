#!/usr/bin/env python3
"""Check retained matrix, input identity, build identities and report coverage."""
from pathlib import Path
import csv, hashlib, json, struct, sys
repo = Path(__file__).resolve().parents[2]
root = Path(sys.argv[1])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
coverage = []
for target, prefix, tiers in [('native', 'timing-', {'scalar','v3','v4x'}), ('arm','timing-',{'scalar','neon'}), ('wasm','timing-wasm-',{'scalar','wasm128'})]:
    obs = root / target / 'observations'
    expected_rows = []
    for mode in ['off','thin']:
        d = json.loads((root / target / (prefix+mode+'.json')).read_text())
        assert not d['incomplete'] and not d['unreliable']
        groups = d['comparisons']
        assert {g['group_name'] for g in groups} == {op+'/'+ty for op in ['poly','dot','div','rem'] for ty in ['f32','f64']}
        for g in groups:
            names = {b['name'] for b in g['benchmarks']}
            forms = {'ordinary','index','token','map','auto'}
            if g['group_name'].startswith(('poly','dot')): forms.add('fused')
            if g['group_name'].startswith('dot'): forms.add('index_ordinary_reduce')
            assert names == {f+'/'+t for f in forms for t in tiers}
            assert min(b['summary']['n'] for b in g['benchmarks']) >= 30
        folder = obs / ('outputs-' + ('wasm-' if target=='wasm' else '') + mode)
        inventory = list(csv.reader((folder/'inventory.tsv').open(),delimiter='\t'))
        expected = {(g['group_name'],band,b['name']) for g in groups for b in g['benchmarks'] for band in ['timed','finite_mixed']}
        assert {(g,band,n) for g,band,n,ty,file in inventory} == expected
        inputs = {}
        for group,band,name,ty,file in inventory:
            # Compare raw input bits, including any signed zeros, before the oracle's group cache is valid.
            data = (folder/file).read_bytes(); size = 4 if ty=='f32' else 8
            assert len(data) == 1024*size*3
            pairs = b''.join(data[i:i+2*size] for i in range(0,len(data),size*3))
            key = (group,band,ty)
            assert inputs.setdefault(key,pairs) == pairs
        expected_rows += inventory
        coverage.append(dict(target=target,lto=mode,cells=sum(len(g['benchmarks']) for g in groups),min_rounds=min(b['summary']['n'] for g in groups for b in g['benchmarks'])))
    acc = list(csv.DictReader((obs/'accuracy.csv').open()))
    assert len(acc) == len(expected_rows)
    assert sum(int(x['nonfinite']) for x in acc) == 0
    ident = json.loads((root/target/'frozen-source'/'identity.json').read_text())
    for b in ident['binaries']:
        p = Path(b['path']); name = p.name
        if target=='native': q=root/target/'bin'/('off' if 'target-off' in b['path'] else 'thin')/name
        elif target=='arm': q=root/target/'bin'/(name+'-'+('off' if 'target-off' in b['path'] else 'thin'))
        else: q=root/target/'bin'/name
        assert sha(q)==b['sha256'],q
    manifest = root/target/'frozen-source'/'source.sha256'
    assert sha(manifest)==ident['source_manifest_sha256']
    # Kernel source identity must survive later report/helper edits.
    for line in manifest.read_text().splitlines():
        h, p = line.split('  ',1)
        if '/consumer/src/' in p or '/ext/src/' in p:
            local = repo / p
            assert sha(local)==h, (target,p)
assert sum(c['cells'] for c in coverage)==644
print(json.dumps(dict(coverage=coverage,cells=644,accuracy_case_band_mode_rows=1288,input_bits_equal_within_oracle_groups=True,frozen_binary_hashes_verified=True,kernel_sources_unchanged=True),indent=2))

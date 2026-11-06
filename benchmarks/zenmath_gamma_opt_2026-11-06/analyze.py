#!/usr/bin/env python3
"""Validate exact timing inventory/gates and preserve every paired comparison."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('input',type=Path)
    parser.add_argument('--mode',choices=['coarse','refine','piecewise','vector'],required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    d=json.loads(args.input.read_text())
    if d.get('incomplete',False) or d['unreliable']:raise ValueError('unreliable timing')
    degrees=[8,12,16,20] if args.mode=='coarse' else list(range(11,18))
    names={'std','std_v3','core_reference','core_v3','midp_full_approx','midp_precondition_approx','lowp_full_approx'}
    names.update(f'cert_{layout}{degree}{context}' for degree in degrees for layout in ['h','e'] for context in ['','_v3'])
    if args.mode in ['piecewise','vector']:
        names={n for n in names if not n.startswith('cert_')}
        names.update(f'piece_{l}{n}{c}' for n in [3,4,5] for l in ['h','e'] for c in ['','_v3'])
    if args.mode=='vector':
        names={n for n in names if not n.startswith(('piece_h','piece_e3'))}
        names.update(f'array{w}_piece_e{n}' for w in [4,8] for n in [4,5])
    expected={f'{kind}/{dist}/{n}' for kind in ['decode','encode'] for dist in ['values','bits'] for n in [64,4096]}
    groups=d['comparisons']
    if len(groups)!=8 or {g['group_name'] for g in groups}!=expected:raise ValueError('group inventory')
    with args.out.open('x') as output:
        writer=csv.writer(output)
        writer.writerow(['group','case','rounds','mean_ns_per_element','time_over_std','paired_diff_ci_lower_ns_per_element','paired_diff_ci_upper_ns_per_element','resolution_limited','timer_ticks_per_sample'])
        for group in groups:
            group_names=names|({f'cert_e{13 if group["group_name"].startswith("decode") else 15}{c}' for c in ['', '_v3']} if args.mode in ['piecewise','vector'] else set())
            if args.mode=='vector':group_names.update(f'array{w}_global_e{13 if group["group_name"].startswith("decode") else 15}' for w in [4,8])
            benches={b['name']:b for b in group['benchmarks']}
            if len(benches)!=len(group_names) or set(benches)!=group_names:raise ValueError('case inventory '+group['group_name'])
            if group['completed_rounds']<50:raise ValueError('insufficient rounds')
            if len(group['samples'])!=group['completed_rounds'] or any(s['gate_clean'] is not True for s in group['samples']):raise ValueError('gate/sample failure')
            n=int(group['group_name'].split('/')[-1]);base=benches['std']['summary']['mean']
            analysis={b:a for a0,b,a in group['analyses'] if a0=='std'}
            for name,b in benches.items():
                a=analysis.get(name)
                writer.writerow([group['group_name'],name,group['completed_rounds'],b['summary']['mean']/n,b['summary']['mean']/base,
                                 a['ci_lower']/n if a else 0,a['ci_upper']/n if a else 0,a['resolution_limited'] if a else False,b['timer_ticks_per_sample']])
            best=min((b for name,b in benches.items() if name.startswith(('cert_','piece_','array'))),key=lambda b:b['summary']['mean'])
            print(group['group_name'],best['name'],f"{best['summary']['mean']/base:.6f}x std",f"{best['summary']['mean']/n:.6f} ns/element")
    print(f"PASS {len(groups)} groups, {sum(len(g["benchmarks"]) for g in groups)} cells, >=50 rounds, all pre-round gates clean; unreliable=false")

if __name__=='__main__':main()

#!/usr/bin/env python3
"""All retained paired-round summaries; never discard flagged rounds or outliers.

Process confirmation remains a separate zenbench verdict. This diagnostic CI
resamples paired rounds within every independent process, preserving equal
process weight; it does not substitute for confirmation across processes.
"""
import gzip,json,random,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for name in ['zen5/v3','zen5/v4','zen5/scalar','i265/v3','mac/neon']:
    p=root/(name+'.json.gz')
    if not p.exists(): raise FileNotFoundError(p)
    data=json.loads(gzip.decompress(p.read_bytes()))
    reps=data.get('replicates',[data])
    for gi,group in enumerate(reps[0]['comparisons']):
        rounds=[r['comparisons'][gi]['samples'] for r in reps]
        assert all(len(s)==50 for s in rounds)
        assert all(r['comparisons'][gi]['group_name']==group['group_name'] for r in reps)
        names=[b['name'] for b in group['benchmarks']]
        by_process=[[[s['compensated_ns'][i]/s['iterations'] for s in samples] for i in range(5)] for samples in rounds]
        means=[statistics.mean(statistics.mean(t[i]) for t in by_process) for i in range(5)]
        clean=sum(s.get('gate_clean') is True for samples in rounds for s in samples)
        cis={};rng=random.Random(0x531324)
        # Shared paired bootstrap indexes for all arms, stratified by process.
        ratios={i:[] for i in range(1,5)}
        parallel_vs_serial=[]
        for _ in range(10000):
            ix=[[rng.randrange(50) for _ in range(50)] for _ in reps]
            totals=[sum(sum(t[i][k] for k in idx) for t,idx in zip(by_process,ix)) for i in range(5)]
            for i in ratios:ratios[i].append(totals[i]/totals[0])
            parallel_vs_serial.append(totals[1]/totals[2])
        for i,rs in ratios.items():
            rs.sort();cis[names[i]]=[rs[249],rs[9749]]
        parallel_vs_serial.sort()
        qualified=(clean==50*len(reps) and name!='mac/neon')
        row={'source':name,'group':group['group_name'],'clean':clean,'retained':50*len(reps),'suite_unreliable':data['unreliable'],'processes':len(reps),'all_rounds_clean':qualified,'ns':dict(zip(names,means)),'ratio_ci95_vs_magetypes':cis,'parallel_vs_serial_ratio':means[1]/means[2],'parallel_vs_serial_ci95':[parallel_vs_serial[249],parallel_vs_serial[9749]]}
        rows.append(row)
        print(name,group['group_name'],f'{clean}/{50*len(reps)} clean',', '.join(f'{n}={v:.6f}ns' for n,v in zip(names,means)),flush=True)
output=root/'qualification-all-rounds.json';assert not output.exists(),output
output.write_text(json.dumps({'method':'10,000 paired within-process stratified bootstrap ratio of arithmetic means, seed 0x531324; all rounds retained; no outlier removal; separate from independent-process zenbench confirmation','rows':rows},indent=2)+'\n')

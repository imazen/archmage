#!/usr/bin/env python3
"""Paired all-round CIs for fused/unfused graphs; never remove flagged samples."""
from pathlib import Path
import json,random,statistics,sys
p=Path(sys.argv[1]);data=json.loads(p.read_text());out=[]
for group in data['comparisons']:
    samples=group['samples'];assert len(samples)==50
    names=[x['name'] for x in group['benchmarks']]
    times=[[s['compensated_ns'][i]/s['iterations'] for s in samples] for i in range(len(names))]
    clean=sum(s.get('gate_clean') is True for s in samples)
    for n,t in zip(names,times):print(f"{group['group_name']} | {n} | {statistics.mean(t):.6f} ns | clean {clean}/50")
    for precision in ['LOWP','MIDP']:
        i=names.index(f'portable {precision} fused')
        for reference in [f'portable {precision} unfused',names[4]]:
            j=names.index(reference);rng=random.Random(0x531324);ratios=[]
            for _ in range(10000):
                ix=[rng.randrange(50) for _ in range(50)]
                ratios.append(sum(times[i][k] for k in ix)/sum(times[j][k] for k in ix))
            ratios.sort();row={'group':group['group_name'],'implementation':names[i],'reference':reference,'ns':statistics.mean(times[i]),'reference_ns':statistics.mean(times[j]),'ratio':statistics.mean(times[i])/statistics.mean(times[j]),'ci95':[ratios[249],ratios[9749]],'clean':clean,'retained':50,'outlier_removal':False}
            out.append(row);print(json.dumps(row))
path=p.with_name('fusion-pairs.json');assert not path.exists();path.write_text(json.dumps({'source':p.name,'unreliable':data['unreliable'],'rows':out},indent=2)+'\n')

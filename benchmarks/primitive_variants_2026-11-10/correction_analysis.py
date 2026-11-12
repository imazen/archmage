#!/usr/bin/env python3
"""All retained paired ratios: proposed correction vs current and legacy graphs."""
from pathlib import Path
import json,random,statistics,sys
p=Path(sys.argv[1]);data=json.loads(p.read_text());rows=[]
for group in data['comparisons']:
    samples=group['samples'];assert len(samples)==50
    names=[v['name'] for v in group['benchmarks']]
    times=[[v['compensated_ns'][i]/v['iterations'] for v in samples] for i in range(4)]
    clean=sum(v.get('gate_clean') is True for v in samples)
    for name,t in zip(names,times): print(f"{group['group_name']} | {name} | {statistics.mean(t):.9f} ns | clean {clean}/50")
    for j in [0,2]:
        rng=random.Random(0x531324);ratios=[]
        for _ in range(10000):
            ix=[rng.randrange(50) for _ in range(50)]
            ratios.append(sum(times[1][k] for k in ix)/sum(times[j][k] for k in ix))
        ratios.sort()
        row={'group':group['group_name'],'implementation':names[1],'reference':names[j],'ns':statistics.mean(times[1]),'reference_ns':statistics.mean(times[j]),'ratio':statistics.mean(times[1])/statistics.mean(times[j]),'ci95':[ratios[249],ratios[9749]],'clean':clean,'retained':50,'outlier_removal':False}
        rows.append(row);print(json.dumps(row))
out=p.with_name('correction-pairs.json');assert not out.exists();out.write_text(json.dumps({'source':p.name,'unreliable':data['unreliable'],'rows':rows},indent=2)+'\n')

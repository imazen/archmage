#!/usr/bin/env python3
"""All-retained-round paired bootstrap of mean time ratios, with no IQR removal."""
from pathlib import Path
import json,random,statistics,sys
source=Path(sys.argv[1]);r=json.loads(source.read_text());rows=[]
for c in r['comparisons']:
    samples=c['samples'];assert len(samples)==50
    times=[[s['compensated_ns'][i]/s['iterations'] for s in samples] for i in range(len(c['benchmarks']))]
    clean=sum(s.get('gate_clean') is True for s in samples)
    for i,b in enumerate(c['benchmarks']):
        mean=statistics.mean(times[i]);ratio=mean/statistics.mean(times[0]);rng=random.Random(0x531324)
        ratios=[]
        if i:
            for _ in range(10000):
                indices=[rng.randrange(50) for _ in range(50)]
                ratios.append(sum(times[i][j] for j in indices)/sum(times[0][j] for j in indices))
            ratios.sort();ci=[ratios[249],ratios[9749]]
        else:ci=[1.,1.]
        rows.append({'group':c['group_name'],'implementation':b['name'],'ns':mean,'ratio':ratio,'ci95':ci,'clean_rounds':clean,'retained_rounds':50,'bootstrap_pairs':10000,'outlier_removal':False})
        print(f"{c['group_name']} | {b['name']} | {mean:.3f} ns | R={ratio:.4f} [{ci[0]:.4f},{ci[1]:.4f}] | clean {clean}/50")
output=source.with_name('all-rounds.json');assert not output.exists();output.write_text(json.dumps({'source':source.name,'unreliable':r['unreliable'],'rows':rows},indent=2)+'\n')

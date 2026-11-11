#!/usr/bin/env python3
"""Independent-process diagnostic intervals, without clearing admission failures.

Match zenbench's paired log-mean Student-t model, Bonferroni family adjustment
and timer-resolution floor. Five processes imply df=4; its CDF has an exact
closed form: F(t)=1/2+3s/4-s^3/4, s=t/sqrt(t^2+4). These are diagnostics on
retained data, NOT zenbench confirmation verdicts for failed suites.
"""
from pathlib import Path
import gzip,json,math,statistics,sys
root=Path(sys.argv[1]); source=root/'qualification-all-rounds.json'
result=json.loads(source.read_text())
def critical(prob):
    lo,hi=0.,1.
    for _ in range(80):
        s=(lo+hi)/2
        if .5+.75*s-.25*s*s*s<prob:lo=s
        else:hi=s
    s=(lo+hi)/2
    return 2*s/math.sqrt(1-s*s)
assert abs(critical(.975)-2.7764451051977987)<1e-10
cache={}
for row in result['rows']:
    name=row['source']
    if name not in cache:cache[name]=json.loads(gzip.decompress((root/(name+'.json.gz')).read_bytes()))
    data=cache[name]; reps=data.get('replicates',[])
    if len(reps)!=5:
        row['process_family95_ratio_ci_vs_magetypes']=None
        continue
    groups=[[g for g in r['comparisons'] if g['group_name']==row['group']][0] for r in reps]
    family=sum(len(g['analyses']) for g in reps[0]['comparisons'])
    t=critical(1-.05/(2*family)); intervals={}
    for i,impl in enumerate(groups[0]['benchmarks']):
        if i==0:continue
        values=[];floor=0.
        for r,g in zip(reps,groups):
            m=[statistics.mean(s['compensated_ns'][j]/s['iterations'] for s in g['samples']) for j in [0,i]]
            values.append(math.log(m[1]/m[0]))
            floor=max(floor,2*r['timer_resolution_ns']/min(s['iterations'] for s in g['samples'])/g['benchmarks'][i]['summary']['mean'])
        mu=statistics.mean(values); half=max(t*statistics.stdev(values)/math.sqrt(5),floor)
        intervals[impl['name']]=[math.exp(mu-half),math.exp(mu+half)]
    row['process_family95_ratio_ci_vs_magetypes']=intervals
    row['process_interval_method']='5 paired log process means; Student-t df4; 95% Bonferroni family '+str(family)+'; timer floor; diagnostic only, admission retained'
output=root/'qualification-with-process-ci.json';assert not output.exists(),output
output.write_text(json.dumps(result,indent=2)+'\n')
for r in result['rows']:
    if r['all_rounds_clean']: print(r['source'],r['group'],r['process_family95_ratio_ci_vs_magetypes'])

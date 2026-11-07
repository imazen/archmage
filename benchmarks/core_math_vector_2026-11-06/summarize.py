#!/usr/bin/env python3
"""Summarize zenbench pairs; raw times and CI are per 1024-element call."""
import csv,json,sys
w=csv.writer(sys.stdout)
w.writerow(['file','group','scalar_ns_per_value','vector_ns_per_value','scalar_over_vector','paired_delta_low_ns_per_value','paired_delta_high_ns_per_value'])
for path in sys.argv[1:]:
    for g in json.load(open(path))['comparisons']:
        a,b=g['benchmarks']; c=g['analyses'][0][2]
        x,y=a['summary']['mean'],b['summary']['mean']
        w.writerow([path.split('/')[-1],g['group_name'],f'{x/1024:.4f}',f'{y/1024:.4f}',f'{x/y:.3f}',f'{c["ci_lower"]/1024:.4f}',f'{c["ci_upper"]/1024:.4f}'])

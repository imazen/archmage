#!/usr/bin/env python3
"""Summarize same-input comparisons without conflating accuracy contracts."""
import csv,json,sys
w=csv.writer(sys.stdout)
w.writerow(['file','group','implementation','ns_per_value','mad_ns_per_value','rounds'])
for p in sys.argv[1:]:
    for g in json.load(open(p))['comparisons']:
        for b in g['benchmarks']:
            s=b['summary']
            w.writerow([p.split('/')[-1],g['group_name'],b['name'],f'{s["mean"]/1024:.4f}',f'{s["mad"]/1024:.4f}',s['n']])

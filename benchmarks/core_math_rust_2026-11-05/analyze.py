#!/usr/bin/env python3
"""Tabulate zenbench stream means; each iteration contains 1024 values."""
import csv
import json
import sys

writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow(['run', 'function', 'implementation', 'rounds', 'ns_per_value', 'mad_ns_per_value'])
for path in sys.argv[1:]:
    with open(path) as source:
        result = json.load(source)
    if result['unreliable']:
        raise SystemExit(f'{path}: zenbench marked run unreliable')
    if len(result['comparisons']) != 33:
        raise SystemExit(f'{path}: expected 33 function groups')
    for group in result['comparisons']:
        for bench in group['benchmarks']:
            stats = bench['summary']
            writer.writerow([path.rsplit('/', 1)[-1], group['group_name'], bench['name'], stats['n'],
                             f"{stats['mean']/1024:.6f}", f"{stats['mad']/1024:.6f}"])

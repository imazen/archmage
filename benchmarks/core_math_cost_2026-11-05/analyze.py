#!/usr/bin/env python3
"""Feed external-worker timings to zenbench's own paired analysis implementation."""
import csv
import json
from pathlib import Path
import subprocess
import sys

source, analyzer = map(Path, sys.argv[1:])
rows = {}
for row in csv.DictReader(source.open()):
    if row['kind'] == 'ns_per_element':
        rows.setdefault((row['op'], row['variant']), []).append(float(row['value']))
baseline = 'rust-std' if source.name.startswith('rust-') else './gcc-x86-64.so'
keys = list(rows)
packets = [dict(baseline=rows[op, baseline], candidate=rows[op, variant],
                iterations=[1] * len(rows[op, variant]),
                timer_resolution_ns=1 / (4096 * 128)) for op, variant in keys]
result = subprocess.run([str(analyzer)], input=json.dumps(packets), text=True,
                        capture_output=True, check=True)
analyses = json.loads(result.stdout)
writer = csv.writer(sys.stdout)
writer.writerow(['op', 'variant', 'ns_per_element', 'pct_vs_baseline',
                 'ci_lower_ns', 'ci_upper_ns', 'significant'])
for (op, variant), analysis in zip(keys, analyses):
    writer.writerow([op, variant, analysis['candidate']['mean'], analysis['pct_change'],
                     analysis['ci_lower'], analysis['ci_upper'], analysis['significant']])

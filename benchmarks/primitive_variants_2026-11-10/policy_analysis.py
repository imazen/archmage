#!/usr/bin/env python3
"""Analyze every retained policy pair, without filtering admission failures.

One packet/context at a time. NumPy is needed only for analysis, not timing.
The process-family interval is a diagnostic, not a replacement for zenbench's
own confirmation result or resource gates.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import re

import numpy as np


def t4_critical(probability):
    low, high = 0.0, 1.0
    for _ in range(80):
        s = (low + high) / 2
        if 0.5 + 0.75 * s - 0.25 * s**3 < probability:
            low = s
        else:
            high = s
    s = (low + high) / 2
    return 2 * s / math.sqrt(1 - s * s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('packet', type=Path)
    p.add_argument('log', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--context', required=True)
    p.add_argument('--exploratory', action='store_true')
    p.add_argument('--wasm', action='store_true')
    a = p.parse_args()
    assert not a.output.exists(), a.output
    packet = json.loads(gzip.decompress(a.packet.read_bytes()))
    reps = packet.get('replicates') or [packet]
    assert len(reps) == (1 if a.exploratory else 5)
    assert all(len(r['comparisons']) == 66 for r in reps)
    assert sum(len(g['benchmarks']) for g in reps[0]['comparisons']) == 1566
    # Only baseline comparisons are reported: include all 1500 in the family.
    family = sum(len(g['benchmarks']) - 1 for g in reps[0]['comparisons'])
    assert family == 1500
    log = a.log.read_text()
    completions = re.findall(r'^run-heavy: done rc=(\d+) ', log, re.MULTILINE)
    assert len(completions) == 1, completions
    confirmation_passed = not a.exploratory and completions[0] == '0'
    errors = {}
    for line in log.splitlines():
        if line.startswith('POLICY_ACCURACY\t'):
            _, group, cell, error = line.split('\t')
            key = (group, cell)
            value = float(error)
            assert math.isfinite(value) and value >= 0
            if key in errors:
                assert errors[key] == value, ('preflight changed between processes', key)
            errors[key] = value
    assert len(errors) == 1566, len(errors)
    suite_valid = all(not r['unreliable'] and not r.get('incomplete', False)
                      for r in [packet, *reps])
    rows = []
    for gi, group in enumerate(reps[0]['comparisons']):
        groups = [r['comparisons'][gi] for r in reps]
        names = [b['name'] for b in group['benchmarks']]
        operation = group['group_name'].split('/')[1]
        assert names[0] == 'existing Magetypes ' + operation
        assert len(set(names)) == len(names)
        assert len(names) == (23 if operation in ('recip', 'rsqrt') else
                              21 if group['group_name'].endswith('/zero-clamp64') else 25)
        assert all(g['group_name'] == group['group_name'] and
                   [b['name'] for b in g['benchmarks']] == names for g in groups)
        assert re.fullmatch(r'(?:wasm-)?[a-z0-9]+/(?:min|max|recip|rsqrt)/f(?:32|64)x(?:2|4|8|16)/(?:stream64|stream4096|chain64|zero-clamp64)', group['group_name'])
        assert all(len(g['samples']) == g['completed_rounds'] == 50 and
                   not g['config']['auto_rounds'] and g['config']['max_rounds'] == 50
                   for g in groups)
        assert all(s['iterations'] > 0 and len(s['compensated_ns']) == len(names)
                   and len(s['elapsed_ns']) == len(names)
                   and sorted(s['execution_order']) == list(range(len(names)))
                   for g in groups for s in g['samples'])
        label = group['group_name'].rsplit('/', 1)[1]
        batch = (4096 if label in ('stream64', 'zero-clamp64') else 256) if a.wasm else 1
        values = np.array([[[s['compensated_ns'][i] / s['iterations'] / batch
                             for i in range(len(names))] for s in g['samples']] for g in groups])
        assert np.all(np.isfinite(values)) and np.all(values > 0)
        means = values.mean(axis=(0, 1))
        rng = np.random.default_rng(0x531324)
        resamples = np.zeros((10000, len(names)))
        for process in values:
            index = rng.integers(0, 50, size=(10000, 50))
            resamples += process[index].mean(axis=1) / len(reps)
        intervals = {}
        for i in range(1, len(names)):
            boot = resamples[:, i] / resamples[:, 0]
            process_ci = None
            if len(reps) == 5:
                process_means = values.mean(axis=1)
                logs = np.log(process_means[:, i] / process_means[:, 0])
                floor = max(2 * r['timer_resolution_ns'] / batch /
                            min(s['iterations'] for s in g['samples']) /
                            min(process_means[j, i], process_means[j, 0])
                            for j, (r, g) in enumerate(zip(reps, groups)))
                half = max(t4_critical(1 - 0.05 / (2 * family)) * logs.std(ddof=1) / math.sqrt(5), floor)
                process_ci = [math.exp(logs.mean() - half), math.exp(logs.mean() + half)]
            intervals[names[i]] = dict(ratio=float(means[i] / means[0]),
                                      paired_round95=np.quantile(boot, [0.025, 0.975]).tolist(),
                                      process_family95=process_ci)
        clean = sum(s.get('gate_clean') is True for g in groups for s in g['samples'])
        fresh_clean = all(s.get('gate_clean') is True and
                          s.get('gate_observation_age_ns') is not None and
                          s['gate_observation_age_ns'] <= 1_000_000_000 and
                          bool(s.get('gate_observation')) for g in groups for s in g['samples'])
        row = dict(context=a.context, group=group['group_name'], processes=len(reps),
                   retained=50 * len(reps), clean=clean, all_rounds_clean=fresh_clean,
                   suite_valid=suite_valid, confirmation_passed=confirmation_passed,
                   exploratory=a.exploratory,
                   admissible=confirmation_passed and suite_valid and fresh_clean,
                   batch=batch, ns=dict(zip(names, means.tolist())), intervals=intervals,
                   max_observed_relative_error={n: errors[(group['group_name'], n)] for n in names})
        rows.append(row)
        print(a.context, row['group'], f'{clean}/{row["retained"]} clean', flush=True)
    a.output.write_text(json.dumps(dict(context=a.context, family=family,
        method='All retained pairs, equal process weight; 10000 within-process paired bootstrap resamples, PCG64 seed 0x531324; df4 log process mean Student-t Bonferroni family95 plus timer floor; no filtering; Mac exploratory.',
        numpy=np.__version__, packet=str(a.packet.resolve()), rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    assert abs(t4_critical(0.975) - 2.7764451051977987) < 1e-10
    main()

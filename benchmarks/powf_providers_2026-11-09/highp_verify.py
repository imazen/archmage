#!/usr/bin/env python3
"""Compare actual library powf outputs to the independent exact graph oracle.

The command is a JSON argv array, permitting explicit physical/cross runners.
Each invocation evaluates broadcasts plus mixed-base vectors grouped by exponent.
NaN payload differences are logged, never counted as promised graph mismatches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('expected', type=Path)
    p.add_argument('directory', type=Path)
    p.add_argument('--command', required=True, help='JSON argv; context appended')
    p.add_argument('--contexts', nargs='+', default=['scalar', 'native', 'automatic'])
    p.add_argument('--widths', nargs='+', type=int, default=[1, 4, 8, 16])
    a = p.parse_args()
    command = json.loads(a.command)
    assert isinstance(command, list) and command and all(isinstance(s, str) for s in command)
    a.directory.mkdir(parents=True, exist_ok=False)
    rows = [tuple(int(v, 16) for v in line.split()) for line in a.expected.read_text().splitlines()]
    assert rows and all(len(row) == 3 for row in rows)
    ordered = sorted(rows, key=lambda r: (r[1], (r[0] - 0x800000) & 0xffffffff))
    report = {'command': command, 'oracle_sha256': hashlib.sha256(a.expected.read_bytes()).hexdigest(),
              'pairs': len(rows), 'cells': []}
    failures = 0
    with (a.directory/'variances.jsonl').open('w') as ledger:
        for ctx in a.contexts:
            for width in a.widths:
                batches = [([row] * width, width) for row in rows]
                batches += [(ordered[i:i+width] + [ordered[i]] * (width-len(ordered[i:i+width])),
                             len(ordered[i:i+width])) for i in range(0, len(ordered), width)]
                requests = ''.join(f'E 0 {width} ' + ' '.join(f'{x:08x} {n:08x}' for x,n,_ in b) + '\n' for b,_ in batches)
                cell = f'{ctx}-x{width}'
                proc = subprocess.run(command + [ctx], input=requests, text=True, capture_output=True)
                (a.directory/f'{cell}.actual.txt').write_text(proc.stdout)
                (a.directory/f'{cell}.stderr.txt').write_text(proc.stderr)
                assert proc.returncode == 0, (cell, proc.returncode, proc.stderr)
                actual = proc.stdout.splitlines()
                assert len(actual) == len(batches), (cell, len(actual), len(batches))
                counts = {'cell': cell, 'lanes': 0, 'bit_or_class_mismatches': 0, 'nan_payload_variances': 0}
                for line, (batch, valid) in zip(actual, batches):
                    values = [int(v,16) for v in line.split()]
                    assert len(values) == width
                    for lane, ((x,n,y), value) in enumerate(zip(batch[:valid],values[:valid])):
                        counts['lanes'] += 1
                        if value == y: continue
                        nan = lambda b: b & 0x7fffffff > 0x7f800000
                        kind = 'nan_payload_variances' if nan(value) and nan(y) else 'bit_or_class_mismatches'
                        counts[kind] += 1
                        ledger.write(json.dumps({'cell':cell, 'lane':lane, 'x':f'{x:08x}', 'n':f'{n:08x}',
                                                 'expected':f'{y:08x}', 'actual':f'{value:08x}', 'kind':kind})+'\n')
                failures += counts['bit_or_class_mismatches']
                report['cells'].append(counts)
                print(json.dumps(counts), flush=True)
                (a.directory/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    assert failures == 0, failures

if __name__ == '__main__':
    main()

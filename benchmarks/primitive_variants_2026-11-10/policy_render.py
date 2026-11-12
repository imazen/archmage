#!/usr/bin/env python3
"""Render bounded nested Rustdoc tables from retained policy analysis packets."""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LABELS = {'dev/scalar': 'Zen 5, 32 MiB L3 class / ScalarToken', 'dev/v3': 'Zen 5, 32 MiB L3 class / V3',
          'dev/v4': 'Zen 5, 32 MiB L3 class / V4', 'dev/v4x': 'Zen 5, 32 MiB L3 class / V4x',
          'i265/v3': 'Core Ultra 7 265K E-cores / V3', 'mac/neon': 'Apple M6 / NEON',
          'wasm/strict': 'Core Ultra 7 265K E-cores / strict WASM',
          'wasm/fusing': 'Core Ultra 7 265K E-cores / relaxed fusing WASM',
          'wasm/nonfusing': 'Core Ultra 7 265K E-cores / relaxed nonfusing WASM'}


def render(context, op, rows):
    selected = [r for r in rows if r['group'].split('/')[1] == op]
    assert len(selected) == (24 if op in ('min', 'max') else 9)
    exploratory = selected[0]['exploratory']
    lines = [f'<details><summary>{LABELS[context]}: {op} timings and observed error</summary>', '',
             'Each timing is **ns per complete workload; ratio (interval) versus existing Magetypes**.',
             'Streams process 64/4096 scalar elements; chain64 performs 64 dependent vector steps.',
             'Error is the largest observed relative error across this width’s fixed stream inputs and chain states; it is not a whole-domain maximum.',
             'Min/max error 0 denotes exact selected values; preflight separately verifies the requested zero signs.',
             ('**Exploratory**: one process, 50 pairs, unrestricted macOS scheduler. Intervals are paired-round 95%; no strict promotion.' if exploratory else
              '**Retained**: five processes × 50 pairs. Intervals are diagnostic process-family 95% (all 1,500 baseline comparisons), including the timer floor.'),
             'The point ratio divides all-retained mean ns. Strict process intervals describe the geometric mean of the five process ratios; raw analysis also retains the paired-round interval.',
             'C marks a group whose 250/250 observations are fresh and clean; D marks a flagged group. E marks exploratory. No observations are filtered.',
             'Display values are rounded; decisions use the full-precision analysis and original samples.',
             'Whole-suite confirmation: ' + ('passed.' if all(r['confirmation_passed'] for r in selected) else 'not established; read cell status separately.'), '']
    if context == 'wasm/nonfusing':
        lines += ['The nonfusing engine uses the baseline x86_64 target, which disables more features than FMA. This is not an isolated FMA-cost comparison.', '']
    for ty in dict.fromkeys(r['group'].split('/')[2] for r in selected):
        rs = [r for r in selected if r['group'].split('/')[2] == ty]
        workloads = ['stream64', 'stream4096', 'chain64'] + (['zero-clamp64'] if op in ('min', 'max') else [])
        by = {r['group'].rsplit('/', 1)[1]: r for r in rs}
        assert set(by) == set(workloads)
        ids = list(by['stream64']['ns'])
        lines += [f'<details><summary>{ty}: all canonical cells × workloads</summary>', '',
                  '| Cell | Observed error | ' + ' | '.join(workloads) + ' |',
                  '|---|---|' + '---|' * len(workloads)]
        for cell in ids:
            errors = [r['max_observed_relative_error'][cell] for r in rs if cell in r['ns']]
            entries = []
            for workload in workloads:
                r = by[workload]
                if cell not in r['ns']:
                    assert workload == 'zero-clamp64' and int(cell[1:], 16) & 3 == 2
                    entries.append('Unsupported input')
                    continue
                ns = r['ns'][cell]
                ci = r['intervals'].get(cell)
                if ci:
                    low, high = ci['paired_round95'] if exploratory else ci['process_family95']
                    entries.append(f'{ns:.3g}; {ci["ratio"]:.3g} ({low:.3g}–{high:.3g})')
                else:
                    entries.append(f'{ns:.3g}; baseline')
            label = 'Existing Magetypes ' + op if cell.startswith('existing ') else f'[{cell}](#{cell.lower()})'
            lines.append('| ' + ' | '.join([label, f'{max(errors):.3g}', *entries]) + ' |')
        lines += ['', '**Retained group status:** ' + '; '.join(
            f'{w}: {"E" if exploratory else "C" if by[w]["all_rounds_clean"] else "D"} '
            f'{by[w]["clean"]}/{by[w]["retained"]} clean' for w in workloads) + '.', '', '</details>', '']
    lines += ['</details>', '']
    text = '\n'.join(lines)
    assert len(text.encode()) <= 30000, (context, op, len(text.encode()))
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('analyses', nargs='+', type=Path)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    documents = {'minmax': [], 'recip': []}
    seen = set()
    for path in a.analyses:
        packet = json.loads(path.read_text())
        context, rows = packet['context'], packet['rows']
        assert context in LABELS and context not in seen
        seen.add(context)
        assert len(rows) == 66 and all(r['context'] == context for r in rows)
        for op in ('min', 'max', 'recip', 'rsqrt'):
            family = 'minmax' if op in ('min', 'max') else 'recip'
            filename = context.replace('/', '-') + '-' + op + '.md'
            target = REPO / 'magemath/src' / family / 'timings' / filename
            text = render(context, op, rows)
            if a.check:
                assert target.read_text() == text, target
            else:
                target.parent.mkdir(exist_ok=True)
                target.write_text(text)
            documents[family].append(filename)
            print(target.relative_to(REPO), len(text.encode()))
    for family, files in documents.items():
        path = REPO / 'magemath/src' / family / 'mod.rs'
        begin, end = '// BEGIN GENERATED TIMING DOCUMENTS', '// END GENERATED TIMING DOCUMENTS'
        block = begin + '\n' + '\n'.join(f'#![doc = include_str!("timings/{f}")]' for f in files) + '\n' + end
        source = path.read_text()
        if begin in source:
            start = source.index(begin)
            stop = source.index(end, start) + len(end)
            updated = source[:start] + block + source[stop:]
        else:
            needle = '#![doc = include_str!("permutations.md")]'
            updated = source.replace(needle, needle + '\n' + block)
        if a.check:
            assert updated == source, path
        else:
            path.write_text(updated)


if __name__ == '__main__':
    main()

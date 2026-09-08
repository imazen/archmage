#!/usr/bin/env python3
"""Collect isa_fixups stdout into the docs dataset; keep every paired round.

Run the example with --release on hardware, or --samples-only under an emulator.
Prefix its stdout with rustc -Vv and lscpu / sysctl -n machdep.cpu.brand_string.
Then: python3 xtask/isa_evidence.py CPU=run.txt ... --output dataset.json
No dependency beyond Python. NaNs and signed zeros are stored as integer bits.
"""
import argparse
import hashlib
import json
import math
import pathlib
import statistics
import struct
import subprocess


def collect(label, path):
    lines = path.read_text().splitlines()
    records = [json.loads(line) for line in lines if line.startswith('{')]
    assert records, (label, 'no records')
    samples = [r for r in records if r['kind'] == 'sample']
    assert samples
    for r in samples:
        assert len(r['input']) == len(r['baseline']) == len(r['fixed'])
        assert len(r['input']) % r['lanes'] == 0
        assert all(isinstance(x, int) and 0 <= x <= 0xffffffff
                   for key in ('input', 'baseline', 'fixed') for x in r[key])
        if r['method'] == 'to_i32_saturating()':
            for bits, actual in zip(r['input'], r['fixed']):
                value = struct.unpack('!f', struct.pack('!I', bits))[0]
                expected = 0 if math.isnan(value) else int(max(-2147483648, min(2147483647, value)))
                assert actual == expected & 0xffffffff, (label, r['isa'], hex(bits))
    timings = [r for r in records if r['kind'] == 'timing']
    for r in timings:
        pairs = r['ns_per_vector_pairs']
        assert len(pairs) == 9 and all(math.isfinite(x) and x > 0 for pair in pairs for x in pair)
        ratios = [b / a for a, b in pairs]
        median = statistics.median(ratios)
        r['median_overhead_percent'] = (median - 1) * 100
        r['ratio_mad_percent'] = statistics.median(abs(x - median) for x in ratios) * 100
        r['comparison'] = ('fixup' if r['isa'].startswith('x86-') else 'identity control')
    return dict(label=label,
                compiler=next((x for x in lines if x.startswith('rustc ')), 'not recorded'),
                target=next((x.removeprefix('target: ') for x in lines if x.startswith('target: ')),
                            next((x.removeprefix('host: ') for x in lines if x.startswith('host: ')), 'see label')),
                cpu=next((x.split(':', 1)[1].strip() for x in lines if x.startswith('Model name:')),
                         next((x for x in lines if x.startswith('Apple ')), label)),
                samples=samples, timings=timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', help='CPU=stdout.txt')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    source = pathlib.Path('magetypes/examples/isa_fixups.rs').read_bytes()
    result = dict(schema=1, library_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  probe_sha256=hashlib.sha256(source).hexdigest(),
                  protocol='2048 f32 lanes, ordinary positive inputs, 9 AB/BA rounds; ns/vector includes load/store and loop; no target-cpu override; ratios are not confidence intervals',
                  runs=[collect(label, pathlib.Path(path)) for label, path in (x.split('=', 1) for x in args.runs)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(',', ':')) + '\n')
    print(f"Validated {sum(len(r['samples']) for r in result['runs'])} sample rows and {sum(len(r['timings']) for r in result['runs'])} timing pairs")


if __name__ == '__main__':
    main()

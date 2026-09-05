#!/usr/bin/env python3
"""Compare PR #85 check/release builds; dependencies warm, magetypes clean."""
import json, os, pathlib, subprocess, sys, time
base, variant = map(pathlib.Path, sys.argv[1:3])
env = dict(os.environ, CARGO_INCREMENTAL='0', CARGO_TERM_COLOR='never')
env.pop('RUSTC_WRAPPER', None)
variants = [("baseline", base), ("arcane", variant)]
if len(sys.argv) > 4: variants.append(("main", pathlib.Path(sys.argv[4])))
rows = []
for mode in ['check', 'release']:
    cmd = ['cargo', 'check' if mode == 'check' else 'build', '--locked', '-p', 'magetypes', '--features', 'avx512']
    if mode == 'release': cmd.append('--release')
    for _, root in variants:
        subprocess.run(cmd, cwd=root, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i in range(6):
        order = variants[:] if len(variants) == 2 else variants[i % len(variants):] + variants[:i % len(variants)]
        if i % 2: order.reverse()
        for label, root in order:
            clean = ['cargo', 'clean', '-p', 'magetypes']
            if mode == 'release': clean.append('--release')
            subprocess.run(clean, cwd=root, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            start = time.perf_counter()
            proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
            elapsed = time.perf_counter() - start
            if proc.returncode: raise RuntimeError(proc.stderr)
            row = dict(mode=mode, variant=label, trial=i+1, seconds=elapsed)
            rows.append(row)
            print(json.dumps(row), flush=True)
pathlib.Path(sys.argv[3]).write_text(json.dumps(rows, indent=2)+'\n')

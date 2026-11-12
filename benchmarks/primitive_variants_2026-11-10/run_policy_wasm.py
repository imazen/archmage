#!/usr/bin/env python3
"""Run the frozen WASM policy matrix once in each verified engine mode.

Optionally wait for this study's native controller to finish first. The wait is
not benchmark admission: zenbench still owns admission and serialization.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--wrapper', type=Path, required=True)
p.add_argument('--wait-for', type=Path)
args = p.parse_args()
root = args.root.resolve()
manifest = json.loads((root / 'SOURCE.json').read_text())
for name, digest in manifest['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
claim = Path('.workongoing')
assert 'codex-portability-fma' in claim.read_text()
if args.wait_for:
    while True:
        state = json.loads(args.wait_for.read_text())
        if state['stage'] == 'complete':
            break
        assert state['stage'] != 'failed-no-packet', state
        time.sleep(30)
done = []
for mode in ('strict', 'fusing', 'nonfusing'):
    module = root / ('strict.wasm' if mode == 'strict' else 'relaxed.wasm')
    output, log = root / (mode + '.json.gz'), root / (mode + '.log')
    assert not output.exists() and not log.exists()
    env = dict(os.environ, TMPDIR=str(Path.home() / 'tmp'),
               ARCHMAGE_PRIMITIVE_RESULTS=str(output), ZENBENCH_GATE_TRACE='1',
               ARCHMAGE_WASM_MODULE=str(module), ARCHMAGE_WASM_MODE=mode,
               RUN_HEAVY_SYSTEM_SCOPE='1')
    command = [str(args.wrapper.resolve()), '--mem', '8G', '--jobs', '2', '--',
               sys.executable, '-B', str(Path(__file__).resolve().parent / 'run_local.py'),
               str(root / 'policy_wasm_host')]
    with log.open('w') as stream:
        child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            now = datetime.now(timezone.utc).isoformat()
            claim.write_text(now + ' codex-portability-fma frozen WASM policy ' + mode + '\n')
            (root / 'progress.json').write_text(json.dumps(dict(
                stage=mode, wrapper_pid=child.pid, command=command,
                last_check_utc=now, completed=done), indent=2) + '\n')
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                pass
    done.append(dict(mode=mode, returncode=code, packet_present=output.is_file()))
    print(done[-1], flush=True)
    if not output.is_file():
        (root / 'progress.json').write_text(json.dumps(dict(
            stage='failed-no-packet', completed=done), indent=2) + '\n')
        raise RuntimeError('No complete packet; inspect ' + str(log))
(root / 'progress.json').write_text(json.dumps(dict(
    stage='complete', completed=done,
    last_check_utc=datetime.now(timezone.utc).isoformat()), indent=2) + '\n')

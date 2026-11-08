"""Stream retained JSON over SSH when the worker has little free disk.

Stdout changes output transport only. Resource gates, lock, samples and
arithmetic are the same. Full worker stdout/stderr is retained for every run.
"""
import json
from pathlib import Path
import shlex
import subprocess
import sys

worker, destination, mode, filter_text = sys.argv[1:5]
out = Path(destination).resolve()
out.mkdir(parents=True)
q = shlex.quote
base = '$HOME/work/archmage-portability-fma-results'
env = ['ZENBENCH_NO_SAVE=1', 'STUDY_CONTEXTS=scalar,neon', 'STUDY_FILTER='+q(filter_text)]
if mode != 'native':
    guest = 'strict-buffer.wasm' if mode == 'strict' else 'relaxed-buffer.wasm'
    env += ['STUDY_WASM_HOST="$PWD/'+guest+'"', 'STUDY_WASM_MODE='+q(mode),
            'STUDY_WASM_EXPECT_FUSION='+('0' if mode=='strict' else '3')]
script = f'''set -eu
cd "{base}"
export TMPDIR="$HOME/tmp/archmage-portability-fma-sol"
{' '.join(env)} STUDY_OUTPUT=- ./run-heavy --mem 4G --jobs 2 -- ./stdout-study-arm-host
'''
(out/'command.sh').write_text(script)
with (out/'worker.log').open('xb') as log:
    run = subprocess.run(['ssh',worker,'bash','-s'],input=script.encode(),stdout=log,stderr=subprocess.STDOUT)
(out/'exit.json').write_text(json.dumps(dict(exit_code=run.returncode))+'\n')
if run.returncode:sys.exit(run.returncode)
packets=[line.removeprefix('STUDY_JSON ') for line in (out/'worker.log').read_text().splitlines() if line.startswith('STUDY_JSON ')]
assert len(packets)==1, 'one complete retained-round packet required'
(out/'result.json').write_text(packets[0]+'\n')
packet=json.loads(packets[0])
clean=bool(packet['comparisons']) and not packet['unreliable'] and not packet.get('incomplete',False) and all(len(g['samples'])>=50 and all(r.get('gate_clean') is True for r in g['samples']) for g in packet['comparisons'])
print(f'{mode}/{filter_text}: groups={len(packet["comparisons"])} admitted={clean}',flush=True)
if not clean:sys.exit(2)

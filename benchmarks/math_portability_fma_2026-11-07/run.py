"""Run frozen per-operation studies; retain every attempt and live process ID."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

binary=Path(sys.argv[1]).resolve()
out=Path(sys.argv[2]).resolve()
out.mkdir(parents=True,exist_ok=True)
identity=hashlib.sha256(binary.read_bytes()).hexdigest()
ops=['exp2_lowp','exp_lowp','pow_lowp','exp2_midp','exp_midp','pow_midp','sigmoid_midp','silu_midp','log2_lowp','ln_lowp','log10_lowp','log2_midp','ln_midp','log10_midp','combined_patch']
for op in ops:
    assert not (out/f'{op}.json').exists(), 'Never overwrite a study'
    env=dict(os.environ,STUDY_FILTER=op+'/',STUDY_OUTPUT=str(out/f'{op}.json'))
    with (out/f'{op}.log').open('x') as log:
        process=subprocess.Popen([str(binary)],env=env,stdout=log,stderr=subprocess.STDOUT)
        state=dict(operation=op,pid=process.pid,binary=str(binary),binary_sha256=identity,started=datetime.now(timezone.utc).isoformat(),completed_operations=ops[:ops.index(op)],required_operations=ops)
        (out/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
        result=process.wait()
        state.update(exit_code=result,finished=datetime.now(timezone.utc).isoformat())
        (out/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
        if result: sys.exit(result)
        packet=json.loads((out/f'{op}.json').read_text())
        invalid=packet['unreliable'] or packet.get('incomplete',False) or any(
            len(g['samples'])<50 or any(r.get('gate_clean') is not True for r in g['samples'])
            for g in packet['comparisons'])
        if invalid:
            state.update(stage='invalid admission; study stopped', reason='No promotion claim; retain all rounds')
            (out/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
            sys.exit(2)
(out/'COMPLETE').write_text(identity+'\n')

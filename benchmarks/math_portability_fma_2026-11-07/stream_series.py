"""Run the preregistered guest cells; preserve every process and stop dirty modes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

worker, destination = sys.argv[1:3]
root = Path(destination).resolve()
root.mkdir(parents=True, exist_ok=True)
ops = ['fma','exp2_lowp','exp_lowp','pow_lowp','exp2_midp','exp_midp','pow_midp','sigmoid_midp','silu_midp','log2_lowp','ln_lowp','log10_lowp','log2_midp','ln_midp','log10_midp','combined_patch']
completed=[]
for mode in ['strict','fusing']:
    for op in ops:
        output=root/f'primed-wasm-{mode}-{op}'
        command=[sys.executable,'-B',str(Path(__file__).with_name('stream_worker.py')),worker,str(output),mode,op+'/']
        with (root/f'{output.name}-supervisor.log').open('x') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            state=dict(mode=mode,operation=op,pid=process.pid,completed=completed,required_operations=ops,command=command,started=datetime.now(timezone.utc).isoformat())
            (root/'primed-progress.json').write_text(json.dumps(state,indent=2)+'\n')
            rc=process.wait()
            state.update(exit_code=rc,finished=datetime.now(timezone.utc).isoformat())
            (root/'primed-progress.json').write_text(json.dumps(state,indent=2)+'\n')
            if rc:break
            completed.append(dict(mode=mode,operation=op))
(root/'primed-completed.json').write_text(json.dumps(completed,indent=2)+'\n')
if len(completed)!=2*len(ops):sys.exit(2)

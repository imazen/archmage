"""Fresh-process counters; includes startup, never isolated FMA latency."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
binary=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
out.mkdir(parents=True)
for kind in ('empty','f32','f64'):
    with (out/f'{kind}.log').open('x') as log:
        subprocess.run(['perf','stat','-r','3','-x,','-e','instructions:u,cycles:u','-o',str(out/f'{kind}.csv'),'--','env',f'STUDY_FIRST_USE={kind}',str(binary)],stdout=log,stderr=subprocess.STDOUT,check=True)
(out/'identity.json').write_text(json.dumps(dict(binary=str(binary),sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),repetitions=3,scope='whole process, includes startup'),indent=2)+'\n')

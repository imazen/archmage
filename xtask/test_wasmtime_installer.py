#!/usr/bin/env python3
"""A successful HTTP response containing wrong bytes must never be installed."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
scratch_root = Path.home() / 'tmp'
scratch_root.mkdir(exist_ok=True)
case = Path(tempfile.mkdtemp(prefix='wasmtime-integrity-', dir=scratch_root))
fake = case / 'bin'
fake.mkdir()
curl = fake / 'curl'
curl.write_text('''#!/usr/bin/env bash
set -eu
while (($#)); do
    if [[ $1 == --output ]]; then
        printf 'HTTP 200 with an invalid archive\\n' > "$2"
        exit 0
    fi
    shift
done
exit 2
''')
curl.chmod(0o755)
destination = case / 'install'
env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ['PATH'],
           WASMTIME_CI_HOME=str(destination))
with (case / 'rejection.log').open('w') as log:
    run = subprocess.run(['bash', str(root / 'xtask/install-wasmtime-ci.sh')],
                         env=env, stdout=log, stderr=subprocess.STDOUT)
assert run.returncode != 0, 'wrong bytes passed the integrity check'
assert not (destination / 'bin/wasmtime').exists(), 'wrong bytes reached installation'
assert 'FAILED' in (case / 'rejection.log').read_text()
print('wrong-archive integrity control passed:', case / 'rejection.log')

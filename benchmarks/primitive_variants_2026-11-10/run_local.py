#!/usr/bin/env python3
"""Constrain only this task's transient run-heavy system scope, then benchmark.

Call under RUN_HEAVY_SYSTEM_SCOPE=1 run-heavy, after a separate release build.
Zenbench still verifies placement/admission and owns benchmark serialization.
"""
from pathlib import Path
import os,subprocess,sys
binary=Path(sys.argv[1]).resolve()
cpus=subprocess.check_output([str(binary),'--cpu-set'],text=True).strip()
assert cpus and all(part.isdigit() for part in cpus.split(','))
path=Path('/proc/self/cgroup').read_text().strip().split('::',1)[1]
unit=path.rsplit('/',1)[1]
assert path.startswith('/system.slice/') and unit.startswith('run-') and unit.endswith('.scope'),path
observed=subprocess.check_output(['systemctl','show',unit,'--property=ControlGroup','--value'],text=True).strip()
assert observed==path,(observed,path)
subprocess.run(['sudo','-n','systemctl','set-property','--runtime',unit,'AllowedCPUs='+cpus],check=True)
print('Owned transient scope',unit,'enforced CPUs',cpus,flush=True)
env=dict(os.environ,ZENBENCH_CPU_SET=cpus,ZENBENCH_LAUNCHER_PIDS=str(os.getpid()))
subprocess.run([str(binary),'--measure'],env=env,check=True)

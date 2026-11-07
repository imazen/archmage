#!/usr/bin/env python3
"""Copy only owned, completed evidence, excluding Cargo outputs. Hosts are caller input."""
import json
from pathlib import Path
import subprocess
import sys
host,remote,dest=sys.argv[1:]
dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
# Explicit owned evidence classes; source checkouts and Cargo targets are absent.
code=r'''
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).expanduser()
paths=[]
for p in root.iterdir():
 if p.is_file() and (p.name.startswith(('pow-','timing-','gamma-','frozen-')) and p.suffix in ['.log','.json','.csv','.resource','.data']):paths.append(p.name)
 if p.is_file() and p.name.startswith(('gamma-timing-arm-','gamma-profile-arm-')):paths.append(p.name)
 if p.is_dir() and p.name.startswith(('frozen-','gamma-full-','gamma-profile-','codegen-','feature-matrix-','family-build-cost-','build-cost-before-features','profiles-controls-','results-native-')):paths.append(p.name)
 if p.name=='.zenbench' and p.is_dir():paths.append(p.name)
print(json.dumps(sorted(set(paths))))
'''
paths=json.loads(subprocess.check_output(['ssh',host,'python3','-c',"'"+code.replace("'","'\\''")+"'",remote],text=True))
assert paths,'no owned evidence'
for name in paths:
    subprocess.run(['rsync','-a','--exclude=target','--exclude=target-*',host+':'+remote.rstrip('/')+'/'+name,str(dest)+'/'],check=True)
(dest/'copy-inventory.json').write_text(json.dumps(dict(source=host+':'+remote,entries=paths,note='Completed owned evidence only; caller checks no active writers before invocation.'),indent=2)+'\n')
print('PASS copied',len(paths),'owned entries')

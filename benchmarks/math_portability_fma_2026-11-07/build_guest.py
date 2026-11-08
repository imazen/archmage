"""Export compiler-assigned guest symbols through the linker, without unsafe attributes."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
mode=sys.argv[2];assert mode in ('strict','relaxed')
features='+simd128'+(',+relaxed-simd' if mode=='relaxed' else '')
flags=f'-Ctarget-feature={features} -Cmetadata=archmage_portability_guest -Clink-dead-code=yes'
manifest=Path(__file__).resolve().parent/'Cargo.toml'
cmd=['cargo','build','--release','--manifest-path',str(manifest),'--target','wasm32-wasip1','--bin','archmage-portability-study']
wasm=manifest.parent/'target/wasm32-wasip1/release/archmage-portability-study.wasm'
with (out/f'{mode}-discover.log').open('w') as log:
    subprocess.run(cmd,env=dict(os.environ,RUSTFLAGS=flags+' -Clink-arg=--export-all'),stdout=log,stderr=subprocess.STDOUT,check=True)
names=set(re.findall(rb'_RNvNtCs[0-9A-Za-z]+_26archmage_portability_study5guest(?:13guest_prepare|9guest_run|11guest_probe)',wasm.read_bytes()))
assert len(names)==3, names
final_flags=flags+' '+ ' '.join('-Clink-arg=--export='+name.decode() for name in sorted(names))
with (out/f'{mode}-export.log').open('w') as log:
    subprocess.run(cmd,env=dict(os.environ,RUSTFLAGS=final_flags),stdout=log,stderr=subprocess.STDOUT,check=True)
shutil.copy2(wasm,out/f'{mode}.wasm')
(out/f'{mode}-build.json').write_text(json.dumps(dict(command=cmd,RUSTFLAGS=final_flags,exports=[n.decode() for n in sorted(names)]),indent=2)+'\n')

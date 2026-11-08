"""Export compiler-assigned WASM symbols; adapted from archmage da295fadb24fa."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
mode=sys.argv[2];assert mode in ['strict','relaxed']
features='+simd128'+(',+relaxed-simd' if mode=='relaxed' else '')
flags=f'-Ctarget-feature={features} -Cmetadata=bounded_powf_guest -Clink-dead-code=yes'
manifest=Path(__file__).resolve().parent/'Cargo.toml'
cmd=['cargo','build','--release','--manifest-path',str(manifest),'--target','wasm32-wasip1','--bin','powf_guest','--features','wasm-guest']
wasm=manifest.parent/'target/wasm32-wasip1/release/powf_guest.wasm'
with (out/f'{mode}-discover.log').open('w') as log:
    subprocess.run(cmd,env=dict(os.environ,RUSTFLAGS=flags+' -Clink-arg=--export-all'),stdout=log,stderr=subprocess.STDOUT,check=True)
def leb(data,i):
    result=shift=0
    while True:
        b=data[i];i+=1;result|=(b&127)<<shift
        if b<128:return result,i
        shift+=7
raw=wasm.read_bytes();assert raw[:8]==b'\0asm\x01\0\0\0'
i=8;exports=[]
while i<len(raw):
    section=raw[i];length,i=leb(raw,i+1);end=i+length
    if section==7:
        count,i=leb(raw,i)
        for _ in range(count):
            n,i=leb(raw,i);name=raw[i:i+n].decode();i+=n;kind=raw[i];i+=1;_,i=leb(raw,i)
            if kind==0:exports.append(name)
    i=end
names=[]
for function in ['guest_prepare','guest_probe','guest_run','guest_verify']:
    found=[n for n in exports if n.endswith(function)]
    assert len(found)==1,(function,found)
    names+=found
final_flags=flags+' '+' '.join('-Clink-arg=--export='+n for n in names)
with (out/f'{mode}-export.log').open('w') as log:
    subprocess.run(cmd,env=dict(os.environ,RUSTFLAGS=final_flags),stdout=log,stderr=subprocess.STDOUT,check=True)
shutil.copy2(wasm,out/f'{mode}.wasm')
(out/f'{mode}-build.json').write_text(json.dumps(dict(command=cmd,RUSTFLAGS=final_flags,exports=names),indent=2)+'\n')

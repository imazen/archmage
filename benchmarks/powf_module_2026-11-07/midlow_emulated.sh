#!/usr/bin/env bash
# Frozen final binaries: correctness only, never emulated performance.
set -euo pipefail
out=$(realpath -m "$1")
mkdir "$out"
rustc -Vv > "$out/rustc.txt"
tests=(--test powf_tails_portable_midp --test powf_tails_portable_lowp --test powf_tails_backend_midp --test powf_tails_backend_lowp --test powf_engine_remap)
for target in x86_64-unknown-linux-gnu i686-unknown-linux-gnu; do
  export CARGO_TARGET_DIR="$out/targets/$target"
  cargo test -p magetypes --release --all-features --target "$target" "${tests[@]}" --no-run --message-format=json > "$out/$target-build.jsonl"
  python3 -B - "$out" "$target" <<'PY'
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
out=Path(sys.argv[1]);target=sys.argv[2]
modules=[Path(x['executable'])for s in(out/f'{target}-build.jsonl').read_text().splitlines()if (x:=json.loads(s)).get('reason')=='compiler-artifact'and x.get('executable')and x['target']['kind']==['test']]
assert len(modules)==5,modules
directory=out/target;directory.mkdir()
runner=['qemu-x86_64','-cpu','Nehalem']if target.startswith('x86_64')else['qemu-i386']
for p in modules:
    frozen=directory/p.name;shutil.copy2(p,frozen)
    (directory/(p.name+'.sha256')).write_text(hashlib.sha256(frozen.read_bytes()).hexdigest()+'\n')
    with(directory/(p.name+'.log')).open('w')as log:
        subprocess.run([*runner,str(frozen),'--nocapture'],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(target,p.name,'PASS',flush=True)
PY
done

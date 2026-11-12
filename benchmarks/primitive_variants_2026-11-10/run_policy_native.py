#!/usr/bin/env python3
"""Freeze a verified native binary, run each prescribed context once, retain all.

The controlled driver refreshes its own claim and durable PID/stage every 30s.
There are no retries, gate changes or data-dependent round extensions.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('binary',type=Path)
p.add_argument('--source',required=True);p.add_argument('--tiers',required=True)
p.add_argument('--wrapper',type=Path,required=True);p.add_argument('--verification',type=Path,required=True)
p.add_argument('--exploratory-macos',action='store_true');args=p.parse_args()
root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
claim=Path('.workongoing');assert 'codex-portability-fma' in claim.read_text()
binary=root/'policy_perf';assert not binary.exists();shutil.copy2(args.binary,binary)
shutil.copy2(args.verification,root/'verification.log')
manifest=dict(source=args.source,compiler=subprocess.check_output(['rustc','-vV'],text=True),
              binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
              zenbench='73302a1c115836b2646f6a9adc32058c20d1a66f',
              groups=66,arms=1566,processes=1 if args.exploratory_macos else 5,rounds_per_process=50,
              entry='typed vector proof retained',flags='default release; no target-cpu=native',
              exploratory=args.exploratory_macos)
(root/'SOURCE.json').write_text(json.dumps(manifest,indent=2)+'\n')
done=[]
for tier in args.tiers.split(','):
    output=root/(tier+'.json.gz');log=root/(tier+'.log');assert not output.exists() and not log.exists()
    env=dict(os.environ,TMPDIR=str(Path.home()/'tmp'),ARCHMAGE_PRIMITIVE_RESULTS=str(output),ZENBENCH_GATE_TRACE='1')
    command=[str(args.wrapper.resolve()),'--mem','6G' if args.exploratory_macos else '8G','--jobs','2','--']
    if args.exploratory_macos:
        command += [str(binary),'--measure','--policy-tier',tier,'--exploratory-macos','--cpu-policy=unrestricted']
    else:
        env['RUN_HEAVY_SYSTEM_SCOPE']='1'
        command += [sys.executable,'-B',str(Path(__file__).resolve().parent/'run_local.py'),str(binary),'--policy-tier',tier]
    with log.open('w') as stream:
        child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        while True:
            now=datetime.now(timezone.utc).isoformat()
            claim.write_text(now+' codex-portability-fma frozen policy '+tier+'\n')
            (root/'progress.json').write_text(json.dumps(dict(stage=tier,wrapper_pid=child.pid,command=command,last_check_utc=now,completed=done),indent=2)+'\n')
            try:code=child.wait(timeout=30);break
            except subprocess.TimeoutExpired:pass
    done.append(dict(tier=tier,returncode=code,packet_present=output.is_file()))
    print(done[-1],flush=True)
    if not output.is_file():
        (root/'progress.json').write_text(json.dumps(dict(stage='failed-no-packet',completed=done),indent=2)+'\n')
        raise RuntimeError('Measurement failed before saving a complete packet; inspect '+str(log))
(root/'progress.json').write_text(json.dumps(dict(stage='complete',completed=done,last_check_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')

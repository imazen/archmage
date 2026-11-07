#!/usr/bin/env python3
"""Bounded orchestration controls: signals, original failure bytes and identity guards."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import unittest

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('plan',HERE/'sweep_plan.py');PLAN=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(PLAN)
BASE=Path(os.environ['GAMMA_DRIVER_TEST_OUT']);BASE.mkdir(exist_ok=False)
class Checks(unittest.TestCase):
    def identity(self,name,body):
        root=BASE/name;root.mkdir();identity=root/'identity';identity.mkdir();binary=identity/'zenmath-gamma-opt'
        binary.write_text('#!'+sys.executable+'\n'+body);binary.chmod(0o755)
        payload=b'synthetic source identity fixture\n';digest=hashlib.sha256(payload).hexdigest();manifest=f'{digest}  dummy.rs\n';(identity/'sources.sha256').write_text(manifest)
        with tarfile.open(identity/'source.tar.gz','w:gz') as tar:
            member=tarfile.TarInfo('source/dummy.rs');member.size=len(payload);tar.addfile(member,io.BytesIO(payload))
        build=dict(binary_sha256=PLAN.sha(binary),source_manifest_sha256=PLAN.sha(identity/'sources.sha256'),source_archive_sha256=PLAN.sha(identity/'source.tar.gz'),timed_kernel_byte_match={'fixture':digest})
        (identity/'build.json').write_text(json.dumps(build));return root,identity
    def command(self,identity,out):return [sys.executable,str(HERE/'sweep_plan.py'),str(identity),str(out)]
    def test_child_signal_preserves_nonutf8_bytes(self):
        body="import os,signal,sys\nfrom pathlib import Path\np=Path(sys.argv[5]);p.mkdir();(p/'FAILURE.txt').write_bytes(b'original\\x00\\xffdiagnostic')\nos.kill(os.getpid(),signal.SIGTERM)\n"
        root,identity=self.identity('child-signal',body);out=root/'run';r=subprocess.run(self.command(identity,out),capture_output=True,timeout=10)
        self.assertNotEqual(r.returncode,0);failure=out/'0-piece-e5-w0/FAILURE.txt';self.assertTrue(failure.read_bytes().startswith(b'original\x00\xffdiagnostic'));self.assertIn(b'returncode=-15',failure.read_bytes());self.assertTrue((out/'FAILURE.txt').exists());self.assertFalse((out/'SUCCESS.txt').exists())
    def test_parent_signal_terminates_recorded_child(self):
        body="import os,time,sys\nfrom pathlib import Path\np=Path(sys.argv[5]);p.mkdir();(p/'child.pid').write_text(str(os.getpid()))\nwhile True:time.sleep(.1)\n"
        root,identity=self.identity('parent-signal',body);out=root/'run';p=subprocess.Popen(self.command(identity,out),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        pidfile=out/'0-piece-e5-w0/child.pid';deadline=time.monotonic()+5
        while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.02)
        self.assertTrue(pidfile.exists());child=int(pidfile.read_text());p.send_signal(signal.SIGTERM);p.communicate(timeout=10)
        self.assertNotEqual(p.returncode,0);self.assertTrue((out/'FAILURE.txt').exists());self.assertTrue((pidfile.parent/'FAILURE.txt').exists());self.assertFalse((out/'SUCCESS.txt').exists());self.assertFalse(Path(f'/proc/{child}').exists())
    def test_archive_bytes_must_match_manifest(self):
        root,identity=self.identity('archive-identity','raise SystemExit(0)\n');build=json.loads((identity/'build.json').read_text())
        PLAN.validate_archive(identity,build)
        with tarfile.open(identity/'source.tar.gz','w:gz') as tar:
            member=tarfile.TarInfo('source/dummy.rs');payload=b'wrong source bytes\n';member.size=len(payload);tar.addfile(member,io.BytesIO(payload))
        build['source_archive_sha256']=PLAN.sha(identity/'source.tar.gz')
        with self.assertRaises(ValueError):PLAN.validate_archive(identity,build)
    def test_existing_prepare_identity_is_refused(self):
        root,identity=self.identity('existing','raise SystemExit(0)\n');out=root/'run';cmd=self.command(identity,out)+['--prepare-only'];subprocess.run(cmd,check=True,capture_output=True);original=(out/'plan.json').read_bytes();r=subprocess.run(cmd,capture_output=True);self.assertNotEqual(r.returncode,0);self.assertEqual((out/'plan.json').read_bytes(),original)

if __name__=='__main__':unittest.main()

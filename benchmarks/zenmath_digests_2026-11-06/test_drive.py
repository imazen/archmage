import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import drive
import audit_matrix

class PrepareTests(unittest.TestCase):
    def test_existing_identity_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as name:
            root=Path(name)
            for name in ['sources.sha256','build.json','rustc.txt']:
                (root/name).write_text('original-'+name)
            with self.assertRaisesRegex(ValueError,'already exists'):
                drive.prepare(SimpleNamespace(out=root))
            for name in ['sources.sha256','build.json','rustc.txt']:
                self.assertEqual((root/name).read_text(),'original-'+name)

    def test_prepare_pins_and_checks_compiler_source_target(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as name:
            root=Path(name); binary=root/'binary';binary.write_text('binary')
            args=SimpleNamespace(out=root/'identity',binary=binary,target=None,revision='5a9cc685',runner='')
            manifest='frozen source\n';digest=hashlib.sha256(manifest.encode()).hexdigest()
            embedded=f'source_sha256={digest}\nrustc=rustc 1.99.0 test\ntarget=test-target\n'
            with patch.object(drive,'sources',return_value=manifest),patch.object(drive.subprocess,'check_output',side_effect=['rustc 1.99.0 test\nhost: test-host\n',embedded]) as command:
                drive.prepare(args)
                self.assertEqual(command.call_args_list[0].kwargs['env']['RUSTUP_TOOLCHAIN'],'1.99.0')
            original=(args.out/'build.json').read_bytes()
            with self.assertRaises(ValueError): drive.prepare(args)
            self.assertEqual((args.out/'build.json').read_bytes(),original)

class SignalTests(unittest.TestCase):
    def test_signalled_child_retires_summary_and_records_failure(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as name:
            root=Path(name);build=root/'build';build.mkdir();out=root/'run';out.mkdir()
            binary=root/'binary';binary.write_text('test')
            manifest='test source\n'
            identity=dict(binary=str(binary),binary_sha256=drive.sha(binary),source_sha256=hashlib.sha256(manifest.encode()).hexdigest(),runner='')
            (build/'build.json').write_text(json.dumps(identity))
            (out/'summary.txt').write_text('status=COMPLETE-F32-DOMAIN\n')
            args=SimpleNamespace(build=build,out=out,op='exp',backend='reference',threads=1,start=0,count=11,chunk=7,compare=True,resume=True)
            with patch.object(drive,'sources',return_value=manifest),patch.object(drive.subprocess,'run',side_effect=subprocess.CalledProcessError(-15,['test'])):
                with self.assertRaises(subprocess.CalledProcessError): drive.run(args)
            self.assertFalse((out/'summary.txt').exists())
            self.assertIn('signal=15',(out/'FAILURE.txt').read_text())
            self.assertEqual(len(list(out.glob('summary.txt.*.bak'))),1)

    def test_subprocess_metadata_preserves_existing_failure_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as name:
            root=Path(name);build=root/'build';build.mkdir();out=root/'run';out.mkdir()
            binary=root/'binary';binary.write_text('test')
            manifest='test source\n'
            identity=dict(binary=str(binary),binary_sha256=drive.sha(binary),source_sha256=hashlib.sha256(manifest.encode()).hexdigest(),runner='')
            (build/'build.json').write_text(json.dumps(identity))
            original=b'MISMATCH chunk=91 input=3f800000 actual=00000000 reference=3f800000\nrerun exact bytes\xff\x00'
            (out/'FAILURE.txt').write_bytes(original)
            args=SimpleNamespace(build=build,out=out,op='exp',backend='reference',threads=1,start=0,count=11,chunk=7,compare=True,resume=True)
            with patch.object(drive,'sources',return_value=manifest),patch.object(drive.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['test'])):
                with self.assertRaises(subprocess.CalledProcessError):drive.run(args)
            result=(out/'FAILURE.txt').read_bytes()
            self.assertEqual(result[:len(original)],original)
            self.assertIn(b'child returncode=1; signal=none',result)

class MatrixLabelTests(unittest.TestCase):
    def test_mislabeled_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'backend'):
            audit_matrix.validate_cell_labels(dict(op='exp',backend='reference',target=audit_matrix.ARM),'exp','neon',audit_matrix.ARM)
    def test_mislabeled_operation_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'op'):
            audit_matrix.validate_cell_labels(dict(op='ln',backend='neon',target=audit_matrix.ARM),'exp','neon',audit_matrix.ARM)
    def test_mislabeled_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'target'):
            audit_matrix.validate_cell_labels(dict(op='exp',backend='neon',target=audit_matrix.X86),'exp','neon',audit_matrix.ARM)

if __name__=='__main__': unittest.main()

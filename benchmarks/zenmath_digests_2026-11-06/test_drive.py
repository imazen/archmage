import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import drive

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

if __name__=='__main__': unittest.main()

#!/usr/bin/env python3
"""A failed attempt to reuse a frozen snapshot must preserve its bytes."""
from pathlib import Path
import subprocess
import sys
import tempfile

script = Path(__file__).with_name("freeze.py").resolve()
with tempfile.TemporaryDirectory(dir=Path.home() / "tmp", prefix="dispatch-freeze-test-") as tmp:
    out = Path(tmp) / "existing"
    out.mkdir()
    identity = out / "identity.json"
    identity.write_bytes(b'{"sentinel":"immutable"}\n')
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    result = subprocess.run([sys.executable, str(script), str(out)], capture_output=True)
    assert result.returncode != 0
    assert b"FileExistsError" in result.stderr
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
print("PASS existing snapshot rejected before any writes")

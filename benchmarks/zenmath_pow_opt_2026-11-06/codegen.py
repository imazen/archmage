#!/usr/bin/env python3
"""Extract linked policy bodies from the exact retained timing executable."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

binary = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=False)
asm = subprocess.check_output(["objdump", "-d", "-C", str(binary)], text=True)
(out / "linked.disasm").write_text(asm)
headers = list(re.finditer(r"^([0-9a-f]+) <([^\n]+)>:\n", asm, re.M))
rows = []
for i, h in enumerate(headers):
    symbol = h.group(2)
    if "zenmath::simd::policy::native_v3::<" not in symbol:
        continue
    body = asm[h.start():headers[i + 1].start() if i + 1 < len(headers) else len(asm)]
    policy = int(re.search(r"::<([0-9]+)>", symbol).group(1))
    calls = [line for line in body.splitlines() if re.search(r"\bcall\s", line)]
    assert not any(any(helper in line for helper in ["fmaf_v3", "floorf_v2", "roundevenf_v2", "fmaf_soft"]) for line in calls), calls
    if (policy >> 9) & 3 == 2:
        assert not calls, calls
    path = f"policy-{policy}.asm"
    (out / path).write_text(body)
    rows.append(dict(symbol=symbol, address=h.group(1), policy=policy, artifact=path,
                     calls=calls, fma_instructions=len(re.findall(r"\bvfmadd", body)),
                     round_instructions=len(re.findall(r"\bvroundss", body))))
assert len(rows) == 8, rows
(out / "identity.json").write_text(json.dumps(dict(binary=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), bodies=rows), indent=2) + "\n")
print(json.dumps(rows, indent=2))

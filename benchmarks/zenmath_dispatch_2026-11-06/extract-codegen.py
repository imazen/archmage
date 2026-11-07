#!/usr/bin/env python3
"""Extract named linked functions; retain full disassembly line references."""

from pathlib import Path
import hashlib
import json
import re
import sys

src = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=False)
patterns = sys.argv[3:]
lines = src.read_text().splitlines(keepends=True)
starts = [
    (i, re.match(r"^([0-9a-f]+) <(.+)>:\s*$", line)) for i, line in enumerate(lines)
]
starts = [(i, m) for i, m in starts if m]
records = []
for pos, (i, match) in enumerate(starts):
    symbol = match.group(2)
    if not any(p in symbol for p in patterns):
        continue
    end = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
    data = "".join(lines[i:end]).encode()
    name = f"{match.group(1)}.asm"
    (out / name).write_bytes(data)
    records.append(
        dict(
            symbol=symbol,
            address=match.group(1),
            source_line=i + 1,
            lines=end - i,
            artifact=name,
            sha256=hashlib.sha256(data).hexdigest(),
        )
    )
assert records, "no linked symbols matched"
(out / "index.json").write_text(
    json.dumps(
        dict(
            source=str(src),
            source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
            symbols=records,
        ),
        indent=2,
    )
    + "\n"
)
print(f"PASS {len(records)} linked functions extracted")

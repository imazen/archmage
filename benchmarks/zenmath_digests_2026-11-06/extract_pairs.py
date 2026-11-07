#!/usr/bin/env python3
"""Preserve the independent, existing MPFR powf fixtures as harness-only data."""
import hashlib
from pathlib import Path
import re

root=Path(__file__).resolve().parents[2]
source=root/'zenmath/tests/core_math.rs'
text=source.read_text()
section=text.split('fn mpfr_powf() {',1)[1]
match=re.search(r'let cases:\s*\[\[u32;\s*3\];\s*(\d+)\]\s*=\s*\[(.*?)\n\];',section,re.S)
assert match, 'MPFR powf fixture shape changed'
count=int(match.group(1))
body=match.group(2)
rows=re.findall(r'\[(0x[0-9a-f]+),(0x[0-9a-f]+),(0x[0-9a-f]+)\]',body)
assert len(rows)==count
output='// Independent MPFR expected bits copied from zenmath/tests/core_math.rs.\n'
output+='// Source SHA256 '+hashlib.sha256(source.read_bytes()).hexdigest()+'\n'
output+='// CORE-MATH origin 660067dfd3e751c8f07acc31c899ff410aa7206e; MIT.\n'
output+='// Regenerate with just pairs-fixture; production fixture remains unchanged.\n'
output+='#[rustfmt::skip]\n'
output+=f'pub(super) const POWF: [[u32; 3]; {count}] = [\n'
output+=''.join(f'    [{a}, {b}, {want}],\n' for a,b,want in rows)
output+='];\n'
(Path(__file__).resolve().parent/'src/pow_mpfr_cases.rs').write_text(output)

#!/usr/bin/env python3
"""Extract the unchanged main implementation, never the PR's forwarding methods."""
from pathlib import Path
import subprocess
import re

ROOT = Path(__file__).resolve().parent
BASE = "91a13739b73f46213ecfe7083c77297c531fc8ad"
NAMES = ["exp2_midp", "exp2_lowp", "exp2_midp_unchecked", "log2_midp", "log2_lowp", "log2_midp_unchecked"]
out = [f"// Frozen from imazen/archmage {BASE}; regenerate with freeze.py.\n"]
for width in [4, 8, 16]:
    source = subprocess.check_output(["git", "show", f"{BASE}:magetypes/src/simd/generic/generated/transcendentals_f32x{width}.rs"], text=True)
    start = source.index("use crate::simd::backends::")
    prefix = source[start:source.index("impl<T:", start)].replace("crate::simd", "magetypes::simd")
    prefix = re.sub(r"\{F32x\d+Backend, (F32x\d+Convert), I32x\d+Backend\}",r"{\1}",prefix)
    prefix = re.sub(r"(i?f?32x\d+)::from_repr_unchecked\(token, <T as \w+>::splat\(token, v\)\)",r"\1::splat_t(token, v)",prefix)
    prefix = re.sub(r"(i32x\d+)::from_repr_unchecked\(token, <T as \w+>::splat\(token, v\)\)",r"\1::splat_t(token, v)",prefix)
    out.append(f"pub mod x{width} {{\n{prefix}\npub trait Frozen<T> {{\n")
    for name in NAMES:
        out.append(f"fn old_{name}(self, token:T)->Self;\n")
    out.append(f"}}\nimpl<T:F32x{width}Convert> Frozen<T> for f32x{width}<T> {{\n")
    for name in NAMES:
        start = source.index(f"pub fn {name}(self)")
        brace = source.index("{", start)
        depth, end = 1, brace + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        body = source[start:end].replace("pub fn", "fn").replace("(self)","(self, token:T)").replace("self.1", "token")
        for old in sorted(NAMES, key=len, reverse=True):
            # Replace complete identifiers, including calls to other frozen methods.
            body = re.sub(r"\b" + old + r"\b", "old_" + old, body)
        body = re.sub(r"\.old_(\w+)\(\)",r".old_\1(token)",body)
        out.append("#[inline(always)]\n" + body + "\n")
    out.append("}\n}\n")
(ROOT / "src/frozen.rs").write_text("".join(out))

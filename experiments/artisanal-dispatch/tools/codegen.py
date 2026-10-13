#!/usr/bin/env python3
"""Assert the two x86 call boundaries; preserve complete expansion and assembly."""
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[1]
out = root / "artifacts"
out.mkdir(exist_ok=True)
asm = root / "target" / "audit" / "codegen.s"
asm.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["cargo", "rustc", "--release", "--example", "codegen", "--features", "std",
                "--", f"--emit=asm={asm}"], cwd=root, check=True)
source = asm.read_text()
excerpts = []
for name in ["leaf_v3", "matching", "proof_boundary"]:
    pattern = rf"^(_R[^\n:]*{name}[^\n:]*):\n(.*?)^\s*\.size\s+\1,.*$"
    matches = list(re.finditer(pattern, source, flags=re.M | re.S))
    assert len(matches) == 1, (name, len(matches))
    body = matches[0].group(2)
    if name == "proof_boundary":
        assert len(re.findall(r"\bjmp\b", body)) == 1 and "leaf_v3" in body, body
        assert not re.search(r"\b(callq?|cpuid|j[enzabgl][a-z]*)\b", body), body
    else:
        assert len(re.findall(r"\bvfmadd213ss\b", body)) == 1, body
        assert not re.search(r"\b(callq?|jmp|cpuid)\b", body), body
    excerpts.append(name + ":\n" + body)
(out / "codegen.txt").write_text(subprocess.check_output(["rustc", "--version"], text=True)
                                + "Command: cargo rustc --release --example codegen --features std -- --emit=asm\n"
                                + "No target-cpu or target-feature flags.\n\n" + "\n".join(excerpts))
for args, name in [(["--example", "codegen"], "codegen-expanded.rs"), (["--lib"], "proofs-expanded.rs")]:
    result = subprocess.run(["cargo", "expand", *args, "--features", "std"], cwd=root, text=True,
                            capture_output=True, check=True)
    path = out / name
    path.write_text(result.stdout)
    print(path.name, len(result.stdout.encode()), "bytes")

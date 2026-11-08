#!/usr/bin/env python3
"""Inventory literal method call sites in local tracked Rust files using ripgrep.

Output can contain private source text. Choose its destination accordingly.
Categories are source heuristics, not Rust name resolution or execution counts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    pattern = (
        r"\.\s*((?:log2|ln|log10|exp2|exp|pow)_"
        r"(?:lowp_unchecked|midp_unchecked|midp_precise|lowp|midp)"
        r"|cbrt_(?:midp_precise|lowp|midp)|sigmoid_midp|silu_midp"
        r"|mul_(?:add|sub)(?:_(?:fused|portable|relaxed))?)\s*\("
    )
    result = {"pattern": pattern, "repositories": []}
    for root in args.roots:
        root = root.resolve()
        tracked = set(git(root, "ls-files", "--", "*.rs").splitlines())
        proc = subprocess.run(
            ["rg", "--json", "-U", "-g", "*.rs", "-g", "!**/target/**",
             "-g", "!**/.git/**", pattern, str(root)],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode not in (0, 1):
            raise RuntimeError(proc.stderr)
        matches = {}
        for line in proc.stdout.splitlines():
            event = json.loads(line)
            if event["type"] != "match":
                continue
            data = event["data"]
            path = Path(data["path"]["text"])
            relative = str(path.relative_to(root))
            if relative not in tracked:
                continue
            source_bytes = path.read_bytes()
            source = source_bytes.decode().splitlines()
            source_sha256 = hashlib.sha256(source_bytes).hexdigest()
            first_test = next((i + 1 for i, text in enumerate(source)
                               if re.match(r"\s*#\[cfg\(test\)\]", text)), len(source) + 1)
            for match in data["submatches"]:
                number = data["line_number"] + data["lines"]["text"][:match["start"]].count("\n")
                text = source[number - 1].strip()
                if text.startswith("//"):
                    continue
                method = re.search(r"[a-z][a-z0-9_]*", match["match"]["text"]).group()
                category = "production candidate"
                if any(part in {"tests", "benches", "benchmarks", "examples", "tools", "dev",
                                "experiments", "experimental"} for part in path.parts) or "bench_" in path.name:
                    category = "test/benchmark/example/experiment"
                elif path.name == "exactness.rs" or number >= first_test:
                    category = "test/audit"
                key = (relative, number, method)
                matches[key] = dict(path=relative, line=number, method=method,
                                    category=category, text=text, source_sha256=source_sha256)
        result["repositories"].append(dict(
            root=str(root), head=git(root, "rev-parse", "HEAD"),
            dirty=bool(git(root, "status", "--porcelain")),
            matches=sorted(matches.values(), key=lambda x: (x["path"], x["line"], x["method"])),
        ))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

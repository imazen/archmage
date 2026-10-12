#!/usr/bin/env python3
"""Generate, compile, and capture the attune matrix. Run under run-heavy.

Raw means cargo expand --ugly, saved byte-for-byte. Positive raw output is also
recompiled with rustc's generated-code lint cap; source inputs forbid unsafe.
No existing artifact directory is overwritten and no failed case is skipped.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from cases import TARGETS, corpus

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def command(argv, directory, stem, env):
    started = time.monotonic()
    with (directory / f"{stem}.stdout").open("wb") as stdout, (directory / f"{stem}.stderr").open("wb") as stderr:
        result = subprocess.run(argv, cwd=directory, env=env, stdout=stdout, stderr=stderr)
    record = dict(command=argv, exit_code=result.returncode, seconds=time.monotonic() - started)
    save(directory / f"{stem}.command.json", record)
    return result.returncode


def manifest(root, raw=None):
    lib = str(raw) if raw else "src/lib.rs"
    return f'''[workspace]
[package]
name = "attune-expansion-corpus"
version = "0.0.0"
edition = "2024"
publish = false
[lib]
path = {json.dumps(lib)}
[dependencies]
archmage = {{ path = {json.dumps(str(root))}, default-features = false }}
magetypes = {{ path = {json.dumps(str(root / 'magetypes'))}, default-features = false }}
[features]
default = []
avx512 = ["archmage/avx512", "magetypes/avx512"]
optional = []
rejections = []
'''


def generate(directory, arch, enabled, groups=None):
    cases = corpus(arch, enabled)
    if groups is not None:
        cases = [case for case in cases if case.group in groups]
        if not cases:
            raise ValueError("requested groups select no cases")
    directory.mkdir(parents=True)
    source = directory / "src" / "cases"
    source.mkdir(parents=True)
    includes = ["#![forbid(unsafe_code)]", "#![deny(warnings)]", "#![allow(dead_code, unused_imports, unused_variables)]"]
    for case in cases:
        (source / f"{case.name}.rs").write_text(case.source + "\n")
        predicate = 'feature = "rejections"' if case.error else 'not(feature = "rejections")'
        includes += [f"#[cfg({predicate})]", f'#[path = "cases/{case.name}.rs"]', f"pub mod {case.name};"]
    (directory / "src/lib.rs").write_text("\n".join(includes) + "\n")
    (directory / "Cargo.toml").write_text(manifest(ROOT))
    save(directory / "cases.json", [asdict(case) for case in cases])
    index = [f"# {arch}: gates {'on' if enabled else 'off'}", "",
             "[Complete positive-case raw expansion](expanded-raw.rs) · "
             "[Rejected raw expansion](rejected-expanded-raw.rs) · "
             "[Diagnostics](rejection-findings.json)", "",
             "Each raw module is an exact excerpt of cargo expand --ugly output. "
             "Rejected modules may contain error placeholders rather than usable Rust.", "",
             "| Case | Group | Expected | Input | Raw module |", "| --- | --- | --- | --- | --- |"]
    for case in cases:
        raw_dir = "rejected-expanded-raw-cases" if case.error else "expanded-raw-cases"
        expectation = case.error or "pass"
        index.append(f"| {case.name} | {case.group} | {expectation} | [source](src/cases/{case.name}.rs) | [raw]({raw_dir}/{case.name}.rs) |")
    (directory / "README.md").write_text("\n".join(index) + "\n")
    return cases


def source_cases(span):
    result = set()
    path = Path(span.get("file_name", ""))
    if path.parent.name == "cases":
        result.add(path.stem)
    expansion = span.get("expansion")
    if expansion:
        result.update(source_cases(expansion.get("span", {})))
    return result


def diagnostics(path):
    by_case = defaultdict(list)
    unassigned = []
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        if event.get("reason") != "compiler-message":
            continue
        message = event["message"]
        if message["level"] != "error":
            continue
        owners = set()
        for span in message["spans"]:
            if span.get("is_primary"):
                owners.update(source_cases(span))
        if owners:
            for owner in owners:
                by_case[owner].append(message["message"])
        else:
            unassigned.append(message["message"])
    return dict(by_case), unassigned


def brace_end(text, start):
    """Find a Rust block end while ignoring comments and string/char literals.

    Used only to extract generated module boundaries, never to rewrite code.
    The complete raw compiler output is always preserved independently.
    """
    depth = 0
    token = re.compile(r'//[^\n]*|/\*.*?\*/|r(#+)?".*?"\1|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])\'|[{}]', re.S)
    for match in token.finditer(text, start):
        if match.group() == "{":
            depth += 1
        elif match.group() == "}":
            depth -= 1
            if depth == 0:
                return match.end()
    raise ValueError("unclosed generated module")


def inspect_raw(directory, cases, raw, rejected=False):
    text = raw.read_text()
    output = directory / ("rejected-expanded-raw-cases" if rejected else "expanded-raw-cases")
    output.mkdir()
    findings = []
    for case in cases:
        if bool(case.error) != rejected:
            continue
        match = re.search(r"\bpub mod " + re.escape(case.name) + r"\s*\{", text)
        if not match:
            findings.append(dict(case=case.name, problem="module absent from raw expansion"))
            continue
        end = brace_end(text, match.end() - 1)
        block = text[match.start():end]
        (output / f"{case.name}.rs").write_text(block + "\n")
        if rejected:
            continue
        for needle in case.absent:
            if needle in block:
                findings.append(dict(case=case.name, problem=f"forbidden output {needle}"))
        for needle in case.present:
            if needle not in block:
                findings.append(dict(case=case.name, problem=f"missing output {needle}"))
        if case.tags.get("inspect_caller"):
            support = re.search(r"\bpub mod support\s*\{", block)
            assert support, case.name
            block = block[:support.start()] + block[brace_end(block, support.end() - 1):]
            probes = bool(re.search(r"SimdToken\s*>\s*::\s*summon\s*\(", block))
            if probes != case.tags["runtime_probe"]:
                findings.append(dict(case=case.name, problem="CPU probe policy mismatch", expected=case.tags["runtime_probe"], actual=probes))
    save(directory / ("rejected-raw-findings.json" if rejected else "raw-findings.json"), findings)
    return findings


def check_config(directory, arch, enabled, cases, target_dir):
    failures = []
    env = dict(os.environ, CARGO_TARGET_DIR=str(target_dir))
    target = ["--target", TARGETS[arch]]
    features = ["--features", "avx512,optional"] if enabled else []
    base = ["--manifest-path", str(directory / "Cargo.toml"), "--lib", *target]
    code = command(["cargo", "check", *base, *features, "--message-format=json"], directory, "input", env)
    errors, unassigned = diagnostics(directory / "input.stdout")
    save(directory / "input-errors.json", dict(cases=errors, unassigned=unassigned))
    if code:
        failures.append("positive inputs failed: input-errors.json")
    code = command(["cargo", "expand", "--ugly", "--color", "never", *base, *features], directory, "expand", env)
    raw = directory / "expanded-raw.rs"
    (directory / "expand.stdout").rename(raw)
    if code:
        failures.append("raw expansion reported errors: expand.stderr")
    findings = inspect_raw(directory, cases, raw)
    if findings:
        failures.append("raw inspection failed: raw-findings.json")

    # Raw rustc expansion exposes compiler-generated prelude attributes and
    # unsafe trampoline blocks. Bootstrap admits the former; the lint cap admits
    # the latter after macro hygiene was serialized away. Type checking is intact.
    replay = directory / "replay"
    replay.mkdir()
    (replay / "Cargo.toml").write_text(manifest(ROOT, raw))
    replay_env = dict(env, RUSTC_BOOTSTRAP="1")
    # Pass the cap only to the replay crate. Global RUSTFLAGS would also hide
    # unsupported-crate-type diagnostics Cargo needs when probing WASM targets.
    code = command(["cargo", "rustc", "--manifest-path", str(replay / "Cargo.toml"), "--lib", *target, *features,
                    "--", "--cap-lints=allow", "--emit=metadata"],
                   directory, "raw-replay", replay_env)
    if code:
        failures.append("standalone raw output failed: raw-replay.stderr")

    reject_features = ["--features", "rejections,avx512,optional" if enabled else "rejections"]
    code = command(["cargo", "check", *base, *reject_features, "--message-format=json"], directory, "rejections", env)
    errors, unassigned = diagnostics(directory / "rejections.stdout")
    problems = list(unassigned)
    for case in cases:
        if case.error and not any(case.error in message for message in errors.get(case.name, [])):
            problems.append(dict(case=case.name, expected=case.error, actual=errors.get(case.name, [])))
    if not code:
        problems.append("rejection corpus compiled successfully")
    # Retain rejected expansions even when rustc reports errors. The check above
    # owns the rejection oracle; cargo expand's exit code alone is insufficient.
    command(["cargo", "expand", "--ugly", "--color", "never", *base, *reject_features], directory, "rejected-expand", env)
    (directory / "rejected-expand.stdout").rename(directory / "rejected-expanded-raw.rs")
    problems.extend(inspect_raw(directory, cases, directory / "rejected-expanded-raw.rs", rejected=True))
    save(directory / "rejection-findings.json", dict(problems=problems, diagnostics=errors))
    if problems:
        failures.append("rejection expectations failed: rejection-findings.json")
    return dict(arch=arch, features=enabled, expected_pass=sum(c.error is None for c in cases),
                expected_reject=sum(c.error is not None for c in cases), ok=not failures, failures=failures,
                groups=dict(Counter(c.group for c in cases)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arches", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--gates", nargs="+", choices=("off", "on"), default=["off", "on"])
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="capture every configuration, then fail if any check failed")
    parser.add_argument("--groups", nargs="+", choices=sorted({case.group for case in corpus("x86_64", True)}),
                        help="explicit subset for a focused audit; omission runs every group")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(command=sys.argv, revision=subprocess.check_output(
                        ["jj", "log", "-r", "@", "--no-graph", "-T", "commit_id"]
                        if (ROOT / ".jj").exists() else ["git", "rev-parse", "HEAD"],
                        cwd=ROOT, text=True).strip(),
                    rustc=subprocess.check_output(["rustc", "-Vv"], text=True), results=[])
    save(out / "results.json", metadata)
    index = ["# Attune raw expansion audit", "",
             "[Measured results](results.json) · [Artifact hashes](artifacts.json)", "",
             "Each configuration links the source inputs, exact raw expansion, and diagnostics.", "",
             "| Target | Gates off | Gates on |", "| --- | --- | --- |"]
    for arch in args.arches:
        links = [f"[index]({arch}-{gate}/README.md)" if gate in args.gates else "not requested"
                 for gate in ("off", "on")]
        index.append(f"| {TARGETS[arch]} | {' | '.join(links)} |")
    (out / "README.md").write_text("\n".join(index) + "\n")
    for arch in args.arches:
        for gate in args.gates:
            directory = out / f"{arch}-{gate}"
            cases = generate(directory, arch, gate == "on", args.groups)
            print(f"{arch}/{gate}: {len(cases)} generated cases", flush=True)
            if not args.generate_only:
                result = check_config(directory, arch, gate == "on", cases, out / "target")
                metadata["results"].append(result)
                save(out / "results.json", metadata)
                if not result["ok"] and not args.keep_going:
                    raise RuntimeError(f"audit failed: {directory}; {result['failures']}")
    files = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and "target" not in path.relative_to(out).parts:
            files.append(dict(path=str(path.relative_to(out)), bytes=path.stat().st_size,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(out / "artifacts.json", files)
    print(json.dumps(metadata["results"], indent=2))
    if any(not result["ok"] for result in metadata["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

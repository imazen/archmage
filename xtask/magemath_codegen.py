#!/usr/bin/env python3
"""Score optimized no_std Magemath and ScalarToken consumer code.

Needs Python 3.12+, rustup with the TARGETS, and Linux binutils (readelf).
Native sizes are ELF STT_FUNC sizes. WASM sizes are linked code-section function
bodies, with names and exports used only to identify them. Scores count static
instructions in the union of reachable local functions, once per function.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from datetime import UTC, datetime
import difflib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from time import monotonic
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "xtask/codegen_baselines/magemath_v1.json"
FIXTURE = Path(__file__).with_name("magemath_codegen_probe.rs")
TARGETS = {
    "x86_64": "x86_64-unknown-linux-gnu",
    "aarch64": "aarch64-unknown-linux-gnu",
    "wasm32": "wasm32-unknown-unknown",
    "wasm32_relaxed": "wasm32-unknown-unknown",
    "i686": "i686-unknown-linux-gnu",
}
WASM_ISAS = {"wasm32", "wasm32_relaxed"}
X86_ISAS = {"x86_64", "i686"}
SCHEMA_VERSION = 2
COST_TABLE_VERSION = 3
WEIGHTS = {
    "alu": 1.0,
    "fma": 1.5,
    "shuffle": 1.5,
    "mem": 2.0,
    "stack_spill": 3.0,
    "branch": 3.0,
    "hw_div_sqrt": 10.0,
    "scalarized_lane_move": 4.0,
    "call_internal": 6.0,
    "call_helper": 30.0,
    "panic_or_trap": 100.0,
}
# First matching ISA-specific rule wins, after control transfers and traps.
# WASM explicitly identifies the stack pointer. Ordinary local accesses are
# VM operands, so they are not assumed to be machine register spills.
COST_TABLES = {
    "x86_64": (
        ("hw_div_sqrt", r"^v?(?:div|sqrt)(?:ps|pd|ss|sd)\b"),
        ("fma", r"^v?f[nm]*m(?:add|sub)"),
        ("scalarized_lane_move", r"^v?(?:pinsr|pextr|insertps|extractps)"),
        ("shuffle", r"^v?(?:(?:p)?shuf|perm|(?:p)?unpck|(?:p)?blend|pack|broadcast)"),
        ("stack_spill", r"^(?:push|pop)\w*\b|^(?!lea).*\([^)]*%[er](?:sp|bp)\b"),
        ("mem", r"^(?!lea)\S+\s+.*(?:\(|@GOT|\.(?:L)?CPI)"),
    ),
    "aarch64": (
        ("stack_spill", r"\[(?:sp|x29)\b"),
        ("hw_div_sqrt", r"^f(?:div|sqrt)\b"),
        ("fma", r"^f(?:mla|mls|madd|msub|nmadd|nmsub)\b"),
        ("scalarized_lane_move", r"^(?:ins|umov|smov|mov\.[bhsd])\b|^mov\s+.*\bv\d+\.[bhsd]\["),
        ("shuffle", r"^(?:zip[12]|uzp[12]|trn[12]|ext|tbl|tbx|dup|rev(?:16|32|64))\b"),
        ("mem", r"^(?:ld|st|prfm)\w*\s"),
    ),
    "wasm32": (
        ("stack_spill", r"^global\.(?:get|set)\s+__stack_pointer\b"),
        ("hw_div_sqrt", r"^f(?:32(?:x4)?|64(?:x2)?)\.(?:div|sqrt)\b"),
        ("fma", r"\.relaxed_(?:madd|nmadd)\b"),
        ("scalarized_lane_move", r"\.(?:replace|extract)_lane"),
        ("shuffle", r"\.(?:shuffle|swizzle|splat)\b"),
        ("mem", r"\.(?:load|store|copy|fill)"),
    ),
}
COST_TABLES["wasm32_relaxed"] = COST_TABLES["wasm32"]
COST_TABLES["i686"] = (
    ("hw_div_sqrt", r"^(?:fi?div(?:r)?[slp]?|fsqrt)\b"),
    *COST_TABLES["x86_64"],
)
COST_RULES = {
    isa: [(bucket, re.compile(pattern)) for bucket, pattern in rules]
    for isa, rules in COST_TABLES.items()
}
PANIC = re.compile(r"panicking|slice_index|panic_bounds_check|rust_begin_unwind")
SOFT_FMA = re.compile(r"soft_fma|fmaf?_soft|fmaf?_exact|powf_soft_f32x4|software_lanes")
CALL_MUT = re.compile(r"call_mut")
GOLDSCHMIDT = re.compile(r"goldschmidt|nostd_math.*sqrt", re.I)
HELPER = re.compile(r"nostd_math|compiler_builtins|\b(?:memcpy|memmove|memset|memcmp)\b")
FLOAT_DIV = re.compile(r"^(?:v?div(?:ps|pd|ss|sd)|fdiv|f(?:32(?:x4)?|64(?:x2)?)\.div)\b")
X87_DIV = re.compile(r"^fi?divr?[slp]?\b")
WASM_RELAXED_MADD = re.compile(r"^f(?:32x4|64x2)\.relaxed_(?:madd|nmadd)\b")
# Moving a byte *into* a vector lane can turn packed I/O into a sequence of
# inserts. A scalar extraction (mov b0, v1.b[15]) in a short tail is different.
BYTE_LANE_MOVE = re.compile(r"^mov\.b\b|^(?:mov|ins)\s+v\d+\.b\[")
QUAD_PAIR = re.compile(r"^(?:ldp|stp)\s+q\d+,\s*q\d+,\s*\[(?!sp\b|x29\b)")
DIRECT_CALLS = {"call", "callq", "calll", "return_call", "bl"}
INDIRECT_CALLS = {"call_indirect", "call_ref", "return_call_indirect",
                  "return_call_ref", "blr", "blraa", "blrab"}
CALLS = DIRECT_CALLS | INDIRECT_CALLS
TAILS = {"b", "jmp", "jmpq", "jmpl"}
BRANCH = re.compile(r"^(?:j\w+|b(?:\.\w+)?|br(?:_\w+)?|cbn?z|tbn?z|ret[ql]?|return|"
                    r"if|else|block|loop|end_(?:block|loop|if))$")
TRAPS = {"ud2", "ud1", "brk", "unreachable"}
SYMBOL = r"[-\w.$]+"
FUNCTION = re.compile(rf"^\s*\.type\s+({SYMBOL}),\s*[@%]function\b", re.M)
ALIAS = re.compile(rf"^\s*(?:\.set\s+({SYMBOL}),\s*|({SYMBOL})\s*=\s*)({SYMBOL})\s*$")
DIRECT_SYMBOL = re.compile(rf"({SYMBOL})(?:@PLT)?")
X86_GOT = {
    "x86_64": re.compile(rf"({SYMBOL})@GOTPCREL(?:X)?\(%rip\)"),
    "i686": re.compile(rf"({SYMBOL})@GOT\(%e(?:ax|bx|cx|dx|si|di|bp)\)"),
}
X86_ADDRESS = {
    "x86_64": re.compile(rf"({SYMBOL})\(%rip\)"),
    "i686": re.compile(rf"({SYMBOL})@GOTOFF\(%e(?:ax|bx|cx|dx|si|di|bp)\)"),
}
X86_REGISTERS = {
    "%" + name: "%" + family[0]
    for family in (
        ("rax", "eax", "ax", "al", "ah"), ("rbx", "ebx", "bx", "bl", "bh"),
        ("rcx", "ecx", "cx", "cl", "ch"), ("rdx", "edx", "dx", "dl", "dh"),
        ("rsi", "esi", "si", "sil"), ("rdi", "edi", "di", "dil"),
        ("rbp", "ebp", "bp", "bpl"), ("rsp", "esp", "sp", "spl"),
        *(tuple(f"r{i}{suffix}" for suffix in ("", "d", "w", "b")) for i in range(8, 16)),
    )
    for name in family
}
X86_CALLEE_SAVED = {"%rbx", "%rbp", "%r12", "%r13", "%r14", "%r15"}
I686_CALLEE_SAVED = {"%rbx", "%rbp", "%rsi", "%rdi"}
I686_PC_THUNK = re.compile(r"__x86\.get_pc_thunk\.(ax|bx|cx|dx|si|di|bp)")
X86_IMPLICIT_WRITE = re.compile(
    r"^(?:cpuid|rdtsc[p]?|rdpmc|syscall|sysenter|xgetbv|xlat[b]?|lahf|"
    r"(?:i?mul|i?div|cqo|cdq|cwd|cmpxchg|loop|movs|stos|lods|scas|cmps|leave)\w*)$"
)

SCALAR = (
    "scalar_transpose", "scalar_deinterleave", "scalar_rgb", "scalar_rsqrt",
    "scalar_round", "scalar_bitmask", "scalar_sqrt_f32", "scalar_sqrt_f64",
)
MATH = (
    "powf_unit_checked", "powf_unit_unchecked", "powf_midp_bit_exact", "powf_midp_backend",
    "exp2_midp_bit_exact", "exp2_midp_backend", "exp2_lowp_bit_exact", "exp2_lowp_backend",
    "log2_midp_bit_exact", "log2_midp_backend",
    "log1p_around", "log1p_greater", "log1p_around_scalar", "log1p_greater_scalar",
    "cbrtf_midp_bit_exact", "cbrtf_midp_backend", "cbrtf_lowp_bit_exact", "cbrtf_lowp_backend",
    "cbrtf_midp_bit_exact_scalar", "cbrtf_midp_backend_scalar",
    "cbrtf_lowp_bit_exact_scalar", "cbrtf_lowp_backend_scalar",
)
HIGHP = ("powf_highp_gamma_one", "powf_highp_gamma_four")
COMMON = ("powf_slice_scalar", *HIGHP, *SCALAR)
RELAXED_LOG1P = ("log1p_around_backend", "log1p_greater_backend")
I686_PROBES = ("powf_slice_scalar", *HIGHP, *SCALAR[:6],
               *(name for name in MATH if name.endswith("_scalar")))


def opcode(line):
    """LLVM may print x86 prefixes as part of an instruction."""
    words = line.split()
    if words[0] in {"lock", "rep", "repe", "repne", "repz", "repnz"} and len(words) > 1:
        return words[1]
    return words[0]


def alias_chain(name, aliases):
    names = [name]
    while name in aliases:
        name = aliases[name]
        if name in names:
            raise ValueError(f"cyclic assembler alias: {name}")
        names.append(name)
    return names


def x86_register_update(isa, line, op, operands, target, registers):
    """Track proven symbol addresses within a basic block of AT&T assembly."""
    if op in CALLS:
        callee_saved = I686_CALLEE_SAVED if isa == "i686" else X86_CALLEE_SAVED
        for reg in registers.keys() - callee_saved:
            del registers[reg]
        # The PIC thunk writes the PC into the named register, even if that
        # register is normally callee-saved.
        if isa == "i686" and target and (thunk := I686_PC_THUNK.fullmatch(target)):
            registers.pop(X86_REGISTERS[f"%e{thunk[1]}"], None)
        return
    if op in TAILS or op in {"ret", "retq", "retl"} or X86_IMPLICIT_WRITE.fullmatch(op):
        registers.clear()
        return
    source, _, dest = operands.rpartition(", ")
    symbol = None
    if op == ("movl" if isa == "i686" else "movq"):
        got = X86_GOT[isa].fullmatch(source)
        symbol = got[1] if got else registers.get(X86_REGISTERS.get(source))
    if op == ("leal" if isa == "i686" else "leaq"):
        address = X86_ADDRESS[isa].fullmatch(source)
        symbol = address[1] if address else None
    # Forget any explicitly used register unless a recognized full-width
    # assignment recovers its value. This also catches writes to subregisters
    # and instructions with more than one output.
    for operand in re.findall(r"%[a-z][a-z0-9]*", line):
        registers.pop(X86_REGISTERS.get(operand), None)
    full_width = dest.startswith("%e") if isa == "i686" else X86_REGISTERS.get(dest) == dest
    if symbol and full_width and dest in X86_REGISTERS:
        registers[X86_REGISTERS[dest]] = symbol


def instructions(isa, body, aliases):
    registers = {}
    for line in body:
        if line.endswith(":"):
            # Joins may have other predecessors with different register values.
            registers.clear()
            continue
        op, _, operands = line.partition(" ")
        target = None
        indirect = op in INDIRECT_CALLS or (isa == "aarch64" and op in {"br", "braa", "brab"})
        if op in DIRECT_CALLS | TAILS:
            if isa in X86_ISAS and operands.startswith("*"):
                indirect = True
                got = X86_GOT[isa].fullmatch(operands[1:])
                target = got[1] if got else registers.get(X86_REGISTERS.get(operands[1:]))
            elif match := DIRECT_SYMBOL.fullmatch(operands):
                symbol = match[1]
                if symbol in aliases or (not symbol.startswith((".L", "$")) and not symbol.isdigit()):
                    target = symbol
        names = alias_chain(target, aliases) if target else ()
        if isa in X86_ISAS and (registers or op in {"movq", "leaq", "movl", "leal"}):
            # Transfer consumes the pre-call value before applying clobbers.
            x86_register_update(isa, line, opcode(line), operands,
                                names[-1] if names else None, registers)
        yield line, names, indirect and target is None


def bucket_for(isa, line, target, local):
    op = opcode(line)
    if op in TRAPS or (target and PANIC.search(target)):
        return "panic_or_trap"
    if op in CALLS or (op in TAILS and target):
        if isa == "i686" and target and I686_PC_THUNK.fullmatch(target):
            return "call_internal"
        if target and (SOFT_FMA.search(target) or HELPER.search(target) or target not in local):
            return "call_helper"
        return "call_internal"
    if BRANCH.fullmatch(op):
        return "branch"
    return next((bucket for bucket, rule in COST_RULES[isa] if rule.search(line)), "alu")


def parse_assembly(isa, text):
    names = set(FUNCTION.findall(text))
    bodies, aliases = {}, {}
    comment = "//" if isa == "aarch64" or isa in WASM_ISAS else "#"
    current = None
    for raw in text.splitlines():
        line = raw.split(comment, 1)[0].strip()
        if not line:
            continue
        if match := ALIAS.fullmatch(line):
            aliases[match[1] or match[2]] = match[3]
        elif line.endswith(":"):
            name = line[:-1]
            if name in names:
                current = []
                bodies[name] = current
            elif current is not None:
                current.append(line)
        elif line.startswith(".size") or line == "end_function":
            current = None
        elif current is not None:
            if line.startswith((".byte", ".long", ".word", ".inst", ".short", ".quad")):
                raise ValueError(f"unclassified inline data/instruction: {raw}")
            if not line.startswith((".", "#")):
                current.append(" ".join(line.split()))
    if not bodies:
        raise ValueError(f"no functions found in {isa} assembly")
    return bodies, aliases


def elf_sizes(path):
    sizes = {}
    for line in command(["readelf", "--wide", "--syms", path]).splitlines():
        fields = line.split()
        if len(fields) == 8 and fields[3] == "FUNC" and fields[6] != "UND":
            if n := int(fields[2], 0):
                sizes[fields[7]] = n
    return sizes


class WasmReader:
    def __init__(self, data):
        self.data = memoryview(data)
        self.pos = 0

    def __bool__(self):
        return self.pos < len(self.data)

    def take(self, size):
        end = self.pos + size
        if end > len(self.data):
            raise ValueError("truncated WASM section")
        data = self.data[self.pos:end]
        self.pos = end
        return data

    def byte(self):
        return self.take(1)[0]

    def uint(self):
        result = 0
        for shift in range(0, 35, 7):
            byte = self.byte()
            result |= (byte & 127) << shift
            if byte < 128:
                return result
        raise ValueError("invalid WASM u32 LEB128")

    def name(self):
        return self.take(self.uint()).tobytes().decode()

    def section(self):
        return self.byte(), WasmReader(self.take(self.uint()))


def wasm_sizes(path):
    reader = WasmReader(path.read_bytes())
    if reader.take(8) != b"\0asm\x01\x00\x00\x00":
        raise ValueError(f"{path}: expected a linked WASM module")
    names, exports = {}, {}
    lengths = []
    while reader:
        kind, section = reader.section()
        if kind == 0 and section.name() == "name":
            while section:
                subsection_id, subsection = section.section()
                if subsection_id == 1:  # function name map
                    for _ in range(subsection.uint()):
                        index = subsection.uint()
                        names[subsection.name()] = index
        elif kind == 2 and section.uint():
            # Standalone no_std probes should link without any host imports.
            # Refuse to misidentify defined-body indexes if that ever changes.
            raise ValueError("unexpected imports in linked WASM probe")
        elif kind == 7:
            for _ in range(section.uint()):
                name, export_kind, index = section.name(), section.byte(), section.uint()
                if export_kind == 0:
                    exports[name] = index
        elif kind == 10:
            for _ in range(section.uint()):
                size = section.uint()
                section.take(size)
                lengths.append(size)
            if section:
                raise ValueError("trailing bytes in WASM code section")
    return {name: lengths[index] for name, index in (names | exports).items()}


def reachable(isa, roots, bodies, aliases):
    live = {}
    todo = [alias_chain(name, aliases)[-1] for name in roots]
    while todo:
        name = todo.pop()
        if name in live:
            continue
        if name not in bodies:
            raise ValueError(f"missing assembly body: {name}")
        live[name] = body = list(instructions(isa, bodies[name], aliases))
        todo.extend(names[-1] for _, names, _ in body if names and names[-1] in bodies)
    return live


def invariant_results(isa, name, decoded):
    # Match each edge once even when more than one alias has a forbidden name.
    def matching_calls(pattern):
        return sum(any(pattern.search(symbol) for symbol in names) for _, names, _ in decoded)

    lines = [line for line, _, _ in decoded]
    results = []
    if name.startswith(("log1p_", "cbrtf_")):
        results.append(["float_div", sum(bool(FLOAT_DIV.match(line) or
                                               (isa == "i686" and X87_DIV.match(line)))
                                         for line in lines)])
    if isa == "aarch64" and name == "powf_unit_unchecked":
        results.extend([
            ["byte_lane_move", sum(bool(BYTE_LANE_MOVE.match(line)) for line in lines)],
            ["quad_pair", sum(bool(QUAD_PAIR.match(line)) for line in lines)],
        ])
    if isa in WASM_ISAS and "backend" in name:
        results.append(["soft_fma", matching_calls(SOFT_FMA)])
        if isa == "wasm32_relaxed" and not name.endswith("_scalar"):
            results.append(["relaxed_madd", sum(bool(WASM_RELAXED_MADD.match(line)) for line in lines)])
    if name in SCALAR[:6]:
        results.append(["bounds_panic", sum(any(PANIC.search(symbol) for symbol in names or (line,))
                                           for line, names, _ in decoded)])
    if name in ("powf_highp_v3", "powf_highp_v4x"):
        results.append(["call_mut", matching_calls(CALL_MUT)])
    if name.startswith("scalar_sqrt_"):
        f32 = name.endswith("f32")
        native = {
            "x86_64": {"sqrtss", "vsqrtss"} if f32 else {"sqrtsd", "vsqrtsd"},
            "aarch64": {"fsqrt"},
            # LLVM can fold a splat, SIMD sqrt, and lane-zero extract into
            # the equally native scalar WASM sqrt opcode.
            "wasm32": {"f32.sqrt", "f32x4.sqrt"} if f32 else {"f64.sqrt", "f64x2.sqrt"},
        }["wasm32" if isa in WASM_ISAS else isa]
        results.extend([
            ["hardware_sqrt", sum(opcode(line) in native for line in lines)],
            ["goldschmidt", matching_calls(GOLDSCHMIDT)],
        ])
    return results


def measure(isa, name, roots, bodies, aliases, sizes):
    live = reachable(isa, roots, bodies, aliases)
    decoded = [insn for body in live.values() for insn in body]
    if not decoded:
        raise ValueError(f"{name}: empty assembly")
    code_bytes = sum(sizes[callee] for callee in live)
    buckets = Counter(bucket_for(isa, line, names[-1] if names else None, bodies)
                      for line, names, _ in decoded)
    results = invariant_results(isa, name, decoded)
    metrics = {
        "code_bytes": code_bytes,
        "insn_count": len(decoded),
        "cost_score": sum(WEIGHTS[key] * count for key, count in buckets.items()),
        "buckets": dict(sorted(buckets.items())),
        "invariants": results,
    }
    if results or name in HIGHP:
        # A missing callee can conceal a forbidden instruction or skew size.
        # Only failures need diagnostics in the compact baseline format.
        unresolved = [f"{callee}: {line}" for callee, body in live.items()
                      for line, _, indirect in body if indirect]
        if unresolved:
            metrics["unresolved_indirect_edges"] = unresolved
    return metrics


def require_invariants(probes):
    errors = []
    for key, metrics in probes.items():
        for edge in metrics.get("unresolved_indirect_edges", ()):
            errors.append(f"{key}: unresolved indirect edge in {edge}")
        for invariant, actual in metrics["invariants"]:
            if invariant in {"quad_pair", "hardware_sqrt", "relaxed_madd"}:
                passes, expected = actual >= 1, ">= 1"
            elif invariant == "gamma_ratio":
                passes, expected = actual <= 1.6, "<= 1.6"
            else:
                passes, expected = actual == 0, "= 0"
            if not passes:
                errors.append(f"{key}: {invariant} {actual} (expected {expected})")
    return errors


def command(args, **kwargs):
    result = subprocess.run(args, check=False, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise ValueError(f"{' '.join(map(str, args))}\n{result.stdout}{result.stderr}")
    return result.stdout


def source_ref(ref, dest):
    if ref in (".", "WORKTREE"):
        return ROOT
    git = ["git"]
    if not (ROOT / ".git").exists() and (ROOT / ".jj").is_dir():
        git += ["--git-dir", command(["jj", "git", "root"], cwd=ROOT).strip()]
    # jj spells remote bookmarks name@remote. Git uses remote/name.
    if "@" in ref and not ref.startswith("@") and "@{" not in ref:
        name, remote = ref.rsplit("@", 1)
        ref = f"refs/remotes/{remote}/{name}"
    archive = subprocess.check_output([*git, "archive", ref], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    return dest


def build_cell(side, source, directory, isa, opt, toolchain, dump):
    directory.mkdir(parents=True)
    (directory / "src").mkdir()
    shutil.copy(FIXTURE, directory / "src/lib.rs")
    shutil.copy(source / "Cargo.lock", directory / "Cargo.lock")
    deps = {"archmage": source, "magemath": source / "magemath",
            "magetypes": source / "magetypes"}
    dependencies = "\n".join(
        f'{name} = {{ path = {json.dumps(str(path))}, default-features = false'
        + (', features = ["w512"]' if name == "magetypes" else "") + " }"
        for name, path in deps.items()
    )
    (directory / "Cargo.toml").write_text(f"""\
[package]
name = "math_guard_probe"
version = "0.0.0"
edition = "2024"

[lib]
crate-type = ["cdylib"]

[dependencies]
{dependencies}

[features]
gamma-four = []
avx512 = ["magemath/avx512"]

[profile.release]
opt-level = {json.dumps(opt) if opt == 'z' else opt}
panic = "abort"
codegen-units = 1
lto = "fat"

[workspace]
""")
    target = TARGETS[isa]
    env = os.environ.copy()
    for variable in (
        "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC", "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_TARGET", "CARGO_BUILD_RUSTFLAGS",
        f"CARGO_TARGET_{target.upper().replace('-', '_')}_RUSTFLAGS",
    ):
        env.pop(variable, None)
    for variable in dict(env):
        if variable.startswith("CARGO_PROFILE_RELEASE_"):
            env.pop(variable)
    env.update(
        TMPDIR=str(directory), CARGO_TARGET_DIR=str(directory / "target"),
        CARGO_BUILD_BUILD_DIR=str(directory / "target"),
        CARGO_INCREMENTAL="0",
    )
    if isa in WASM_ISAS:
        relaxed = "+" if isa == "wasm32_relaxed" else "-"
        env["RUSTFLAGS"] = f"-Ctarget-feature=+simd128,{relaxed}relaxed-simd"
    shared = {}
    if isa == "wasm32_relaxed":
        names = (*COMMON, *MATH, *RELAXED_LOG1P)
    elif isa == "i686":
        names = I686_PROBES
    else:
        names = COMMON if opt == "z" else (*COMMON, *MATH)
    if opt == "3" and isa == "x86_64":
        names += ("powf_highp_v3", "powf_highp_v4x")
    for four in (False, True):
        variant = "four" if four else "one"
        asm = directory / f"{variant}.s"
        obj = directory / f"{variant}.o"
        flags = [f"--emit=asm={asm},link"] if isa in WASM_ISAS else [
            "-Clinker=rust-lld", "-Clinker-flavor=ld.lld", f"--emit=asm={asm},obj={obj}",
        ]
        features = ["avx512"] if isa == "x86_64" else []
        if four:
            features.append("gamma-four")
        args = ["cargo", toolchain, "rustc", "--release", "--lib", "--target", target, "-j", "2"]
        if features:
            args.extend(("--features", ",".join(features)))
        command([*args, "--", *flags], cwd=directory, env=env)
        assembly = asm.read_text()
        bodies, aliases = parse_assembly(isa, assembly)
        sizes = (wasm_sizes(directory / f"target/{target}/release/math_guard_probe.wasm")
                 if isa in WASM_ISAS else elf_sizes(obj))
        if dump:
            out = dump / side / (isa if isa == "wasm32_relaxed" else target) / f"opt-{opt}"
            out.mkdir(parents=True, exist_ok=True)
            # Canonical whitespace, omit comments and non-code directives, retain
            # operands and function attribution for human review.
            (out / f"gamma-{variant}.s").write_text("".join(
                f"{name}:\n" + "".join(f"\t{line}\n" for line in body)
                for name, body in bodies.items()
            ) + "".join(f"{name} = {dest}\n" for name, dest in aliases.items()))
        for name in ([HIGHP[1]] if four else (n for n in names if n != HIGHP[1])):
            roots = (tuple(f"powf_highp_gamma_{i}" for i in range(4 if four else 1))
                     if name in HIGHP else (name,))
            shared[name] = measure(isa, name, roots, bodies, aliases, sizes)
    ratio = shared[HIGHP[1]]["code_bytes"] / shared[HIGHP[0]]["code_bytes"]
    shared[HIGHP[1]]["invariants"].append(["gamma_ratio", ratio])
    return {f"{isa}/opt-{opt}/{key}": row for key, row in sorted(shared.items())}


def collect(refs, scratch, toolchain, dump):
    sources = {side: source_ref(ref, scratch / f"{side}-src") for side, ref in refs.items()}
    tasks = [
        (side, source, scratch / side / f"{isa}-{opt}", isa, opt, toolchain, dump)
        for side, source in sources.items() for isa in TARGETS for opt in ("3", "z")
    ]
    batches = {side: {} for side in sources}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for task, probes in zip(tasks, pool.map(lambda args: build_cell(*args), tasks)):
            batches[task[0]].update(probes)
            print(f"{task[0]} {task[3]} opt-{task[4]}: {len(probes)} probes", flush=True)
    return batches


def metadata(toolchain):
    info = command(["rustc", toolchain, "-vV"]).splitlines()
    return {
        "schema_version": SCHEMA_VERSION,
        "cost_table_version": COST_TABLE_VERSION,
        "rustc_version": info[0],
        "llvm_version": next(line.split(": ", 1)[1] for line in info if line.startswith("LLVM version: ")),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def baseline_text(data):
    # Share the field names across probes. A dense bucket vector is shorter
    # than repeating dictionary keys, even when some counts are zero.
    columns = ("code_bytes", "insn_count", "cost_score", "buckets", "invariants")
    header = {
        **{key: value for key, value in data.items() if key != "probes"},
        "probe_columns": columns,
        "bucket_columns": tuple(WEIGHTS),
    }
    summary = json.dumps(header, indent=2).removesuffix("\n}")
    rows = []
    for key, row in data["probes"].items():
        counts = [row["buckets"].get(bucket, 0) for bucket in WEIGHTS]
        cells = [row["code_bytes"], row["insn_count"], row["cost_score"], counts, row["invariants"]]
        rows.append("    " + json.dumps(key) + ":" + json.dumps(cells, separators=(",", ":")))
    return summary + ',\n  "probes": {\n' + ",\n".join(rows) + "\n  }\n}\n"


def baseline_probes(data):
    probes = {}
    for key, cells in data["probes"].items():
        row = dict(zip(data["probe_columns"], cells, strict=True))
        counts = zip(data["bucket_columns"], row["buckets"], strict=True)
        row["buckets"] = {bucket: count for bucket, count in counts if count}
        probes[key] = row
    return probes


def check_versions(before, after):
    for key in ("schema_version", "cost_table_version"):
        if before[key] != after[key]:
            raise ValueError(f"{key} mismatch; review and update the baseline")


def drift_tolerance(before, after, percent):
    keys = ("rustc_version", "llvm_version")
    if any(before[key] != after[key] for key in keys):
        old = f'{before["rustc_version"]}, LLVM {before["llvm_version"]}'
        new = f'{after["rustc_version"]}, LLVM {after["llvm_version"]}'
        print(f"Compiler drift: {old} → {new}. Percentage-only tolerance {percent:g}%. "
              "Hard invariants unchanged.")
        return percent
    return None


def comparison(before, after, *, drift_pct=None, quiet=False):
    issues = require_invariants(after)
    if before.keys() != after.keys():
        issues.append(f"missing probes: {sorted(before.keys() - after.keys())}; "
                      f"new probes: {sorted(after.keys() - before.keys())}")
    if not quiet:
        print("| Probe | Code bytes before → after (Δ) | Instructions Δ | Cost before → after (Δ) | Bucket Δ |")
        print("|---|---:|---:|---:|---|")
    for name in sorted(before.keys() & after.keys()):
        old, new = before[name], after[name]
        for field, floor in (("cost_score", 5), ("code_bytes", 64)):
            growth = new[field] - old[field]
            budget = (max(floor, old[field] * 0.05) if drift_pct is None
                      else old[field] * drift_pct / 100)
            if growth > budget:
                issues.append(f"{name}: {field} +{growth:g} ({old[field]:g} → {new[field]:g})")
        if not quiet:
            changes = Counter(new["buckets"])
            changes.subtract(old["buckets"])
            buckets = ", ".join(f"{count:+} {key}" for key, count in sorted(changes.items()) if count)
            byte_delta = new["code_bytes"] - old["code_bytes"]
            insn_delta = new["insn_count"] - old["insn_count"]
            cost_delta = new["cost_score"] - old["cost_score"]
            print(f'| {name} | {old["code_bytes"]} → {new["code_bytes"]} ({byte_delta:+}) | '
                  f'{insn_delta:+} | {old["cost_score"]:g} → {new["cost_score"]:g} ({cost_delta:+g}) | '
                  f'{buckets or "—"} |')
    return issues


class GuardTests(unittest.TestCase):
    def score(self, isa, name, body):
        assembly = f"\t.type {name},@function\n{name}:\n{body}\n\t.size {name}, .-{name}\n"
        bodies, aliases = parse_assembly(isa, assembly)
        return measure(isa, name, (name,), bodies, aliases, {name: 1000})

    def compare_shift(self, growth):
        old = self.score("x86_64", "ordinary", "\taddl %eax, %ebx\n" * 100)
        new = self.score("x86_64", "ordinary", "\taddl %eax, %ebx\n" * (100 + growth))
        new["code_bytes"] = 1000 + 10 * growth
        base = {"rustc_version": "rustc 1.100", "llvm_version": "23.1.5"}
        current = {**base, "llvm_version": "24.1.0"}
        output = io.StringIO()
        with redirect_stdout(output):
            relaxed = drift_tolerance(base, current, 15)
        self.assertIn("23.1.5", output.getvalue())
        self.assertIn("24.1.0", output.getvalue())
        return [comparison({"probe": old}, {"probe": new}, drift_pct=percent, quiet=True)
                for percent in (None, relaxed)]

    def test_register_renames_and_reordering_preserve_cost(self):
        first = self.score("x86_64", "probe", "\tvaddps %ymm0, %ymm1, %ymm2\n"
                          "\tvmovups (%rsi), %ymm3\n\tretq")
        second = self.score("x86_64", "probe", "\tvmovups (%rdi), %ymm9\n"
                           "\tvaddps %ymm4, %ymm6, %ymm1\n\tretq")
        self.assertEqual(first, second)
        self.assertEqual(first["cost_score"], 6.0)

    def test_same_toolchain_rejects_eight_percent_drift_accepts_it(self):
        strict, relaxed = self.compare_shift(8)
        self.assertEqual(len(strict), 2)
        self.assertEqual(relaxed, [])

    def test_different_llvm_still_rejects_thirty_percent(self):
        strict, relaxed = self.compare_shift(30)
        self.assertEqual(len(strict), 2)
        self.assertEqual(len(relaxed), 2)

    def test_drift_budget_has_no_absolute_floor(self):
        old = self.score("x86_64", "scalar_round", "\tretq")
        old.update(code_bytes=113, cost_score=20)
        new = {**old, "code_bytes": 153, "cost_score": 24}
        self.assertEqual(comparison({"round": old}, {"round": new}, quiet=True), [])
        issues = comparison({"round": old}, {"round": new}, drift_pct=15, quiet=True)
        self.assertEqual(issues, [
            "round: cost_score +4 (20 → 24)",
            "round: code_bytes +40 (113 → 153)",
        ])
        boundary = {**old, "code_bytes": 129, "cost_score": 23}
        self.assertEqual(comparison({"round": old}, {"round": boundary}, drift_pct=15,
                                    quiet=True), [])

    def test_hard_injections_fail_even_with_drift(self):
        cases = (
            ("x86_64", "log1p_around", "vdivps %ymm0, %ymm1, %ymm2", "float_div"),
            ("aarch64", "powf_unit_unchecked", "mov.b v0[0], w0", "byte_lane_move"),
            ("aarch64", "powf_unit_unchecked", "mov v0.b[0], w0", "byte_lane_move"),
            ("x86_64", "scalar_transpose", "callq core::panicking::panic_fmt", "bounds_panic"),
            ("wasm32", "cbrtf_midp_backend", "call _RNv8soft_fma9fmaf_soft", "soft_fma"),
            ("x86_64", "powf_highp_v3", "callq _RNv5FnMut8call_mut", "call_mut"),
            ("wasm32", "scalar_sqrt_f32", "call goldschmidt", "goldschmidt"),
        )
        for isa, name, injected, invariant in cases:
            with self.subTest(invariant=invariant, injection=injected):
                clean = self.score(isa, name, "\tldp q0, q1, [x0]\n\tf32x4.sqrt")
                broken = self.score(isa, name, f"\tldp q0, q1, [x0]\n\tf32x4.sqrt\n\t{injected}")
                errors = comparison({name: clean}, {name: broken}, drift_pct=15, quiet=True)
                self.assertTrue(any(f"{invariant} 1" in error for error in errors), errors)
        one = self.score("x86_64", HIGHP[0], "\tretq")
        four = {**one, "code_bytes": 4 * one["code_bytes"], "invariants": [["gamma_ratio", 4.0]]}
        errors = comparison({HIGHP[1]: one}, {HIGHP[1]: four}, drift_pct=15, quiet=True)
        self.assertTrue(any("gamma_ratio 4.0" in error for error in errors))

    def test_aliases_and_shared_transitive_callees_count_once(self):
        bodies = {"wrapper": ["callq child", "jmp child"], "child": ["retq"]}
        row = measure("x86_64", "test", ("wrapper", "alias"), bodies, {"alias": "child"},
                      {"wrapper": 12, "child": 1})
        self.assertEqual(row["code_bytes"], 13)
        self.assertEqual(row["buckets"], {"branch": 1, "call_internal": 2})

    def test_packed_wasm_software_fma_symbols_are_forbidden(self):
        for symbol in (
            "_RNvNtCsabcdef_8magemath8wasm_fma15powf_soft_f32x4",
            "_RINvNtCsabcdef_8magemath8wasm_fma14software_lanesKj8_E",
        ):
            with self.subTest(symbol=symbol):
                name = "powf_midp_backend"
                row = self.score("wasm32", name, f"\tcall {symbol}")
                self.assertEqual(row["buckets"], {"call_helper": 1})
                self.assertEqual(require_invariants({name: row}), [f"{name}: soft_fma 1 (expected = 0)"])

    def test_relaxed_wasm_requires_madd_without_software_fma(self):
        name = "log1p_greater_backend"
        for op in ("f32x4.relaxed_madd", "f32x4.relaxed_nmadd"):
            row = self.score("wasm32_relaxed", name, f"\t{op}")
            self.assertEqual(row["buckets"], {"fma": 1})
            self.assertEqual(require_invariants({name: row}), [])
        unfused = self.score("wasm32_relaxed", name, "\tf32x4.mul\n\tf32x4.add")
        self.assertEqual(require_invariants({name: unfused}), [f"{name}: relaxed_madd 0 (expected >= 1)"])
        software = self.score("wasm32_relaxed", name, "\tf32x4.relaxed_madd\n\tcall powf_soft_f32x4")
        self.assertEqual(require_invariants({name: software}), [f"{name}: soft_fma 1 (expected = 0)"])

    def test_i686_pic_thunk_and_got_targets(self):
        name = "log1p_around_scalar"
        bodies = {
            name: [
                "calll __x86.get_pc_thunk.bx",
                "addl $_GLOBAL_OFFSET_TABLE_, %ebx",
                "calll *divide@GOT(%ebx)",
                "movl divide@GOT(%ebx), %esi",
                "calll *%esi",
                "retl",
            ],
            "divide": ["fdivrp %st, %st(1)", "retl"],
        }
        row = measure("i686", name, (name,), bodies, {}, {name: 23, "divide": 3})
        self.assertEqual(row["code_bytes"], 26)
        self.assertEqual(row["buckets"], {"alu": 1, "branch": 2, "call_internal": 3,
                                          "hw_div_sqrt": 1, "mem": 1})
        self.assertEqual(require_invariants({name: row}), [f"{name}: float_div 1 (expected = 0)"])
        name = "scalar_transpose"
        panic = self.score("i686", name, "\tcalll *_RNv4core9panicking9panic_fmt@GOT(%ebx)")
        self.assertEqual(require_invariants({name: panic}), [f"{name}: bounds_panic 1 (expected = 0)"])

    def test_compact_baseline_round_trips_measurement_rows(self):
        name = "log1p_greater_backend"
        row = self.score("wasm32_relaxed", name, "\tf32x4.relaxed_madd\n\tret")
        probes = {f"wasm32_relaxed/opt-z/{name}": row}
        compact = json.loads(baseline_text({"schema_version": SCHEMA_VERSION, "probes": probes}))
        self.assertEqual(baseline_probes(compact), probes)

    def test_got_calls_reach_divide_helper(self):
        name = "log1p_around"
        helpers = {"divide": ["divss %xmm1, %xmm0", "retq"]}
        sizes = {name: 16, "divide": 5}
        for caller in (
            ["callq *divide_alias@GOTPCREL(%rip)"],
            ["movq divide_alias@GOTPCREL(%rip), %r15", "callq *%r15", "callq *%r15"],
            ["movq divide_alias@GOTPCREL(%rip), %r15", "movq %r15, %rax", "jmpq *%rax"],
        ):
            with self.subTest(caller=caller):
                row = measure("x86_64", name, (name,), {**helpers, name: caller},
                              {"divide_alias": "divide"}, sizes)
                self.assertEqual(row["code_bytes"], 21)
                self.assertEqual(require_invariants({name: row}), [f"{name}: float_div 1 (expected = 0)"])

    def test_alias_targets_are_checked_forbidden_symbols(self):
        name = "powf_highp_v3"
        symbol = "_RNv5FnMut8call_mut"
        bodies, aliases = parse_assembly("x86_64", f"""
            .type {name},@function
        {name}:
            callq first
            .size {name}, .-{name}
            .type {symbol},@function
        {symbol}:
            retq
            .size {symbol}, .-{symbol}
            .set first, second
            second = {symbol}
        """)
        row = measure("x86_64", name, (name,), bodies, aliases, {name: 5, symbol: 1})
        self.assertEqual(row["code_bytes"], 6)
        self.assertEqual(require_invariants({name: row}), [f"{name}: call_mut 1 (expected = 0)"])

    def test_reverse_aliases_cannot_hide_forbidden_symbols(self):
        cases = (
            ("x86_64", "powf_highp_v3", "callq", "retq", "_RNv5FnMut8call_mut", "call_mut"),
            ("wasm32", "powf_midp_backend", "call", "return", "powf_soft_f32x4", "soft_fma"),
        )
        for isa, name, op, ret, forbidden, invariant in cases:
            for called in (forbidden, ".Louter"):
                with self.subTest(isa=isa, called=called):
                    bodies, aliases = parse_assembly(isa, f"""
                        .type {name},@function
                    {name}:
                        {op} {called}
                        {op} canonical_impl
                        .size {name}, .-{name}
                        .type canonical_impl,@function
                    canonical_impl:
                        {ret}
                        .size canonical_impl, .-canonical_impl
                        .set .Louter, {forbidden}
                        .set {forbidden}, canonical_impl
                    """)
                    sizes = {name: 10, "canonical_impl": 1}
                    row = measure(isa, name, (name,), bodies, aliases, sizes)
                    self.assertEqual(row["code_bytes"], 11)
                    self.assertEqual(require_invariants({name: row}),
                                     [f"{name}: {invariant} 1 (expected = 0)"])

    def test_unresolved_indirect_edges_report_instruction_and_function(self):
        cases = (
            ("x86_64", "", "jmpq *%r15"),
            ("x86_64", "\tmovq memcpy@GOTPCREL(%rip), %r15\n\tmovb $0, %r15b\n",
             "callq *%r15"),
            ("x86_64", "\tjz .Ljoin\n\tmovq memcpy@GOTPCREL(%rip), %r15\n.Ljoin:\n",
             "callq *%r15"),
            ("x86_64", "\tmovq memcpy@GOTPCREL(%rip), %rax\n\tcallq other\n",
             "callq *%rax"),
            ("aarch64", "", "blr x8"),
            ("aarch64", "", "br x8"),
            ("wasm32", "", "call_indirect () -> ()"),
        )
        for isa, setup, call in cases:
            with self.subTest(isa=isa, setup=setup, call=call):
                name = "log1p_around"
                row = self.score(isa, name, f"{setup}\t{call}")
                expected = f"{name}: unresolved indirect edge in {name}: {call}"
                self.assertEqual(require_invariants({name: row}), [expected])
    def test_missing_required_native_instruction_is_a_hard_failure(self):
        for isa, name in (("aarch64", "powf_unit_unchecked"), ("wasm32", "scalar_sqrt_f64")):
            row = self.score(isa, name, "\tret")
            self.assertEqual(len(require_invariants({name: row})), 1)

    def test_native_wasm_scalar_sqrt_and_neon_tail_extract(self):
        sqrt = self.score("wasm32", "scalar_sqrt_f32", "\tlocal.get 0\n\tf32.sqrt")
        arm = self.score("aarch64", "powf_unit_unchecked", "\tldp q0, q1, [x0]\n\tmov b0, v1.b[15]")
        self.assertEqual(require_invariants({"sqrt": sqrt, "arm": arm}), [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--compare", nargs=2, metavar=("BASE_REF", "HEAD_REF"))
    modes.add_argument("--check-baseline", nargs="?", const=BASELINE, type=Path, metavar="PATH")
    modes.add_argument("--update-baseline", nargs="?", const=BASELINE, type=Path, metavar="PATH")
    modes.add_argument("--self-test", action="store_true")
    parser.add_argument("--dump-asm", type=Path, metavar="DIR")
    parser.add_argument("--llvm-drift-pct", type=float, default=15.0)
    args = parser.parse_args()
    if args.self_test:
        return 0 if unittest.main(argv=[sys.argv[0]], exit=False, verbosity=2).result.wasSuccessful() else 1
    if not math.isfinite(args.llvm_drift_pct) or args.llvm_drift_pct < 0:
        parser.error("--llvm-drift-pct must be finite and nonnegative")
    tmp = Path.home() / "tmp"
    tmp.mkdir(exist_ok=True)
    dump = args.dump_asm.expanduser().resolve() if args.dump_asm else None
    if dump and not dump.is_relative_to(tmp.resolve()):
        parser.error("--dump-asm must be under ~/tmp")
    start = monotonic()
    toolchain = "+" + os.environ.get("RUSTUP_TOOLCHAIN", "stable")
    compiler = metadata(toolchain)
    print(f'{compiler["rustc_version"]}, LLVM {compiler["llvm_version"]}', flush=True)
    with tempfile.TemporaryDirectory(prefix="magemath-codegen-", dir=tmp) as directory:
        scratch = Path(directory)
        if args.compare:
            batches = collect(dict(zip(("base", "head"), args.compare)), scratch, toolchain, dump)
            # A PR can fix a hard violation in its base. Enforce contracts on
            # the checked revision, and surface any pre-existing violations.
            for issue in require_invariants(batches["base"]):
                print(f"BASE: {issue}")
            issues = comparison(batches["base"], batches["head"])
        else:
            probes = collect({"worktree": "."}, scratch, toolchain, dump)["worktree"]
            current = {**compiler, "probes": probes}
            if args.check_baseline:
                old = json.loads(args.check_baseline.read_text())
                check_versions(old, current)
                drift_pct = drift_tolerance(old, current, args.llvm_drift_pct)
                issues = comparison(baseline_probes(old), probes, drift_pct=drift_pct)
            else:
                issues = require_invariants(probes)
                path = args.update_baseline
                if not issues:
                    previous = path.read_text() if path.exists() else ""
                    updated = baseline_text(current)
                    if path.resolve() == BASELINE.resolve() and len(updated.encode()) >= 30_000:
                        raise ValueError("committed baseline must be under 30,000 bytes")
                    print("".join(difflib.unified_diff(previous.splitlines(True), updated.splitlines(True),
                                                     str(path) + " (old)", str(path) + " (new)")), end="")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(updated)
                    print(f"Wrote {path} ({len(updated.encode())} bytes).")
    for issue in issues:
        print(f"FAIL: {issue}")
    print(f"{len(issues)} regressions or invariant failures in {monotonic() - start:.3f} s")
    return int(bool(issues))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)

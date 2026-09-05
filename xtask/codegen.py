#!/usr/bin/env python3
"""Compare release consumer codegen by ISA, resolving aliases and local callees.

Run after changing a generator, macro, or storage abstraction:
    python3 xtask/codegen.py BASE_REF WORKTREE
Requires Python 3.12+ and stable Rust with x86_64, aarch64 and wasm32 unknown-unknown targets.
Each API escapes through a no-inline exported consumer function so loads and stores
cannot cancel. Immediate operands and branch/call/memory instructions remain in the comparison.
Only narrowly specified, reviewed improvements are accepted below; everything else
fails with before/after instructions. Scratch output goes exclusively under ~/tmp.
"""

import collections
import difflib
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tarfile
import tempfile
import hashlib


def normalize(s):
    s = re.sub(r"^\s*\.file.*\n", "", s, flags=re.M)
    return re.sub(r"Cs[0-9a-zA-Z]+_13storage_probe", "CsHASH_13storage_probe", s)


def parse(s):
    s = re.sub(r"Cs[0-9a-zA-Z]+_", "CsHASH_", s)
    # Resolve local data by contents, not LLVM's unstable label numbers. Keep
    # relocation targets and offsets: identical loads can load different data.
    data = {}
    for m in re.finditer(
        r"^(\.L(?:CPI|anon)[^:\s]+):[^\n]*\n((?:\s*\.(?:byte|short|long|quad|word|xword|zero|ascii|asciz|int8|int16|int32|int64)\s+[^\n]*\n)+)",
        s,
        re.M,
    ):
        alignment = re.search(r"(\.p2align[^\n]*)\n\s*$", s[: m.start()])
        data[m[1]] = (alignment[1] + "\n" if alignment else "") + m[2].strip()

    def data_ref(m, active=()):
        name = m[0]
        assert name in data, ("unparsed constant", name)
        assert name not in active, ("cyclic constant", name)
        contents = re.sub(
            r"\.L(?:CPI|anon)[\w.]+", lambda x: data_ref(x, (*active, name)), data[name]
        )
        return "DATA_" + hashlib.sha256(contents.encode()).hexdigest()

    names = re.findall(r"^\s*\.type\s+(\S+),[@%]function", s, re.M)
    assert names, "No functions parsed; refusing a vacuous comparison"
    funcs = {}
    aliases = {}
    bodies = dict(
        re.findall(
            r"^([A-Za-z_][^:\s]*):[^\n]*\n(.*?)(?:^\s*\.size\s|^\s*end_function)",
            s,
            re.M | re.S,
        )
    )
    for name in names:
        body = bodies.get(name)
        if body is not None:
            # Keep label positions as well as branch operands. Renumbering is
            # harmless; moving a label relative to instructions is not.
            labels = re.findall(r"^(\.LBB\w+):", body, re.M)
            label_ids = {label: f"BLOCK_{i}" for i, label in enumerate(labels)}
            lines = []
            for x in body.splitlines():
                label = x.strip().removesuffix(":")
                if label in label_ids:
                    lines.append("LABEL " + label_ids[label])
                elif re.match(r"^\t[a-z][a-z0-9_.]*(?:\s|$)", x):
                    x = re.sub(r"\.LBB\w+", lambda m: label_ids[m[0]], x.strip())
                    x = re.sub(r"\.L(?:CPI|anon)[\w.]+", data_ref, x)
                    lines.append(x)
                elif x.lstrip().startswith(
                    (".byte", ".long", ".short", ".quad", ".ascii")
                ):
                    raise ValueError(
                        ("inline data/instruction encoding requires review", name, x)
                    )
                elif x.strip() and not x.lstrip().startswith((".", "#", "//")):
                    raise ValueError(("unparsed assembly", name, x))
            assert lines, ("empty function", name)
            funcs[name] = lines
        else:
            m = re.search(
                r"^(?:\s*\.set\s+)?" + re.escape(name) + r"\s*(?:=|,)\s*(\S+)", s, re.M
            )
            if m:
                aliases[name] = m[1]
            else:
                raise ValueError(name)

    def resolve(n):
        return funcs[n] if n in funcs else resolve(aliases[n])

    return {n: resolve(n) for n in names}


def opcode(line):
    return line.split()[0]


def memory_accesses(body, target):
    if target.startswith("x86"):
        return [
            (opcode(x), re.findall(r"\([^)]*\)", x))
            for x in body
            if "(" in x and not opcode(x).startswith("lea")
        ]
    if target.startswith("aarch64"):
        return [(opcode(x), re.findall(r"\[[^]]*\]", x)) for x in body if "[" in x]
    return [x for x in body if ".load" in opcode(x) or ".store" in opcode(x)]


def transfers(body):
    return [
        x
        for x in body
        if opcode(x)
        in [
            "call",
            "callq",
            "call_indirect",
            "call_ref",
            "bl",
            "blr",
            "b",
            "jmp",
            "br",
            "br_if",
            "br_table",
        ]
        or opcode(x).startswith("j")
        or opcode(x).startswith("b.")
        or opcode(x).startswith(("cbz", "cbnz", "tbz", "tbnz"))
    ]


def normalized(body):
    return [
        re.sub(
            r"\.Lanon\.[a-f0-9]+\.",
            ".Lanon.HASH.",
            re.sub(
                r"\.L(?:BB|CPI)\d+_",
                lambda m: ".LBB_" if m[0].startswith(".LBB") else ".LCPI_",
                x,
            ),
        )
        for x in body
    ]


def expand(name, functions, depth=0):
    assert depth < 10
    result = []
    for line in normalized(functions[name]):
        op = opcode(line)
        if (
            op in ["callq", "call", "bl", "jmp", "jmpq", "b", "return_call"]
            and line.split()[-1] in functions
        ):
            marker = "INTERNAL_CALL" if op in ["callq", "call", "bl"] else "TAIL_CALL"
            result += [marker] + expand(line.split()[-1], functions, depth + 1)
        else:
            result.append(line)
    return result


def aligned_store_equivalent(before, after, name):
    if "from_bytes" not in name:
        return before == after
    if len(before) != len(after):
        return False
    for a, b in zip(before, after):
        if a == b:
            continue
        # Only stronger destination alignment; never erase a load alignment or offset.
        if (
            re.match(r"v?movups\t", a)
            and b == a.replace("movups", "movaps", 1)
            and (store := re.search(r"%([xyz])mm\d+, (\d*)\(%rdi\)$", a))
            and int(store[2] or 0) % {"x": 16, "y": 32, "z": 64}[store[1]] == 0
        ):
            continue
        if (
            a.startswith("v128.store\t")
            and a.endswith(":p2align=0")
            and b == a.removesuffix(":p2align=0")
        ):
            continue
        return False
    return True


def neon_fma_dataflow(body):
    # Symbolically track the straight-line paired loads/FMA/stores. Preserve
    # operand order (including NaN selection), lane shape, addresses and offsets.
    registers = {}
    stores = []
    accesses = []
    for line in body:
        if line == "ret":
            continue
        pair = re.fullmatch(
            r"(ldp|stp)\s+q(\d+), q(\d+), \[(x\d+)(?:, #(\d+))?\]", line
        )
        if pair:
            op, a, b, base, offset = pair.groups()
            offset = int(offset or 0)
            for register, delta in [(a, 0), (b, 16)]:
                address = (base, offset + delta)
                accesses.append((op, address))
                if op == "ldp":
                    registers[register] = address
                else:
                    stores.append((address, registers[register]))
            continue
        fma = re.fullmatch(r"fmla\s+v(\d+)\.(\w+), v(\d+)\.\2, v(\d+)\.\2", line)
        if not fma:
            return None
        dest, shape, left, right = fma.groups()
        registers[dest] = (
            "fma",
            shape,
            registers[dest],
            registers[left],
            registers[right],
        )
    return stores, sorted(accesses)


def partition_dataflow(body, target):
    """Interpret only the straight-line integer subset used by as_chunks.

    Equal opcode counts are insufficient: retain masks, offsets, shift counts,
    return values and stores. Unknown instructions require a fresh review.
    """
    registers = {}
    stores = []

    def mask(value, bits):
        if isinstance(value, int):
            return value & bits
        if isinstance(value, tuple) and value[0] == "and":
            return mask(value[1], value[2] & bits)
        return value if bits == (1 << 64) - 1 else ("and", value, bits)

    def reg(name):
        return name.replace("%e", "%r") if name.startswith("%e") else name

    def get(name):
        if name.startswith(("$", "#")):
            return int(name[1:], 0)
        return registers.get(reg(name), reg(name))

    def put(name, value):
        registers[reg(name)] = (
            mask(value, (1 << 32) - 1) if name.startswith("%e") else value
        )

    def address(base, offset=0):
        return (get(base), int(offset or 0))

    for line in body:
        if line in ("ret", "retq"):
            continue
        if target.startswith("x86"):
            line = re.sub(r"^shrq\s+(%\w+)$", r"shrq $1, \1", line)
            m = re.fullmatch(
                r"(movq|movl|movabsq|andq|andl|shrq|addq)\s+([^,]+), (%\w+)", line
            )
            if m:
                op, src, dst = m.groups()
                value = get(src)
                if op.startswith("and"):
                    left, right = get(dst), value
                    if isinstance(left, int):
                        left, right = right, left
                    if not isinstance(right, int):
                        return None
                    value = mask(left, right & ((1 << 64) - 1))
                elif op == "shrq":
                    value = ("shr", get(dst), value)
                elif op == "addq":
                    value = ("add", get(dst), value)
                put(dst, value)
                continue
            m = re.fullmatch(r"leaq\s+\((%\w+),(%\w+),(\d+)\), (%\w+)", line)
            if m:
                base, index, scale, dst = m.groups()
                put(dst, ("add", get(base), ("mul", get(index), int(scale))))
                continue
            m = re.fullmatch(r"movq\s+(%\w+), (\d*)\((%\w+)\)", line)
            if m:
                src, offset, base = m.groups()
                stores.append((address(base, offset), get(src)))
                continue
        else:
            m = re.fullmatch(r"(lsr|and)\s+(x\d+), (x\d+), #(\w+)", line)
            if m:
                op, dst, src, immediate = m.groups()
                value = int(immediate, 0)
                put(
                    dst,
                    mask(get(src), value) if op == "and" else ("shr", get(src), value),
                )
                continue
            m = re.fullmatch(r"add\s+(x\d+), (x\d+), (x\d+), lsl #(\d+)", line)
            if m:
                dst, base, index, shift = m.groups()
                put(dst, ("add", get(base), ("mul", get(index), 1 << int(shift))))
                continue
            m = re.fullmatch(r"add\s+(x\d+), (x\d+), (x\d+)", line)
            if m:
                dst, a, b = m.groups()
                put(dst, ("add", get(a), get(b)))
                continue
            m = re.fullmatch(r"stp\s+(x\d+), (x\d+), \[(x\d+)(?:, #(\d+))?\]", line)
            if m:
                a, b, base, offset = m.groups()
                offset = int(offset or 0)
                stores.extend(
                    [
                        (address(base, offset), get(a)),
                        (address(base, offset + 8), get(b)),
                    ]
                )
                continue
        return None
    return stores, get("%rax") if target.startswith("x86") else None


def direct_storage_copy(body, target, name):
    """Prove a call-free replacement copies exactly the API's lane bytes."""
    m = re.fullmatch(
        r"\w+_([fiu])(\d+)x(\d+)_(load|store|to_array|from_array)_plain", name
    )
    if not m:
        return False
    _, bits, lanes, operation = m.groups()
    size = int(bits) * int(lanes) // 8
    copied = {}
    if target.startswith("x86"):
        source, dest = ("rdi", "rsi") if operation == "store" else ("rsi", "rdi")
        vector_align = min(size, 32 if name.startswith("v3_") else 64)
        source_align = (
            int(bits) // 8 if operation in ("load", "from_array") else vector_align
        )
        dest_align = (
            vector_align if operation in ("load", "from_array") else int(bits) // 8
        )
        registers = {}
        for line in body:
            if line in ("retq", "vzeroupper", "movq\t%rdi, %rax"):
                continue
            m = re.fullmatch(
                r"v?mov(?:aps|ups|apd|upd|dqa(?:32|64)?|dqu(?:32|64)?)\s+(.+), (.+)",
                line,
            )
            if not m:
                return False
            src, dst = m.groups()
            load = re.fullmatch(r"(\d*)\(%" + source + r"\)", src)
            if load and re.fullmatch(r"%[xyz]mm\d+", dst):
                length = {"x": 16, "y": 32, "z": 64}[dst[1]]
                if re.match(r"v?mov(?:ap|dqa)", line) and (
                    source_align < length or int(load[1] or 0) % length
                ):
                    return False
                registers[dst] = list(
                    range(int(load[1] or 0), int(load[1] or 0) + length)
                )
                continue
            store = re.fullmatch(r"(\d*)\(%" + dest + r"\)", dst)
            if not store or src not in registers:
                return False
            if re.match(r"v?mov(?:ap|dqa)", line) and (
                dest_align < len(registers[src])
                or int(store[1] or 0) % len(registers[src])
            ):
                return False
            for i, byte in enumerate(registers[src], int(store[1] or 0)):
                if i in copied:
                    return False
                copied[i] = byte
    elif target.startswith("wasm"):
        source, dest = (0, 1) if operation == "store" else (1, 0)
        stack = []
        for line in body:
            m = re.fullmatch(r"local.get\s+(\d+)", line)
            if m:
                stack.append(int(m[1]))
                continue
            m = re.fullmatch(r"v128.(load|store)\s+(\d+)(?::p2align=\d+)?", line)
            if not m:
                return False
            op, offset = m[1], int(m[2])
            if op == "load":
                if stack.pop() != source:
                    return False
                stack.append(list(range(offset, offset + 16)))
            else:
                value = stack.pop()
                if stack.pop() != dest or not isinstance(value, list):
                    return False
                for i, byte in enumerate(value, offset):
                    if i in copied:
                        return False
                    copied[i] = byte
        if stack:
            return False
    else:
        return False
    return copied == dict(enumerate(range(size)))


def x86_bitand_equivalent(old, new, name):
    if not re.fullmatch(r"v4x?_[iu]\d+x\d+_bitand_plain", name):
        return False

    # AVX-512 integer/float spelling of the same aligned moves and bitwise AND.
    # Preserve every operand, register width, offset, label and call marker.
    def canonical(lines):
        return [
            re.sub(
                r"^(vmovdqa64|vpandd)\b",
                lambda m: {"vmovdqa64": "vmovaps", "vpandd": "vandps"}[m[0]],
                line,
            )
            for line in lines
        ]

    return canonical(old) == canonical(new)


def x86_halves_dataflow(body, size):
    """Check the three reviewed concatenations, including stack temporaries.

    Interpret only pointer bookkeeping, vector copies, and insertion of the
    high half. Unsupported instructions fail closed; no floating arithmetic is
    approximated. Bytes from the two inputs remain distinct symbolic values.
    """
    registers = {
        "%rdi": ("out", 0),
        "%rsi": ("lo", 0),
        "%rdx": ("hi", 0),
        "%rsp": ("stack", 0),
    }
    vectors, memory = {}, {}

    def shifted(pointer, offset):
        return pointer[0], pointer[1] + offset

    def address(text):
        m = re.fullmatch(r"(-?\d*)\((%\w+)\)", text)
        return shifted(registers[m[2]], int(m[1] or 0)) if m else None

    def read(pointer, length):
        return [
            memory.get(shifted(pointer, i), shifted(pointer, i)) for i in range(length)
        ]

    for line in body:
        if line in ("retq", "vzeroupper", "INTERNAL_CALL", "TAIL_CALL"):
            continue
        m = re.fullmatch(r"(pushq|popq)\s+(%\w+)", line)
        if m:
            if m[1] == "pushq":
                registers["%rsp"] = shifted(registers["%rsp"], -8)
                memory[registers["%rsp"]] = registers.get(m[2], (m[2], 0))
            else:
                registers[m[2]] = memory[registers["%rsp"]]
                registers["%rsp"] = shifted(registers["%rsp"], 8)
            continue
        m = re.fullmatch(r"(movq|leaq)\s+([^,]+), (%\w+)", line)
        if m:
            registers[m[3]] = registers.get(m[2]) if m[1] == "movq" else address(m[2])
            if registers[m[3]] is None:
                return False
            continue
        m = re.fullmatch(r"(subq|addq|andq)\s+\$(-?\d+), %rsp", line)
        if m:
            op, value = m[1], int(m[2])
            registers["%rsp"] = (
                ("aligned_stack", 0)
                if op == "andq"
                else shifted(registers["%rsp"], value if op == "addq" else -value)
            )
            continue
        m = re.fullmatch(r"(v?movaps)\s+([^,]+), ([^,]+)", line)
        if m:
            op, src, dst = m.groups()
            vr = re.fullmatch(r"%([xyz])mm(\d+)", dst)
            if vr:
                length = {"x": 16, "y": 32, "z": 64}[vr[1]]
                pointer = address(src)
                if pointer is None:
                    return False
                vectors[vr[2]] = read(pointer, length) + [
                    0 if op.startswith("v") else None
                ] * (64 - length)
            else:
                vr = re.fullmatch(r"%([xyz])mm(\d+)", src)
                if not vr or address(dst) is None:
                    return False
                pointer = address(dst)
                for i, byte in enumerate(
                    vectors[vr[2]][: {"x": 16, "y": 32, "z": 64}[vr[1]]]
                ):
                    memory[shifted(pointer, i)] = byte
            continue
        m = re.fullmatch(
            r"vinsertf(?:128|64x4)\s+\$1, ([^,]+), %[yz]mm(\d+), %[yz]mm(\d+)", line
        )
        if m:
            pointer = address(m[1])
            if pointer is None:
                return False
            half = size // 2
            vectors[m[3]] = vectors[m[2]][:half] + read(pointer, half)
            continue
        return False
    expected = [("lo", i) for i in range(size // 2)] + [
        ("hi", i) for i in range(size // 2)
    ]
    actual = [memory.get(("out", i)) for i in range(size)]
    return actual == expected and {i for base, i in memory if base == "out"} == set(
        range(size)
    )


def self_test():
    def assembly(body, data=""):
        return (
            "\t.type probe,@function\nprobe:\n"
            + body
            + "\t.size probe, .-probe\n"
            + data
        )

    a = assembly("\tje .LBB0_1\n\txorl %eax, %eax\n.LBB0_1:\n\tretq\n")
    b = a.replace("\txorl %eax, %eax\n.LBB0_1:", ".LBB0_1:\n\txorl %eax, %eax")
    assert parse(a) != parse(b), "branch destinations must retain their positions"
    assert parse(a) == parse(a.replace("LBB0_1", "LBB99_8"))
    a = assembly(
        "\tmovss .LCPI0_0(%rip), %xmm0\n\tretq\n", ".LCPI0_0:\n\t.long 1065353216\n"
    )
    assert parse(a) != parse(a.replace("1065353216", "1073741824"))
    assert parse(a) == parse(a.replace("LCPI0_0", "LCPI99_2"))
    for op in ("callq", "jmp", "jmpq", "bl", "b", "return_call"):
        a = {"probe": [op + "\t_helper"], "_helper": ["ret"]}
        b = {**a, "_helper": ["call\texpensive", "ret"]}
        assert expand("probe", a) != expand("probe", b), op
    for invalid in ("", assembly("\tretq\n").replace("\tretq", "    retq")):
        try:
            parse(invalid)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("unparsed assembly must fail closed")
    copy = ["movups\t(%rsi), %xmm0", "movaps\t%xmm0, (%rdi)", "retq"]
    assert direct_storage_copy(copy, "x86", "v3_f32x4_load_plain")
    for wrong in (
        [line.replace("(%rsi)", "16(%rsi)") for line in copy],
        [line.replace("movups", "movaps") for line in copy],
        [*copy, "callq\textra"],
    ):
        assert not direct_storage_copy(wrong, "x86", "v3_f32x4_load_plain")
    partition = ["movq\t%rdx, %rcx", "andl\t$7, %ecx", "movq\t%rcx, (%rdi)", "retq"]
    assert partition_dataflow(partition, "x86") is not None
    assert partition_dataflow(partition, "x86") != partition_dataflow(
        [line.replace("$7", "$3") for line in partition], "x86"
    )
    halves = [
        "movaps\t(%rsi), %xmm0",
        "movaps\t(%rdx), %xmm1",
        "movaps\t%xmm0, (%rdi)",
        "movaps\t%xmm1, 16(%rdi)",
        "retq",
    ]
    assert x86_halves_dataflow(halves, 32)
    assert not x86_halves_dataflow(
        [line.replace("(%rdx)", "(%rsi)") for line in halves], 32
    )
    print("codegen adversarial self-tests passed")


def main():
    if not __debug__:
        raise RuntimeError("Run without Python -O: assertions must be enabled")
    self_test()
    if sys.argv[1:] == ["--self-test"]:
        return
    if not __debug__:
        raise RuntimeError(
            "Run without Python -O: the codegen assertions must be enabled."
        )
    (pathlib.Path.home() / "tmp").mkdir(parents=True, exist_ok=True)
    repository = pathlib.Path(__file__).resolve().parent.parent
    root = pathlib.Path(
        tempfile.mkdtemp(prefix="archmage-codegen-", dir=pathlib.Path.home() / "tmp")
    )
    assert len(sys.argv) == 3, (
        "usage: python3 xtask/codegen.py BEFORE_REF AFTER_REF_or_WORKTREE"
    )
    for side, ref in zip(["before", "after"], sys.argv[1:]):
        dest = root / (side + "-src")
        dest.mkdir()
        archive = subprocess.check_output(
            ["git", "archive", "HEAD" if ref == "WORKTREE" else ref], cwd=repository
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(dest, filter="data")
        if ref == "WORKTREE":
            changed = subprocess.check_output(
                ["git", "diff", "HEAD", "--name-only"], cwd=repository, text=True
            ).splitlines()
            added = subprocess.check_output(
                [
                    "git",
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "--",
                    "*.rs",
                    "*.toml",
                ],
                cwd=repository,
                text=True,
            ).splitlines()
            for name in sorted(set(changed + added)):
                source = repository / name
                target = dest / name
                if source.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
                elif target.exists():
                    target.unlink()
    print("Assembly and diagnostics:", root, flush=True)
    env = dict(
        os.environ, TMPDIR=str(pathlib.Path.home() / "tmp"), CARGO_INCREMENTAL="0"
    )
    for key in [
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "CARGO_TARGET_DIR",
        "CARGO_BUILD_TARGET",
    ]:
        env.pop(key, None)
    targets = {
        "x86_64-unknown-linux-gnu": [
            ("scalar", "ScalarToken", [128, 256, 512]),
            ("v3", "X64V3Token", [128, 256, 512]),
            ("v4", "X64V4Token", [512]),
            ("v4x", "X64V4xToken", [512]),
        ],
        "aarch64-unknown-linux-gnu": [("neon", "NeonToken", [128, 256, 512])],
        "wasm32-unknown-unknown": [("wasm", "Wasm128Token", [128, 256, 512])],
    }
    elems = [
        ("f32", 32),
        ("f64", 64),
        ("i8", 8),
        ("u8", 8),
        ("i16", 16),
        ("u16", 16),
        ("i32", 32),
        ("u32", 32),
        ("i64", 64),
        ("u64", 64),
    ]
    counts = {}
    expected = {}
    for variant in ["before", "after"]:
        p = root / variant
        (p / "src").mkdir(parents=True, exist_ok=True)
        dep = root / (variant + "-src")
        (p / "Cargo.toml").write_text(
            f'[package]\nname="storage_probe"\nversion="0.0.0"\nedition="2024"\n[dependencies]\narchmage={{path="{dep}",features=["avx512"]}}\nmagetypes={{path="{dep}/magetypes",features=["avx512"]}}\n[workspace]\n'
        )
        for target, tiers in targets.items():
            parts = [
                "#![allow(unused_variables)]\nuse archmage::*;\nuse magetypes::simd::generic::*;\n"
            ]
            count = 0
            probe_names = set()
            for tier, token, widths in tiers:
                for elem, bits in elems:
                    for width in widths:
                        n = width // bits
                        ty = f"{elem}x{n}"
                        simd = f"{ty}<{token}>"
                        arr = f"[{elem};{n}]"
                        name = f"{tier}_{ty}"
                        funcs = [
                            (
                                "load",
                                f"data: &{arr}",
                                simd,
                                f"{ty}::<{token}>::load(token,data)",
                            ),
                            (
                                "from_array",
                                f"data: {arr}",
                                simd,
                                f"{ty}::<{token}>::from_array(token,data)",
                            ),
                            (
                                "store",
                                f"value: {simd}, out: &mut {arr}",
                                "()",
                                "value.store(out)",
                            ),
                            ("to_array", f"value: {simd}", arr, "value.to_array()"),
                            (
                                "partition",
                                f"data: &[{elem}]",
                                f"(&[{arr}], &[{elem}])",
                                f"{ty}::<{token}>::partition_slice(token,data)",
                            ),
                            (
                                "partition_mut",
                                f"data: &mut [{elem}]",
                                f"(&mut [{arr}], &mut [{elem}])",
                                f"{ty}::<{token}>::partition_slice_mut(token,data)",
                            ),
                        ]
                        funcs += [
                            (
                                "index",
                                f"value: &{simd}, i: usize",
                                f"&{elem}",
                                "&value[i]",
                            ),
                            (
                                "index_mut",
                                f"value: &mut {simd}, i: usize",
                                f"&mut {elem}",
                                "&mut value[i]",
                            ),
                        ]
                        if ty in [
                            "f32x4",
                            "f32x8",
                            "f64x2",
                            "f64x4",
                            "i32x4",
                            "i32x8",
                            "i8x16",
                            "u32x4",
                        ]:
                            bytearr = f"[u8;{width // 8}]"
                            funcs += [
                                (
                                    "as_array",
                                    f"value: &{simd}",
                                    f"&{arr}",
                                    "value.as_array()",
                                ),
                                (
                                    "as_array_mut",
                                    f"value: &mut {simd}",
                                    f"&mut {arr}",
                                    "value.as_array_mut()",
                                ),
                                (
                                    "as_bytes",
                                    f"value: &{simd}",
                                    f"&{bytearr}",
                                    "value.as_bytes()",
                                ),
                                (
                                    "as_bytes_mut",
                                    f"value: &mut {simd}",
                                    f"&mut {bytearr}",
                                    "value.as_bytes_mut()",
                                ),
                                (
                                    "from_bytes",
                                    f"value: &{bytearr}",
                                    simd,
                                    f"{ty}::<{token}>::from_bytes(token,value)",
                                ),
                                (
                                    "from_bytes_owned",
                                    f"value: {bytearr}",
                                    simd,
                                    f"{ty}::<{token}>::from_bytes_owned(token,value)",
                                ),
                            ]
                        impl_text = (
                            dep
                            / "magetypes/src/simd/generic/generated"
                            / f"{ty}_impl.rs"
                        ).read_text()
                        block_file = (
                            dep
                            / "magetypes/src/simd/generic/generated"
                            / f"block_ops_{ty}.rs"
                        )
                        if block_file.exists():
                            impl_text += block_file.read_text()
                        for method, mutable, dest in re.findall(
                            r"pub fn (bitcast_(?:ref|mut)_\w+)\(&(?:mut )?self\) -> &(mut )?super::(\w+)<T>",
                            impl_text,
                        ):
                            borrow = "&mut " if mutable else "&"
                            funcs.append(
                                (
                                    method,
                                    f"value: {borrow}{simd}",
                                    f"{borrow}{dest}<{token}>",
                                    f"value.{method}()",
                                )
                            )
                        if ty in [
                            "f32x4",
                            "f32x8",
                            "f64x2",
                            "f64x4",
                            "i32x4",
                            "i32x8",
                            "i8x16",
                            "u32x4",
                        ]:
                            funcs += [
                                (
                                    "cast_slice",
                                    f"data: &[{elem}]",
                                    f"Option<&[{simd}]>",
                                    f"{ty}::<{token}>::cast_slice(token,data)",
                                ),
                                (
                                    "cast_slice_mut",
                                    f"data: &mut [{elem}]",
                                    f"Option<&mut [{simd}]>",
                                    f"{ty}::<{token}>::cast_slice_mut(token,data)",
                                ),
                            ]
                        if ty == "u32x4":
                            funcs.append(
                                (
                                    "bitcast_value",
                                    f"value: {simd}",
                                    f"f32x4<{token}>",
                                    "value.bitcast_f32x4()",
                                )
                            )
                        if ty in ["f32x8", "f32x16"]:
                            half = f"f32x{n // 2}<{token}>"
                            funcs += [
                                (
                                    "from_halves",
                                    f"lo: {half}, hi: {half}",
                                    simd,
                                    f"{ty}::<{token}>::from_halves(token,lo,hi)",
                                ),
                                ("low", f"value: {simd}", half, "value.low()"),
                                ("high", f"value: {simd}", half, "value.high()"),
                            ]
                        funcs += [
                            ("add", f"a: {simd}, b: {simd}", simd, "a + b"),
                            ("bitand", f"a: {simd}, b: {simd}", simd, "a & b"),
                        ]
                        if elem.startswith("f"):
                            funcs += [
                                ("sqrt", f"a: {simd}", simd, "a.sqrt()"),
                                (
                                    "fma",
                                    f"a: {simd}, b: {simd}, c: {simd}",
                                    simd,
                                    "a.mul_add(b,c)",
                                ),
                            ]
                        # Ordinary callers must not acquire hidden feature calls,
                        # stack temporaries or dispatch overhead. Check both contexts.
                        if tier != "scalar":
                            funcs += [
                                (op + "_plain", args, result, body)
                                for op, args, result, body in funcs
                                if not op.startswith("partition")
                            ]
                        for op, args, result, body in funcs:
                            attr = (
                                "#[arcane]\n"
                                if tier != "scalar"
                                and not op.startswith("partition")
                                and not op.endswith("_plain")
                                else ""
                            )
                            parts.append(
                                f"#[unsafe(no_mangle)]\n{attr}#[inline(never)]\npub fn {name}_{op}(token: {token}, {args}) -> {result} {{ {body} }}\n"
                            )
                            count += 1
                            assert f"{name}_{op}" not in probe_names
                            probe_names.add(f"{name}_{op}")
            src = "\n".join(parts)
            (p / "src/lib.rs").write_text(src)
            (root / f"{target}.rs").write_text(src)
            counts[target] = count
            assert expected.setdefault(target, probe_names) == probe_names
            cmd = [
                "cargo",
                "+stable",
                "rustc",
                "--manifest-path",
                str(p / "Cargo.toml"),
                "--release",
                "--lib",
                "--target",
                target,
                "--",
                "--emit=asm",
            ]
            out = subprocess.run(cmd, env=env, capture_output=True, text=True)
            (root / f"{variant}-{target}.log").write_text(out.stdout + out.stderr)
            if out.returncode:
                raise RuntimeError(out.stderr)
            files = list(
                (p / "target" / target / "release/deps").glob("storage_probe-*.s")
            )
            assert len(files) == 1, files
            (root / f"{variant}-{target}.s").write_text(files[0].read_text())
            print(variant, target, count, flush=True)
    result = {}
    for target in targets:
        a, b = [
            normalize((root / f"{v}-{target}.s").read_text())
            for v in ["before", "after"]
        ]
        diff = "".join(
            difflib.unified_diff(
                a.splitlines(True),
                b.splitlines(True),
                fromfile="before",
                tofile="after",
            )
        )
        (root / f"{target}.diff").write_text(diff)
        result[target] = {
            "probes": counts[target],
            "identical": a == b,
            "diff_lines": len(diff.splitlines()),
        }
        print(target, result[target], flush=True)
    result["rustc"] = subprocess.check_output(["rustc", "+stable", "-Vv"], text=True)
    (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    compare(root, expected)


def compare(root, expected):
    summary = {}
    for target in [
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "wasm32-unknown-unknown",
    ]:
        a, b = [
            parse((root / f"{v}-{target}.s").read_text()) for v in ["before", "after"]
        ]
        names = {n for n in a if not n.startswith("_")}
        assert names == {n for n in b if not n.startswith("_")}
        assert names == expected[target], (
            target,
            "missing or unexpected probes",
            names ^ expected[target],
        )
        changes = []
        for n in sorted(names):
            old, new = expand(n, a), expand(n, b)
            if old == new:
                continue
            kind = "stronger destination alignment"
            if target.startswith("aarch64") and n.removesuffix("_plain") in [
                "neon_f32x8_fma",
                "neon_f64x4_fma",
            ]:
                assert list(map(opcode, old)) == list(map(opcode, new))
                assert neon_fma_dataflow(old) is not None and neon_fma_dataflow(
                    old
                ) == neon_fma_dataflow(new)
                kind = "identical FMA dataflow and instruction sequence; register allocation only"
            elif "_partition" in n:
                # Soundness comes from core::slice::as_chunks[_mut]; this gate checks
                # instruction/control-flow/memory cost despite register rescheduling.
                assert len(old) == len(new)
                assert transfers(old) == transfers(new)
                assert memory_accesses(old, target) == memory_accesses(new, target)
                assert partition_dataflow(
                    old, target
                ) is not None and partition_dataflow(old, target) == partition_dataflow(
                    new, target
                ), (target, n, old, new)
                before_ops = collections.Counter(map(opcode, old))
                after_ops = collections.Counter(map(opcode, new))
                assert before_ops == after_ops or (
                    target.startswith("x86")
                    and after_ops - before_ops == {"movl": 1}
                    and before_ops - after_ops == {"movq": 1}
                ), (target, n, old, new)
                kind = "equal symbolic partition results and instruction/memory counts"
            elif len(new) <= len(old) and direct_storage_copy(new, target, n):
                assert "INTERNAL_CALL" in old or "TAIL_CALL" in old
                kind = "exact byte copy; eliminates feature calls without increasing instructions"
            elif target.startswith("x86") and x86_bitand_equivalent(old, new, n):
                kind = "identical AVX-512 bitwise operands and instruction counts"
            elif not aligned_store_equivalent(old, new, n):
                assert target.startswith("x86") and n in [
                    "v3_f32x8_from_halves_plain",
                    "v4_f32x16_from_halves_plain",
                    "v4x_f32x16_from_halves_plain",
                ], (target, n, old, new)
                assert len(new) < len(old)
                assert (
                    new.count("INTERNAL_CALL") == 1 and old.count("INTERNAL_CALL") == 2
                )
                size = 32 if n.startswith("v3_") else 64
                assert x86_halves_dataflow(old, size) and x86_halves_dataflow(
                    new, size
                ), (target, n, old, new)
                assert len(memory_accesses(new, target)) <= len(
                    memory_accesses(old, target)
                )
                assert transfers(new) == transfers(old) == []
                kind = "one feature boundary instead of two; fewer total instructions and memory accesses"
            changes.append(
                {"name": n, "kind": kind, "before_expanded": old, "after_expanded": new}
            )
        summary[target] = {
            "api_probes": len(names),
            "equivalent_expanded_bodies": len(names) - len(changes),
            "reviewed_changes": changes,
        }
        print(target, len(names), "probes;", len(changes), "reviewed differences")
    (root / "gate-results.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarise the dumps written by dump.sh: widest register, AVX-512-only
features, loops, vector spills, calls, and whether v3 and v4 differ after
register-name and label normalisation. Usage: analyze.py [asm-dir]"""
import os, re, sys, glob, collections

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/tmp/handoff/archmage-v4ctx-asm/asm")
LABEL = re.compile(r"^(\.L[A-Za-z0-9_]+):")
VREG = re.compile(r"\b([xyz]mm)(\d+)\b")

def load(path):
    lines = [l.rstrip("\n") for l in open(path)]
    return lines[1:]  # first line is the symbol

def instrs(lines):
    out = []  # (lineno, label_or_None, text)
    for n, l in enumerate(lines, 2):
        out.append((n, l))
    return out

def widest(lines):
    w = "none"
    for _, l in instrs(lines):
        for m in VREG.finditer(l):
            if m.group(1) == "zmm": return "zmm"
            if m.group(1) == "ymm": w = "ymm"
            elif w == "none": w = "xmm"
    return w

def mnemonic(l):
    t = l.strip()
    if not t or t.endswith(":"): return None
    return t.split()[0]

def avx512_markers(lines):
    k = collections.Counter()
    for _, l in instrs(lines):
        if re.search(r"\bk[0-7]\b", l): k["kreg"] += 1
        if "vpternlog" in l: k["vpternlog"] += 1
        if "{1to" in l: k["bcast{1toN}"] += 1
        for m in VREG.finditer(l):
            if int(m.group(2)) >= 16: k["reg16+"] += 1; break
        if re.search(r"\bzmm\d+", l): k["zmm"] += 1
    return k

def loops(lines):
    labels = {}
    for n, l in instrs(lines):
        m = LABEL.match(l)
        if m: labels[m.group(1)] = n
    res = []
    for n, l in instrs(lines):
        m = re.match(r"\s*(j\w+)\s+(\.L\w+)", l)
        if m and m.group(2) in labels and labels[m.group(2)] <= n:
            res.append((labels[m.group(2)], n))
    out = []
    for a, b in res:
        body = [(i, t) for i, t in instrs(lines) if a <= i <= b]
        ins = [t for _, t in body if mnemonic(t) and not LABEL.match(t)]
        vec = [t for t in ins if VREG.search(t)]
        out.append({"lines": (a, b), "n": len(ins), "nvec": len(vec),
                    "w": widest([t for t in ins]),
                    "spill": sum(1 for t in vec if re.search(r"\[(rsp|rbp)", t)),
                    "calls": [t.strip() for t in ins if mnemonic(t) == "call"]})
    return out

def spills(lines):
    return sum(1 for _, l in instrs(lines)
               if VREG.search(l) and re.search(r"\[(rsp|rbp)", l) and mnemonic(l) and mnemonic(l).startswith("v"))

def calls(lines):
    return [l.strip() for _, l in instrs(lines) if mnemonic(l) == "call"]

def norm(lines):
    out = []
    for _, l in instrs(lines):
        t = l.strip()
        if not t: continue
        t = re.sub(r"\.L[A-Za-z0-9_]+", "L", t)
        t = VREG.sub(lambda m: "V" + m.group(1)[0], t)  # keep width class only
        t = re.sub(r"\b(r[a-z0-9]+|e[a-z]{2}|[a-d]l)\b", "G", t)
        out.append(t)
    return out

def mnems(lines):
    return collections.Counter(mnemonic(l) for _, l in instrs(lines) if mnemonic(l) and not LABEL.match(l))

ids = sorted({os.path.basename(p).rsplit("_", 1)[0] if p.endswith(("_v3.s", "_v4.s")) else None
              for p in glob.glob(D + "/*.s")} - {None})
ids = sorted({re.sub(r"_(v3|v4_via_v3|v4)\.s$", "", os.path.basename(p)) for p in glob.glob(D + "/*.s")})
for i in ids:
    p3 = f"{D}/{i}_v3.s"
    p4 = f"{D}/{i}_v4.s" if os.path.exists(f"{D}/{i}_v4.s") else f"{D}/{i}_v4_via_v3.s"
    if not (os.path.exists(p3) and os.path.exists(p4)): continue
    a, b = load(p3), load(p4)
    print(f"=== {i}   v3 {os.path.basename(p3)} ({len(a)} lines)   v4 {os.path.basename(p4)} ({len(b)} lines)")
    for tag, L in (("v3", a), ("v4", b)):
        print(f"  {tag}: widest={widest(L)} avx512={dict(avx512_markers(L))} vec-spills={spills(L)} calls={calls(L)}")
        for lp in loops(L):
            print(f"      loop L{lp['lines'][0]}-{lp['lines'][1]} n={lp['n']} vec={lp['nvec']} w={lp['w']} spills={lp['spill']} calls={lp['calls']}")
    same = norm(a) == norm(b)
    print(f"  normalised-identical (labels, register numbers, GPRs erased): {same}")
    ma, mb = mnems(a), mnems(b)
    only4 = {k: v for k, v in (mb - ma).items()}
    only3 = {k: v for k, v in (ma - mb).items()}
    if not same:
        print(f"  mnemonic count delta  v4-v3: {only4}")
        print(f"                        v3-v4: {only3}")

#!/usr/bin/env python3
"""Static codegen equivalence-class & llvm-mca analyzer for magetypes/magemath kernels.

Compiles standalone `#[unsafe(no_mangle)] #[inline(never)]` stream64 and chain64
probes across x86-64 (V3, V4, Scalar), AArch64 (NEON), and WASM (simd128,
relaxed-simd), groups all policy flags into exact normalized assembly
equivalence classes, flags any unexpected hot-path `call`/`bl` or stack frame
spills, and runs `llvm-mca` on one representative per unique equivalence class.

Usage examples:
  # Fast static equivalence + llvm-mca across V3, NEON, WASM strict & relaxed (building on i265):
  python3 -B xtask/mca.py --remote i265 --out ~/tmp/mca-audit

  # Full width matrix (all 6 widths, all 4 ops) with deduplication map for policy benchmarks:
  python3 -B xtask/mca.py --remote i265 --widths all --tiers v3,v4,neon,scalar,wasm128,wasm128-relaxed --out ~/tmp/mca-full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

REPO = Path(__file__).resolve().parent.parent
TOKEN_REGISTRY_TOML = REPO / "token-registry.toml"
REGISTRY_RS = REPO / "benchmarks/primitive_variants_2026-11-10/src/policy_registry.rs"


def _load_registry_target_features() -> dict[str, str]:
    data = tomllib.loads(TOKEN_REGISTRY_TOML.read_text())
    out: dict[str, str] = {}
    for tok in data.get("token", []):
        feats = tok.get("features", [])
        if feats:
            out[tok["name"]] = f'#[target_feature(enable = "{",".join(feats)}")]'
    return out


_REGISTRY_TF = _load_registry_target_features()

TIER_CONFIG = {
    "v3": {
        "target": "x86_64-unknown-linux-gnu",
        "rustflags": "",
        "token_ty": "archmage::X64V3Token",
        "target_feature": _REGISTRY_TF["X64V3Token"],
        "mca_triple": "x86_64-unknown-linux-gnu",
        "mca_cpus": ["znver5", "arrowlake"],
    },
    "v4": {
        "target": "x86_64-unknown-linux-gnu",
        "rustflags": "",
        "token_ty": "archmage::X64V4Token",
        "target_feature": _REGISTRY_TF["X64V4Token"],
        "mca_triple": "x86_64-unknown-linux-gnu",
        "mca_cpus": ["znver5", "skylake-avx512"],
    },
    "v4x": {
        "target": "x86_64-unknown-linux-gnu",
        "rustflags": "",
        "token_ty": "archmage::X64V4xToken",
        "target_feature": _REGISTRY_TF["X64V4xToken"],
        "mca_triple": "x86_64-unknown-linux-gnu",
        "mca_cpus": ["znver5", "icelake-server"],
    },
    "scalar": {
        "target": "x86_64-unknown-linux-gnu",
        "rustflags": "",
        "token_ty": "archmage::ScalarToken",
        "target_feature": "",
        "mca_triple": "x86_64-unknown-linux-gnu",
        "mca_cpus": ["znver5"],
    },
    "neon": {
        "target": "aarch64-unknown-linux-gnu",
        "rustflags": "",
        "token_ty": "archmage::NeonToken",
        "target_feature": _REGISTRY_TF["NeonToken"],
        "mca_triple": "aarch64-unknown-linux-gnu",
        "mca_cpus": ["apple-m4", "neoverse-n1"],
    },
    "wasm128": {
        "target": "wasm32-wasip1",
        "rustflags": "-Ctarget-feature=+simd128",
        "token_ty": "archmage::Wasm128Token",
        "target_feature": "",
        "mca_triple": None,
        "mca_cpus": [],
    },
    "wasm128-relaxed": {
        "target": "wasm32-wasip1",
        "rustflags": "-Ctarget-feature=+simd128,+relaxed-simd",
        "token_ty": "archmage::Wasm128Token",
        "target_feature": "",
        "mca_triple": None,
        "mca_cpus": [],
    },
}

WIDTH_SPECS = {
    "f32x4": ("f32", 4, "f32x4", "F32x4Backend", True),
    "f32x8": ("f32", 8, "f32x8", "F32x8Backend", True),
    "f32x16": ("f32", 16, "f32x16", "F32x16Backend", True),
    "f64x2": ("f64", 2, "f64x2", "F64x2Backend", False),
    "f64x4": ("f64", 4, "f64x4", "F64x4Backend", False),
    "f64x8": ("f64", 8, "f64x8", "F64x8Backend", False),
}


def load_policy_flags() -> dict[str, list[tuple[int, str]]]:
    text = REGISTRY_RS.read_text()
    blocks: dict[str, list[tuple[int, str]]] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"macro_rules!\s+(\w+)\s*\{", line)
        if m:
            current = m.group(1)
            blocks[current] = []
            continue
        if current and line.startswith("}"):
            current = None
            continue
        if current:
            m2 = re.search(r'\$apply!\((\d+),\s*"([^"]+)"\);', line)
            if m2:
                blocks[current].append((int(m2.group(1)), m2.group(2)))
    return blocks


def generate_probe_crate(
    probe_dir: Path,
    repo_rel: str,
    tiers_for_target: list[str],
    ops: list[str],
    widths: list[str],
    flags_map: dict[str, list[tuple[int, str]]],
) -> list[dict]:
    (probe_dir / "src").mkdir(parents=True, exist_ok=True)
    cargo_toml = f"""[package]
name = "archmage-mca-probe"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[lib]
crate-type = ["rlib"]

[dependencies]
archmage = {{ path = "{repo_rel}", features = ["std", "avx512"] }}
magetypes = {{ path = "{repo_rel}/magetypes", features = ["std", "w512", "avx512"] }}
magemath = {{ path = "{repo_rel}/magemath", features = ["std", "w512", "avx512"] }}
"""
    (probe_dir / "Cargo.toml").write_text(cargo_toml)

    lines = [
        "#![allow(unused_imports, dead_code, non_snake_case)]",
        "use archmage::SimdToken;",
    ]
    manifest: list[dict] = []

    for tier in tiers_for_target:
        tcfg = TIER_CONFIG[tier]
        tok_ty = tcfg["token_ty"]
        tf_attr = tcfg["target_feature"]
        tier_ident = tier.replace("-", "_")

        for wname in widths:
            elem, lanes, vec_ty, _bound, has_recip = WIDTH_SPECS[wname]
            for op in ops:
                if op in ("recip", "rsqrt") and not has_recip:
                    continue
                is_recip = op in ("recip", "rsqrt")
                arms: list[tuple[int, str, bool]] = [(-1, "existing", True)]
                case_key = "recip_cases" if is_recip else "minmax_cases"
                for bits, label in flags_map[case_key]:
                    arms.append((bits, label, False))

                for bits, label, legacy in arms:
                    arm_ident = "legacy" if legacy else label
                    if op == "min":
                        expr = (
                            "va.min(vb)"
                            if legacy
                            else f"magemath::minmax::try_min_{vec_ty}_with::<{bits}, _>(t, va, vb).unwrap()"
                        )
                    elif op == "max":
                        expr = (
                            "va.max(vb)"
                            if legacy
                            else f"magemath::minmax::try_max_{vec_ty}_with::<{bits}, _>(t, va, vb).unwrap()"
                        )
                    elif op == "recip":
                        expr = (
                            "va.recip()"
                            if legacy
                            else f"magemath::recip::try_recip_{vec_ty}_with::<{bits}, _>(t, va).unwrap()"
                        )
                    elif op == "rsqrt":
                        expr = (
                            "va.rsqrt()"
                            if legacy
                            else f"magemath::recip::try_rsqrt_{vec_ty}_with::<{bits}, _>(t, va).unwrap()"
                        )
                    else:
                        raise ValueError(op)

                    # 1. stream64 (matches policy_workload.rs)
                    sym_stream = f"mca_{tier_ident}_{op}_{wname}_stream64_{arm_ident}"
                    if is_recip:
                        lines.append(f"""
#[unsafe(no_mangle)]
#[inline(never)]
{tf_attr}
pub fn {sym_stream}(t: {tok_ty}, a: &[{elem}; 64], out: &mut [{elem}; 64]) {{
    use magetypes::simd::generic::{vec_ty} as V;
    for (ca, cout) in a.as_chunks::<{lanes}>().0.iter().zip(out.as_chunks_mut::<{lanes}>().0) {{
        let va = V::from_array_t(t, *ca);
        *cout = ({expr}).to_array();
    }}
}}""")
                    else:
                        lines.append(f"""
#[unsafe(no_mangle)]
#[inline(never)]
{tf_attr}
pub fn {sym_stream}(t: {tok_ty}, a: &[{elem}; 64], b: &[{elem}; 64], out: &mut [{elem}; 64]) {{
    use magetypes::simd::generic::{vec_ty} as V;
    for ((ca, cb), cout) in a.as_chunks::<{lanes}>().0.iter().zip(b.as_chunks::<{lanes}>().0).zip(out.as_chunks_mut::<{lanes}>().0) {{
        let va = V::from_array_t(t, *ca);
        let vb = V::from_array_t(t, *cb);
        *cout = ({expr}).to_array();
    }}
}}""")
                    manifest.append(
                        {
                            "symbol": sym_stream,
                            "tier": tier,
                            "op": op,
                            "width": wname,
                            "shape": "stream64",
                            "arm": f"existing Magetypes {op}" if legacy else label,
                            "bits": bits,
                            "chunks": 64 // lanes,
                        }
                    )

                    # 2. chain64 (matches policy_workload.rs)
                    sym_chain = f"mca_{tier_ident}_{op}_{wname}_chain64_{arm_ident}"
                    chain_step = (
                        expr.replace("va", "x")
                        if is_recip
                        else expr.replace("va", "x").replace("vb", f"V::splat_t(t, 2.0 as {elem})")
                    )
                    add_const = "0.25" if is_recip else "0.03125"
                    lines.append(f"""
#[unsafe(no_mangle)]
#[inline(never)]
{tf_attr}
pub fn {sym_chain}(t: {tok_ty}, initial: {elem}) -> [{elem}; {lanes}] {{
    use magetypes::simd::generic::{vec_ty} as V;
    let mut x = V::splat_t(t, initial);
    for _ in 0..64 {{
        x = ({chain_step}) + V::splat_t(t, {add_const} as {elem});
    }}
    x.to_array()
}}""")
                    manifest.append(
                        {
                            "symbol": sym_chain,
                            "tier": tier,
                            "op": op,
                            "width": wname,
                            "shape": "chain64",
                            "arm": f"existing Magetypes {op}" if legacy else label,
                            "bits": bits,
                            "chunks": 64,
                        }
                    )

    (probe_dir / "src/lib.rs").write_text("\n".join(lines) + "\n")
    return manifest


def parse_asm_functions(asm_text: str, is_wasm: bool) -> dict[str, list[str]]:
    funcs: dict[str, list[str]] = {}
    aliases: dict[str, str] = {}
    cur_names: list[str] = []
    cur_lines: list[str] = []
    for raw in asm_text.splitlines():
        s = raw.strip()
        # LLVM Identical Code Folding emits symbol aliases:
        #   mca_foo = mca_bar
        #   .set mca_foo, mca_bar
        m_alias = re.match(
            r"^(?:\.set\s+)?_?(mca_[A-Za-z0-9_]+)\s*(?:=|,)\s*_?(mca_[A-Za-z0-9_]+)\s*$",
            s,
        )
        if m_alias:
            aliases[m_alias.group(1)] = m_alias.group(2)
            continue

        m_lbl = re.match(r"^_?(mca_[A-Za-z0-9_]+):(?:\s*(?:#|//).*)?$", s)
        if m_lbl:
            sym = m_lbl.group(1)
            if not cur_names:
                cur_names = [sym]
                cur_lines = []
            elif all(
                not ln.strip() or any(ln.strip().startswith(d) for d in SKIP_DIRECTIVES)
                for ln in cur_lines
            ):
                # Consecutive entry labels sharing one body before first instruction
                cur_names.append(sym)
            else:
                cur_lines.append(raw)
            continue

        if cur_names:
            if s in (".cfi_endproc", "end_function") or s.startswith(".Lfunc_end"):
                for name in cur_names:
                    funcs[name] = list(cur_lines)
                cur_names = []
                cur_lines = []
            else:
                cur_lines.append(raw)

    def resolve(sym: str, seen: set[str] | None = None) -> list[str]:
        if sym in funcs:
            return funcs[sym]
        seen = seen or set()
        if sym in aliases and sym not in seen:
            seen.add(sym)
            return resolve(aliases[sym], seen)
        return []

    for src in list(aliases):
        resolved = resolve(src)
        if resolved:
            funcs[src] = resolved
    return funcs


SKIP_DIRECTIVES = (
    ".cfi_",
    ".loc",
    ".file",
    ".p2align",
    ".align",
    ".type",
    ".size",
    ".section",
    ".text",
    ".globl",
    ".hidden",
    ".local",
    ".functype",
    ".local ",
    "#",
    "//",
)


def normalize_body(lines: list[str]) -> tuple[list[str], list[str], dict]:
    """Return (normalized_lines, hot_mca_lines, metrics)."""
    label_map: dict[str, str] = {}
    const_map: dict[str, str] = {}

    def map_label(m: re.Match) -> str:
        tok = m.group(0)
        return label_map.setdefault(tok, f".L{len(label_map)}")

    def map_const(m: re.Match) -> str:
        tok = m.group(0)
        return const_map.setdefault(tok, f".LCPI{len(const_map)}")

    norm: list[str] = []
    hot_norm: list[str] = []
    hot_lines: list[str] = []
    past_ret = False
    hot_calls: list[str] = []
    cold_calls: list[str] = []
    stack_ops = 0
    total_insts = 0

    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        # Strip trailing comments
        if "//" in s:
            s = s.split("//", 1)[0].strip()
        if "#" in s and not s.startswith("#") and " #" in s:
            s = s.split(" #", 1)[0].strip()
        if not s or any(s.startswith(d) for d in SKIP_DIRECTIVES):
            continue

        # Label definition
        if s.endswith(":"):
            n_lbl = re.sub(r"\.LBB\d+_\d+|\.Ltmp\d+", map_label, s)
            norm.append(n_lbl)
            if not past_ret:
                hot_norm.append(n_lbl)
                hot_lines.append(s)
            continue

        total_insts += 1
        n_inst = re.sub(r"\.LBB\d+_\d+|\.Ltmp\d+", map_label, s)
        n_inst = re.sub(r"\.LCPI\d+_\d+", map_const, n_inst)
        n_inst = re.sub(r"\.Lanon\.[0-9a-f]+\.\d+", ".Lanon", n_inst)
        n_inst = re.sub(r"\s+", " ", n_inst)
        norm.append(n_inst)

        op = s.split()[0]
        # Stack frame adjustments
        if (
            re.match(r"subq?\s+\$\d+,\s*%rsp", s)
            or re.match(r"sub\s+sp,\s*sp,", s)
            or "global.get\t$__stack_pointer" in raw
            or "global.set\t$__stack_pointer" in raw
        ):
            stack_ops += 1

        # Call detection
        if op in ("call", "callq", "bl", "blr"):
            is_cold_panic = any(
                k in s for k in ("unwrap_failed", "panic", "handle_error")
            )
            if past_ret or is_cold_panic:
                cold_calls.append(s)
            else:
                hot_calls.append(s)

        if not past_ret:
            hot_norm.append(n_inst)
            if op in ("ret", "retq", "end_function"):
                past_ret = True
            else:
                hot_lines.append(raw)

    # Extract innermost backward loop block if one exists for clean llvm-mca simulation,
    # otherwise simulate the straight-line hot body (excluding ret and panic jumps).
    mca_snippet = extract_mca_region(hot_lines)
    metrics = {
        "total_insts": total_insts,
        "hot_insts": sum(1 for ln in hot_lines if not ln.strip().endswith(":")),
        "mca_insts": len(mca_snippet["lines"]),
        "mca_is_loop": mca_snippet["is_loop"],
        "hot_calls": hot_calls,
        "cold_calls": cold_calls,
        "stack_ops": stack_ops,
    }
    # Use hot_norm (instructions up to ret/end_function) for equivalence hashing so
    # cold panic source-location constants after `ret` never split identical hot kernels.
    return hot_norm or norm, mca_snippet["lines"], metrics


def extract_mca_region(hot_lines: list[str]) -> dict:
    """Find the main inner loop block `.LBBx_y: ... jcc .LBBx_y` if present."""
    label_positions: dict[str, int] = {}
    for idx, raw in enumerate(hot_lines):
        s = raw.strip()
        if s.endswith(":"):
            label_positions[s[:-1]] = idx

    best_loop: list[str] | None = None
    for idx, raw in enumerate(hot_lines):
        s = raw.strip()
        if s.endswith(":"):
            continue
        for lbl, pos in label_positions.items():
            if pos < idx and re.search(rf"\b{re.escape(lbl)}\b", s):
                # Found backward branch to `lbl`!
                region = [
                    ln
                    for ln in hot_lines[pos + 1 : idx + 1]
                    if not ln.strip().endswith(":")
                ]
                if region and (best_loop is None or len(region) > len(best_loop)):
                    best_loop = region

    if best_loop:
        return {"is_loop": True, "lines": best_loop}

    # Unrolled straight-line kernel: strip conditional jumps to cold panic blocks
    straight = []
    for raw in hot_lines:
        s = raw.strip()
        if s.endswith(":"):
            continue
        op = s.split()[0]
        if op.startswith(("j", "b.")) and ".LBB" in s:
            continue
        if op in ("call", "callq", "bl") and any(
            k in s for k in ("unwrap_failed", "panic")
        ):
            continue
        straight.append(raw)
    return {"is_loop": False, "lines": straight}


def find_llvm_mca() -> str | None:
    for cand in ("llvm-mca-21", "llvm-mca-20", "llvm-mca-19", "llvm-mca"):
        found = shutil.which(cand)
        if found:
            return found
    return None


def run_llvm_mca(mca_bin: str, triple: str, cpu: str, inst_lines: list[str]) -> dict:
    if not inst_lines:
        return {"error": "empty snippet"}
    # Replace RIP/ADRP constant-pool symbol relocations with a zero displacement so
    # llvm-mca's standalone MC parser accepts load operands without object symbols.
    cleaned = []
    for raw in inst_lines:
        s = raw
        # x86 RIP-relative constant pool: .LCPI0_0(%rip) -> 0(%rip)
        s = re.sub(r"\.?LCPI[0-9_]+\(%rip\)", "0(%rip)", s)
        # aarch64 adrp + ldr/add symbol page relocations:
        if re.match(r"\s*adrp\s+\w+,\s*\S+", s):
            continue
        s = re.sub(r":lo12:\S+\]", "#0]", s)
        s = re.sub(r",\s*:lo12:\S+", ", #0", s)
        cleaned.append(s)

    asm_input = "\n".join(cleaned) + "\n"
    cmd = [
        mca_bin,
        f"-mtriple={triple}",
        f"-mcpu={cpu}",
        "-iterations=100",
        "-timeline=false",
    ]
    res = subprocess.run(cmd, input=asm_input, capture_output=True, text=True)
    if res.returncode != 0:
        return {"error": res.stderr.splitlines()[0] if res.stderr else "mca failed"}

    out = res.stdout
    def grab(pattern: str) -> float | None:
        m = re.search(pattern, out)
        return float(m.group(1)) if m else None

    iters = grab(r"Iterations:\s+(\d+)") or 100.0
    total_cycles = grab(r"Total Cycles:\s+(\d+)")
    total_uops = grab(r"Total uOps:\s+(\d+)")
    rthroughput = grab(r"Block RThroughput:\s+([0-9.]+)")
    ipc = grab(r"IPC:\s+([0-9.]+)")
    return {
        "cycles_per_iter": round(total_cycles / iters, 2) if total_cycles is not None else None,
        "uops_per_iter": round(total_uops / iters, 2) if total_uops is not None else None,
        "block_rthroughput": rthroughput,
        "ipc": ipc,
    }


def compile_asm(
    probe_dir: Path,
    target: str,
    rustflags: str,
    remote: str | None,
    remote_repo: str,
) -> str:
    if remote is None:
        env = dict(os.environ, RUSTFLAGS=rustflags, TMPDIR=str(Path.home() / "tmp"))
        cmd = [
            "cargo",
            "rustc",
            "--release",
            "--lib",
            "--manifest-path",
            str(probe_dir / "Cargo.toml"),
            "--target",
            target,
            "--",
            "--emit=asm",
        ]
        subprocess.run(cmd, env=env, check=True)
        files = sorted(
            (probe_dir / "target" / target / "release/deps").glob("archmage_mca_probe-*.s"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        assert files, f"No .s emitted for {target}"
        return files[0].read_text()

    # Remote compilation via run-heavy on worker host (keeps dev free of builds)
    remote_probe = "~/tmp/archmage-mca-probe"
    subprocess.run(
        ["ssh", remote, f"mkdir -p {remote_probe}/src"],
        check=True,
    )
    subprocess.run(
        [
            "rsync",
            "-az",
            str(probe_dir / "Cargo.toml"),
            str(probe_dir / "src/lib.rs"),
            f"{remote}:{remote_probe}/src_sync/",
        ],
        check=True,
    )
    remote_cmd = (
        f'bash -lc \'mv {remote_probe}/src_sync/Cargo.toml {remote_probe}/Cargo.toml && '
        f'mv {remote_probe}/src_sync/lib.rs {remote_probe}/src/lib.rs && '
        f'rmdir {remote_probe}/src_sync && '
        f'cd {remote_probe} && '
        f'RUSTFLAGS="{rustflags}" ~/work/claudehints/scripts/run-heavy --mem 16G -- '
        f'cargo rustc --release --lib --target {target} -- --emit=asm >&2 && '
        f'ls -t target/{target}/release/deps/archmage_mca_probe-*.s | head -n1 | xargs cat\''
    )
    out = subprocess.run(
        ["ssh", remote, remote_cmd],
        capture_output=True,
        text=True,
        check=True,
    )
    if out.stderr:
        for line in out.stderr.splitlines():
            if "run-heavy:" in line:
                print(f"  [{remote} {target}] {line}", file=sys.stderr)
    return out.stdout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiers", default="v3,neon,wasm128,wasm128-relaxed")
    p.add_argument("--ops", default="min,max,recip,rsqrt")
    p.add_argument("--widths", default="f32x8,f64x4")
    p.add_argument("--remote", default=None, help="SSH build host (e.g. i265)")
    p.add_argument(
        "--remote-repo",
        default="/home/lilith/tmp/archmage-fma-verify",
        help="Path to synced archmage checkout on --remote",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    widths = (
        list(WIDTH_SPECS.keys())
        if args.widths == "all"
        else [w.strip() for w in args.widths.split(",") if w.strip()]
    )

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = Path.home() / "tmp" / "archmage-mca-probe-local"
    probe_dir.mkdir(parents=True, exist_ok=True)

    flags_map = load_policy_flags()
    mca_bin = find_llvm_mca()

    # Group tiers by (target, rustflags) so each target compiles only once!
    compile_groups: dict[tuple[str, str], list[str]] = {}
    for tier in tiers:
        cfg = TIER_CONFIG[tier]
        compile_groups.setdefault((cfg["target"], cfg["rustflags"]), []).append(tier)

    repo_path_for_cargo = args.remote_repo if args.remote else str(REPO)
    all_classes: list[dict] = []
    dedup_map: dict[str, list[dict]] = {}
    total_arms = 0
    total_unique = 0

    for (target, rustflags), group_tiers in compile_groups.items():
        manifest = generate_probe_crate(
            probe_dir, repo_path_for_cargo, group_tiers, ops, widths, flags_map
        )
        asm_text = compile_asm(probe_dir, target, rustflags, args.remote, args.remote_repo)
        (out_dir / f"{'_'.join(group_tiers)}.s").write_text(asm_text)
        asm_funcs = parse_asm_functions(asm_text, target.startswith("wasm32"))

        # Group manifest entries by (tier, op, width, shape)
        cells: dict[tuple[str, str, str, str], list[dict]] = {}
        for item in manifest:
            cells.setdefault(
                (item["tier"], item["op"], item["width"], item["shape"]), []
            ).append(item)

        for (tier, op, wname, shape), items in cells.items():
            tcfg = TIER_CONFIG[tier]
            by_hash: dict[str, dict] = {}
            for item in items:
                total_arms += 1
                raw_lines = asm_funcs.get(item["symbol"], [])
                norm_lines, mca_lines, metrics = normalize_body(raw_lines)
                digest = hashlib.sha256("\n".join(norm_lines).encode()).hexdigest()[:12]
                if digest not in by_hash:
                    mca_scores = {}
                    if mca_bin and tcfg["mca_triple"]:
                        for cpu in tcfg["mca_cpus"]:
                            mca_scores[cpu] = run_llvm_mca(
                                mca_bin, tcfg["mca_triple"], cpu, mca_lines
                            )
                    by_hash[digest] = {
                        "tier": tier,
                        "op": op,
                        "width": wname,
                        "shape": shape,
                        "class_id": digest,
                        "representative": item["arm"],
                        "members": [item["arm"]],
                        "metrics": metrics,
                        "mca": mca_scores,
                        "sample_asm": norm_lines[:24],
                    }
                else:
                    by_hash[digest]["members"].append(item["arm"])

            classes_list = list(by_hash.values())
            total_unique += len(classes_list)
            all_classes.extend(classes_list)
            cell_key = f"{tier}/{op}/{wname}/{shape}"
            dedup_map[cell_key] = [
                {
                    "class_id": c["class_id"],
                    "representative": c["representative"],
                    "members": c["members"],
                }
                for c in classes_list
            ]

    # Build Markdown report
    md = [
        "# Static Codegen Equivalence & `llvm-mca` Report",
        "",
        f"- **Total `(tier, op, width, shape, policy)` arms analyzed:** `{total_arms}`",
        f"- **Unique normalized machine-code classes:** `{total_unique}` "
        f"(**`{total_arms / max(total_unique, 1):.2f}x` benchmark deduplication factor**)",
        "",
        "| Tier | Cell | Class | Rep | Members | Hot Insts | Hot Calls | Stack Ops | Primary MCA (cyc/iter, RThr) | Secondary MCA (cyc/iter, RThr) |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    unexpected_simd_calls = []
    for c in all_classes:
        tcfg = TIER_CONFIG[c["tier"]]
        cpus = tcfg["mca_cpus"]
        def fmt_mca(idx: int) -> str:
            if idx >= len(cpus):
                return "—"
            cpu = cpus[idx]
            s = c["mca"].get(cpu, {})
            if "error" in s:
                return f"{cpu}: {s['error']}"
            return f"{cpu}: {s.get('cycles_per_iter')}c (RThr {s.get('block_rthroughput')})"

        m = c["metrics"]
        members_preview = (
            ", ".join(c["members"])
            if len(c["members"]) <= 4
            else f"{', '.join(c['members'][:3])}, +{len(c['members']) - 3} more ({len(c['members'])})"
        )
        md.append(
            f"| `{c['tier']}` | `{c['op']}/{c['width']}/{c['shape']}` | `{c['class_id'][:8]}` | "
            f"`{c['representative']}` | {len(c['members'])} ({members_preview}) | "
            f"{m['hot_insts']} | {len(m['hot_calls'])} | {m['stack_ops']} | "
            f"{fmt_mca(0)} | {fmt_mca(1)} |"
        )
        if c["tier"] in ("v3", "v4", "neon", "wasm128", "wasm128-relaxed") and m["hot_calls"]:
            # Note: WASM f64 fused software path intentionally keeps 1 cold call (`soft_f64x2`)
            unexpected_simd_calls.append(
                (c["tier"], f"{c['op']}/{c['width']}/{c['shape']}", c["representative"], m["hot_calls"])
            )

    if unexpected_simd_calls:
        md.extend(["", "## Hot-Path Call Sites in SIMD Tiers", ""])
        for tier, cell, rep, calls in unexpected_simd_calls:
            md.append(f"- `{tier}` `{cell}` (`{rep}`): `{calls}`")
    else:
        md.extend(["", "## Hot-Path Call Gate", "", "- **PASS:** `0` hot-path `call`/`bl` instructions across all analyzed SIMD (`v3`, `v4`, `neon`, `wasm128`, `wasm128-relaxed`) kernels."])

    report_text = "\n".join(md) + "\n"
    (out_dir / "report.md").write_text(report_text)
    (out_dir / "mca-results.json").write_text(
        json.dumps(
            {
                "total_arms": total_arms,
                "unique_classes": total_unique,
                "dedup_factor": round(total_arms / max(total_unique, 1), 3),
                "unexpected_simd_calls": unexpected_simd_calls,
                "classes": all_classes,
            },
            indent=2,
        )
        + "\n"
    )
    (out_dir / "dedup-classes.json").write_text(json.dumps(dedup_map, indent=2) + "\n")
    tsv_lines = []
    for cell_key, entries in dedup_map.items():
        for entry in entries:
            tsv_lines.append(
                f"{cell_key}\t{entry['representative']}\t{entry['class_id']}\t{','.join(entry['members'])}"
            )
    (out_dir / "dedup-classes.tsv").write_text("\n".join(tsv_lines) + "\n")
    print(
        f"Analyzed {total_arms} arms -> {total_unique} unique classes "
        f"({total_arms / max(total_unique, 1):.2f}x dedup); "
        f"SIMD hot calls: {len(unexpected_simd_calls)}"
    )
    shutil.rmtree(probe_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a two-crate codegen matrix; keep this experiment off main.

Run under run-heavy. Generated sources, complete logs and assembly go to --out.
The checked-in xtask/codegen.py parser resolves assembly aliases before counting.
"""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

STAGES = (1, 16, 128)
HINTS = ("none", "hint", "never")
WRAPPERS = ("none", "hint", "always", "never")


def run(command, cwd, log, env, expected=0):
    print("RUN", " ".join(map(str, command)), flush=True)
    with log.open("w") as output:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=output,
                                stderr=subprocess.STDOUT)
    if result.returncode != expected:
        raise RuntimeError(f"exit {result.returncode}, expected {expected}: {log}")
    print("OK ", log.name, flush=True)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def body(stages):
    lines = ["let mut value = _mm_loadu_ps(input);"]
    for i in range(stages):
        lines += [f"value = _mm_add_ps(_mm_mul_ps(value, _mm_loadu_ps(&coeffs[{i}]))",
                  f"    , _mm_loadu_ps(&coeffs[{127-i}]));"]
    return "\n".join(lines + ["let mut out = [0.0; 4];",
                               "_mm_storeu_ps(&mut out, value);", "out"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit("Use a new output directory; previous artifacts are preserved.")
    out.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[2]
    workspace = out / "workspace"
    env = dict(os.environ, TMPDIR=str(Path.home() / "tmp"),
               RUSTFLAGS="", CARGO_ENCODED_RUSTFLAGS="", CARGO_INCREMENTAL="0")
    arch = json.dumps(str(repo))
    write(workspace / "Cargo.toml", '''[workspace]
members = ["attributes", "provider", "consumer"]
resolver = "2"
[profile.release]
lto = "off"
codegen-units = 1
opt-level = 3
''')
    write(workspace / "attributes/Cargo.toml", '''[package]
name = "inline-probe-attributes"
version = "0.0.0"
edition = "2024"
[lib]
proc-macro = true
''')
    # After rite/arcane expansion, replace only this function's inline hint.
    # This allows the control that existing rite cannot express: inline(never).
    write(workspace / "attributes/src/lib.rs", '''#![forbid(unsafe_code)]
use proc_macro::{Delimiter, TokenStream, TokenTree};
#[proc_macro_attribute]
pub fn hint(mode: TokenStream, item: TokenStream) -> TokenStream {
    let mode = mode.to_string();
    let trees: Vec<_> = item.into_iter().collect();
    let mut output = TokenStream::new();
    let mut i = 0;
    while i < trees.len() {
        if matches!(&trees[i], TokenTree::Punct(p) if p.as_char() == '#')
            && let Some(TokenTree::Group(g)) = trees.get(i + 1)
            && g.delimiter() == Delimiter::Bracket
            && matches!(g.stream().into_iter().next(), Some(TokenTree::Ident(id)) if id.to_string() == "inline")
        {
            i += 2;
        } else {
            output.extend([trees[i].clone()]);
            i += 1;
        }
    }
    let attribute = match mode.as_str() {
        "none" => "",
        "hint" => "#[inline]",
        "never" => "#[inline(never)]",
        "always" => "#[inline(always)]",
        _ => panic!("unknown inline mode"),
    };
    let mut result: TokenStream = attribute.parse().unwrap();
    result.extend(output);
    result
}
''')
    write(workspace / "provider/Cargo.toml", f'''[package]
name = "inline-probe-provider"
version = "0.0.0"
edition = "2024"
[dependencies]
archmage = {{ path = {arch} }}
inline-probe-attributes = {{ path = "../attributes" }}
''')
    prefix = '''#![forbid(unsafe_code)]
use archmage::{arcane, rite, X64V2Token, X64V3Token};
use inline_probe_attributes::hint;
'''
    provider = [prefix, "use archmage::intrinsics::x86_64::*;"]
    sig = "input: &[f32; 4], coeffs: &[[f32; 4]; 128]"
    function_names = []
    for tier in ("v2", "v3"):
        token = f"X64{tier.upper()}Token"
        for stages in STAGES:
            for ih in HINTS:
                core = f"core_{tier}_{ih}_s{stages}"
                function_names.append(core)
                provider.append(f"#[rite({tier})]\n#[hint({ih})]\npub fn {core}({sig}) -> [f32; 4] {{ {body(stages)} }}")
                for wh in WRAPPERS:
                    entry = f"entry_{tier}_{ih}_s{stages}_{wh}"
                    function_names += [entry, "__arcane_" + entry]
                    provider.append(f"#[arcane]\n#[hint({wh})]\npub fn {entry}(_token: {token}, {sig}) -> [f32; 4] {{ {core}(input, coeffs) }}")
    provider += ['''pub fn plain_tiny(value: u32) -> u32 { value.wrapping_add(17) }
pub(crate) fn crate_only(value: u32) -> u32 { value.wrapping_add(17) }
fn private_only(value: u32) -> u32 { value.wrapping_add(17) }
#[macro_export]
macro_rules! private_via_macro {
    ($value:expr) => { $crate::private_only($value) };
}
''']
    # Keep the visibility controls live without changing their visibility.
    provider += ["pub fn visibility_control(x: u32) -> u32 { crate_only(x) ^ private_only(x) }"]
    write(workspace / "provider/src/lib.rs", "\n".join(provider))
    write(workspace / "consumer/Cargo.toml", f'''[package]
name = "inline-probe-consumer"
version = "0.0.0"
edition = "2024"
[features]
private-function = []
crate-function = []
private-macro = []
[dependencies]
archmage = {{ path = {arch} }}
inline-probe-attributes = {{ path = "../attributes" }}
inline-probe-provider = {{ path = "../provider" }}
''')
    consumer = [prefix.replace("arcane, ", ""), "use inline_probe_provider as provider;"]
    cases = []
    checks = []
    for stages in STAGES:
        for ih in HINTS:
            for complexity in ("small", "large"):
                for context in ("baseline", "matching", "superset"):
                    tier = "v2" if context == "superset" else "v3"
                    routes = list(WRAPPERS)
                    if context != "baseline":
                        routes.insert(0, "direct")
                    for route in routes:
                        name = f"case_{len(cases):03}_{tier}_{ih}_s{stages}_{complexity}_{context}_{route}"
                        token_arg = "token: X64V3Token, " if context == "baseline" else ""
                        attrs = "" if context == "baseline" else "#[rite(v3)]\n"
                        if route == "direct":
                            call = f"provider::core_{tier}_{ih}_s{stages}(input, coeffs)"
                        else:
                            proof = "token" if context == "baseline" else f"X64{tier.upper()}Token::from_context()"
                            call = f"provider::entry_{tier}_{ih}_s{stages}_{route}({proof}, input, coeffs)"
                        noise = ["let mut checksum = salt;"]
                        for n in range(1 if complexity == "small" else 128):
                            noise.append(f"checksum = checksum.wrapping_add(noise[{n}]).rotate_left({1+n%29}) ^ {0x9e3779b9 ^ (n*431)}u32;")
                        consumer.append(f"{attrs}#[hint(never)]\npub fn {name}({token_arg}{sig}, salt: u32, noise: &[u32; 128]) -> ([f32; 4], u32) {{\n" + "\n".join(noise) + f"\n({call}, checksum)\n}}")
                        args_ = "token, " if context == "baseline" else ""
                        checks.append(f"let (value, checksum) = {name}({args_}&input, &coeffs, 17, &noise);\n"
                                      f"assert_eq!(value.map(f32::to_bits), expected_{stages}.map(f32::to_bits), \"{name}\");\n"
                                      f"assert_eq!(checksum, checksum_{complexity}, \"{name}\");")
                        cases.append(dict(name=name, tier=tier, body_hint=ih,
                                          stages=stages, caller=complexity,
                                          context=context, route=route))
    consumer += ['''#[inline(never)]
pub fn plain_tiny_caller(x: u32) -> u32 { provider::plain_tiny(x) }
#[cfg(feature = "private-function")]
pub fn invalid_private(x: u32) -> u32 { provider::private_only(x) }
#[cfg(feature = "crate-function")]
pub fn invalid_crate(x: u32) -> u32 { provider::crate_only(x) }
#[cfg(feature = "private-macro")]
pub fn invalid_macro(x: u32) -> u32 { provider::private_via_macro!(x) }
''']
    consumer += ['''#[cfg(test)]
mod tests {
    use super::*;
    use archmage::{arcane, SimdToken};
    fn reference(input: [f32; 4], coeffs: &[[f32; 4]; 128], stages: usize) -> [f32; 4] {
        let mut out = input;
        for i in 0..stages {
            for lane in 0..4 { out[lane] = out[lane] * coeffs[i][lane] + coeffs[127-i][lane]; }
        }
        out
    }
    #[arcane]
    fn verify(token: X64V3Token) {
      for sample in 0..4 {
        let input = [0.125 + sample as f32, -0.5, 1.0, 3.25];
        let coeffs = std::array::from_fn(|i| std::array::from_fn(|j| 0.9 + ((i+j)%7) as f32 * 0.015));
        let noise = std::array::from_fn(|i| i as u32 * 77);
        let checksum_reference = |count| (0..count).fold(17u32, |v, n: usize| {
            v.wrapping_add(noise[n]).rotate_left((1+n%29) as u32) ^ (0x9e3779b9 ^ (n as u32*431))
        });
        let checksum_small = checksum_reference(1);
        let checksum_large = checksum_reference(128);
''']
    consumer += [f"let expected_{n} = reference(input, &coeffs, {n});" for n in STAGES]
    consumer += checks + ['''}}
    #[test]
    fn every_generated_path_matches_scalar_bits() {
        verify(X64V3Token::summon().expect("this x86 codegen experiment requires V3 to execute"));
    }
}
''']
    write(workspace / "consumer/src/lib.rs", "\n".join(consumer))
    write(out / "cases.json", json.dumps(cases, indent=2))
    write(out / "environment.txt", subprocess.check_output(["rustc", "-vV"], text=True)
          + "\nbase/source HEAD: " + subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)
          + "profile: opt-level=3, lto=off, codegen-units=1; RUSTFLAGS empty\n")
    # Rustfmt only touches these generated local workspace members.
    run(["cargo", "fmt", "--all"], workspace, out / "fmt.log", env)
    run(["cargo", "test", "-p", "inline-probe-consumer", "--release", "--lib"], workspace, out / "test.log", env)
    for feature in ("private-function", "crate-function", "private-macro"):
        log = out / f"privacy-{feature}.log"
        run(["cargo", "check", "-p", "inline-probe-consumer", "--features", feature], workspace, log, env, expected=101)
        assert "error[E0603]" in log.read_text(), log
    for package, label in (("inline-probe-consumer", "consumer"), ("inline-probe-provider", "provider")):
        run(["cargo", "asm", "-p", package, "--lib", "--everything", "--keep-mangled", "--no-color"], workspace, out / f"{label}.asm", env)
    spec = importlib.util.spec_from_file_location("codegen", repo / "xtask/codegen.py")
    codegen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codegen)
    # cargo-show-asm 0.2.62 --everything --keep-mangled still demangles
    # .type/.size directives. Parse its original rustc artifact instead.
    for label in ("consumer", "provider"):
        paths = list((workspace / "target/release/deps").glob(f"inline_probe_{label}-*.s"))
        assert len(paths) == 1, paths
        write(out / f"{label}.raw.s", paths[0].read_text())
    asm = codegen.parse((out / "consumer.raw.s").read_text())
    rows = []
    groups = {}
    for case in cases:
        matches = [name for name in asm if case["name"] in name]
        assert len(matches) == 1, (case["name"], matches)
        instructions = asm[matches[0]]
        key = tuple(case[x] for x in ("stages", "body_hint", "caller", "context"))
        groups.setdefault(key, {})[case["route"]] = instructions
        calls = codegen.transfers(instructions)
        external = [x for x in calls if "inline_probe_provider" in x]
        named = []
        for call in external:
            found = [name for name in function_names if name in call]
            assert found, call
            named.append(max(found, key=len))
        rows.append(dict(case, instructions=sum(not x.startswith("LABEL ") for x in instructions),
                         provider_transfers=len(external), targets=";".join(named)))
    with (out / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = []
    for key, bodies in groups.items():
        if "direct" in bodies:
            for route in ("hint", "always"):
                comparisons.append(dict(zip(("stages", "body_hint", "caller", "context", "wrapper", "identical"),
                                            (*key, route, bodies[route] == bodies["direct"]))))
    with (out / "comparisons.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    plain = [name for name in asm if "plain_tiny_caller" in name]
    assert len(plain) == 1
    write(out / "plain-tiny.txt", "\n".join(asm[plain[0]]) + "\n")
    print(f"COMPLETE: {len(rows)} caller cases; assembly, privacy diagnostics and correctness log in {out}", flush=True)


if __name__ == "__main__":
    main()

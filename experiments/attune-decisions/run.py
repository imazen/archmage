#!/usr/bin/env python3
"""Run bounded Rust design probes. No attune implementation is implied.

Run under run-heavy. Every compiler/runtime command retains a full log.
Use a fresh --out directory; previous evidence is never overwritten.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    env = dict(os.environ, TMPDIR=str(Path.home() / "tmp"),
               RUSTFLAGS="", CARGO_ENCODED_RUSTFLAGS="", CARGO_INCREMENTAL="0")
    rows = []

    def write(name, body):
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        return p

    def run(name, command, expected=0, diagnostic=None, cwd=None):
        print("RUN", name, flush=True)
        log = out / f"{name}.log"
        with log.open("w") as f:
            result = subprocess.run(list(map(str, command)), cwd=cwd, env=env,
                                    stdout=f, stderr=subprocess.STDOUT)
        text = log.read_text()
        assert result.returncode == expected, (name, result.returncode, log)
        if diagnostic:
            assert diagnostic in text, (name, diagnostic, log)
        rows.append(dict(case=name, exit=result.returncode, required=diagnostic or "success"))

    def rust_case(name, body, diagnostic=None, execute=False, extra=()):
        source = write(f"sources/{name}.rs", body)
        binary = out / name
        run(name, ["rustc", "--edition=2024", "-Copt-level=3", source, "-o", binary, *extra],
            expected=1 if diagnostic else 0, diagnostic=diagnostic)
        if execute:
            run(name + "_run", [binary])

    write("environment.txt", subprocess.check_output(["rustc", "-vV"], text=True)
          + "source HEAD: " + subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True))

    for provider_fast in (False, True):
        library = out / f"libprovider_{int(provider_fast)}.rlib"
        flags = ["--cfg", 'feature="fast"'] if provider_fast else []
        run(f"provider_{int(provider_fast)}", ["rustc", "--edition=2024", "--crate-type=rlib",
            "--crate-name=provider", "-Copt-level=3", here / "provider.rs", "-o", library, *flags])
        for consumer_fast in (False, True):
            name = f"namespace_p{int(provider_fast)}_c{int(consumer_fast)}"
            binary = out / name
            consumer_flags = ["--cfg", 'feature="fast"'] if consumer_fast else []
            if provider_fast:
                consumer_flags += ["--cfg", "expected_fast"]
            run(name, ["rustc", "--edition=2024", "-Copt-level=3", here / "consumer.rs",
                "--extern", f"dependency_renamed={library}", "-o", binary, *consumer_flags])
            run(name + "_run", [binary])

    rust_case("cfg_leak_missing", '''fn main() {
    let _ = renamed::consumer_cfg_leak!(1);
}''', "E0425", extra=("--extern", f"renamed={out}/libprovider_0.rlib", "--cfg", 'feature="fast"'))
    rust_case("cfg_leak_wrong_selection", '''fn main() {
    // Provider fast is enabled; consumer fast is disabled.
    assert_eq!(renamed::consumer_cfg_leak!(10), 13);
    assert_eq!(renamed::macro_alias!(10), 14);
}''', execute=True, extra=("--extern", f"renamed={out}/libprovider_1.rlib"))

    rust_case("macro_export_collision", '''mod a {
    #[macro_export] macro_rules! __family_work { () => { 1 }; }
} mod b {
    #[macro_export] macro_rules! __family_work { () => { 2 }; }
} fn main() {}''', "E0428")
    rust_case("macro_public_reexport_without_export", '''mod api {
    macro_rules! work { () => { 1 }; }
    pub use work;
} fn main() {}''', "E0364")
    rust_case("descriptor_existing_type_collision", '''struct work;
    enum work {}
    fn main() {}''', "E0428")
    rust_case("descriptor_in_impl", '''struct P;
    impl P { pub enum work {} }
    fn main() {}''', "enum is not supported in")
    rust_case("macro_in_impl", '''struct P;
    impl P { macro_rules! work { () => { 1 }; } }
    fn main() {}''', "macro definition is not supported in")

    rust_case("associated_forwarding_bare", '''struct P;
    impl P {
        fn direct(x: u32) -> u32 { x + 1 }
        fn entry(x: u32) -> u32 { direct(x) }
    } fn main() { assert_eq!(P::entry(1), 2); }''', "E0425")
    rust_case("associated_forwarding_qualified", '''struct P;
    impl P {
        fn direct(x: u32) -> u32 { x + 1 }
        fn entry(x: u32) -> u32 { Self::direct(x) }
    } fn main() { assert_eq!(P::entry(1), 2); }''', execute=True)
    rust_case("nested_outer_generic_capture", '''struct P<T>(T);
    impl<T: Copy> P<T> {
        fn value(&self) -> T {
            fn inner(x: &P<T>) -> T { x.0 }
            inner(self)
        }
    } fn main() {}''', "E0401")
    rust_case("ambiguous_associated_output", '''trait K { type Output; }
    struct P;
    impl K for P { type Output = u8; }
    fn helper() -> P::Output { 1 }
    fn main() {}''', "E0223")
    rust_case("foreign_inherent_helper", '''impl Vec<f32> {
        fn helper(&self) -> f32 { self[0] }
    } fn main() {}''', "E0116")

    rust_case("track_caller_chain", '''#![forbid(unsafe_code)]
    use std::panic::Location;
    #[track_caller] fn both_outer() -> u32 { both_inner() }
    #[track_caller] fn both_inner() -> u32 { Location::caller().line() }
    #[track_caller] fn outer_only() -> u32 { plain_inner() }
    fn plain_inner() -> u32 { Location::caller().line() }
    fn main() {
        let wanted = line!() + 1;
        let got = both_outer();
        assert_eq!(got, wanted);
        let unwanted = line!() + 1;
        let lost = outer_only();
        assert_ne!(lost, unwanted);
        println!("both-chain={got}; outer-only={lost}; external-line={unwanted}");
    }''', execute=True)
    rust_case("expect_duplicated", '''#![deny(unfulfilled_lint_expectations)]
    #[expect(unused_variables)] fn outer() { inner(); }
    #[expect(unused_variables)] fn inner() { let unused = 1; }
    fn main() { outer(); }''', "this lint expectation is unfulfilled")
    rust_case("expect_body_only", '''#![deny(unfulfilled_lint_expectations)]
    fn outer() { inner(); }
    #[expect(unused_variables)] fn inner() { let unused = 1; }
    fn main() { outer(); }''', execute=True)

    # Demonstrates why identifier replacement must respect nested item scopes.
    original = '''struct Outer(u32);
    impl Outer { fn value(&self) -> u32 {
        struct Inner(u32);
        impl Inner { fn value(&self) -> u32 { self.0 } }
        let captured = || self.0;
        captured() + Inner(2).value()
    }} fn main() { assert_eq!(Outer(3).value(), 5); }'''
    rust_case("receiver_original_scopes", original, execute=True)
    rust_case("receiver_blind_rewrite", original.replace("self", "__receiver"),
              "expected one of")

    workspace = out / "workspace"
    write("workspace/Cargo.toml", '''[workspace]
members = ["provider", "traits", "defaults"]
resolver = "2"
[profile.release]
lto = "off"
codegen-units = 1
''')
    write("workspace/provider/Cargo.toml", '''[package]
name = "decision-provider"
version = "0.0.0"
edition = "2024"
[features]
fast = []
''')
    write("workspace/provider/src/lib.rs", (here / "provider.rs").read_text())
    write("workspace/traits/Cargo.toml", f'''[package]
name = "decision-traits"
version = "0.0.0"
edition = "2024"
[dependencies]
archmage = {{ path = {json.dumps(str(repo))}, features = ["avx512"] }}
dependency_renamed = {{ package = "decision-provider", path = "../provider" }}
''')
    write("workspace/traits/src/lib.rs", (here / "traits.rs").read_text() + "\n#[cfg(test)]\nmod call_policy;\n")
    write("workspace/traits/src/call_policy.rs", (here / "call_policy.rs").read_text())
    write("workspace/defaults/Cargo.toml", f'''[package]
name = "decision-defaults"
version = "0.0.0"
edition = "2024"
[features]
avx512 = ["archmage/avx512"]
simd_opt = []
check_auto_v4 = []
check_family_v4 = []
require_omitted = []
[dependencies]
archmage = {{ path = {json.dumps(str(repo))} }}
''')
    write("workspace/defaults/src/lib.rs", (here / "defaults.rs").read_text())
    run("traits_fmt", ["cargo", "fmt", "--all"], cwd=workspace)
    run("traits_test", ["cargo", "test", "--release", "-p", "decision-traits", "--lib", "--", "--nocapture"], cwd=workspace)
    run("traits_clippy", ["cargo", "clippy", "--release", "-p", "decision-traits", "--all-targets", "--", "-Dwarnings"], cwd=workspace)
    run("defaults_auto_without_avx512", ["cargo", "check", "-p", "decision-defaults", "--features", "check_auto_v4"], cwd=workspace)
    run("defaults_family_without_avx512", ["cargo", "check", "-p", "decision-defaults", "--features", "check_family_v4"], expected=101, diagnostic="E0425", cwd=workspace)
    run("defaults_both_with_avx512", ["cargo", "check", "-p", "decision-defaults", "--features", "avx512,check_auto_v4,check_family_v4"], cwd=workspace)
    run("defaults_cfg_disabled", ["cargo", "test", "-p", "decision-defaults", "--lib"], cwd=workspace)
    run("defaults_cfg_enabled", ["cargo", "test", "-p", "decision-defaults", "--lib", "--features", "simd_opt"], cwd=workspace)
    run("defaults_outer_cfg_omits", ["cargo", "check", "-p", "decision-defaults", "--features", "require_omitted"], expected=101, diagnostic="E0425", cwd=workspace)
    shutil.copyfile(workspace / "Cargo.lock", out / "dependencies.lock")

    with (out / "results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=("case", "exit", "required"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"PASS: {len(rows)} command outcomes, including eight SIMD trait/call-policy tests and two cfg fallback runs", flush=True)


if __name__ == "__main__":
    main()

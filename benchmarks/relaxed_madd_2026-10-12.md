# Relaxed SIMD FMA and an independent software regression, 2026-10-12

The earlier relaxed-SIMD profile measured portable FMA but omitted bare
`mul_add`. This follow-up measures the latter explicitly. The portable method
stays software-backed even with `+relaxed-simd`; bare SIMD128 `mul_add` uses
`f32x4.relaxed_madd` / `f64x2.relaxed_madd` in that configuration.

## Measured SIMD128 results

Source: a3372cea0c5596d25e42ac16c7dc3f64417257f1. Rust 1.99.0
(b940084d7, LLVM 23.1.1), Wasmtime 49.0.2, Intel Core Ultra 7 265K, CPU-pinned.
No target-cpu=native. Strict flags are `+simd128,-relaxed-simd`; relaxed flags
are `+simd128,+relaxed-simd`. Both modules run on the same engine and host.

Microseconds per 1,024-vector stream or 1,024-step dependent chain;
median of three process medians. This measures f32x4<Wasm128Token>.

| Operation | Strict stream | Relaxed stream | Strict chain | Relaxed chain |
|---|---:|---:|---:|---:|
| `a * b + c` | 0.452 | 0.452 | 1.079 | 1.079 |
| `mul_add` | 0.447 | 0.438 | 1.079 | 0.679 |
| `mul_add_portable` | 6.138 | 6.140 | 9.848 | 9.847 |

The full compact CSV also includes f64x2 and portable subtraction. Six
benchmark processes passed their exact expected rounding checks, including
cancellation inputs which distinguish fused from unfused results. All 16
cases per process have 30–100 samples, and artifact hashes were verified.
These checks are evidence for the tested inputs, not exhaustive correctness.
The scalar-backend regression below is not covered by these six processes.

## Engine semantics and references

The [WebAssembly numeric specification](https://webassembly.github.io/spec/core/exec/numerics.html#op-frelaxed-madd)
allows fused or unfused relaxed madd. Rust exposes that through
[`f32x4_relaxed_madd`](https://doc.rust-lang.org/core/arch/wasm32/fn.f32x4_relaxed_madd.html).
That permission is different from a guaranteed, correctly rounded FMA.

The version-pinned [Wasmtime 49.0.2 implementation](https://github.com/bytecodealliance/wasmtime/blob/v49.0.2/crates/cranelift/src/func_environ.rs#L5413-L5451)
uses native FMA when available. Without it, deterministic mode uses fused
software, while ordinary mode emits separate multiply and add. This differs
from the current specification page's description of its deterministic
profile; the engine's option must be checked against its implementation.

An independent WAT probe computes `(1 + 2^-23) * (1 - 2^-23) - 1` using the
explicit relaxed instruction. Observations on the same host:

| Wasmtime configuration | Result bits | Rounding |
|---|---|---|
| Default | `0xa8800000` | fused |
| Deterministic mode | `0xa8800000` | fused |
| Engine FMA code generation disabled | `0x00000000` | unfused |
| FMA code generation disabled, deterministic | `0xa8800000` | fused |

Disabling the engine's FMA code generation is not a physical no-FMA CPU
benchmark. The first deterministic timing attempt incorrectly expected
unfused behavior and failed its assertion. It is preserved and excluded
from the timings above; the separate WAT probe establishes the actual result.

## Independent scalar correctness failure

[Sergey Davidoff's FMA analysis](https://shnatsel.github.io/implementing-fma-finding-bugs-in-std/)
describes a subnormal double-rounding bug and a corrected f64/TwoSum/round-to-odd
implementation. Its numerical foundation is [Boldo and Melquiond's paper,
Emulation of FMA and correctly-rounded sums: proved algorithms using rounding
to odd](https://guillaume.melquiond.fr/doc/08-tc.pdf).

The published counterexample is:

```text
a = f32::from_bits(0x97000800)
b = f32::from_bits(0x1cfff001)
c = f32::from_bits(0x00010002)
correct fma result = 0x00010001
```

The scalar f32x1 portable method in the direct-libm source above returns
`0x00010002` in the strict WASM build. Its SIMD128 portable method passes the
same assertion. This was a correctness bug in that revision, not an acceptable
performance tradeoff; the resolution below restores the corrected helper. Native no-FMA execution of this case has not been measured.

An independent rational-arithmetic calculation gives the exact result in
units of the minimum f32 subnormal as `9007405413171199 / 137438953472`.
The distance to 0x00010001 is 68719476735 units of a common denominator;
the distance to 0x00010002 is 68719476737 units. Thus the expected bit pattern
does not rely on another implementation of libm as an oracle.

The earlier software timing data remain measurements, but passing their
selected input checks did not establish correctness on this example. The
native-std candidate still explicitly leaves WASM on libm and therefore does
not address this failure. No production FMA policy is changed by this report.

## Correctness fix validation

The follow-up fix routes WASM scalar f32 through the existing `fmaf_soft`
helper in both std and no_std builds. Native libm selection, f64, and bare
relaxed-SIMD madd are unchanged. This is a correctness fix; no new speedup
is claimed for the corrected snapshot.

`software_subnormal_rounding` first failed on the old WASM helper with
65538 instead of 65537. It then passed on the corrected helper and portable
add/sub for scalar f32x1/x4/x8/x16, including the sign-negated example.
Expected bits come from the exact rational calculation above, independently
of Rust or libm's FMA implementations.

The fused-arithmetic, documentation-example, and scalar-parity suites passed
with strict SIMD and relaxed SIMD, each with std enabled and disabled.
They also passed with baseline compile flags `-simd128,-relaxed-simd`
(the test runner still enabled SIMD for explicitly annotated test functions).
Relaxed SIMD also passed with Wasmtime's host FMA code generation disabled,
using the existing unfused-result assertions. Native fused-arithmetic tests,
Clippy with warnings denied, generator/registry checks, and intrinsic
soundness checks passed. Existing WASM warnings and the documentation test's
preexisting ignored panic case were not changed.

The validation used Rust 1.99.0 and Wasmtime 49.0.2 on the same host as the
measurements above. Representative resource reports from `run-heavy`:

```text
native fused tests: rc=0 3s | peak-RSS 0.47GiB | min-avail 25927MiB | peak-load 0.44
strict no_std suites: rc=0 3s | peak-RSS 0.36GiB | min-avail 25948MiB | peak-load 0.58
relaxed, engine FMA off: rc=0 2s | peak-RSS 0.37GiB | min-avail 26030MiB | peak-load 0.45
```

Native scalar performance policy and the known libm/Miri CPUID failure are
separate unresolved work; this WASM fix does not change either.

## Reproduction and resources

The isolated experiment's `just wasm-madd` preserves the full failing scalar
regression. `just wasm-simd-madd` runs the successful strict/relaxed SIMD128
comparisons and then the rejected deterministic expectation. `just probe-relaxed`
runs the separate WAT observation on all four engine configurations. Each
recipe takes the Wasmtime executable path; timing recipes additionally take a
CPU index. Existing output files are never overwritten.

```text
Full scalar/SIMD attempt: rc=1 20s | peak-RSS 0.37GiB | min-avail 25683MiB | peak-load 1.19
SIMD128 timing + rejected deterministic attempt: rc=1 30s | peak-RSS 0.37GiB | min-avail 25526MiB | peak-load 1.06
Independent WAT probe: rc=0 1s | peak-RSS 0.02GiB | min-avail 26193MiB | peak-load 0.29
```

Raw outputs, binaries, exact commands, source-generating scripts, and SHA256
manifest are archived on dev at `/mnt/v/output/archmage/fma_relaxed_review_2026-10-12/`.
The timing CSV accompanies this report. These results cover Wasmtime on one
CPU; browsers and other engines have not been measured here.

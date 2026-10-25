# Fused multiply-add vs a separate multiply and add, 2026-10-25

Which is faster depends on where the dependency runs, not on the data.
Interleaved zenbench comparison in one binary (`fused`: `mul_add` / std
`mul_add`, one `vfmadd`; `separate`: `a * b + c`, a `vmul` and a `vadd`, checked
in the disassembly), V3 kernels on `f32x8`, `f64x4` and scalar `f64` inside a
V3 function, 1,024 operations per sample. Rust 1.99.0, release,
`codegen-units=1`, CPU 2 pinned, under `run-heavy`. zenbench reported both runs
reliable.

- i265: Intel Core Ultra 7 265K (Lion Cove P-core).
- r7900x: AMD Ryzen 9 7900X (Zen 4).

Separate time / fused time (95% paired CI where it matters):

| Shape | i265 | r7900x |
|---|---:|---:|
| Multiplicand chain `x = x * y + z` (Horner) | 1.58–1.61× | 1.50–1.56× |
| Accumulator chain `acc = a[i] * b[i] + acc` | 0.49–0.58× | 0.78–0.80× |
| Four vector accumulators | 1.02× | 1.02× |
| Four scalar `f64` accumulators | 0.58× | 0.84× |
| Stream `out = a * b + c` | 0.81–1.04× | 0.85–1.00× |

When the chain runs through the multiplicand, both halves of a separate
multiply-add sit on it and the fused instruction is shorter. When it runs only
through the addend, the multiplies leave the critical path and only the add
remains; an add has lower latency than an FMA on both cores, much lower on the
Intel P-core. With enough independent work the two are equal. Fused rounding
is also more accurate, so this is a speed trade, not a free choice.

Consequences: no single lowering of a "fastest" multiply-add wins everywhere;
reductions with one or few accumulators are faster written as `a * b + acc`
on these CPUs, and Horner-style chains as a fused multiply-add.

Data: `/mnt/v/output/archmage/fma_vs_separate_2026-10-25/` on `dev`.

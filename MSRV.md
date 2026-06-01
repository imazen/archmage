# How Rust 1.89 brought the safe SIMD story together for Archmage

Archmage requires Rust 1.89+. This isn't arbitrary — 1.89 is the version where Rust's SIMD story finally came together. Four releases turned SIMD from "unsafe everything" to "zero unsafe with full feature coverage."

## Rust 1.86 — safe `#[target_feature]` calls (April 2025)

**Stabilization:** [`target_feature_11`](https://github.com/rust-lang/rust/issues/69098)

Before 1.86, every `#[target_feature]` function was implicitly `unsafe fn`. Calling one required `unsafe`, even from another function with the same features. This made composing SIMD code painful — you'd chain `unsafe` blocks through every layer.

1.86 changed the rules:

```
  Rust's #[target_feature] call rules (1.86+)

  ┌─────────────────────────┐         ┌──────────────────────────────┐
  │  fn normal_code()       │ unsafe  │ #[target_feature(avx2, fma)] │
  │                         │────────▶│ fn simd_work()               │
  │  (no target features)   │         │                              │
  └─────────────────────────┘         └──────────────┬───────────────┘
                                                     │
          Calling simd_work() from                   │ safe
          normal_code() requires                     │ (subset of
          unsafe { }. The caller                     ▼ caller's features)
          has fewer features.          ┌──────────────────────────────┐
                                       │ #[target_feature(avx2)]      │
                                       │ fn simd_helper()             │
                                       └──────────────────────────────┘

  Caller has same or superset features? Safe call. No unsafe needed.
  Caller has fewer features?            Rust requires an unsafe block.
```

**What this enabled for archmage:** `#[arcane]` generates an inner `#[target_feature]` function and a safe outer wrapper. The inner function can call `#[rite]` functions (which also carry `#[target_feature]`) safely — no `unsafe` needed at call sites between matching feature contexts. Before 1.86, every such call required `unsafe`.

## Rust 1.87 — safe value-based intrinsics (May 2025)

**Stabilization:** [rust-lang/rust#134790](https://github.com/rust-lang/rust/pull/134790)

Before 1.87, every `std::arch` intrinsic was `unsafe fn` — even pure value operations like `_mm256_add_ps` that can't cause UB regardless of CPU state. You'd write `unsafe { _mm256_add_ps(a, b) }` for an operation that's semantically identical to `a + b` on wider registers.

1.87 split intrinsics into two categories:

| Category | Example | Safety |
|----------|---------|--------|
| Value-based (arithmetic, shuffle, compare, bitwise) | `_mm256_add_ps`, `_mm256_shuffle_epi32` | **Safe** inside `#[target_feature]` |
| Pointer-based (loads, stores) | `_mm256_loadu_ps(*const f32)` | Still `unsafe` (raw pointer) |

```rust
#[target_feature(enable = "avx2,fma")]
fn example() {
    let a = _mm256_setzero_ps();        // safe
    let b = _mm256_set1_ps(2.0);        // safe
    let c = _mm256_fmadd_ps(a, a, b);   // safe — value-based

    // Only memory ops still need unsafe (raw pointer):
    let v = unsafe { _mm256_loadu_ps(ptr) };
}
```

**What this enabled for archmage:** Inside `#[arcane]` and `#[rite]` functions, most intrinsic calls are safe with no wrappers needed. The remaining gap — pointer-based memory ops — is closed by [`safe_unaligned_simd`](https://crates.io/crates/safe_unaligned_simd), which shadows `core::arch` load/store functions with reference-based versions (`_mm256_loadu_ps` takes `&[f32; 8]` instead of `*const f32`). Archmage re-exports these through `import_intrinsics`.

Combined with 1.86, this means the body of an `#[arcane(import_intrinsics)]` function contains zero `unsafe`. Your crate can use `#![forbid(unsafe_code)]`.

## Rust 1.88 — `as_chunks` on slices (June 2025)

**Stabilization:** [rust-lang/rust#142208](https://github.com/rust-lang/rust/pull/142208)

`as_chunks::<N>()` and `as_chunks_mut::<N>()` split a slice into fixed-size array references and a remainder, at zero cost:

```rust
let (chunks, remainder) = data.as_chunks::<8>();
for chunk in chunks {
    // chunk: &[f32; 8] — exact type needed by safe SIMD loads
    let v = _mm256_loadu_ps(chunk);
}
```

Before this, the pattern was:

```rust
for chunk in data.chunks_exact(8) {
    let arr: &[f32; 8] = chunk.try_into().unwrap();
    let v = _mm256_loadu_ps(arr);
}
```

The `try_into().unwrap()` is optimized away, but it's noisy and obscures intent. `as_chunks` makes SIMD-width iteration a one-liner.

## Rust 1.89 — AVX-512 and SHA-512/SM3/SM4 intrinsics (August 2025)

**Tracking issues:** [rust-lang/rust#111137](https://github.com/rust-lang/rust/issues/111137) (AVX-512), [rust-lang/rust#126624](https://github.com/rust-lang/rust/issues/126624) (SHA-512/SM3/SM4)

This was the big one. Before 1.89, writing `#[target_feature(enable = "avx512f")]` on stable Rust was a compiler error. The entire AVX-512 family — the most capable SIMD ISA on x86 — was locked behind `#![feature(avx512_target_feature)]` on nightly.

1.89 stabilized two feature gates at once:

### `avx512_target_feature` — 22 target features, ~857 intrinsics

**Target feature PR:** [rust-lang/rust#138940](https://github.com/rust-lang/rust/pull/138940)
**Intrinsics PR:** [rust-lang/stdarch#1819](https://github.com/rust-lang/stdarch/pull/1819)

**AVX-512 core (14 features):** `avx512f`, `avx512bw`, `avx512cd`, `avx512dq`, `avx512vl`, `avx512bf16`, `avx512bitalg`, `avx512fp16`, `avx512ifma`, `avx512vbmi`, `avx512vbmi2`, `avx512vnni`, `avx512vp2intersect`, `avx512vpopcntdq`

**VEX variants (5 features):** `avxifma`, `avxneconvert`, `avxvnni`, `avxvnniint8`, `avxvnniint16`

**Crypto / Galois field (3 features):** `gfni`, `vaes`, `vpclmulqdq`

Plus mask types (`__mmask8/16/32/64`), mask operations (`_kadd_mask*`, `_kand_mask*`, etc.), comparison enums (`_MM_CMPINT_*`), and 256 permutation constants (`_MM_PERM_*`).

[Full intrinsics-per-module breakdown →](https://imazen.github.io/archmage/archmage/reference/rust-189-simd/)

### `sha512_sm_x86` — 3 target features, 10 intrinsics

**Target feature PR:** [rust-lang/rust#140767](https://github.com/rust-lang/rust/pull/140767)
**Intrinsics PR:** [rust-lang/stdarch#1796](https://github.com/rust-lang/stdarch/pull/1796)

**Target features:** `sha512`, `sm3`, `sm4`

**Intrinsics:** `_mm256_sha512msg1_epi64`, `_mm256_sha512msg2_epi64`, `_mm256_sha512rnds2_epi64`, `_mm_sm3msg1_epi32`, `_mm_sm3msg2_epi32`, `_mm_sm3rnds2_epi32`, `_mm256_sm4key4_epi32`, `_mm256_sm4rnds4_epi32`, `_mm_sm4key4_epi32`, `_mm_sm4rnds4_epi32`

### What this enabled for archmage

Archmage maps capability tokens to `#[target_feature]` attributes. Every AVX-512 token depends on these features compiling on stable:

| Token | Features |
|-------|----------|
| `X64V4Token` | `avx512f`, `avx512bw`, `avx512cd`, `avx512dq`, `avx512vl` |
| `X64V4xToken` | Above + `avx512vbmi`, `avx512vbmi2`, `avx512bitalg`, `avx512vnni`, `avx512vpopcntdq`, `avx512ifma`, `avx512bf16`, `gfni`, `vaes`, `vpclmulqdq` |
| `Avx512Fp16Token` | Above + `avx512fp16` |

Runtime detection (`is_x86_feature_detected!("avx512f")`) also works on stable now, so `summon()` compiles without feature gates.

## The result

On Rust 1.89, you can write AVX-512 code with zero `unsafe`:

```rust
use archmage::prelude::*;

#[arcane(import_intrinsics)]
fn sum_512(_token: X64V4Token, data: &[f32; 16]) -> f32 {
    let v = _mm512_loadu_ps(data);            // safe: reference-based
    let sum = _mm512_reduce_add_ps(v);        // safe: value-based
    sum
}

fn main() {
    if let Some(token) = X64V4Token::summon() {
        println!("{}", sum_512(token, &[1.0; 16]));
    }
}
```

No `#![feature(...)]`, no `unsafe`, no nightly. `#![forbid(unsafe_code)]` works. That's why the MSRV is 1.89.

## Newer-stable intrinsics *above* the MSRV — the version gate

Some hardware paths want an intrinsic that became `#[stable]` *after* 1.89.
Example: the aarch64 NEON half-precision converters `vcvt_f32_f16` /
`vcvt_f16_f32` are stable only since **1.94** (`stdarch_neon_f16`). A static
`cfg` on them would drag the whole crate's MSRV up to 1.94 — losing every user
on 1.89–1.93.

archmage uses **two different mechanisms** here, picked by whether the intrinsic
has a stable version yet:

| Intrinsic state | Mechanism | Why |
|---|---|---|
| **Stable since a known version** (e.g. NEON-f16 @ 1.94) | `rustversion` + `target_arch` gate; both arms covered by the **normal** CI (MSRV `cargo check` below the bound, stable `test-aarch64`/`test-cross` above it) | The stabilization version is a fact you can name; `#[rustversion::since(X)]` selects the path by toolchain version with **no build script** and a trusted, dep-light proc-macro. No dedicated boundary job — see "Both arms are covered by the normal CI matrix" below. |
| **Nightly-only** (no stable version yet) | a tiny **try-compile probe** that runs *only* under nightly | Nightly feature gates churn (renamed/removed between nightlies), so a blind `#![feature(...)]` breaks. The probe enables the path iff it actually compiles on *this* nightly, else falls back. This is the one place feature-detection genuinely beats version-matching. |

### Stable case — `rustversion` + arch (the NEON-f16 gate)

The NEON-f16 hardware kernels live in `magetypes/src/simd/generic/convert_f16.rs`,
inside `#[cfg(target_arch = "aarch64")]`, gated by toolchain version:

```rust
// HW kernel: only compiled where the intrinsics are stable.
#[cfg(target_arch = "aarch64")]
#[rustversion::since(1.94)]
#[archmage::arcane(import_intrinsics)]
fn neon_f16_decode_hw(token: Arm64V2Token, input: &[u16], output: &mut [f32]) { /* vcvt_f32_f16 */ }

// Path selector — two paired items, one per side of the bound:
#[cfg(target_arch = "aarch64")]
#[rustversion::since(1.94)]   // HW exists → runtime token decides HW vs software
fn neon_f16_decode_select(token: NeonToken, ..) { if fp16 { hw } else { software } }

#[cfg(target_arch = "aarch64")]
#[rustversion::before(1.94)]  // HW not compiled → software only, no MSRV bump
fn neon_f16_decode_select(token: NeonToken, ..) { software }
```

- The **version gate** (`since`/`before`) selects whether the HW impl *exists*.
- The **runtime token** (`Arm64V2Token::summon()` → proves `fp16`) selects
  whether to *use* it on the actual CPU. The two are orthogonal.
- On **rustc ≥ 1.94** the HW kernel compiles and the runtime picks it on
  `fp16` hardware (Cortex-A55+/Apple M1+/Graviton 2+).
- On **rustc 1.89–1.93** the HW kernel is not compiled at all; the same source
  builds clean with the branchless **software** kernel. **No compile error, no
  MSRV bump** (proven by `cargo +1.93 check --target aarch64-unknown-linux-gnu`).

Why not the build-script capability probe we trialed first? It worked (~80 ms
one-time cold compile, ~0 hot — see `benchmarks/build_script_overhead_*.md`),
but `rustversion` is more maintainable: no build script, a trusted dtolnay
proc-macro, and none of the bespoke-probe gotchas (e.g. a cross-`core`
false-negative when probing an uninstalled target).

#### Both arms are covered by the normal CI matrix

The stabilization version is a documented fact (`#[stable(since = "1.94.0")]`),
not a guess, and both arms of the gate are already exercised by the existing CI
— **no dedicated boundary job**:

- **below the bound** — the **MSRV 1.89 `aarch64 Linux`** job `cargo check`s the
  crate, compiling the `before(1.94)` software arm with no MSRV violation.
- **above the bound** — the **`test-aarch64`** (native ubuntu-24.04-arm) and
  **`test-cross`** (aarch64 QEMU) jobs run at stable, compiling **and running**
  the `since(1.94)` HW arm so `vcvt_*` resolves and the bit-identity test passes.
- the software arm's *correctness* also runs on every x86 test job (the kernel is
  generic, branchless, arch-independent) and under Miri (nightly) for UB.

What this does **not** independently re-validate is the exact `1.94` literal —
a too-low/too-high bound is only detectable by a cell at `1.94 ± 1`, which
neither 1.89 nor stable is. That was judged not worth a perpetual dedicated
`cross` job: the version is documented in std, and a wrong literal surfaces at
the next toolchain reaching the gap. If a *future* intrinsic's stabilization
version is genuinely uncertain, add a **transient** 2-cell `cross` matrix at
`<ver> - 1` / `<ver>` to pin it, then remove it once confirmed.

### Nightly-only case — the try-compile probe pattern (not currently shipped)

There is **no nightly-only intrinsic in archmage today**, so **no `build.rs`
ships** (keeping build time + complexity at zero for downstream consumers). This
is the documented pattern to re-introduce when one is actually needed:

1. A `build.rs` (re-introduced for this case only) that, **gated on
   `version_check`/`rustversion`-style nightly detection**, try-compiles a tiny
   `#![feature(<gate>)] #![no_std]` snippet using the *intrinsic*. On success it
   emits `cargo:rustc-cfg=archmage_nightly_<name>` (+ matching
   `cargo:rustc-check-cfg`).
2. The path is gated `#[rustversion::nightly] #[cfg(archmage_nightly_<name>)]`,
   with a `#[rustversion::not(nightly)]`-or-`cfg(not(...))` software fallback.
3. A CI nightly cell exercises that the probe + gated path compile.

The probe (not a version compare) is correct here precisely because the answer
"does *this* nightly still accept this feature + intrinsic" can change build to
build, with no stable version to name.

### Adding the next stable intrinsic

For an intrinsic that has stabilized at a known version (AVX-512-FP16
`_mm512_cvtph_ps`, SVE, …):

1. Write the HW kernel inside `#[cfg(target_arch = "<arch>")]`, gated
   `#[rustversion::since(<ver>)]`.
2. Write the paired `*_select` helper: `#[rustversion::since(<ver>)]` (HW +
   runtime token) and `#[rustversion::before(<ver>)]` (software only).
3. The normal CI already covers both arms (MSRV `cargo check` below `<ver>`,
   stable `test-aarch64`/`test-cross` above it). If `<ver>` itself is uncertain,
   add a **transient** 2-cell `cross` matrix at `<ver> - 1` / `<ver>` to pin it,
   then remove it once confirmed.

The crate adapts to whatever toolchain compiles it, rather than baking one
static MSRV decision.

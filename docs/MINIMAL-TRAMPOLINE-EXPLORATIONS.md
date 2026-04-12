# Minimal Trampoline Explorations

Design iterations exploring how small a `#[target_feature]` dispatch macro can be
while remaining sound and `#![forbid(unsafe_code)]`-compatible. Each iteration
builds on lessons from the previous one.

Produced during a design analysis session comparing archmage's macro stack (~6,200
lines) against the theoretical minimum. Third-party crates surveyed: multiversion
0.8, pulp 0.22, fearless_simd 0.4, target-feature-dispatch 3.1, safe_arch 1.0.

## Table of Contents

1. [Iteration 1: Bare wrapper (not a trampoline)](#iteration-1-bare-wrapper)
2. [Iteration 2: Single-tier trampoline with CPUID](#iteration-2-single-tier-trampoline)
3. [Iteration 3: Attribute partitioning and rejection](#iteration-3-attribute-partitioning)
4. [Iteration 4: Constant-CPUID assumption (reject list drops to zero)](#iteration-4-constant-cpuid)
5. [Iteration 5: Rich signatures](#iteration-5-rich-signatures)
6. [Iteration 6: Multi-tier multi-arch dispatch](#iteration-6-multi-tier)
7. [Iteration 7: Compile-time elision](#iteration-7-compile-time-elision)
8. [Iteration 8: Suffix-mangle convention](#iteration-8-suffix-mangle)
9. [Iteration 9: Unified two-macro design](#iteration-9-unified-design)
10. [Key findings](#key-findings)

---

## Iteration 1: Bare wrapper

**Problem:** Not a real trampoline. Just wraps an `unsafe { inner() }` call.
No CPUID check — relies on the caller to have already proved features exist
(e.g., via a token parameter).

**~30 lines.** Does NOT perform any CPU detection.

```rust
#[proc_macro_attribute]
pub fn trampoline(attr: TokenStream, item: TokenStream) -> TokenStream {
    let features = parse_macro_input!(attr as LitStr);     // "avx2,fma"
    let f = parse_macro_input!(item as ItemFn);
    let (vis, sig, body) = (&f.vis, &f.sig, &f.block);
    let inner = format_ident!("__trampoline_{}", sig.ident);

    let args = sig.inputs.iter().map(|a| match a {
        FnArg::Typed(PatType { pat: box Pat::Ident(p), .. }) => Ok(&p.ident),
        _ => Err(syn::Error::new_spanned(a, "trampoline: only `name: T` params")),
    }).collect::<Result<Vec<_>, _>>();
    let args = match args { Ok(v) => v, Err(e) => return e.to_compile_error().into() };

    let mut inner_sig = sig.clone();
    inner_sig.ident = inner.clone();

    quote!(
        #[target_feature(enable = #features)]
        #[inline]
        fn #inner_sig #body

        #[inline(always)]
        #vis #sig {
            // SAFETY: caller asserts CPU supports `#features`.
            unsafe { #inner(#(#args),*) }
        }
    ).into()
}
```

**Verdict:** Not useful alone. The caller must write `unsafe { }` to call the
wrapper (since the wrapper has no `#[target_feature]` but calls one that does),
which defeats `#![forbid(unsafe_code)]`. Shown only as a baseline.

---

## Iteration 2: Single-tier trampoline

**Problem:** Add the actual CPUID check — cached in an `AtomicU8`, with a
scalar fallback. This is the first "real" trampoline.

**~60 lines.**

```rust
#[proc_macro_attribute]
pub fn trampoline(attr: TokenStream, item: TokenStream) -> TokenStream {
    // attr: features = "avx2,fma", fallback = path::to::scalar_impl
    let TrampolineArgs { features, fallback } = parse_macro_input!(attr);
    let f = parse_macro_input!(item as ItemFn);
    let (vis, sig, body) = (&f.vis, &f.sig, &f.block);
    let inner = format_ident!("__tramp_{}", sig.ident);

    let args = collect_ident_params(sig)?;

    // One is_x86_feature_detected! per feature, ANDed
    let checks = features.split(',').map(|f| {
        let f = f.trim();
        quote!(std::is_x86_feature_detected!(#f))
    });

    let mut inner_sig = sig.clone();
    inner_sig.ident = inner.clone();

    quote!(
        #[target_feature(enable = #features)]
        #[inline]
        fn #inner_sig #body

        #vis #sig {
            use core::sync::atomic::{AtomicU8, Ordering};
            static CACHE: AtomicU8 = AtomicU8::new(0); // 0=unknown 1=no 2=yes
            let supported = match CACHE.load(Ordering::Relaxed) {
                2 => true,
                1 => false,
                _ => {
                    let v = #(#checks)&&*;
                    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
                    v
                }
            };
            if supported {
                // SAFETY: CPUID for these features just succeeded above.
                unsafe { #inner(#(#args),*) }
            } else {
                #fallback(#(#args),*)
            }
        }
    ).into()
}
```

**Audit surface:** One `unsafe { }` block with the CPUID check 5 lines above it.

**forbid(unsafe_code):** Compatible — the inner is a safe `fn` (Rust 2024
edition), the `unsafe { }` block is proc-macro-generated.

---

## Iteration 3: Attribute partitioning

**Problem:** What attributes on the user's function should go where? What
function signature forms should be rejected?

### Attribute routing

| Attribute | Inner | Wrapper | Reason |
|---|---|---|---|
| `#[target_feature]` (user's) | Reject | Reject | Macro owns this |
| `#[inline]` / `#[inline(always)]` | Strip (macro sets its own) | Pass through | User's hint applies to the public symbol |
| `#[no_mangle]`, `#[export_name]` | Strip | Keep | Symbol-table attrs — link error if both have them |
| `#[allow/warn/deny/forbid/expect]` | Keep | Keep | Lint scopes match the body |
| `#[track_caller]` | Keep | Keep | Panic locations |
| `#[cold]` | Strip | Keep | Trampoline branch is cold; inner is hot |
| `#[must_use]`, `#[deprecated]`, `#[doc]` | Strip | Keep | Public API metadata |
| `extern "ABI"` | Keep | Keep | ABI is part of the call type |
| Visibility | Private | User's vis | Inner is hidden |

### Rejection list (later revised)

| Modifier | Action | Why |
|---|---|---|
| `async fn` | Reject | Future's `poll` can run in non-feature context |
| `const fn` | Accept | CTFE doesn't run on the host CPU |
| `#[naked]` | Defer to rustc | E0658 already |
| Returning `impl Fn*` / `Box<dyn Fn*>` / `fn(...)` | Reject | Closure-escape launders feature requirement |
| User `#[target_feature]` | Reject | Macro manages this |

**~90 lines** total (30 base + 30 partitioning + 30 rejection).

---

## Iteration 4: Constant-CPUID assumption

**Key insight:** CPU features do not change during process lifetime on tier-1
Rust targets. Once CPUID returns "avx2 available," it's available for the
entire process — including closures, futures, fn pointers, struct fields,
thread-spawned work, and anything else constructed inside a trampolined call.

**Consequence:** The entire rejection list from iteration 3 drops to zero.

- `async fn`: Sound — future constructed inside trampolined call, polled on same CPU.
- `impl Fn` / `Box<dyn Fn>` return: Sound — closure inherits features, CPU still has them.
- `fn` pointer coercion: Sound — pointer callable from anywhere in same process.
- Thread spawn: Sound — same CPU, same features.

All soundness concerns were about feature availability changing between
construction and use. Under constant-CPUID, it doesn't.

The reject list becomes:
- `#[naked]` — rustc rejects already (E0658)
- User `#[target_feature]` — reject for clarity, not soundness

**Enforcement for forbid(unsafe) users:** Under `#![forbid(unsafe_code)]`,
the user physically cannot bypass the trampoline — calling the inner directly
requires an `unsafe { }` block the lint forbids. This mechanically enforces
"all SIMD entry goes through a trampoline" without relying on user discipline.

**~30 lines** (same as iteration 2 — the reject logic is removed, not added).

---

## Iteration 5: Rich signatures

**Problem:** Support generics, self receivers, pattern parameters.

### Generic type/const forwarding (+20 lines)

Turbofish forwarding so `__inner::<T, N>(args)` resolves:

```rust
fn build_turbofish(generics: &syn::Generics) -> TokenStream {
    let params: Vec<_> = generics.params.iter().filter_map(|p| match p {
        GenericParam::Type(tp)  => Some(&tp.ident),
        GenericParam::Const(cp) => Some(&cp.ident),
        GenericParam::Lifetime(_) => None,
    }).map(|i| quote!(#i)).collect();
    if params.is_empty() { quote!() } else { quote!(::<#(#params),*>) }
}
```

### Self receivers in inherent impls (+25 lines)

Detect `FnArg::Receiver`, emit method-style call `self.__inner(args)`.
Both the wrapper and inner live in the same impl scope, so `Self` resolves
naturally. Trait method impls are rejected (sibling expansion can't add
non-trait methods).

### Pattern parameter rebinding (+15 lines)

Wildcards (`_: T`) and destructured patterns (`(a, b): (T, U)`) are renamed
to `__tramp_arg_N` in both sigs, with `let original_pattern: T = __tramp_arg_N;`
prepended to the inner body.

### `unsafe fn` propagation (+5 lines)

```rust
let was_unsafe = sig.unsafety.is_some();
let fallback_call = if was_unsafe {
    quote!(#[allow(unused_unsafe)] unsafe { #fallback(#(#args),*) })
} else {
    quote!(#fallback(#(#args),*))
};
```

**Running total: ~125 lines.**

---

## Iteration 6: Multi-tier multi-arch dispatch

**Problem:** Generate dispatchers that try multiple tiers in priority order.

### Const registry (~80 lines)

One field per tier. The features string is the single source of truth for
BOTH `#[target_feature(enable = ...)]` AND runtime `is_*_feature_detected!()`.
No separate "detect" list — that's a soundness bug (see below).

```rust
pub struct Tier {
    pub suffix:   &'static str,  // "v3", "v4", "neon"
    pub features: &'static str,  // comma-separated, used for BOTH target_feature AND detection
    pub arch:     &'static str,  // "x86_64", "aarch64", "wasm32"
    pub priority: u8,            // higher = tried first within same arch
}

pub const TIERS: &[Tier] = &[
    Tier { suffix: "v4x", arch: "x86_64", priority: 90,
        features: "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,vpclmulqdq,vaes,gfni,bmi1,bmi2" },
    Tier { suffix: "v4",  arch: "x86_64", priority: 80,
        features: "avx512f,avx512bw,avx512cd,avx512dq,avx512vl" },
    Tier { suffix: "v3",  arch: "x86_64", priority: 50,
        features: "avx2,fma,bmi1,bmi2,f16c,lzcnt,popcnt,movbe" },
    Tier { suffix: "v2",  arch: "x86_64", priority: 30,
        features: "sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b" },
    Tier { suffix: "v1",  arch: "x86_64", priority: 10,
        features: "sse,sse2" },
    Tier { suffix: "arm_v3", arch: "aarch64", priority: 80,
        features: "neon,crc,rdm,dotprod,fp16,aes,sha2,sha3,fhm,fcma,i8mm,bf16" },
    Tier { suffix: "arm_v2", arch: "aarch64", priority: 50,
        features: "neon,crc,rdm,dotprod,fp16,aes,sha2" },
    Tier { suffix: "neon",   arch: "aarch64", priority: 30,
        features: "neon" },
    Tier { suffix: "wasm",   arch: "wasm32",  priority: 50,
        features: "simd128" },
];
```

### Soundness rule: features == detect

**features and detection MUST be the same list.** Every feature in
`#[target_feature(enable = ...)]` must be checked by
`is_*_feature_detected!()`. If you enable 8 features but check 3, the proof
obligation doesn't cover the other 5.

Concrete failure mode: a VM masks CPUID to expose AVX2 but hide LZCNT.
`is_x86_feature_detected!("avx2")` returns true, the trampoline calls the
inner, LLVM emits `lzcnt` (because `#[target_feature]` enabled it), and the
program traps.

archmage's `token-registry.toml` gets this right. The comment at the top:

> *LLVM deduplicates redundant features in `#[target_feature]` — listing
> the full set is harmless and eliminates the class of bugs where
> "minimal" lists diverge from "cumulative" lists.*

One field. Derive both uses from it. No drift.

### Dispatch expansion

Multi-tier generates one cache + check per tier, grouped by arch:

```rust
pub fn process(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")] {
        static C_V4: AtomicU8 = AtomicU8::new(0);
        if check(&C_V4, is_x86_feature_detected!("avx512f")
                     && is_x86_feature_detected!("avx512bw") && /* ... */) {
            return unsafe { process_v4(data) };
        }
        static C_V3: AtomicU8 = AtomicU8::new(0);
        if check(&C_V3, is_x86_feature_detected!("avx2")
                     && is_x86_feature_detected!("fma") && /* ... */) {
            return unsafe { process_v3(data) };
        }
    }
    #[cfg(target_arch = "aarch64")] {
        static C_NEON: AtomicU8 = AtomicU8::new(0);
        if check(&C_NEON, is_aarch64_feature_detected!("neon")) {
            return unsafe { process_neon(data) };
        }
    }
    process_scalar(data) // unconditional, every arch, always the last line
}
```

The scalar fallback is always the last line — no cfg, no check, no unsafe.
On uncovered arches (riscv, etc.), no cfg block matches, every SIMD check
is absent, and execution falls straight through to scalar.

**Running total: ~285 lines** (125 rich-sig + 80 registry + 65 multi-tier + 15 multi-arch).

---

## Iteration 7: Compile-time elision

**Problem:** When the binary is compiled with `-Ctarget-cpu=native` or
`-Ctarget-feature=+avx2,+fma,...`, the runtime dispatch should disappear
entirely — no atomic load, no branch, direct call to the best tier.

### Mechanism

Use `#[cfg(target_feature = "...")]` to detect ambient features at compile
time. Emit one elision block per tier before the runtime dispatch:

```rust
pub fn process(data: &[f32]) -> f32 {
    // v4 features in ambient target → direct call, no runtime check.
    #[cfg(all(
        target_feature = "avx512f", target_feature = "avx512bw",
        target_feature = "avx512cd", target_feature = "avx512dq",
        target_feature = "avx512vl",
    ))]
    { return unsafe { process_v4(data) }; }

    // v3 features in ambient, but NOT v4 (mutually exclusive with above).
    #[cfg(all(
        all(target_feature = "avx2", target_feature = "fma", /* ... */),
        not(all(target_feature = "avx512f", /* ... */)),
    ))]
    { return unsafe { process_v3(data) }; }

    // Runtime path — only compiled when no tier is fully ambient.
    #[cfg(not(any(
        all(target_feature = "avx512f", /* v4 */),
        all(target_feature = "avx2", /* v3 */),
    )))]
    {
        // cached CPUID dispatch (same as iteration 6)
        ...
        process_scalar(data)
    }
}
```

Higher tiers are excluded from lower tiers' cfgs via `not(all(...))` to
ensure mutual exclusivity — exactly one block is compiled for any given build.
The runtime path gets a `#[cfg(not(any(...)))]` wrapping everything.

When compiled with `-Ctarget-cpu=haswell`, the v3 elision block fires, the
function becomes `unsafe { process_v3(data) }`, and since `process_v3`'s
features are a subset of ambient, rustc can inline the body directly. The
`AtomicU8`, `is_x86_feature_detected!` calls, and scalar fallback all disappear.

archmage uses the same pattern in its generated `summon()`:
```rust
#[cfg(all(target_feature = "avx2", target_feature = "fma", /* ... */))]
{ Some(unsafe { Self::forge_token_dangerously() }) }
#[cfg(not(all(/* ... */)))]
{ match CACHE.load(Ordering::Relaxed) { /* runtime path */ } }
```

**Added lines: ~30.** Running total: **~315 lines.**

---

## Iteration 8: Suffix-mangle convention

**Key insight:** The function name suffix (`_v3`, `_v4`, `_neon`) is the
natural lookup key for the tier registry. The macro can infer the
`#[target_feature]` set from the suffix instead of requiring it as a macro
argument.

### `#[cpu_tier]` — the primitive annotation

```rust
#[cpu_tier]
fn dot_v3(a: &[f32; 8], b: &[f32; 8]) -> f32 { /* avx2 body */ }
```

Macro splits the name at the last `_`, looks up `v3` in `TIERS`, applies
`#[target_feature(enable = "avx2,fma,...")]` + `#[inline]` + `#[cfg(target_arch = "x86_64")]`.

Implementation (~30 lines):

```rust
#[proc_macro_attribute]
pub fn cpu_tier(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let f = parse_macro_input!(item as ItemFn);
    let name = f.sig.ident.to_string();

    let (_base, suffix) = match name.rsplit_once('_') {
        Some(p) => p,
        None => return err(f.sig.ident.span(),
            "#[cpu_tier] requires a suffixed name like `foo_v3`"),
    };
    let tier = match find_tier(suffix) {
        Some(t) => t,
        None => return err(f.sig.ident.span(),
            &format!("unknown tier suffix `_{suffix}`")),
    };

    let features = tier.features;
    let arch     = tier.arch;
    let attrs    = &f.attrs;
    let vis      = &f.vis;
    let sig      = &f.sig;
    let body     = &f.block;

    quote!(
        #[cfg(target_arch = #arch)]
        #(#attrs)*
        #[target_feature(enable = #features)]
        #[inline]
        #vis #sig #body
    ).into()
}
```

### Dispatch with suffix list

```rust
#[dispatch(self, _v3, _neon)]
fn compute_v4(data: &[f32]) -> f32 { /* avx-512 body */ }

#[cpu_tier]
fn compute_v3(data: &[f32]) -> f32 { /* avx2 body */ }

#[cpu_tier]
fn compute_neon(data: &[f32]) -> f32 { /* neon body */ }

fn compute_scalar(data: &[f32]) -> f32 { data.iter().sum() }

// User calls: compute(&data)
```

The `#[dispatch]` attribute on `compute_v4`:
1. Promotes `compute_v4` itself to `#[cpu_tier]` (applies target_feature from suffix).
2. Generates a dispatcher `compute` (base name, suffix stripped).
3. The dispatch list `(self, _v3, _neon)` specifies priority order.
4. The macro reads each suffix's arch from TIERS, groups by arch, emits cfg blocks.
5. `compute_scalar` is the unconditional fallback at the bottom.

---

## Iteration 9: Unified two-macro design

**Key insight:** The presence or absence of a token parameter determines the
dispatch mode. Two macros cover all use cases:

### `#[cpu_tier]` — raw annotation

Apply `#[target_feature]` from suffix. No wrapper, no dispatch. The caller
must be in a matching-feature context.

```rust
#[cpu_tier]
fn helper_v3(chunk: &mut [f32; 8]) { /* inner helper */ }
```

### `#[dispatch]` — two modes from one macro

**Mode A: No token parameter → CPUID dispatcher with scalar fallback.**

```rust
#[dispatch(self, _v3, _v2)]
fn compute_v4(data: &[f32]) -> f32 { /* avx-512 body */ }
```

The function IS the top-tier implementation. The macro generates a dispatcher
`compute()` that does cached CPUID, tries tiers in order, falls through to
`compute_scalar()`. The scalar fallback is **always required** in this mode —
without a token, the caller hasn't proved anything, so the function must
handle "CPU supports nothing."

**Mode B: Token parameter → safe wrapper (no CPUID, no fallback).**

```rust
#[dispatch]
fn compute_v4(token: X64V4Token, data: &[f32]) -> f32 { /* avx-512 body */ }
```

The token proves features exist. No CPUID check. No scalar fallback. The macro
generates a safe wrapper that calls the `#[target_feature]` inner via proc-macro
`unsafe { }`. This is archmage's `#[arcane]` under the unified name.

### Mode switch rule

| | No token parameter | Has token parameter |
|---|---|---|
| What it generates | CPUID dispatcher + `_scalar` fallback | Safe wrapper (token = proof) |
| Scalar fallback | **Required** | Not needed |
| CPUID check | Yes (cached) | No |
| Equivalent archmage macro | `incant!` + dispatch | `#[arcane]` |

### Summary

| Macro | Lines | Role |
|---|---|---|
| `#[cpu_tier]` | ~30 | Apply `#[target_feature]` from suffix |
| `#[dispatch]` (tokenless, with tier list) | ~120 | Runtime CPUID dispatcher |
| `#[dispatch]` (with token) | ~50 | Safe wrapper (arcane-equivalent) |
| `registry.rs` (const TIERS) | ~80 | Feature/arch/suffix lookup |
| Parse helpers | ~40 | Shared syn utilities |
| Safe intrinsics re-exports | ~200 | Memory-op wrappers for forbid(unsafe) |
| **Total** | **~520** | |

---

## Key findings

### Soundness

- **Constant-CPUID assumption** (tier-1 Rust targets): CPU features don't
  change during process lifetime. This collapses all closure-escape,
  async-future, fn-pointer-laundering concerns. The proof at the dispatch
  site covers every later use of any value constructed inside the dispatched
  call.

- **features == detect**: The feature list used for `#[target_feature(enable)]`
  and the feature list checked by `is_*_feature_detected!()` must be identical.
  Checking only "headline" features (e.g., `avx2` for the v3 bundle) while
  enabling more (e.g., `lzcnt`, `bmi2`) is a soundness bug. archmage's
  `token-registry.toml` gets this right with a single `features` array per token.

- **`unsafe fn` vs `unsafe { }`**: `#![forbid(unsafe_code)]` rejects `unsafe fn`
  declarations from proc macros but allows `unsafe { }` blocks from proc macros.
  The trampoline inner must be a safe `fn` with `#[target_feature]` (Rust 2024
  edition), never `unsafe fn`. multiversion 0.8 generates `unsafe fn` inners
  and therefore breaks forbid(unsafe).

### Architecture

- **forbid(unsafe_code) is the spec, not a feature.** The library must enable
  forbid for users who want it. It must not require it. Macros emit
  forbid-compatible code; users who don't set forbid can write their own
  `unsafe { }` blocks to bypass the macros when useful.

- **Token parameter presence/absence** cleanly switches between "I am the
  runtime dispatch entry (CPUID + scalar fallback required)" and "the caller
  already proved features (no CPUID, no fallback)."

- **The scalar fallback** is always the unconditional last line of the
  dispatcher. No cfg. No check. No unsafe. It's just `compute_scalar(data)`.
  On uncovered arches, no SIMD cfg block matches and execution falls through.

### Comparison with archmage

| | Minimal (iteration 9) | archmage |
|---|---|---|
| Macro LoC | ~520 | ~6,200 |
| Macros | 2 (`#[cpu_tier]`, `#[dispatch]`) | 5+ (`#[arcane]`, `#[rite]`, `incant!`, `#[magetypes]`, `#[autoversion]`) |
| Registry | Const array (~80 lines) | TOML + xtask codegen (~3,500 lines) |
| Body inspection | None | `rewrite.rs` walks token trees for `incant!` rewriting |
| Token hoisting | Via `#[dispatch]` with token param | Via `Token::summon()` + pass-through |
| forbid(unsafe) | Yes | Yes |
| Compile-time elision | Yes | Yes (via `compiled_with()`) |

### What the minimal version does not cover

- **Token-as-proof type system.** No zero-sized proof types, no trait
  hierarchy, no downcast methods. The constant-CPUID assumption replaces
  type-level feature tracking.
- **Body-level `incant!` rewriting.** No body walking. Inner calls are
  explicit: `helper_v3(args)` from `process_v3`, safe under Rust 1.86+
  matching-feature rules.
- **`#[magetypes]` text substitution.** No `Token`/`f32xN`/`LANES` placeholders.
  Write per-tier implementations manually, or use a `macro_rules!` to clone
  one body across tiers.
- **`#[autoversion]` auto-vectorization.** No auto-clone of scalar code across
  tiers. Add as a ~50-line extension if needed.
- **Cross-tier dispatch from inside a tier.** Archmage's `incant!` rewriter
  turns `incant!(helper(x))` inside `process_v3` into a direct call to
  `helper_v3(x)`. The minimal version makes this the user's responsibility.

### Upstream issue

[imazen/archmage#26](https://github.com/imazen/archmage/issues/26) — proposed
adding suffix inference to `#[rite]` as an additive, non-breaking change.

# Spec: Zero-Overhead Cross-Macro Calls via incant! Rewriting

## Problem

When one `#[autoversion]` function calls another, the inner call goes through
the runtime dispatcher on every iteration — 5.8x slower than the gold
standard (`#[arcane]` + `#[rite]`). The dispatcher can't be inlined because
it lacks `#[target_feature]`, creating an LLVM optimization boundary.

```rust
#[autoversion]
fn transform(v: &[f32; 4], scale: f32) -> [f32; 4] {
    let n = normalize(v);  // calls normalize's DISPATCHER every time
    // ...
}
```

## Solution

Tier macros (`#[autoversion]`, `#[arcane]`, `#[rite]`) scan the function body
for `incant!()` calls and rewrite them to direct calls to the matching tier's
trampoline — bypassing the dispatcher entirely.

```rust
#[autoversion]
fn transform(v: &[f32; 4], scale: f32) -> [f32; 4] {
    let n = incant!(normalize(v));  // rewritten per tier
    // ...
}

// In V3 variant: incant!(normalize(v)) → normalize_v3(__token, v)
// In V4 variant: incant!(normalize(v)) → normalize_v4(__token, v)
// In scalar variant: incant!(normalize(v)) → normalize_scalar(ScalarToken, v)
// In dispatcher: incant!(normalize(v)) → full dispatch (unchanged)
```

## Proven: Trampoline Bypass = Zero Overhead

Benchmark (`examples/autoversion_call_chain.rs`):

| Version | ns/vec | vs gold |
|---------|--------|---------|
| autoversion re-dispatch | 5.6 | 6x slower |
| **trampoline bypass** | **0.94** | **1.0x** |
| arcane+rite gold standard | 0.93 | 1.0x |
| scalar | 1.25 | 1.3x slower |

LLVM inlines through the `#[inline(always)]` trampoline, then inlines the
`#[target_feature]` inner function (matching features). Assembly confirms:
zero `call` instructions, full loop vectorization across both functions.

## Design Principles

### Tokens everywhere

Every tier variant takes its tier's token. No tokenless variants, no token
forging. The token is zero-sized — passing it has zero runtime cost. It
maintains the safety invariant: you can only call a `#[target_feature]`
function if you can prove the features are available.

### Soundness over performance

The trampoline pattern is preserved. The ONLY `unsafe` is calling a
`#[target_feature]` function from non-`#[target_feature]` context. If the
user's function is `unsafe fn`, the compiler still checks the body normally —
the generated `unsafe` block wraps only the call, not the body.

### The trampoline evaporates

The trampoline `fn_v3(token, args)` is `#[inline(always)]` with a single
`unsafe { __arcane_fn_v3(token, args) }` call. LLVM eliminates it completely.
The `__arcane_fn_v3` inner has matching `#[target_feature]`, so it inlines
into the caller. Net cost: zero.

## Architecture

### Current call chain (3 layers)

```
normalize(v)                              — dispatcher (summon + branch)
  └→ normalize_v3(token, v)               — trampoline (#[inline(always)])
       └→ __arcane_normalize_v3(token, v)  — #[target_feature] inner
```

### With incant! rewriting

The tier macro rewrites `incant!(normalize(v))` in V3 body to:
```
normalize_v3(__token, v)                  — direct call to trampoline
  └→ __arcane_normalize_v3(token, v)      — inlines (matching features)
```

Dispatcher is bypassed. Trampoline inlines. Inner inlines. Zero overhead.

## Rewriting Rules

### When does rewriting happen?

Only when `incant!()` appears inside a function body processed by a tier
macro (`#[autoversion]`, `#[arcane]`, `#[rite]`). The outer macro walks the
body token stream, finds `incant!(...)` invocations, and rewrites them before
emitting the code.

### What does it rewrite to?

| Outer context | Token available as | `incant!(foo(args))` becomes |
|---------------|-------------------|------------------------------|
| autoversion V3 variant | `__token: X64V3Token` | `foo_v3(__token, args)` |
| autoversion V4 variant | `__token: X64V4Token` | `foo_v4(__token, args)` |
| autoversion scalar variant | `ScalarToken` | `foo_scalar(ScalarToken, args)` |
| autoversion dispatcher | (none) | full dispatch (unchanged) |
| `#[arcane]` with `X64V3Token` | `token: X64V3Token` | `foo_v3(token, args)` |
| `#[rite(v3)]` with token | `token: X64V3Token` | `foo_v3(token, args)` |
| `#[rite(v3)]` tokenless | (no token) | **not supported** — use token form |
| plain code (no macro) | (none) | full dispatch (unchanged) |

### Token passing

The macro knows the token ident because:
- `#[autoversion]` auto-injects it (e.g., `__token`) or parses it from `ScalarToken` param
- `#[arcane]` parses it from the function signature via `find_token_param`
- `#[rite]` with token parses it the same way

The rewritten call passes this token to the callee's trampoline.

### Downcasting (caller tier > callee tier)

When V4 code calls a function that only has V3 variants:

```rust
incant!(normalize(v), [v3, neon, scalar])
```

The macro sees: caller is V4, callee's highest compatible x86 tier is V3.
V4 ⊃ V3, so it emits:

```rust
normalize_v3(__token.v3(), v)
```

The `.v3()` method downcasts `X64V4Token → X64V3Token`. This is a no-op at
runtime (both are zero-sized) and safe (V4 implies V3).

**Downcast methods exist on all tokens**: `.v3()`, `.v2()`, `.v1()`, etc.

### Upgrading (caller tier < callee tier)

Not supported by rewriting. V3 code wanting a V4 fast path must use
conditional dispatch explicitly:

```rust
if let Some(v4) = X64V4Token::summon() {
    fast_path(v4, data)
} else {
    // ...
}
```

### Tier list matching

When `incant!` has an explicit tier list:

```rust
incant!(foo(args), [v4(cfg(avx512)), v3, neon, scalar])
```

The rewriter picks the **highest compatible tier** from the list for the
current context:

| Caller tier | Available tiers | Picks | Downcast |
|-------------|----------------|-------|----------|
| V4 | v4, v3, neon, scalar | v4 | none |
| V3 | v4, v3, neon, scalar | v3 | none |
| V4x | v4, v3, neon, scalar | v4 | `.v4()` |
| neon | v4, v3, neon, scalar | neon | none |
| scalar | v4, v3, neon, scalar | scalar | none |

Without an explicit tier list, the default tier list is used.

If no compatible tier exists in the list, it's a **compile error**.

### Cross-architecture

`incant!` inside a V3 body on x86 won't try to call `foo_neon` — cfg guards
handle this. The rewriter only considers tiers that match the caller's
architecture.

## What Changes in Each Macro

### `#[autoversion]`

**Body scanning**: When generating each tier variant, walk the token stream
for `incant!(...)` patterns. Replace with direct suffixed call + token.

**Token injection**: Already injects a token param (`__token`) into each
variant. The token ident is known at rewrite time.

**Dispatcher**: unchanged — still generates `summon()` + branch.
`incant!()` in the dispatcher body is NOT rewritten (no tier context).

### `#[arcane]`

**Body scanning**: The sibling/inner function body is scanned for `incant!()`.
The tier is derived from the token type via `canonical_token_to_tier_suffix`.

**Token ident**: parsed from the function signature by `find_token_param`.

### `#[rite]`

**Single-tier with token**: Same as arcane — scan body, derive tier from
token type, rewrite.

**Single-tier tokenless** (`#[rite(v3)]`): **Cannot participate** in incant!
rewriting as a caller because there's no token to pass. The function itself
is a valid **callee** (other code can `incant!` into it).

**Multi-tier**: Each generated variant knows its tier. Scan body per variant,
rewrite incant! calls to matching tier.

### `incant!`

**In tier context (rewritten by outer macro)**: Never reaches proc macro
expansion — the outer macro rewrites it to a direct call before `incant!`
even runs.

**In plain context (no outer macro)**: Expands as today — if-let-else chain
with `summon()` for each tier.

**Simplification**: Replace labeled block `'__incant: { break ... }` with
plain `if-let-else` chain. Both work for expression and void-return
contexts.

## Deprecations (Prerequisites)

### Tokenless `#[rite(v3)]` (no token param)

Tokenless rite functions can't participate as incant! rewriting callers (no
token to pass). They can still be callees, but the asymmetry is confusing.

**Deprecate**: Emit a warning suggesting token-based form. No removal yet.
Users calling tokenless rite from `#[arcane]` or `#[rite]` contexts can call
them directly (they have matching `#[target_feature]`, no incant! needed).

### `stub` option

**Removed** (done). Now a compile error pointing to `incant!`.

## Implementation Order

1. **Replace labeled blocks with if-let-else chains** in `incant!` codegen
   (simplification, no behavior change)

2. **Add body scanning infrastructure** — a function that walks a
   `proc_macro2::TokenStream` looking for `incant!(...)` patterns and
   returns the locations + parsed arguments

3. **Implement rewriting in `#[autoversion]`** — the highest-value target
   (autoversion-to-autoversion calls are the main pain point). Add snapshot
   tests for rewritten expansion.

4. **Implement rewriting in `#[arcane]`** — scan the sibling body for
   `incant!()` and rewrite based on the token type.

5. **Implement rewriting in `#[rite]` with token** — same as arcane.

6. **Add tier compatibility logic** — for downcasting (V4 caller → V3
   callee), determine the right downcast method and emit it.

7. **Deprecate tokenless `#[rite(v3)]`** — warning only.

8. **Benchmark** — reproduce the autoversion chain benchmark and verify
   zero overhead.

## Non-Goals

- **Rewriting plain function calls**: `normalize(v)` is NOT rewritten.
  Only `incant!(normalize(v))` is. The user opts in explicitly.

- **Cross-crate visibility**: The `_v3` trampolines must be visible to the
  caller. Same-module is automatic. Cross-module requires `pub` variants.
  Cross-crate requires the callee crate to export them.

- **Token forging**: Not used anywhere. Tokens flow from `summon()` through
  the call chain. The trampoline pattern handles the `unsafe` boundary.

- **Changing the trampoline pattern**: The 3-layer structure (dispatcher →
  trampoline → inner) is preserved. The trampoline is the safety firewall.
  incant! rewriting just skips the dispatcher, calling the trampoline
  directly.

## Example: Full Expansion

```rust
// User writes:
#[autoversion]
fn transform(v: &[f32; 4], scale: f32) -> [f32; 4] {
    let n = incant!(normalize(v));
    [n[0] * scale, n[1] * scale, n[2] * scale, n[3] * scale]
}

// V3 variant after rewriting:
#[arcane]
fn transform_v3(__token: X64V3Token, v: &[f32; 4], scale: f32) -> [f32; 4] {
    let n = normalize_v3(__token, v);  // direct trampoline call
    [n[0] * scale, n[1] * scale, n[2] * scale, n[3] * scale]
}

// Dispatcher (no rewriting — no tier context):
fn transform(v: &[f32; 4], scale: f32) -> [f32; 4] {
    if let Some(t) = X64V4Token::summon() { transform_v4(t, v, scale) }
    else if let Some(t) = X64V3Token::summon() { transform_v3(t, v, scale) }
    else { transform_scalar(ScalarToken, v, scale) }
}
```

At runtime with V3: `transform()` → `transform_v3(token, v, scale)` →
(trampoline inlines) → `__arcane_transform_v3(token, v, scale)` which calls
`normalize_v3(token, v)` → (trampoline inlines) →
`__arcane_normalize_v3(token, v)`. LLVM sees one big `#[target_feature]`
region and vectorizes across both functions.

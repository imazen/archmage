# Attune migration contract — draft

Status: design requirement, 2026-10-07. No legacy-to-attune converter or production
deprecation warning is implemented. The [diagnostic probe](experiments/migration-diagnostics/README.md)
tests warning delivery only. Keep implementation experiments off main.

The requested endpoint is a complete migration converter. Every legacy attribute
and incant invocation should receive an invocation-specific replacement, including
attributes whose behavior would otherwise change. An LLM should be able to apply
the supplied edits without guessing names, tiers, token positions or inline policy.
This extends the [attune specification](ATTUNE_SPEC.md).

## One conversion planner, two outputs

Use the same conversion rules to produce:

1. Deprecation notes at legacy macro invocation sites. Show the exact replacement
   for the identified source range, not a generic link to the new attribute.
2. An explicit source migration command producing reviewable edits, with check,
   diff and apply modes. A possible spelling is `cargo archmage migrate`; this
   command does not exist yet and its packaging is not selected.

Rules consume already-parsed macro arguments and a normalized description of
the old expansion: signature, proof input, body context, outputs, names,
visibility, feature requirements, cfg, dispatch policy and effective attributes.
The compile-time path must not parse an enclosing impl, scan files or discover
cross-crate definitions. The explicitly invoked source tool can inspect source
context and Cargo metadata; that cost stays outside normal builds.

The planner returns minimal range edits and their preconditions, not a string
representing an entire reformatted source file. The source tool preserves comments
and untouched bytes. TokenStream display text is not a faithful source editor:
comments and original formatting are not retained. The warning can print a
replacement attribute/call and separately identified insertions; coupled changes
need a complete patch from the source tool.

Do not independently maintain a warning template and a different CLI rewrite.
Input macro version matters: use the locked dependency/version and versioned
conversion rules. Published behavior is the migration baseline, not an older
unpublished attune experiment. Shared implementation should not force full source
parsing or formatting dependencies onto every proc-macro user.

## Warning feasibility and limits

On installed stable Rust 1.98.1, a macro-generated deprecated local constant and
reference emitted a dynamic, multiline replacement note at the attribute/call
site. allow and expect suppressed/fulfilled the diagnostic; deny promoted it to
an error. The expression probe evaluated its argument once. A deprecated
proc-macro definition also warned, but its static note cannot vary by invocation.

The emitted JSON contained no suggested replacement edits. This technique does
not provide cargo fix integration by itself. Direct proc_macro Diagnostic use
failed with E0658; the [Rust documentation](https://doc.rust-lang.org/proc_macro/struct.Diagnostic.html)
also marks that API unstable. Test the chosen mechanism on the actual MSRV before
shipping; the probe is not MSRV evidence.

Illustrative future diagnostic once the target syntax is implemented:

```text
warning: legacy #[rite(v3)]
replace this attribute with:
    #[inline]
    #[attune(v3)]
```

This preserves the old direct body's inline policy explicitly. It is not a
promise that every conversion is an attribute-only edit. Do not deprecate the
user's generated public function just to warn about its defining macro: that
would warn downstream callers at the wrong place. Avoid duplicate warnings for
aliases, nested rewrites and each generated tier.

Retain normal Rust lint behavior, including deny(warnings). Do not override the
consumer's lint policy to force or hide a warning. Adding deprecations can break
deny-warnings builds, so the conversion path must be available at rollout.

## Required semantic coverage

This is a required coverage matrix, not completed converter coverage.

| Legacy surface | Replacement must preserve |
|---|---|
| arcane | Same public signature/name, concrete or generic proof, argument position, trait bounds, associated output, private target-feature body and effective wrapper/body inline policies |
| simd_fn, token_target_features_boundary | Same arcane contract; identify the actual alias/import spelling at the edit site |
| rite, token_target_features | Direct feature context, same-name versus suffixed mode, imports, generics and effective inline policy |
| magetypes | Exact generated surface, placeholders, define aliases, proof placement, generic substitution, architecture and feature gates |
| autoversion | Dispatcher and variant names/signatures, old tier set and gates, scalar/default behavior, receiver handling, attribute routing and dispatch policy |
| incant runtime dispatch | Exact available entry convention, implicit legacy fallback made explicit, detection/reselection behavior, argument evaluation and token placement |
| incant without token | Covered direct call, own-tier selection and no hidden runtime upgrade |
| incant with a supplied token | Exact token handling, unmatched-token behavior, generic contracts and absence of newly introduced CPU probing |
| simd_route, dispatch_variant | The corresponding incant behavior and actual imported name |

All aliases above are present in archmage-macros/src/lib.rs at the inspected
baseline. Import aliases, qualified paths, renamed dependencies, cfg-disabled
code, examples and macro_rules templates require source-tool coverage too.
The existing [inventory](experiments/attune-migration/README.md) supplies cases,
not proof that they are converted.

Preserve the old effective behavior rather than assuming old spelling had the
intended effect. For example, current arcane filters user inline attributes and
gives its ordinary wrapper inline(always); current autoversion routes user
attributes to the dispatcher. A converter cannot indiscriminately copy them onto
every output. Route expect only where fulfilled and track_caller across the
necessary forwarding chain.

Old default sets must become explicit where new defaults differ. In particular,
autoversion's unconditional V4 and magetypes' gated V4 cannot both become a new
wildcard without changing one contract. Changes to default selection order or
fallback can change observable numerical behavior too.

## What this requires from the new API

- Same-signature generic proof entries must be expressible. The
  [cargo expand probe](experiments/generic-token-trampoline/README.md) establishes
  the existing lowering: preserve T and its bounds in wrapper and body. Keeping
  arcane forever is not the completed migration plan. Entry syntax remains open.
- The new surface must express explicit-proof dispatch semantics where legacy
  incant uses them. Replacing every such call with reattune would introduce a
  different selection policy. Syntax remains open.
- Naming dictionaries and explicit selectors must preserve old output names,
  gates and visibility. A rename alone cannot adapt an argument position.
- Inline and other attribute placement must be specified per generated role so
  the planner can render an exact equivalent instead of relying on new defaults.
- Unsupported trait/method shapes need a concrete extraction or explicit adapter
  migration. Function attributes remain function-only; any source restructuring
  is an explicit converter operation, with enclosing bounds made explicit.

If an old invocation cannot yet be represented, report the specific missing
capability during development and implement it or a validated source rewrite
before claiming complete migration. Do not print a plausible but unverified
replacement, silently change a signature, or turn a runtime behavior into a
different compile-time contract and call it equivalent. Safety/bug corrections
that intentionally change behavior need a separate, explicit migration policy.

## Verification and rollout gates

Keep existing legacy tests. Add conversion tests which:

- Compile the original and the converted program against their intended macro
  versions, including forbid(unsafe_code) and strict lint configurations.
- Compare public API surfaces, types, generic/associated contracts, cfg matrices,
  runtime outputs and argument evaluation/ownership.
- Inspect expansion and cross-crate assembly for direct calls, wrappers and
  dispatcher placement; do not use code-text snapshots alone as correctness proof.
- Apply converter edits to source, build the result and verify a second conversion
  makes no edits. Preserve comments, import aliases and unrelated attributes.
- Check diagnostics and replacement ranges, including multiple legacy macros,
  nesting, lint scopes, methods, generated templates and unavailable source spans.
- Compile/run converted examples and docs across supported architectures, and
  cover old scalar/default and AVX-512 behavior explicitly.

No end-user compile-time regression remains a release gate. Measure the legacy
warning path as well as converted consumers, with matched compiler, dependencies,
tiers and outputs. Warning construction/output is work too; do not claim it free.
Full source analysis and pretty-printing belong to the explicit converter, not
to every macro expansion. Avoid repeated body parsing and emitting one giant
replacement for each generated variant.

Publish the working replacement API and converter before enabling deprecations
that direct users to them. The release plan must identify any supported syntax
still lacking an exact conversion; none may be hidden behind a claim of a full
converter. The current work establishes design requirements and diagnostic
feasibility only, with no implemented legacy conversion rules.

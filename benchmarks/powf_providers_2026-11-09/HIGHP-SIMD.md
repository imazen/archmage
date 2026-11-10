# Canonical HIGHP integration and SIMD

The maintained [ISA reference](../../docs/POWF_REFERENCE.md) owns the numerical
bounds, implementation characteristics, measured performance and promotion
status. This phase changes production HIGHP source; the older provider/scalar
studies retain their original source identities.

Review of the integration covers const encodings and runtime arms, whole-vector
validation order, canonical special-value precedence, bounded table indexing,
fused node order, packed lane ordering, scalar fallback, std policy preservation,
WASM f64 probe wiring, no_std and MSRV. Every table index is explicitly masked;
no unchecked indexing, fast-math switch or global floating mode is added.

Review corrected a wide-context x4 selector that left work in a scalar tail,
a WASM probe accidentally nested in the f32 getter, scalar-target unused
helpers, and a duplicate Just recipe. All failed attempts remain in the
measurement root. No established assertion, stack limit or admission gate
was weakened. The final public additions are the two canonical HIGHP presets;
the API snapshots include them on x86-64, AArch64 and WASM.

A first MPFR manifest referenced mutable Cargo output. Its identity guard
rejected a later changed executable. Authoritative full, hard-witness and
Gamma replays instead use the separately frozen `dev-wide` worker, with
16, 16 and 46 provider/width executions respectively. The earlier outputs
remain identified as earlier evidence. M6 V1/V2 workers are both frozen and
relocation manifests preserve their original hashes.

Actual V3 x4 disassembly contains the nine packed f64 FMA nodes and no calls
in the vector entry. V4x x4 retains YMM FMAs; V4x x8 uses ZMM FMAs. Actual
worker sizes include numerical-protocol and batch adapters and are not
isolated-kernel instruction counts. Both native ARM machines execute the
same canonical graph and the exhaustive NaN-family scan.

The performance conclusion is held. Streams and chains, widths, domains and
validation policies have measured ties and reversals. The reference lists
both observed costs and outstanding timing axes. Source-level savings and
lower instruction counts do not establish the required per-cell speed bound.

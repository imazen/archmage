# Power policy and gamma optimization study

Unpublished study based on draft 4ccaf1cf. Production changes remain unpushed.
The initial benchmark covers exponent-spread positive bases (about 2^-60 to
2^60), linear-uniform C0 sRGB gamma bases, and negative/integer/special controls.
These differ from historical corpora; comparisons use identical inputs within
this study, never ratios inferred from old reports. Boundary/log-spaced and
whole-transfer coverage is being added before the final matrix.

The independently frozen baseline contains 79 Rust files byte-identical to
4ccaf1cf, with renamed unpublished package manifests only. Run
`python3 validate-baseline.py` before builds. It supplies old scalar, selected
scalar, whole-loop scalar feature context and packed policy controls. Current
reference, current selected policy and std controls have distinct labels.
`std` is a timing control, never the correct-rounding oracle.

Actual const-generic policies include correct, MIDP/LOWP Full and
PositiveNormalPrecondition, each fused/separate where valid. Uniform gamma
exponents are separate const-generic loop instantiations using exact f32 bits.
Whole-loop V3/NEON/WASM scalar contexts distinguish feature-boundary overhead
from polynomial work. Packed precondition tails use valid (1,0) filler.

Initial perf profiles found software f64 FMA in the generic correct policy and
software f32 FMA in fused MIDP. Routing fix 329fdc54 keeps the reference policy
non-dispatch while ordinary policies select the ordinary correct core. A shared
positive-normal entry avoids the longer special classifier without changing
polynomial steps or the accurate fallback. Quiet paired data and all limitations
will be retained, including unresolved direct-context differences and std wins.

Use fresh target directories after source transport. `freeze.py` refuses an
existing snapshot and copies binaries outside Cargo before further builds.
Raw data/binaries/profiles go to /mnt/v/output/archmage/zenmath-pow-opt-2026-11-06/
and the corresponding NAS mirror. No large generated data belongs in git.

The current driver also covers cutoff/base neighbors and logarithmically
spaced positive f32 bit codes. A historical corpus control uses bases in
[0.01,4.01) and the exact f32 exponent 2.4, matching that input definition only.
`transfer` measures C0 whole-transfer work separately: inactive lanes receive
valid (1,0) before eager power, then a shared postpass selects the linear branch.
Fused and separate encoding composition are separate groups for every power
candidate. This includes preprocessing/postprocessing memory passes; it is not a
single fused pixel loop. `linear-srgb =0.6.12` scalar/slice rational functions
are whole-transfer controls with their own approximation contracts and natural
compile-time composition. Source comparisons against local 8900ffc8 are recorded
in `linear-srgb-provenance.json`; no sibling repository is changed.

The initial native policy stage has four effective clean groups after replacing
one rejected dirty group. Correct rescue/reference behavior remains bit checked.
Linked native policy extraction found zero calls in all eight examined bodies;
Fused MIDP/LOWP use 10/7 scalar FMA instructions, separate variants use none.
Ordinary callers still pay a feature boundary per value. LOWP separate explicit
precondition retains its inline reference path because paired intervals did not
establish a native-context benefit. This is a design tie-break, not a universal
fastest-policy claim. Std still wins many scalar correct controls.

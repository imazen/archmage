# Real-consumer inline policy experiment

Draft-only investigation. This changes neither the published attributes nor the
attune parser. It measures actual zenav1-svt encode and rav1d-safe decode calls,
using separate copies of pinned consumer sources and macro-emission variants.

`run.py prepare` records exact source revisions, compiler version, input hashes,
resolved dependencies and lockfile hashes in its output `plan.json`. The
consumer Rust files are copied unchanged. Only the rav1d-safe manifest's
archmage dependency source is redirected to the selected macro stack.
`driver.rs` uses zenbench 0.1.10; no new dependency enters archmage itself.

Policies, each changing one layer from the baseline:

| Policy | Changed emission |
|---|---|
| baseline | Fixed main's existing attributes: body hints, always-inline proof wrappers, existing dispatcher attributes |
| body_default | Original buggy rule: resolves from emitted body visibility |
| body_operation | Corrected rule: resolves from source operation visibility before private lowering |
| body_none | Omit implicit body inline attributes in arcane and rite |
| body_never | Replace those body hints with inline(never) |
| proof_none | Omit implicit inline(always) on arcane proof wrappers |
| proof_hint | Replace proof-wrapper inline(always) with inline |
| dispatcher_always | Apply inline(always) to autoversion dispatchers |

Explicit arcane `inline_always` remains explicit. Handwritten consumer inline
attributes, ordinary magetypes method implementations and unrelated functions
are unchanged. Scalar bodies that do not pass through arcane or rite keep their
existing policy. These are controlled legacy-emitter ablations, not a claim
that the unpublished attune expansion has been measured end to end.

## Workload and timing

Two centered photographic crops, 256×256 and 512×512, are converted once to
I420 before benchmarking. The pinned image path/hash and generated input hashes
are recorded. No input pixels or encoded files are committed. The 256 case uses
QP 32/preset 8; the 512 case uses QP 20/preset 6. Both are single still-frame,
8-bit 4:2:0 encodes; the decoder reads the resulting AV1 payload in memory with
one thread. These are not video, film-grain, HDR or broad-corpus measurements.

The benchmark directly invokes the public encoder and decoder APIs. File I/O,
process startup, input preparation, pipeline/decoder construction and destruction
of returned output/state are outside the timer. `with_input` supplies fresh
state for every call; the closure returns state along with its output so that
state destruction is also untimed. Each group has 20 measured one-call rounds,
100 ms warmup, and ordinary zenbench resource gating. Stack jitter is disabled
consistently. The command runner pins runtime workers to CPU 2.

The Python runner randomizes policy order within each process-level pass.
Zenbench samples inside a process measure one compiled policy; they are **not**
paired samples across binaries. The process-level pass is the comparison unit.
All per-process means, medians and MADs are retained. Runtime process startup is
not part of the reported API timings. The generated payload and decoded plane
bytes must have identical SHA-256 hashes across every policy and build profile.

Two release profiles are built serially: LTO off / 16 codegen units, and fat LTO
/ 1 codegen unit. The second follows zenav1-svt's shipping profile, but the two
profiles change both LTO and codegen units, so their difference cannot be
attributed to LTO alone. Neither profile uses native CPU targeting or incremental
compilation. Reference binaries are retained outside Cargo target directories. GNU
`size` text/data/BSS counts and one build wall-time/RSS observation are saved; a single build is not a compile-time regression estimate.

## Running

Run these serially through the host's resource-limited heavy-command wrapper.
The runner refuses to overwrite prior artifacts.

```sh
python3 run.py prepare --out /absolute/new/output \
  --sources /absolute/pinned/consumer/sources --image /absolute/photo.png \
  --revision <archmage-commit>
python3 run.py build --out /absolute/new/output
python3 run.py measure --out /absolute/new/output --passes 5
python3 run.py export --out /absolute/new/output --report /absolute/new/report
```

The consumer source capture must contain `zenav1-svt`, `rav1d-safe` and the
`provenance.json` used by `benchmarks/consumer_compile.py`. The manifest override
and macro edits have exact occurrence checks: a changed upstream emitter fails
preparation rather than silently measuring the wrong policy.

Full compiler/measurement logs and generated artifacts stay in the output
folder. Compact committed results must name the source pins and reference the
resource wrapper's peak-RSS lines. Do not infer generic defaults from a single
CPU and these two workloads, or interpret inline(never) as equivalent to
omitting an attribute.

The export command requires the complete five-pass matrix for the explicitly
selected `--profiles` and `--policies`, 20 samples per group,
reliable runs and output parity before writing compact CSV/JSON evidence. Full
samples stay in the raw logs, indexed by hash in `raw-artifacts.csv`.

To compare only the visibility-based policy, pass `--policies baseline
body_default` to prepare, build, measure and export. Rebuild and remeasure the
baseline alongside it; do not compare runs made in different sessions as if
paired. The native arcane operation is private even when its proof wrapper is
public. Scalar/wasm arcane bodies and rite functions use their emitted visibility.
Proof-wrapper attributes and all explicit `inline_always` requests stay intact.

## Inventory of visibility-policy decisions

`inventory.py --measurement /absolute/measured/output --out /absolute/new/inventory`
collects decisions from the actual macro emitters in a fresh copy of the
`body_default` case. Run it through the resource-limited heavy-command wrapper.
It does not modify measured sources/binaries or production macro code.

The diagnostic helper records package, macro, function, identifier/body source
locations, written visibility, generated body visibility, body kind, token/tier,
architecture/feature guard and emitted inline attribute. `decisions.csv` retains
every event, including wrappers and cfg-filtered variants; `summary.csv` groups
them and `by-file.csv` groups matching body decisions by source file. The
`provenance.json` includes hashes of the raw inventory and the measurement plan.

An added calibration module checks public/private/restricted direct functions,
a public native wrapper, a scalar body and an explicit always-inline override.
Its package is `inline-consumer-probe`: exclude it from consumer totals. Its
added direct dependency enables no new features, and the collector verifies
that resolved package versions/features match the measured case.

`generated_guards_match` checks generated architecture and Cargo-feature guards
against the current compilation. It does not report execution, linker retention,
LLVM inlining or instantiated generic counts. AVX-512 code can match build guards
without being executable on the current CPU. `inherited` describes absent
visibility syntax; it does not assert that a trait method is private. Body-source
locations distinguish repeated functions generated by the same macro template.

## Corrected operation-visibility comparison

Use `--policies baseline body_default body_operation` with each runner action
(or the `*-operation` recipes) to retain the original policy as a control.
`body_operation` resolves a native arcane body's hint from the source function's
visibility before emitting a private helper. Rite output visibility and wrapper
policies are unchanged. The earlier `body_default` remains reproducible.

Inventory either rule with `inventory.py --policy body_operation` or
`--policy body_default`. The CSV records `policy_visibility` separately from
`body_visibility`; the calibration requires a public native operation's hidden
body to receive a hint only under the corrected rule. This experiment changes
legacy emitter decisions across the pinned stack, including macro-generated
magetypes kernels. It does not measure the complete attune rewrite.

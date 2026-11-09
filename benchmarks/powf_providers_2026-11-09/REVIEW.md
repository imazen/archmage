# Provider-study review

The production powf implementation, generators and workspace dependencies have
no changes from the previously verified 5f2817ec head. This follow-up adds
benchmarks, numerical comparisons and source research. External libraries stay
in standalone unpublished studies. Existing legacy, BitExact and HIGHP contracts
and every established accuracy, assembly and resource-admission gate remain.

Review found and corrected these issues in the study:

- The first Gamma analysis included subnormal exponents. The final independent
  predicate uses the actual preset's normal-or-zero exponent requirement. The
  initial reports remain visible and are identified as superseded analyses.
- Neoverse inherited a CPU target flag. Its original timing remains labeled
  CPU-targeted. A separately identified generic numerical build reproduces
  all 35 contexts and complete trajectories byte-for-byte; it has no new timing.
- Native evaluator parsing read failed input before checking stream state.
  The corrected worker rejects a truncated request before using those values.
  Frozen earlier workers and their valid-input measurements are retained.
- Replaying a manifest previously required manually restoring provider DSO
  environment variables. The loader now restores only known frozen selections,
  clears stale ones, verifies worker/explicit/resolved DSO hashes, and records
  the macOS shared-cache build for future discovery. Negative controls reject
  changed explicit and system DSOs. Original macOS runs retain their observed
  OS build separately; they do not gain a retroactive automated build check.
- This QEMU lacks mnemonic disassembly in its raw log. The instruction proof
  decodes its OBJD-T bytes and requires FMA block addresses in actual execution
  records. A translated-but-unexecuted block fails the negative control.
- The upstream hard-to-round finder's COUNT formula omits its final inclusive
  endpoint: it reports 16,777,216 while the search reports 16,777,217 checks.
  The helper now admits the independent inclusive bit-pattern count and checks
  the actual search total. Both are below the unchanged 20-million budget;
  all 515 witnesses and the original raw outputs remain unchanged.

The whole-function CORE-MATH FMA kernel enables AVX+FMA and disables AVX2.
Its guard requires usable AVX+FMA, the AVX2-disabled run selects it, and the
no-FMA profile rejects it. GNU/AOCL x8 entry points retain their stronger
AVX2+FMA checks. Each external symbol is loaded from its named DSO without
preloading or interposing Rust std. The native C reference shares its adapter
with the C libraries. EMPTY transport work is measured and never subtracted.

Validation: full `just ci` passes with its original tests/limits. Resource line:
`650s | peak-RSS 0.71GiB | min-avail 23107MiB | peak-load 4.21`.
The preceding failure was a caller environment error: an explicitly empty
CARGO_ENCODED_RUSTFLAGS caused the existing Miri launcher to receive an empty
rustc argument. The successful invocation leaves that variable unset; no
expectation or test is relaxed. Both logs are preserved. The standalone Rust
studies pass formatting and all-target Clippy, and the native workers compile
with warnings as errors on the measured x86/Arm platforms. Focused parser,
identity and executed-instruction negative controls pass.

Remaining limits are material: zero new provider cells meet all promotion
requirements; Intel was admission-blocked and Linux pilots were unreliable.
M6 timings are explicitly exploratory. First-use and exceptional/bit-distributed
provider timing matrices are not measured. Sample agreement cannot establish
whole-domain canonical behavior or a maximum mathematical error bound.
The 515 hard-to-round witnesses are bounded adversarial coverage. Production
promotion remains held, and the PR stays draft.

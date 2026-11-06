# Zenmath exhaustive f32 digests (local draft)

Independent unpublished harness for source freeze `5a9cc685`. It changes no
production crate and uses `zenmath::scalar::reference` explicitly. Hash equality
is regression evidence, not a proof of correctness or an accuracy oracle. Existing
independent MPFR and exact FMA fixtures remain unchanged.

Each cell evaluates input bit patterns in ascending unsigned-u32 order. The
complete unary domain has exactly **4,294,967,296** inputs. A power cell covers
that domain for one explicitly named exponent, never all input pairs. For
example `pow:4019999a` means the exact f32 representation of 2.4. Suggested
additional families are `pow:3f000000` (0.5), `pow:40000000` (2), and
`pow:bf800000` (-1). No exhaustive f64 or FMA-ternary claim is made.

`reference` runs the non-dispatch scalar algorithms. `packed-scalar` runs their
four-lane software counterpart; `scalar-context` runs the complete scalar
context entries. `v3`, `neon`, `wasm128`, and `v4x` run four-input direct context
bodies inside one proof boundary per block. `scalar-v3`, `scalar-neon`,
`scalar-wasm128`, and `scalar-v4x` run the scalar context bodies. `v4x8` uses the
actual eight-binary64 AVX-512 implementation. For primitives, `v1` supports
sqrt; `v2` supports floor/ceil/trunc/round/roundeven; `neon` and `wasm128`
support all six. An unsupported tier/operation fails. Trunc's reference is
`reference::floorf(x.abs()).copysign(x)` because the frozen reference module has
no standalone truncf. NaN payloads are outside the numerical contract.

## Format and correctness checks

Zenhash is pinned to upstream revision
[`4c4a993ba4207f2f28529ce0406be8650d270118`](https://github.com/imazen/zenhash/tree/4c4a993ba4207f2f28529ce0406be8650d270118),
with only `xxh3,std` and defaults disabled. The harness patches its Archmage
requirement to the same local source used by Zenmath. No hash dependency is
added to production crates.

Chunk boundaries are fixed independently of thread scheduling (default 2^20
inputs). Each worker buffers 4,096 output values. It feeds explicit little-endian
u32 bytes into seed-zero streaming XXH3-128, once per block. Two hashes are
retained: raw output bits and canonicalized NaNs (`0x7fc00000`). Canonicalization
retains both signs of zero and every non-NaN bit. The ordered aggregate hashes
LE u64 index/start/count followed by the LE u128 per-chunk digest, with a fixed
domain header. **This is an aggregate of chunk hashes, not a hash of the raw
output stream.** Use matching domain/chunk/scheme when comparing results.

Chunk rows include their index, start, count, raw/canonical digest and a checksum
of those fields. Resume validates full configuration and source/binary identity,
then archives the previous summary before loading chunks. A corrupt/range-invalid
chunk fails with `FAILURE.txt`; missing chunks are recomputed. Archived `.bak`
files preserve prior evidence. Only exact ordered coverage can produce
`COMPLETE-F32-DOMAIN`; bounded runs say `COMPLETE-BOUNDED-DOMAIN`.

`--compare` directly checks every output against the scalar reference, treating
NaNs by NaN-ness. A mismatch prints input/actual/reference bits, reruns the chunk,
and fails. A completion marker is valid only if no FAILURE exists and all chunk
checksums, ranges and total coverage validate. Use the driver for an exclusive
run-directory lock. Do not launch the binary concurrently into one directory.

## Running on an admitted compute host

Set a disk-backed TMPDIR and an isolated CARGO_TARGET_DIR. Use explicit Rust
1.99.0 and clear both RUSTFLAGS and CARGO_ENCODED_RUSTFLAGS; never use native CPU
flags. CPU admission is external to this harness. Run builds/sweeps through the
shared `run-heavy` resource wrapper with the admitted thread and memory caps.
No build or benchmark belongs on the coordinator host.

```sh
cargo +1.99.0 fmt
just check
just build
just health
python3 drive.py prepare --binary "$CARGO_TARGET_DIR/release/zenmath-digests" \
  --revision 5a9cc685 --out "$RUN_ROOT/build"
python3 drive.py run --build "$RUN_ROOT/build" --op exp --backend neon \
  --threads 4 --compare --out "$RUN_ROOT/exp-neon"
```

`prepare` refuses existing identity directories. It pins Rust1.99.0 and verifies the compiler, target and source SHA embedded at build time before recording a SHA-256 source-file manifest, executable SHA-256, rustc version,
target and configured rounding/underflow assumption. Full rustc identity is
saved separately. `run` verifies the source and executable have not changed,
records its exact command, and invokes the binary. Add `--count 65537` for bounded
smoke checks; use a new directory for every configuration. Add `--resume` only
for an interrupted matching run. WASM requires threads1 and an explicit runner
and target at `prepare`; grant its output directory through the WASI runner.

`python3 drive.py compare --auditor <local-compatible-binary> <run>...` checks source revision/digest, operation,
domain, chunking and hash scheme before comparing canonical aggregates and
per-chunk canonical digests. The read-only auditor independently verifies every chunk checksum, exact ordered coverage, saved chunk files and completion summary; any failure marker rejects the run. Binary hashes may differ between target builds;
source identities must match. Preserve differing-source results separately.
Raw equality is reported separately because NaN payloads may differ.

Tests cover byte order, NaN/zero policy, tails, chunk corruption, gaps/order/count,
parallel ordering, exact mismatch bits, resume identity, stale completion markers,
and directed mixed-bit data for the four packed operations and exponent families.
Reports must list missing matrix cells before claiming coverage. Large generated
results belong outside git with a pointer; archive and verify their mirror before
reporting them secured. Digest elapsed time is harness elapsed only, not a kernel
speed benchmark.

`matrix.py` runs admitted cells sequentially, with per-cell logs and durable JSON
recording its PID, current child PID, exact command and completed/required cells.
Groups are top functions, primitives, and additional fixed-power families.
Never infer liveness from the JSON alone: check the recorded process and chunk
artifacts. All matrix cells use direct per-output reference comparisons.

## Driver updates and preserved numerical evidence

The initial exhaustive matrix uses harness source stamp
`59e78dde6dee12778662beff2c215a88029e73381bc159fb5dcc77f47a46f64c`
on implementation freeze `5a9cc685`. Keep that exact numerical/build source
across ISA builds; do not relabel earlier results after driver updates.

The subsequent driver revision catches a signalled subprocess, archives an old
summary and persists its return code/signal in `FAILURE.txt`. Its regression
covers SIGTERM15 without relying on the child to write a failure marker. Matrix
subprocesses now use `sys.executable`, preserving the selected interpreter.
This matters when a PATH shim reintroduces environment variables. Existing
binaries do not change behavior when runtime RUSTFLAGS changes; compilation
fingerprints establish effective build flags. Interrupted chunks remain partial
and are preserved. Driver-only validation is separate from the frozen exhaustive
matrix and does not establish a new numerical source matrix.

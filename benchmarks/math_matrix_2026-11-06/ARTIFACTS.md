# Retained artifacts and run identities

Raw generated data, binaries and full logs are outside git on **dev**:
`/mnt/v/output/archmage/math-matrix-2026-11-06/`.
NAS mirror: `/mnt/tower/output/archmage/math-matrix-2026-11-06/`.
`SHA256SUMS` covers every retained file; `MANIFEST.meta.json` records date,
file count, bytes and manifest hash. The final report records verification.
No R2 upload was made. These local paths were checked on dev; other hosts'
`/mnt/v` trees are not assumed identical.

Each platform folder retains output x/y/result bit triples and inventory,
raw zenbench JSON with all rounds, codegen and full logs. The original
incomplete timing passes are preserved in the remote-stage logs and snapshots;
only final accepted JSON is summarized. A standalone snapshot manifest/tarball
freezes source independently of later feedback/report additions.

## Per-build source manifests

The hash below is SHA256 of `source.sha256`, not a claim that the final report
source tree exactly equals every earlier measured executable.

| Build snapshot | Manifest SHA256 | Used for |
|---|---|---|
| i265/frozen-throughput-source | `df782a40fcf582dd071e3a4d8491a23ec2a27db42a849f7e0d36007e525a2b99` | Intel/Ryzen actual throughput and algebraic binary |
| r7900x/frozen-codegen-source | `7ae39f65ca21e6916fd313e5be3d018302064048b102787fa44005b26f66a0bc` | Original native validation and non-LTO x86 codegen source; copied Intel timing binaries separately identified |
| arm/frozen-throughput-source | `bf73098116ad810b378fe0dd2110bfa2b6f4a6da8c4c042996d2768a04d6abfb` | ARMbig throughput/algebraic binary built on ARMxl |
| wasm/frozen-wasm-source | `84c4537502d849bd7ac792c37775e1e216c9390dea6cca08a4e5b6de0bdea818` | WASM module/validator and embedded host before later report helpers |
| r7900x/frozen-feedback-source | `e34db79ee3a8fbd48910c8b34b0fa8790f336360003d09bc712d9808151228a2` | Intel/Ryzen feedback binary |
| arm/frozen-feedback-source | `e34db79ee3a8fbd48910c8b34b0fa8790f336360003d09bc712d9808151228a2` | ARMbig feedback binary built on ARMxl |

Each `identity.json` includes exact binary paths/SHA256s, compiler/LLVM and uname.
The actual linked disassembly and symbols live in `arm/timed-codegen/` and
`r7900x/timed-codegen/`; selected small extracts are committed in `codegen/`.
Runtime CPU/libc observations are in `runtime-*.txt`. Exact dependency versions
and checksums are pinned in the committed Cargo.lock (SHA256
`7b030ae8a8ff238161ba697bbc04d7d98ebeccc53ac44866113d52c0e5c767c0`).

## Validation evidence

- `accuracy.csv`:4568 independent MPFR file results; each1024records.
- `wasm49/`: the identical retained module under official Wasmtime49.0.5,
  including1046same-corpus output files and independent MPFR results.
  `reference/equality.csv` confirms every file is byte-identical to40.0.1.
  CLI source: https://github.com/bytecodealliance/wasmtime/releases/tag/v49.0.5
  CLI SHA256 `04c48a2507d614209dfbaa45397e842a5c79f64fe75e5bd4d3b95cdf667fec5d`;
  validator module `2693722331f2f828746fbd84875eeb1d2e54f8109766e4c7484d0cb12c1b3c5c`.
- `wasm/isolation/`, `wasm49/isolation/`: per-op/band external candidate logs.
  Both engines retain the metallic powf/special invalid-integer-conversion trap.
- `tested-docs/`: full actual source files, final health/doc log, and
  `current-vs-tested.json`. Two guide fences plus one reductions fence passed
  native/default, AVX-512 and strict WASM. Tested guide hash
  `0f05e5b1ce417bd8c772e20856ce2a5b70a6f3b54bd32e470476fded0f589469`;
  reductions `5fc5ae004caf8f2d1f0d0c8bc0bdec5898b5d9f41c5386a7ac393363d007ce66`.
  Later guide prose differs, but its two Rust fences are byte-identical; combined
  fence hash `e60d3c5524602422b8dbabfc451f614c91a9e1d79647f8ec34cd1d3a15f0063a`.
- `r7900x/algebraic-evidence/`: actual1.89.0 compile failure and1.98.1 success,
  retained optimized per-operator IR with the four precise algebraic flags.
- `windows/windows-exact-probe.log`: fresh existing executable probe,
  command `timeout 15s /mnt/c/Windows/System32/cmd.exe /c ver`, internal
  UtilAcceptVsock accept4 timeout110 and rc1. No Windows execution claimed.
- `workspace-safety/`: backup and every-file hash verification around the
  shared-store jj stale-workspace recovery; no WIP was lost.

## Resource evidence

Each value below is copied from the full run-heavy summary, not estimated.

| Accepted run/log | rc / wall | Peak RSS | Min available | Peak load |
|---|---|---|---|---|
| Intel matrix+algebraic `i265/logs/matrix-i265-timing.log` |0 /192s |0.04GiB |22565MiB |1.74 |
| Ryzen matrix+algebraic `r7900x/logs/matrix-timing-final.log` |0 /328s |0.11GiB |18143MiB |2.06 |
| Neoverse matrix+algebraic `arm/logs/matrix-timing-final.log` |0 /194s |0.03GiB |14875MiB |1.02 |
| WASM matrix+algebraic `wasm/logs/matrix-wasm-timing.log` |0 /466s |0.43GiB |47565MiB |1.82 |
| Final all-feature native/scoped WASM lint `wasm/logs/matrix-final-lint.log` |0 /51s |1.23GiB |46324MiB |2.43 |
| Four health checks+three doc configurations `tested-docs/matrix-health-docs.log` |0 /26s |0.57GiB |17708MiB |0.60 |

Additional full logs retain build failures/fixes, empty first-pass cells,
initial stale zenbench reservation diagnosis, feedback runs, ARM builds,
version probes, and exact source snapshot identities. A dead reservation was
renamed only after recorded PID absence and an unlocked guard were verified;
no live benchmark lock was bypassed. No benchmark/build process remains ours
at completion. The coordinator owns integration and jj-workspace cleanup.

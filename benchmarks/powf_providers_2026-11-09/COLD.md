# Const policy: cold-target compile and linked size

Neoverse-N1, Rust 1.99.0, release, thin LTO, one codegen unit, two build jobs,
6 GiB cap per build. Three fresh Cargo targets per variant; frozen order rotates
across repetitions. Dependency resolution/downloads are excluded and all four
consumer lockfiles match byte-for-byte. Background system work remains visible
in the resource records, so these are diagnostics, not qualified runtime timings.

Base source is main 6bfd83d744e41259ec758054631dcb0928fc4ea1. Patched source is
5f2817ecb2440c37e1df03d71d315b1e3bfa9b35. No experimental provider dependency
enters these clients. Every client calls the same legacy scalar x4 MIDP, LOWP
and std f32 operations. The two policy clients each add the same Gamma MIDP
array call; one uses const `.bits()`, the other opaque runtime options.

| Consumer | Fresh-target wall seconds, repetitions 0 / 1 / 2 | File bytes | GNU size text / data / bss |
|---|---|---:|---|
| Main, legacy only | 16.06 / 15.85 / 15.70 | 457,848 | 279,394 / 10,152 / 232 |
| Patched, legacy only | 16.65 / 16.17 / 16.11 | 457,848 | 279,394 / 10,152 / 232 |
| Patched, plus const Gamma MIDP | 17.08 / 16.82 / 16.54 | 458,640 | 281,218 / 10,272 / 232 |
| Patched, plus runtime options | 27.55 / 27.42 / 27.03 | 460,768 | 318,162 / 10,272 / 232 |

The const client has 36,944 fewer GNU-size text bytes than the runtime-options
client. The legacy-only size columns match between source trees; their binary
hashes differ. File length, GNU-size text and instruction counts are different
measurements and are not interchangeable. Every repetition's actual binary,
hash and `size` output is retained.

GNU time peak RSS ranges: main legacy 394,300–394,660 KiB; patched legacy
423,328–423,592 KiB; const policy 417,636–423,560 KiB; runtime options
763,576–764,244 KiB. The full resource wrapper lines and GNU time records are
preserved per build. No memory estimate is derived from source or struct sizes.

The runtime client deliberately hides all five option fields from LLVM. It can
retain dispatch and graph families that a caller's const flags remove. This
measures a representative width and policy, not compile time for every possible
consumer or a promise that const code is always faster at runtime.

Reproduce with `just powf-policy-cold-build BASE_TAR PATCHED_TAR DIRECTORY
TOOLCHAIN`. `cold_build.py` retains binaries and logs outside Cargo targets
before reclaiming each completed, task-owned build directory. First-use runtime
cost remains unmeasured; these builds do not substitute for that benchmark.

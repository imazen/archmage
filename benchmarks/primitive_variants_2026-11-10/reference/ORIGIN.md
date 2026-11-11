# Intel RCP14 reference

`RECIP14.c` is the unmodified Intel reference downloaded on 2026-11-11 from
[RECIP14.c](https://www.intel.com/content/dam/develop/external/us/en/documents/recip14.c),
linked by [Intel's reference implementation article](https://www.intel.com/content/www/us/en/developer/articles/code-sample/reference-implementations-for-ia-approximation-instructions-vrcp14-vrsqrt14-vrcp28-vrsqrt28-vexp2.html).
The file retains its Intel copyright and BSD three-clause redistribution notice.
It is a diagnostic oracle, outside the shipped crates. The wrapper compiles with
strict-aliasing optimization disabled to preserve the reference's pointer casts,
with FP contraction and fast-math disabled. Hardware checks are CPU-gated.

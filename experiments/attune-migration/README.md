# Proposed attune migration inventory

Draft only: no attune implementation or existing API changes.

Start with the [annotated from/to patterns](migration.md) and
[generic, trait, inline and visibility contracts](migration-contracts.md).
Every proposed replacement has `API`, `INLINE` and `BEHAVIOR` comments.
The [full inventory report](report.md) links the occurrence index and explains
its scope: **1,009 attributes and 153 dispatch calls**, including fixtures and
macro templates, with comments and generated snapshots counted separately.

The [cross-crate assembly experiment](../cross-crate-inline/README.md) supports
the distinction between public name access and optimization through a public
wrapper. It does not measure the proposed attune implementation.

## Independent review

The coordinating agent verified all 435 scoped archive files against their git
blob IDs in baseline `cf07592212e96294ef9ca5dca9a364fa8d15d8ad`, reran the
inventory into a fresh directory, and byte-compared every compact index,
failure index, summary, source manifest and raw occurrence part. They matched.
The macro implementation was spot-checked for visibility, inline filtering
and attribute placement; those behaviors are linked in the guides.

The checked-in scope manifests retain source hashes. The 57 verbose raw
occurrence parts are reproducible with `inventory.py`; they are not checked in.
Use a fresh output directory to preserve earlier artifacts. Supply a source
archive of the pinned baseline through `--source`; no build is necessary.

Existing tests and snapshots remain unchanged. Future syntax and unresolved
trait/generic/entry contracts are labeled as proposals rather than compiling
replacements. The experiment remains on a draft bookmark, outside main.

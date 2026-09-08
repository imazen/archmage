+++
title = "Plane Operations"
weight = 1
+++

The complete runnable [zenfilters plane-scaling chain](@/magetypes/examples/generic-kernels.md)
is the starting point: public entry → dispatch → generated per-tier context →
array chunks → scalar tail. It includes the pinned production source and every
function needed by the standalone adaptation.

In the production exposure filter, Oklab L/a/b channels are all scaled by
`2^(stops/3)`, because they are related to cube-root linear light. A generic
plane multiplication helper should not silently assume that doubling an Oklab
component means doubling linear-light intensity.

For nonlinear transfer curves, see the complete
[linear-srgb gamma conversion](@/magetypes/math/transcendentals.md). Its scalar
tail uses powf while the vector body uses a polynomial: the documented error
budget matters at chunk boundaries. The old partial power-contrast and sigmoid
snippets omitted scalar helper definitions, so they are no longer presented as
standalone examples here.

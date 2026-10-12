+++
title = "See the Expansion"
description = "Compare the source with real cargo-expand output"
weight = 5
+++

Choose a target, then compare what you write with what Rust receives.
Desktop shows both columns; on mobile, the Source and Expansion controls stay
visible while you scroll. The same selection applies to every example.

These are complete module expansions from **Rust 1.99.0 and cargo-expand
1.0.126**. Each fixture is type-checked before capture. Rust has already
evaluated architecture and Cargo gates: an empty module means its function was
excluded on that target. The generated architecture guards are automatic.
This view shows macro expansion, not optimized assembly or a performance promise.

{{ expansion_browser() }}

The [full reference](@/attunement/reference.md) covers the remaining syntax and
restrictions. Reproduce these views from the repository with `just docs-expand`;
`just docs-expand-check` verifies the committed outputs.

+++
title = "Archmage"
description = "Safe SIMD via capability tokens for Rust"
template = "landing.html"

[extra]
section_order = ["hero", "features", "easy_command", "final_cta"]

[extra.hero]
title = "Archmage"
description = "Safely invoke your intrinsic power, using the tokens granted to you by the CPU. Zero overhead. Zero unsafe. Full SIMD."
badge = "Rust 1.87+"
gradient_opacity = 15
cta_buttons = [
    { text = "Get Started", url = "/archmage/getting-started/installation/", style = "primary" },
    { text = "Magetypes (Experimental)", url = "/magetypes/", style = "secondary" },
]

[[extra.features]]
title = "Zero Unsafe"
desc = "Capability tokens prove CPU features at the type level. #[arcane] enables #[target_feature] so intrinsics are safe. Your crate uses #![forbid(unsafe_code)]."
icon = "fa-solid fa-shield-halved"

[[extra.features]]
title = "Zero Overhead"
desc = "Generates identical assembly to hand-written #[target_feature] + unsafe. The safety abstractions exist only at compile time — at runtime, you get raw SIMD instructions."
icon = "fa-solid fa-bolt"

[[extra.features]]
title = "Every Platform"
desc = "x86-64 (SSE2 through AVX-512), AArch64 (NEON through v3), and WASM SIMD128. Tokens compile on all platforms — summon() returns None on unsupported architectures."
icon = "fa-solid fa-globe"

[[extra.features]]
title = "Runtime Dispatch"
desc = "incant! dispatches to the best available SIMD tier at runtime. Write _v3, _neon, _wasm128, and _scalar variants — the macro handles detection, cfg guards, and fallback."
icon = "fa-solid fa-route"

[[extra.features]]
title = "Magetypes (Experimental)"
desc = "Optional SIMD vector types with natural Rust operators. f32x8 wraps __m256 with +, -, *, /, FMA, comparisons, reductions, and transcendentals. Cross-platform polyfills included."
icon = "fa-solid fa-shapes"

[[extra.features]]
title = "12,000+ Intrinsics Indexed"
desc = "Every x86 and AArch64 intrinsic cataloged by token, safety status, and stability. Browse which intrinsics each token unlocks."
icon = "fa-solid fa-magnifying-glass"

[extra.easy_command_section]
title = "Quick Start"
description = "Add archmage to your project and start writing safe SIMD."
tabs = [
    { name = "Cargo.toml", command = "[dependencies]\narchmage = \"0.8\"" },
    { name = "With Magetypes", command = "[dependencies]\narchmage = \"0.8\"\nmagetypes = \"0.8\"" },
    { name = "docs.rs", link = "https://docs.rs/archmage" },
]

[extra.final_cta_section]
title = "Start Writing Safe SIMD"
description = "Archmage is stable, battle-tested, and generates identical assembly to hand-written unsafe code. Magetypes adds ergonomic vector types if you want them."
button = { text = "Read the Docs", url = "/archmage/" }
+++

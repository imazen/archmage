+++
title = "Archmage"
description = "Safe SIMD via capability tokens for Rust"
template = "landing.html"

[extra]
section_order = ["hero", "features", "easy_command", "final_cta"]

[extra.hero]
title = "Archmage"
description = "Safely invoke your intrinsic power, using the tokens granted to you by the CPU. Portable kernels, explicit CPU capabilities, and safe SIMD."
badge = "Rust 1.89+"
gradient_opacity = 15
cta_buttons = [
    { text = "Get Started", url = "/archmage/getting-started/installation/", style = "primary" },
    { text = "Magetypes", url = "/magetypes/", style = "secondary" },
]

[[extra.features]]
title = "Zero Unsafe"
desc = "Capability tokens prove CPU features at the type level. #[arcane] enables #[target_feature] so intrinsics are safe. Your crate uses #![forbid(unsafe_code)]."
icon = "fa-solid fa-shield-halved"

[[extra.features]]
title = "Auditable Performance"
desc = "Place the hot loop inside its target-feature context. Inspect codegen and benchmark the complete kernel against the same numerical contract."
icon = "fa-solid fa-bolt"

[[extra.features]]
title = "Every Platform"
desc = "x86-64 (SSE2 through AVX-512), AArch64 (NEON through v3), and WASM SIMD128. Tokens compile on all platforms — summon() returns None on unsupported architectures."
icon = "fa-solid fa-globe"

[[extra.features]]
title = "Runtime Dispatch"
desc = "incant! dispatches to the best available SIMD tier at runtime. Generate _v3, _neon, _wasm128, and _scalar variants with #[magetypes] — the macro handles detection, cfg guards, and fallback."
icon = "fa-solid fa-route"

[[extra.features]]
title = "Magetypes"
desc = "Optional SIMD vector types with natural Rust operators. f32x8 has eight logical lanes with +, -, *, /, FMA, comparisons, reductions, and transcendentals. Cross-platform polyfills included."
icon = "fa-solid fa-shapes"

[[extra.features]]
title = "12,000+ Intrinsics Indexed"
desc = "Every x86 and AArch64 intrinsic cataloged by token, safety status, and stability. Browse which intrinsics each token unlocks."
icon = "fa-solid fa-magnifying-glass"

[extra.easy_command_section]
title = "Quick Start"
description = "Add archmage to your project and start writing safe SIMD."
tabs = [
    { name = "Cargo.toml", command = "[dependencies]\narchmage = \"0.9\"" },
    { name = "With Magetypes", command = "[dependencies]\narchmage = \"0.9\"\nmagetypes = \"0.9\"" },
    { name = "docs.rs", link = "https://docs.rs/archmage" },
]

[extra.final_cta_section]
title = "Start Writing Safe SIMD"
description = "Learn from complete image and codec kernels: generics, target-feature contexts, safe memory access, and measured ISA tradeoffs."
button = { text = "Read the Docs", url = "/archmage/" }
+++

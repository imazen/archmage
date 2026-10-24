//! Named imports: each macro is used only inside another macro's body, the
//! way the body rewriters see it. An import the rewrite consumes would be
//! reported as unused.

/// `attuned!` and `reattune!` only inside `#[attune]` bodies.
pub mod attune_calls {
    use archmage::{attune, attuned, reattune};

    #[attune(_*, _*_t)]
    fn step(x: u32) -> u32 {
        x.wrapping_add(1)
    }

    /// Covered call.
    #[attune(_*, dispatch)]
    #[must_use]
    pub fn twice(x: u32) -> u32 {
        attuned!(step(attuned!(step(x))))
    }

    /// Reconsidered call.
    #[attune(_*, dispatch)]
    #[must_use]
    pub fn reconsidered(x: u32) -> u32 {
        reattune!(step(x), [_v3, _scalar])
    }
}

/// `incant!` only inside `#[rite]` bodies. `#[rite(v3, neon, wasm128)]`
/// emits nothing on other architectures, so code (and imports) that use it
/// is gated the same way.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
pub mod rite_calls {
    use archmage::{incant, rite};

    #[rite(v3, neon, wasm128)]
    fn halve(x: f32) -> f32 {
        x * 0.5
    }

    #[rite(v3, neon, wasm128)]
    fn quarter(x: f32) -> f32 {
        incant!(halve(x) without token) * 0.5
    }

    // A qualified path at statement start.
    #[rite(v3, neon, wasm128)]
    fn eighth(x: f32) -> f32 {
        archmage::incant!(quarter(x) without token) * 0.5
    }

    /// Calls the v3 variant from a v3 context.
    #[cfg(target_arch = "x86_64")]
    #[archmage::arcane]
    #[must_use]
    pub fn quarter_v3_entry(_token: archmage::X64V3Token, x: f32) -> f32 {
        quarter_v3(x) + eighth_v3(x)
    }
}

/// `incant!` only inside a `#[magetypes]` body.
pub mod magetypes_calls {
    use archmage::{incant, magetypes};

    #[magetypes(v3, neon, wasm128, scalar)]
    fn inner(_token: Token, x: f32) -> f32 {
        x * 2.0
    }

    #[magetypes(v3, neon, wasm128, scalar)]
    fn outer(token: Token, x: f32) -> f32 {
        incant!(inner(x) with token, [v3, neon, wasm128, scalar])
    }

    /// Dispatches `outer`.
    #[must_use]
    pub fn run(x: f32) -> f32 {
        archmage::incant!(outer(x), [v3, neon, wasm128, scalar])
    }
}

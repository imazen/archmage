//! `#[attune]`, `attuned!` and `reattune!` (the website examples, plus calls
//! in every value position).

/// A single direct body: `#[attune(v3)]`.
pub mod direct {
    use archmage::prelude::*;
    /// Adds one.
    #[attune(v3)]
    #[must_use]
    pub fn add_one(x: u32) -> u32 {
        x.wrapping_add(1)
    }
}

/// Tier inferred from the suffix.
pub mod inferred {
    use archmage::prelude::*;
    /// Adds one.
    #[attune]
    #[must_use]
    pub fn add_one_v3(x: u32) -> u32 {
        x.wrapping_add(1)
    }
}

/// A proof wrapper around a token-taking body.
pub mod wrap {
    use archmage::prelude::*;
    /// Adds one.
    #[attune(wrap)]
    #[must_use]
    pub fn add_one(_token: X64V3Token, x: u32) -> u32 {
        x.wrapping_add(1)
    }
}

/// Every tier as a direct body, every tier as a proof wrapper, a dispatcher.
pub mod families {
    use archmage::prelude::*;
    /// Adds one (direct bodies).
    #[attune(_*)]
    #[must_use]
    pub fn add_one(x: u32) -> u32 {
        x.wrapping_add(1)
    }
    /// Adds two (proof wrappers).
    #[attune(_*_t)]
    #[must_use]
    pub fn add_two(x: u32) -> u32 {
        x.wrapping_add(2)
    }
    /// Adds three (all outputs).
    #[attune(all)]
    #[must_use]
    pub fn add_three(x: u32) -> u32 {
        x.wrapping_add(3)
    }
    /// Gated tiers.
    #[attune(_*, dispatch, +v4x(cfg(avx512)), -_neon)]
    #[must_use]
    pub fn add_four(x: u32) -> u32 {
        x.wrapping_add(4)
    }
    /// Per-output options. The author asked for `inline(always)`, so the
    /// author allows its lint.
    #[attune(_v3(pub(crate), inline(never)), _scalar, dispatch(pub, inline(always)))]
    #[must_use]
    #[allow(clippy::inline_always)]
    pub fn add_five(x: u32) -> u32 {
        x.wrapping_add(5)
    }
}

/// Dispatchers over slices, vectors, generics and methods.
pub mod dispatchers {
    use archmage::prelude::*;
    /// Scales in place.
    #[attune(dispatch)]
    pub fn scale(values: &mut [f32], factor: f32) {
        for value in values {
            *value *= factor;
        }
    }
    /// Scales in place with `f32x8`.
    #[attune(dispatch, define(f32x8))]
    pub fn scale_vector(values: &mut [f32], factor: f32) {
        let token = Token::from_context();
        let multiplier = f32x8::splat_t(token, factor);
        let (chunks, tail) = f32x8::partition_slice_mut_t(token, values);
        for chunk in chunks {
            (f32x8::load_t(token, chunk) * multiplier).store(chunk);
        }
        for value in tail {
            *value *= factor;
        }
    }
    /// Maps in place.
    #[attune(dispatch)]
    pub fn map<F: Fn(f32) -> f32 + Copy>(values: &mut [f32], f: F) {
        for value in values {
            *value = f(*value);
        }
    }
    /// A gain applied by a method.
    #[derive(Debug)]
    pub struct Gain(pub f32);
    impl Gain {
        /// Scales in place.
        #[attune(_*, dispatch)]
        pub fn apply(&self, values: &mut [f32]) {
            for value in values {
                *value *= self.0;
            }
        }
    }
}

/// x86 intrinsics through a feature context.
pub mod raw_register {
    use archmage::prelude::*;
    /// Broadcasts one value.
    #[attune(v3, import_intrinsics, define(f32x8))]
    #[must_use]
    pub fn broadcast(value: f32) -> [f32; 8] {
        f32x8::from_raw(_mm256_set1_ps(value)).to_array()
    }
}

/// `attuned!` and `reattune!` in every value position.
pub mod calls {
    use archmage::prelude::*;

    #[attune(_*, _*_t, +v4(cfg(avx512)))]
    fn bump(x: u32) -> u32 {
        x.wrapping_add(1)
    }

    #[attune(_*, _*_t)]
    fn bump_in_place(x: &mut u32) {
        *x = x.wrapping_add(1);
    }

    fn identity(x: u32) -> u32 {
        x
    }

    /// Plain caller, default tiers, tail position.
    #[must_use]
    pub fn tail(x: u32) -> u32 {
        attuned!(bump(x))
    }

    /// Plain caller, scalar-only list in `let`, argument and assignment.
    #[must_use]
    pub fn scalar_positions(x: u32) -> u32 {
        let a = attuned!(bump(x), [_scalar]);
        let mut b = identity(attuned!(bump(a), [_scalar]));
        b = attuned!(bump(b), [_scalar]);
        attuned!(bump_in_place(&mut b), [_scalar]);
        b
    }

    /// Plain caller with a held token.
    #[must_use]
    pub fn using_token(token: X64V3Token, x: u32) -> u32 {
        let y = attuned!(bump(x), [_v3, _scalar], using(token));
        y.wrapping_mul(2)
    }

    /// Covered calls inside a feature body, every position.
    #[attune(_*, dispatch)]
    #[must_use]
    pub fn covered(x: u32) -> u32 {
        let mut y = attuned!(bump(x));
        y = identity(attuned!(bump(y)));
        y += attuned!(bump(y));
        attuned!(bump_in_place(&mut y));
        if attuned!(bump(y)) > 3 {
            y = y.wrapping_add(1);
        }
        let z = match attuned!(bump(y)) {
            0 => 0,
            n => n.wrapping_sub(1),
        };
        if z == 0 {
            return attuned!(bump(z));
        }
        attuned!(bump(z)).wrapping_add(attuned!(bump(y)))
    }

    /// A brace-delimited call is a statement and needs no `;`; the rewrite
    /// keeps the caller's delimiter so the next statement still parses.
    #[attune(_*, dispatch)]
    #[must_use]
    pub fn braced(x: u32) -> u32 {
        let mut y = x;
        attuned! { bump_in_place(&mut y) }
        y = y.wrapping_add(1);
        y
    }

    /// `reattune!` probes for stronger tiers from a covered context.
    #[attune(_*, dispatch)]
    #[must_use]
    pub fn reconsider(x: u32) -> u32 {
        let y = reattune!(bump(x), [_v4(cfg(avx512)), _v3, _scalar]);
        y.wrapping_add(1)
    }
}

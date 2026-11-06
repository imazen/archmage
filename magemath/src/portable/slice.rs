//! Bulk operations on `&mut [f32]`, dispatched through archmage tiers.
//!
//! Each op is an `#[attune]` family: the public name dispatches at runtime,
//! `_v3_t`-style proof wrappers take a token, and `_v3`-style direct variants
//! are for callers already in that feature context. Lanes left over after the
//! last whole `f32x8` go through the same kernel in a padded vector, so every
//! element of a slice, whatever its position, takes one code path.
//!
//! Dispatch happens once per call: pass a whole buffer, not one element at a
//! time. To chain several operations per element, write one `#[attune]` body
//! over [`kernels`] instead of calling these in sequence.

crate::surface::slice_surface!();

/// Correctly rounded exp over a slice, dispatching once per buffer.
#[archmage::attune(_*, _*_t, dispatch(inline(hint)), +v4x(cfg(avx512)))]
pub fn exp_f32(data: &mut [f32]) {
    let token = Token::from_context();
    let (chunks, tail) = data.as_chunks_mut::<8>();
    let mut pad = [1.0; 8];
    pad[..tail.len()].copy_from_slice(tail);
    for i in 0..chunks.len() + usize::from(!tail.is_empty()) {
        let a = if i < chunks.len() {
            &mut chunks[i]
        } else {
            &mut pad
        };
        let lo = crate::correct::exp(token, core::array::from_fn(|i| a[i]));
        let hi = crate::correct::exp(token, core::array::from_fn(|i| a[i + 4]));
        *a = core::array::from_fn(|i| if i < 4 { lo[i] } else { hi[i - 4] });
    }
    tail.copy_from_slice(&pad[..tail.len()]);
}

/// Correctly rounded ln over a slice, dispatching once per buffer.
#[archmage::attune(_*, _*_t, dispatch(inline(hint)), +v4x(cfg(avx512)))]
pub fn ln_f32(data: &mut [f32]) {
    let token = Token::from_context();
    let (chunks, tail) = data.as_chunks_mut::<8>();
    let mut pad = [1.0; 8];
    pad[..tail.len()].copy_from_slice(tail);
    for i in 0..chunks.len() + usize::from(!tail.is_empty()) {
        let a = if i < chunks.len() {
            &mut chunks[i]
        } else {
            &mut pad
        };
        let lo = crate::correct::log::<_, false>(token, core::array::from_fn(|i| a[i]));
        let hi = crate::correct::log::<_, false>(token, core::array::from_fn(|i| a[i + 4]));
        *a = core::array::from_fn(|i| if i < 4 { lo[i] } else { hi[i - 4] });
    }
    tail.copy_from_slice(&pad[..tail.len()]);
}

/// Correctly rounded log2 over a slice, dispatching once per buffer.
#[archmage::attune(_*, _*_t, dispatch(inline(hint)), +v4x(cfg(avx512)))]
pub fn log2_f32(data: &mut [f32]) {
    let token = Token::from_context();
    let (chunks, tail) = data.as_chunks_mut::<8>();
    let mut pad = [1.0; 8];
    pad[..tail.len()].copy_from_slice(tail);
    for i in 0..chunks.len() + usize::from(!tail.is_empty()) {
        let a = if i < chunks.len() {
            &mut chunks[i]
        } else {
            &mut pad
        };
        let lo = crate::correct::log::<_, true>(token, core::array::from_fn(|i| a[i]));
        let hi = crate::correct::log::<_, true>(token, core::array::from_fn(|i| a[i + 4]));
        *a = core::array::from_fn(|i| if i < 4 { lo[i] } else { hi[i - 4] });
    }
    tail.copy_from_slice(&pad[..tail.len()]);
}

/// Correctly rounded power with a per-element exponent. Lengths must match.
#[archmage::attune(_*, _*_t, dispatch(inline(hint)), +v4x(cfg(avx512)))]
pub fn pow_f32(data: &mut [f32], exponents: &[f32]) {
    assert_eq!(
        data.len(),
        exponents.len(),
        "power slice lengths must match"
    );
    let token = Token::from_context();
    let (chunks, tail) = data.as_chunks_mut::<8>();
    let (ys, yt) = exponents.as_chunks::<8>();
    let mut pad = [1.0; 8];
    pad[..tail.len()].copy_from_slice(tail);
    let mut py = [1.0; 8];
    py[..yt.len()].copy_from_slice(yt);
    for i in 0..chunks.len() + usize::from(!tail.is_empty()) {
        let (a, b) = if i < chunks.len() {
            (&mut chunks[i], ys[i])
        } else {
            (&mut pad, py)
        };
        let lo = crate::correct::pow(
            token,
            core::array::from_fn(|i| a[i]),
            core::array::from_fn(|i| b[i]),
        );
        let hi = crate::correct::pow(
            token,
            core::array::from_fn(|i| a[i + 4]),
            core::array::from_fn(|i| b[i + 4]),
        );
        *a = core::array::from_fn(|i| if i < 4 { lo[i] } else { hi[i - 4] });
    }
    tail.copy_from_slice(&pad[..tail.len()]);
}

//! Shared precise implementation; normal currently uses the same bits.

/// Correctly rounded exp over a slice, dispatching once per buffer.
#[archmage::attune(_*, _*_t, dispatch(inline(hint)), +v4x(cfg(avx512)))]
pub fn exp_f32(data: &mut [f32]) {
    let token = Token::from_context();
    let (chunks, tail) = data.as_chunks_mut::<4>();
    let mut pad = [1.0; 4];
    pad[..tail.len()].copy_from_slice(tail);
    for i in 0..chunks.len() + usize::from(!tail.is_empty()) {
        let a = if i < chunks.len() {
            &mut chunks[i]
        } else {
            &mut pad
        };
        *a = crate::correct::exp(token, *a);
    }
    tail.copy_from_slice(&pad[..tail.len()]);
}

/// Correctly rounded ln over a slice, dispatching once per buffer.
#[archmage::attune(_*, _*_t, dispatch(inline(hint)), +v4x(cfg(avx512)))]
pub fn ln_f32(data: &mut [f32]) {
    let token = Token::from_context();
    let (chunks, tail) = data.as_chunks_mut::<4>();
    let mut pad = [1.0; 4];
    pad[..tail.len()].copy_from_slice(tail);
    for i in 0..chunks.len() + usize::from(!tail.is_empty()) {
        let a = if i < chunks.len() {
            &mut chunks[i]
        } else {
            &mut pad
        };
        *a = crate::correct::log::<_, false>(token, *a);
    }
    tail.copy_from_slice(&pad[..tail.len()]);
}

/// Correctly rounded log2 over a slice, dispatching once per buffer.
#[archmage::attune(_*, _*_t, dispatch(inline(hint)), +v4x(cfg(avx512)))]
pub fn log2_f32(data: &mut [f32]) {
    let token = Token::from_context();
    let (chunks, tail) = data.as_chunks_mut::<4>();
    let mut pad = [1.0; 4];
    pad[..tail.len()].copy_from_slice(tail);
    for i in 0..chunks.len() + usize::from(!tail.is_empty()) {
        let a = if i < chunks.len() {
            &mut chunks[i]
        } else {
            &mut pad
        };
        *a = crate::correct::log::<_, true>(token, *a);
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
    let (chunks, tail) = data.as_chunks_mut::<4>();
    let (ys, yt) = exponents.as_chunks::<4>();
    let mut pad = [1.0; 4];
    pad[..tail.len()].copy_from_slice(tail);
    let mut py = [1.0; 4];
    py[..yt.len()].copy_from_slice(yt);
    for i in 0..chunks.len() + usize::from(!tail.is_empty()) {
        let (a, b) = if i < chunks.len() {
            (&mut chunks[i], ys[i])
        } else {
            (&mut pad, py)
        };
        *a = crate::correct::pow(token, *a, b);
    }
    tail.copy_from_slice(&pad[..tail.len()]);
}

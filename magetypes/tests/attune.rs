#![forbid(unsafe_code)]
#![deny(warnings)]

use archmage::attune;

#[attune(make(all), define(f32x8))]
fn scale(data: &mut [f32], factor: f32) {
    let proof = Token::from_context();
    let multiplier = f32x8::splat_t(proof, factor);
    let (chunks, tail) = f32x8::partition_slice_mut_t(proof, data);
    for chunk in chunks {
        (f32x8::load_t(proof, chunk) * multiplier).store(chunk);
    }
    for value in tail {
        *value *= factor;
    }
}

#[attune(make(all))]
fn compose(data: &mut [f32], factor: f32) {
    archmage::attuned!(scale(data, factor));
}

#[test]
fn fixed_width_aliases_and_tail_work_in_every_available_context() {
    let original = [2.0f32; 19];
    let mut data = original;
    scale(&mut data, 0.5);
    assert_eq!(data, [1.0; 19]);
    compose(&mut data, 3.0);
    assert_eq!(data, [3.0; 19]);
    let mut scalar = original;
    scale_scalar(&mut scalar, 0.5);
    assert_eq!(scalar, [1.0; 19]);
}

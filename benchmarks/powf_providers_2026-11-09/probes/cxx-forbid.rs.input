//! Compiler probe for a type-checked C++ provider boundary.
#![forbid(unsafe_code)]
#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("provider_probe.h");
        fn provider_probe(x: f32) -> f32;
    }
}
fn main() {
    std::hint::black_box(ffi::provider_probe(std::hint::black_box(1.)));
}

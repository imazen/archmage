#![forbid(unsafe_code)]
#[archmage::rite(v3, use(f32xN))]
fn helper() {
    fn inner() { let _ = f32xN::splat(1.0); }
    inner();
}
fn main() {}

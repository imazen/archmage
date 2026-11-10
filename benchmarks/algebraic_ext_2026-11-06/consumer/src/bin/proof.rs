#![forbid(unsafe_code)]
fn main() {
    algebraic_ext_consumer::tier_inventory();
    println!(
        "exact fixture operations {}",
        algebraic_ext_consumer::fixtures::check_all()
    );
}

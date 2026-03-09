// When incant! is given an explicit tier list, `scalar` must be
// listed explicitly. Omitting it is a compile error — this forces
// acknowledgment that the scalar fallback path exists.

use archmage::incant;

fn add_v3(_token: archmage::X64V3Token, a: i32, b: i32) -> i32 {
    a + b
}

fn add_scalar(_token: archmage::ScalarToken, a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    // The _scalar function exists, but scalar isn't in the tier list — error!
    let _ = incant!(add(1, 2), [v3]);
}

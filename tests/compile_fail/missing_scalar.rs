// incant! always emits a call to fn_scalar(ScalarToken, ...) as the
// unconditional fallback. If the _scalar function doesn't exist, this
// is a compile error — not a runtime error.

use archmage::incant;

#[cfg(target_arch = "x86_64")]
fn add_v3(_token: archmage::X64V3Token, a: i32, b: i32) -> i32 {
    a + b
}

// No add_scalar defined!

fn main() {
    let _ = incant!(add(1, 2));
}

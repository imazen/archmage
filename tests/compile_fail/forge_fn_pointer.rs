// A #[target_feature] function has no call site left for rustc to check once
// it is behind a function pointer, so the coercion is rejected outright.
fn main() {
    let f: fn() -> archmage::X64V3Token = archmage::X64V3Token::forge_token_dangerously;
    let _ = f();
}

// On a foreign architecture no #[target_feature] context for those features can
// exist, so no `from_context()` is generated at all — only the `unsafe fn` alias.
fn main() {
    let _ = archmage::NeonToken::from_context();
}

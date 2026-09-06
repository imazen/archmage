// A caller with no #[target_feature] attribute cannot construct the proof.
fn main() {
    let _ = archmage::X64V3Token::from_context();
}

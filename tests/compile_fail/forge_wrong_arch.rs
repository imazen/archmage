// On a foreign architecture the constructor stays `unsafe fn`: no
// #[target_feature] context can exist for rustc to check against.
fn main() {
    let _ = archmage::NeonToken::forge_token_dangerously();
}

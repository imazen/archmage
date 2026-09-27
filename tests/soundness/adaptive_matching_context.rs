#![forbid(unsafe_code)]
#[archmage::rite(v3, use(f32xN))]
fn helper() { let _ = f32xN::splat(1.0); }
#[archmage::rite(v3)]
fn matching() { helper(); }
#[archmage::rite(v3_crypto)]
fn stronger() { helper(); }
#[archmage::rite(v3, use(f32xN))]
fn with_token() {
    fn inner(token: archmage::X64V3Token) {
        let _ = f32xN::splat_with_token(token, 1.0);
    }
    inner(archmage::X64V3Token::from_context());
}
fn main() {}

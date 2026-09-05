#[allow(dead_code)]
#[target_feature(enable = "sse,sse2")]
fn weaker_context() {
    let _ = archmage::X64V3Token::from_context();
}

fn main() {}

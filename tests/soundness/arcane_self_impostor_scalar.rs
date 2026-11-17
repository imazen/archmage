// REGRESSION CASE: `_self = ScalarToken` on an impostor receiver must fail to
// type-check even though ScalarToken has no `#[target_feature]` list.
#![forbid(unsafe_code)]
use archmage::{ScalarToken, arcane};

struct Impostor;

impl Impostor {
    #[arcane(_self = ScalarToken)]
    fn forge(&self) -> u32 {
        let _ = _self;
        1
    }
}

fn main() {
    let _ = Impostor.forge();
}

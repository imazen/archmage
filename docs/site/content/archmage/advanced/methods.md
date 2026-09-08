+++
title = "Methods with #[arcane]"
weight = 1
+++

`#[arcane]` supports inherent methods. Its default sibling expansion remains
inside the `impl`, so `self` and `Self` resolve there. Trait implementations
need nested expansion because adding an undeclared sibling method is invalid.

This is a reference exercise for receiver handling, not the main zen kernel
architecture. Most application examples use free generated kernels called by
ordinary image/converter methods.

```rust
use archmage::prelude::*;
struct Plane([f32; 8]);
impl Plane {
    #[arcane]
    fn sum(&self, _token: X64V3Token) -> f32 { self.0.iter().sum() }
}
trait Sum {
    fn sum_trait(&self, token: X64V3Token) -> f32;
}
// The whole impl must be cfg-gated: on another architecture the macro
// omits the method, which would otherwise leave a required trait item missing.
#[cfg(target_arch = "x86_64")]
impl Sum for Plane {
    #[arcane(_self = Plane)]
    fn sum_trait(&self, _token: X64V3Token) -> f32 { _self.0.iter().sum() }
}
#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() {
    let plane = Plane([1.0; 8]);
    assert_eq!(plane.sum(token), 8.0);
    assert_eq!(plane.sum_trait(token), 8.0);
}
```

`_self = Plane` supplies the concrete receiver type to the nested function and
uses `_self` in its body. Receiver-free trait methods can use `nested`. For a
generic impl, keep its actual type parameters in the receiver type as required
by the macro. The API test suite covers these expansion rules.

Do not introduce a token-bearing trait hierarchy just to dispatch an ordinary
pixel loop. A public method can call a free `incant!` entry; that is easier to
read and keeps ISA details out of the public data model.

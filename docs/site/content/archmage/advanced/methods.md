+++
title = "Methods with #[attune]"
weight = 1
+++

`#[attune]` supports inherent methods. Its default sibling expansion remains
inside the `impl`, so `self` and `Self` resolve there (`#[attune(wrap)]` and
family selectors generate method siblings). Two shapes need a flag, because the
macro sees only the function: an associated function without a receiver
(`in_impl`), and a method in a trait implementation (`in_trait`, alias
`nested`, or `_self = ConcreteType` when the method has a receiver), where an
undeclared sibling method is invalid.

This is a reference exercise for receiver handling, not the main zen kernel
architecture. Most application examples use free generated kernels called by
ordinary image/converter methods.

```rust
use archmage::prelude::*;
struct Plane([f32; 8]);
impl Plane {
    #[attune(wrap)]
    fn sum(&self, _token: X64V3Token) -> f32 { self.0.iter().sum() }
    // No receiver: the wrapper must call `Self::`, which the macro cannot infer.
    #[attune(wrap, in_impl)]
    fn filled(_token: X64V3Token, value: f32) -> Self { Self([value; 8]) }
}
trait Sum {
    fn sum_trait(&self, token: X64V3Token) -> f32;
}
// The whole impl must be cfg-gated: on another architecture the macro
// omits the method, which would otherwise leave a required trait item missing.
#[cfg(target_arch = "x86_64")]
impl Sum for Plane {
    #[attune(wrap, _self = Plane)]
    fn sum_trait(&self, _token: X64V3Token) -> f32 { self.0.iter().sum() }
}
#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() {
    let plane = Plane::filled(token, 1.0);
    assert_eq!(plane.sum(token), 8.0);
    assert_eq!(plane.sum_trait(token), 8.0);
}
```

In a trait implementation the feature-enabled function nests inside the
method, where `self` and `Self` do not exist. `_self = Plane` implies `in_trait`
and names the concrete receiver type: the nested function takes `_self: &Plane`,
and the macro rewrites `Self` to `Plane` and `self` to `_self`, so the body reads
as written. Receiver-free trait methods need `#[attune(wrap, in_trait)]`.
Nested functions cannot capture outer `impl<T>` type parameters (`E0401`); for a
generic `impl<T>`, move the kernel to a free function and pass `self` explicitly.
`in_impl` and `in_trait` describe different places and are rejected together; a
method with a receiver in an inherent impl needs neither. Family generation inside
a trait impl is also rejected because families emit sibling functions.

Do not introduce a token-bearing trait hierarchy just to dispatch an ordinary
pixel loop. A public method can call a free `attuned!` or `#[attune(dispatch)]`
entry; that is easier to read and keeps ISA details out of the public data model.

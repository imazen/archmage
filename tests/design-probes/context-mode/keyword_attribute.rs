//! Verify keyword tokens reach an attribute macro unchanged.
#![forbid(unsafe_code)]
extern crate proc_macro;

#[proc_macro_attribute]
pub fn accept(
    args: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    assert_eq!(args.to_string().replace(' ', ""), "use(f32x8)");
    item
}

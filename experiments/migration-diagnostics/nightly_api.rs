extern crate proc_macro;

#[proc_macro]
pub fn diagnose(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro::Diagnostic::new(proc_macro::Level::Warning, "probe").emit();
    input
}

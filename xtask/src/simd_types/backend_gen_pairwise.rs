//! Unsigned pairwise widening sums. Output lane k covers input lanes 2k, 2k+1.
use indoc::formatdoc;

pub(super) fn trait_names() -> Vec<(usize, String)> {
    [128, 256, 512]
        .into_iter()
        .flat_map(|w| [8, 16].map(|bits| (w, format!("U{bits}x{}Pairwise", w / bits))))
        .collect()
}

pub(super) fn traits() -> String {
    let mut out = String::new();
    for (w, name) in trait_names() {
        let bits = if name.starts_with("U8") { 8 } else { 16 };
        let src = format!("U{bits}x{}", w / bits);
        let dst = format!("U{}x{}", bits * 2, w / bits / 2);
        let gate = if w == 512 {
            "#[cfg(feature = \"w512\")]"
        } else {
            ""
        };
        out += &formatdoc! {r#"
            {gate}
            /// Exact unsigned adjacent sums in lanes twice as wide.
            pub trait {name}: super::{src}Backend + super::{dst}Backend {{
                /// Output lane k = widened a[2k] + widened a[2k+1].
                fn pairwise_widen_add(self, a: <Self as super::{src}Backend>::Repr) -> <Self as super::{dst}Backend>::Repr;
            }}
        "#};
    }
    out
}

pub(super) fn generic(name: &str) -> String {
    let Some((elem, n)) = name.split_once('x') else {
        return String::new();
    };
    let bits = match elem {
        "u8" => 8,
        "u16" => 16,
        _ => return String::new(),
    };
    let n: usize = n.parse().unwrap();
    let bound = format!("U{bits}x{n}Pairwise");
    let dst = format!("u{}x{}", bits * 2, n / 2);
    formatdoc! {r#"
        impl<T: crate::simd::backends::{bound}> {name}<T> {{
            /// Sum adjacent pairs into unsigned lanes twice as wide.
            ///
            /// Output lane `k` is `self[2*k] + self[2*k+1]`, with both
            /// inputs widened before addition. The result is exact for the
            /// full input range; no lane wraps or saturates. Pair ordering
            /// is unchanged across native and polyfilled widths.
            #[inline(always)]
            pub fn pairwise_widen_add(self) -> super::{dst}<T> {{
                super::{dst}::from_repr_unchecked(self.1, T::pairwise_widen_add(self.1, self.0))
            }}
        }}
    "#}
}

pub(super) fn impls(arch: &str, token: &str, w512: bool) -> String {
    let mut out = String::new();
    for w in [128, 256, 512].into_iter().filter(|w| (*w == 512) == w512) {
        let native = match arch {
            "v4" => 512,
            "x86" => w.min(256),
            _ => 128,
        };
        for bits in [8, 16] {
            let src = format!("u{bits}");
            let dst = format!("u{}", bits * 2);
            let sr = super::backend_gen_integer_ops::repr(arch, &src, w);
            let dr = super::backend_gen_integer_ops::repr(arch, &dst, w);
            let name = format!("U{bits}x{}Pairwise", w / bits);
            let attr = match arch {
                "scalar" | "wasm" => "#[inline(always)]".into(),
                "x86" if w == 128 => "sse2_baseline! {".into(),
                _ => super::backend_syntax::arcane(token),
            };
            let close = if arch == "x86" && w == 128 { "}" } else { "" };
            let body = if arch == "scalar" {
                format!("core::array::from_fn(|k| {dst}::from(a[2*k]) + {dst}::from(a[2*k+1]))")
            } else {
                let parts = w / native;
                let expressions: Vec<_> = (0..parts)
                    .map(|i| {
                        let a = if parts == 1 {
                            "a".into()
                        } else {
                            format!("a[{i}]")
                        };
                        native_expr(arch, native, bits, &a)
                    })
                    .collect();
                if parts == 1 {
                    expressions[0].clone()
                } else {
                    format!("[{}]", expressions.join(", "))
                }
            };
            out += &formatdoc! {r#"
                impl {name} for archmage::{token} {{
                    {attr}
                    fn pairwise_widen_add(self, a: {sr}) -> {dr} {{ {body} }}
                    {close}
                }}
            "#};
        }
    }
    out
}

fn native_expr(arch: &str, width: usize, bits: usize, a: &str) -> String {
    match arch {
        "neon" => format!("vpaddlq_u{bits}({a})"),
        "wasm" => format!(
            "u{}x{}_extadd_pairwise_u{bits}x{}({a})",
            bits * 2,
            128 / bits / 2,
            128 / bits
        ),
        "x86" | "v4" => {
            let p = if width == 128 {
                "_mm".into()
            } else {
                format!("_mm{width}")
            };
            let wide = bits * 2;
            let mask = (1u32 << bits) - 1;
            // Each wide lane contains one little-endian adjacent pair.
            // Mask/shift zero-extend both terms; their sum fits wide bits.
            // This also stays inside SSE2 for the baseline 128-bit methods.
            format!(
                "{p}_add_epi{wide}({p}_and_si{width}({a}, {p}_set1_epi{wide}({mask})), {p}_srli_epi{wide}::<{bits}>({a}))"
            )
        }
        _ => unreachable!(),
    }
}

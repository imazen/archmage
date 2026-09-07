//! Unsigned pairwise widening sums. Output lane k covers input lanes 2k, 2k+1.
//! Reuse the unsigned widening shapes; emit into their existing source backends.
use super::backend_gen_widen_narrow::{WidenPair, all_widen_pairs};
use indoc::formatdoc;

fn pair(name: &str) -> Option<WidenPair> {
    all_widen_pairs()
        .into_iter()
        .find(|p| p.src == name && !p.signed)
}

pub(super) fn trait_methods(name: &str) -> String {
    let Some(p) = pair(name) else {
        return String::new();
    };
    let src = p.src_backend();
    let dst = p.dst_backend();
    formatdoc! {r#"
        /// Exact unsigned adjacent sums: output[k] = widened a[2*k] + widened a[2*k+1].
        fn pairwise_widen_add(self, a: <Self as super::{src}>::Repr) -> <Self as super::{dst}>::Repr
        where Self: super::{dst};
    "#}
}

pub(super) fn generic(name: &str) -> String {
    let Some(p) = pair(name) else {
        return String::new();
    };
    let src_backend = p.src_backend();
    let dst_backend = p.dst_backend();
    let dst = p.dst;
    formatdoc! {r#"
        impl<T: crate::simd::backends::{src_backend}> {name}<T> {{
            /// Sum adjacent pairs into unsigned lanes twice as wide.
            ///
            /// Output lane `k` is `self[2*k] + self[2*k+1]`, with both
            /// inputs widened before addition. The result is exact for the
            /// full input range; no lane wraps or saturates. Pair ordering
            /// is unchanged across native and polyfilled widths.
            /// Subsequent accumulation uses the destination's normal wrapping addition.
            #[inline(always)]
            pub fn pairwise_widen_add(self) -> super::{dst}<T>
            where T: crate::simd::backends::{dst_backend} {{
                super::{dst}::from_repr_unchecked(self.1, <T as crate::simd::backends::{src_backend}>::pairwise_widen_add(self.1, self.0))
            }}
        }}
    "#}
}

pub(super) fn methods(arch: &str, token: &str, name: &str) -> String {
    let Some(p) = pair(name) else {
        return String::new();
    };
    let w = p.width_bits;
    let bits = w / p.src_lanes;
    let native = match arch {
        "v4" => 512,
        "x86" => w.min(256),
        _ => 128,
    };
    let sr = super::backend_gen_integer_ops::repr(arch, p.src_elem, w);
    let dr = super::backend_gen_integer_ops::repr(arch, p.dst_elem, w);
    let attr = match arch {
        "scalar" | "wasm" => "#[inline(always)]".into(),
        "x86" if w == 128 => "sse2_baseline! {".into(),
        _ => super::backend_syntax::arcane(token),
    };
    let close = if arch == "x86" && w == 128 { "}" } else { "" };
    let body = if arch == "scalar" {
        let dst = p.dst_elem;
        // Widen BEFORE adding: maxima are 510 (u8 -> u16) and 131070
        // (u16 -> u32). Wrapping addition is therefore exact, and also avoids
        // an unnecessary overflow check in unoptimized scalar code.
        format!("core::array::from_fn(|k| {dst}::from(a[2*k]).wrapping_add({dst}::from(a[2*k+1])))")
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
            // Native segments have an even number of input lanes, so no
            // adjacent pair crosses a segment boundary. Concatenation is exact.
            format!("[{}]", expressions.join(", "))
        }
    };
    formatdoc! {r#"
        {attr}
        fn pairwise_widen_add(self, a: {sr}) -> {dr} {{ {body} }}
        {close}
    "#}
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

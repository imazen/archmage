//! Integer pairwise products and full-range absolute differences.
//! Keep the scalar definitions and ISA formulations together for review.
use indoc::formatdoc;

fn upper(s: &str) -> String {
    s[..1].to_uppercase() + &s[1..]
}

pub(super) fn trait_names() -> Vec<(usize, String)> {
    [128, 256, 512]
        .into_iter()
        .flat_map(|w| {
            [
                (w, format!("I16x{}Pairwise", w / 16)),
                (w, format!("I16x{}AbsDiff", w / 16)),
                (w, format!("U8x{}AbsDiff", w / 8)),
            ]
        })
        .collect()
}

pub(super) fn traits() -> String {
    let mut out = String::new();
    for w in [128, 256, 512] {
        let gate = if w == 512 {
            "#[cfg(feature = \"w512\")]"
        } else {
            ""
        };
        let (n, m) = (w / 16, w / 32);
        out += &formatdoc! {r#"
            {gate}
            /// Adjacent signed 16-bit dot products with wrapping 32-bit results.
            pub trait I16x{n}Pairwise: super::I16x{n}Backend + super::I32x{m}Backend {{
                /// Lane k = a[2k]*b[2k] + a[2k+1]*b[2k+1], modulo 2^32.
                fn madd_adjacent(self, a: <Self as super::I16x{n}Backend>::Repr, b: <Self as super::I16x{n}Backend>::Repr) -> <Self as super::I32x{m}Backend>::Repr;
            }}
        "#};
        for (elem, n, dst) in [("i16", w / 16, "u16"), ("u8", w / 8, "u8")] {
            let src = format!("{}{n}", upper(&format!("{elem}x")));
            let dest = format!("{}{n}", upper(&format!("{dst}x")));
            let sum = if elem == "u8" {
                format!(
                    "/// Exact widening sum, at most {}.\nfn reduce_add_u32(self, a: <Self as super::{src}Backend>::Repr) -> u32;\n/// Exact sum of absolute byte differences, using a native SAD where available.\nfn sum_abs_diff(self, a: <Self as super::{src}Backend>::Repr, b: <Self as super::{src}Backend>::Repr) -> u32;",
                    n * 255
                )
            } else {
                String::new()
            };
            let extra = if src == dest {
                String::new()
            } else {
                format!(" + super::{dest}Backend")
            };
            out += &formatdoc! {r#"
                {gate}
                /// Full-range absolute differences, with unsigned output lanes.
                pub trait {src}AbsDiff: super::{src}Backend{extra} {{
                    /// Exact |a[i] - b[i]|; signed inputs can span the full unsigned range.
                    fn abs_diff(self, a: <Self as super::{src}Backend>::Repr, b: <Self as super::{src}Backend>::Repr) -> <Self as super::{dest}Backend>::Repr;
                    {sum}
                }}
            "#};
        }
    }
    out
}

pub(super) fn generic(name: &str) -> String {
    let Some((elem, lanes)) = name.split_once('x') else {
        return String::new();
    };
    if !matches!(elem, "i16" | "u8") {
        return String::new();
    }
    let n: usize = lanes.parse().unwrap();
    let bound = upper(name);
    let dest = if elem == "i16" {
        format!("u16x{n}")
    } else {
        name.to_owned()
    };
    let signed_note = if elem == "i16" {
        "/// MIN.abs_diff(MAX) is u16::MAX."
    } else {
        ""
    };
    let mut out = formatdoc! {r#"
        impl<T: crate::simd::backends::{bound}AbsDiff> {name}<T> {{
            {signed_note}
            /// Exact lane-wise absolute difference, without saturation or wrapping.
            #[inline(always)]
            pub fn abs_diff(self, rhs: Self) -> super::{dest}<T> {{
                super::{dest}::from_repr_unchecked(self.1, T::abs_diff(self.1, self.0, rhs.0))
            }}
        }}
    "#};
    if elem == "i16" {
        let m = n / 2;
        out += &formatdoc! {r#"
            impl<T: crate::simd::backends::{bound}Pairwise> {name}<T> {{
                /// Multiply signed lanes, then sum adjacent pairs into i32 lanes.
                /// Lane k uses exactly input lanes 2k and 2k+1, in that order.
                /// The sum wraps modulo 2^32: two MIN*MIN products yield i32::MIN.
                /// Neither this operation nor the WASM dot instruction saturates.
                #[inline(always)]
                pub fn madd_adjacent(self, rhs: Self) -> super::i32x{m}<T> {{
                    super::i32x{m}::from_repr_unchecked(self.1, T::madd_adjacent(self.1, self.0, rhs.0))
                }}

                /// Subtract the adjacent pairwise dot product from accumulator.
                /// Each i32 lane wraps modulo 2^32. This is the accumulator form
                /// used by Wiener statistics, not a difference between products.
                #[inline(always)]
                pub fn msub_adjacent(self, rhs: Self, accumulator: super::i32x{m}<T>) -> super::i32x{m}<T> {{
                    accumulator - self.madd_adjacent(rhs)
                }}
            }}
        "#};
    } else {
        out += &formatdoc! {r#"
            impl<T: crate::simd::backends::{bound}AbsDiff> {name}<T> {{
                /// Sum all lanes exactly into u32 (unlike wrapping reduce_add).
                #[inline(always)]
                pub fn reduce_add_u32(self) -> u32 {{ T::reduce_add_u32(self.1, self.0) }}

                /// Exact sum of absolute byte differences (SAD).
                /// Prefer this fused operation when only the sum is needed:
                /// x86 can use psadbw directly instead of materializing abs_diff.
                #[inline(always)]
                pub fn sum_abs_diff(self, rhs: Self) -> u32 {{ T::sum_abs_diff(self.1, self.0, rhs.0) }}
            }}
        "#};
    }
    out
}

fn repr(arch: &str, elem: &str, width: usize) -> String {
    let native = match arch {
        "x86" => 256.min(width),
        "v4" => width,
        _ => 128.min(width),
    };
    let bits = if elem == "u8" {
        8
    } else if elem == "i32" {
        32
    } else {
        16
    };
    if arch == "scalar" {
        return format!("[{elem}; {}]", width / bits);
    }
    let ty = match arch {
        "x86" | "v4" => format!("__m{native}i"),
        "wasm" => "v128".into(),
        "neon" => format!(
            "{}{}x{}_t",
            if elem.starts_with('u') { "uint" } else { "int" },
            bits,
            128 / bits
        ),
        _ => unreachable!(),
    };
    if width == native {
        ty
    } else {
        format!("[{ty}; {}]", width / native)
    }
}

pub(super) fn impls(arch: &str, token: &str, w512: bool) -> String {
    let mut out = String::new();
    for w in [128, 256, 512].into_iter().filter(|w| (*w == 512) == w512) {
        let native = if arch == "v4" {
            512
        } else if arch == "x86" {
            256.min(w)
        } else {
            128
        };
        let parts = w / native;
        for op in ["madd_adjacent", "abs_i16", "abs_u8"] {
            let elem = if matches!(op, "madd_adjacent" | "abs_i16") {
                "i16"
            } else {
                "u8"
            };
            let dst = match op {
                "madd_adjacent" => "i32",
                "abs_i16" => "u16",
                _ => "u8",
            };
            let n = w / if elem == "u8" { 8 } else { 16 };
            let src_name = format!("{}{n}", upper(&format!("{elem}x")));
            let tn = format!(
                "{src_name}{}",
                if op == "madd_adjacent" {
                    "Pairwise"
                } else {
                    "AbsDiff"
                }
            );
            let sr = repr(arch, elem, w);
            let dr = repr(arch, dst, w);
            let method = if op == "madd_adjacent" {
                op
            } else {
                "abs_diff"
            };
            let attr = if arch == "scalar" || arch == "wasm" {
                "#[inline(always)]".into()
            } else if arch == "x86" && w == 128 {
                "sse2_baseline! {".into()
            } else {
                super::backend_syntax::arcane(token)
            };
            let close = if arch == "x86" && w == 128 { "}" } else { "" };
            let body = if arch == "scalar" {
                if op == "madd_adjacent" {
                    "core::array::from_fn(|k| (a[2*k] as i32 * b[2*k] as i32).wrapping_add(a[2*k+1] as i32 * b[2*k+1] as i32))".into()
                } else {
                    "core::array::from_fn(|k| a[k].abs_diff(b[k]))".into()
                }
            } else {
                let expressions: Vec<_> = (0..parts)
                    .map(|part| {
                        let a = if parts == 1 {
                            "a".into()
                        } else {
                            format!("a[{part}]")
                        };
                        let b = if parts == 1 {
                            "b".into()
                        } else {
                            format!("b[{part}]")
                        };
                        native_expr(arch, native, op, &a, &b)
                    })
                    .collect();
                if parts == 1 {
                    expressions[0].clone()
                } else {
                    format!("[{}]", expressions.join(", "))
                }
            };
            out += &format!(
                "impl {tn} for archmage::{token} {{\n{attr}\nfn {method}(self, a: {sr}, b: {sr}) -> {dr} {{ {body} }}\n{close}\n"
            );
            if op == "abs_u8" {
                let body = if arch == "scalar" {
                    "a.into_iter().map(u32::from).sum()".into()
                } else {
                    let sums: Vec<_> = (0..parts)
                        .map(|i| {
                            native_sum(
                                arch,
                                native,
                                &if parts == 1 {
                                    "a".into()
                                } else {
                                    format!("a[{i}]")
                                },
                                None,
                            )
                        })
                        .collect();
                    if sums.len() == 1 {
                        sums[0].clone()
                    } else {
                        sums.into_iter()
                            .map(|sum| format!("({sum})"))
                            .collect::<Vec<_>>()
                            .join(" + ")
                    }
                };
                out += &format!(
                    "{attr}\nfn reduce_add_u32(self, a: {sr}) -> u32 {{ {body} }}\n{close}\n"
                );
                let sad = if arch == "scalar" {
                    "a.into_iter().zip(b).map(|(a,b)| u32::from(a.abs_diff(b))).sum()".into()
                } else {
                    let sums: Vec<_> = (0..parts)
                        .map(|i| {
                            let a = if parts == 1 {
                                "a".into()
                            } else {
                                format!("a[{i}]")
                            };
                            let b = if parts == 1 {
                                "b".into()
                            } else {
                                format!("b[{i}]")
                            };
                            native_sum(arch, native, &a, Some(&b))
                        })
                        .collect();
                    if sums.len() == 1 {
                        sums[0].clone()
                    } else {
                        sums.into_iter()
                            .map(|sum| format!("({sum})"))
                            .collect::<Vec<_>>()
                            .join(" + ")
                    }
                };
                out += &format!(
                    "{attr}\nfn sum_abs_diff(self, a: {sr}, b: {sr}) -> u32 {{ {sad} }}\n{close}\n"
                );
            }
            out += "}\n";
        }
    }
    out
}

fn native_expr(arch: &str, width: usize, op: &str, a: &str, b: &str) -> String {
    if matches!(arch, "x86" | "v4") {
        let p = if width == 128 {
            "_mm".into()
        } else {
            format!("_mm{width}")
        };
        return match op {
            "madd_adjacent" => format!("{p}_madd_epi16({a}, {b})"),
            // Signed max/min followed by wrapping subtraction is an exact
            // unsigned magnitude, even when the mathematical distance is 65535.
            "abs_i16" => format!("{p}_sub_epi16({p}_max_epi16({a}, {b}), {p}_min_epi16({a}, {b}))"),
            "abs_u8" => format!("{p}_sub_epi8({p}_max_epu8({a}, {b}), {p}_min_epu8({a}, {b}))"),
            _ => unreachable!(),
        };
    }
    match (arch, op) {
        // Two signed widening multiplies and a 32-bit pairwise add, NOT vpaddlq
        // (which would widen again to 64 bits and halve the lane count).
        ("neon", "madd_adjacent") => format!(
            "vpaddq_s32(vmull_s16(vget_low_s16({a}), vget_low_s16({b})), vmull_high_s16({a}, {b}))"
        ),
        ("neon", "abs_i16") => format!("vreinterpretq_u16_s16(vabdq_s16({a}, {b}))"),
        ("neon", "abs_u8") => format!("vabdq_u8({a}, {b})"),
        ("wasm", "madd_adjacent") => format!("i32x4_dot_i16x8({a}, {b})"),
        ("wasm", "abs_i16") => format!("i16x8_sub(i16x8_max({a}, {b}), i16x8_min({a}, {b}))"),
        ("wasm", "abs_u8") => format!("u8x16_sub(u8x16_max({a}, {b}), u8x16_min({a}, {b}))"),
        _ => unreachable!(),
    }
}

// Each native block sums at most 64 * 255 = 16320, so all widening
// intermediate additions and the final u32 sum are exact. x86 SAD produces
// eight-byte group sums in u64 lanes; NEON widens before reducing; WASM
// widens twice before the scalar horizontal sum. No narrowing loses bits.
fn native_sum(arch: &str, width: usize, a: &str, b: Option<&str>) -> String {
    let difference;
    let a = if let Some(b) = b.filter(|_| arch == "neon" || arch == "wasm") {
        difference = native_expr(arch, width, "abs_u8", a, b);
        &difference
    } else {
        a
    };
    match arch {
        "neon" => format!("vaddlvq_u8({a}) as u32"),
        "wasm" => format!(
            "{{ let s = u32x4_extadd_pairwise_u16x8(u16x8_extadd_pairwise_u8x16({a})); u32x4_extract_lane::<0>(s) + u32x4_extract_lane::<1>(s) + u32x4_extract_lane::<2>(s) + u32x4_extract_lane::<3>(s) }}"
        ),
        "x86" | "v4" => {
            let p = if width == 128 {
                "_mm".into()
            } else {
                format!("_mm{width}")
            };
            let reduction = match width {
                128 => "_mm_cvtsi128_si64(_mm_add_epi64(s, _mm_srli_si128::<8>(s))) as u32",
                256 => {
                    "{ let s = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s)); _mm_cvtsi128_si64(_mm_add_epi64(s, _mm_srli_si128::<8>(s))) as u32 }"
                }
                512 => "_mm512_reduce_add_epi64(s) as u32",
                _ => unreachable!(),
            };
            {
                let rhs = b
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("{p}_setzero_si{width}()"));
                format!("{{ let s = {p}_sad_epu8({a}, {rhs}); {reduction} }}")
            }
        }
        _ => unreachable!(),
    }
}

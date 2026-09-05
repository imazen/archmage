//! Backend codegen for integer **widening** and **saturating narrowing**.
//!
//! Scope is deliberately narrower than "all the conversions the ISAs offer";
//! `docs/CROSS-ISA-INT-PRIMITIVES.md` §3 has the full audit. In short:
//!
//! * **Widening** (`widen_low` / `widen_high`, `u8→u16`, `i8→i16`, `u16→u32`,
//!   `i16→i32`) is native at every tier, in natural lane order, with no
//!   semantic divergence — one instruction per half everywhere.
//! * **Narrowing** is exposed only in the **signed-source, saturating** shape
//!   (`i16 → i8`/`u8`, `i32 → i16`/`u16`). x86 and wasm have *no*
//!   unsigned-source narrowing at all: `packus` clamps its input as `i16`, so a
//!   generic `u16 → u8` saturating narrow would return `0` on x86/wasm and
//!   `255` on NEON for every lane above `0x7FFF`. Typing the op on the signed
//!   source types makes that divergence unrepresentable rather than merely
//!   documented.
//! * Truncating (non-saturating) narrowing and the rounding narrowing shift are
//!   NEON-only as single instructions and are **not** exposed.
//!
//! Two lane-order traps are handled here rather than left to the caller:
//!
//! 1. `_mm256_packs/packus_*` interleave in 64-bit groups —
//!    `[a.lo, b.lo, a.hi, b.hi]` (`core_arch/x86/avx2.rs`, the `IDXS` constant
//!    in each `pack*` body) — so the AVX2 arm applies
//!    `_mm256_permute4x64_epi64::<0xD8>`.
//! 2. `_mm512_packs/packus_*` is worse: it interleaves across all four 128-bit
//!    lanes (`core_arch/x86/avx512bw.rs`, `_mm512_packus_epi16`'s `IDXS`). The
//!    AVX-512 arm therefore does not use `pack*` at all — it uses the
//!    `vpmovswb`/`vpmovuswb` (`_mm512_cvtsepi16_epi8` / `_mm512_cvtusepi16_epi8`)
//!    family, which is natural by construction, and concatenates the two
//!    256-bit halves with `_mm512_inserti64x4::<1>`.
//!
//! Widening has no such trap on AVX2: `_mm256_cvtepu8_epi16` is a plain
//! element-wise `simd_cast` of its 128-bit input (`core_arch/x86/avx2.rs:884`),
//! not an `unpack`, so there is nothing to undo.

use indoc::{formatdoc, indoc};

// ============================================================================
// Data model
// ============================================================================

/// One `src -> dst` widening family (both `widen_low` and `widen_high`).
#[derive(Clone, Copy, Debug)]
pub(super) struct WidenPair {
    /// Source type name, e.g. `"u8x16"`.
    pub src: &'static str,
    /// Destination type name, e.g. `"u16x8"`.
    pub dst: &'static str,
    /// Source element, e.g. `"u8"`.
    pub src_elem: &'static str,
    /// Destination element, e.g. `"u16"`.
    pub dst_elem: &'static str,
    /// Signed source (sign-extend) vs unsigned source (zero-extend).
    pub signed: bool,
    /// Register width in bits — identical for src and dst (128 / 256 / 512).
    pub width_bits: usize,
    /// Lane count of the source (twice the destination's).
    pub src_lanes: usize,
}

/// One saturating-narrowing family: two signed-source vectors to one vector of
/// twice as many half-width lanes, in both the signed-destination and
/// unsigned-destination flavours.
#[derive(Clone, Copy, Debug)]
pub(super) struct NarrowPair {
    /// Source type name, e.g. `"i16x8"` — always signed.
    pub src: &'static str,
    /// Source element, e.g. `"i16"`.
    pub src_elem: &'static str,
    /// Signed destination type name, e.g. `"i8x16"`.
    pub sdst: &'static str,
    /// Signed destination element, e.g. `"i8"`.
    pub sdst_elem: &'static str,
    /// Unsigned destination type name, e.g. `"u8x16"`.
    pub udst: &'static str,
    /// Unsigned destination element, e.g. `"u8"`.
    pub udst_elem: &'static str,
    /// Register width in bits (128 / 256 / 512).
    pub width_bits: usize,
    /// Lane count of one source operand.
    pub src_lanes: usize,
}

/// `"u8x16"` -> `"U8x16"` (leading letter only; `to_uppercase` would give
/// `U8X16`).
fn upper(name: &str) -> String {
    let mut cs = name.chars();
    match cs.next() {
        Some(c) => c.to_uppercase().collect::<String>() + cs.as_str(),
        None => String::new(),
    }
}

impl WidenPair {
    /// `"U8x16Widen"`.
    pub fn trait_name(&self) -> String {
        format!("{}Widen", upper(self.src))
    }
    /// `"U8x16Backend"`.
    fn src_backend(&self) -> String {
        format!("{}Backend", upper(self.src))
    }
    /// `"U16x8Backend"`.
    fn dst_backend(&self) -> String {
        format!("{}Backend", upper(self.dst))
    }
    /// `"widen_low_u8_to_u16"` / `"widen_high_u8_to_u16"`.
    pub fn method(&self, half: Half) -> String {
        format!(
            "widen_{}_{}_to_{}",
            half.name(),
            self.src_elem,
            self.dst_elem
        )
    }
    fn dst_lanes(&self) -> usize {
        self.src_lanes / 2
    }
    /// How many native 128-bit sub-vectors the polyfilled repr holds
    /// (1 = native single register).
    fn subs(&self) -> usize {
        self.width_bits / 128
    }
}

impl NarrowPair {
    /// `"I16x8Narrow"`.
    pub fn trait_name(&self) -> String {
        format!("{}Narrow", upper(self.src))
    }
    fn src_backend(&self) -> String {
        format!("{}Backend", upper(self.src))
    }
    fn sdst_backend(&self) -> String {
        format!("{}Backend", upper(self.sdst))
    }
    fn udst_backend(&self) -> String {
        format!("{}Backend", upper(self.udst))
    }
    /// `"narrow_saturating_i16_to_i8"` / `"narrow_saturating_i16_to_u8"`.
    pub fn method(&self, dst_signed: bool) -> String {
        let d = if dst_signed {
            self.sdst_elem
        } else {
            self.udst_elem
        };
        format!("narrow_saturating_{}_to_{}", self.src_elem, d)
    }
    fn subs(&self) -> usize {
        self.width_bits / 128
    }
}

/// Which half of the source a widening reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Half {
    Low,
    High,
}

impl Half {
    fn name(self) -> &'static str {
        match self {
            Half::Low => "low",
            Half::High => "high",
        }
    }
}

/// Every widening family, in generation order.
pub(super) fn all_widen_pairs() -> Vec<WidenPair> {
    vec![
        WidenPair {
            src: "u8x16",
            dst: "u16x8",
            src_elem: "u8",
            dst_elem: "u16",
            signed: false,
            width_bits: 128,
            src_lanes: 16,
        },
        WidenPair {
            src: "u16x8",
            dst: "u32x4",
            src_elem: "u16",
            dst_elem: "u32",
            signed: false,
            width_bits: 128,
            src_lanes: 8,
        },
        WidenPair {
            src: "i8x16",
            dst: "i16x8",
            src_elem: "i8",
            dst_elem: "i16",
            signed: true,
            width_bits: 128,
            src_lanes: 16,
        },
        WidenPair {
            src: "i16x8",
            dst: "i32x4",
            src_elem: "i16",
            dst_elem: "i32",
            signed: true,
            width_bits: 128,
            src_lanes: 8,
        },
        WidenPair {
            src: "u8x32",
            dst: "u16x16",
            src_elem: "u8",
            dst_elem: "u16",
            signed: false,
            width_bits: 256,
            src_lanes: 32,
        },
        WidenPair {
            src: "u16x16",
            dst: "u32x8",
            src_elem: "u16",
            dst_elem: "u32",
            signed: false,
            width_bits: 256,
            src_lanes: 16,
        },
        WidenPair {
            src: "i8x32",
            dst: "i16x16",
            src_elem: "i8",
            dst_elem: "i16",
            signed: true,
            width_bits: 256,
            src_lanes: 32,
        },
        WidenPair {
            src: "i16x16",
            dst: "i32x8",
            src_elem: "i16",
            dst_elem: "i32",
            signed: true,
            width_bits: 256,
            src_lanes: 16,
        },
        WidenPair {
            src: "u8x64",
            dst: "u16x32",
            src_elem: "u8",
            dst_elem: "u16",
            signed: false,
            width_bits: 512,
            src_lanes: 64,
        },
        WidenPair {
            src: "u16x32",
            dst: "u32x16",
            src_elem: "u16",
            dst_elem: "u32",
            signed: false,
            width_bits: 512,
            src_lanes: 32,
        },
        WidenPair {
            src: "i8x64",
            dst: "i16x32",
            src_elem: "i8",
            dst_elem: "i16",
            signed: true,
            width_bits: 512,
            src_lanes: 64,
        },
        WidenPair {
            src: "i16x32",
            dst: "i32x16",
            src_elem: "i16",
            dst_elem: "i32",
            signed: true,
            width_bits: 512,
            src_lanes: 32,
        },
    ]
}

/// Every saturating-narrowing family, in generation order.
pub(super) fn all_narrow_pairs() -> Vec<NarrowPair> {
    vec![
        NarrowPair {
            src: "i16x8",
            src_elem: "i16",
            sdst: "i8x16",
            sdst_elem: "i8",
            udst: "u8x16",
            udst_elem: "u8",
            width_bits: 128,
            src_lanes: 8,
        },
        NarrowPair {
            src: "i32x4",
            src_elem: "i32",
            sdst: "i16x8",
            sdst_elem: "i16",
            udst: "u16x8",
            udst_elem: "u16",
            width_bits: 128,
            src_lanes: 4,
        },
        NarrowPair {
            src: "i16x16",
            src_elem: "i16",
            sdst: "i8x32",
            sdst_elem: "i8",
            udst: "u8x32",
            udst_elem: "u8",
            width_bits: 256,
            src_lanes: 16,
        },
        NarrowPair {
            src: "i32x8",
            src_elem: "i32",
            sdst: "i16x16",
            sdst_elem: "i16",
            udst: "u16x16",
            udst_elem: "u16",
            width_bits: 256,
            src_lanes: 8,
        },
        NarrowPair {
            src: "i16x32",
            src_elem: "i16",
            sdst: "i8x64",
            sdst_elem: "i8",
            udst: "u8x64",
            udst_elem: "u8",
            width_bits: 512,
            src_lanes: 32,
        },
        NarrowPair {
            src: "i32x16",
            src_elem: "i32",
            sdst: "i16x32",
            sdst_elem: "i16",
            udst: "u16x32",
            udst_elem: "u16",
            width_bits: 512,
            src_lanes: 16,
        },
    ]
}

// ============================================================================
// Trait definitions -> backends/widen_narrow.rs
// ============================================================================

pub(super) fn generate_widen_narrow_traits() -> String {
    let mut code = String::from(indoc_header());

    for p in all_widen_pairs() {
        let gate = w512_gate(p.width_bits);
        let tn = p.trait_name();
        let sb = p.src_backend();
        let db = p.dst_backend();
        let (src, dst) = (p.src, p.dst);
        let de = p.dst_elem;
        let half_lanes = p.dst_lanes();
        let extend = if p.signed { "Sign" } else { "Zero" };
        let lo = p.method(Half::Low);
        let hi = p.method(Half::High);
        code.push_str(&formatdoc! {r#"
            /// {extend}-extending widening from `{src}` to `{dst}`.
            ///
            /// Native at every tier and in natural lane order everywhere; see
            /// `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
            {gate}pub trait {tn}: super::{sb} + super::{db} + SimdToken + Sealed + Copy + 'static {{
                /// {extend}-extend source lanes `0..{half_lanes}` to `{de}`.
                ///
                /// Result lane `i` is `a[i] as {de}` for `i` in `0..{half_lanes}`.
                fn {lo}(self, a: <Self as super::{sb}>::Repr) -> <Self as super::{db}>::Repr;

                /// {extend}-extend source lanes `{half_lanes}..{src_lanes}` to `{de}`.
                ///
                /// Result lane `i` is `a[i + {half_lanes}] as {de}`.
                fn {hi}(self, a: <Self as super::{sb}>::Repr) -> <Self as super::{db}>::Repr;
            }}

        "#, src_lanes = p.src_lanes});
    }

    for p in all_narrow_pairs() {
        let gate = w512_gate(p.width_bits);
        let tn = p.trait_name();
        let sb = p.src_backend();
        let sdb = p.sdst_backend();
        let udb = p.udst_backend();
        let (src, sdst, udst) = (p.src, p.sdst, p.udst);
        let (se, sde, ude) = (p.src_elem, p.sdst_elem, p.udst_elem);
        let n = p.src_lanes;
        let sm = p.method(true);
        let um = p.method(false);
        code.push_str(&formatdoc! {r#"
            /// Saturating narrowing from two `{src}` operands.
            ///
            /// Only the **signed-source** shape is offered: x86 and wasm have no
            /// unsigned-source narrowing instruction, so a `u{srcw} -> u{dstw}`
            /// saturating narrow would disagree across ISAs above `0x{hi_hex}`.
            /// See `docs/CROSS-ISA-INT-PRIMITIVES.md` §3.
            {gate}pub trait {tn}:
                super::{sb} + super::{sdb} + super::{udb} + SimdToken + Sealed + Copy + 'static
            {{
                /// Signed-saturating narrow to `{sdst}`.
                ///
                /// Result lane `i` is `a[i].clamp({sde}::MIN, {sde}::MAX)` for
                /// `i < {n}` and `b[i - {n}].clamp(..)` for `i >= {n}`.
                fn {sm}(
                    self,
                    a: <Self as super::{sb}>::Repr,
                    b: <Self as super::{sb}>::Repr,
                ) -> <Self as super::{sdb}>::Repr;

                /// Unsigned-saturating narrow to `{udst}`.
                ///
                /// Result lane `i` is `a[i].clamp(0, {ude}::MAX as {se})` for
                /// `i < {n}` and `b[i - {n}].clamp(..)` for `i >= {n}`.
                fn {um}(
                    self,
                    a: <Self as super::{sb}>::Repr,
                    b: <Self as super::{sb}>::Repr,
                ) -> <Self as super::{udb}>::Repr;
            }}

        "#,
            srcw = elem_bits(p.src_elem),
            dstw = elem_bits(p.sdst_elem),
            hi_hex = format!("{:X}", (1u64 << (elem_bits(p.src_elem) - 1)) - 1),
        });
    }

    code.push_str(&super::backend_gen_integer_ops::traits());
    code
}

fn indoc_header() -> &'static str {
    indoc! {r#"
        //! Widening and saturating-narrowing conversion traits for integer types.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        use super::sealed::Sealed;
        use archmage::SimdToken;

    "#}
}

fn w512_gate(width_bits: usize) -> &'static str {
    if width_bits == 512 {
        "#[cfg(feature = \"w512\")]\n"
    } else {
        ""
    }
}

// ============================================================================
// Shared helpers
// ============================================================================

/// Element bit width from an element name (`"u8"` -> 8).
fn elem_bits(elem: &str) -> usize {
    elem[1..]
        .parse()
        .expect("element name is a letter + bit width")
}

/// The 128-bit magetypes/wasm type name for an element (`"u8"` -> `"u8x16"`).
fn native128_name(elem: &str) -> String {
    format!("{elem}x{}", 128 / elem_bits(elem))
}

/// NEON 128-bit vector type for an element (`"i16"` -> `"int16x8_t"`).
fn neon_type(elem: &str) -> String {
    let bits = elem_bits(elem);
    let sign = if elem.starts_with('i') { "int" } else { "uint" };
    format!("{sign}{bits}x{}_t", 128 / bits)
}

/// NEON intrinsic suffix for an element (`"i16"` -> `"s16"`, `"u8"` -> `"u8"`).
fn neon_suffix(elem: &str) -> String {
    let bits = elem_bits(elem);
    let sign = if elem.starts_with('i') { 's' } else { 'u' };
    format!("{sign}{bits}")
}

/// Repr text for a scalar-backend type (`"u8x16"` with elem `"u8"`).
fn scalar_repr(elem: &str, lanes: usize) -> String {
    format!("[{elem}; {lanes}]")
}

/// Repr text for an x86 V3-backend type at a given register width.
fn x86_v3_repr(width_bits: usize) -> &'static str {
    match width_bits {
        128 => "__m128i",
        256 => "__m256i",
        512 => "[__m256i; 2]",
        _ => unreachable!("only 128/256/512-bit types are generated"),
    }
}

/// Repr text for a NEON-backend type at a given register width.
fn neon_repr(elem: &str, width_bits: usize) -> String {
    let sub = neon_type(elem);
    match width_bits {
        128 => sub,
        n => format!("[{sub}; {}]", n / 128),
    }
}

/// Repr text for a wasm128-backend type at a given register width.
fn wasm_repr(width_bits: usize) -> String {
    match width_bits {
        128 => "v128".to_string(),
        n => format!("[v128; {}]", n / 128),
    }
}

/// `a` when the repr is a single register, `a[k]` when it is an array.
fn sub_expr(name: &str, subs: usize, k: usize) -> String {
    if subs == 1 {
        name.to_string()
    } else {
        format!("{name}[{k}]")
    }
}

/// Wrap a list of per-sub expressions into a repr-shaped expression.
fn pack_subs(items: &[String]) -> String {
    if items.len() == 1 {
        items[0].clone()
    } else {
        format!("[{}]", items.join(", "))
    }
}

/// `impl Trait for archmage::Token {` header with the right cfg attributes.
fn impl_header(trait_name: &str, token: &str, arch: Option<&str>) -> String {
    let arch_attr = match arch {
        Some(a) => format!("#[cfg(target_arch = \"{a}\")]\n"),
        None => String::new(),
    };
    format!("\n{arch_attr}impl {trait_name} for archmage::{token} {{\n")
}

// ============================================================================
// Scalar backend
// ============================================================================

/// Scalar widen/narrow impls. This backend is the reference the differential
/// tests compare every other backend against, so both families are written as
/// literal per-lane arithmetic rather than delegating to anything clever.
///
/// **Widening reads all source lanes, then selects the half** (issue #77).
/// The obvious body — `from_fn(|i| a[i] as D)` over half the lanes — reads
/// only an 8-byte slice of the by-value source array, which LLVM's SROA
/// promotes to an `i64`; the lane reads become shift/truncate extracts that
/// the SLP vectorizer reassembles as scalar-to-vector inserts instead of one
/// vector load + extend (measured 4.6x slower on Zen 5, 1.8x on Apple M4).
/// Widening the *full* array first keeps the source as a memory-resident
/// aggregate, so each half auto-vectorizes to a plain
/// `load <N x i8>` + `zext`; the unused half is dead-code-eliminated, making
/// each method byte-identical to the casts written inline at the call site.
/// Extracting the half as a sub-array first (`split_at`/`as_chunks` + `map`)
/// is WORSE — the 8-byte half copy itself gets integer-promoted and both
/// halves degrade.
pub(super) fn generate_scalar_widen_narrow_impls(w512: bool) -> String {
    let mut code = String::new();

    for p in all_widen_pairs()
        .into_iter()
        .filter(|p| is_w512(p.width_bits) == w512)
    {
        let src_repr = scalar_repr(p.src_elem, p.src_lanes);
        let dst_repr = scalar_repr(p.dst_elem, p.dst_lanes());
        let de = p.dst_elem;
        let half = p.dst_lanes();
        let full_repr = scalar_repr(p.dst_elem, p.src_lanes);
        code.push_str(&impl_header(&p.trait_name(), "ScalarToken", None));
        code.push_str(&formatdoc! {r#"
                #[inline(always)]
                fn {lo}(self, a: {src_repr}) -> {dst_repr} {{
                    // Widen all lanes, keep the low half: dodges SROA integer
                    // promotion of a half-array read (#77); the high half is DCE'd.
                    let f: {full_repr} = core::array::from_fn(|i| a[i] as {de});
                    core::array::from_fn(|i| f[i])
                }}

                #[inline(always)]
                fn {hi}(self, a: {src_repr}) -> {dst_repr} {{
                    // Widen all lanes, keep the high half: dodges SROA integer
                    // promotion of a half-array read (#77); the low half is DCE'd.
                    let f: {full_repr} = core::array::from_fn(|i| a[i] as {de});
                    core::array::from_fn(|i| f[i + {half}])
                }}
        "#, lo = p.method(Half::Low), hi = p.method(Half::High)});
        code.push_str("}\n");
    }

    for p in all_narrow_pairs()
        .into_iter()
        .filter(|p| is_w512(p.width_bits) == w512)
    {
        let src_repr = scalar_repr(p.src_elem, p.src_lanes);
        let sdst_repr = scalar_repr(p.sdst_elem, p.src_lanes * 2);
        let udst_repr = scalar_repr(p.udst_elem, p.src_lanes * 2);
        let (se, sde, ude) = (p.src_elem, p.sdst_elem, p.udst_elem);
        let n = p.src_lanes;
        code.push_str(&impl_header(&p.trait_name(), "ScalarToken", None));
        code.push_str(&formatdoc! {r#"
                #[inline(always)]
                fn {sm}(self, a: {src_repr}, b: {src_repr}) -> {sdst_repr} {{
                    core::array::from_fn(|i| {{
                        let v: {se} = if i < {n} {{ a[i] }} else {{ b[i - {n}] }};
                        v.clamp({sde}::MIN as {se}, {sde}::MAX as {se}) as {sde}
                    }})
                }}

                #[inline(always)]
                fn {um}(self, a: {src_repr}, b: {src_repr}) -> {udst_repr} {{
                    core::array::from_fn(|i| {{
                        let v: {se} = if i < {n} {{ a[i] }} else {{ b[i - {n}] }};
                        v.clamp(0, {ude}::MAX as {se}) as {ude}
                    }})
                }}
        "#, sm = p.method(true), um = p.method(false)});
        code.push_str("}\n");
    }

    code.push_str(&super::backend_gen_integer_ops::impls(
        "scalar",
        "ScalarToken",
        w512,
    ));
    code
}

fn is_w512(width_bits: usize) -> bool {
    width_bits == 512
}

// ============================================================================
// x86 V3 backend (AVX2; 512-bit types are polyfilled as `[__m256i; 2]`)
// ============================================================================

/// `_mm_cvtepu8_epi16` / `_mm256_cvtepi16_epi32` / ... for one widen pair.
fn x86_cvt(p: &WidenPair, prefix: &str) -> String {
    let sign = if p.signed { 'i' } else { 'u' };
    format!(
        "{prefix}_cvtep{sign}{}_epi{}",
        elem_bits(p.src_elem),
        elem_bits(p.dst_elem)
    )
}

/// `_mm_packus_epi16` / `_mm256_packs_epi32` / ... for one narrow pair.
fn x86_pack(p: &NarrowPair, prefix: &str, dst_signed: bool) -> String {
    let kind = if dst_signed { "packs" } else { "packus" };
    format!("{prefix}_{kind}_epi{}", elem_bits(p.src_elem))
}

/// x86 V3 widen/narrow impls.
///
/// Widening needs no lane-order fixup at any width: `_mm256_cvtepu8_epi16` is
/// `simd_cast` of a 16-byte input (`core_arch/x86/avx2.rs:884`), not an
/// `unpack`, so the result is already `[a0, a1, ...]`.
///
/// Narrowing at 256-bit does: `_mm256_packs/packus_*` emit
/// `[a.lo, b.lo, a.hi, b.hi]` in 64-bit groups (the `IDXS` constant in each
/// `pack*` body in `core_arch/x86/avx2.rs`), so a
/// `_mm256_permute4x64_epi64::<0xD8>` — dst = src lanes `[0, 2, 1, 3]` —
/// restores `[a..., b...]`. Skipping it would make this the one tier that
/// silently produces different bytes.
pub(super) fn generate_x86_v3_widen_narrow_impls(w512: bool) -> String {
    let arcane = super::backend_syntax::arcane("X64V3Token");
    let mut code = String::new();

    for p in all_widen_pairs()
        .into_iter()
        .filter(|p| is_w512(p.width_bits) == w512)
    {
        let repr = x86_v3_repr(p.width_bits);
        code.push_str(&impl_header(&p.trait_name(), "X64V3Token", Some("x86_64")));
        for half in [Half::Low, Half::High] {
            let body = match p.width_bits {
                128 => {
                    let cvt = x86_cvt(&p, "_mm");
                    match half {
                        Half::Low => format!("{cvt}(a)"),
                        Half::High => format!("{cvt}(_mm_srli_si128::<8>(a))"),
                    }
                }
                256 => {
                    let cvt = x86_cvt(&p, "_mm256");
                    match half {
                        Half::Low => format!("{cvt}(_mm256_castsi256_si128(a))"),
                        Half::High => format!("{cvt}(_mm256_extracti128_si256::<1>(a))"),
                    }
                }
                _ => {
                    let cvt = x86_cvt(&p, "_mm256");
                    let k = if half == Half::Low { 0 } else { 1 };
                    format!(
                        "[{cvt}(_mm256_castsi256_si128(a[{k}])), {cvt}(_mm256_extracti128_si256::<1>(a[{k}]))]"
                    )
                }
            };
            code.push_str(&formatdoc! {r#"
                    {arcane}
                    fn {m}(self, a: {repr}) -> {repr} {{
                        {body}
                    }}

            "#, m = p.method(half)});
        }
        code.push_str("}\n");
    }

    for p in all_narrow_pairs()
        .into_iter()
        .filter(|p| is_w512(p.width_bits) == w512)
    {
        let repr = x86_v3_repr(p.width_bits);
        code.push_str(&impl_header(&p.trait_name(), "X64V3Token", Some("x86_64")));
        for dst_signed in [true, false] {
            let body = match p.width_bits {
                128 => format!("{}(a, b)", x86_pack(&p, "_mm", dst_signed)),
                256 => format!(
                    "_mm256_permute4x64_epi64::<0xD8>({}(a, b))",
                    x86_pack(&p, "_mm256", dst_signed)
                ),
                _ => {
                    let pk = x86_pack(&p, "_mm256", dst_signed);
                    format!(
                        "[_mm256_permute4x64_epi64::<0xD8>({pk}(a[0], a[1])), _mm256_permute4x64_epi64::<0xD8>({pk}(b[0], b[1]))]"
                    )
                }
            };
            code.push_str(&formatdoc! {r#"
                    {arcane}
                    fn {m}(self, a: {repr}, b: {repr}) -> {repr} {{
                        {body}
                    }}

            "#, m = p.method(dst_signed)});
        }
        code.push_str("}\n");
    }

    code.push_str(&super::backend_gen_integer_ops::impls(
        "x86",
        "X64V3Token",
        w512,
    ));
    code
}

// ============================================================================
// x86 V4 / V4x backends (native AVX-512, 512-bit types only)
// ============================================================================

/// AVX-512 widen/narrow impls for one 512-bit token.
///
/// Narrowing deliberately avoids `_mm512_packs/packus_*`: at 512-bit that
/// family interleaves across all four 128-bit lanes
/// (`core_arch/x86/avx512bw.rs`, `_mm512_packus_epi16`'s `IDXS`), which is a
/// worse scramble than AVX2's. `vpmovswb`/`vpmovsdw` (`_mm512_cvtsepi16_epi8`,
/// `_mm512_cvtsepi32_epi16`) narrow one register to a natural-order half, and
/// `_mm512_inserti64x4::<1>` concatenates the two halves.
///
/// The unsigned-destination form has no signed-source instruction
/// (`_mm512_cvtusepi16_epi8` reads its source as **unsigned**), so the source
/// is first clamped at zero with `_mm512_max_epi16`. After that clamp every
/// lane is in `0..=0x7FFF`, where the signed and unsigned readings coincide,
/// and the instruction's saturation to `0..=255` is exactly `packus`'s.
pub(super) fn generate_x86_v4_widen_narrow_impls(token: &str) -> String {
    let arcane = super::backend_syntax::arcane(token);
    let mut code = String::new();

    for p in all_widen_pairs()
        .into_iter()
        .filter(|p| is_w512(p.width_bits))
    {
        let cvt = x86_cvt(&p, "_mm512");
        code.push_str(&impl_header(&p.trait_name(), token, Some("x86_64")));
        for half in [Half::Low, Half::High] {
            let extract = match half {
                Half::Low => "_mm512_castsi512_si256(a)",
                Half::High => "_mm512_extracti64x4_epi64::<1>(a)",
            };
            code.push_str(&formatdoc! {r#"
                    {arcane}
                    fn {m}(self, a: __m512i) -> __m512i {{
                        {cvt}({extract})
                    }}

            "#, m = p.method(half)});
        }
        code.push_str("}\n");
    }

    for p in all_narrow_pairs()
        .into_iter()
        .filter(|p| is_w512(p.width_bits))
    {
        let sb = elem_bits(p.src_elem);
        let db = sb / 2;
        code.push_str(&impl_header(&p.trait_name(), token, Some("x86_64")));

        let signed_cvt = format!("_mm512_cvtsepi{sb}_epi{db}");
        code.push_str(&formatdoc! {r#"
                {arcane}
                fn {m}(self, a: __m512i, b: __m512i) -> __m512i {{
                    _mm512_inserti64x4::<1>(
                        _mm512_castsi256_si512({signed_cvt}(a)),
                        {signed_cvt}(b),
                    )
                }}

        "#, m = p.method(true)});

        let unsigned_cvt = format!("_mm512_cvtusepi{sb}_epi{db}");
        let max = format!("_mm512_max_epi{sb}");
        code.push_str(&formatdoc! {r#"
                {arcane}
                fn {m}(self, a: __m512i, b: __m512i) -> __m512i {{
                    // Clamp at zero so the unsigned-source instruction sees
                    // the same value the signed source held; above the
                    // clamp both readings agree.
                    let zero = _mm512_setzero_si512();
                    _mm512_inserti64x4::<1>(
                        _mm512_castsi256_si512({unsigned_cvt}({max}(a, zero))),
                        {unsigned_cvt}({max}(b, zero)),
                    )
                }}

        "#, m = p.method(false)});
        code.push_str("}\n");
    }

    code.push_str(&super::backend_gen_integer_ops::impls("v4", token, true));
    code
}

// ============================================================================
// NEON backend (128-bit native, wider types polyfilled as arrays)
// ============================================================================

/// NEON widen/narrow impls.
///
/// NEON is the one tier whose widening needs an explicit low/high *pair* of
/// intrinsics rather than one taking a half-register: `vmovl_u8` consumes a
/// 64-bit `uint8x8_t` (hence `vget_low_u8`) while `vmovl_high_u8` consumes the
/// full 128-bit register. That is a signature difference, not a semantic one —
/// the lane order is natural, as everywhere else.
pub(super) fn generate_neon_widen_narrow_impls(w512: bool) -> String {
    let arcane = super::backend_syntax::arcane("NeonToken");
    let mut code = String::new();

    for p in all_widen_pairs()
        .into_iter()
        .filter(|p| is_w512(p.width_bits) == w512)
    {
        let src_repr = neon_repr(p.src_elem, p.width_bits);
        let dst_repr = neon_repr(p.dst_elem, p.width_bits);
        let ns = neon_suffix(p.src_elem);
        let subs = p.subs();
        code.push_str(&impl_header(&p.trait_name(), "NeonToken", Some("aarch64")));
        for half in [Half::Low, Half::High] {
            // Dest sub `j` of this half is global half-register `g`; the source
            // 128-bit sub it comes from is `g / 2` and the half within that sub
            // is `g % 2`.
            let items: Vec<String> = (0..subs)
                .map(|j| {
                    let g = j + if half == Half::Low { 0 } else { subs };
                    let src = sub_expr("a", subs, g / 2);
                    if g % 2 == 0 {
                        format!("vmovl_{ns}(vget_low_{ns}({src}))")
                    } else {
                        format!("vmovl_high_{ns}({src})")
                    }
                })
                .collect();
            code.push_str(&formatdoc! {r#"
                    {arcane}
                    fn {m}(self, a: {src_repr}) -> {dst_repr} {{
                        {body}
                    }}

            "#, m = p.method(half), body = pack_subs(&items)});
        }
        code.push_str("}\n");
    }

    for p in all_narrow_pairs()
        .into_iter()
        .filter(|p| is_w512(p.width_bits) == w512)
    {
        let src_repr = neon_repr(p.src_elem, p.width_bits);
        let ns = neon_suffix(p.src_elem);
        let subs = p.subs();
        code.push_str(&impl_header(&p.trait_name(), "NeonToken", Some("aarch64")));
        for dst_signed in [true, false] {
            let dst_elem = if dst_signed { p.sdst_elem } else { p.udst_elem };
            let dst_repr = neon_repr(dst_elem, p.width_bits);
            let nds = neon_suffix(dst_elem);
            let narrow = if dst_signed {
                format!("vqmovn_{ns}")
            } else {
                format!("vqmovun_{ns}")
            };
            // Logical source order is `a`'s subs followed by `b`'s; dest sub
            // `j` combines logical subs `2j` and `2j + 1`.
            let logical: Vec<String> = (0..subs)
                .map(|k| sub_expr("a", subs, k))
                .chain((0..subs).map(|k| sub_expr("b", subs, k)))
                .collect();
            let items: Vec<String> = (0..subs)
                .map(|j| {
                    format!(
                        "vcombine_{nds}({narrow}({}), {narrow}({}))",
                        logical[2 * j],
                        logical[2 * j + 1]
                    )
                })
                .collect();
            code.push_str(&formatdoc! {r#"
                    {arcane}
                    fn {m}(self, a: {src_repr}, b: {src_repr}) -> {dst_repr} {{
                        {body}
                    }}

            "#, m = p.method(dst_signed), body = pack_subs(&items)});
        }
        code.push_str("}\n");
    }

    code.push_str(&super::backend_gen_integer_ops::impls(
        "neon",
        "NeonToken",
        w512,
    ));
    code
}

// ============================================================================
// wasm128 backend (128-bit native, wider types polyfilled as arrays)
// ============================================================================

/// wasm128 widen/narrow impls.
///
/// Both families map to single simd128 instructions in natural lane order:
/// `i16x8.extend_low_i8x16` / `_high_`, and `i8x16.narrow_i16x8_s` /
/// `_u` — whose Rust spellings already say "the input lanes are always
/// interpreted as signed integers" (`core_arch/wasm32/simd128.rs`), which is
/// the same signed-source restriction this trait family is typed around.
pub(super) fn generate_wasm_widen_narrow_impls(w512: bool) -> String {
    let mut code = String::new();

    for p in all_widen_pairs()
        .into_iter()
        .filter(|p| is_w512(p.width_bits) == w512)
    {
        let repr = wasm_repr(p.width_bits);
        let src128 = native128_name(p.src_elem);
        let dst128 = native128_name(p.dst_elem);
        let subs = p.subs();
        code.push_str(&impl_header(
            &p.trait_name(),
            "Wasm128Token",
            Some("wasm32"),
        ));
        for half in [Half::Low, Half::High] {
            let items: Vec<String> = (0..subs)
                .map(|j| {
                    let g = j + if half == Half::Low { 0 } else { subs };
                    let src = sub_expr("a", subs, g / 2);
                    let which = if g % 2 == 0 { "low" } else { "high" };
                    format!("{dst128}_extend_{which}_{src128}({src})")
                })
                .collect();
            code.push_str(&formatdoc! {r#"
                    #[inline(always)]
                    fn {m}(self, a: {repr}) -> {repr} {{
                        {body}
                    }}

            "#, m = p.method(half), body = pack_subs(&items)});
        }
        code.push_str("}\n");
    }

    for p in all_narrow_pairs()
        .into_iter()
        .filter(|p| is_w512(p.width_bits) == w512)
    {
        let repr = wasm_repr(p.width_bits);
        let src128 = native128_name(p.src_elem);
        let subs = p.subs();
        code.push_str(&impl_header(
            &p.trait_name(),
            "Wasm128Token",
            Some("wasm32"),
        ));
        for dst_signed in [true, false] {
            let dst_elem = if dst_signed { p.sdst_elem } else { p.udst_elem };
            let dst128 = native128_name(dst_elem);
            let logical: Vec<String> = (0..subs)
                .map(|k| sub_expr("a", subs, k))
                .chain((0..subs).map(|k| sub_expr("b", subs, k)))
                .collect();
            let items: Vec<String> = (0..subs)
                .map(|j| {
                    format!(
                        "{dst128}_narrow_{src128}({}, {})",
                        logical[2 * j],
                        logical[2 * j + 1]
                    )
                })
                .collect();
            code.push_str(&formatdoc! {r#"
                    #[inline(always)]
                    fn {m}(self, a: {repr}, b: {repr}) -> {repr} {{
                        {body}
                    }}

            "#, m = p.method(dst_signed), body = pack_subs(&items)});
        }
        code.push_str("}\n");
    }

    code.push_str(&super::backend_gen_integer_ops::impls(
        "wasm",
        "Wasm128Token",
        w512,
    ));
    code
}

//! `concat_shift` — cross-vector element shift, on the generated backend traits.
//!
//! The operation: `concat_shift::<N>(lo, hi)` yields lanes `N..N+LANES` of the
//! concatenation `[lo, hi]`. Every ISA has it under a different name —
//! `valignd` (AVX-512), `vperm2f128`+`vpalignr` (AVX2), `palignr` (SSSE3),
//! `EXT` (NEON), `i8x16.shuffle` (wasm) — and for every element type, because
//! the x86 and wasm forms are byte-granular and NEON has a `vextq_` per type.
//!
//! ## Why it lives on the backend trait rather than its own trait
//!
//! A standalone `ConcatShift` trait works, but every layer of generic code
//! between an entry point and the kernel then has to carry a
//! `V<T>: ConcatShift` bound. As a provided method on the generated backend
//! traits, code that is already generic over the backend gets it with no new
//! bound and no new trait.
//!
//! ## Why every backend that can, overrides the default
//!
//! The default body is a lane gather. MEASURED 2026-09-08 on a 7950X, reading
//! the disassembly of the gather inlined into a `target_feature` region:
//!
//! | | default body | native |
//! |---|---|---|
//! | f32x8, AVX2, N=1 | 6 ops (scalar `vmovss`/`vmovsd`) | 2 |
//! | f32x16, AVX-512, N=1 | 7 ops, mixed scalar | 1 |
//!
//! LLVM does not recover the funnel shift from the gather, so the default is
//! the correctness fallback (and the differential-test reference), not a fast
//! path.
//!
//! Which instruction the native form becomes is LLVM's choice and moves with
//! optimization level — the AVX-512 shift has been observed as `valignd`,
//! `vpermi2ps` and `vpermt2ps`, all one instruction. `scripts/verify-asm.sh`
//! therefore gates the ABSENCE of per-lane scalar moves rather than a mnemonic.
//!
//! ## Why `N` is `i32`
//!
//! An intrinsic immediate is a const argument, and a const generic parameter
//! cannot appear in a const *operation* on stable — `_mm512_alignr_epi32::<{N
//! as i32}>` does not compile. Typing `N` as the immediate's own type lets the
//! AVX-512 dword and qword paths pass it straight through. Everywhere the
//! immediate is a multiple of `N` (`palignr` counts bytes, not elements) that
//! trick is unavailable and the cases are spelled out in a `match`; `N` is
//! const, so one arm survives monomorphisation.

use indoc::formatdoc;

/// Bytes per element — the multiplier that turns a lane shift into the byte
/// immediate `palignr` actually takes.
fn elem_bytes(elem: &str) -> usize {
    match elem {
        "i8" | "u8" => 1,
        "i16" | "u16" => 2,
        "f32" | "i32" | "u32" => 4,
        "f64" | "i64" | "u64" => 8,
        other => panic!("concat_shift: unknown element type {other}"),
    }
}

/// Wrap `expr` in `cast` unless the cast is empty (integer vectors are already
/// in the domain `palignr` works in).
fn cast(c: &str, expr: &str) -> String {
    if c.is_empty() {
        expr.to_string()
    } else {
        format!("{c}({expr})")
    }
}

/// The provided method on a backend trait. Correct for every element type and
/// every backend; overridden wherever the ISA has the instruction.
pub(super) fn trait_decl(elem: &str, lanes: usize, trait_name: &str) -> String {
    formatdoc! {r#"

        // ====== Cross-vector element shift ======

        /// Lanes `N..N+{lanes}` of the concatenation `[lo, hi]`.
        ///
        /// `N == 0` returns `lo`; `N == {lanes}` would return `hi` and is rejected,
        /// since a caller that wants `hi` should just use it. This is the
        /// "funnel shift" — `valignd` on AVX-512, `vperm2f128` + `vpalignr` on
        /// AVX2, `EXT` on NEON, `i8x16.shuffle` on wasm — and it is what a
        /// 3-tap horizontal filter needs to derive the `x-1` and `x+1` vectors
        /// from two loads instead of three, and what a byte-shuffling kernel
        /// needs to slide a window.
        ///
        /// `N` is `i32` because that is the type of the ISA immediates it
        /// forwards to; a const generic cannot be cast in a const position.
        ///
        /// The default body is a lane gather, which LLVM does **not** recover
        /// into a funnel shift (measured 2026-09-08: 6-7 scalar moves where the
        /// native form is 1-2 instructions). It is the correctness fallback and
        /// the differential-test reference; every backend whose ISA has the
        /// instruction overrides it.
        #[inline(always)]
        fn concat_shift<const N: i32>(self, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {{
            const {{ assert!(N >= 0 && N < {lanes}, "concat_shift: N must be in 0..{lanes}") }};
            let n = N as usize;
            let a = <Self as {trait_name}>::to_array(self, lo);
            let b = <Self as {trait_name}>::to_array(self, hi);
            <Self as {trait_name}>::from_array(
                self,
                core::array::from_fn(|i| if n + i < {lanes} {{ a[n + i] }} else {{ b[n + i - {lanes}] }}),
            )
        }}
    "#}
}

/// Assemble a `match N` whose arms are already rendered, wrapping the whole
/// expression in `out` when the result has to be cast back out of the integer
/// domain.
fn match_expr(arms: &[String], out: &str) -> String {
    let body = format!(
        "match N {{\n                    0 => a,\n{}\n                    _ => unreachable!(),\n                }}",
        arms.join("\n")
    );
    if out.is_empty() {
        body
    } else {
        format!("{out}({body})")
    }
}

/// x86 native override for a 128- or 256-bit vector of any element type.
///
/// 128 bits is one `palignr` at a byte immediate of `N * elem_bytes`.
///
/// 256 bits is two instructions, because `vpalignr` shifts *within* each
/// 128-bit lane while a funnel shift crosses them: build the operand that
/// already holds the next 128 bits (`vperm2f128` / `vperm2x128`), then align
/// against it. Below the lane boundary the pair is `(cross, lo)`; above it,
/// `(hi, cross)`; exactly on it the crossing operand is already the answer.
pub(super) fn x86_native(elem: &str, lanes: usize, bits: usize, repr: &str, attr: &str) -> String {
    let es = elem_bytes(elem);
    let (to_i, from_i) = match (elem, bits) {
        ("f32", 128) => ("_mm_castps_si128", "_mm_castsi128_ps"),
        ("f64", 128) => ("_mm_castpd_si128", "_mm_castsi128_pd"),
        ("f32", 256) => ("_mm256_castps_si256", "_mm256_castsi256_ps"),
        ("f64", 256) => ("_mm256_castpd_si256", "_mm256_castsi256_pd"),
        _ => ("", ""),
    };
    let (a, b) = (cast(to_i, "lo"), cast(to_i, "hi"));

    if bits == 128 {
        let arms: Vec<String> = (1..lanes)
            .map(|k| {
                format!(
                    "                    {k} => _mm_alignr_epi8::<{}>(b, a),",
                    k * es
                )
            })
            .collect();
        return formatdoc! {r#"

            {attr}
            fn concat_shift<const N: i32>(self, lo: {repr}, hi: {repr}) -> {repr} {{
                const {{ assert!(N >= 0 && N < {lanes}, "concat_shift: N must be in 0..{lanes}") }};
                let a = {a};
                let b = {b};
                {body}
            }}
        "#, body = match_expr(&arms, from_i)};
    }

    let perm = match elem {
        "f32" => "_mm256_permute2f128_ps::<0x21>(lo, hi)",
        "f64" => "_mm256_permute2f128_pd::<0x21>(lo, hi)",
        _ => "_mm256_permute2x128_si256::<0x21>(lo, hi)",
    };
    let c = cast(to_i, "cross");
    let arms: Vec<String> = (1..lanes)
        .map(|k| {
            let s = k * es;
            match s.cmp(&16) {
                std::cmp::Ordering::Less => {
                    format!("                    {k} => _mm256_alignr_epi8::<{s}>(c, a),")
                }
                std::cmp::Ordering::Equal => format!("                    {k} => c,"),
                std::cmp::Ordering::Greater => format!(
                    "                    {k} => _mm256_alignr_epi8::<{}>(b, c),",
                    s - 16
                ),
            }
        })
        .collect();
    formatdoc! {r#"

        {attr}
        fn concat_shift<const N: i32>(self, lo: {repr}, hi: {repr}) -> {repr} {{
            const {{ assert!(N >= 0 && N < {lanes}, "concat_shift: N must be in 0..{lanes}") }};
            // `vpalignr` works within each 128-bit lane, so the operand for the
            // upper lane must already hold the next 128 bits: build
            // [lo.hi128, hi.lo128] and align against that.
            let cross = {perm};
            let a = {a};
            let b = {b};
            let c = {c};
            {body}
        }}
    "#, body = match_expr(&arms, from_i)}
}

/// AVX-512 native override for a 512-bit vector.
///
/// 32- and 64-bit elements are a single `valignd`/`valignq`, and `N` is already
/// the immediate they want, so it passes straight through with no `match`.
///
/// 8- and 16-bit elements have no full-width byte funnel shift: `vpalignr` is
/// per-128-bit-lane at every width. They are built from the dword shift, which
/// IS full-width. For a byte offset `s`, take `q = s / 16` whole 128-bit lanes
/// and `r = s % 16` bytes: `A` is the window at `16q` and `B` the window at
/// `16q + 16`, both dword-aligned so `valignd` reaches them, and `B`'s lane `i`
/// starts exactly 16 bytes after `A`'s — which is what makes the per-lane
/// `vpalignr` of `(B, A)` by `r` land on `[s + 16i, s + 16i + 16)`. At `q == 3`
/// the upper window is `hi` itself, which is also the only case where the
/// immediate would leave range. Three instructions at worst, one at best.
pub(super) fn avx512_native(elem: &str, lanes: usize, repr: &str, attr: &str) -> String {
    let es = elem_bytes(elem);
    let (to_i, from_i) = match elem {
        "f32" => ("_mm512_castps_si512", "_mm512_castsi512_ps"),
        "f64" => ("_mm512_castpd_si512", "_mm512_castsi512_pd"),
        _ => ("", ""),
    };
    let (a, b) = (cast(to_i, "lo"), cast(to_i, "hi"));

    if es >= 4 {
        let op = if es == 4 { "epi32" } else { "epi64" };
        let inner = format!("_mm512_alignr_{op}::<N>(b, a)");
        return formatdoc! {r#"

            {attr}
            fn concat_shift<const N: i32>(self, lo: {repr}, hi: {repr}) -> {repr} {{
                const {{ assert!(N >= 0 && N < {lanes}, "concat_shift: N must be in 0..{lanes}") }};
                let a = {a};
                let b = {b};
                {out}
            }}
        "#, out = cast(from_i, &inner)};
    }

    let arms: Vec<String> = (1..lanes)
        .map(|k| {
            let s = k * es;
            let (q, r) = (s / 16, s % 16);
            let lower = if q == 0 {
                "a".to_string()
            } else {
                format!("_mm512_alignr_epi32::<{}>(b, a)", q * 4)
            };
            let expr = if r == 0 {
                lower
            } else {
                let upper = if q == 3 {
                    "b".to_string()
                } else {
                    format!("_mm512_alignr_epi32::<{}>(b, a)", (q + 1) * 4)
                };
                format!("_mm512_alignr_epi8::<{r}>({upper}, {lower})")
            };
            format!("                    {k} => {expr},")
        })
        .collect();
    formatdoc! {r#"

        {attr}
        fn concat_shift<const N: i32>(self, lo: {repr}, hi: {repr}) -> {repr} {{
            const {{ assert!(N >= 0 && N < {lanes}, "concat_shift: N must be in 0..{lanes}") }};
            let a = {a};
            let b = {b};
            {body}
        }}
    "#, body = match_expr(&arms, from_i)}
}

/// NEON native override for a 128-bit vector: `vextq_*` is exactly this
/// operation, and exists for every element type.
pub(super) fn neon_native(elem: &str, lanes: usize, repr: &str, attr: &str) -> String {
    let suffix = match elem {
        "f32" => "f32",
        "f64" => "f64",
        "i8" => "s8",
        "u8" => "u8",
        "i16" => "s16",
        "u16" => "u16",
        "i32" => "s32",
        "u32" => "u32",
        "i64" => "s64",
        "u64" => "u64",
        other => panic!("concat_shift: no NEON suffix for {other}"),
    };
    let arms: Vec<String> = (1..lanes)
        .map(|k| format!("                    {k} => vextq_{suffix}::<{k}>(lo, hi),"))
        .collect();
    formatdoc! {r#"

        {attr}
        fn concat_shift<const N: i32>(self, lo: {repr}, hi: {repr}) -> {repr} {{
            const {{ assert!(N >= 0 && N < {lanes}, "concat_shift: N must be in 0..{lanes}") }};
            match N {{
                0 => lo,
        {arms}
                _ => unreachable!(),
            }}
        }}
    "#, arms = arms.join("\n")}
}

/// wasm SIMD128 native override: `i8x16.shuffle` selects 16 bytes from the
/// concatenation of two vectors, which is a funnel shift by construction, and
/// is byte-granular so one form covers every element type.
pub(super) fn wasm_native(elem: &str, lanes: usize) -> String {
    let es = elem_bytes(elem);
    let arms: Vec<String> = (1..lanes)
        .map(|k| {
            let idx: Vec<String> = (0..16).map(|i| (k * es + i).to_string()).collect();
            format!(
                "                    {k} => i8x16_shuffle::<{}>(lo, hi),",
                idx.join(", ")
            )
        })
        .collect();
    formatdoc! {r#"

        #[inline(always)]
        fn concat_shift<const N: i32>(self, lo: v128, hi: v128) -> v128 {{
            const {{ assert!(N >= 0 && N < {lanes}, "concat_shift: N must be in 0..{lanes}") }};
            match N {{
                0 => lo,
        {arms}
                _ => unreachable!(),
            }}
        }}
    "#, arms = arms.join("\n")}
}

/// Width-polyfill override: a `[Q; K]` repr shifts by taking, for each output
/// sub-vector, the sub-trait's shift over the adjacent pair of the 2K-long
/// sequence `[lo.., hi..]`. Pure delegation — the native instruction is at the
/// leaf, so no arch-specific code appears here.
pub(super) fn polyfill_delegate(
    lanes: usize,
    repr: &str,
    token: &str,
    sub_trait: &str,
    sub_lanes: usize,
) -> String {
    let subs = lanes / sub_lanes;
    let seq: Vec<String> = (0..subs)
        .map(|i| format!("lo[{i}]"))
        .chain((0..subs).map(|i| format!("hi[{i}]")))
        .collect();
    let arms: Vec<String> = (0..sub_lanes)
        .map(|e| {
            let pat = if e + 1 == sub_lanes {
                "_".to_string()
            } else {
                e.to_string()
            };
            format!(
                "{pat} => <archmage::{token} as {sub_trait}>::concat_shift::<{e}>(self, s[k + i], s[k + i + 1]),"
            )
        })
        .collect();
    formatdoc! {r#"

        #[inline(always)]
        fn concat_shift<const N: i32>(self, lo: {repr}, hi: {repr}) -> {repr} {{
            const {{ assert!(N >= 0 && N < {lanes}, "concat_shift: N must be in 0..{lanes}") }};
            let s = [{seq}];
            let k = (N as usize) / {sub_lanes};
            core::array::from_fn(|i| match N % {sub_lanes} {{
            {arms}
            }})
        }}
    "#, seq = seq.join(", "), arms = arms.join("\n            ")}
}

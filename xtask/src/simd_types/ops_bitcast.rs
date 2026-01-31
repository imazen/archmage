//! Bitcast code generation for SIMD types.
//!
//! Generates zero-cost reinterpret-cast methods between same-width SIMD types.
//! No token required — bitcast is purely a type-level operation.
//!
//! Three variants per pair:
//! - `bitcast_<target>(self) -> Target` — owned, zero-cost
//! - `bitcast_ref_<target>(&self) -> &Target` — pointer cast
//! - `bitcast_mut_<target>(&mut self) -> &mut Target` — pointer cast

use crate::simd_types::types::{ElementType, SimdType, SimdWidth};
use std::fmt::Write;

/// Bitcast pairs to generate. (source_elem, target_elem).
/// Full NxN is the goal; we allowlist to control sprawl.
const ALLOWED_BITCAST_PAIRS: &[(ElementType, ElementType)] = &[
    // Float ↔ same-size signed int
    (ElementType::F32, ElementType::I32),
    (ElementType::I32, ElementType::F32),
    (ElementType::F64, ElementType::I64),
    (ElementType::I64, ElementType::F64),
    // Float ↔ same-size unsigned int
    (ElementType::F32, ElementType::U32),
    (ElementType::U32, ElementType::F32),
    (ElementType::F64, ElementType::U64),
    (ElementType::U64, ElementType::F64),
    // Signed ↔ unsigned (same element size)
    (ElementType::I8, ElementType::U8),
    (ElementType::U8, ElementType::I8),
    (ElementType::I16, ElementType::U16),
    (ElementType::U16, ElementType::I16),
    (ElementType::I32, ElementType::U32),
    (ElementType::U32, ElementType::I32),
    (ElementType::I64, ElementType::U64),
    (ElementType::U64, ElementType::I64),
];

/// Get the allowed bitcast targets for a given source element type.
pub fn bitcast_targets(src: ElementType) -> Vec<ElementType> {
    ALLOWED_BITCAST_PAIRS
        .iter()
        .filter(|(s, _)| *s == src)
        .map(|(_, t)| *t)
        .collect()
}

/// Classify inner type for x86 cast intrinsic selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InnerKind {
    Ps, // __m128, __m256, __m512 (f32)
    Pd, // __m128d, __m256d, __m512d (f64)
    Si, // __m128i, __m256i, __m512i (all integers)
}

fn inner_kind(elem: ElementType) -> InnerKind {
    match elem {
        ElementType::F32 => InnerKind::Ps,
        ElementType::F64 => InnerKind::Pd,
        _ => InnerKind::Si,
    }
}

/// Generate bitcast methods for a single x86 SIMD type.
pub fn generate_x86_bitcasts(ty: &SimdType) -> String {
    let targets = bitcast_targets(ty.elem);
    if targets.is_empty() {
        return String::new();
    }

    let mut code = String::new();
    writeln!(
        code,
        "\n    // ========== Bitcast (reinterpret bits, zero-cost) ==========\n"
    )
    .unwrap();

    let src_kind = inner_kind(ty.elem);
    let prefix = ty.width.x86_prefix();
    let bits = ty.width.bits();

    for target_elem in &targets {
        let target_ty = SimdType::new(*target_elem, ty.width);
        let target_name = target_ty.name();
        let target_kind = inner_kind(*target_elem);

        // Owned variant
        let body = generate_x86_bitcast_body(src_kind, target_kind, prefix, bits);
        writeln!(
            code,
            "    /// Reinterpret bits as `{target_name}` (zero-cost)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn bitcast_{target_name}(self) -> {target_name} {{"
        )
        .unwrap();
        writeln!(code, "        {body}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // Ref variant
        writeln!(
            code,
            "    /// Reinterpret bits as `&{target_name}` (zero-cost pointer cast)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn bitcast_ref_{target_name}(&self) -> &{target_name} {{"
        )
        .unwrap();
        writeln!(
            code,
            "        unsafe {{ &*(self as *const Self as *const {target_name}) }}"
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        // Mut variant
        writeln!(
            code,
            "    /// Reinterpret bits as `&mut {target_name}` (zero-cost pointer cast)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn bitcast_mut_{target_name}(&mut self) -> &mut {target_name} {{"
        )
        .unwrap();
        writeln!(
            code,
            "        unsafe {{ &mut *(self as *mut Self as *mut {target_name}) }}"
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    code
}

/// Generate the body of an owned x86 bitcast.
fn generate_x86_bitcast_body(
    src_kind: InnerKind,
    dst_kind: InnerKind,
    prefix: &str,
    bits: usize,
) -> String {
    match (src_kind, dst_kind) {
        // Same inner kind → just wrap (transmute handles newtype differences)
        (InnerKind::Si, InnerKind::Si)
        | (InnerKind::Ps, InnerKind::Ps)
        | (InnerKind::Pd, InnerKind::Pd) => "unsafe { core::mem::transmute(self) }".to_string(),
        // Ps → Si
        (InnerKind::Ps, InnerKind::Si) => {
            format!("unsafe {{ core::mem::transmute({prefix}_castps_si{bits}(self.0)) }}")
        }
        // Si → Ps
        (InnerKind::Si, InnerKind::Ps) => {
            format!("unsafe {{ core::mem::transmute({prefix}_castsi{bits}_ps(self.0)) }}")
        }
        // Pd → Si
        (InnerKind::Pd, InnerKind::Si) => {
            format!("unsafe {{ core::mem::transmute({prefix}_castpd_si{bits}(self.0)) }}")
        }
        // Si → Pd
        (InnerKind::Si, InnerKind::Pd) => {
            format!("unsafe {{ core::mem::transmute({prefix}_castsi{bits}_pd(self.0)) }}")
        }
        // Ps → Pd (not currently allowlisted, but infrastructure is here)
        (InnerKind::Ps, InnerKind::Pd) => {
            format!("unsafe {{ core::mem::transmute({prefix}_castps_pd(self.0)) }}")
        }
        // Pd → Ps
        (InnerKind::Pd, InnerKind::Ps) => {
            format!("unsafe {{ core::mem::transmute({prefix}_castpd_ps(self.0)) }}")
        }
    }
}

/// Generate bitcast methods for a single ARM NEON type.
pub fn generate_arm_bitcasts(ty: &SimdType) -> String {
    let targets = bitcast_targets(ty.elem);
    if targets.is_empty() {
        return String::new();
    }

    let mut code = String::new();
    writeln!(
        code,
        "\n    // ========== Bitcast (reinterpret bits, zero-cost) ==========\n"
    )
    .unwrap();

    let src_suffix = arm_type_suffix(ty.elem);

    for target_elem in &targets {
        let target_ty = SimdType::new(*target_elem, ty.width);
        let target_name = target_ty.name();
        let dst_suffix = arm_type_suffix(*target_elem);

        // Owned variant — use vreinterpretq intrinsic
        let needs_intrinsic = ty.elem != *target_elem;
        let body = if needs_intrinsic && inner_kind(ty.elem) != InnerKind::Si
            || inner_kind(*target_elem) != InnerKind::Si
            || ty.elem.size_bytes() != target_elem.size_bytes()
        {
            // Different NEON inner types → use vreinterpretq
            format!("{target_name}(unsafe {{ vreinterpretq_{dst_suffix}_{src_suffix}(self.0) }})")
        } else if needs_intrinsic {
            // Both are integer types with same element size — inner type might still differ
            // (e.g., int32x4_t vs uint32x4_t)
            format!("{target_name}(unsafe {{ vreinterpretq_{dst_suffix}_{src_suffix}(self.0) }})")
        } else {
            // Same element type (shouldn't happen with allowlist, but handle it)
            "self".to_string()
        };

        writeln!(
            code,
            "    /// Reinterpret bits as `{target_name}` (zero-cost)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn bitcast_{target_name}(self) -> {target_name} {{"
        )
        .unwrap();
        writeln!(code, "        {body}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // Ref variant
        writeln!(
            code,
            "    /// Reinterpret bits as `&{target_name}` (zero-cost pointer cast)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn bitcast_ref_{target_name}(&self) -> &{target_name} {{"
        )
        .unwrap();
        writeln!(
            code,
            "        unsafe {{ &*(self as *const Self as *const {target_name}) }}"
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        // Mut variant
        writeln!(
            code,
            "    /// Reinterpret bits as `&mut {target_name}` (zero-cost pointer cast)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn bitcast_mut_{target_name}(&mut self) -> &mut {target_name} {{"
        )
        .unwrap();
        writeln!(
            code,
            "        unsafe {{ &mut *(self as *mut Self as *mut {target_name}) }}"
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    code
}

/// Generate bitcast methods for a single WASM SIMD type.
pub fn generate_wasm_bitcasts(ty: &SimdType) -> String {
    let targets = bitcast_targets(ty.elem);
    if targets.is_empty() {
        return String::new();
    }

    let mut code = String::new();
    writeln!(
        code,
        "\n    // ========== Bitcast (reinterpret bits, zero-cost) ==========\n"
    )
    .unwrap();

    for target_elem in &targets {
        let target_ty = SimdType::new(*target_elem, ty.width);
        let target_name = target_ty.name();

        // All WASM types wrap v128 — just rewrap
        writeln!(
            code,
            "    /// Reinterpret bits as `{target_name}` (zero-cost)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn bitcast_{target_name}(self) -> {target_name} {{"
        )
        .unwrap();
        writeln!(code, "        {target_name}(self.0)").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // Ref variant
        writeln!(
            code,
            "    /// Reinterpret bits as `&{target_name}` (zero-cost pointer cast)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn bitcast_ref_{target_name}(&self) -> &{target_name} {{"
        )
        .unwrap();
        writeln!(
            code,
            "        unsafe {{ &*(self as *const Self as *const {target_name}) }}"
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        // Mut variant
        writeln!(
            code,
            "    /// Reinterpret bits as `&mut {target_name}` (zero-cost pointer cast)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn bitcast_mut_{target_name}(&mut self) -> &mut {target_name} {{"
        )
        .unwrap();
        writeln!(
            code,
            "        unsafe {{ &mut *(self as *mut Self as *mut {target_name}) }}"
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    code
}

/// Generate bitcast methods for polyfill types (delegates to lo/hi).
pub fn generate_polyfill_bitcasts(
    src_elem: ElementType,
    width: SimdWidth,
    half_width: SimdWidth,
) -> String {
    let targets = bitcast_targets(src_elem);
    if targets.is_empty() {
        return String::new();
    }

    let mut code = String::new();
    writeln!(
        code,
        "\n    // ========== Bitcast (reinterpret bits, zero-cost) ==========\n"
    )
    .unwrap();

    for target_elem in &targets {
        let target_ty = SimdType::new(*target_elem, width);
        let target_name = target_ty.name();

        let half_target = SimdType::new(*target_elem, half_width);
        let half_target_name = half_target.name();

        // Owned variant — delegate to lo/hi bitcasts
        writeln!(
            code,
            "    /// Reinterpret bits as `{target_name}` (zero-cost)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn bitcast_{target_name}(self) -> {target_name} {{"
        )
        .unwrap();
        writeln!(code, "        {target_name} {{").unwrap();
        writeln!(
            code,
            "            lo: self.lo.bitcast_{half_target_name}(),"
        )
        .unwrap();
        writeln!(
            code,
            "            hi: self.hi.bitcast_{half_target_name}(),"
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // Ref variant
        writeln!(
            code,
            "    /// Reinterpret bits as `&{target_name}` (zero-cost pointer cast)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn bitcast_ref_{target_name}(&self) -> &{target_name} {{"
        )
        .unwrap();
        writeln!(
            code,
            "        unsafe {{ &*(self as *const Self as *const {target_name}) }}"
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        // Mut variant
        writeln!(
            code,
            "    /// Reinterpret bits as `&mut {target_name}` (zero-cost pointer cast)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn bitcast_mut_{target_name}(&mut self) -> &mut {target_name} {{"
        )
        .unwrap();
        writeln!(
            code,
            "        unsafe {{ &mut *(self as *mut Self as *mut {target_name}) }}"
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    code
}

/// ARM NEON type suffix for vreinterpretq intrinsics.
fn arm_type_suffix(elem: ElementType) -> &'static str {
    match elem {
        ElementType::F32 => "f32",
        ElementType::F64 => "f64",
        ElementType::I8 => "s8",
        ElementType::U8 => "u8",
        ElementType::I16 => "s16",
        ElementType::U16 => "u16",
        ElementType::I32 => "s32",
        ElementType::U32 => "u32",
        ElementType::I64 => "s64",
        ElementType::U64 => "u64",
    }
}

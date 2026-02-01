//! Bitcast code generation for SIMD types.
//!
//! Generates zero-cost reinterpret-cast methods between same-width SIMD types.

use crate::simd_types::types::{ElementType, SimdType, SimdWidth};
use indoc::formatdoc;

/// Bitcast pairs to generate. (source_elem, target_elem).
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

pub fn bitcast_targets(src: ElementType) -> Vec<ElementType> {
    ALLOWED_BITCAST_PAIRS
        .iter()
        .filter(|(s, _)| *s == src)
        .map(|(_, t)| *t)
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InnerKind {
    Ps,
    Pd,
    Si,
}

fn inner_kind(elem: ElementType) -> InnerKind {
    match elem {
        ElementType::F32 => InnerKind::Ps,
        ElementType::F64 => InnerKind::Pd,
        _ => InnerKind::Si,
    }
}

/// Generate all three bitcast variants (owned, ref, mut)
fn gen_bitcast_methods(target: &str, body: &str) -> String {
    formatdoc! {"
        /// Reinterpret bits as `{target}` (zero-cost).
        #[inline(always)]
        pub fn bitcast_{target}(self) -> {target} {{
        {body}
        }}

        /// Reinterpret bits as `&{target}` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_ref_{target}(&self) -> &{target} {{
        unsafe {{ &*(self as *const Self as *const {target}) }}
        }}

        /// Reinterpret bits as `&mut {target}` (zero-cost pointer cast).
        #[inline(always)]
        pub fn bitcast_mut_{target}(&mut self) -> &mut {target} {{
        unsafe {{ &mut *(self as *mut Self as *mut {target}) }}
        }}
    "}
}

pub fn generate_x86_bitcasts(ty: &SimdType) -> String {
    let targets = bitcast_targets(ty.elem);
    if targets.is_empty() {
        return String::new();
    }

    let src_kind = inner_kind(ty.elem);
    let prefix = ty.width.x86_prefix();
    let bits = ty.width.bits();

    let mut code = String::from("// ========== Bitcast ==========\n");
    for target_elem in &targets {
        let target = SimdType::new(*target_elem, ty.width).name();
        let body = x86_bitcast_body(src_kind, inner_kind(*target_elem), prefix, bits);
        code.push_str(&gen_bitcast_methods(&target, &body));
    }
    code
}

fn x86_bitcast_body(src: InnerKind, dst: InnerKind, prefix: &str, bits: usize) -> String {
    match (src, dst) {
        (InnerKind::Si, InnerKind::Si)
        | (InnerKind::Ps, InnerKind::Ps)
        | (InnerKind::Pd, InnerKind::Pd) => "unsafe { core::mem::transmute(self) }".into(),
        (InnerKind::Ps, InnerKind::Si) => {
            format!("unsafe {{ core::mem::transmute({prefix}_castps_si{bits}(self.0)) }}")
        }
        (InnerKind::Si, InnerKind::Ps) => {
            format!("unsafe {{ core::mem::transmute({prefix}_castsi{bits}_ps(self.0)) }}")
        }
        (InnerKind::Pd, InnerKind::Si) => {
            format!("unsafe {{ core::mem::transmute({prefix}_castpd_si{bits}(self.0)) }}")
        }
        (InnerKind::Si, InnerKind::Pd) => {
            format!("unsafe {{ core::mem::transmute({prefix}_castsi{bits}_pd(self.0)) }}")
        }
        (InnerKind::Ps, InnerKind::Pd) => {
            format!("unsafe {{ core::mem::transmute({prefix}_castps_pd(self.0)) }}")
        }
        (InnerKind::Pd, InnerKind::Ps) => {
            format!("unsafe {{ core::mem::transmute({prefix}_castpd_ps(self.0)) }}")
        }
    }
}

pub fn generate_arm_bitcasts(ty: &SimdType) -> String {
    let targets = bitcast_targets(ty.elem);
    if targets.is_empty() {
        return String::new();
    }

    let src_suffix = arm_suffix(ty.elem);
    let mut code = String::from("// ========== Bitcast ==========\n");

    for target_elem in &targets {
        let target = SimdType::new(*target_elem, ty.width).name();
        let dst_suffix = arm_suffix(*target_elem);
        let body =
            format!("{target}(unsafe {{ vreinterpretq_{dst_suffix}_{src_suffix}(self.0) }})");
        code.push_str(&gen_bitcast_methods(&target, &body));
    }
    code
}

fn arm_suffix(elem: ElementType) -> &'static str {
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

pub fn generate_wasm_bitcasts(ty: &SimdType) -> String {
    let targets = bitcast_targets(ty.elem);
    if targets.is_empty() {
        return String::new();
    }

    let mut code = String::from("// ========== Bitcast ==========\n");
    for target_elem in &targets {
        let target = SimdType::new(*target_elem, ty.width).name();
        let body = format!("{target}(self.0)");
        code.push_str(&gen_bitcast_methods(&target, &body));
    }
    code
}

pub fn generate_polyfill_bitcasts(
    src_elem: ElementType,
    width: SimdWidth,
    half_width: SimdWidth,
) -> String {
    let targets = bitcast_targets(src_elem);
    if targets.is_empty() {
        return String::new();
    }

    let mut code = String::from("// ========== Bitcast ==========\n");
    for target_elem in &targets {
        let target = SimdType::new(*target_elem, width).name();
        let half = SimdType::new(*target_elem, half_width).name();

        code.push_str(&formatdoc! {"
            /// Reinterpret bits as `{target}` (zero-cost).
            #[inline(always)]
            pub fn bitcast_{target}(self) -> {target} {{
            {target} {{
            lo: self.lo.bitcast_{half}(),
            hi: self.hi.bitcast_{half}(),
            }}
            }}

            /// Reinterpret bits as `&{target}` (zero-cost pointer cast).
            #[inline(always)]
            pub fn bitcast_ref_{target}(&self) -> &{target} {{
            unsafe {{ &*(self as *const Self as *const {target}) }}
            }}

            /// Reinterpret bits as `&mut {target}` (zero-cost pointer cast).
            #[inline(always)]
            pub fn bitcast_mut_{target}(&mut self) -> &mut {target} {{
            unsafe {{ &mut *(self as *mut Self as *mut {target}) }}
            }}
        "});
    }
    code
}

//! Asm-inspection helper: proves the backend `__tf` nested-target_feature
//! fns fully inline into a matching-feature kernel (no residual calls).
//!
//! x86_64-only by design (it inspects AVX2 codegen). Build release and
//! disassemble the `kernel_v3` symbol:
//! ```bash
//! cargo build -p magetypes --release --example tf_inline_check
//! objdump -d target/release/examples/tf_inline_check | \
//!     awk '/<.*kernel_v3.*>:/,/^$/' | grep call
//! ```
//! Expected: no `call` into `__tf` (only straight-line vector code).

#[cfg(target_arch = "x86_64")]
mod demo {
    use archmage::{SimdToken, X64V3Token, arcane};
    use magetypes::simd::f32x8;

    #[arcane]
    #[inline(never)]
    pub fn kernel_v3(token: X64V3Token, data: &[[f32; 8]], out: &mut [f32]) {
        for (chunk, o) in data.iter().zip(out.iter_mut()) {
            let v = f32x8::load(token, chunk);
            let d = (v * v + v).recip();
            *o = d.reduce_add();
        }
    }

    pub fn run() {
        let data = vec![[1.5f32; 8]; 1024];
        let mut out = vec![0.0f32; 1024];
        if let Some(token) = X64V3Token::summon() {
            kernel_v3(token, &data, &mut out);
        }
        println!("{}", out.iter().sum::<f32>());
    }
}

fn main() {
    #[cfg(target_arch = "x86_64")]
    demo::run();
    #[cfg(not(target_arch = "x86_64"))]
    println!("tf_inline_check is an x86_64-only asm-inspection example");
}

#![forbid(unsafe_code)]
#[cfg(target_arch = "x86_64")]
mod kernels {
    use artisanal_dispatch::{V3, bridge, tier};

    #[bridge(v3)]
    fn leaf_v3(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }

    #[tier(v3)]
    #[inline(never)]
    fn matching(a: f32, b: f32, c: f32) -> f32 {
        leaf_v3(a, b, c)
    }

    #[inline(never)]
    fn proof_boundary(t: V3, a: f32, b: f32, c: f32) -> f32 {
        leaf_v3_t(t, a, b, c)
    }

    #[bridge(v3)]
    #[inline(never)]
    fn enter_v3(a: f32, b: f32, c: f32) -> f32 {
        matching(a, b, c)
    }

    pub fn run() {
        let t = V3::summon().expect("run this example with std on a V3 CPU");
        let x = std::hint::black_box(1.25);
        println!("{} {}", proof_boundary(t, x, x, x), enter_v3_t(t, x, x, x));
    }
}
fn main() {
    #[cfg(target_arch = "x86_64")]
    kernels::run();
}

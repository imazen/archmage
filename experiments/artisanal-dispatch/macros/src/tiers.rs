// This is the whole registry, including inheritance. Detection and codegen use
// the same fully expanded lists. Names match Archmage c7897905, not just headline
// ISA names (e.g. v4 also includes AES/PCLMULQDQ).
#[derive(Clone, Copy)]
pub struct Tier {
    pub name: &'static str,
    pub proof: &'static str,
    pub arch: &'static str,
    pub parent: &'static str,
    pub extra: &'static str,
}
const fn tier(
    name: &'static str,
    proof: &'static str,
    arch: &'static str,
    parent: &'static str,
    extra: &'static str,
) -> Tier {
    Tier {
        name,
        proof,
        arch,
        parent,
        extra,
    }
}
pub const TIERS: &[Tier] = &[
    tier("scalar", "Scalar", "", "", ""),
    tier("v1", "V1", "x86_64", "", "sse,sse2"),
    tier(
        "v2",
        "V2",
        "x86_64",
        "v1",
        "sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b",
    ),
    tier(
        "v3",
        "V3",
        "x86_64",
        "v2",
        "avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe",
    ),
    tier(
        "v4",
        "V4",
        "x86_64",
        "v3",
        "pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl",
    ),
    tier(
        "v4x",
        "V4x",
        "x86_64",
        "v4",
        "avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes",
    ),
    tier("neon", "Neon", "aarch64", "", "neon"),
    tier(
        "arm_v2",
        "ArmV2",
        "aarch64",
        "neon",
        "crc,rdm,dotprod,fp16,aes,sha2",
    ),
    tier(
        "arm_v3",
        "ArmV3",
        "aarch64",
        "arm_v2",
        "fhm,fcma,sha3,i8mm,bf16",
    ),
    tier("wasm128", "Wasm128", "wasm32", "", "simd128"),
    tier(
        "wasm128_relaxed",
        "Wasm128Relaxed",
        "wasm32",
        "wasm128",
        "relaxed-simd",
    ),
];
impl Tier {
    pub fn features(self) -> Vec<&'static str> {
        let mut features = TIERS
            .iter()
            .find(|t| t.name == self.parent)
            .map_or_else(Vec::new, |t| t.features());
        features.extend(self.extra.split(',').filter(|s| !s.is_empty()));
        features
    }
}

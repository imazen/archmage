# NeonToken (Arm64) — NEON

Proof that NEON is available.

**Architecture:** aarch64 | **Features:** neon
**Total intrinsics:** 3302 (2756 safe, 546 unsafe, 3266 stable, 36 unstable/unknown)

## Usage

```rust
use archmage::{NeonToken, SimdToken};

if let Some(token) = NeonToken::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: NeonToken, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(4) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: NeonToken, chunk: &mut [f32; 4]) {
    let v = vld1q_f32(chunk);  // safe!
    let doubled = vaddq_f32(v, v);  // value intrinsic (safe inside #[rite])
    vst1q_f32(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```



## All Intrinsics

### Stable, Safe (2744 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `vaba_s16` | Absolute difference and accumulate (64-bit) | "vaba.s16" | — |
| `vaba_s32` | Absolute difference and accumulate (64-bit) | "vaba.s32" | — |
| `vaba_s8` | Absolute difference and accumulate (64-bit) | "vaba.s8" | — |
| `vaba_u16` | Absolute difference and accumulate (64-bit) | "vaba.u16" | — |
| `vaba_u32` | Absolute difference and accumulate (64-bit) | "vaba.u32" | — |
| `vaba_u8` | Absolute difference and accumulate (64-bit) | "vaba.u8" | — |
| `vabal_high_s16` | Signed Absolute difference and Accumulate Long | sabal2 | — |
| `vabal_high_s32` | Signed Absolute difference and Accumulate Long | sabal2 | — |
| `vabal_high_s8` | Signed Absolute difference and Accumulate Long | sabal2 | — |
| `vabal_high_u16` | Unsigned Absolute difference and Accumulate Long | uabal2 | — |
| `vabal_high_u32` | Unsigned Absolute difference and Accumulate Long | uabal2 | — |
| `vabal_high_u8` | Unsigned Absolute difference and Accumulate Long | uabal2 | — |
| `vabal_s16` | Signed Absolute difference and Accumulate Long | "vabal.s16" | — |
| `vabal_s32` | Signed Absolute difference and Accumulate Long | "vabal.s32" | — |
| `vabal_s8` | Signed Absolute difference and Accumulate Long | "vabal.s8" | — |
| `vabal_u16` | Unsigned Absolute difference and Accumulate Long | "vabal.u16" | — |
| `vabal_u32` | Unsigned Absolute difference and Accumulate Long | "vabal.u32" | — |
| `vabal_u8` | Unsigned Absolute difference and Accumulate Long | "vabal.u8" | — |
| `vabaq_s16` | Absolute difference and accumulate (128-bit) | "vaba.s16" | — |
| `vabaq_s32` | Absolute difference and accumulate (128-bit) | "vaba.s32" | — |
| `vabaq_s8` | Absolute difference and accumulate (128-bit) | "vaba.s8" | — |
| `vabaq_u16` | Absolute difference and accumulate (128-bit) | "vaba.u16" | — |
| `vabaq_u32` | Absolute difference and accumulate (128-bit) | "vaba.u32" | — |
| `vabaq_u8` | Absolute difference and accumulate (128-bit) | "vaba.u8" | — |
| `vabd_f32` | Absolute difference between the arguments of Floating | "vabd.f32" | — |
| `vabd_f64` | Absolute difference between the arguments of Floating | fabd | — |
| `vabd_s16` | Absolute difference between the arguments | "vabd.s16" | — |
| `vabd_s32` | Absolute difference between the arguments | "vabd.s32" | — |
| `vabd_s8` | Absolute difference between the arguments | "vabd.s8" | — |
| `vabd_u16` | Absolute difference between the arguments | "vabd.u16" | — |
| `vabd_u32` | Absolute difference between the arguments | "vabd.u32" | — |
| `vabd_u8` | Absolute difference between the arguments | "vabd.u8" | — |
| `vabdd_f64` | Floating-point absolute difference | fabd | — |
| `vabdl_high_s16` | Signed Absolute difference Long | sabdl2 | — |
| `vabdl_high_s32` | Signed Absolute difference Long | sabdl2 | — |
| `vabdl_high_s8` | Signed Absolute difference Long | sabdl2 | — |
| `vabdl_high_u16` | Unsigned Absolute difference Long | uabdl2 | — |
| `vabdl_high_u32` | Unsigned Absolute difference Long | uabdl2 | — |
| `vabdl_high_u8` | Unsigned Absolute difference Long | uabdl2 | — |
| `vabdl_s16` | Signed Absolute difference Long | "vabdl.s16" | — |
| `vabdl_s32` | Signed Absolute difference Long | "vabdl.s32" | — |
| `vabdl_s8` | Signed Absolute difference Long | "vabdl.s8" | — |
| `vabdl_u16` | Unsigned Absolute difference Long | "vabdl.u16" | — |
| `vabdl_u32` | Unsigned Absolute difference Long | "vabdl.u32" | — |
| `vabdl_u8` | Unsigned Absolute difference Long | "vabdl.u8" | — |
| `vabdq_f32` | Absolute difference between the arguments of Floating | "vabd.f32" | — |
| `vabdq_f64` | Absolute difference between the arguments of Floating | fabd | — |
| `vabdq_s16` | Absolute difference between the arguments | "vabd.s16" | — |
| `vabdq_s32` | Absolute difference between the arguments | "vabd.s32" | — |
| `vabdq_s8` | Absolute difference between the arguments | "vabd.s8" | — |
| `vabdq_u16` | Absolute difference between the arguments | "vabd.u16" | — |
| `vabdq_u32` | Absolute difference between the arguments | "vabd.u32" | — |
| `vabdq_u8` | Absolute difference between the arguments | "vabd.u8" | — |
| `vabds_f32` | Floating-point absolute difference | fabd | — |
| `vabs_f32` | Floating-point absolute value | vabs | — |
| `vabs_f64` | Floating-point absolute value | fabs | — |
| `vabs_s16` | Absolute value (wrapping) | vabs | — |
| `vabs_s32` | Absolute value (wrapping) | vabs | — |
| `vabs_s64` | Absolute Value (wrapping) | abs | — |
| `vabs_s8` | Absolute value (wrapping) | vabs | — |
| `vabsd_s64` | Absolute Value (wrapping) | abs | — |
| `vabsq_f32` | Floating-point absolute value | vabs | — |
| `vabsq_f64` | Floating-point absolute value | fabs | — |
| `vabsq_s16` | Absolute value (wrapping) | vabs | — |
| `vabsq_s32` | Absolute value (wrapping) | vabs | — |
| `vabsq_s64` | Absolute Value (wrapping) | abs | — |
| `vabsq_s8` | Absolute value (wrapping) | vabs | — |
| `vadd_f32` | Vector add | vadd | — |
| `vadd_f64` | Vector add | fadd | — |
| `vadd_p16` | Bitwise exclusive OR | nop | — |
| `vadd_p64` | Bitwise exclusive OR | nop | — |
| `vadd_p8` | Bitwise exclusive OR | nop | — |
| `vadd_s16` | Vector add | vadd | — |
| `vadd_s32` | Vector add | vadd | — |
| `vadd_s64` | Vector add | add | — |
| `vadd_s8` | Vector add | vadd | — |
| `vadd_u16` | Vector add | vadd | — |
| `vadd_u32` | Vector add | vadd | — |
| `vadd_u64` | Vector add | add | — |
| `vadd_u8` | Vector add | vadd | — |
| `vaddd_s64` | Add | nop | — |
| `vaddd_u64` | Add | nop | — |
| `vaddhn_high_s16` | Add returning High Narrow (high half) | vaddhn | — |
| `vaddhn_high_s32` | Add returning High Narrow (high half) | vaddhn | — |
| `vaddhn_high_s64` | Add returning High Narrow (high half) | vaddhn | — |
| `vaddhn_high_u16` | Add returning High Narrow (high half) | vaddhn | — |
| `vaddhn_high_u32` | Add returning High Narrow (high half) | vaddhn | — |
| `vaddhn_high_u64` | Add returning High Narrow (high half) | vaddhn | — |
| `vaddhn_s16` | Add returning High Narrow | vaddhn | — |
| `vaddhn_s32` | Add returning High Narrow | vaddhn | — |
| `vaddhn_s64` | Add returning High Narrow | vaddhn | — |
| `vaddhn_u16` | Add returning High Narrow | vaddhn | — |
| `vaddhn_u32` | Add returning High Narrow | vaddhn | — |
| `vaddhn_u64` | Add returning High Narrow | vaddhn | — |
| `vaddl_high_s16` | Signed Add Long (vector, high half) | vaddl | — |
| `vaddl_high_s32` | Signed Add Long (vector, high half) | vaddl | — |
| `vaddl_high_s8` | Signed Add Long (vector, high half) | vaddl | — |
| `vaddl_high_u16` | Signed Add Long (vector, high half) | vaddl | — |
| `vaddl_high_u32` | Signed Add Long (vector, high half) | vaddl | — |
| `vaddl_high_u8` | Signed Add Long (vector, high half) | vaddl | — |
| `vaddl_s16` | Add Long (vector) | vaddl | — |
| `vaddl_s32` | Add Long (vector) | vaddl | — |
| `vaddl_s8` | Add Long (vector) | vaddl | — |
| `vaddl_u16` | Add Long (vector) | vaddl | — |
| `vaddl_u32` | Add Long (vector) | vaddl | — |
| `vaddl_u8` | Add Long (vector) | vaddl | — |
| `vaddlv_s16` | Signed Add Long across Vector | saddlv | — |
| `vaddlv_s32` | Signed Add Long across Vector | saddlp | — |
| `vaddlv_s8` | Signed Add Long across Vector | saddlv | — |
| `vaddlv_u16` | Unsigned Add Long across Vector | uaddlv | — |
| `vaddlv_u32` | Unsigned Add Long across Vector | uaddlp | — |
| `vaddlv_u8` | Unsigned Add Long across Vector | uaddlv | — |
| `vaddlvq_s16` | Signed Add Long across Vector | saddlv | — |
| `vaddlvq_s32` | Signed Add Long across Vector | saddlv | — |
| `vaddlvq_s8` | Signed Add Long across Vector | saddlv | — |
| `vaddlvq_u16` | Unsigned Add Long across Vector | uaddlv | — |
| `vaddlvq_u32` | Unsigned Add Long across Vector | uaddlv | — |
| `vaddlvq_u8` | Unsigned Add Long across Vector | uaddlv | — |
| `vaddq_f32` | Vector add | vadd | — |
| `vaddq_f64` | Vector add | fadd | — |
| `vaddq_p128` | Bitwise exclusive OR | nop | — |
| `vaddq_p16` | Bitwise exclusive OR | nop | — |
| `vaddq_p64` | Bitwise exclusive OR | nop | — |
| `vaddq_p8` | Bitwise exclusive OR | nop | — |
| `vaddq_s16` | Vector add | vadd | — |
| `vaddq_s32` | Vector add | vadd | — |
| `vaddq_s64` | Vector add | vadd | — |
| `vaddq_s8` | Vector add | vadd | — |
| `vaddq_u16` | Vector add | vadd | — |
| `vaddq_u32` | Vector add | vadd | — |
| `vaddq_u64` | Vector add | vadd | — |
| `vaddq_u8` | Vector add | vadd | — |
| `vaddv_f32` | Floating-point add across vector | faddp | — |
| `vaddv_s16` | Add across vector | addv | — |
| `vaddv_s32` | Add across vector | addp | — |
| `vaddv_s8` | Add across vector | addv | — |
| `vaddv_u16` | Add across vector | addv | — |
| `vaddv_u32` | Add across vector | addp | — |
| `vaddv_u8` | Add across vector | addv | — |
| `vaddvq_f32` | Floating-point add across vector | faddp | — |
| `vaddvq_f64` | Floating-point add across vector | faddp | — |
| `vaddvq_s16` | Add across vector | addv | — |
| `vaddvq_s32` | Add across vector | addv | — |
| `vaddvq_s64` | Add across vector | addp | — |
| `vaddvq_s8` | Add across vector | addv | — |
| `vaddvq_u16` | Add across vector | addv | — |
| `vaddvq_u32` | Add across vector | addv | — |
| `vaddvq_u64` | Add across vector | addp | — |
| `vaddvq_u8` | Add across vector | addv | — |
| `vaddw_high_s16` | Add Wide (high half) | vaddw | — |
| `vaddw_high_s32` | Add Wide (high half) | vaddw | — |
| `vaddw_high_s8` | Add Wide (high half) | vaddw | — |
| `vaddw_high_u16` | Add Wide (high half) | vaddw | — |
| `vaddw_high_u32` | Add Wide (high half) | vaddw | — |
| `vaddw_high_u8` | Add Wide (high half) | vaddw | — |
| `vaddw_s16` | Add Wide | vaddw | — |
| `vaddw_s32` | Add Wide | vaddw | — |
| `vaddw_s8` | Add Wide | vaddw | — |
| `vaddw_u16` | Add Wide | vaddw | — |
| `vaddw_u32` | Add Wide | vaddw | — |
| `vaddw_u8` | Add Wide | vaddw | — |
| `vand_s16` | Vector bitwise and | vand | — |
| `vand_s32` | Vector bitwise and | vand | — |
| `vand_s64` | Vector bitwise and | vand | — |
| `vand_s8` | Vector bitwise and | vand | — |
| `vand_u16` | Vector bitwise and | vand | — |
| `vand_u32` | Vector bitwise and | vand | — |
| `vand_u64` | Vector bitwise and | vand | — |
| `vand_u8` | Vector bitwise and | vand | — |
| `vandq_s16` | Vector bitwise and | vand | — |
| `vandq_s32` | Vector bitwise and | vand | — |
| `vandq_s64` | Vector bitwise and | vand | — |
| `vandq_s8` | Vector bitwise and | vand | — |
| `vandq_u16` | Vector bitwise and | vand | — |
| `vandq_u32` | Vector bitwise and | vand | — |
| `vandq_u64` | Vector bitwise and | vand | — |
| `vandq_u8` | Vector bitwise and | vand | — |
| `vbic_s16` | Vector bitwise bit clear | vbic | — |
| `vbic_s32` | Vector bitwise bit clear | vbic | — |
| `vbic_s64` | Vector bitwise bit clear | vbic | — |
| `vbic_s8` | Vector bitwise bit clear | vbic | — |
| `vbic_u16` | Vector bitwise bit clear | vbic | — |
| `vbic_u32` | Vector bitwise bit clear | vbic | — |
| `vbic_u64` | Vector bitwise bit clear | vbic | — |
| `vbic_u8` | Vector bitwise bit clear | vbic | — |
| `vbicq_s16` | Vector bitwise bit clear | vbic | — |
| `vbicq_s32` | Vector bitwise bit clear | vbic | — |
| `vbicq_s64` | Vector bitwise bit clear | vbic | — |
| `vbicq_s8` | Vector bitwise bit clear | vbic | — |
| `vbicq_u16` | Vector bitwise bit clear | vbic | — |
| `vbicq_u32` | Vector bitwise bit clear | vbic | — |
| `vbicq_u64` | Vector bitwise bit clear | vbic | — |
| `vbicq_u8` | Vector bitwise bit clear | vbic | — |
| `vbsl_f32` | Bitwise Select | vbsl | — |
| `vbsl_f64` | Bitwise Select instructions. This instruction sets each bit ... | bsl | — |
| `vbsl_p16` | Bitwise Select | vbsl | — |
| `vbsl_p64` | Bitwise Select | bsl | — |
| `vbsl_p8` | Bitwise Select | vbsl | — |
| `vbsl_s16` | Bitwise Select | vbsl | — |
| `vbsl_s32` | Bitwise Select | vbsl | — |
| `vbsl_s64` | Bitwise Select | vbsl | — |
| `vbsl_s8` | Bitwise Select | vbsl | — |
| `vbsl_u16` | Bitwise Select | vbsl | — |
| `vbsl_u32` | Bitwise Select | vbsl | — |
| `vbsl_u64` | Bitwise Select | vbsl | — |
| `vbsl_u8` | Bitwise Select | vbsl | — |
| `vbslq_f32` | Bitwise Select | vbsl | — |
| `vbslq_f64` | Bitwise Select. (128-bit) | bsl | — |
| `vbslq_p16` | Bitwise Select | vbsl | — |
| `vbslq_p64` | Bitwise Select. (128-bit) | bsl | — |
| `vbslq_p8` | Bitwise Select | vbsl | — |
| `vbslq_s16` | Bitwise Select | vbsl | — |
| `vbslq_s32` | Bitwise Select | vbsl | — |
| `vbslq_s64` | Bitwise Select | vbsl | — |
| `vbslq_s8` | Bitwise Select | vbsl | — |
| `vbslq_u16` | Bitwise Select | vbsl | — |
| `vbslq_u32` | Bitwise Select | vbsl | — |
| `vbslq_u64` | Bitwise Select | vbsl | — |
| `vbslq_u8` | Bitwise Select | vbsl | — |
| `vcage_f32` | Floating-point absolute compare greater than or equal | "vacge.f32" | — |
| `vcage_f64` | Floating-point absolute compare greater than or equal | facge | — |
| `vcaged_f64` | Floating-point absolute compare greater than or equal | facge | — |
| `vcageq_f32` | Floating-point absolute compare greater than or equal | "vacge.f32" | — |
| `vcageq_f64` | Floating-point absolute compare greater than or equal | facge | — |
| `vcages_f32` | Floating-point absolute compare greater than or equal | facge | — |
| `vcagt_f32` | Floating-point absolute compare greater than | "vacgt.f32" | — |
| `vcagt_f64` | Floating-point absolute compare greater than | facgt | — |
| `vcagtd_f64` | Floating-point absolute compare greater than | facgt | — |
| `vcagtq_f32` | Floating-point absolute compare greater than | "vacgt.f32" | — |
| `vcagtq_f64` | Floating-point absolute compare greater than | facgt | — |
| `vcagts_f32` | Floating-point absolute compare greater than | facgt | — |
| `vcale_f32` | Floating-point absolute compare less than or equal | "vacge.f32" | — |
| `vcale_f64` | Floating-point absolute compare less than or equal | facge | — |
| `vcaled_f64` | Floating-point absolute compare less than or equal | facge | — |
| `vcaleq_f32` | Floating-point absolute compare less than or equal | "vacge.f32" | — |
| `vcaleq_f64` | Floating-point absolute compare less than or equal | facge | — |
| `vcales_f32` | Floating-point absolute compare less than or equal | facge | — |
| `vcalt_f32` | Floating-point absolute compare less than | "vacgt.f32" | — |
| `vcalt_f64` | Floating-point absolute compare less than | facgt | — |
| `vcaltd_f64` | Floating-point absolute compare less than | facgt | — |
| `vcaltq_f32` | Floating-point absolute compare less than | "vacgt.f32" | — |
| `vcaltq_f64` | Floating-point absolute compare less than | facgt | — |
| `vcalts_f32` | Floating-point absolute compare less than | facgt | — |
| `vceq_f32` | Floating-point compare equal | "vceq.f32" | — |
| `vceq_f64` | Floating-point compare equal | fcmeq | — |
| `vceq_p64` | Compare bitwise Equal (vector) | cmeq | — |
| `vceq_p8` | Compare bitwise Equal (vector) | "vceq.i8" | — |
| `vceq_s16` | Compare bitwise Equal (vector) | "vceq.i16" | — |
| `vceq_s32` | Compare bitwise Equal (vector) | "vceq.i32" | — |
| `vceq_s64` | Compare bitwise Equal (vector) | cmeq | — |
| `vceq_s8` | Compare bitwise Equal (vector) | "vceq.i8" | — |
| `vceq_u16` | Compare bitwise Equal (vector) | "vceq.i16" | — |
| `vceq_u32` | Compare bitwise Equal (vector) | "vceq.i32" | — |
| `vceq_u64` | Compare bitwise Equal (vector) | cmeq | — |
| `vceq_u8` | Compare bitwise Equal (vector) | "vceq.i8" | — |
| `vceqd_f64` | Floating-point compare equal | fcmp | — |
| `vceqd_s64` | Compare bitwise equal | cmp | — |
| `vceqd_u64` | Compare bitwise equal | cmp | — |
| `vceqq_f32` | Floating-point compare equal | "vceq.f32" | — |
| `vceqq_f64` | Floating-point compare equal | fcmeq | — |
| `vceqq_p64` | Compare bitwise Equal (vector) | cmeq | — |
| `vceqq_p8` | Compare bitwise Equal (vector) | "vceq.i8" | — |
| `vceqq_s16` | Compare bitwise Equal (vector) | "vceq.i16" | — |
| `vceqq_s32` | Compare bitwise Equal (vector) | "vceq.i32" | — |
| `vceqq_s64` | Compare bitwise Equal (vector) | cmeq | — |
| `vceqq_s8` | Compare bitwise Equal (vector) | "vceq.i8" | — |
| `vceqq_u16` | Compare bitwise Equal (vector) | "vceq.i16" | — |
| `vceqq_u32` | Compare bitwise Equal (vector) | "vceq.i32" | — |
| `vceqq_u64` | Compare bitwise Equal (vector) | cmeq | — |
| `vceqq_u8` | Compare bitwise Equal (vector) | "vceq.i8" | — |
| `vceqs_f32` | Floating-point compare equal | fcmp | — |
| `vceqz_f32` | Floating-point compare bitwise equal to zero | fcmeq | — |
| `vceqz_f64` | Floating-point compare bitwise equal to zero | fcmeq | — |
| `vceqz_p64` | Signed compare bitwise equal to zero | cmeq | — |
| `vceqz_p8` | Signed compare bitwise equal to zero | cmeq | — |
| `vceqz_s16` | Signed compare bitwise equal to zero | cmeq | — |
| `vceqz_s32` | Signed compare bitwise equal to zero | cmeq | — |
| `vceqz_s64` | Signed compare bitwise equal to zero | cmeq | — |
| `vceqz_s8` | Signed compare bitwise equal to zero | cmeq | — |
| `vceqz_u16` | Unsigned compare bitwise equal to zero | cmeq | — |
| `vceqz_u32` | Unsigned compare bitwise equal to zero | cmeq | — |
| `vceqz_u64` | Unsigned compare bitwise equal to zero | cmeq | — |
| `vceqz_u8` | Unsigned compare bitwise equal to zero | cmeq | — |
| `vceqzd_f64` | Floating-point compare bitwise equal to zero | fcmp | — |
| `vceqzd_s64` | Compare bitwise equal to zero | cmp | — |
| `vceqzd_u64` | Compare bitwise equal to zero | cmp | — |
| `vceqzq_f32` | Floating-point compare bitwise equal to zero | fcmeq | — |
| `vceqzq_f64` | Floating-point compare bitwise equal to zero | fcmeq | — |
| `vceqzq_p64` | Signed compare bitwise equal to zero | cmeq | — |
| `vceqzq_p8` | Signed compare bitwise equal to zero | cmeq | — |
| `vceqzq_s16` | Signed compare bitwise equal to zero | cmeq | — |
| `vceqzq_s32` | Signed compare bitwise equal to zero | cmeq | — |
| `vceqzq_s64` | Signed compare bitwise equal to zero | cmeq | — |
| `vceqzq_s8` | Signed compare bitwise equal to zero | cmeq | — |
| `vceqzq_u16` | Unsigned compare bitwise equal to zero | cmeq | — |
| `vceqzq_u32` | Unsigned compare bitwise equal to zero | cmeq | — |
| `vceqzq_u64` | Unsigned compare bitwise equal to zero | cmeq | — |
| `vceqzq_u8` | Unsigned compare bitwise equal to zero | cmeq | — |
| `vceqzs_f32` | Floating-point compare bitwise equal to zero | fcmp | — |
| `vcge_f32` | Floating-point compare greater than or equal | "vcge.f32" | — |
| `vcge_f64` | Floating-point compare greater than or equal | fcmge | — |
| `vcge_s16` | Compare signed greater than or equal | "vcge.s16" | — |
| `vcge_s32` | Compare signed greater than or equal | "vcge.s32" | — |
| `vcge_s64` | Compare signed greater than or equal | cmge | — |
| `vcge_s8` | Compare signed greater than or equal | "vcge.s8" | — |
| `vcge_u16` | Compare unsigned greater than or equal | "vcge.u16" | — |
| `vcge_u32` | Compare unsigned greater than or equal | "vcge.u32" | — |
| `vcge_u64` | Compare unsigned greater than or equal | cmhs | — |
| `vcge_u8` | Compare unsigned greater than or equal | "vcge.u8" | — |
| `vcged_f64` | Floating-point compare greater than or equal | fcmp | — |
| `vcged_s64` | Compare greater than or equal | cmp | — |
| `vcged_u64` | Compare greater than or equal | cmp | — |
| `vcgeq_f32` | Floating-point compare greater than or equal | "vcge.f32" | — |
| `vcgeq_f64` | Floating-point compare greater than or equal | fcmge | — |
| `vcgeq_s16` | Compare signed greater than or equal | "vcge.s16" | — |
| `vcgeq_s32` | Compare signed greater than or equal | "vcge.s32" | — |
| `vcgeq_s64` | Compare signed greater than or equal | cmge | — |
| `vcgeq_s8` | Compare signed greater than or equal | "vcge.s8" | — |
| `vcgeq_u16` | Compare unsigned greater than or equal | "vcge.u16" | — |
| `vcgeq_u32` | Compare unsigned greater than or equal | "vcge.u32" | — |
| `vcgeq_u64` | Compare unsigned greater than or equal | cmhs | — |
| `vcgeq_u8` | Compare unsigned greater than or equal | "vcge.u8" | — |
| `vcges_f32` | Floating-point compare greater than or equal | fcmp | — |
| `vcgez_f32` | Floating-point compare greater than or equal to zero | fcmge | — |
| `vcgez_f64` | Floating-point compare greater than or equal to zero | fcmge | — |
| `vcgez_s16` | Compare signed greater than or equal to zero | cmge | — |
| `vcgez_s32` | Compare signed greater than or equal to zero | cmge | — |
| `vcgez_s64` | Compare signed greater than or equal to zero | cmge | — |
| `vcgez_s8` | Compare signed greater than or equal to zero | cmge | — |
| `vcgezd_f64` | Floating-point compare greater than or equal to zero | fcmp | — |
| `vcgezd_s64` | Compare signed greater than or equal to zero | nop | — |
| `vcgezq_f32` | Floating-point compare greater than or equal to zero | fcmge | — |
| `vcgezq_f64` | Floating-point compare greater than or equal to zero | fcmge | — |
| `vcgezq_s16` | Compare signed greater than or equal to zero | cmge | — |
| `vcgezq_s32` | Compare signed greater than or equal to zero | cmge | — |
| `vcgezq_s64` | Compare signed greater than or equal to zero | cmge | — |
| `vcgezq_s8` | Compare signed greater than or equal to zero | cmge | — |
| `vcgezs_f32` | Floating-point compare greater than or equal to zero | fcmp | — |
| `vcgt_f32` | Floating-point compare greater than | "vcgt.f32" | — |
| `vcgt_f64` | Floating-point compare greater than | fcmgt | — |
| `vcgt_s16` | Compare signed greater than | "vcgt.s16" | — |
| `vcgt_s32` | Compare signed greater than | "vcgt.s32" | — |
| `vcgt_s64` | Compare signed greater than | cmgt | — |
| `vcgt_s8` | Compare signed greater than | "vcgt.s8" | — |
| `vcgt_u16` | Compare unsigned greater than | "vcgt.u16" | — |
| `vcgt_u32` | Compare unsigned greater than | "vcgt.u32" | — |
| `vcgt_u64` | Compare unsigned greater than | cmhi | — |
| `vcgt_u8` | Compare unsigned greater than | "vcgt.u8" | — |
| `vcgtd_f64` | Floating-point compare greater than | fcmp | — |
| `vcgtd_s64` | Compare greater than | cmp | — |
| `vcgtd_u64` | Compare greater than | cmp | — |
| `vcgtq_f32` | Floating-point compare greater than | "vcgt.f32" | — |
| `vcgtq_f64` | Floating-point compare greater than | fcmgt | — |
| `vcgtq_s16` | Compare signed greater than | "vcgt.s16" | — |
| `vcgtq_s32` | Compare signed greater than | "vcgt.s32" | — |
| `vcgtq_s64` | Compare signed greater than | cmgt | — |
| `vcgtq_s8` | Compare signed greater than | "vcgt.s8" | — |
| `vcgtq_u16` | Compare unsigned greater than | "vcgt.u16" | — |
| `vcgtq_u32` | Compare unsigned greater than | "vcgt.u32" | — |
| `vcgtq_u64` | Compare unsigned greater than | cmhi | — |
| `vcgtq_u8` | Compare unsigned greater than | "vcgt.u8" | — |
| `vcgts_f32` | Floating-point compare greater than | fcmp | — |
| `vcgtz_f32` | Floating-point compare greater than zero | fcmgt | — |
| `vcgtz_f64` | Floating-point compare greater than zero | fcmgt | — |
| `vcgtz_s16` | Compare signed greater than zero | cmgt | — |
| `vcgtz_s32` | Compare signed greater than zero | cmgt | — |
| `vcgtz_s64` | Compare signed greater than zero | cmgt | — |
| `vcgtz_s8` | Compare signed greater than zero | cmgt | — |
| `vcgtzd_f64` | Floating-point compare greater than zero | fcmp | — |
| `vcgtzd_s64` | Compare signed greater than zero | cmp | — |
| `vcgtzq_f32` | Floating-point compare greater than zero | fcmgt | — |
| `vcgtzq_f64` | Floating-point compare greater than zero | fcmgt | — |
| `vcgtzq_s16` | Compare signed greater than zero | cmgt | — |
| `vcgtzq_s32` | Compare signed greater than zero | cmgt | — |
| `vcgtzq_s64` | Compare signed greater than zero | cmgt | — |
| `vcgtzq_s8` | Compare signed greater than zero | cmgt | — |
| `vcgtzs_f32` | Floating-point compare greater than zero | fcmp | — |
| `vcle_f32` | Floating-point compare less than or equal | "vcge.f32" | — |
| `vcle_f64` | Floating-point compare less than or equal | fcmge | — |
| `vcle_s16` | Compare signed less than or equal | "vcge.s16" | — |
| `vcle_s32` | Compare signed less than or equal | "vcge.s32" | — |
| `vcle_s64` | Compare signed less than or equal | cmge | — |
| `vcle_s8` | Compare signed less than or equal | "vcge.s8" | — |
| `vcle_u16` | Compare unsigned less than or equal | "vcge.u16" | — |
| `vcle_u32` | Compare unsigned less than or equal | "vcge.u32" | — |
| `vcle_u64` | Compare unsigned less than or equal | cmhs | — |
| `vcle_u8` | Compare unsigned less than or equal | "vcge.u8" | — |
| `vcled_f64` | Floating-point compare less than or equal | fcmp | — |
| `vcled_s64` | Compare less than or equal | cmp | — |
| `vcled_u64` | Compare less than or equal | cmp | — |
| `vcleq_f32` | Floating-point compare less than or equal | "vcge.f32" | — |
| `vcleq_f64` | Floating-point compare less than or equal | fcmge | — |
| `vcleq_s16` | Compare signed less than or equal | "vcge.s16" | — |
| `vcleq_s32` | Compare signed less than or equal | "vcge.s32" | — |
| `vcleq_s64` | Compare signed less than or equal | cmge | — |
| `vcleq_s8` | Compare signed less than or equal | "vcge.s8" | — |
| `vcleq_u16` | Compare unsigned less than or equal | "vcge.u16" | — |
| `vcleq_u32` | Compare unsigned less than or equal | "vcge.u32" | — |
| `vcleq_u64` | Compare unsigned less than or equal | cmhs | — |
| `vcleq_u8` | Compare unsigned less than or equal | "vcge.u8" | — |
| `vcles_f32` | Floating-point compare less than or equal | fcmp | — |
| `vclez_f32` | Floating-point compare less than or equal to zero | fcmle | — |
| `vclez_f64` | Floating-point compare less than or equal to zero | fcmle | — |
| `vclez_s16` | Compare signed less than or equal to zero | cmle | — |
| `vclez_s32` | Compare signed less than or equal to zero | cmle | — |
| `vclez_s64` | Compare signed less than or equal to zero | cmle | — |
| `vclez_s8` | Compare signed less than or equal to zero | cmle | — |
| `vclezd_f64` | Floating-point compare less than or equal to zero | fcmp | — |
| `vclezd_s64` | Compare less than or equal to zero | cmp | — |
| `vclezq_f32` | Floating-point compare less than or equal to zero | fcmle | — |
| `vclezq_f64` | Floating-point compare less than or equal to zero | fcmle | — |
| `vclezq_s16` | Compare signed less than or equal to zero | cmle | — |
| `vclezq_s32` | Compare signed less than or equal to zero | cmle | — |
| `vclezq_s64` | Compare signed less than or equal to zero | cmle | — |
| `vclezq_s8` | Compare signed less than or equal to zero | cmle | — |
| `vclezs_f32` | Floating-point compare less than or equal to zero | fcmp | — |
| `vcls_s16` | Count leading sign bits | "vcls.s16" | — |
| `vcls_s32` | Count leading sign bits | "vcls.s32" | — |
| `vcls_s8` | Count leading sign bits | "vcls.s8" | — |
| `vcls_u16` | Count leading sign bits | vcls | — |
| `vcls_u32` | Count leading sign bits | vcls | — |
| `vcls_u8` | Count leading sign bits | vcls | — |
| `vclsq_s16` | Count leading sign bits | "vcls.s16" | — |
| `vclsq_s32` | Count leading sign bits | "vcls.s32" | — |
| `vclsq_s8` | Count leading sign bits | "vcls.s8" | — |
| `vclsq_u16` | Count leading sign bits | vcls | — |
| `vclsq_u32` | Count leading sign bits | vcls | — |
| `vclsq_u8` | Count leading sign bits | vcls | — |
| `vclt_f32` | Floating-point compare less than | "vcgt.f32" | — |
| `vclt_f64` | Floating-point compare less than | fcmgt | — |
| `vclt_s16` | Compare signed less than | "vcgt.s16" | — |
| `vclt_s32` | Compare signed less than | "vcgt.s32" | — |
| `vclt_s64` | Compare signed less than | cmgt | — |
| `vclt_s8` | Compare signed less than | "vcgt.s8" | — |
| `vclt_u16` | Compare unsigned less than | "vcgt.u16" | — |
| `vclt_u32` | Compare unsigned less than | "vcgt.u32" | — |
| `vclt_u64` | Compare unsigned less than | cmhi | — |
| `vclt_u8` | Compare unsigned less than | "vcgt.u8" | — |
| `vcltd_f64` | Floating-point compare less than | fcmp | — |
| `vcltd_s64` | Compare less than | cmp | — |
| `vcltd_u64` | Compare less than | cmp | — |
| `vcltq_f32` | Floating-point compare less than | "vcgt.f32" | — |
| `vcltq_f64` | Floating-point compare less than | fcmgt | — |
| `vcltq_s16` | Compare signed less than | "vcgt.s16" | — |
| `vcltq_s32` | Compare signed less than | "vcgt.s32" | — |
| `vcltq_s64` | Compare signed less than | cmgt | — |
| `vcltq_s8` | Compare signed less than | "vcgt.s8" | — |
| `vcltq_u16` | Compare unsigned less than | "vcgt.u16" | — |
| `vcltq_u32` | Compare unsigned less than | "vcgt.u32" | — |
| `vcltq_u64` | Compare unsigned less than | cmhi | — |
| `vcltq_u8` | Compare unsigned less than | "vcgt.u8" | — |
| `vclts_f32` | Floating-point compare less than | fcmp | — |
| `vcltz_f32` | Floating-point compare less than zero | fcmlt | — |
| `vcltz_f64` | Floating-point compare less than zero | fcmlt | — |
| `vcltz_s16` | Compare signed less than zero | cmlt | — |
| `vcltz_s32` | Compare signed less than zero | cmlt | — |
| `vcltz_s64` | Compare signed less than zero | cmlt | — |
| `vcltz_s8` | Compare signed less than zero | cmlt | — |
| `vcltzd_f64` | Floating-point compare less than zero | fcmp | — |
| `vcltzd_s64` | Compare less than zero | asr | — |
| `vcltzq_f32` | Floating-point compare less than zero | fcmlt | — |
| `vcltzq_f64` | Floating-point compare less than zero | fcmlt | — |
| `vcltzq_s16` | Compare signed less than zero | cmlt | — |
| `vcltzq_s32` | Compare signed less than zero | cmlt | — |
| `vcltzq_s64` | Compare signed less than zero | cmlt | — |
| `vcltzq_s8` | Compare signed less than zero | cmlt | — |
| `vcltzs_f32` | Floating-point compare less than zero | fcmp | — |
| `vclz_s16` | Count leading zero bits | "vclz.i16" | — |
| `vclz_s32` | Count leading zero bits | "vclz.i32" | — |
| `vclz_s8` | Count leading zero bits | "vclz.i8" | — |
| `vclz_u16` | Count leading zero bits | "vclz.i16" | — |
| `vclz_u32` | Count leading zero bits | "vclz.i32" | — |
| `vclz_u8` | Count leading zero bits | "vclz.i8" | — |
| `vclzq_s16` | Count leading zero bits | "vclz.i16" | — |
| `vclzq_s32` | Count leading zero bits | "vclz.i32" | — |
| `vclzq_s8` | Count leading zero bits | "vclz.i8" | — |
| `vclzq_u16` | Count leading zero bits | "vclz.i16" | — |
| `vclzq_u32` | Count leading zero bits | "vclz.i32" | — |
| `vclzq_u8` | Count leading zero bits | "vclz.i8" | — |
| `vcnt_p8` | Population count per byte | vcnt | — |
| `vcnt_s8` | Population count per byte | vcnt | — |
| `vcnt_u8` | Population count per byte | vcnt | — |
| `vcntq_p8` | Population count per byte | vcnt | — |
| `vcntq_s8` | Population count per byte | vcnt | — |
| `vcntq_u8` | Population count per byte | vcnt | — |
| `vcombine_f32` | Join two smaller vectors into a single larger vector | nop | — |
| `vcombine_f64` | Vector combine | mov | — |
| `vcombine_p16` | Join two smaller vectors into a single larger vector | nop | — |
| `vcombine_p64` | Join two smaller vectors into a single larger vector | nop | — |
| `vcombine_p8` | Join two smaller vectors into a single larger vector | nop | — |
| `vcombine_s16` | Join two smaller vectors into a single larger vector | nop | — |
| `vcombine_s32` | Join two smaller vectors into a single larger vector | nop | — |
| `vcombine_s64` | Join two smaller vectors into a single larger vector | nop | — |
| `vcombine_s8` | Join two smaller vectors into a single larger vector | nop | — |
| `vcombine_u16` | Join two smaller vectors into a single larger vector | nop | — |
| `vcombine_u32` | Join two smaller vectors into a single larger vector | nop | — |
| `vcombine_u64` | Join two smaller vectors into a single larger vector | nop | — |
| `vcombine_u8` | Join two smaller vectors into a single larger vector | nop | — |
| `vcopy_lane_f32` | Insert vector element from another vector element | mov | — |
| `vcopy_lane_f64` | Duplicate vector element to vector or scalar | nop | — |
| `vcopy_lane_p16` | Insert vector element from another vector element | mov | — |
| `vcopy_lane_p64` | Duplicate vector element to vector or scalar | nop | — |
| `vcopy_lane_p8` | Insert vector element from another vector element | mov | — |
| `vcopy_lane_s16` | Insert vector element from another vector element | mov | — |
| `vcopy_lane_s32` | Insert vector element from another vector element | mov | — |
| `vcopy_lane_s64` | Duplicate vector element to vector or scalar | nop | — |
| `vcopy_lane_s8` | Insert vector element from another vector element | mov | — |
| `vcopy_lane_u16` | Insert vector element from another vector element | mov | — |
| `vcopy_lane_u32` | Insert vector element from another vector element | mov | — |
| `vcopy_lane_u64` | Duplicate vector element to vector or scalar | nop | — |
| `vcopy_lane_u8` | Insert vector element from another vector element | mov | — |
| `vcopy_laneq_f32` | Insert vector element from another vector element | mov | — |
| `vcopy_laneq_f64` | Duplicate vector element to vector or scalar | nop | — |
| `vcopy_laneq_p16` | Insert vector element from another vector element | mov | — |
| `vcopy_laneq_p64` | Duplicate vector element to vector or scalar | nop | — |
| `vcopy_laneq_p8` | Insert vector element from another vector element | mov | — |
| `vcopy_laneq_s16` | Insert vector element from another vector element | mov | — |
| `vcopy_laneq_s32` | Insert vector element from another vector element | mov | — |
| `vcopy_laneq_s64` | Duplicate vector element to vector or scalar | nop | — |
| `vcopy_laneq_s8` | Insert vector element from another vector element | mov | — |
| `vcopy_laneq_u16` | Insert vector element from another vector element | mov | — |
| `vcopy_laneq_u32` | Insert vector element from another vector element | mov | — |
| `vcopy_laneq_u64` | Duplicate vector element to vector or scalar | nop | — |
| `vcopy_laneq_u8` | Insert vector element from another vector element | mov | — |
| `vcopyq_lane_f32` | Insert vector element from another vector element | mov | — |
| `vcopyq_lane_f64` | Insert vector element from another vector element | mov | — |
| `vcopyq_lane_p16` | Insert vector element from another vector element | mov | — |
| `vcopyq_lane_p64` | Insert vector element from another vector element | mov | — |
| `vcopyq_lane_p8` | Insert vector element from another vector element | mov | — |
| `vcopyq_lane_s16` | Insert vector element from another vector element | mov | — |
| `vcopyq_lane_s32` | Insert vector element from another vector element | mov | — |
| `vcopyq_lane_s64` | Insert vector element from another vector element | mov | — |
| `vcopyq_lane_s8` | Insert vector element from another vector element | mov | — |
| `vcopyq_lane_u16` | Insert vector element from another vector element | mov | — |
| `vcopyq_lane_u32` | Insert vector element from another vector element | mov | — |
| `vcopyq_lane_u64` | Insert vector element from another vector element | mov | — |
| `vcopyq_lane_u8` | Insert vector element from another vector element | mov | — |
| `vcopyq_laneq_f32` | Insert vector element from another vector element | mov | — |
| `vcopyq_laneq_f64` | Insert vector element from another vector element | mov | — |
| `vcopyq_laneq_p16` | Insert vector element from another vector element | mov | — |
| `vcopyq_laneq_p64` | Insert vector element from another vector element | mov | — |
| `vcopyq_laneq_p8` | Insert vector element from another vector element | mov | — |
| `vcopyq_laneq_s16` | Insert vector element from another vector element | mov | — |
| `vcopyq_laneq_s32` | Insert vector element from another vector element | mov | — |
| `vcopyq_laneq_s64` | Insert vector element from another vector element | mov | — |
| `vcopyq_laneq_s8` | Insert vector element from another vector element | mov | — |
| `vcopyq_laneq_u16` | Insert vector element from another vector element | mov | — |
| `vcopyq_laneq_u32` | Insert vector element from another vector element | mov | — |
| `vcopyq_laneq_u64` | Insert vector element from another vector element | mov | — |
| `vcopyq_laneq_u8` | Insert vector element from another vector element | mov | — |
| `vcreate_f32` | Insert vector element from another vector element | nop | — |
| `vcreate_f64` | Insert vector element from another vector element | nop | — |
| `vcreate_p16` | Insert vector element from another vector element | nop | — |
| `vcreate_p8` | Insert vector element from another vector element | nop | — |
| `vcreate_s16` | Insert vector element from another vector element | nop | — |
| `vcreate_s32` | Insert vector element from another vector element | nop | — |
| `vcreate_s64` | Insert vector element from another vector element | nop | — |
| `vcreate_s8` | Insert vector element from another vector element | nop | — |
| `vcreate_u16` | Insert vector element from another vector element | nop | — |
| `vcreate_u32` | Insert vector element from another vector element | nop | — |
| `vcreate_u64` | Insert vector element from another vector element | nop | — |
| `vcreate_u8` | Insert vector element from another vector element | nop | — |
| `vcvt_f32_f64` | Floating-point convert | fcvtn | — |
| `vcvt_f32_s32` | Fixed-point convert to floating-point | vcvt | — |
| `vcvt_f32_u32` | Fixed-point convert to floating-point | vcvt | — |
| `vcvt_f64_f32` | Floating-point convert to higher precision long | fcvtl | — |
| `vcvt_f64_s64` | Fixed-point convert to floating-point | scvtf | — |
| `vcvt_f64_u64` | Fixed-point convert to floating-point | ucvtf | — |
| `vcvt_high_f32_f64` | Floating-point convert to lower precision narrow | fcvtn2 | — |
| `vcvt_high_f64_f32` | Floating-point convert to higher precision long | fcvtl2 | — |
| `vcvt_n_f64_s64` | Fixed-point convert to floating-point | scvtf | — |
| `vcvt_n_f64_u64` | Fixed-point convert to floating-point | ucvtf | — |
| `vcvt_n_s64_f64` | Floating-point convert to fixed-point, rounding toward zero | fcvtzs | — |
| `vcvt_n_u64_f64` | Floating-point convert to fixed-point, rounding toward zero | fcvtzu | — |
| `vcvt_s32_f32` | Floating-point convert to signed fixed-point, rounding towar... | vcvt | — |
| `vcvt_s64_f64` | Floating-point convert to signed fixed-point, rounding towar... | fcvtzs | — |
| `vcvt_u32_f32` | Floating-point convert to unsigned fixed-point, rounding tow... | vcvt | — |
| `vcvt_u64_f64` | Floating-point convert to unsigned fixed-point, rounding tow... | fcvtzu | — |
| `vcvta_s32_f32` | Floating-point convert to signed integer, rounding to neares... | fcvtas | — |
| `vcvta_s64_f64` | Floating-point convert to signed integer, rounding to neares... | fcvtas | — |
| `vcvta_u32_f32` | Floating-point convert to unsigned integer, rounding to near... | fcvtau | — |
| `vcvta_u64_f64` | Floating-point convert to unsigned integer, rounding to near... | fcvtau | — |
| `vcvtad_s64_f64` | Floating-point convert to integer, rounding to nearest with ... | fcvtas | — |
| `vcvtad_u64_f64` | Floating-point convert to integer, rounding to nearest with ... | fcvtau | — |
| `vcvtaq_s32_f32` | Floating-point convert to signed integer, rounding to neares... | fcvtas | — |
| `vcvtaq_s64_f64` | Floating-point convert to signed integer, rounding to neares... | fcvtas | — |
| `vcvtaq_u32_f32` | Floating-point convert to unsigned integer, rounding to near... | fcvtau | — |
| `vcvtaq_u64_f64` | Floating-point convert to unsigned integer, rounding to near... | fcvtau | — |
| `vcvtas_s32_f32` | Floating-point convert to integer, rounding to nearest with ... | fcvtas | — |
| `vcvtas_u32_f32` | Floating-point convert to integer, rounding to nearest with ... | fcvtau | — |
| `vcvtd_f64_s64` | Fixed-point convert to floating-point | scvtf | — |
| `vcvtd_f64_u64` | Fixed-point convert to floating-point | ucvtf | — |
| `vcvtd_n_f64_s64` | Fixed-point convert to floating-point | scvtf | — |
| `vcvtd_n_f64_u64` | Fixed-point convert to floating-point | ucvtf | — |
| `vcvtd_n_s64_f64` | Floating-point convert to fixed-point, rounding toward zero | fcvtzs | — |
| `vcvtd_n_u64_f64` | Floating-point convert to fixed-point, rounding toward zero | fcvtzu | — |
| `vcvtd_s64_f64` | Fixed-point convert to floating-point | fcvtzs | — |
| `vcvtd_u64_f64` | Fixed-point convert to floating-point | fcvtzu | — |
| `vcvtm_s32_f32` | Floating-point convert to signed integer, rounding toward mi... | fcvtms | — |
| `vcvtm_s64_f64` | Floating-point convert to signed integer, rounding toward mi... | fcvtms | — |
| `vcvtm_u32_f32` | Floating-point convert to unsigned integer, rounding toward ... | fcvtmu | — |
| `vcvtm_u64_f64` | Floating-point convert to unsigned integer, rounding toward ... | fcvtmu | — |
| `vcvtmd_s64_f64` | Floating-point convert to signed integer, rounding toward mi... | fcvtms | — |
| `vcvtmd_u64_f64` | Floating-point convert to unsigned integer, rounding toward ... | fcvtmu | — |
| `vcvtmq_s32_f32` | Floating-point convert to signed integer, rounding toward mi... | fcvtms | — |
| `vcvtmq_s64_f64` | Floating-point convert to signed integer, rounding toward mi... | fcvtms | — |
| `vcvtmq_u32_f32` | Floating-point convert to unsigned integer, rounding toward ... | fcvtmu | — |
| `vcvtmq_u64_f64` | Floating-point convert to unsigned integer, rounding toward ... | fcvtmu | — |
| `vcvtms_s32_f32` | Floating-point convert to signed integer, rounding toward mi... | fcvtms | — |
| `vcvtms_u32_f32` | Floating-point convert to unsigned integer, rounding toward ... | fcvtmu | — |
| `vcvtn_s32_f32` | Floating-point convert to signed integer, rounding to neares... | fcvtns | — |
| `vcvtn_s64_f64` | Floating-point convert to signed integer, rounding to neares... | fcvtns | — |
| `vcvtn_u32_f32` | Floating-point convert to unsigned integer, rounding to near... | fcvtnu | — |
| `vcvtn_u64_f64` | Floating-point convert to unsigned integer, rounding to near... | fcvtnu | — |
| `vcvtnd_s64_f64` | Floating-point convert to signed integer, rounding to neares... | fcvtns | — |
| `vcvtnd_u64_f64` | Floating-point convert to unsigned integer, rounding to near... | fcvtnu | — |
| `vcvtnq_s32_f32` | Floating-point convert to signed integer, rounding to neares... | fcvtns | — |
| `vcvtnq_s64_f64` | Floating-point convert to signed integer, rounding to neares... | fcvtns | — |
| `vcvtnq_u32_f32` | Floating-point convert to unsigned integer, rounding to near... | fcvtnu | — |
| `vcvtnq_u64_f64` | Floating-point convert to unsigned integer, rounding to near... | fcvtnu | — |
| `vcvtns_s32_f32` | Floating-point convert to signed integer, rounding to neares... | fcvtns | — |
| `vcvtns_u32_f32` | Floating-point convert to unsigned integer, rounding to near... | fcvtnu | — |
| `vcvtp_s32_f32` | Floating-point convert to signed integer, rounding toward pl... | fcvtps | — |
| `vcvtp_s64_f64` | Floating-point convert to signed integer, rounding toward pl... | fcvtps | — |
| `vcvtp_u32_f32` | Floating-point convert to unsigned integer, rounding toward ... | fcvtpu | — |
| `vcvtp_u64_f64` | Floating-point convert to unsigned integer, rounding toward ... | fcvtpu | — |
| `vcvtpd_s64_f64` | Floating-point convert to signed integer, rounding toward pl... | fcvtps | — |
| `vcvtpd_u64_f64` | Floating-point convert to unsigned integer, rounding toward ... | fcvtpu | — |
| `vcvtpq_s32_f32` | Floating-point convert to signed integer, rounding toward pl... | fcvtps | — |
| `vcvtpq_s64_f64` | Floating-point convert to signed integer, rounding toward pl... | fcvtps | — |
| `vcvtpq_u32_f32` | Floating-point convert to unsigned integer, rounding toward ... | fcvtpu | — |
| `vcvtpq_u64_f64` | Floating-point convert to unsigned integer, rounding toward ... | fcvtpu | — |
| `vcvtps_s32_f32` | Floating-point convert to signed integer, rounding toward pl... | fcvtps | — |
| `vcvtps_u32_f32` | Floating-point convert to unsigned integer, rounding toward ... | fcvtpu | — |
| `vcvtq_f32_s32` | Fixed-point convert to floating-point | vcvt | — |
| `vcvtq_f32_u32` | Fixed-point convert to floating-point | vcvt | — |
| `vcvtq_f64_s64` | Fixed-point convert to floating-point | scvtf | — |
| `vcvtq_f64_u64` | Fixed-point convert to floating-point | ucvtf | — |
| `vcvtq_n_f64_s64` | Fixed-point convert to floating-point | scvtf | — |
| `vcvtq_n_f64_u64` | Fixed-point convert to floating-point | ucvtf | — |
| `vcvtq_n_s64_f64` | Floating-point convert to fixed-point, rounding toward zero | fcvtzs | — |
| `vcvtq_n_u64_f64` | Floating-point convert to fixed-point, rounding toward zero | fcvtzu | — |
| `vcvtq_s32_f32` | Floating-point convert to signed fixed-point, rounding towar... | vcvt | — |
| `vcvtq_s64_f64` | Floating-point convert to signed fixed-point, rounding towar... | fcvtzs | — |
| `vcvtq_u32_f32` | Floating-point convert to unsigned fixed-point, rounding tow... | vcvt | — |
| `vcvtq_u64_f64` | Floating-point convert to unsigned fixed-point, rounding tow... | fcvtzu | — |
| `vcvts_f32_s32` | Fixed-point convert to floating-point | scvtf | — |
| `vcvts_f32_u32` | Fixed-point convert to floating-point | ucvtf | — |
| `vcvts_n_f32_s32` | Fixed-point convert to floating-point | scvtf | — |
| `vcvts_n_f32_u32` | Fixed-point convert to floating-point | ucvtf | — |
| `vcvts_n_s32_f32` | Floating-point convert to fixed-point, rounding toward zero | fcvtzs | — |
| `vcvts_n_u32_f32` | Floating-point convert to fixed-point, rounding toward zero | fcvtzu | — |
| `vcvts_s32_f32` | Fixed-point convert to floating-point | fcvtzs | — |
| `vcvts_u32_f32` | Fixed-point convert to floating-point | fcvtzu | — |
| `vcvtx_f32_f64` | Floating-point convert to lower precision narrow, rounding t... | fcvtxn | — |
| `vcvtx_high_f32_f64` | Floating-point convert to lower precision narrow, rounding t... | fcvtxn2 | — |
| `vcvtxd_f32_f64` | Floating-point convert to lower precision narrow, rounding t... | fcvtxn | — |
| `vdiv_f32` | Divide | fdiv | — |
| `vdiv_f64` | Divide | fdiv | — |
| `vdivq_f32` | Divide | fdiv | — |
| `vdivq_f64` | Divide | fdiv | — |
| `vdup_lane_f32` | Set all vector lanes to the same value | "vdup.32" | — |
| `vdup_lane_f64` | Set all vector lanes to the same value | nop | — |
| `vdup_lane_p16` | Set all vector lanes to the same value | "vdup.16" | — |
| `vdup_lane_p64` | Set all vector lanes to the same value | nop | — |
| `vdup_lane_p8` | Set all vector lanes to the same value | "vdup.8" | — |
| `vdup_lane_s16` | Set all vector lanes to the same value | "vdup.16" | — |
| `vdup_lane_s32` | Set all vector lanes to the same value | "vdup.32" | — |
| `vdup_lane_s64` | Set all vector lanes to the same value | nop | — |
| `vdup_lane_s8` | Set all vector lanes to the same value | "vdup.8" | — |
| `vdup_lane_u16` | Set all vector lanes to the same value | "vdup.16" | — |
| `vdup_lane_u32` | Set all vector lanes to the same value | "vdup.32" | — |
| `vdup_lane_u64` | Set all vector lanes to the same value | nop | — |
| `vdup_lane_u8` | Set all vector lanes to the same value | "vdup.8" | — |
| `vdup_laneq_f32` | Set all vector lanes to the same value | "vdup.32" | — |
| `vdup_laneq_f64` | Set all vector lanes to the same value | nop | — |
| `vdup_laneq_p16` | Set all vector lanes to the same value | "vdup.16" | — |
| `vdup_laneq_p64` | Set all vector lanes to the same value | nop | — |
| `vdup_laneq_p8` | Set all vector lanes to the same value | "vdup.8" | — |
| `vdup_laneq_s16` | Set all vector lanes to the same value | "vdup.16" | — |
| `vdup_laneq_s32` | Set all vector lanes to the same value | "vdup.32" | — |
| `vdup_laneq_s64` | Set all vector lanes to the same value | vmov | — |
| `vdup_laneq_s8` | Set all vector lanes to the same value | "vdup.8" | — |
| `vdup_laneq_u16` | Set all vector lanes to the same value | "vdup.16" | — |
| `vdup_laneq_u32` | Set all vector lanes to the same value | "vdup.32" | — |
| `vdup_laneq_u64` | Set all vector lanes to the same value | vmov | — |
| `vdup_laneq_u8` | Set all vector lanes to the same value | "vdup.8" | — |
| `vdup_n_f32` | Duplicate vector element to vector or scalar | "vdup.32" | — |
| `vdup_n_f64` | Duplicate vector element to vector or scalar | nop | — |
| `vdup_n_p16` | Duplicate vector element to vector or scalar | "vdup.16" | — |
| `vdup_n_p64` | Duplicate vector element to vector or scalar | fmov | — |
| `vdup_n_p8` | Duplicate vector element to vector or scalar | "vdup.8" | — |
| `vdup_n_s16` | Duplicate vector element to vector or scalar | "vdup.16" | — |
| `vdup_n_s32` | Duplicate vector element to vector or scalar | "vdup.32" | — |
| `vdup_n_s64` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vdup_n_s8` | Duplicate vector element to vector or scalar | "vdup.8" | — |
| `vdup_n_u16` | Duplicate vector element to vector or scalar | "vdup.16" | — |
| `vdup_n_u32` | Duplicate vector element to vector or scalar | "vdup.32" | — |
| `vdup_n_u64` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vdup_n_u8` | Duplicate vector element to vector or scalar | "vdup.8" | — |
| `vdupb_lane_p8` | Set all vector lanes to the same value | nop | — |
| `vdupb_lane_s8` | Set all vector lanes to the same value | nop | — |
| `vdupb_lane_u8` | Set all vector lanes to the same value | nop | — |
| `vdupb_laneq_p8` | Extract an element from a vector | nop | — |
| `vdupb_laneq_s8` | Extract an element from a vector | nop | — |
| `vdupb_laneq_u8` | Extract an element from a vector | nop | — |
| `vdupd_lane_f64` | Set all vector lanes to the same value | nop | — |
| `vdupd_lane_s64` | Set all vector lanes to the same value | nop | — |
| `vdupd_lane_u64` | Set all vector lanes to the same value | nop | — |
| `vdupd_laneq_f64` | Set all vector lanes to the same value | nop | — |
| `vdupd_laneq_s64` | Set all vector lanes to the same value | nop | — |
| `vdupd_laneq_u64` | Set all vector lanes to the same value | nop | — |
| `vduph_lane_p16` | Set all vector lanes to the same value | nop | — |
| `vduph_lane_s16` | Set all vector lanes to the same value | nop | — |
| `vduph_lane_u16` | Set all vector lanes to the same value | nop | — |
| `vduph_laneq_p16` | Set all vector lanes to the same value | nop | — |
| `vduph_laneq_s16` | Set all vector lanes to the same value | nop | — |
| `vduph_laneq_u16` | Set all vector lanes to the same value | nop | — |
| `vdupq_lane_f32` | Set all vector lanes to the same value | "vdup.32" | — |
| `vdupq_lane_f64` | Set all vector lanes to the same value | dup | — |
| `vdupq_lane_p16` | Set all vector lanes to the same value | "vdup.16" | — |
| `vdupq_lane_p64` | Set all vector lanes to the same value | dup | — |
| `vdupq_lane_p8` | Set all vector lanes to the same value | "vdup.8" | — |
| `vdupq_lane_s16` | Set all vector lanes to the same value | "vdup.16" | — |
| `vdupq_lane_s32` | Set all vector lanes to the same value | "vdup.32" | — |
| `vdupq_lane_s64` | Set all vector lanes to the same value | vmov | — |
| `vdupq_lane_s8` | Set all vector lanes to the same value | "vdup.8" | — |
| `vdupq_lane_u16` | Set all vector lanes to the same value | "vdup.16" | — |
| `vdupq_lane_u32` | Set all vector lanes to the same value | "vdup.32" | — |
| `vdupq_lane_u64` | Set all vector lanes to the same value | vmov | — |
| `vdupq_lane_u8` | Set all vector lanes to the same value | "vdup.8" | — |
| `vdupq_laneq_f32` | Set all vector lanes to the same value | "vdup.32" | — |
| `vdupq_laneq_f64` | Set all vector lanes to the same value | dup | — |
| `vdupq_laneq_p16` | Set all vector lanes to the same value | "vdup.16" | — |
| `vdupq_laneq_p64` | Set all vector lanes to the same value | dup | — |
| `vdupq_laneq_p8` | Set all vector lanes to the same value | "vdup.8" | — |
| `vdupq_laneq_s16` | Set all vector lanes to the same value | "vdup.16" | — |
| `vdupq_laneq_s32` | Set all vector lanes to the same value | "vdup.32" | — |
| `vdupq_laneq_s64` | Set all vector lanes to the same value | vmov | — |
| `vdupq_laneq_s8` | Set all vector lanes to the same value | "vdup.8" | — |
| `vdupq_laneq_u16` | Set all vector lanes to the same value | "vdup.16" | — |
| `vdupq_laneq_u32` | Set all vector lanes to the same value | "vdup.32" | — |
| `vdupq_laneq_u64` | Set all vector lanes to the same value | vmov | — |
| `vdupq_laneq_u8` | Set all vector lanes to the same value | "vdup.8" | — |
| `vdupq_n_f32` | Duplicate vector element to vector or scalar | "vdup.32" | — |
| `vdupq_n_f64` | Duplicate vector element to vector or scalar | dup | — |
| `vdupq_n_p16` | Duplicate vector element to vector or scalar | "vdup.16" | — |
| `vdupq_n_p64` | Duplicate vector element to vector or scalar | dup | — |
| `vdupq_n_p8` | Duplicate vector element to vector or scalar | "vdup.8" | — |
| `vdupq_n_s16` | Duplicate vector element to vector or scalar | "vdup.16" | — |
| `vdupq_n_s32` | Duplicate vector element to vector or scalar | "vdup.32" | — |
| `vdupq_n_s64` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vdupq_n_s8` | Duplicate vector element to vector or scalar | "vdup.8" | — |
| `vdupq_n_u16` | Duplicate vector element to vector or scalar | "vdup.16" | — |
| `vdupq_n_u32` | Duplicate vector element to vector or scalar | "vdup.32" | — |
| `vdupq_n_u64` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vdupq_n_u8` | Duplicate vector element to vector or scalar | "vdup.8" | — |
| `vdups_lane_f32` | Set all vector lanes to the same value | nop | — |
| `vdups_lane_s32` | Set all vector lanes to the same value | nop | — |
| `vdups_lane_u32` | Set all vector lanes to the same value | nop | — |
| `vdups_laneq_f32` | Set all vector lanes to the same value | nop | — |
| `vdups_laneq_s32` | Set all vector lanes to the same value | nop | — |
| `vdups_laneq_u32` | Set all vector lanes to the same value | nop | — |
| `veor_s16` | Vector bitwise exclusive or (vector) | veor | — |
| `veor_s32` | Vector bitwise exclusive or (vector) | veor | — |
| `veor_s64` | Vector bitwise exclusive or (vector) | veor | — |
| `veor_s8` | Vector bitwise exclusive or (vector) | veor | — |
| `veor_u16` | Vector bitwise exclusive or (vector) | veor | — |
| `veor_u32` | Vector bitwise exclusive or (vector) | veor | — |
| `veor_u64` | Vector bitwise exclusive or (vector) | veor | — |
| `veor_u8` | Vector bitwise exclusive or (vector) | veor | — |
| `veorq_s16` | Vector bitwise exclusive or (vector) | veor | — |
| `veorq_s32` | Vector bitwise exclusive or (vector) | veor | — |
| `veorq_s64` | Vector bitwise exclusive or (vector) | veor | — |
| `veorq_s8` | Vector bitwise exclusive or (vector) | veor | — |
| `veorq_u16` | Vector bitwise exclusive or (vector) | veor | — |
| `veorq_u32` | Vector bitwise exclusive or (vector) | veor | — |
| `veorq_u64` | Vector bitwise exclusive or (vector) | veor | — |
| `veorq_u8` | Vector bitwise exclusive or (vector) | veor | — |
| `vext_f32` | Extract vector from pair of vectors | "vext.8" | — |
| `vext_f64` | Extract vector from pair of vectors | nop | — |
| `vext_p16` | Extract vector from pair of vectors | "vext.8" | — |
| `vext_p64` | Extract vector from pair of vectors | nop | — |
| `vext_p8` | Extract vector from pair of vectors | "vext.8" | — |
| `vext_s16` | Extract vector from pair of vectors | "vext.8" | — |
| `vext_s32` | Extract vector from pair of vectors | "vext.8" | — |
| `vext_s8` | Extract vector from pair of vectors | "vext.8" | — |
| `vext_u16` | Extract vector from pair of vectors | "vext.8" | — |
| `vext_u32` | Extract vector from pair of vectors | "vext.8" | — |
| `vext_u8` | Extract vector from pair of vectors | "vext.8" | — |
| `vextq_f32` | Extract vector from pair of vectors | "vext.8" | — |
| `vextq_f64` | Extract vector from pair of vectors | ext | — |
| `vextq_p16` | Extract vector from pair of vectors | "vext.8" | — |
| `vextq_p64` | Extract vector from pair of vectors | ext | — |
| `vextq_p8` | Extract vector from pair of vectors | "vext.8" | — |
| `vextq_s16` | Extract vector from pair of vectors | "vext.8" | — |
| `vextq_s32` | Extract vector from pair of vectors | "vext.8" | — |
| `vextq_s64` | Extract vector from pair of vectors | vmov | — |
| `vextq_s8` | Extract vector from pair of vectors | "vext.8" | — |
| `vextq_u16` | Extract vector from pair of vectors | "vext.8" | — |
| `vextq_u32` | Extract vector from pair of vectors | "vext.8" | — |
| `vextq_u64` | Extract vector from pair of vectors | vmov | — |
| `vextq_u8` | Extract vector from pair of vectors | "vext.8" | — |
| `vfma_f32` | Floating-point fused Multiply-Add to accumulator(vector) | vfma | — |
| `vfma_f64` | Floating-point fused Multiply-Add to accumulator(vector) | fmadd | — |
| `vfma_lane_f32` | Floating-point fused multiply-add to accumulator | fmla | — |
| `vfma_lane_f64` | Floating-point fused multiply-add to accumulator | fmadd | — |
| `vfma_laneq_f32` | Floating-point fused multiply-add to accumulator | fmla | — |
| `vfma_laneq_f64` | Floating-point fused multiply-add to accumulator | fmadd | — |
| `vfma_n_f32` | Floating-point fused Multiply-Add to accumulator(vector) | vfma | — |
| `vfma_n_f64` | Floating-point fused Multiply-Add to accumulator(vector) | fmadd | — |
| `vfmad_lane_f64` | Floating-point fused multiply-add to accumulator | fmadd | — |
| `vfmad_laneq_f64` | Floating-point fused multiply-add to accumulator | fmadd | — |
| `vfmaq_f32` | Floating-point fused Multiply-Add to accumulator(vector) | vfma | — |
| `vfmaq_f64` | Floating-point fused Multiply-Add to accumulator(vector) | fmla | — |
| `vfmaq_lane_f32` | Floating-point fused multiply-add to accumulator | fmla | — |
| `vfmaq_lane_f64` | Floating-point fused multiply-add to accumulator | fmla | — |
| `vfmaq_laneq_f32` | Floating-point fused multiply-add to accumulator | fmla | — |
| `vfmaq_laneq_f64` | Floating-point fused multiply-add to accumulator | fmla | — |
| `vfmaq_n_f32` | Floating-point fused Multiply-Add to accumulator(vector) | vfma | — |
| `vfmaq_n_f64` | Floating-point fused Multiply-Add to accumulator(vector) | fmla | — |
| `vfmas_lane_f32` | Floating-point fused multiply-add to accumulator | fmadd | — |
| `vfmas_laneq_f32` | Floating-point fused multiply-add to accumulator | fmadd | — |
| `vfms_f32` | Floating-point fused multiply-subtract from accumulator | vfms | — |
| `vfms_f64` | Floating-point fused multiply-subtract from accumulator | fmsub | — |
| `vfms_lane_f32` | Floating-point fused multiply-subtract to accumulator | fmls | — |
| `vfms_lane_f64` | Floating-point fused multiply-subtract to accumulator | fmsub | — |
| `vfms_laneq_f32` | Floating-point fused multiply-subtract to accumulator | fmls | — |
| `vfms_laneq_f64` | Floating-point fused multiply-subtract to accumulator | fmsub | — |
| `vfms_n_f32` | Floating-point fused Multiply-subtract to accumulator(vector... | vfms | — |
| `vfms_n_f64` | Floating-point fused Multiply-subtract to accumulator(vector... | fmsub | — |
| `vfmsd_lane_f64` | Floating-point fused multiply-subtract to accumulator | fmsub | — |
| `vfmsd_laneq_f64` | Floating-point fused multiply-subtract to accumulator | fmsub | — |
| `vfmsq_f32` | Floating-point fused multiply-subtract from accumulator | vfms | — |
| `vfmsq_f64` | Floating-point fused multiply-subtract from accumulator | fmls | — |
| `vfmsq_lane_f32` | Floating-point fused multiply-subtract to accumulator | fmls | — |
| `vfmsq_lane_f64` | Floating-point fused multiply-subtract to accumulator | fmls | — |
| `vfmsq_laneq_f32` | Floating-point fused multiply-subtract to accumulator | fmls | — |
| `vfmsq_laneq_f64` | Floating-point fused multiply-subtract to accumulator | fmls | — |
| `vfmsq_n_f32` | Floating-point fused Multiply-subtract to accumulator(vector... | vfms | — |
| `vfmsq_n_f64` | Floating-point fused Multiply-subtract to accumulator(vector... | fmls | — |
| `vfmss_lane_f32` | Floating-point fused multiply-subtract to accumulator | fmsub | — |
| `vfmss_laneq_f32` | Floating-point fused multiply-subtract to accumulator | fmsub | — |
| `vget_high_f32` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vget_high_f64` | Duplicate vector element to vector or scalar | nop | — |
| `vget_high_p16` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vget_high_p64` | Duplicate vector element to vector or scalar | ext | — |
| `vget_high_p8` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vget_high_s16` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vget_high_s32` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vget_high_s64` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vget_high_s8` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vget_high_u16` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vget_high_u32` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vget_high_u64` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vget_high_u8` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vget_lane_f32` | Move vector element to general-purpose register | nop | — |
| `vget_lane_f64` | Duplicate vector element to vector or scalar | nop | — |
| `vget_lane_p16` | Move vector element to general-purpose register | nop | — |
| `vget_lane_p64` | Move vector element to general-purpose register | nop | — |
| `vget_lane_p8` | Move vector element to general-purpose register | nop | — |
| `vget_lane_s16` | Move vector element to general-purpose register | nop | — |
| `vget_lane_s32` | Move vector element to general-purpose register | nop | — |
| `vget_lane_s64` | Move vector element to general-purpose register | nop | — |
| `vget_lane_s8` | Move vector element to general-purpose register | nop | — |
| `vget_lane_u16` | Move vector element to general-purpose register | nop | — |
| `vget_lane_u32` | Move vector element to general-purpose register | nop | — |
| `vget_lane_u64` | Move vector element to general-purpose register | nop | — |
| `vget_lane_u8` | Move vector element to general-purpose register | nop | — |
| `vget_low_f32` | Duplicate vector element to vector or scalar | nop | — |
| `vget_low_f64` | Duplicate vector element to vector or scalar | nop | — |
| `vget_low_p16` | Duplicate vector element to vector or scalar | nop | — |
| `vget_low_p64` | Duplicate vector element to vector or scalar | nop | — |
| `vget_low_p8` | Duplicate vector element to vector or scalar | nop | — |
| `vget_low_s16` | Duplicate vector element to vector or scalar | nop | — |
| `vget_low_s32` | Duplicate vector element to vector or scalar | nop | — |
| `vget_low_s64` | Duplicate vector element to vector or scalar | nop | — |
| `vget_low_s8` | Duplicate vector element to vector or scalar | nop | — |
| `vget_low_u16` | Duplicate vector element to vector or scalar | nop | — |
| `vget_low_u32` | Duplicate vector element to vector or scalar | nop | — |
| `vget_low_u64` | Duplicate vector element to vector or scalar | nop | — |
| `vget_low_u8` | Duplicate vector element to vector or scalar | nop | — |
| `vgetq_lane_f32` | Move vector element to general-purpose register | nop | — |
| `vgetq_lane_f64` | Duplicate vector element to vector or scalar | nop | — |
| `vgetq_lane_p16` | Move vector element to general-purpose register | nop | — |
| `vgetq_lane_p64` | Move vector element to general-purpose register | nop | — |
| `vgetq_lane_p8` | Move vector element to general-purpose register | nop | — |
| `vgetq_lane_s16` | Move vector element to general-purpose register | nop | — |
| `vgetq_lane_s32` | Move vector element to general-purpose register | nop | — |
| `vgetq_lane_s64` | Move vector element to general-purpose register | nop | — |
| `vgetq_lane_s8` | Move vector element to general-purpose register | nop | — |
| `vgetq_lane_u16` | Move vector element to general-purpose register | nop | — |
| `vgetq_lane_u32` | Move vector element to general-purpose register | nop | — |
| `vgetq_lane_u64` | Move vector element to general-purpose register | nop | — |
| `vgetq_lane_u8` | Move vector element to general-purpose register | nop | — |
| `vhadd_s16` | Halving add | "vhadd.s16" | — |
| `vhadd_s32` | Halving add | "vhadd.s32" | — |
| `vhadd_s8` | Halving add | "vhadd.s8" | — |
| `vhadd_u16` | Halving add | "vhadd.u16" | — |
| `vhadd_u32` | Halving add | "vhadd.u32" | — |
| `vhadd_u8` | Halving add | "vhadd.u8" | — |
| `vhaddq_s16` | Halving add | "vhadd.s16" | — |
| `vhaddq_s32` | Halving add | "vhadd.s32" | — |
| `vhaddq_s8` | Halving add | "vhadd.s8" | — |
| `vhaddq_u16` | Halving add | "vhadd.u16" | — |
| `vhaddq_u32` | Halving add | "vhadd.u32" | — |
| `vhaddq_u8` | Halving add | "vhadd.u8" | — |
| `vhsub_s16` | Signed halving subtract | "vhsub.s16" | — |
| `vhsub_s32` | Signed halving subtract | "vhsub.s32" | — |
| `vhsub_s8` | Signed halving subtract | "vhsub.s8" | — |
| `vhsub_u16` | Signed halving subtract | "vhsub.u16" | — |
| `vhsub_u32` | Signed halving subtract | "vhsub.u32" | — |
| `vhsub_u8` | Signed halving subtract | "vhsub.u8" | — |
| `vhsubq_s16` | Signed halving subtract | "vhsub.s16" | — |
| `vhsubq_s32` | Signed halving subtract | "vhsub.s32" | — |
| `vhsubq_s8` | Signed halving subtract | "vhsub.s8" | — |
| `vhsubq_u16` | Signed halving subtract | "vhsub.u16" | — |
| `vhsubq_u32` | Signed halving subtract | "vhsub.u32" | — |
| `vhsubq_u8` | Signed halving subtract | "vhsub.u8" | — |
| `vmax_f32` | Maximum (vector) | vmax | — |
| `vmax_f64` | Maximum (vector) | fmax | — |
| `vmax_s16` | Maximum (vector) | vmax | — |
| `vmax_s32` | Maximum (vector) | vmax | — |
| `vmax_s8` | Maximum (vector) | vmax | — |
| `vmax_u16` | Maximum (vector) | vmax | — |
| `vmax_u32` | Maximum (vector) | vmax | — |
| `vmax_u8` | Maximum (vector) | vmax | — |
| `vmaxnm_f32` | Floating-point Maximum Number (vector) | vmaxnm | — |
| `vmaxnm_f64` | Floating-point Maximum Number (vector) | fmaxnm | — |
| `vmaxnmq_f32` | Floating-point Maximum Number (vector) | vmaxnm | — |
| `vmaxnmq_f64` | Floating-point Maximum Number (vector) | fmaxnm | — |
| `vmaxnmv_f32` | Floating-point maximum number across vector | fmaxnmp | — |
| `vmaxnmvq_f32` | Floating-point maximum number across vector | fmaxnmv | — |
| `vmaxnmvq_f64` | Floating-point maximum number across vector | fmaxnmp | — |
| `vmaxq_f32` | Maximum (vector) | vmax | — |
| `vmaxq_f64` | Maximum (vector) | fmax | — |
| `vmaxq_s16` | Maximum (vector) | vmax | — |
| `vmaxq_s32` | Maximum (vector) | vmax | — |
| `vmaxq_s8` | Maximum (vector) | vmax | — |
| `vmaxq_u16` | Maximum (vector) | vmax | — |
| `vmaxq_u32` | Maximum (vector) | vmax | — |
| `vmaxq_u8` | Maximum (vector) | vmax | — |
| `vmaxv_f32` | Horizontal vector max | fmaxp | — |
| `vmaxv_s16` | Horizontal vector max | smaxv | — |
| `vmaxv_s32` | Horizontal vector max | smaxp | — |
| `vmaxv_s8` | Horizontal vector max | smaxv | — |
| `vmaxv_u16` | Horizontal vector max | umaxv | — |
| `vmaxv_u32` | Horizontal vector max | umaxp | — |
| `vmaxv_u8` | Horizontal vector max | umaxv | — |
| `vmaxvq_f32` | Horizontal vector max | fmaxv | — |
| `vmaxvq_f64` | Horizontal vector max | fmaxp | — |
| `vmaxvq_s16` | Horizontal vector max | smaxv | — |
| `vmaxvq_s32` | Horizontal vector max | smaxv | — |
| `vmaxvq_s8` | Horizontal vector max | smaxv | — |
| `vmaxvq_u16` | Horizontal vector max | umaxv | — |
| `vmaxvq_u32` | Horizontal vector max | umaxv | — |
| `vmaxvq_u8` | Horizontal vector max | umaxv | — |
| `vmin_f32` | Minimum (vector) | vmin | — |
| `vmin_f64` | Minimum (vector) | fmin | — |
| `vmin_s16` | Minimum (vector) | vmin | — |
| `vmin_s32` | Minimum (vector) | vmin | — |
| `vmin_s8` | Minimum (vector) | vmin | — |
| `vmin_u16` | Minimum (vector) | vmin | — |
| `vmin_u32` | Minimum (vector) | vmin | — |
| `vmin_u8` | Minimum (vector) | vmin | — |
| `vminnm_f32` | Floating-point Minimum Number (vector) | vminnm | — |
| `vminnm_f64` | Floating-point Minimum Number (vector) | fminnm | — |
| `vminnmq_f32` | Floating-point Minimum Number (vector) | vminnm | — |
| `vminnmq_f64` | Floating-point Minimum Number (vector) | fminnm | — |
| `vminnmv_f32` | Floating-point minimum number across vector | fminnmp | — |
| `vminnmvq_f32` | Floating-point minimum number across vector | fminnmv | — |
| `vminnmvq_f64` | Floating-point minimum number across vector | fminnmp | — |
| `vminq_f32` | Minimum (vector) | vmin | — |
| `vminq_f64` | Minimum (vector) | fmin | — |
| `vminq_s16` | Minimum (vector) | vmin | — |
| `vminq_s32` | Minimum (vector) | vmin | — |
| `vminq_s8` | Minimum (vector) | vmin | — |
| `vminq_u16` | Minimum (vector) | vmin | — |
| `vminq_u32` | Minimum (vector) | vmin | — |
| `vminq_u8` | Minimum (vector) | vmin | — |
| `vminv_f32` | Horizontal vector min | fminp | — |
| `vminv_s16` | Horizontal vector min | sminv | — |
| `vminv_s32` | Horizontal vector min | sminp | — |
| `vminv_s8` | Horizontal vector min | sminv | — |
| `vminv_u16` | Horizontal vector min | uminv | — |
| `vminv_u32` | Horizontal vector min | uminp | — |
| `vminv_u8` | Horizontal vector min | uminv | — |
| `vminvq_f32` | Horizontal vector min | fminv | — |
| `vminvq_f64` | Horizontal vector min | fminp | — |
| `vminvq_s16` | Horizontal vector min | sminv | — |
| `vminvq_s32` | Horizontal vector min | sminv | — |
| `vminvq_s8` | Horizontal vector min | sminv | — |
| `vminvq_u16` | Horizontal vector min | uminv | — |
| `vminvq_u32` | Horizontal vector min | uminv | — |
| `vminvq_u8` | Horizontal vector min | uminv | — |
| `vmla_f32` | Floating-point multiply-add to accumulator | "vmla.f32" | — |
| `vmla_f64` | Floating-point multiply-add to accumulator | fmul | — |
| `vmla_lane_f32` | Vector multiply accumulate with scalar | "vmla.f32" | — |
| `vmla_lane_s16` | Vector multiply accumulate with scalar | "vmla.i16" | — |
| `vmla_lane_s32` | Vector multiply accumulate with scalar | "vmla.i32" | — |
| `vmla_lane_u16` | Vector multiply accumulate with scalar | "vmla.i16" | — |
| `vmla_lane_u32` | Vector multiply accumulate with scalar | "vmla.i32" | — |
| `vmla_laneq_f32` | Vector multiply accumulate with scalar | "vmla.f32" | — |
| `vmla_laneq_s16` | Vector multiply accumulate with scalar | "vmla.i16" | — |
| `vmla_laneq_s32` | Vector multiply accumulate with scalar | "vmla.i32" | — |
| `vmla_laneq_u16` | Vector multiply accumulate with scalar | "vmla.i16" | — |
| `vmla_laneq_u32` | Vector multiply accumulate with scalar | "vmla.i32" | — |
| `vmla_n_f32` | Vector multiply accumulate with scalar | "vmla.f32" | — |
| `vmla_n_s16` | Vector multiply accumulate with scalar | "vmla.i16" | — |
| `vmla_n_s32` | Vector multiply accumulate with scalar | "vmla.i32" | — |
| `vmla_n_u16` | Vector multiply accumulate with scalar | "vmla.i16" | — |
| `vmla_n_u32` | Vector multiply accumulate with scalar | "vmla.i32" | — |
| `vmla_s16` | Multiply-add to accumulator | "vmla.i16" | — |
| `vmla_s32` | Multiply-add to accumulator | "vmla.i32" | — |
| `vmla_s8` | Multiply-add to accumulator | "vmla.i8" | — |
| `vmla_u16` | Multiply-add to accumulator | "vmla.i16" | — |
| `vmla_u32` | Multiply-add to accumulator | "vmla.i32" | — |
| `vmla_u8` | Multiply-add to accumulator | "vmla.i8" | — |
| `vmlal_high_lane_s16` | Multiply-add long | smlal2 | — |
| `vmlal_high_lane_s32` | Multiply-add long | smlal2 | — |
| `vmlal_high_lane_u16` | Multiply-add long | umlal2 | — |
| `vmlal_high_lane_u32` | Multiply-add long | umlal2 | — |
| `vmlal_high_laneq_s16` | Multiply-add long | smlal2 | — |
| `vmlal_high_laneq_s32` | Multiply-add long | smlal2 | — |
| `vmlal_high_laneq_u16` | Multiply-add long | umlal2 | — |
| `vmlal_high_laneq_u32` | Multiply-add long | umlal2 | — |
| `vmlal_high_n_s16` | Multiply-add long | smlal2 | — |
| `vmlal_high_n_s32` | Multiply-add long | smlal2 | — |
| `vmlal_high_n_u16` | Multiply-add long | umlal2 | — |
| `vmlal_high_n_u32` | Multiply-add long | umlal2 | — |
| `vmlal_high_s16` | Signed multiply-add long | smlal2 | — |
| `vmlal_high_s32` | Signed multiply-add long | smlal2 | — |
| `vmlal_high_s8` | Signed multiply-add long | smlal2 | — |
| `vmlal_high_u16` | Unsigned multiply-add long | umlal2 | — |
| `vmlal_high_u32` | Unsigned multiply-add long | umlal2 | — |
| `vmlal_high_u8` | Unsigned multiply-add long | umlal2 | — |
| `vmlal_lane_s16` | Vector widening multiply accumulate with scalar | "vmlal.s16" | — |
| `vmlal_lane_s32` | Vector widening multiply accumulate with scalar | "vmlal.s32" | — |
| `vmlal_lane_u16` | Vector widening multiply accumulate with scalar | "vmlal.u16" | — |
| `vmlal_lane_u32` | Vector widening multiply accumulate with scalar | "vmlal.u32" | — |
| `vmlal_laneq_s16` | Vector widening multiply accumulate with scalar | "vmlal.s16" | — |
| `vmlal_laneq_s32` | Vector widening multiply accumulate with scalar | "vmlal.s32" | — |
| `vmlal_laneq_u16` | Vector widening multiply accumulate with scalar | "vmlal.u16" | — |
| `vmlal_laneq_u32` | Vector widening multiply accumulate with scalar | "vmlal.u32" | — |
| `vmlal_n_s16` | Vector widening multiply accumulate with scalar | "vmlal.s16" | — |
| `vmlal_n_s32` | Vector widening multiply accumulate with scalar | "vmlal.s32" | — |
| `vmlal_n_u16` | Vector widening multiply accumulate with scalar | "vmlal.u16" | — |
| `vmlal_n_u32` | Vector widening multiply accumulate with scalar | "vmlal.u32" | — |
| `vmlal_s16` | Signed multiply-add long | "vmlal.s16" | — |
| `vmlal_s32` | Signed multiply-add long | "vmlal.s32" | — |
| `vmlal_s8` | Signed multiply-add long | "vmlal.s8" | — |
| `vmlal_u16` | Unsigned multiply-add long | "vmlal.u16" | — |
| `vmlal_u32` | Unsigned multiply-add long | "vmlal.u32" | — |
| `vmlal_u8` | Unsigned multiply-add long | "vmlal.u8" | — |
| `vmlaq_f32` | Floating-point multiply-add to accumulator | "vmla.f32" | — |
| `vmlaq_f64` | Floating-point multiply-add to accumulator | fmul | — |
| `vmlaq_lane_f32` | Vector multiply accumulate with scalar | "vmla.f32" | — |
| `vmlaq_lane_s16` | Vector multiply accumulate with scalar | "vmla.i16" | — |
| `vmlaq_lane_s32` | Vector multiply accumulate with scalar | "vmla.i32" | — |
| `vmlaq_lane_u16` | Vector multiply accumulate with scalar | "vmla.i16" | — |
| `vmlaq_lane_u32` | Vector multiply accumulate with scalar | "vmla.i32" | — |
| `vmlaq_laneq_f32` | Vector multiply accumulate with scalar | "vmla.f32" | — |
| `vmlaq_laneq_s16` | Vector multiply accumulate with scalar | "vmla.i16" | — |
| `vmlaq_laneq_s32` | Vector multiply accumulate with scalar | "vmla.i32" | — |
| `vmlaq_laneq_u16` | Vector multiply accumulate with scalar | "vmla.i16" | — |
| `vmlaq_laneq_u32` | Vector multiply accumulate with scalar | "vmla.i32" | — |
| `vmlaq_n_f32` | Vector multiply accumulate with scalar | "vmla.f32" | — |
| `vmlaq_n_s16` | Vector multiply accumulate with scalar | "vmla.i16" | — |
| `vmlaq_n_s32` | Vector multiply accumulate with scalar | "vmla.i32" | — |
| `vmlaq_n_u16` | Vector multiply accumulate with scalar | "vmla.i16" | — |
| `vmlaq_n_u32` | Vector multiply accumulate with scalar | "vmla.i32" | — |
| `vmlaq_s16` | Multiply-add to accumulator | "vmla.i16" | — |
| `vmlaq_s32` | Multiply-add to accumulator | "vmla.i32" | — |
| `vmlaq_s8` | Multiply-add to accumulator | "vmla.i8" | — |
| `vmlaq_u16` | Multiply-add to accumulator | "vmla.i16" | — |
| `vmlaq_u32` | Multiply-add to accumulator | "vmla.i32" | — |
| `vmlaq_u8` | Multiply-add to accumulator | "vmla.i8" | — |
| `vmls_f32` | Floating-point multiply-subtract from accumulator | "vmls.f32" | — |
| `vmls_f64` | Floating-point multiply-subtract from accumulator | fmul | — |
| `vmls_lane_f32` | Vector multiply subtract with scalar | "vmls.f32" | — |
| `vmls_lane_s16` | Vector multiply subtract with scalar | "vmls.i16" | — |
| `vmls_lane_s32` | Vector multiply subtract with scalar | "vmls.i32" | — |
| `vmls_lane_u16` | Vector multiply subtract with scalar | "vmls.i16" | — |
| `vmls_lane_u32` | Vector multiply subtract with scalar | "vmls.i32" | — |
| `vmls_laneq_f32` | Vector multiply subtract with scalar | "vmls.f32" | — |
| `vmls_laneq_s16` | Vector multiply subtract with scalar | "vmls.i16" | — |
| `vmls_laneq_s32` | Vector multiply subtract with scalar | "vmls.i32" | — |
| `vmls_laneq_u16` | Vector multiply subtract with scalar | "vmls.i16" | — |
| `vmls_laneq_u32` | Vector multiply subtract with scalar | "vmls.i32" | — |
| `vmls_n_f32` | Vector multiply subtract with scalar | "vmls.f32" | — |
| `vmls_n_s16` | Vector multiply subtract with scalar | "vmls.i16" | — |
| `vmls_n_s32` | Vector multiply subtract with scalar | "vmls.i32" | — |
| `vmls_n_u16` | Vector multiply subtract with scalar | "vmls.i16" | — |
| `vmls_n_u32` | Vector multiply subtract with scalar | "vmls.i32" | — |
| `vmls_s16` | Multiply-subtract from accumulator | "vmls.i16" | — |
| `vmls_s32` | Multiply-subtract from accumulator | "vmls.i32" | — |
| `vmls_s8` | Multiply-subtract from accumulator | "vmls.i8" | — |
| `vmls_u16` | Multiply-subtract from accumulator | "vmls.i16" | — |
| `vmls_u32` | Multiply-subtract from accumulator | "vmls.i32" | — |
| `vmls_u8` | Multiply-subtract from accumulator | "vmls.i8" | — |
| `vmlsl_high_lane_s16` | Multiply-subtract long | smlsl2 | — |
| `vmlsl_high_lane_s32` | Multiply-subtract long | smlsl2 | — |
| `vmlsl_high_lane_u16` | Multiply-subtract long | umlsl2 | — |
| `vmlsl_high_lane_u32` | Multiply-subtract long | umlsl2 | — |
| `vmlsl_high_laneq_s16` | Multiply-subtract long | smlsl2 | — |
| `vmlsl_high_laneq_s32` | Multiply-subtract long | smlsl2 | — |
| `vmlsl_high_laneq_u16` | Multiply-subtract long | umlsl2 | — |
| `vmlsl_high_laneq_u32` | Multiply-subtract long | umlsl2 | — |
| `vmlsl_high_n_s16` | Multiply-subtract long | smlsl2 | — |
| `vmlsl_high_n_s32` | Multiply-subtract long | smlsl2 | — |
| `vmlsl_high_n_u16` | Multiply-subtract long | umlsl2 | — |
| `vmlsl_high_n_u32` | Multiply-subtract long | umlsl2 | — |
| `vmlsl_high_s16` | Signed multiply-subtract long | smlsl2 | — |
| `vmlsl_high_s32` | Signed multiply-subtract long | smlsl2 | — |
| `vmlsl_high_s8` | Signed multiply-subtract long | smlsl2 | — |
| `vmlsl_high_u16` | Unsigned multiply-subtract long | umlsl2 | — |
| `vmlsl_high_u32` | Unsigned multiply-subtract long | umlsl2 | — |
| `vmlsl_high_u8` | Unsigned multiply-subtract long | umlsl2 | — |
| `vmlsl_lane_s16` | Vector widening multiply subtract with scalar | "vmlsl.s16" | — |
| `vmlsl_lane_s32` | Vector widening multiply subtract with scalar | "vmlsl.s32" | — |
| `vmlsl_lane_u16` | Vector widening multiply subtract with scalar | "vmlsl.u16" | — |
| `vmlsl_lane_u32` | Vector widening multiply subtract with scalar | "vmlsl.u32" | — |
| `vmlsl_laneq_s16` | Vector widening multiply subtract with scalar | "vmlsl.s16" | — |
| `vmlsl_laneq_s32` | Vector widening multiply subtract with scalar | "vmlsl.s32" | — |
| `vmlsl_laneq_u16` | Vector widening multiply subtract with scalar | "vmlsl.u16" | — |
| `vmlsl_laneq_u32` | Vector widening multiply subtract with scalar | "vmlsl.u32" | — |
| `vmlsl_n_s16` | Vector widening multiply subtract with scalar | "vmlsl.s16" | — |
| `vmlsl_n_s32` | Vector widening multiply subtract with scalar | "vmlsl.s32" | — |
| `vmlsl_n_u16` | Vector widening multiply subtract with scalar | "vmlsl.u16" | — |
| `vmlsl_n_u32` | Vector widening multiply subtract with scalar | "vmlsl.u32" | — |
| `vmlsl_s16` | Signed multiply-subtract long | "vmlsl.s16" | — |
| `vmlsl_s32` | Signed multiply-subtract long | "vmlsl.s32" | — |
| `vmlsl_s8` | Signed multiply-subtract long | "vmlsl.s8" | — |
| `vmlsl_u16` | Unsigned multiply-subtract long | "vmlsl.u16" | — |
| `vmlsl_u32` | Unsigned multiply-subtract long | "vmlsl.u32" | — |
| `vmlsl_u8` | Unsigned multiply-subtract long | "vmlsl.u8" | — |
| `vmlsq_f32` | Floating-point multiply-subtract from accumulator | "vmls.f32" | — |
| `vmlsq_f64` | Floating-point multiply-subtract from accumulator | fmul | — |
| `vmlsq_lane_f32` | Vector multiply subtract with scalar | "vmls.f32" | — |
| `vmlsq_lane_s16` | Vector multiply subtract with scalar | "vmls.i16" | — |
| `vmlsq_lane_s32` | Vector multiply subtract with scalar | "vmls.i32" | — |
| `vmlsq_lane_u16` | Vector multiply subtract with scalar | "vmls.i16" | — |
| `vmlsq_lane_u32` | Vector multiply subtract with scalar | "vmls.i32" | — |
| `vmlsq_laneq_f32` | Vector multiply subtract with scalar | "vmls.f32" | — |
| `vmlsq_laneq_s16` | Vector multiply subtract with scalar | "vmls.i16" | — |
| `vmlsq_laneq_s32` | Vector multiply subtract with scalar | "vmls.i32" | — |
| `vmlsq_laneq_u16` | Vector multiply subtract with scalar | "vmls.i16" | — |
| `vmlsq_laneq_u32` | Vector multiply subtract with scalar | "vmls.i32" | — |
| `vmlsq_n_f32` | Vector multiply subtract with scalar | "vmls.f32" | — |
| `vmlsq_n_s16` | Vector multiply subtract with scalar | "vmls.i16" | — |
| `vmlsq_n_s32` | Vector multiply subtract with scalar | "vmls.i32" | — |
| `vmlsq_n_u16` | Vector multiply subtract with scalar | "vmls.i16" | — |
| `vmlsq_n_u32` | Vector multiply subtract with scalar | "vmls.i32" | — |
| `vmlsq_s16` | Multiply-subtract from accumulator | "vmls.i16" | — |
| `vmlsq_s32` | Multiply-subtract from accumulator | "vmls.i32" | — |
| `vmlsq_s8` | Multiply-subtract from accumulator | "vmls.i8" | — |
| `vmlsq_u16` | Multiply-subtract from accumulator | "vmls.i16" | — |
| `vmlsq_u32` | Multiply-subtract from accumulator | "vmls.i32" | — |
| `vmlsq_u8` | Multiply-subtract from accumulator | "vmls.i8" | — |
| `vmov_n_f32` | Duplicate vector element to vector or scalar | "vdup.32" | — |
| `vmov_n_f64` | Duplicate vector element to vector or scalar | nop | — |
| `vmov_n_p16` | Duplicate vector element to vector or scalar | "vdup.16" | — |
| `vmov_n_p64` | Duplicate vector element to vector or scalar | fmov | — |
| `vmov_n_p8` | Duplicate vector element to vector or scalar | "vdup.8" | — |
| `vmov_n_s16` | Duplicate vector element to vector or scalar | "vdup.16" | — |
| `vmov_n_s32` | Duplicate vector element to vector or scalar | "vdup.32" | — |
| `vmov_n_s64` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vmov_n_s8` | Duplicate vector element to vector or scalar | "vdup.8" | — |
| `vmov_n_u16` | Duplicate vector element to vector or scalar | "vdup.16" | — |
| `vmov_n_u32` | Duplicate vector element to vector or scalar | "vdup.32" | — |
| `vmov_n_u64` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vmov_n_u8` | Duplicate vector element to vector or scalar | "vdup.8" | — |
| `vmovl_high_s16` | Vector move | sxtl2 | — |
| `vmovl_high_s32` | Vector move | sxtl2 | — |
| `vmovl_high_s8` | Vector move | sxtl2 | — |
| `vmovl_high_u16` | Vector move | uxtl2 | — |
| `vmovl_high_u32` | Vector move | uxtl2 | — |
| `vmovl_high_u8` | Vector move | uxtl2 | — |
| `vmovl_s16` | Vector long move | vmovl | — |
| `vmovl_s32` | Vector long move | vmovl | — |
| `vmovl_s8` | Vector long move | vmovl | — |
| `vmovl_u16` | Vector long move | vmovl | — |
| `vmovl_u32` | Vector long move | vmovl | — |
| `vmovl_u8` | Vector long move | vmovl | — |
| `vmovn_high_s16` | Extract narrow | xtn2 | — |
| `vmovn_high_s32` | Extract narrow | xtn2 | — |
| `vmovn_high_s64` | Extract narrow | xtn2 | — |
| `vmovn_high_u16` | Extract narrow | xtn2 | — |
| `vmovn_high_u32` | Extract narrow | xtn2 | — |
| `vmovn_high_u64` | Extract narrow | xtn2 | — |
| `vmovn_s16` | Vector narrow integer | vmovn | — |
| `vmovn_s32` | Vector narrow integer | vmovn | — |
| `vmovn_s64` | Vector narrow integer | vmovn | — |
| `vmovn_u16` | Vector narrow integer | vmovn | — |
| `vmovn_u32` | Vector narrow integer | vmovn | — |
| `vmovn_u64` | Vector narrow integer | vmovn | — |
| `vmovq_n_f32` | Duplicate vector element to vector or scalar | "vdup.32" | — |
| `vmovq_n_f64` | Duplicate vector element to vector or scalar | dup | — |
| `vmovq_n_p16` | Duplicate vector element to vector or scalar | "vdup.16" | — |
| `vmovq_n_p64` | Duplicate vector element to vector or scalar | dup | — |
| `vmovq_n_p8` | Duplicate vector element to vector or scalar | "vdup.8" | — |
| `vmovq_n_s16` | Duplicate vector element to vector or scalar | "vdup.16" | — |
| `vmovq_n_s32` | Duplicate vector element to vector or scalar | "vdup.32" | — |
| `vmovq_n_s64` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vmovq_n_s8` | Duplicate vector element to vector or scalar | "vdup.8" | — |
| `vmovq_n_u16` | Duplicate vector element to vector or scalar | "vdup.16" | — |
| `vmovq_n_u32` | Duplicate vector element to vector or scalar | "vdup.32" | — |
| `vmovq_n_u64` | Duplicate vector element to vector or scalar | "vmov" | — |
| `vmovq_n_u8` | Duplicate vector element to vector or scalar | "vdup.8" | — |
| `vmul_f32` | Multiply | "vmul.f32" | — |
| `vmul_f64` | Multiply | fmul | — |
| `vmul_lane_f32` | Floating-point multiply | vmul | — |
| `vmul_lane_f64` | Floating-point multiply | fmul | — |
| `vmul_lane_s16` | Multiply | vmul | — |
| `vmul_lane_s32` | Multiply | vmul | — |
| `vmul_lane_u16` | Multiply | vmul | — |
| `vmul_lane_u32` | Multiply | vmul | — |
| `vmul_laneq_f32` | Floating-point multiply | vmul | — |
| `vmul_laneq_f64` | Floating-point multiply | fmul | — |
| `vmul_laneq_s16` | Multiply | vmul | — |
| `vmul_laneq_s32` | Multiply | vmul | — |
| `vmul_laneq_u16` | Multiply | vmul | — |
| `vmul_laneq_u32` | Multiply | vmul | — |
| `vmul_n_f32` | Vector multiply by scalar | vmul | — |
| `vmul_n_f64` | Vector multiply by scalar | fmul | — |
| `vmul_n_s16` | Vector multiply by scalar | vmul | — |
| `vmul_n_s32` | Vector multiply by scalar | vmul | — |
| `vmul_n_u16` | Vector multiply by scalar | vmul | — |
| `vmul_n_u32` | Vector multiply by scalar | vmul | — |
| `vmul_p8` | Polynomial multiply | vmul | — |
| `vmul_s16` | Multiply | "vmul.i16" | — |
| `vmul_s32` | Multiply | "vmul.i32" | — |
| `vmul_s8` | Multiply | "vmul.i8" | — |
| `vmul_u16` | Multiply | "vmul.i16" | — |
| `vmul_u32` | Multiply | "vmul.i32" | — |
| `vmul_u8` | Multiply | "vmul.i8" | — |
| `vmuld_lane_f64` | Floating-point multiply | fmul | — |
| `vmuld_laneq_f64` | Floating-point multiply | fmul | — |
| `vmull_high_lane_s16` | Multiply long | smull2 | — |
| `vmull_high_lane_s32` | Multiply long | smull2 | — |
| `vmull_high_lane_u16` | Multiply long | umull2 | — |
| `vmull_high_lane_u32` | Multiply long | umull2 | — |
| `vmull_high_laneq_s16` | Multiply long | smull2 | — |
| `vmull_high_laneq_s32` | Multiply long | smull2 | — |
| `vmull_high_laneq_u16` | Multiply long | umull2 | — |
| `vmull_high_laneq_u32` | Multiply long | umull2 | — |
| `vmull_high_n_s16` | Multiply long | smull2 | — |
| `vmull_high_n_s32` | Multiply long | smull2 | — |
| `vmull_high_n_u16` | Multiply long | umull2 | — |
| `vmull_high_n_u32` | Multiply long | umull2 | — |
| `vmull_high_p8` | Polynomial multiply long | pmull2 | — |
| `vmull_high_s16` | Signed multiply long | smull2 | — |
| `vmull_high_s32` | Signed multiply long | smull2 | — |
| `vmull_high_s8` | Signed multiply long | smull2 | — |
| `vmull_high_u16` | Unsigned multiply long | umull2 | — |
| `vmull_high_u32` | Unsigned multiply long | umull2 | — |
| `vmull_high_u8` | Unsigned multiply long | umull2 | — |
| `vmull_lane_s16` | Vector long multiply by scalar | vmull | — |
| `vmull_lane_s32` | Vector long multiply by scalar | vmull | — |
| `vmull_lane_u16` | Vector long multiply by scalar | vmull | — |
| `vmull_lane_u32` | Vector long multiply by scalar | vmull | — |
| `vmull_laneq_s16` | Vector long multiply by scalar | vmull | — |
| `vmull_laneq_s32` | Vector long multiply by scalar | vmull | — |
| `vmull_laneq_u16` | Vector long multiply by scalar | vmull | — |
| `vmull_laneq_u32` | Vector long multiply by scalar | vmull | — |
| `vmull_n_s16` | Vector long multiply with scalar | vmull | — |
| `vmull_n_s32` | Vector long multiply with scalar | vmull | — |
| `vmull_n_u16` | Vector long multiply with scalar | vmull | — |
| `vmull_n_u32` | Vector long multiply with scalar | vmull | — |
| `vmull_p8` | Polynomial multiply long | "vmull.p8" | — |
| `vmull_s16` | Signed multiply long | "vmull.s16" | — |
| `vmull_s32` | Signed multiply long | "vmull.s32" | — |
| `vmull_s8` | Signed multiply long | "vmull.s8" | — |
| `vmull_u16` | Unsigned multiply long | "vmull.u16" | — |
| `vmull_u32` | Unsigned multiply long | "vmull.u32" | — |
| `vmull_u8` | Unsigned multiply long | "vmull.u8" | — |
| `vmulq_f32` | Multiply | "vmul.f32" | — |
| `vmulq_f64` | Multiply | fmul | — |
| `vmulq_lane_f32` | Floating-point multiply | vmul | — |
| `vmulq_lane_f64` | Floating-point multiply | fmul | — |
| `vmulq_lane_s16` | Multiply | vmul | — |
| `vmulq_lane_s32` | Multiply | vmul | — |
| `vmulq_lane_u16` | Multiply | vmul | — |
| `vmulq_lane_u32` | Multiply | vmul | — |
| `vmulq_laneq_f32` | Floating-point multiply | vmul | — |
| `vmulq_laneq_f64` | Floating-point multiply | fmul | — |
| `vmulq_laneq_s16` | Multiply | vmul | — |
| `vmulq_laneq_s32` | Multiply | vmul | — |
| `vmulq_laneq_u16` | Multiply | vmul | — |
| `vmulq_laneq_u32` | Multiply | vmul | — |
| `vmulq_n_f32` | Vector multiply by scalar | vmul | — |
| `vmulq_n_f64` | Vector multiply by scalar | fmul | — |
| `vmulq_n_s16` | Vector multiply by scalar | vmul | — |
| `vmulq_n_s32` | Vector multiply by scalar | vmul | — |
| `vmulq_n_u16` | Vector multiply by scalar | vmul | — |
| `vmulq_n_u32` | Vector multiply by scalar | vmul | — |
| `vmulq_p8` | Polynomial multiply | vmul | — |
| `vmulq_s16` | Multiply | "vmul.i16" | — |
| `vmulq_s32` | Multiply | "vmul.i32" | — |
| `vmulq_s8` | Multiply | "vmul.i8" | — |
| `vmulq_u16` | Multiply | "vmul.i16" | — |
| `vmulq_u32` | Multiply | "vmul.i32" | — |
| `vmulq_u8` | Multiply | "vmul.i8" | — |
| `vmuls_lane_f32` | Floating-point multiply | fmul | — |
| `vmuls_laneq_f32` | Floating-point multiply | fmul | — |
| `vmulx_f32` | Floating-point multiply extended | fmulx | — |
| `vmulx_f64` | Floating-point multiply extended | fmulx | — |
| `vmulx_lane_f32` | Floating-point multiply extended | fmulx | — |
| `vmulx_lane_f64` | Floating-point multiply extended | fmulx | — |
| `vmulx_laneq_f32` | Floating-point multiply extended | fmulx | — |
| `vmulx_laneq_f64` | Floating-point multiply extended | fmulx | — |
| `vmulxd_f64` | Floating-point multiply extended | fmulx | — |
| `vmulxd_lane_f64` | Floating-point multiply extended | fmulx | — |
| `vmulxd_laneq_f64` | Floating-point multiply extended | fmulx | — |
| `vmulxq_f32` | Floating-point multiply extended | fmulx | — |
| `vmulxq_f64` | Floating-point multiply extended | fmulx | — |
| `vmulxq_lane_f32` | Floating-point multiply extended | fmulx | — |
| `vmulxq_lane_f64` | Floating-point multiply extended | fmulx | — |
| `vmulxq_laneq_f32` | Floating-point multiply extended | fmulx | — |
| `vmulxq_laneq_f64` | Floating-point multiply extended | fmulx | — |
| `vmulxs_f32` | Floating-point multiply extended | fmulx | — |
| `vmulxs_lane_f32` | Floating-point multiply extended | fmulx | — |
| `vmulxs_laneq_f32` | Floating-point multiply extended | fmulx | — |
| `vmvn_p8` | Vector bitwise not | vmvn | — |
| `vmvn_s16` | Vector bitwise not | vmvn | — |
| `vmvn_s32` | Vector bitwise not | vmvn | — |
| `vmvn_s8` | Vector bitwise not | vmvn | — |
| `vmvn_u16` | Vector bitwise not | vmvn | — |
| `vmvn_u32` | Vector bitwise not | vmvn | — |
| `vmvn_u8` | Vector bitwise not | vmvn | — |
| `vmvnq_p8` | Vector bitwise not | vmvn | — |
| `vmvnq_s16` | Vector bitwise not | vmvn | — |
| `vmvnq_s32` | Vector bitwise not | vmvn | — |
| `vmvnq_s8` | Vector bitwise not | vmvn | — |
| `vmvnq_u16` | Vector bitwise not | vmvn | — |
| `vmvnq_u32` | Vector bitwise not | vmvn | — |
| `vmvnq_u8` | Vector bitwise not | vmvn | — |
| `vneg_f32` | Negate | "vneg.f32" | — |
| `vneg_f64` | Negate | fneg | — |
| `vneg_s16` | Negate | "vneg.s16" | — |
| `vneg_s32` | Negate | "vneg.s32" | — |
| `vneg_s64` | Negate | neg | — |
| `vneg_s8` | Negate | "vneg.s8" | — |
| `vnegd_s64` | Negate | neg | — |
| `vnegq_f32` | Negate | "vneg.f32" | — |
| `vnegq_f64` | Negate | fneg | — |
| `vnegq_s16` | Negate | "vneg.s16" | — |
| `vnegq_s32` | Negate | "vneg.s32" | — |
| `vnegq_s64` | Negate | neg | — |
| `vnegq_s8` | Negate | "vneg.s8" | — |
| `vorn_s16` | Vector bitwise inclusive OR NOT | vorn | — |
| `vorn_s32` | Vector bitwise inclusive OR NOT | vorn | — |
| `vorn_s64` | Vector bitwise inclusive OR NOT | vorn | — |
| `vorn_s8` | Vector bitwise inclusive OR NOT | vorn | — |
| `vorn_u16` | Vector bitwise inclusive OR NOT | vorn | — |
| `vorn_u32` | Vector bitwise inclusive OR NOT | vorn | — |
| `vorn_u64` | Vector bitwise inclusive OR NOT | vorn | — |
| `vorn_u8` | Vector bitwise inclusive OR NOT | vorn | — |
| `vornq_s16` | Vector bitwise inclusive OR NOT | vorn | — |
| `vornq_s32` | Vector bitwise inclusive OR NOT | vorn | — |
| `vornq_s64` | Vector bitwise inclusive OR NOT | vorn | — |
| `vornq_s8` | Vector bitwise inclusive OR NOT | vorn | — |
| `vornq_u16` | Vector bitwise inclusive OR NOT | vorn | — |
| `vornq_u32` | Vector bitwise inclusive OR NOT | vorn | — |
| `vornq_u64` | Vector bitwise inclusive OR NOT | vorn | — |
| `vornq_u8` | Vector bitwise inclusive OR NOT | vorn | — |
| `vorr_s16` | Vector bitwise or (immediate, inclusive) | vorr | — |
| `vorr_s32` | Vector bitwise or (immediate, inclusive) | vorr | — |
| `vorr_s64` | Vector bitwise or (immediate, inclusive) | vorr | — |
| `vorr_s8` | Vector bitwise or (immediate, inclusive) | vorr | — |
| `vorr_u16` | Vector bitwise or (immediate, inclusive) | vorr | — |
| `vorr_u32` | Vector bitwise or (immediate, inclusive) | vorr | — |
| `vorr_u64` | Vector bitwise or (immediate, inclusive) | vorr | — |
| `vorr_u8` | Vector bitwise or (immediate, inclusive) | vorr | — |
| `vorrq_s16` | Vector bitwise or (immediate, inclusive) | vorr | — |
| `vorrq_s32` | Vector bitwise or (immediate, inclusive) | vorr | — |
| `vorrq_s64` | Vector bitwise or (immediate, inclusive) | vorr | — |
| `vorrq_s8` | Vector bitwise or (immediate, inclusive) | vorr | — |
| `vorrq_u16` | Vector bitwise or (immediate, inclusive) | vorr | — |
| `vorrq_u32` | Vector bitwise or (immediate, inclusive) | vorr | — |
| `vorrq_u64` | Vector bitwise or (immediate, inclusive) | vorr | — |
| `vorrq_u8` | Vector bitwise or (immediate, inclusive) | vorr | — |
| `vpadal_s16` | Signed Add and Accumulate Long Pairwise | "vpadal.s16" | — |
| `vpadal_s32` | Signed Add and Accumulate Long Pairwise | "vpadal.s32" | — |
| `vpadal_s8` | Signed Add and Accumulate Long Pairwise | "vpadal.s8" | — |
| `vpadal_u16` | Unsigned Add and Accumulate Long Pairwise | "vpadal.u16" | — |
| `vpadal_u32` | Unsigned Add and Accumulate Long Pairwise | "vpadal.u32" | — |
| `vpadal_u8` | Unsigned Add and Accumulate Long Pairwise | "vpadal.u8" | — |
| `vpadalq_s16` | Signed Add and Accumulate Long Pairwise | "vpadal.s16" | — |
| `vpadalq_s32` | Signed Add and Accumulate Long Pairwise | "vpadal.s32" | — |
| `vpadalq_s8` | Signed Add and Accumulate Long Pairwise | "vpadal.s8" | — |
| `vpadalq_u16` | Unsigned Add and Accumulate Long Pairwise | "vpadal.u16" | — |
| `vpadalq_u32` | Unsigned Add and Accumulate Long Pairwise | "vpadal.u32" | — |
| `vpadalq_u8` | Unsigned Add and Accumulate Long Pairwise | "vpadal.u8" | — |
| `vpadd_f32` | Floating-point add pairwise | vpadd | — |
| `vpadd_s16` | Add pairwise | vpadd | — |
| `vpadd_s32` | Add pairwise | vpadd | — |
| `vpadd_s8` | Add pairwise | vpadd | — |
| `vpadd_u16` | Add pairwise | vpadd | — |
| `vpadd_u32` | Add pairwise | vpadd | — |
| `vpadd_u8` | Add pairwise | vpadd | — |
| `vpaddd_f64` | Floating-point add pairwise | nop | — |
| `vpaddd_s64` | Add pairwise | addp | — |
| `vpaddd_u64` | Add pairwise | addp | — |
| `vpaddl_s16` | Signed Add and Accumulate Long Pairwise | "vpaddl.s16" | — |
| `vpaddl_s32` | Signed Add and Accumulate Long Pairwise | "vpaddl.s32" | — |
| `vpaddl_s8` | Signed Add and Accumulate Long Pairwise | "vpaddl.s8" | — |
| `vpaddl_u16` | Unsigned Add and Accumulate Long Pairwise | "vpaddl.u16" | — |
| `vpaddl_u32` | Unsigned Add and Accumulate Long Pairwise | "vpaddl.u32" | — |
| `vpaddl_u8` | Unsigned Add and Accumulate Long Pairwise | "vpaddl.u8" | — |
| `vpaddlq_s16` | Signed Add and Accumulate Long Pairwise | "vpaddl.s16" | — |
| `vpaddlq_s32` | Signed Add and Accumulate Long Pairwise | "vpaddl.s32" | — |
| `vpaddlq_s8` | Signed Add and Accumulate Long Pairwise | "vpaddl.s8" | — |
| `vpaddlq_u16` | Unsigned Add and Accumulate Long Pairwise | "vpaddl.u16" | — |
| `vpaddlq_u32` | Unsigned Add and Accumulate Long Pairwise | "vpaddl.u32" | — |
| `vpaddlq_u8` | Unsigned Add and Accumulate Long Pairwise | "vpaddl.u8" | — |
| `vpaddq_f32` | Floating-point add pairwise | faddp | — |
| `vpaddq_f64` | Floating-point add pairwise | faddp | — |
| `vpaddq_s16` | Add Pairwise | addp | — |
| `vpaddq_s32` | Add Pairwise | addp | — |
| `vpaddq_s64` | Add Pairwise | addp | — |
| `vpaddq_s8` | Add Pairwise | addp | — |
| `vpaddq_u16` | Add Pairwise | addp | — |
| `vpaddq_u32` | Add Pairwise | addp | — |
| `vpaddq_u64` | Add Pairwise | addp | — |
| `vpaddq_u8` | Add Pairwise | addp | — |
| `vpadds_f32` | Floating-point add pairwise | nop | — |
| `vpmax_f32` | Folding maximum of adjacent pairs | vpmax | — |
| `vpmax_s16` | Folding maximum of adjacent pairs | vpmax | — |
| `vpmax_s32` | Folding maximum of adjacent pairs | vpmax | — |
| `vpmax_s8` | Folding maximum of adjacent pairs | vpmax | — |
| `vpmax_u16` | Folding maximum of adjacent pairs | vpmax | — |
| `vpmax_u32` | Folding maximum of adjacent pairs | vpmax | — |
| `vpmax_u8` | Folding maximum of adjacent pairs | vpmax | — |
| `vpmaxnm_f32` | Floating-point Maximum Number Pairwise (vector) | fmaxnmp | — |
| `vpmaxnmq_f32` | Floating-point Maximum Number Pairwise (vector) | fmaxnmp | — |
| `vpmaxnmq_f64` | Floating-point Maximum Number Pairwise (vector) | fmaxnmp | — |
| `vpmaxnmqd_f64` | Floating-point maximum number pairwise | fmaxnmp | — |
| `vpmaxnms_f32` | Floating-point maximum number pairwise | fmaxnmp | — |
| `vpmaxq_f32` | Folding maximum of adjacent pairs | fmaxp | — |
| `vpmaxq_f64` | Folding maximum of adjacent pairs | fmaxp | — |
| `vpmaxq_s16` | Folding maximum of adjacent pairs | smaxp | — |
| `vpmaxq_s32` | Folding maximum of adjacent pairs | smaxp | — |
| `vpmaxq_s8` | Folding maximum of adjacent pairs | smaxp | — |
| `vpmaxq_u16` | Folding maximum of adjacent pairs | umaxp | — |
| `vpmaxq_u32` | Folding maximum of adjacent pairs | umaxp | — |
| `vpmaxq_u8` | Folding maximum of adjacent pairs | umaxp | — |
| `vpmaxqd_f64` | Floating-point maximum pairwise | fmaxp | — |
| `vpmaxs_f32` | Floating-point maximum pairwise | fmaxp | — |
| `vpmin_f32` | Folding minimum of adjacent pairs | vpmin | — |
| `vpmin_s16` | Folding minimum of adjacent pairs | vpmin | — |
| `vpmin_s32` | Folding minimum of adjacent pairs | vpmin | — |
| `vpmin_s8` | Folding minimum of adjacent pairs | vpmin | — |
| `vpmin_u16` | Folding minimum of adjacent pairs | vpmin | — |
| `vpmin_u32` | Folding minimum of adjacent pairs | vpmin | — |
| `vpmin_u8` | Folding minimum of adjacent pairs | vpmin | — |
| `vpminnm_f32` | Floating-point Minimum Number Pairwise (vector) | fminnmp | — |
| `vpminnmq_f32` | Floating-point Minimum Number Pairwise (vector) | fminnmp | — |
| `vpminnmq_f64` | Floating-point Minimum Number Pairwise (vector) | fminnmp | — |
| `vpminnmqd_f64` | Floating-point minimum number pairwise | fminnmp | — |
| `vpminnms_f32` | Floating-point minimum number pairwise | fminnmp | — |
| `vpminq_f32` | Folding minimum of adjacent pairs | fminp | — |
| `vpminq_f64` | Folding minimum of adjacent pairs | fminp | — |
| `vpminq_s16` | Folding minimum of adjacent pairs | sminp | — |
| `vpminq_s32` | Folding minimum of adjacent pairs | sminp | — |
| `vpminq_s8` | Folding minimum of adjacent pairs | sminp | — |
| `vpminq_u16` | Folding minimum of adjacent pairs | uminp | — |
| `vpminq_u32` | Folding minimum of adjacent pairs | uminp | — |
| `vpminq_u8` | Folding minimum of adjacent pairs | uminp | — |
| `vpminqd_f64` | Floating-point minimum pairwise | fminp | — |
| `vpmins_f32` | Floating-point minimum pairwise | fminp | — |
| `vqabs_s16` | Signed saturating Absolute value | "vqabs.s16" | — |
| `vqabs_s32` | Signed saturating Absolute value | "vqabs.s32" | — |
| `vqabs_s64` | Signed saturating Absolute value | sqabs | — |
| `vqabs_s8` | Signed saturating Absolute value | "vqabs.s8" | — |
| `vqabsb_s8` | Signed saturating absolute value | sqabs | — |
| `vqabsd_s64` | Signed saturating absolute value | sqabs | — |
| `vqabsh_s16` | Signed saturating absolute value | sqabs | — |
| `vqabsq_s16` | Signed saturating Absolute value | "vqabs.s16" | — |
| `vqabsq_s32` | Signed saturating Absolute value | "vqabs.s32" | — |
| `vqabsq_s64` | Signed saturating Absolute value | sqabs | — |
| `vqabsq_s8` | Signed saturating Absolute value | "vqabs.s8" | — |
| `vqabss_s32` | Signed saturating absolute value | sqabs | — |
| `vqadd_s16` | Saturating add | "vqadd.s16" | — |
| `vqadd_s32` | Saturating add | "vqadd.s32" | — |
| `vqadd_s64` | Saturating add | "vqadd.s64" | — |
| `vqadd_s8` | Saturating add | "vqadd.s8" | — |
| `vqadd_u16` | Saturating add | "vqadd.u16" | — |
| `vqadd_u32` | Saturating add | "vqadd.u32" | — |
| `vqadd_u64` | Saturating add | "vqadd.u64" | — |
| `vqadd_u8` | Saturating add | "vqadd.u8" | — |
| `vqaddb_s8` | Saturating add | sqadd | — |
| `vqaddb_u8` | Saturating add | uqadd | — |
| `vqaddd_s64` | Saturating add | sqadd | — |
| `vqaddd_u64` | Saturating add | uqadd | — |
| `vqaddh_s16` | Saturating add | sqadd | — |
| `vqaddh_u16` | Saturating add | uqadd | — |
| `vqaddq_s16` | Saturating add | "vqadd.s16" | — |
| `vqaddq_s32` | Saturating add | "vqadd.s32" | — |
| `vqaddq_s64` | Saturating add | "vqadd.s64" | — |
| `vqaddq_s8` | Saturating add | "vqadd.s8" | — |
| `vqaddq_u16` | Saturating add | "vqadd.u16" | — |
| `vqaddq_u32` | Saturating add | "vqadd.u32" | — |
| `vqaddq_u64` | Saturating add | "vqadd.u64" | — |
| `vqaddq_u8` | Saturating add | "vqadd.u8" | — |
| `vqadds_s32` | Saturating add | sqadd | — |
| `vqadds_u32` | Saturating add | uqadd | — |
| `vqdmlal_high_lane_s16` | Signed saturating doubling multiply-add long | sqdmlal2 | — |
| `vqdmlal_high_lane_s32` | Signed saturating doubling multiply-add long | sqdmlal2 | — |
| `vqdmlal_high_laneq_s16` | Signed saturating doubling multiply-add long | sqdmlal2 | — |
| `vqdmlal_high_laneq_s32` | Signed saturating doubling multiply-add long | sqdmlal2 | — |
| `vqdmlal_high_n_s16` | Signed saturating doubling multiply-add long | sqdmlal2 | — |
| `vqdmlal_high_n_s32` | Signed saturating doubling multiply-add long | sqdmlal2 | — |
| `vqdmlal_high_s16` | Signed saturating doubling multiply-add long | sqdmlal2 | — |
| `vqdmlal_high_s32` | Signed saturating doubling multiply-add long | sqdmlal2 | — |
| `vqdmlal_lane_s16` | Vector widening saturating doubling multiply accumulate with... | vqdmlal | — |
| `vqdmlal_lane_s32` | Vector widening saturating doubling multiply accumulate with... | vqdmlal | — |
| `vqdmlal_laneq_s16` | Vector widening saturating doubling multiply accumulate with... | sqdmlal | — |
| `vqdmlal_laneq_s32` | Vector widening saturating doubling multiply accumulate with... | sqdmlal | — |
| `vqdmlal_n_s16` | Vector widening saturating doubling multiply accumulate with... | vqdmlal | — |
| `vqdmlal_n_s32` | Vector widening saturating doubling multiply accumulate with... | vqdmlal | — |
| `vqdmlal_s16` | Signed saturating doubling multiply-add long | vqdmlal | — |
| `vqdmlal_s32` | Signed saturating doubling multiply-add long | vqdmlal | — |
| `vqdmlalh_lane_s16` | Signed saturating doubling multiply-add long | sqdmlal | — |
| `vqdmlalh_laneq_s16` | Signed saturating doubling multiply-add long | sqdmlal | — |
| `vqdmlalh_s16` | Signed saturating doubling multiply-add long | sqdmlal | — |
| `vqdmlals_lane_s32` | Signed saturating doubling multiply-add long | sqdmlal | — |
| `vqdmlals_laneq_s32` | Signed saturating doubling multiply-add long | sqdmlal | — |
| `vqdmlals_s32` | Signed saturating doubling multiply-add long | sqdmlal | — |
| `vqdmlsl_high_lane_s16` | Signed saturating doubling multiply-subtract long | sqdmlsl2 | — |
| `vqdmlsl_high_lane_s32` | Signed saturating doubling multiply-subtract long | sqdmlsl2 | — |
| `vqdmlsl_high_laneq_s16` | Signed saturating doubling multiply-subtract long | sqdmlsl2 | — |
| `vqdmlsl_high_laneq_s32` | Signed saturating doubling multiply-subtract long | sqdmlsl2 | — |
| `vqdmlsl_high_n_s16` | Signed saturating doubling multiply-subtract long | sqdmlsl2 | — |
| `vqdmlsl_high_n_s32` | Signed saturating doubling multiply-subtract long | sqdmlsl2 | — |
| `vqdmlsl_high_s16` | Signed saturating doubling multiply-subtract long | sqdmlsl2 | — |
| `vqdmlsl_high_s32` | Signed saturating doubling multiply-subtract long | sqdmlsl2 | — |
| `vqdmlsl_lane_s16` | Vector widening saturating doubling multiply subtract with s... | vqdmlsl | — |
| `vqdmlsl_lane_s32` | Vector widening saturating doubling multiply subtract with s... | vqdmlsl | — |
| `vqdmlsl_laneq_s16` | Vector widening saturating doubling multiply subtract with s... | sqdmlsl | — |
| `vqdmlsl_laneq_s32` | Vector widening saturating doubling multiply subtract with s... | sqdmlsl | — |
| `vqdmlsl_n_s16` | Vector widening saturating doubling multiply subtract with s... | vqdmlsl | — |
| `vqdmlsl_n_s32` | Vector widening saturating doubling multiply subtract with s... | vqdmlsl | — |
| `vqdmlsl_s16` | Signed saturating doubling multiply-subtract long | vqdmlsl | — |
| `vqdmlsl_s32` | Signed saturating doubling multiply-subtract long | vqdmlsl | — |
| `vqdmlslh_lane_s16` | Signed saturating doubling multiply-subtract long | sqdmlsl | — |
| `vqdmlslh_laneq_s16` | Signed saturating doubling multiply-subtract long | sqdmlsl | — |
| `vqdmlslh_s16` | Signed saturating doubling multiply-subtract long | sqdmlsl | — |
| `vqdmlsls_lane_s32` | Signed saturating doubling multiply-subtract long | sqdmlsl | — |
| `vqdmlsls_laneq_s32` | Signed saturating doubling multiply-subtract long | sqdmlsl | — |
| `vqdmlsls_s32` | Signed saturating doubling multiply-subtract long | sqdmlsl | — |
| `vqdmulh_lane_s16` | Vector saturating doubling multiply high by scalar | sqdmulh | — |
| `vqdmulh_lane_s32` | Vector saturating doubling multiply high by scalar | sqdmulh | — |
| `vqdmulh_laneq_s16` | Vector saturating doubling multiply high by scalar | vqdmulh | — |
| `vqdmulh_laneq_s32` | Vector saturating doubling multiply high by scalar | vqdmulh | — |
| `vqdmulh_n_s16` | Vector saturating doubling multiply high with scalar | vqdmulh | — |
| `vqdmulh_n_s32` | Vector saturating doubling multiply high with scalar | vqdmulh | — |
| `vqdmulh_s16` | Signed saturating doubling multiply returning high half | vqdmulh | — |
| `vqdmulh_s32` | Signed saturating doubling multiply returning high half | vqdmulh | — |
| `vqdmulhh_lane_s16` | Signed saturating doubling multiply returning high half | sqdmulh | — |
| `vqdmulhh_laneq_s16` | Signed saturating doubling multiply returning high half | sqdmulh | — |
| `vqdmulhh_s16` | Signed saturating doubling multiply returning high half | sqdmulh | — |
| `vqdmulhq_lane_s16` | Vector saturating doubling multiply high by scalar | sqdmulh | — |
| `vqdmulhq_lane_s32` | Vector saturating doubling multiply high by scalar | sqdmulh | — |
| `vqdmulhq_laneq_s16` | Vector saturating doubling multiply high by scalar | vqdmulh | — |
| `vqdmulhq_laneq_s32` | Vector saturating doubling multiply high by scalar | vqdmulh | — |
| `vqdmulhq_n_s16` | Vector saturating doubling multiply high with scalar | vqdmulh | — |
| `vqdmulhq_n_s32` | Vector saturating doubling multiply high with scalar | vqdmulh | — |
| `vqdmulhq_s16` | Signed saturating doubling multiply returning high half | vqdmulh | — |
| `vqdmulhq_s32` | Signed saturating doubling multiply returning high half | vqdmulh | — |
| `vqdmulhs_lane_s32` | Signed saturating doubling multiply returning high half | sqdmulh | — |
| `vqdmulhs_laneq_s32` | Signed saturating doubling multiply returning high half | sqdmulh | — |
| `vqdmulhs_s32` | Signed saturating doubling multiply returning high half | sqdmulh | — |
| `vqdmull_high_lane_s16` | Signed saturating doubling multiply long | sqdmull2 | — |
| `vqdmull_high_lane_s32` | Signed saturating doubling multiply long | sqdmull2 | — |
| `vqdmull_high_laneq_s16` | Signed saturating doubling multiply long | sqdmull2 | — |
| `vqdmull_high_laneq_s32` | Signed saturating doubling multiply long | sqdmull2 | — |
| `vqdmull_high_n_s16` | Signed saturating doubling multiply long | sqdmull2 | — |
| `vqdmull_high_n_s32` | Signed saturating doubling multiply long | sqdmull2 | — |
| `vqdmull_high_s16` | Signed saturating doubling multiply long | sqdmull2 | — |
| `vqdmull_high_s32` | Signed saturating doubling multiply long | sqdmull2 | — |
| `vqdmull_lane_s16` | Vector saturating doubling long multiply by scalar | vqdmull | — |
| `vqdmull_lane_s32` | Vector saturating doubling long multiply by scalar | vqdmull | — |
| `vqdmull_laneq_s16` | Vector saturating doubling long multiply by scalar | sqdmull | — |
| `vqdmull_laneq_s32` | Vector saturating doubling long multiply by scalar | sqdmull | — |
| `vqdmull_n_s16` | Vector saturating doubling long multiply with scalar | vqdmull | — |
| `vqdmull_n_s32` | Vector saturating doubling long multiply with scalar | vqdmull | — |
| `vqdmull_s16` | Signed saturating doubling multiply long | vqdmull | — |
| `vqdmull_s32` | Signed saturating doubling multiply long | vqdmull | — |
| `vqdmullh_lane_s16` | Signed saturating doubling multiply long | sqdmull | — |
| `vqdmullh_laneq_s16` | Signed saturating doubling multiply long | sqdmull | — |
| `vqdmullh_s16` | Signed saturating doubling multiply long | sqdmull | — |
| `vqdmulls_lane_s32` | Signed saturating doubling multiply long | sqdmull | — |
| `vqdmulls_laneq_s32` | Signed saturating doubling multiply long | sqdmull | — |
| `vqdmulls_s32` | Signed saturating doubling multiply long | sqdmull | — |
| `vqmovn_high_s16` | Signed saturating extract narrow | sqxtn2 | — |
| `vqmovn_high_s32` | Signed saturating extract narrow | sqxtn2 | — |
| `vqmovn_high_s64` | Signed saturating extract narrow | sqxtn2 | — |
| `vqmovn_high_u16` | Signed saturating extract narrow | uqxtn2 | — |
| `vqmovn_high_u32` | Signed saturating extract narrow | uqxtn2 | — |
| `vqmovn_high_u64` | Signed saturating extract narrow | uqxtn2 | — |
| `vqmovn_s16` | Signed saturating extract narrow | vqmovn | — |
| `vqmovn_s32` | Signed saturating extract narrow | vqmovn | — |
| `vqmovn_s64` | Signed saturating extract narrow | vqmovn | — |
| `vqmovn_u16` | Unsigned saturating extract narrow | vqmovn | — |
| `vqmovn_u32` | Unsigned saturating extract narrow | vqmovn | — |
| `vqmovn_u64` | Unsigned saturating extract narrow | vqmovn | — |
| `vqmovnd_s64` | Saturating extract narrow | sqxtn | — |
| `vqmovnd_u64` | Saturating extract narrow | uqxtn | — |
| `vqmovnh_s16` | Saturating extract narrow | sqxtn | — |
| `vqmovnh_u16` | Saturating extract narrow | uqxtn | — |
| `vqmovns_s32` | Saturating extract narrow | sqxtn | — |
| `vqmovns_u32` | Saturating extract narrow | uqxtn | — |
| `vqmovun_high_s16` | Signed saturating extract unsigned narrow | sqxtun2 | — |
| `vqmovun_high_s32` | Signed saturating extract unsigned narrow | sqxtun2 | — |
| `vqmovun_high_s64` | Signed saturating extract unsigned narrow | sqxtun2 | — |
| `vqmovun_s16` | Signed saturating extract unsigned narrow | vqmovun | — |
| `vqmovun_s32` | Signed saturating extract unsigned narrow | vqmovun | — |
| `vqmovun_s64` | Signed saturating extract unsigned narrow | vqmovun | — |
| `vqmovund_s64` | Signed saturating extract unsigned narrow | sqxtun | — |
| `vqmovunh_s16` | Signed saturating extract unsigned narrow | sqxtun | — |
| `vqmovuns_s32` | Signed saturating extract unsigned narrow | sqxtun | — |
| `vqneg_s16` | Signed saturating negate | "vqneg.s16" | — |
| `vqneg_s32` | Signed saturating negate | "vqneg.s32" | — |
| `vqneg_s64` | Signed saturating negate | sqneg | — |
| `vqneg_s8` | Signed saturating negate | "vqneg.s8" | — |
| `vqnegb_s8` | Signed saturating negate | sqneg | — |
| `vqnegd_s64` | Signed saturating negate | sqneg | — |
| `vqnegh_s16` | Signed saturating negate | sqneg | — |
| `vqnegq_s16` | Signed saturating negate | "vqneg.s16" | — |
| `vqnegq_s32` | Signed saturating negate | "vqneg.s32" | — |
| `vqnegq_s64` | Signed saturating negate | sqneg | — |
| `vqnegq_s8` | Signed saturating negate | "vqneg.s8" | — |
| `vqnegs_s32` | Signed saturating negate | sqneg | — |
| `vqrdmulh_lane_s16` | Vector rounding saturating doubling multiply high by scalar | vqrdmulh | — |
| `vqrdmulh_lane_s32` | Vector rounding saturating doubling multiply high by scalar | vqrdmulh | — |
| `vqrdmulh_laneq_s16` | Vector rounding saturating doubling multiply high by scalar | vqrdmulh | — |
| `vqrdmulh_laneq_s32` | Vector rounding saturating doubling multiply high by scalar | vqrdmulh | — |
| `vqrdmulh_n_s16` | Vector saturating rounding doubling multiply high with scala... | vqrdmulh | — |
| `vqrdmulh_n_s32` | Vector saturating rounding doubling multiply high with scala... | vqrdmulh | — |
| `vqrdmulh_s16` | Signed saturating rounding doubling multiply returning high ... | vqrdmulh | — |
| `vqrdmulh_s32` | Signed saturating rounding doubling multiply returning high ... | vqrdmulh | — |
| `vqrdmulhh_lane_s16` | Signed saturating rounding doubling multiply returning high ... | sqrdmulh | — |
| `vqrdmulhh_laneq_s16` | Signed saturating rounding doubling multiply returning high ... | sqrdmulh | — |
| `vqrdmulhh_s16` | Signed saturating rounding doubling multiply returning high ... | sqrdmulh | — |
| `vqrdmulhq_lane_s16` | Vector rounding saturating doubling multiply high by scalar | vqrdmulh | — |
| `vqrdmulhq_lane_s32` | Vector rounding saturating doubling multiply high by scalar | vqrdmulh | — |
| `vqrdmulhq_laneq_s16` | Vector rounding saturating doubling multiply high by scalar | vqrdmulh | — |
| `vqrdmulhq_laneq_s32` | Vector rounding saturating doubling multiply high by scalar | vqrdmulh | — |
| `vqrdmulhq_n_s16` | Vector saturating rounding doubling multiply high with scala... | vqrdmulh | — |
| `vqrdmulhq_n_s32` | Vector saturating rounding doubling multiply high with scala... | vqrdmulh | — |
| `vqrdmulhq_s16` | Signed saturating rounding doubling multiply returning high ... | vqrdmulh | — |
| `vqrdmulhq_s32` | Signed saturating rounding doubling multiply returning high ... | vqrdmulh | — |
| `vqrdmulhs_lane_s32` | Signed saturating rounding doubling multiply returning high ... | sqrdmulh | — |
| `vqrdmulhs_laneq_s32` | Signed saturating rounding doubling multiply returning high ... | sqrdmulh | — |
| `vqrdmulhs_s32` | Signed saturating rounding doubling multiply returning high ... | sqrdmulh | — |
| `vqrshl_s16` | Signed saturating rounding shift left | vqrshl | — |
| `vqrshl_s32` | Signed saturating rounding shift left | vqrshl | — |
| `vqrshl_s64` | Signed saturating rounding shift left | vqrshl | — |
| `vqrshl_s8` | Signed saturating rounding shift left | vqrshl | — |
| `vqrshl_u16` | Unsigned signed saturating rounding shift left | vqrshl | — |
| `vqrshl_u32` | Unsigned signed saturating rounding shift left | vqrshl | — |
| `vqrshl_u64` | Unsigned signed saturating rounding shift left | vqrshl | — |
| `vqrshl_u8` | Unsigned signed saturating rounding shift left | vqrshl | — |
| `vqrshlb_s8` | Signed saturating rounding shift left | sqrshl | — |
| `vqrshlb_u8` | Unsigned signed saturating rounding shift left | uqrshl | — |
| `vqrshld_s64` | Signed saturating rounding shift left | sqrshl | — |
| `vqrshld_u64` | Unsigned signed saturating rounding shift left | uqrshl | — |
| `vqrshlh_s16` | Signed saturating rounding shift left | sqrshl | — |
| `vqrshlh_u16` | Unsigned signed saturating rounding shift left | uqrshl | — |
| `vqrshlq_s16` | Signed saturating rounding shift left | vqrshl | — |
| `vqrshlq_s32` | Signed saturating rounding shift left | vqrshl | — |
| `vqrshlq_s64` | Signed saturating rounding shift left | vqrshl | — |
| `vqrshlq_s8` | Signed saturating rounding shift left | vqrshl | — |
| `vqrshlq_u16` | Unsigned signed saturating rounding shift left | vqrshl | — |
| `vqrshlq_u32` | Unsigned signed saturating rounding shift left | vqrshl | — |
| `vqrshlq_u64` | Unsigned signed saturating rounding shift left | vqrshl | — |
| `vqrshlq_u8` | Unsigned signed saturating rounding shift left | vqrshl | — |
| `vqrshls_s32` | Signed saturating rounding shift left | sqrshl | — |
| `vqrshls_u32` | Unsigned signed saturating rounding shift left | uqrshl | — |
| `vqrshrn_high_n_s16` | Signed saturating rounded shift right narrow | sqrshrn2 | — |
| `vqrshrn_high_n_s32` | Signed saturating rounded shift right narrow | sqrshrn2 | — |
| `vqrshrn_high_n_s64` | Signed saturating rounded shift right narrow | sqrshrn2 | — |
| `vqrshrn_high_n_u16` | Unsigned saturating rounded shift right narrow | uqrshrn2 | — |
| `vqrshrn_high_n_u32` | Unsigned saturating rounded shift right narrow | uqrshrn2 | — |
| `vqrshrn_high_n_u64` | Unsigned saturating rounded shift right narrow | uqrshrn2 | — |
| `vqrshrnd_n_s64` | Signed saturating rounded shift right narrow | sqrshrn | — |
| `vqrshrnd_n_u64` | Unsigned saturating rounded shift right narrow | uqrshrn | — |
| `vqrshrnh_n_s16` | Signed saturating rounded shift right narrow | sqrshrn | — |
| `vqrshrnh_n_u16` | Unsigned saturating rounded shift right narrow | uqrshrn | — |
| `vqrshrns_n_s32` | Signed saturating rounded shift right narrow | sqrshrn | — |
| `vqrshrns_n_u32` | Unsigned saturating rounded shift right narrow | uqrshrn | — |
| `vqrshrun_high_n_s16` | Signed saturating rounded shift right unsigned narrow | sqrshrun2 | — |
| `vqrshrun_high_n_s32` | Signed saturating rounded shift right unsigned narrow | sqrshrun2 | — |
| `vqrshrun_high_n_s64` | Signed saturating rounded shift right unsigned narrow | sqrshrun2 | — |
| `vqrshrund_n_s64` | Signed saturating rounded shift right unsigned narrow | sqrshrun | — |
| `vqrshrunh_n_s16` | Signed saturating rounded shift right unsigned narrow | sqrshrun | — |
| `vqrshruns_n_s32` | Signed saturating rounded shift right unsigned narrow | sqrshrun | — |
| `vqshl_n_s16` | Signed saturating shift left | vqshl | — |
| `vqshl_n_s32` | Signed saturating shift left | vqshl | — |
| `vqshl_n_s64` | Signed saturating shift left | vqshl | — |
| `vqshl_n_s8` | Signed saturating shift left | vqshl | — |
| `vqshl_n_u16` | Unsigned saturating shift left | vqshl | — |
| `vqshl_n_u32` | Unsigned saturating shift left | vqshl | — |
| `vqshl_n_u64` | Unsigned saturating shift left | vqshl | — |
| `vqshl_n_u8` | Unsigned saturating shift left | vqshl | — |
| `vqshl_s16` | Signed saturating shift left | vqshl | — |
| `vqshl_s32` | Signed saturating shift left | vqshl | — |
| `vqshl_s64` | Signed saturating shift left | vqshl | — |
| `vqshl_s8` | Signed saturating shift left | vqshl | — |
| `vqshl_u16` | Unsigned saturating shift left | vqshl | — |
| `vqshl_u32` | Unsigned saturating shift left | vqshl | — |
| `vqshl_u64` | Unsigned saturating shift left | vqshl | — |
| `vqshl_u8` | Unsigned saturating shift left | vqshl | — |
| `vqshlb_n_s8` | Signed saturating shift left | sqshl | — |
| `vqshlb_n_u8` | Unsigned saturating shift left | uqshl | — |
| `vqshlb_s8` | Signed saturating shift left | sqshl | — |
| `vqshlb_u8` | Unsigned saturating shift left | uqshl | — |
| `vqshld_n_s64` | Signed saturating shift left | sqshl | — |
| `vqshld_n_u64` | Unsigned saturating shift left | uqshl | — |
| `vqshld_s64` | Signed saturating shift left | sqshl | — |
| `vqshld_u64` | Unsigned saturating shift left | uqshl | — |
| `vqshlh_n_s16` | Signed saturating shift left | sqshl | — |
| `vqshlh_n_u16` | Unsigned saturating shift left | uqshl | — |
| `vqshlh_s16` | Signed saturating shift left | sqshl | — |
| `vqshlh_u16` | Unsigned saturating shift left | uqshl | — |
| `vqshlq_n_s16` | Signed saturating shift left | vqshl | — |
| `vqshlq_n_s32` | Signed saturating shift left | vqshl | — |
| `vqshlq_n_s64` | Signed saturating shift left | vqshl | — |
| `vqshlq_n_s8` | Signed saturating shift left | vqshl | — |
| `vqshlq_n_u16` | Unsigned saturating shift left | vqshl | — |
| `vqshlq_n_u32` | Unsigned saturating shift left | vqshl | — |
| `vqshlq_n_u64` | Unsigned saturating shift left | vqshl | — |
| `vqshlq_n_u8` | Unsigned saturating shift left | vqshl | — |
| `vqshlq_s16` | Signed saturating shift left | vqshl | — |
| `vqshlq_s32` | Signed saturating shift left | vqshl | — |
| `vqshlq_s64` | Signed saturating shift left | vqshl | — |
| `vqshlq_s8` | Signed saturating shift left | vqshl | — |
| `vqshlq_u16` | Unsigned saturating shift left | vqshl | — |
| `vqshlq_u32` | Unsigned saturating shift left | vqshl | — |
| `vqshlq_u64` | Unsigned saturating shift left | vqshl | — |
| `vqshlq_u8` | Unsigned saturating shift left | vqshl | — |
| `vqshls_n_s32` | Signed saturating shift left | sqshl | — |
| `vqshls_n_u32` | Unsigned saturating shift left | uqshl | — |
| `vqshls_s32` | Signed saturating shift left | sqshl | — |
| `vqshls_u32` | Unsigned saturating shift left | uqshl | — |
| `vqshlub_n_s8` | Signed saturating shift left unsigned | sqshlu | — |
| `vqshlud_n_s64` | Signed saturating shift left unsigned | sqshlu | — |
| `vqshluh_n_s16` | Signed saturating shift left unsigned | sqshlu | — |
| `vqshlus_n_s32` | Signed saturating shift left unsigned | sqshlu | — |
| `vqshrn_high_n_s16` | Signed saturating shift right narrow | sqshrn2 | — |
| `vqshrn_high_n_s32` | Signed saturating shift right narrow | sqshrn2 | — |
| `vqshrn_high_n_s64` | Signed saturating shift right narrow | sqshrn2 | — |
| `vqshrn_high_n_u16` | Unsigned saturating shift right narrow | uqshrn2 | — |
| `vqshrn_high_n_u32` | Unsigned saturating shift right narrow | uqshrn2 | — |
| `vqshrn_high_n_u64` | Unsigned saturating shift right narrow | uqshrn2 | — |
| `vqshrnd_n_s64` | Signed saturating shift right narrow | sqshrn | — |
| `vqshrnd_n_u64` | Unsigned saturating shift right narrow | uqshrn | — |
| `vqshrnh_n_s16` | Signed saturating shift right narrow | sqshrn | — |
| `vqshrnh_n_u16` | Unsigned saturating shift right narrow | uqshrn | — |
| `vqshrns_n_s32` | Signed saturating shift right narrow | sqshrn | — |
| `vqshrns_n_u32` | Unsigned saturating shift right narrow | uqshrn | — |
| `vqshrun_high_n_s16` | Signed saturating shift right unsigned narrow | sqshrun2 | — |
| `vqshrun_high_n_s32` | Signed saturating shift right unsigned narrow | sqshrun2 | — |
| `vqshrun_high_n_s64` | Signed saturating shift right unsigned narrow | sqshrun2 | — |
| `vqshrund_n_s64` | Signed saturating shift right unsigned narrow | sqshrun | — |
| `vqshrunh_n_s16` | Signed saturating shift right unsigned narrow | sqshrun | — |
| `vqshruns_n_s32` | Signed saturating shift right unsigned narrow | sqshrun | — |
| `vqsub_s16` | Saturating subtract | "vqsub.s16" | — |
| `vqsub_s32` | Saturating subtract | "vqsub.s32" | — |
| `vqsub_s64` | Saturating subtract | "vqsub.s64" | — |
| `vqsub_s8` | Saturating subtract | "vqsub.s8" | — |
| `vqsub_u16` | Saturating subtract | "vqsub.u16" | — |
| `vqsub_u32` | Saturating subtract | "vqsub.u32" | — |
| `vqsub_u64` | Saturating subtract | "vqsub.u64" | — |
| `vqsub_u8` | Saturating subtract | "vqsub.u8" | — |
| `vqsubb_s8` | Saturating subtract | sqsub | — |
| `vqsubb_u8` | Saturating subtract | uqsub | — |
| `vqsubd_s64` | Saturating subtract | sqsub | — |
| `vqsubd_u64` | Saturating subtract | uqsub | — |
| `vqsubh_s16` | Saturating subtract | sqsub | — |
| `vqsubh_u16` | Saturating subtract | uqsub | — |
| `vqsubq_s16` | Saturating subtract | "vqsub.s16" | — |
| `vqsubq_s32` | Saturating subtract | "vqsub.s32" | — |
| `vqsubq_s64` | Saturating subtract | "vqsub.s64" | — |
| `vqsubq_s8` | Saturating subtract | "vqsub.s8" | — |
| `vqsubq_u16` | Saturating subtract | "vqsub.u16" | — |
| `vqsubq_u32` | Saturating subtract | "vqsub.u32" | — |
| `vqsubq_u64` | Saturating subtract | "vqsub.u64" | — |
| `vqsubq_u8` | Saturating subtract | "vqsub.u8" | — |
| `vqsubs_s32` | Saturating subtract | sqsub | — |
| `vqsubs_u32` | Saturating subtract | uqsub | — |
| `vqtbl1_p8` | Table look-up | tbl | — |
| `vqtbl1_s8` | Table look-up | tbl | — |
| `vqtbl1_u8` | Table look-up | tbl | — |
| `vqtbl1q_p8` | Table look-up | tbl | — |
| `vqtbl1q_s8` | Table look-up | tbl | — |
| `vqtbl1q_u8` | Table look-up | tbl | — |
| `vqtbl2_p8` | Table look-up | tbl | — |
| `vqtbl2_s8` | Table look-up | tbl | — |
| `vqtbl2_u8` | Table look-up | tbl | — |
| `vqtbl2q_p8` | Table look-up | tbl | — |
| `vqtbl2q_s8` | Table look-up | tbl | — |
| `vqtbl2q_u8` | Table look-up | tbl | — |
| `vqtbl3_p8` | Table look-up | tbl | — |
| `vqtbl3_s8` | Table look-up | tbl | — |
| `vqtbl3_u8` | Table look-up | tbl | — |
| `vqtbl3q_p8` | Table look-up | tbl | — |
| `vqtbl3q_s8` | Table look-up | tbl | — |
| `vqtbl3q_u8` | Table look-up | tbl | — |
| `vqtbl4_p8` | Table look-up | tbl | — |
| `vqtbl4_s8` | Table look-up | tbl | — |
| `vqtbl4_u8` | Table look-up | tbl | — |
| `vqtbl4q_p8` | Table look-up | tbl | — |
| `vqtbl4q_s8` | Table look-up | tbl | — |
| `vqtbl4q_u8` | Table look-up | tbl | — |
| `vqtbx1_p8` | Extended table look-up | tbx | — |
| `vqtbx1_s8` | Extended table look-up | tbx | — |
| `vqtbx1_u8` | Extended table look-up | tbx | — |
| `vqtbx1q_p8` | Extended table look-up | tbx | — |
| `vqtbx1q_s8` | Extended table look-up | tbx | — |
| `vqtbx1q_u8` | Extended table look-up | tbx | — |
| `vqtbx2_p8` | Extended table look-up | tbx | — |
| `vqtbx2_s8` | Extended table look-up | tbx | — |
| `vqtbx2_u8` | Extended table look-up | tbx | — |
| `vqtbx2q_p8` | Extended table look-up | tbx | — |
| `vqtbx2q_s8` | Extended table look-up | tbx | — |
| `vqtbx2q_u8` | Extended table look-up | tbx | — |
| `vqtbx3_p8` | Extended table look-up | tbx | — |
| `vqtbx3_s8` | Extended table look-up | tbx | — |
| `vqtbx3_u8` | Extended table look-up | tbx | — |
| `vqtbx3q_p8` | Extended table look-up | tbx | — |
| `vqtbx3q_s8` | Extended table look-up | tbx | — |
| `vqtbx3q_u8` | Extended table look-up | tbx | — |
| `vqtbx4_p8` | Extended table look-up | tbx | — |
| `vqtbx4_s8` | Extended table look-up | tbx | — |
| `vqtbx4_u8` | Extended table look-up | tbx | — |
| `vqtbx4q_p8` | Extended table look-up | tbx | — |
| `vqtbx4q_s8` | Extended table look-up | tbx | — |
| `vqtbx4q_u8` | Extended table look-up | tbx | — |
| `vraddhn_high_s16` | Rounding Add returning High Narrow (high half) | "vraddhn.i16" | — |
| `vraddhn_high_s32` | Rounding Add returning High Narrow (high half) | "vraddhn.i32" | — |
| `vraddhn_high_s64` | Rounding Add returning High Narrow (high half) | "vraddhn.i64" | — |
| `vraddhn_high_u16` | Rounding Add returning High Narrow (high half) | "vraddhn.i16" | — |
| `vraddhn_high_u32` | Rounding Add returning High Narrow (high half) | "vraddhn.i32" | — |
| `vraddhn_high_u64` | Rounding Add returning High Narrow (high half) | "vraddhn.i64" | — |
| `vraddhn_s16` | Rounding Add returning High Narrow | "vraddhn.i16" | — |
| `vraddhn_s32` | Rounding Add returning High Narrow | "vraddhn.i32" | — |
| `vraddhn_s64` | Rounding Add returning High Narrow | "vraddhn.i64" | — |
| `vraddhn_u16` | Rounding Add returning High Narrow | "vraddhn.i16" | — |
| `vraddhn_u32` | Rounding Add returning High Narrow | "vraddhn.i32" | — |
| `vraddhn_u64` | Rounding Add returning High Narrow | "vraddhn.i64" | — |
| `vrbit_p8` | Reverse bit order | rbit | — |
| `vrbit_s8` | Reverse bit order | rbit | — |
| `vrbit_u8` | Reverse bit order | rbit | — |
| `vrbitq_p8` | Reverse bit order | rbit | — |
| `vrbitq_s8` | Reverse bit order | rbit | — |
| `vrbitq_u8` | Reverse bit order | rbit | — |
| `vrecpe_f32` | Reciprocal estimate | vrecpe | — |
| `vrecpe_f64` | Reciprocal estimate | frecpe | — |
| `vrecpe_u32` | Unsigned reciprocal estimate | vrecpe | — |
| `vrecped_f64` | Reciprocal estimate | frecpe | — |
| `vrecpeq_f32` | Reciprocal estimate | vrecpe | — |
| `vrecpeq_f64` | Reciprocal estimate | frecpe | — |
| `vrecpeq_u32` | Unsigned reciprocal estimate | vrecpe | — |
| `vrecpes_f32` | Reciprocal estimate | frecpe | — |
| `vrecps_f32` | Floating-point reciprocal step | vrecps | — |
| `vrecps_f64` | Floating-point reciprocal step | frecps | — |
| `vrecpsd_f64` | Floating-point reciprocal step | frecps | — |
| `vrecpsq_f32` | Floating-point reciprocal step | vrecps | — |
| `vrecpsq_f64` | Floating-point reciprocal step | frecps | — |
| `vrecpss_f32` | Floating-point reciprocal step | frecps | — |
| `vrecpxd_f64` | Floating-point reciprocal exponent | frecpx | — |
| `vrecpxs_f32` | Floating-point reciprocal exponent | frecpx | — |
| `vreinterpret_f32_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f32_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f32_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f32_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f32_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f32_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f32_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f32_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f32_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f32_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f32_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f32_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f64_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f64_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f64_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f64_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f64_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f64_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f64_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f64_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f64_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f64_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f64_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_f64_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p16_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p16_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p16_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p16_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p16_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p16_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p16_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p16_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p16_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p16_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p16_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p64_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p64_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p64_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p64_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p8_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p8_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p8_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p8_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p8_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p8_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p8_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p8_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p8_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p8_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_p8_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s16_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s16_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s16_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s16_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s16_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s16_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s16_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s16_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s16_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s16_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s16_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s32_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s32_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s32_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s32_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s32_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s32_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s32_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s32_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s32_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s32_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s32_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s64_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s64_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s64_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s64_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s64_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s64_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s64_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s64_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s64_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s64_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s64_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s64_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s8_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s8_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s8_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s8_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s8_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s8_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s8_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s8_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s8_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s8_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_s8_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u16_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u16_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u16_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u16_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u16_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u16_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u16_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u16_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u16_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u16_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u16_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u32_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u32_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u32_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u32_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u32_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u32_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u32_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u32_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u32_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u32_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u32_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u64_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u64_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u64_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u64_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u64_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u64_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u64_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u64_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u64_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u64_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u64_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u64_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u8_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u8_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u8_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u8_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u8_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u8_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u8_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u8_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u8_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u8_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpret_u8_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f32_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f32_p128` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f32_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f32_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f32_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f32_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f32_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f32_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f32_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f32_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f32_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f32_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f32_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f64_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f64_p128` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f64_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f64_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f64_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f64_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f64_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f64_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f64_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f64_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f64_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f64_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_f64_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p128_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p128_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p16_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p16_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p16_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p16_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p16_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p16_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p16_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p16_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p16_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p16_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p16_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p64_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p64_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p64_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p64_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p8_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p8_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p8_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p8_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p8_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p8_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p8_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p8_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p8_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p8_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_p8_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s16_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s16_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s16_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s16_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s16_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s16_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s16_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s16_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s16_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s16_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s16_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s32_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s32_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s32_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s32_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s32_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s32_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s32_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s32_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s32_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s32_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s32_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s64_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s64_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s64_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s64_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s64_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s64_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s64_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s64_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s64_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s64_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s64_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s64_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s8_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s8_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s8_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s8_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s8_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s8_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s8_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s8_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s8_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s8_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_s8_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u16_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u16_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u16_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u16_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u16_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u16_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u16_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u16_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u16_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u16_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u16_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u32_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u32_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u32_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u32_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u32_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u32_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u32_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u32_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u32_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u32_u64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u32_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u64_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u64_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u64_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u64_p64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u64_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u64_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u64_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u64_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u64_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u64_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u64_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u64_u8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u8_f32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u8_f64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u8_p16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u8_p8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u8_s16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u8_s32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u8_s64` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u8_s8` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u8_u16` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u8_u32` | Vector reinterpret cast operation | nop | — |
| `vreinterpretq_u8_u64` | Vector reinterpret cast operation | nop | — |
| `vrev16_p8` | Reversing vector elements (swap endianness) | "vrev16.8" | — |
| `vrev16_s8` | Reversing vector elements (swap endianness) | "vrev16.8" | — |
| `vrev16_u8` | Reversing vector elements (swap endianness) | "vrev16.8" | — |
| `vrev16q_p8` | Reversing vector elements (swap endianness) | "vrev16.8" | — |
| `vrev16q_s8` | Reversing vector elements (swap endianness) | "vrev16.8" | — |
| `vrev16q_u8` | Reversing vector elements (swap endianness) | "vrev16.8" | — |
| `vrev32_p16` | Reversing vector elements (swap endianness) | "vrev32.16" | — |
| `vrev32_p8` | Reversing vector elements (swap endianness) | "vrev32.8" | — |
| `vrev32_s16` | Reversing vector elements (swap endianness) | "vrev32.16" | — |
| `vrev32_s8` | Reversing vector elements (swap endianness) | "vrev32.8" | — |
| `vrev32_u16` | Reversing vector elements (swap endianness) | "vrev32.16" | — |
| `vrev32_u8` | Reversing vector elements (swap endianness) | "vrev32.8" | — |
| `vrev32q_p16` | Reversing vector elements (swap endianness) | "vrev32.16" | — |
| `vrev32q_p8` | Reversing vector elements (swap endianness) | "vrev32.8" | — |
| `vrev32q_s16` | Reversing vector elements (swap endianness) | "vrev32.16" | — |
| `vrev32q_s8` | Reversing vector elements (swap endianness) | "vrev32.8" | — |
| `vrev32q_u16` | Reversing vector elements (swap endianness) | "vrev32.16" | — |
| `vrev32q_u8` | Reversing vector elements (swap endianness) | "vrev32.8" | — |
| `vrev64_f32` | Reversing vector elements (swap endianness) | "vrev64.32" | — |
| `vrev64_p16` | Reversing vector elements (swap endianness) | "vrev64.16" | — |
| `vrev64_p8` | Reversing vector elements (swap endianness) | "vrev64.8" | — |
| `vrev64_s16` | Reversing vector elements (swap endianness) | "vrev64.16" | — |
| `vrev64_s32` | Reversing vector elements (swap endianness) | "vrev64.32" | — |
| `vrev64_s8` | Reversing vector elements (swap endianness) | "vrev64.8" | — |
| `vrev64_u16` | Reversing vector elements (swap endianness) | "vrev64.16" | — |
| `vrev64_u32` | Reversing vector elements (swap endianness) | "vrev64.32" | — |
| `vrev64_u8` | Reversing vector elements (swap endianness) | "vrev64.8" | — |
| `vrev64q_f32` | Reversing vector elements (swap endianness) | "vrev64.32" | — |
| `vrev64q_p16` | Reversing vector elements (swap endianness) | "vrev64.16" | — |
| `vrev64q_p8` | Reversing vector elements (swap endianness) | "vrev64.8" | — |
| `vrev64q_s16` | Reversing vector elements (swap endianness) | "vrev64.16" | — |
| `vrev64q_s32` | Reversing vector elements (swap endianness) | "vrev64.32" | — |
| `vrev64q_s8` | Reversing vector elements (swap endianness) | "vrev64.8" | — |
| `vrev64q_u16` | Reversing vector elements (swap endianness) | "vrev64.16" | — |
| `vrev64q_u32` | Reversing vector elements (swap endianness) | "vrev64.32" | — |
| `vrev64q_u8` | Reversing vector elements (swap endianness) | "vrev64.8" | — |
| `vrhadd_s16` | Rounding halving add | "vrhadd.s16" | — |
| `vrhadd_s32` | Rounding halving add | "vrhadd.s32" | — |
| `vrhadd_s8` | Rounding halving add | "vrhadd.s8" | — |
| `vrhadd_u16` | Rounding halving add | "vrhadd.u16" | — |
| `vrhadd_u32` | Rounding halving add | "vrhadd.u32" | — |
| `vrhadd_u8` | Rounding halving add | "vrhadd.u8" | — |
| `vrhaddq_s16` | Rounding halving add | "vrhadd.s16" | — |
| `vrhaddq_s32` | Rounding halving add | "vrhadd.s32" | — |
| `vrhaddq_s8` | Rounding halving add | "vrhadd.s8" | — |
| `vrhaddq_u16` | Rounding halving add | "vrhadd.u16" | — |
| `vrhaddq_u32` | Rounding halving add | "vrhadd.u32" | — |
| `vrhaddq_u8` | Rounding halving add | "vrhadd.u8" | — |
| `vrnd_f32` | Floating-point round to integral, toward zero | frintz | — |
| `vrnd_f64` | Floating-point round to integral, toward zero | frintz | — |
| `vrnda_f32` | Floating-point round to integral, to nearest with ties to aw... | frinta | — |
| `vrnda_f64` | Floating-point round to integral, to nearest with ties to aw... | frinta | — |
| `vrndaq_f32` | Floating-point round to integral, to nearest with ties to aw... | frinta | — |
| `vrndaq_f64` | Floating-point round to integral, to nearest with ties to aw... | frinta | — |
| `vrndi_f32` | Floating-point round to integral, using current rounding mod... | frinti | — |
| `vrndi_f64` | Floating-point round to integral, using current rounding mod... | frinti | — |
| `vrndiq_f32` | Floating-point round to integral, using current rounding mod... | frinti | — |
| `vrndiq_f64` | Floating-point round to integral, using current rounding mod... | frinti | — |
| `vrndm_f32` | Floating-point round to integral, toward minus infinity | frintm | — |
| `vrndm_f64` | Floating-point round to integral, toward minus infinity | frintm | — |
| `vrndmq_f32` | Floating-point round to integral, toward minus infinity | frintm | — |
| `vrndmq_f64` | Floating-point round to integral, toward minus infinity | frintm | — |
| `vrndn_f32` | Floating-point round to integral, to nearest with ties to ev... | vrintn | — |
| `vrndn_f64` | Floating-point round to integral, to nearest with ties to ev... | frintn | — |
| `vrndnq_f32` | Floating-point round to integral, to nearest with ties to ev... | vrintn | — |
| `vrndnq_f64` | Floating-point round to integral, to nearest with ties to ev... | frintn | — |
| `vrndns_f32` | Floating-point round to integral, to nearest with ties to ev... | frintn | — |
| `vrndp_f32` | Floating-point round to integral, toward plus infinity | frintp | — |
| `vrndp_f64` | Floating-point round to integral, toward plus infinity | frintp | — |
| `vrndpq_f32` | Floating-point round to integral, toward plus infinity | frintp | — |
| `vrndpq_f64` | Floating-point round to integral, toward plus infinity | frintp | — |
| `vrndq_f32` | Floating-point round to integral, toward zero | frintz | — |
| `vrndq_f64` | Floating-point round to integral, toward zero | frintz | — |
| `vrndx_f32` | Floating-point round to integral exact, using current roundi... | frintx | — |
| `vrndx_f64` | Floating-point round to integral exact, using current roundi... | frintx | — |
| `vrndxq_f32` | Floating-point round to integral exact, using current roundi... | frintx | — |
| `vrndxq_f64` | Floating-point round to integral exact, using current roundi... | frintx | — |
| `vrshl_s16` | Signed rounding shift left | vrshl | — |
| `vrshl_s32` | Signed rounding shift left | vrshl | — |
| `vrshl_s64` | Signed rounding shift left | vrshl | — |
| `vrshl_s8` | Signed rounding shift left | vrshl | — |
| `vrshl_u16` | Unsigned rounding shift left | vrshl | — |
| `vrshl_u32` | Unsigned rounding shift left | vrshl | — |
| `vrshl_u64` | Unsigned rounding shift left | vrshl | — |
| `vrshl_u8` | Unsigned rounding shift left | vrshl | — |
| `vrshld_s64` | Signed rounding shift left | srshl | — |
| `vrshld_u64` | Unsigned rounding shift left | urshl | — |
| `vrshlq_s16` | Signed rounding shift left | vrshl | — |
| `vrshlq_s32` | Signed rounding shift left | vrshl | — |
| `vrshlq_s64` | Signed rounding shift left | vrshl | — |
| `vrshlq_s8` | Signed rounding shift left | vrshl | — |
| `vrshlq_u16` | Unsigned rounding shift left | vrshl | — |
| `vrshlq_u32` | Unsigned rounding shift left | vrshl | — |
| `vrshlq_u64` | Unsigned rounding shift left | vrshl | — |
| `vrshlq_u8` | Unsigned rounding shift left | vrshl | — |
| `vrshr_n_s16` | Signed rounding shift right | vrshr | — |
| `vrshr_n_s32` | Signed rounding shift right | vrshr | — |
| `vrshr_n_s64` | Signed rounding shift right | vrshr | — |
| `vrshr_n_s8` | Signed rounding shift right | vrshr | — |
| `vrshr_n_u16` | Unsigned rounding shift right | vrshr | — |
| `vrshr_n_u32` | Unsigned rounding shift right | vrshr | — |
| `vrshr_n_u64` | Unsigned rounding shift right | vrshr | — |
| `vrshr_n_u8` | Unsigned rounding shift right | vrshr | — |
| `vrshrd_n_s64` | Signed rounding shift right | srshr | — |
| `vrshrd_n_u64` | Unsigned rounding shift right | urshr | — |
| `vrshrn_high_n_s16` | Rounding shift right narrow | rshrn2 | — |
| `vrshrn_high_n_s32` | Rounding shift right narrow | rshrn2 | — |
| `vrshrn_high_n_s64` | Rounding shift right narrow | rshrn2 | — |
| `vrshrn_high_n_u16` | Rounding shift right narrow | rshrn2 | — |
| `vrshrn_high_n_u32` | Rounding shift right narrow | rshrn2 | — |
| `vrshrn_high_n_u64` | Rounding shift right narrow | rshrn2 | — |
| `vrshrn_n_u16` | Rounding shift right narrow | vrshrn | — |
| `vrshrn_n_u32` | Rounding shift right narrow | vrshrn | — |
| `vrshrn_n_u64` | Rounding shift right narrow | vrshrn | — |
| `vrshrq_n_s16` | Signed rounding shift right | vrshr | — |
| `vrshrq_n_s32` | Signed rounding shift right | vrshr | — |
| `vrshrq_n_s64` | Signed rounding shift right | vrshr | — |
| `vrshrq_n_s8` | Signed rounding shift right | vrshr | — |
| `vrshrq_n_u16` | Unsigned rounding shift right | vrshr | — |
| `vrshrq_n_u32` | Unsigned rounding shift right | vrshr | — |
| `vrshrq_n_u64` | Unsigned rounding shift right | vrshr | — |
| `vrshrq_n_u8` | Unsigned rounding shift right | vrshr | — |
| `vrsqrte_f32` | Reciprocal square-root estimate | vrsqrte | — |
| `vrsqrte_f64` | Reciprocal square-root estimate | frsqrte | — |
| `vrsqrte_u32` | Unsigned reciprocal square root estimate | vrsqrte | — |
| `vrsqrted_f64` | Reciprocal square-root estimate | frsqrte | — |
| `vrsqrteq_f32` | Reciprocal square-root estimate | vrsqrte | — |
| `vrsqrteq_f64` | Reciprocal square-root estimate | frsqrte | — |
| `vrsqrteq_u32` | Unsigned reciprocal square root estimate | vrsqrte | — |
| `vrsqrtes_f32` | Reciprocal square-root estimate | frsqrte | — |
| `vrsqrts_f32` | Floating-point reciprocal square root step | vrsqrts | — |
| `vrsqrts_f64` | Floating-point reciprocal square root step | frsqrts | — |
| `vrsqrtsd_f64` | Floating-point reciprocal square root step | frsqrts | — |
| `vrsqrtsq_f32` | Floating-point reciprocal square root step | vrsqrts | — |
| `vrsqrtsq_f64` | Floating-point reciprocal square root step | frsqrts | — |
| `vrsqrtss_f32` | Floating-point reciprocal square root step | frsqrts | — |
| `vrsra_n_s16` | Signed rounding shift right and accumulate | vrsra | — |
| `vrsra_n_s32` | Signed rounding shift right and accumulate | vrsra | — |
| `vrsra_n_s64` | Signed rounding shift right and accumulate | vrsra | — |
| `vrsra_n_s8` | Signed rounding shift right and accumulate | vrsra | — |
| `vrsra_n_u16` | Unsigned rounding shift right and accumulate | vrsra | — |
| `vrsra_n_u32` | Unsigned rounding shift right and accumulate | vrsra | — |
| `vrsra_n_u64` | Unsigned rounding shift right and accumulate | vrsra | — |
| `vrsra_n_u8` | Unsigned rounding shift right and accumulate | vrsra | — |
| `vrsrad_n_s64` | Signed rounding shift right and accumulate | srshr | — |
| `vrsrad_n_u64` | Unsigned rounding shift right and accumulate | urshr | — |
| `vrsraq_n_s16` | Signed rounding shift right and accumulate | vrsra | — |
| `vrsraq_n_s32` | Signed rounding shift right and accumulate | vrsra | — |
| `vrsraq_n_s64` | Signed rounding shift right and accumulate | vrsra | — |
| `vrsraq_n_s8` | Signed rounding shift right and accumulate | vrsra | — |
| `vrsraq_n_u16` | Unsigned rounding shift right and accumulate | vrsra | — |
| `vrsraq_n_u32` | Unsigned rounding shift right and accumulate | vrsra | — |
| `vrsraq_n_u64` | Unsigned rounding shift right and accumulate | vrsra | — |
| `vrsraq_n_u8` | Unsigned rounding shift right and accumulate | vrsra | — |
| `vrsubhn_high_s16` | Rounding subtract returning high narrow | rsubhn2 | — |
| `vrsubhn_high_s32` | Rounding subtract returning high narrow | rsubhn2 | — |
| `vrsubhn_high_s64` | Rounding subtract returning high narrow | rsubhn2 | — |
| `vrsubhn_high_u16` | Rounding subtract returning high narrow | rsubhn2 | — |
| `vrsubhn_high_u32` | Rounding subtract returning high narrow | rsubhn2 | — |
| `vrsubhn_high_u64` | Rounding subtract returning high narrow | rsubhn2 | — |
| `vrsubhn_s16` | Rounding subtract returning high narrow | vrsubhn | — |
| `vrsubhn_s32` | Rounding subtract returning high narrow | vrsubhn | — |
| `vrsubhn_s64` | Rounding subtract returning high narrow | vrsubhn | — |
| `vrsubhn_u16` | Rounding subtract returning high narrow | vrsubhn | — |
| `vrsubhn_u32` | Rounding subtract returning high narrow | vrsubhn | — |
| `vrsubhn_u64` | Rounding subtract returning high narrow | vrsubhn | — |
| `vset_lane_f32` | Insert vector element from another vector element | nop | — |
| `vset_lane_f64` | Insert vector element from another vector element | nop | — |
| `vset_lane_p16` | Insert vector element from another vector element | nop | — |
| `vset_lane_p8` | Insert vector element from another vector element | nop | — |
| `vset_lane_s16` | Insert vector element from another vector element | nop | — |
| `vset_lane_s32` | Insert vector element from another vector element | nop | — |
| `vset_lane_s64` | Insert vector element from another vector element | nop | — |
| `vset_lane_s8` | Insert vector element from another vector element | nop | — |
| `vset_lane_u16` | Insert vector element from another vector element | nop | — |
| `vset_lane_u32` | Insert vector element from another vector element | nop | — |
| `vset_lane_u64` | Insert vector element from another vector element | nop | — |
| `vset_lane_u8` | Insert vector element from another vector element | nop | — |
| `vsetq_lane_f32` | Insert vector element from another vector element | nop | — |
| `vsetq_lane_f64` | Insert vector element from another vector element | nop | — |
| `vsetq_lane_p16` | Insert vector element from another vector element | nop | — |
| `vsetq_lane_p8` | Insert vector element from another vector element | nop | — |
| `vsetq_lane_s16` | Insert vector element from another vector element | nop | — |
| `vsetq_lane_s32` | Insert vector element from another vector element | nop | — |
| `vsetq_lane_s64` | Insert vector element from another vector element | nop | — |
| `vsetq_lane_s8` | Insert vector element from another vector element | nop | — |
| `vsetq_lane_u16` | Insert vector element from another vector element | nop | — |
| `vsetq_lane_u32` | Insert vector element from another vector element | nop | — |
| `vsetq_lane_u64` | Insert vector element from another vector element | nop | — |
| `vsetq_lane_u8` | Insert vector element from another vector element | nop | — |
| `vshl_n_s16` | Shift left | vshl | — |
| `vshl_n_s32` | Shift left | vshl | — |
| `vshl_n_s64` | Shift left | vshl | — |
| `vshl_n_s8` | Shift left | vshl | — |
| `vshl_n_u16` | Shift left | vshl | — |
| `vshl_n_u32` | Shift left | vshl | — |
| `vshl_n_u64` | Shift left | vshl | — |
| `vshl_n_u8` | Shift left | vshl | — |
| `vshl_s16` | Signed Shift left | vshl | — |
| `vshl_s32` | Signed Shift left | vshl | — |
| `vshl_s64` | Signed Shift left | vshl | — |
| `vshl_s8` | Signed Shift left | vshl | — |
| `vshl_u16` | Unsigned Shift left | vshl | — |
| `vshl_u32` | Unsigned Shift left | vshl | — |
| `vshl_u64` | Unsigned Shift left | vshl | — |
| `vshl_u8` | Unsigned Shift left | vshl | — |
| `vshld_n_s64` | Shift left | nop | — |
| `vshld_n_u64` | Shift left | nop | — |
| `vshld_s64` | Signed Shift left | sshl | — |
| `vshld_u64` | Unsigned Shift left | ushl | — |
| `vshll_high_n_s16` | Signed shift left long | sshll2 | — |
| `vshll_high_n_s32` | Signed shift left long | sshll2 | — |
| `vshll_high_n_s8` | Signed shift left long | sshll2 | — |
| `vshll_high_n_u16` | Signed shift left long | ushll2 | — |
| `vshll_high_n_u32` | Signed shift left long | ushll2 | — |
| `vshll_high_n_u8` | Signed shift left long | ushll2 | — |
| `vshll_n_s16` | Signed shift left long | "vshll.s16" | — |
| `vshll_n_s32` | Signed shift left long | "vshll.s32" | — |
| `vshll_n_s8` | Signed shift left long | "vshll.s8" | — |
| `vshll_n_u16` | Signed shift left long | "vshll.u16" | — |
| `vshll_n_u32` | Signed shift left long | "vshll.u32" | — |
| `vshll_n_u8` | Signed shift left long | "vshll.u8" | — |
| `vshlq_n_s16` | Shift left | vshl | — |
| `vshlq_n_s32` | Shift left | vshl | — |
| `vshlq_n_s64` | Shift left | vshl | — |
| `vshlq_n_s8` | Shift left | vshl | — |
| `vshlq_n_u16` | Shift left | vshl | — |
| `vshlq_n_u32` | Shift left | vshl | — |
| `vshlq_n_u64` | Shift left | vshl | — |
| `vshlq_n_u8` | Shift left | vshl | — |
| `vshlq_s16` | Signed Shift left | vshl | — |
| `vshlq_s32` | Signed Shift left | vshl | — |
| `vshlq_s64` | Signed Shift left | vshl | — |
| `vshlq_s8` | Signed Shift left | vshl | — |
| `vshlq_u16` | Unsigned Shift left | vshl | — |
| `vshlq_u32` | Unsigned Shift left | vshl | — |
| `vshlq_u64` | Unsigned Shift left | vshl | — |
| `vshlq_u8` | Unsigned Shift left | vshl | — |
| `vshr_n_s16` | Shift right | "vshr.s16" | — |
| `vshr_n_s32` | Shift right | "vshr.s32" | — |
| `vshr_n_s64` | Shift right | "vshr.s64" | — |
| `vshr_n_s8` | Shift right | "vshr.s8" | — |
| `vshr_n_u16` | Shift right | "vshr.u16" | — |
| `vshr_n_u32` | Shift right | "vshr.u32" | — |
| `vshr_n_u64` | Shift right | "vshr.u64" | — |
| `vshr_n_u8` | Shift right | "vshr.u8" | — |
| `vshrd_n_s64` | Signed shift right | nop | — |
| `vshrd_n_u64` | Unsigned shift right | nop | — |
| `vshrn_high_n_s16` | Shift right narrow | shrn2 | — |
| `vshrn_high_n_s32` | Shift right narrow | shrn2 | — |
| `vshrn_high_n_s64` | Shift right narrow | shrn2 | — |
| `vshrn_high_n_u16` | Shift right narrow | shrn2 | — |
| `vshrn_high_n_u32` | Shift right narrow | shrn2 | — |
| `vshrn_high_n_u64` | Shift right narrow | shrn2 | — |
| `vshrn_n_s16` | Shift right narrow | "vshrn.i16" | — |
| `vshrn_n_s32` | Shift right narrow | "vshrn.i32" | — |
| `vshrn_n_s64` | Shift right narrow | "vshrn.i64" | — |
| `vshrn_n_u16` | Shift right narrow | "vshrn.i16" | — |
| `vshrn_n_u32` | Shift right narrow | "vshrn.i32" | — |
| `vshrn_n_u64` | Shift right narrow | "vshrn.i64" | — |
| `vshrq_n_s16` | Shift right | "vshr.s16" | — |
| `vshrq_n_s32` | Shift right | "vshr.s32" | — |
| `vshrq_n_s64` | Shift right | "vshr.s64" | — |
| `vshrq_n_s8` | Shift right | "vshr.s8" | — |
| `vshrq_n_u16` | Shift right | "vshr.u16" | — |
| `vshrq_n_u32` | Shift right | "vshr.u32" | — |
| `vshrq_n_u64` | Shift right | "vshr.u64" | — |
| `vshrq_n_u8` | Shift right | "vshr.u8" | — |
| `vslid_n_s64` | Shift left and insert | sli | — |
| `vslid_n_u64` | Shift left and insert | sli | — |
| `vsqadd_u16` | Unsigned saturating Accumulate of Signed value | usqadd | — |
| `vsqadd_u32` | Unsigned saturating Accumulate of Signed value | usqadd | — |
| `vsqadd_u64` | Unsigned saturating Accumulate of Signed value | usqadd | — |
| `vsqadd_u8` | Unsigned saturating Accumulate of Signed value | usqadd | — |
| `vsqaddb_u8` | Unsigned saturating accumulate of signed value | usqadd | — |
| `vsqaddd_u64` | Unsigned saturating accumulate of signed value | usqadd | — |
| `vsqaddh_u16` | Unsigned saturating accumulate of signed value | usqadd | — |
| `vsqaddq_u16` | Unsigned saturating Accumulate of Signed value | usqadd | — |
| `vsqaddq_u32` | Unsigned saturating Accumulate of Signed value | usqadd | — |
| `vsqaddq_u64` | Unsigned saturating Accumulate of Signed value | usqadd | — |
| `vsqaddq_u8` | Unsigned saturating Accumulate of Signed value | usqadd | — |
| `vsqadds_u32` | Unsigned saturating accumulate of signed value | usqadd | — |
| `vsqrt_f32` | Calculates the square root of each lane | fsqrt | — |
| `vsqrt_f64` | Calculates the square root of each lane | fsqrt | — |
| `vsqrtq_f32` | Calculates the square root of each lane | fsqrt | — |
| `vsqrtq_f64` | Calculates the square root of each lane | fsqrt | — |
| `vsra_n_s16` | Signed shift right and accumulate | vsra | — |
| `vsra_n_s32` | Signed shift right and accumulate | vsra | — |
| `vsra_n_s64` | Signed shift right and accumulate | vsra | — |
| `vsra_n_s8` | Signed shift right and accumulate | vsra | — |
| `vsra_n_u16` | Unsigned shift right and accumulate | vsra | — |
| `vsra_n_u32` | Unsigned shift right and accumulate | vsra | — |
| `vsra_n_u64` | Unsigned shift right and accumulate | vsra | — |
| `vsra_n_u8` | Unsigned shift right and accumulate | vsra | — |
| `vsrad_n_s64` | Signed shift right and accumulate | nop | — |
| `vsrad_n_u64` | Unsigned shift right and accumulate | nop | — |
| `vsraq_n_s16` | Signed shift right and accumulate | vsra | — |
| `vsraq_n_s32` | Signed shift right and accumulate | vsra | — |
| `vsraq_n_s64` | Signed shift right and accumulate | vsra | — |
| `vsraq_n_s8` | Signed shift right and accumulate | vsra | — |
| `vsraq_n_u16` | Unsigned shift right and accumulate | vsra | — |
| `vsraq_n_u32` | Unsigned shift right and accumulate | vsra | — |
| `vsraq_n_u64` | Unsigned shift right and accumulate | vsra | — |
| `vsraq_n_u8` | Unsigned shift right and accumulate | vsra | — |
| `vsrid_n_s64` | Shift right and insert | sri | — |
| `vsrid_n_u64` | Shift right and insert | sri | — |
| `vsub_f32` | Subtract | "vsub.f32" | — |
| `vsub_f64` | Subtract | fsub | — |
| `vsub_s16` | Subtract | "vsub.i16" | — |
| `vsub_s32` | Subtract | "vsub.i32" | — |
| `vsub_s64` | Subtract | "vsub.i64" | — |
| `vsub_s8` | Subtract | "vsub.i8" | — |
| `vsub_u16` | Subtract | "vsub.i16" | — |
| `vsub_u32` | Subtract | "vsub.i32" | — |
| `vsub_u64` | Subtract | "vsub.i64" | — |
| `vsub_u8` | Subtract | "vsub.i8" | — |
| `vsubd_s64` | Subtract | nop | — |
| `vsubd_u64` | Subtract | nop | — |
| `vsubhn_high_s16` | Subtract returning high narrow | vsubhn | — |
| `vsubhn_high_s32` | Subtract returning high narrow | vsubhn | — |
| `vsubhn_high_s64` | Subtract returning high narrow | vsubhn | — |
| `vsubhn_high_u16` | Subtract returning high narrow | vsubhn | — |
| `vsubhn_high_u32` | Subtract returning high narrow | vsubhn | — |
| `vsubhn_high_u64` | Subtract returning high narrow | vsubhn | — |
| `vsubhn_s16` | Subtract returning high narrow | vsubhn | — |
| `vsubhn_s32` | Subtract returning high narrow | vsubhn | — |
| `vsubhn_s64` | Subtract returning high narrow | vsubhn | — |
| `vsubhn_u16` | Subtract returning high narrow | vsubhn | — |
| `vsubhn_u32` | Subtract returning high narrow | vsubhn | — |
| `vsubhn_u64` | Subtract returning high narrow | vsubhn | — |
| `vsubl_high_s16` | Signed Subtract Long | ssubl2 | — |
| `vsubl_high_s32` | Signed Subtract Long | ssubl2 | — |
| `vsubl_high_s8` | Signed Subtract Long | ssubl2 | — |
| `vsubl_high_u16` | Unsigned Subtract Long | usubl2 | — |
| `vsubl_high_u32` | Unsigned Subtract Long | usubl2 | — |
| `vsubl_high_u8` | Unsigned Subtract Long | usubl2 | — |
| `vsubl_s16` | Signed Subtract Long | vsubl | — |
| `vsubl_s32` | Signed Subtract Long | vsubl | — |
| `vsubl_s8` | Signed Subtract Long | vsubl | — |
| `vsubl_u16` | Unsigned Subtract Long | vsubl | — |
| `vsubl_u32` | Unsigned Subtract Long | vsubl | — |
| `vsubl_u8` | Unsigned Subtract Long | vsubl | — |
| `vsubq_f32` | Subtract | "vsub.f32" | — |
| `vsubq_f64` | Subtract | fsub | — |
| `vsubq_s16` | Subtract | "vsub.i16" | — |
| `vsubq_s32` | Subtract | "vsub.i32" | — |
| `vsubq_s64` | Subtract | "vsub.i64" | — |
| `vsubq_s8` | Subtract | "vsub.i8" | — |
| `vsubq_u16` | Subtract | "vsub.i16" | — |
| `vsubq_u32` | Subtract | "vsub.i32" | — |
| `vsubq_u64` | Subtract | "vsub.i64" | — |
| `vsubq_u8` | Subtract | "vsub.i8" | — |
| `vsubw_high_s16` | Signed Subtract Wide | ssubw2 | — |
| `vsubw_high_s32` | Signed Subtract Wide | ssubw2 | — |
| `vsubw_high_s8` | Signed Subtract Wide | ssubw2 | — |
| `vsubw_high_u16` | Unsigned Subtract Wide | usubw2 | — |
| `vsubw_high_u32` | Unsigned Subtract Wide | usubw2 | — |
| `vsubw_high_u8` | Unsigned Subtract Wide | usubw2 | — |
| `vsubw_s16` | Signed Subtract Wide | vsubw | — |
| `vsubw_s32` | Signed Subtract Wide | vsubw | — |
| `vsubw_s8` | Signed Subtract Wide | vsubw | — |
| `vsubw_u16` | Unsigned Subtract Wide | vsubw | — |
| `vsubw_u32` | Unsigned Subtract Wide | vsubw | — |
| `vsubw_u8` | Unsigned Subtract Wide | vsubw | — |
| `vtrn1_f32` | Transpose vectors | zip1 | — |
| `vtrn1_p16` | Transpose vectors | trn1 | — |
| `vtrn1_p8` | Transpose vectors | trn1 | — |
| `vtrn1_s16` | Transpose vectors | trn1 | — |
| `vtrn1_s32` | Transpose vectors | zip1 | — |
| `vtrn1_s8` | Transpose vectors | trn1 | — |
| `vtrn1_u16` | Transpose vectors | trn1 | — |
| `vtrn1_u32` | Transpose vectors | zip1 | — |
| `vtrn1_u8` | Transpose vectors | trn1 | — |
| `vtrn1q_f32` | Transpose vectors | trn1 | — |
| `vtrn1q_f64` | Transpose vectors | zip1 | — |
| `vtrn1q_p16` | Transpose vectors | trn1 | — |
| `vtrn1q_p64` | Transpose vectors | zip1 | — |
| `vtrn1q_p8` | Transpose vectors | trn1 | — |
| `vtrn1q_s16` | Transpose vectors | trn1 | — |
| `vtrn1q_s32` | Transpose vectors | trn1 | — |
| `vtrn1q_s64` | Transpose vectors | zip1 | — |
| `vtrn1q_s8` | Transpose vectors | trn1 | — |
| `vtrn1q_u16` | Transpose vectors | trn1 | — |
| `vtrn1q_u32` | Transpose vectors | trn1 | — |
| `vtrn1q_u64` | Transpose vectors | zip1 | — |
| `vtrn1q_u8` | Transpose vectors | trn1 | — |
| `vtrn2_f32` | Transpose vectors | zip2 | — |
| `vtrn2_p16` | Transpose vectors | trn2 | — |
| `vtrn2_p8` | Transpose vectors | trn2 | — |
| `vtrn2_s16` | Transpose vectors | trn2 | — |
| `vtrn2_s32` | Transpose vectors | zip2 | — |
| `vtrn2_s8` | Transpose vectors | trn2 | — |
| `vtrn2_u16` | Transpose vectors | trn2 | — |
| `vtrn2_u32` | Transpose vectors | zip2 | — |
| `vtrn2_u8` | Transpose vectors | trn2 | — |
| `vtrn2q_f32` | Transpose vectors | trn2 | — |
| `vtrn2q_f64` | Transpose vectors | zip2 | — |
| `vtrn2q_p16` | Transpose vectors | trn2 | — |
| `vtrn2q_p64` | Transpose vectors | zip2 | — |
| `vtrn2q_p8` | Transpose vectors | trn2 | — |
| `vtrn2q_s16` | Transpose vectors | trn2 | — |
| `vtrn2q_s32` | Transpose vectors | trn2 | — |
| `vtrn2q_s64` | Transpose vectors | zip2 | — |
| `vtrn2q_s8` | Transpose vectors | trn2 | — |
| `vtrn2q_u16` | Transpose vectors | trn2 | — |
| `vtrn2q_u32` | Transpose vectors | trn2 | — |
| `vtrn2q_u64` | Transpose vectors | zip2 | — |
| `vtrn2q_u8` | Transpose vectors | trn2 | — |
| `vtrn_f32` | Transpose elements | vtrn | — |
| `vtrn_p16` | Transpose elements | vtrn | — |
| `vtrn_p8` | Transpose elements | vtrn | — |
| `vtrn_s16` | Transpose elements | vtrn | — |
| `vtrn_s32` | Transpose elements | vtrn | — |
| `vtrn_s8` | Transpose elements | vtrn | — |
| `vtrn_u16` | Transpose elements | vtrn | — |
| `vtrn_u32` | Transpose elements | vtrn | — |
| `vtrn_u8` | Transpose elements | vtrn | — |
| `vtrnq_f32` | Transpose elements | vtrn | — |
| `vtrnq_p16` | Transpose elements | vtrn | — |
| `vtrnq_p8` | Transpose elements | vtrn | — |
| `vtrnq_s16` | Transpose elements | vtrn | — |
| `vtrnq_s32` | Transpose elements | vtrn | — |
| `vtrnq_s8` | Transpose elements | vtrn | — |
| `vtrnq_u16` | Transpose elements | vtrn | — |
| `vtrnq_u32` | Transpose elements | vtrn | — |
| `vtrnq_u8` | Transpose elements | vtrn | — |
| `vtst_p16` | Signed compare bitwise Test bits nonzero | vtst | — |
| `vtst_p64` | Signed compare bitwise Test bits nonzero | cmtst | — |
| `vtst_p8` | Signed compare bitwise Test bits nonzero | vtst | — |
| `vtst_s16` | Signed compare bitwise Test bits nonzero | vtst | — |
| `vtst_s32` | Signed compare bitwise Test bits nonzero | vtst | — |
| `vtst_s64` | Signed compare bitwise Test bits nonzero | cmtst | — |
| `vtst_s8` | Signed compare bitwise Test bits nonzero | vtst | — |
| `vtst_u16` | Unsigned compare bitwise Test bits nonzero | vtst | — |
| `vtst_u32` | Unsigned compare bitwise Test bits nonzero | vtst | — |
| `vtst_u64` | Unsigned compare bitwise Test bits nonzero | cmtst | — |
| `vtst_u8` | Unsigned compare bitwise Test bits nonzero | vtst | — |
| `vtstd_s64` | Compare bitwise test bits nonzero | tst | — |
| `vtstd_u64` | Compare bitwise test bits nonzero | tst | — |
| `vtstq_p16` | Signed compare bitwise Test bits nonzero | vtst | — |
| `vtstq_p64` | Signed compare bitwise Test bits nonzero | cmtst | — |
| `vtstq_p8` | Signed compare bitwise Test bits nonzero | vtst | — |
| `vtstq_s16` | Signed compare bitwise Test bits nonzero | vtst | — |
| `vtstq_s32` | Signed compare bitwise Test bits nonzero | vtst | — |
| `vtstq_s64` | Signed compare bitwise Test bits nonzero | cmtst | — |
| `vtstq_s8` | Signed compare bitwise Test bits nonzero | vtst | — |
| `vtstq_u16` | Unsigned compare bitwise Test bits nonzero | vtst | — |
| `vtstq_u32` | Unsigned compare bitwise Test bits nonzero | vtst | — |
| `vtstq_u64` | Unsigned compare bitwise Test bits nonzero | cmtst | — |
| `vtstq_u8` | Unsigned compare bitwise Test bits nonzero | vtst | — |
| `vuqadd_s16` | Signed saturating Accumulate of Unsigned value | suqadd | — |
| `vuqadd_s32` | Signed saturating Accumulate of Unsigned value | suqadd | — |
| `vuqadd_s64` | Signed saturating Accumulate of Unsigned value | suqadd | — |
| `vuqadd_s8` | Signed saturating Accumulate of Unsigned value | suqadd | — |
| `vuqaddb_s8` | Signed saturating accumulate of unsigned value | suqadd | — |
| `vuqaddd_s64` | Signed saturating accumulate of unsigned value | suqadd | — |
| `vuqaddh_s16` | Signed saturating accumulate of unsigned value | suqadd | — |
| `vuqaddq_s16` | Signed saturating Accumulate of Unsigned value | suqadd | — |
| `vuqaddq_s32` | Signed saturating Accumulate of Unsigned value | suqadd | — |
| `vuqaddq_s64` | Signed saturating Accumulate of Unsigned value | suqadd | — |
| `vuqaddq_s8` | Signed saturating Accumulate of Unsigned value | suqadd | — |
| `vuqadds_s32` | Signed saturating accumulate of unsigned value | suqadd | — |
| `vuzp1_f32` | Unzip vectors | zip1 | — |
| `vuzp1_p16` | Unzip vectors | uzp1 | — |
| `vuzp1_p8` | Unzip vectors | uzp1 | — |
| `vuzp1_s16` | Unzip vectors | uzp1 | — |
| `vuzp1_s32` | Unzip vectors | zip1 | — |
| `vuzp1_s8` | Unzip vectors | uzp1 | — |
| `vuzp1_u16` | Unzip vectors | uzp1 | — |
| `vuzp1_u32` | Unzip vectors | zip1 | — |
| `vuzp1_u8` | Unzip vectors | uzp1 | — |
| `vuzp1q_f32` | Unzip vectors | uzp1 | — |
| `vuzp1q_f64` | Unzip vectors | zip1 | — |
| `vuzp1q_p16` | Unzip vectors | uzp1 | — |
| `vuzp1q_p64` | Unzip vectors | zip1 | — |
| `vuzp1q_p8` | Unzip vectors | uzp1 | — |
| `vuzp1q_s16` | Unzip vectors | uzp1 | — |
| `vuzp1q_s32` | Unzip vectors | uzp1 | — |
| `vuzp1q_s64` | Unzip vectors | zip1 | — |
| `vuzp1q_s8` | Unzip vectors | uzp1 | — |
| `vuzp1q_u16` | Unzip vectors | uzp1 | — |
| `vuzp1q_u32` | Unzip vectors | uzp1 | — |
| `vuzp1q_u64` | Unzip vectors | zip1 | — |
| `vuzp1q_u8` | Unzip vectors | uzp1 | — |
| `vuzp2_f32` | Unzip vectors | zip2 | — |
| `vuzp2_p16` | Unzip vectors | uzp2 | — |
| `vuzp2_p8` | Unzip vectors | uzp2 | — |
| `vuzp2_s16` | Unzip vectors | uzp2 | — |
| `vuzp2_s32` | Unzip vectors | zip2 | — |
| `vuzp2_s8` | Unzip vectors | uzp2 | — |
| `vuzp2_u16` | Unzip vectors | uzp2 | — |
| `vuzp2_u32` | Unzip vectors | zip2 | — |
| `vuzp2_u8` | Unzip vectors | uzp2 | — |
| `vuzp2q_f32` | Unzip vectors | uzp2 | — |
| `vuzp2q_f64` | Unzip vectors | zip2 | — |
| `vuzp2q_p16` | Unzip vectors | uzp2 | — |
| `vuzp2q_p64` | Unzip vectors | zip2 | — |
| `vuzp2q_p8` | Unzip vectors | uzp2 | — |
| `vuzp2q_s16` | Unzip vectors | uzp2 | — |
| `vuzp2q_s32` | Unzip vectors | uzp2 | — |
| `vuzp2q_s64` | Unzip vectors | zip2 | — |
| `vuzp2q_s8` | Unzip vectors | uzp2 | — |
| `vuzp2q_u16` | Unzip vectors | uzp2 | — |
| `vuzp2q_u32` | Unzip vectors | uzp2 | — |
| `vuzp2q_u64` | Unzip vectors | zip2 | — |
| `vuzp2q_u8` | Unzip vectors | uzp2 | — |
| `vuzp_f32` | Unzip vectors | vtrn | — |
| `vuzp_p16` | Unzip vectors | vuzp | — |
| `vuzp_p8` | Unzip vectors | vuzp | — |
| `vuzp_s16` | Unzip vectors | vuzp | — |
| `vuzp_s32` | Unzip vectors | vtrn | — |
| `vuzp_s8` | Unzip vectors | vuzp | — |
| `vuzp_u16` | Unzip vectors | vuzp | — |
| `vuzp_u32` | Unzip vectors | vtrn | — |
| `vuzp_u8` | Unzip vectors | vuzp | — |
| `vuzpq_f32` | Unzip vectors | vuzp | — |
| `vuzpq_p16` | Unzip vectors | vuzp | — |
| `vuzpq_p8` | Unzip vectors | vuzp | — |
| `vuzpq_s16` | Unzip vectors | vuzp | — |
| `vuzpq_s32` | Unzip vectors | vuzp | — |
| `vuzpq_s8` | Unzip vectors | vuzp | — |
| `vuzpq_u16` | Unzip vectors | vuzp | — |
| `vuzpq_u32` | Unzip vectors | vuzp | — |
| `vuzpq_u8` | Unzip vectors | vuzp | — |
| `vzip1_f32` | Zip vectors | zip1 | — |
| `vzip1_p16` | Zip vectors | zip1 | — |
| `vzip1_p8` | Zip vectors | zip1 | — |
| `vzip1_s16` | Zip vectors | zip1 | — |
| `vzip1_s32` | Zip vectors | zip1 | — |
| `vzip1_s8` | Zip vectors | zip1 | — |
| `vzip1_u16` | Zip vectors | zip1 | — |
| `vzip1_u32` | Zip vectors | zip1 | — |
| `vzip1_u8` | Zip vectors | zip1 | — |
| `vzip1q_f32` | Zip vectors | zip1 | — |
| `vzip1q_f64` | Zip vectors | zip1 | — |
| `vzip1q_p16` | Zip vectors | zip1 | — |
| `vzip1q_p64` | Zip vectors | zip1 | — |
| `vzip1q_p8` | Zip vectors | zip1 | — |
| `vzip1q_s16` | Zip vectors | zip1 | — |
| `vzip1q_s32` | Zip vectors | zip1 | — |
| `vzip1q_s64` | Zip vectors | zip1 | — |
| `vzip1q_s8` | Zip vectors | zip1 | — |
| `vzip1q_u16` | Zip vectors | zip1 | — |
| `vzip1q_u32` | Zip vectors | zip1 | — |
| `vzip1q_u64` | Zip vectors | zip1 | — |
| `vzip1q_u8` | Zip vectors | zip1 | — |
| `vzip2_f32` | Zip vectors | zip2 | — |
| `vzip2_p16` | Zip vectors | zip2 | — |
| `vzip2_p8` | Zip vectors | zip2 | — |
| `vzip2_s16` | Zip vectors | zip2 | — |
| `vzip2_s32` | Zip vectors | zip2 | — |
| `vzip2_s8` | Zip vectors | zip2 | — |
| `vzip2_u16` | Zip vectors | zip2 | — |
| `vzip2_u32` | Zip vectors | zip2 | — |
| `vzip2_u8` | Zip vectors | zip2 | — |
| `vzip2q_f32` | Zip vectors | zip2 | — |
| `vzip2q_f64` | Zip vectors | zip2 | — |
| `vzip2q_p16` | Zip vectors | zip2 | — |
| `vzip2q_p64` | Zip vectors | zip2 | — |
| `vzip2q_p8` | Zip vectors | zip2 | — |
| `vzip2q_s16` | Zip vectors | zip2 | — |
| `vzip2q_s32` | Zip vectors | zip2 | — |
| `vzip2q_s64` | Zip vectors | zip2 | — |
| `vzip2q_s8` | Zip vectors | zip2 | — |
| `vzip2q_u16` | Zip vectors | zip2 | — |
| `vzip2q_u32` | Zip vectors | zip2 | — |
| `vzip2q_u64` | Zip vectors | zip2 | — |
| `vzip2q_u8` | Zip vectors | zip2 | — |
| `vzip_f32` | Zip vectors | vtrn | — |
| `vzip_p16` | Zip vectors | vzip | — |
| `vzip_p8` | Zip vectors | vzip | — |
| `vzip_s16` | Zip vectors | vzip | — |
| `vzip_s32` | Zip vectors | vtrn | — |
| `vzip_s8` | Zip vectors | vzip | — |
| `vzip_u16` | Zip vectors | vzip | — |
| `vzip_u32` | Zip vectors | vtrn | — |
| `vzip_u8` | Zip vectors | vzip | — |
| `vzipq_f32` | Zip vectors | vorr | — |
| `vzipq_p16` | Zip vectors | vorr | — |
| `vzipq_p8` | Zip vectors | vorr | — |
| `vzipq_s16` | Zip vectors | vorr | — |
| `vzipq_s32` | Zip vectors | vorr | — |
| `vzipq_s8` | Zip vectors | vorr | — |
| `vzipq_u16` | Zip vectors | vorr | — |
| `vzipq_u32` | Zip vectors | vorr | — |
| `vzipq_u8` | Zip vectors | vorr | — |

### Stable, Unsafe (522 intrinsics) — use import_intrinsics for safe versions

| Name | Description | Safe Variant |
|------|-------------|--------------|
| `vext_s64` | Extract vector from pair of vectors | — |
| `vext_u64` | Extract vector from pair of vectors | — |
| `vld1_dup_f32` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1_dup_f64` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_dup_p16` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1_dup_p8` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1_dup_s16` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1_dup_s32` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1_dup_s64` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1_dup_s8` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1_dup_u16` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1_dup_u32` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1_dup_u64` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1_dup_u8` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1_f32_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_f32_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_f32_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_f64` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_f64_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_f64_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_f64_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_lane_f32` | Load one single-element structure to one lane of one registe... | — |
| `vld1_lane_f64` | Load one single-element structure to one lane of one registe... | — |
| `vld1_lane_p16` | Load one single-element structure to one lane of one registe... | — |
| `vld1_lane_p8` | Load one single-element structure to one lane of one registe... | — |
| `vld1_lane_s16` | Load one single-element structure to one lane of one registe... | — |
| `vld1_lane_s32` | Load one single-element structure to one lane of one registe... | — |
| `vld1_lane_s64` | Load one single-element structure to one lane of one registe... | — |
| `vld1_lane_s8` | Load one single-element structure to one lane of one registe... | — |
| `vld1_lane_u16` | Load one single-element structure to one lane of one registe... | — |
| `vld1_lane_u32` | Load one single-element structure to one lane of one registe... | — |
| `vld1_lane_u64` | Load one single-element structure to one lane of one registe... | — |
| `vld1_lane_u8` | Load one single-element structure to one lane of one registe... | — |
| `vld1_p16_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_p16_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_p16_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_p8_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_p8_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_p8_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_s16_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_s16_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_s16_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_s32_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_s32_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_s32_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_s64_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_s64_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_s64_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_s8_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_s8_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_s8_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_u16_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_u16_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_u16_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_u32_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_u32_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_u32_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_u64_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_u64_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_u64_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_u8_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_u8_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1_u8_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_dup_f32` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1q_dup_f64` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_dup_p16` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1q_dup_p8` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1q_dup_s16` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1q_dup_s32` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1q_dup_s64` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1q_dup_s8` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1q_dup_u16` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1q_dup_u32` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1q_dup_u64` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1q_dup_u8` | Load one single-element structure and Replicate to all lanes... | — |
| `vld1q_f32_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_f32_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_f32_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_f64` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_f64_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_f64_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_f64_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_lane_f32` | Load one single-element structure to one lane of one registe... | — |
| `vld1q_lane_f64` | Load one single-element structure to one lane of one registe... | — |
| `vld1q_lane_p16` | Load one single-element structure to one lane of one registe... | — |
| `vld1q_lane_p8` | Load one single-element structure to one lane of one registe... | — |
| `vld1q_lane_s16` | Load one single-element structure to one lane of one registe... | — |
| `vld1q_lane_s32` | Load one single-element structure to one lane of one registe... | — |
| `vld1q_lane_s64` | Load one single-element structure to one lane of one registe... | — |
| `vld1q_lane_s8` | Load one single-element structure to one lane of one registe... | — |
| `vld1q_lane_u16` | Load one single-element structure to one lane of one registe... | — |
| `vld1q_lane_u32` | Load one single-element structure to one lane of one registe... | — |
| `vld1q_lane_u64` | Load one single-element structure to one lane of one registe... | — |
| `vld1q_lane_u8` | Load one single-element structure to one lane of one registe... | — |
| `vld1q_p16_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_p16_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_p16_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_p8_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_p8_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_p8_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_s16_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_s16_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_s16_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_s32_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_s32_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_s32_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_s64_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_s64_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_s64_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_s8_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_s8_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_s8_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_u16_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_u16_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_u16_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_u32_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_u32_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_u32_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_u64_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_u64_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_u64_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_u8_x2` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_u8_x3` | Load multiple single-element structures to one, two, three, ... | — |
| `vld1q_u8_x4` | Load multiple single-element structures to one, two, three, ... | — |
| `vld2_dup_f64` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2_dup_p16` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2_dup_p8` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2_dup_u16` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2_dup_u32` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2_dup_u64` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2_dup_u8` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2_f64` | Load multiple 2-element structures to two registers | — |
| `vld2_lane_f32` | Load multiple 2-element structures to two registers | — |
| `vld2_lane_f64` | Load multiple 2-element structures to two registers | — |
| `vld2_lane_p16` | Load multiple 2-element structures to two registers | — |
| `vld2_lane_p8` | Load multiple 2-element structures to two registers | — |
| `vld2_lane_s16` | Load multiple 2-element structures to two registers | — |
| `vld2_lane_s32` | Load multiple 2-element structures to two registers | — |
| `vld2_lane_s64` | Load multiple 2-element structures to two registers | — |
| `vld2_lane_s8` | Load multiple 2-element structures to two registers | — |
| `vld2_lane_u16` | Load multiple 2-element structures to two registers | — |
| `vld2_lane_u32` | Load multiple 2-element structures to two registers | — |
| `vld2_lane_u64` | Load multiple 2-element structures to two registers | — |
| `vld2_lane_u8` | Load multiple 2-element structures to two registers | — |
| `vld2_p16` | Load multiple 2-element structures to two registers | — |
| `vld2_p8` | Load multiple 2-element structures to two registers | — |
| `vld2_u16` | Load multiple 2-element structures to two registers | — |
| `vld2_u32` | Load multiple 2-element structures to two registers | — |
| `vld2_u64` | Load multiple 2-element structures to two registers | — |
| `vld2_u8` | Load multiple 2-element structures to two registers | — |
| `vld2q_dup_f64` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2q_dup_p16` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2q_dup_p8` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2q_dup_s64` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2q_dup_u16` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2q_dup_u32` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2q_dup_u64` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2q_dup_u8` | Load single 2-element structure and replicate to all lanes o... | — |
| `vld2q_f64` | Load multiple 2-element structures to two registers | — |
| `vld2q_lane_f32` | Load multiple 2-element structures to two registers | — |
| `vld2q_lane_f64` | Load multiple 2-element structures to two registers | — |
| `vld2q_lane_p16` | Load multiple 2-element structures to two registers | — |
| `vld2q_lane_p8` | Load multiple 2-element structures to two registers | — |
| `vld2q_lane_s16` | Load multiple 2-element structures to two registers | — |
| `vld2q_lane_s32` | Load multiple 2-element structures to two registers | — |
| `vld2q_lane_s64` | Load multiple 2-element structures to two registers | — |
| `vld2q_lane_s8` | Load multiple 2-element structures to two registers | — |
| `vld2q_lane_u16` | Load multiple 2-element structures to two registers | — |
| `vld2q_lane_u32` | Load multiple 2-element structures to two registers | — |
| `vld2q_lane_u64` | Load multiple 2-element structures to two registers | — |
| `vld2q_lane_u8` | Load multiple 2-element structures to two registers | — |
| `vld2q_p16` | Load multiple 2-element structures to two registers | — |
| `vld2q_p8` | Load multiple 2-element structures to two registers | — |
| `vld2q_s64` | Load multiple 2-element structures to two registers | — |
| `vld2q_u16` | Load multiple 2-element structures to two registers | — |
| `vld2q_u32` | Load multiple 2-element structures to two registers | — |
| `vld2q_u64` | Load multiple 2-element structures to two registers | — |
| `vld2q_u8` | Load multiple 2-element structures to two registers | — |
| `vld3_dup_f32` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3_dup_f64` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3_dup_p16` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3_dup_p8` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3_dup_s16` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3_dup_s32` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3_dup_s64` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3_dup_s8` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3_dup_u16` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3_dup_u32` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3_dup_u64` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3_dup_u8` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3_f32` | Load multiple 3-element structures to three registers | — |
| `vld3_f64` | Load multiple 3-element structures to three registers | — |
| `vld3_lane_f32` | Load multiple 3-element structures to three registers | — |
| `vld3_lane_f64` | Load multiple 3-element structures to three registers | — |
| `vld3_lane_p16` | Load multiple 3-element structures to three registers | — |
| `vld3_lane_p8` | Load multiple 3-element structures to three registers | — |
| `vld3_lane_s16` | Load multiple 3-element structures to two registers | — |
| `vld3_lane_s32` | Load multiple 3-element structures to two registers | — |
| `vld3_lane_s64` | Load multiple 3-element structures to two registers | — |
| `vld3_lane_s8` | Load multiple 3-element structures to two registers | — |
| `vld3_lane_u16` | Load multiple 3-element structures to three registers | — |
| `vld3_lane_u32` | Load multiple 3-element structures to three registers | — |
| `vld3_lane_u64` | Load multiple 3-element structures to three registers | — |
| `vld3_lane_u8` | Load multiple 3-element structures to three registers | — |
| `vld3_p16` | Load multiple 3-element structures to three registers | — |
| `vld3_p8` | Load multiple 3-element structures to three registers | — |
| `vld3_s16` | Load multiple 3-element structures to three registers | — |
| `vld3_s32` | Load multiple 3-element structures to three registers | — |
| `vld3_s64` | Load multiple 3-element structures to three registers | — |
| `vld3_s8` | Load multiple 3-element structures to three registers | — |
| `vld3_u16` | Load multiple 3-element structures to three registers | — |
| `vld3_u32` | Load multiple 3-element structures to three registers | — |
| `vld3_u64` | Load multiple 3-element structures to three registers | — |
| `vld3_u8` | Load multiple 3-element structures to three registers | — |
| `vld3q_dup_f32` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3q_dup_f64` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3q_dup_p16` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3q_dup_p8` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3q_dup_s16` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3q_dup_s32` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3q_dup_s64` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3q_dup_s8` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3q_dup_u16` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3q_dup_u32` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3q_dup_u64` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3q_dup_u8` | Load single 3-element structure and replicate to all lanes o... | — |
| `vld3q_f32` | Load multiple 3-element structures to three registers | — |
| `vld3q_f64` | Load multiple 3-element structures to three registers | — |
| `vld3q_lane_f32` | Load multiple 3-element structures to three registers | — |
| `vld3q_lane_f64` | Load multiple 3-element structures to three registers | — |
| `vld3q_lane_p16` | Load multiple 3-element structures to three registers | — |
| `vld3q_lane_p8` | Load multiple 3-element structures to three registers | — |
| `vld3q_lane_s16` | Load multiple 3-element structures to two registers | — |
| `vld3q_lane_s32` | Load multiple 3-element structures to two registers | — |
| `vld3q_lane_s64` | Load multiple 3-element structures to two registers | — |
| `vld3q_lane_s8` | Load multiple 3-element structures to two registers | — |
| `vld3q_lane_u16` | Load multiple 3-element structures to three registers | — |
| `vld3q_lane_u32` | Load multiple 3-element structures to three registers | — |
| `vld3q_lane_u64` | Load multiple 3-element structures to three registers | — |
| `vld3q_lane_u8` | Load multiple 3-element structures to three registers | — |
| `vld3q_p16` | Load multiple 3-element structures to three registers | — |
| `vld3q_p8` | Load multiple 3-element structures to three registers | — |
| `vld3q_s16` | Load multiple 3-element structures to three registers | — |
| `vld3q_s32` | Load multiple 3-element structures to three registers | — |
| `vld3q_s64` | Load multiple 3-element structures to three registers | — |
| `vld3q_s8` | Load multiple 3-element structures to three registers | — |
| `vld3q_u16` | Load multiple 3-element structures to three registers | — |
| `vld3q_u32` | Load multiple 3-element structures to three registers | — |
| `vld3q_u64` | Load multiple 3-element structures to three registers | — |
| `vld3q_u8` | Load multiple 3-element structures to three registers | — |
| `vld4_dup_f64` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4_dup_p16` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4_dup_p8` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4_dup_s64` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4_dup_u16` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4_dup_u32` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4_dup_u64` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4_dup_u8` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4_f32` | Load multiple 4-element structures to four registers | — |
| `vld4_f64` | Load multiple 4-element structures to four registers | — |
| `vld4_lane_f32` | Load multiple 4-element structures to four registers | — |
| `vld4_lane_f64` | Load multiple 4-element structures to four registers | — |
| `vld4_lane_p16` | Load multiple 4-element structures to four registers | — |
| `vld4_lane_p8` | Load multiple 4-element structures to four registers | — |
| `vld4_lane_s16` | Load multiple 4-element structures to four registers | — |
| `vld4_lane_s32` | Load multiple 4-element structures to four registers | — |
| `vld4_lane_s64` | Load multiple 4-element structures to four registers | — |
| `vld4_lane_s8` | Load multiple 4-element structures to four registers | — |
| `vld4_lane_u16` | Load multiple 4-element structures to four registers | — |
| `vld4_lane_u32` | Load multiple 4-element structures to four registers | — |
| `vld4_lane_u64` | Load multiple 4-element structures to four registers | — |
| `vld4_lane_u8` | Load multiple 4-element structures to four registers | — |
| `vld4_p16` | Load multiple 4-element structures to four registers | — |
| `vld4_p8` | Load multiple 4-element structures to four registers | — |
| `vld4_s16` | Load multiple 4-element structures to four registers | — |
| `vld4_s32` | Load multiple 4-element structures to four registers | — |
| `vld4_s64` | Load multiple 4-element structures to four registers | — |
| `vld4_s8` | Load multiple 4-element structures to four registers | — |
| `vld4_u16` | Load multiple 4-element structures to four registers | — |
| `vld4_u32` | Load multiple 4-element structures to four registers | — |
| `vld4_u64` | Load multiple 4-element structures to four registers | — |
| `vld4_u8` | Load multiple 4-element structures to four registers | — |
| `vld4q_dup_f64` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4q_dup_p16` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4q_dup_p8` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4q_dup_s64` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4q_dup_u16` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4q_dup_u32` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4q_dup_u64` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4q_dup_u8` | Load single 4-element structure and replicate to all lanes o... | — |
| `vld4q_f32` | Load multiple 4-element structures to four registers | — |
| `vld4q_f64` | Load multiple 4-element structures to four registers | — |
| `vld4q_lane_f32` | Load multiple 4-element structures to four registers | — |
| `vld4q_lane_f64` | Load multiple 4-element structures to four registers | — |
| `vld4q_lane_p16` | Load multiple 4-element structures to four registers | — |
| `vld4q_lane_p8` | Load multiple 4-element structures to four registers | — |
| `vld4q_lane_s16` | Load multiple 4-element structures to four registers | — |
| `vld4q_lane_s32` | Load multiple 4-element structures to four registers | — |
| `vld4q_lane_s64` | Load multiple 4-element structures to four registers | — |
| `vld4q_lane_s8` | Load multiple 4-element structures to four registers | — |
| `vld4q_lane_u16` | Load multiple 4-element structures to four registers | — |
| `vld4q_lane_u32` | Load multiple 4-element structures to four registers | — |
| `vld4q_lane_u64` | Load multiple 4-element structures to four registers | — |
| `vld4q_lane_u8` | Load multiple 4-element structures to four registers | — |
| `vld4q_p16` | Load multiple 4-element structures to four registers | — |
| `vld4q_p8` | Load multiple 4-element structures to four registers | — |
| `vld4q_s16` | Load multiple 4-element structures to four registers | — |
| `vld4q_s32` | Load multiple 4-element structures to four registers | — |
| `vld4q_s64` | Load multiple 4-element structures to four registers | — |
| `vld4q_s8` | Load multiple 4-element structures to four registers | — |
| `vld4q_u16` | Load multiple 4-element structures to four registers | — |
| `vld4q_u32` | Load multiple 4-element structures to four registers | — |
| `vld4q_u64` | Load multiple 4-element structures to four registers | — |
| `vld4q_u8` | Load multiple 4-element structures to four registers | — |
| `vldrq_p128` | Store SIMD&FP register (immediate offset) | — |
| `vst1_f32_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_f64` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_f64_x2` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_f64_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_f64_x4` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_lane_f32` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_lane_f64` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_lane_p16` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_lane_p8` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_lane_s16` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_lane_s32` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_lane_s64` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_lane_s8` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_lane_u16` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_lane_u32` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_lane_u64` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_lane_u8` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_p16_x2` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_p16_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_p16_x4` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_p8_x2` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_p8_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_p8_x4` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_s16_x2` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_s16_x3` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_s16_x4` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_s32_x2` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_s32_x3` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_s32_x4` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_s64_x2` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_s64_x3` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_s64_x4` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_s8_x2` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_s8_x3` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_s8_x4` | Store multiple single-element structures from one, two, thre... | — |
| `vst1_u16_x2` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_u16_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_u16_x4` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_u32_x2` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_u32_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_u32_x4` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_u64_x2` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_u64_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_u64_x4` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_u8_x2` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_u8_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1_u8_x4` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_f32_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_f64` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_f64_x2` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_f64_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_f64_x4` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_lane_f32` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_lane_f64` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_lane_p16` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_lane_p8` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_lane_s16` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_lane_s32` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_lane_s64` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_lane_s8` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_lane_u16` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_lane_u32` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_lane_u64` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_lane_u8` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_p16_x2` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_p16_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_p16_x4` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_p8_x2` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_p8_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_p8_x4` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_s16_x2` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_s16_x3` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_s16_x4` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_s32_x2` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_s32_x3` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_s32_x4` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_s64_x2` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_s64_x3` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_s64_x4` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_s8_x2` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_s8_x3` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_s8_x4` | Store multiple single-element structures from one, two, thre... | — |
| `vst1q_u16_x2` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_u16_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_u16_x4` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_u32_x2` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_u32_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_u32_x4` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_u64_x2` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_u64_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_u64_x4` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_u8_x2` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_u8_x3` | Store multiple single-element structures to one, two, three,... | — |
| `vst1q_u8_x4` | Store multiple single-element structures to one, two, three,... | — |
| `vst2_f32` | Store multiple 2-element structures from two registers | — |
| `vst2_f64` | Store multiple 2-element structures from two registers | — |
| `vst2_lane_f32` | Store multiple 2-element structures from two registers | — |
| `vst2_lane_f64` | Store multiple 2-element structures from two registers | — |
| `vst2_lane_p16` | Store multiple 2-element structures from two registers | — |
| `vst2_lane_p8` | Store multiple 2-element structures from two registers | — |
| `vst2_lane_s16` | Store multiple 2-element structures from two registers | — |
| `vst2_lane_s32` | Store multiple 2-element structures from two registers | — |
| `vst2_lane_s64` | Store multiple 2-element structures from two registers | — |
| `vst2_lane_s8` | Store multiple 2-element structures from two registers | — |
| `vst2_lane_u16` | Store multiple 2-element structures from two registers | — |
| `vst2_lane_u32` | Store multiple 2-element structures from two registers | — |
| `vst2_lane_u64` | Store multiple 2-element structures from two registers | — |
| `vst2_lane_u8` | Store multiple 2-element structures from two registers | — |
| `vst2_p16` | Store multiple 2-element structures from two registers | — |
| `vst2_p8` | Store multiple 2-element structures from two registers | — |
| `vst2_s16` | Store multiple 2-element structures from two registers | — |
| `vst2_s32` | Store multiple 2-element structures from two registers | — |
| `vst2_s8` | Store multiple 2-element structures from two registers | — |
| `vst2_u16` | Store multiple 2-element structures from two registers | — |
| `vst2_u32` | Store multiple 2-element structures from two registers | — |
| `vst2_u64` | Store multiple 2-element structures from two registers | — |
| `vst2_u8` | Store multiple 2-element structures from two registers | — |
| `vst2q_f32` | Store multiple 2-element structures from two registers | — |
| `vst2q_f64` | Store multiple 2-element structures from two registers | — |
| `vst2q_lane_f32` | Store multiple 2-element structures from two registers | — |
| `vst2q_lane_f64` | Store multiple 2-element structures from two registers | — |
| `vst2q_lane_p16` | Store multiple 2-element structures from two registers | — |
| `vst2q_lane_p8` | Store multiple 2-element structures from two registers | — |
| `vst2q_lane_s16` | Store multiple 2-element structures from two registers | — |
| `vst2q_lane_s32` | Store multiple 2-element structures from two registers | — |
| `vst2q_lane_s64` | Store multiple 2-element structures from two registers | — |
| `vst2q_lane_s8` | Store multiple 2-element structures from two registers | — |
| `vst2q_lane_u16` | Store multiple 2-element structures from two registers | — |
| `vst2q_lane_u32` | Store multiple 2-element structures from two registers | — |
| `vst2q_lane_u64` | Store multiple 2-element structures from two registers | — |
| `vst2q_lane_u8` | Store multiple 2-element structures from two registers | — |
| `vst2q_p16` | Store multiple 2-element structures from two registers | — |
| `vst2q_p8` | Store multiple 2-element structures from two registers | — |
| `vst2q_s16` | Store multiple 2-element structures from two registers | — |
| `vst2q_s32` | Store multiple 2-element structures from two registers | — |
| `vst2q_s64` | Store multiple 2-element structures from two registers | — |
| `vst2q_s8` | Store multiple 2-element structures from two registers | — |
| `vst2q_u16` | Store multiple 2-element structures from two registers | — |
| `vst2q_u32` | Store multiple 2-element structures from two registers | — |
| `vst2q_u64` | Store multiple 2-element structures from two registers | — |
| `vst2q_u8` | Store multiple 2-element structures from two registers | — |
| `vst3_f64` | Store multiple 3-element structures from three registers | — |
| `vst3_lane_f64` | Store multiple 3-element structures from three registers | — |
| `vst3_lane_p16` | Store multiple 3-element structures from three registers | — |
| `vst3_lane_p8` | Store multiple 3-element structures from three registers | — |
| `vst3_lane_s64` | Store multiple 3-element structures from three registers | — |
| `vst3_lane_u16` | Store multiple 3-element structures from three registers | — |
| `vst3_lane_u32` | Store multiple 3-element structures from three registers | — |
| `vst3_lane_u64` | Store multiple 3-element structures from three registers | — |
| `vst3_lane_u8` | Store multiple 3-element structures from three registers | — |
| `vst3_p16` | Store multiple 3-element structures from three registers | — |
| `vst3_p8` | Store multiple 3-element structures from three registers | — |
| `vst3_s64` | Store multiple 3-element structures from three registers | — |
| `vst3_u16` | Store multiple 3-element structures from three registers | — |
| `vst3_u32` | Store multiple 3-element structures from three registers | — |
| `vst3_u64` | Store multiple 3-element structures from three registers | — |
| `vst3_u8` | Store multiple 3-element structures from three registers | — |
| `vst3q_f64` | Store multiple 3-element structures from three registers | — |
| `vst3q_lane_f64` | Store multiple 3-element structures from three registers | — |
| `vst3q_lane_p16` | Store multiple 3-element structures from three registers | — |
| `vst3q_lane_p8` | Store multiple 3-element structures from three registers | — |
| `vst3q_lane_s64` | Store multiple 3-element structures from three registers | — |
| `vst3q_lane_s8` | Store multiple 3-element structures from three registers | — |
| `vst3q_lane_u16` | Store multiple 3-element structures from three registers | — |
| `vst3q_lane_u32` | Store multiple 3-element structures from three registers | — |
| `vst3q_lane_u64` | Store multiple 3-element structures from three registers | — |
| `vst3q_lane_u8` | Store multiple 3-element structures from three registers | — |
| `vst3q_p16` | Store multiple 3-element structures from three registers | — |
| `vst3q_p8` | Store multiple 3-element structures from three registers | — |
| `vst3q_s64` | Store multiple 3-element structures from three registers | — |
| `vst3q_u16` | Store multiple 3-element structures from three registers | — |
| `vst3q_u32` | Store multiple 3-element structures from three registers | — |
| `vst3q_u64` | Store multiple 3-element structures from three registers | — |
| `vst3q_u8` | Store multiple 3-element structures from three registers | — |
| `vst4_f64` | Store multiple 4-element structures from four registers | — |
| `vst4_lane_f64` | Store multiple 4-element structures from four registers | — |
| `vst4_lane_p16` | Store multiple 4-element structures from four registers | — |
| `vst4_lane_p8` | Store multiple 4-element structures from four registers | — |
| `vst4_lane_s64` | Store multiple 4-element structures from four registers | — |
| `vst4_lane_u16` | Store multiple 4-element structures from four registers | — |
| `vst4_lane_u32` | Store multiple 4-element structures from four registers | — |
| `vst4_lane_u64` | Store multiple 4-element structures from four registers | — |
| `vst4_lane_u8` | Store multiple 4-element structures from four registers | — |
| `vst4_p16` | Store multiple 4-element structures from four registers | — |
| `vst4_p8` | Store multiple 4-element structures from four registers | — |
| `vst4_u16` | Store multiple 4-element structures from four registers | — |
| `vst4_u32` | Store multiple 4-element structures from four registers | — |
| `vst4_u64` | Store multiple 4-element structures from four registers | — |
| `vst4_u8` | Store multiple 4-element structures from four registers | — |
| `vst4q_f64` | Store multiple 4-element structures from four registers | — |
| `vst4q_lane_f64` | Store multiple 4-element structures from four registers | — |
| `vst4q_lane_p16` | Store multiple 4-element structures from four registers | — |
| `vst4q_lane_p8` | Store multiple 4-element structures from four registers | — |
| `vst4q_lane_s64` | Store multiple 4-element structures from four registers | — |
| `vst4q_lane_s8` | Store multiple 4-element structures from four registers | — |
| `vst4q_lane_u16` | Store multiple 4-element structures from four registers | — |
| `vst4q_lane_u32` | Store multiple 4-element structures from four registers | — |
| `vst4q_lane_u64` | Store multiple 4-element structures from four registers | — |
| `vst4q_lane_u8` | Store multiple 4-element structures from four registers | — |
| `vst4q_p16` | Store multiple 4-element structures from four registers | — |
| `vst4q_p8` | Store multiple 4-element structures from four registers | — |
| `vst4q_s64` | Store multiple 4-element structures from four registers | — |
| `vst4q_u16` | Store multiple 4-element structures from four registers | — |
| `vst4q_u32` | Store multiple 4-element structures from four registers | — |
| `vst4q_u64` | Store multiple 4-element structures from four registers | — |
| `vst4q_u8` | Store multiple 4-element structures from four registers | — |
| `vstrq_p128` | Store SIMD&FP register (immediate offset) | — |

### Unstable/Nightly (36 intrinsics)

| Name | Description | Instruction |
|------|-------------|-------------|
| `vst1_f32` | Store multiple single-element structures from one, two, thre... | "vst1.32" |
| `vst1_p16` | Store multiple single-element structures from one, two, thre... | "vst1.16" |
| `vst1_p64` | Store multiple single-element structures from one, two, thre... | "vst1.64" |
| `vst1_p8` | Store multiple single-element structures from one, two, thre... | "vst1.8" |
| `vst1_s16` | Store multiple single-element structures from one, two, thre... | "vst1.16" |
| `vst1_s32` | Store multiple single-element structures from one, two, thre... | "vst1.32" |
| `vst1_s64` | Store multiple single-element structures from one, two, thre... | "vst1.64" |
| `vst1_s8` | Store multiple single-element structures from one, two, thre... | "vst1.8" |
| `vst1_u16` | Store multiple single-element structures from one, two, thre... | "vst1.16" |
| `vst1_u32` | Store multiple single-element structures from one, two, thre... | "vst1.32" |
| `vst1_u64` | Store multiple single-element structures from one, two, thre... | "vst1.64" |
| `vst1_u8` | Store multiple single-element structures from one, two, thre... | "vst1.8" |
| `vst1q_f32` | Store multiple single-element structures from one, two, thre... | "vst1.32" |
| `vst1q_p16` | Store multiple single-element structures from one, two, thre... | "vst1.16" |
| `vst1q_p64` | Store multiple single-element structures from one, two, thre... | "vst1.64" |
| `vst1q_p8` | Store multiple single-element structures from one, two, thre... | "vst1.8" |
| `vst1q_s16` | Store multiple single-element structures from one, two, thre... | "vst1.16" |
| `vst1q_s32` | Store multiple single-element structures from one, two, thre... | "vst1.32" |
| `vst1q_s64` | Store multiple single-element structures from one, two, thre... | "vst1.64" |
| `vst1q_s8` | Store multiple single-element structures from one, two, thre... | "vst1.8" |
| `vst1q_u16` | Store multiple single-element structures from one, two, thre... | "vst1.16" |
| `vst1q_u32` | Store multiple single-element structures from one, two, thre... | "vst1.32" |
| `vst1q_u64` | Store multiple single-element structures from one, two, thre... | "vst1.64" |
| `vst1q_u8` | Store multiple single-element structures from one, two, thre... | "vst1.8" |
| `vtbl1_p8` | Table look-up | vtbl |
| `vtbl1_s8` | Table look-up | vtbl |
| `vtbl1_u8` | Table look-up | vtbl |
| `vtbl2_p8` | Table look-up | vtbl |
| `vtbl2_s8` | Table look-up | vtbl |
| `vtbl2_u8` | Table look-up | vtbl |
| `vtbl3_p8` | Table look-up | vtbl |
| `vtbl3_s8` | Table look-up | vtbl |
| `vtbl3_u8` | Table look-up | vtbl |
| `vtbl4_p8` | Table look-up | vtbl |
| `vtbl4_s8` | Table look-up | vtbl |
| `vtbl4_u8` | Table look-up | vtbl |



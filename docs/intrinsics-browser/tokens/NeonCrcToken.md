# NeonCrcToken — NEON+CRC

Proof that NEON + CRC is available.

**Architecture:** aarch64 | **Features:** neon, crc
**Total intrinsics:** 8 (8 safe, 0 unsafe, 6 stable, 2 unstable/unknown)

## Usage

```rust
use archmage::prelude::*;

if let Some(token) = NeonCrcToken::summon() {
    process(token, &mut data);
}

#[arcane(import_intrinsics)]  // Entry point only
fn process(token: NeonCrcToken, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(4) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

#[rite(import_intrinsics)]  // All inner helpers
fn process_chunk(_: NeonCrcToken, chunk: &mut [f32; 4]) {
    let v = vld1q_f32(chunk);  // safe!
    let doubled = vaddq_f32(v, v);  // value intrinsic (safe inside #[rite])
    vst1q_f32(chunk, doubled);  // safe!
}
// No unsafe anywhere. Use #![forbid(unsafe_code)] in your crate.
```



## All Intrinsics

### Stable, Safe (6 intrinsics)

| Name | Description | Instruction | Timing (H/Z4) |
|------|-------------|-------------|---------------|
| `__crc32b` | CRC32 single round checksum for bytes (8 bits) | crc32b | — |
| `__crc32cb` | CRC32-C single round checksum for bytes (8 bits) | crc32cb | — |
| `__crc32ch` | CRC32-C single round checksum for bytes (16 bits) | crc32ch | — |
| `__crc32cw` | CRC32-C single round checksum for bytes (32 bits) | crc32cw | — |
| `__crc32h` | CRC32 single round checksum for bytes (16 bits) | crc32h | — |
| `__crc32w` | CRC32 single round checksum for bytes (32 bits) | crc32w | — |

### Unstable/Nightly (2 intrinsics)

| Name | Description | Instruction |
|------|-------------|-------------|
| `__crc32cd` | CRC32-C single round checksum for quad words (64 bits) | crc32cw |
| `__crc32d` | CRC32 single round checksum for quad words (64 bits) | crc32w |



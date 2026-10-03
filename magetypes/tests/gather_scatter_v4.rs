//! AVX-512 `gather_wrapping`, `gather_or` and `scatter_select` against scalar
//! indexing, with hostile indices: negative values read as unsigned, the 2^31
//! boundary, `u32::MAX`, indices at and just past the slice end, empty slices,
//! and frequent duplicate scatter targets.
//!
//! Runs where the CPU has AVX-512. Without one, use Intel SDE:
//! `sde64 -icl -- cargo test -p magetypes --features avx512 --test gather_scatter_v4`.
#![cfg(all(target_arch = "x86_64", feature = "avx512"))]

use archmage::{SimdToken, X64V4Token};
use magetypes::simd::generic::{f32x16, i32x16, u32x16};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    /// An index biased toward the cases that break unsanitized gathers.
    fn index(&mut self, len: usize) -> u32 {
        let r = self.next();
        match r % 8 {
            0 => (-1 - (self.next() % 100) as i32) as u32,
            1 => i32::MAX as u32 - self.next() % 3,
            2 => (1u32 << 31) + self.next() % 3,
            3 => u32::MAX - self.next() % 3,
            4 => (len as u32).wrapping_add(self.next() % 3).wrapping_sub(1),
            5 => self.next() % 4,
            _ => self.next() % (2 * len as u32 + 1),
        }
    }

    fn indices(&mut self, len: usize) -> [u32; 16] {
        core::array::from_fn(|_| self.index(len))
    }
}

/// Scalar reference for `gather_or`: lanes below `min(len, 2^31)` read the
/// table, the rest keep `or`.
fn gather_or_ref<E: Copy>(table: &[E], idx: &[u32; 16], or: &[E; 16]) -> [E; 16] {
    core::array::from_fn(|i| {
        let j = idx[i] as usize;
        if j < table.len().min(1 << 31) {
            table[j]
        } else {
            or[i]
        }
    })
}

/// Scalar reference for `scatter_select`: enabled, in-range lanes write in
/// lane order, so the highest lane wins on a shared index.
fn scatter_ref<E: Copy>(dst: &mut [E], enable: u16, idx: &[u32; 16], v: &[E; 16]) {
    for i in 0..16 {
        let j = idx[i] as usize;
        if enable >> i & 1 == 1 && j < dst.len().min(1 << 31) {
            dst[j] = v[i];
        }
    }
}

/// `f32` values whose bits must survive a gather or scatter unchanged:
/// signed zeros, infinities, subnormals and NaNs with payloads.
fn f32_value(i: usize) -> f32 {
    match i % 7 {
        0 => -0.0,
        1 => f32::INFINITY,
        2 => f32::from_bits(0x0000_0001),
        3 => f32::from_bits(0x7fc0_1234),
        4 => f32::from_bits(0xffa0_0042),
        _ => i as f32 * -1.25,
    }
}

const SLICE_LENS: [usize; 5] = [0, 1, 2, 37, 1000];

macro_rules! check_token {
    ($token:ty, $t:expr) => {{
        let t: $token = $t;
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let u32_table: Vec<u32> = (0..1024u32)
            .map(|i| i.wrapping_mul(2_654_435_761))
            .collect();
        let i32_table: Vec<i32> = (0..1024i32).map(|i| i * 7 - 3000).collect();
        let f32_table: Vec<f32> = (0..1024).map(f32_value).collect();

        for _ in 0..2_000 {
            // gather_wrapping over power-of-two tables, including N = 1.
            let idx = rng.indices(1024);
            let iv = u32x16::<$token>::from_array_t(t, idx);
            let u: &[u32; 1024] = u32_table.as_slice().try_into().unwrap();
            let got = u32x16::<$token>::gather_wrapping(u, iv).to_array();
            assert_eq!(got, core::array::from_fn(|i| u[idx[i] as usize & 1023]));
            let s: &[i32; 16] = i32_table[..16].try_into().unwrap();
            let got = i32x16::<$token>::gather_wrapping(s, iv).to_array();
            assert_eq!(got, core::array::from_fn(|i| s[idx[i] as usize & 15]));
            let one: &[f32; 1] = f32_table[3..4].try_into().unwrap();
            let got = f32x16::<$token>::gather_wrapping(one, iv).to_array();
            assert_eq!(got.map(f32::to_bits), [one[0].to_bits(); 16]);
            let f: &[f32; 1024] = f32_table.as_slice().try_into().unwrap();
            let got = f32x16::<$token>::gather_wrapping(f, iv).to_array();
            let want: [f32; 16] = core::array::from_fn(|i| f[idx[i] as usize & 1023]);
            assert_eq!(got.map(f32::to_bits), want.map(f32::to_bits));

            for len in SLICE_LENS {
                let idx = rng.indices(len);
                let iv = u32x16::<$token>::from_array_t(t, idx);
                let enable = rng.next() as u16;

                // gather_or
                let or_u: [u32; 16] = core::array::from_fn(|i| 0xdead_0000 + i as u32);
                let got = u32x16::<$token>::gather_or(
                    &u32_table[..len],
                    iv,
                    u32x16::<$token>::from_array_t(t, or_u),
                )
                .to_array();
                assert_eq!(
                    got,
                    gather_or_ref(&u32_table[..len], &idx, &or_u),
                    "len {len}"
                );
                let or_i: [i32; 16] = core::array::from_fn(|i| -1 - i as i32);
                let got = i32x16::<$token>::gather_or(
                    &i32_table[..len],
                    iv,
                    i32x16::<$token>::from_array_t(t, or_i),
                )
                .to_array();
                assert_eq!(
                    got,
                    gather_or_ref(&i32_table[..len], &idx, &or_i),
                    "len {len}"
                );
                let or_f: [f32; 16] = core::array::from_fn(|i| f32_value(i + 3));
                let got = f32x16::<$token>::gather_or(
                    &f32_table[..len],
                    iv,
                    f32x16::<$token>::from_array_t(t, or_f),
                )
                .to_array();
                let want = gather_or_ref(&f32_table[..len], &idx, &or_f);
                assert_eq!(got.map(f32::to_bits), want.map(f32::to_bits), "len {len}");

                // scatter_select
                let vals_u: [u32; 16] = core::array::from_fn(|i| 0x5ca7_0000 + i as u32);
                let mut got = vec![7u32; len];
                let mut want = got.clone();
                u32x16::<$token>::from_array_t(t, vals_u).scatter_select(&mut got, enable, iv);
                scatter_ref(&mut want, enable, &idx, &vals_u);
                assert_eq!(got, want, "len {len} enable {enable:#06x}");
                let vals_i: [i32; 16] = core::array::from_fn(|i| i as i32 * 3 - 20);
                let mut got = vec![-9i32; len];
                let mut want = got.clone();
                i32x16::<$token>::from_array_t(t, vals_i).scatter_select(&mut got, enable, iv);
                scatter_ref(&mut want, enable, &idx, &vals_i);
                assert_eq!(got, want, "len {len} enable {enable:#06x}");
                let vals_f: [f32; 16] = core::array::from_fn(f32_value);
                let mut got = vec![0.5f32; len];
                let mut want = got.clone();
                f32x16::<$token>::from_array_t(t, vals_f).scatter_select(&mut got, enable, iv);
                scatter_ref(&mut want, enable, &idx, &vals_f);
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&got), bits(&want), "len {len} enable {enable:#06x}");
            }
        }
    }};
}

#[test]
fn v4_matches_scalar_indexing() {
    match X64V4Token::summon() {
        Some(t) => check_token!(X64V4Token, t),
        None => eprintln!("skipped: this CPU lacks AVX-512 (see the file header for SDE)"),
    }
}

/// Every index collides: the highest enabled lane must win.
#[test]
fn scatter_duplicates_highest_lane_wins() {
    let Some(t) = X64V4Token::summon() else {
        eprintln!("skipped: this CPU lacks AVX-512 (see the file header for SDE)");
        return;
    };
    let vals = u32x16::from_array_t(t, core::array::from_fn(|i| i as u32 + 100));
    let same = u32x16::splat_t(t, 2);
    let mut dst = [0u32; 4];
    vals.scatter_select(&mut dst, 0xffff, same);
    assert_eq!(dst, [0, 0, 115, 0]);
    let mut dst = [0u32; 4];
    vals.scatter_select(&mut dst, 0x00ff, same);
    assert_eq!(dst, [0, 0, 107, 0]);
}

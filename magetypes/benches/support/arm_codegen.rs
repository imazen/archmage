//! ARM codegen comparisons for the lossless add-green row primitive.
//! Scalar baselines can auto-vectorize: NEON is baseline on aarch64.

use archmage::prelude::*;
use std::hint::black_box;
use zenbench::prelude::*;

#[inline(never)]
fn scalar_row(row: &mut [u8]) {
    for pixel in row.as_chunks_mut::<4>().0 {
        pixel[0] = pixel[0].wrapping_add(pixel[1]);
        pixel[2] = pixel[2].wrapping_add(pixel[1]);
    }
}

// Preserve the codec's old four-pixel unrolling as a separate baseline.
#[inline(never)]
fn scalar_unrolled(row: &mut [u8]) {
    let (chunks, tail) = row.as_chunks_mut::<16>();
    for c in chunks {
        let [g0, g1, g2, g3] = [c[1], c[5], c[9], c[13]];
        c[0] = c[0].wrapping_add(g0);
        c[2] = c[2].wrapping_add(g0);
        c[4] = c[4].wrapping_add(g1);
        c[6] = c[6].wrapping_add(g1);
        c[8] = c[8].wrapping_add(g2);
        c[10] = c[10].wrapping_add(g2);
        c[12] = c[12].wrapping_add(g3);
        c[14] = c[14].wrapping_add(g3);
    }
    scalar_row(tail);
}

#[magetypes(define(u8x16), neon, scalar)]
fn green16(token: Token, row: &mut [u8]) {
    let (chunks, tail) = row.as_chunks_mut::<16>();
    for chunk in chunks {
        let input = u8x16::load(token, chunk);
        let a = input.to_array();
        let mask = u8x16::from_array(
            token,
            [
                a[1], 0, a[1], 0, a[5], 0, a[5], 0, a[9], 0, a[9], 0, a[13], 0, a[13], 0,
            ],
        );
        (input + mask).store(chunk);
    }
    scalar_row(tail);
}

#[magetypes(define(u8x32), neon)]
fn green32(token: Token, row: &mut [u8]) {
    let (chunks, tail) = row.as_chunks_mut::<32>();
    for chunk in chunks {
        let input = u8x32::load(token, chunk);
        let a = input.to_array();
        let mask = u8x32::from_array(
            token,
            core::array::from_fn(|i| if i % 2 == 0 { a[(i & !3) + 1] } else { 0 }),
        );
        (input + mask).store(chunk);
    }
    scalar_row(tail);
}

#[arcane(import_intrinsics)]
fn direct_neon_inner(_token: NeonToken, row: &mut [u8]) {
    let mask = vld1q_u8(&[1, 16, 1, 16, 5, 16, 5, 16, 9, 16, 9, 16, 13, 16, 13, 16]);
    let (chunks, tail) = row.as_chunks_mut::<16>();
    for chunk in chunks {
        let input = vld1q_u8(chunk);
        vst1q_u8(chunk, vaddq_u8(input, vqtbl1q_u8(input, mask)));
    }
    scalar_row(tail);
}

// All variants have one opaque function call per row, allowing assembly
// inspection without giving the scalar or vector variant different callers.
#[inline(never)]
fn magetypes16(row: &mut [u8]) {
    green16_neon(NeonToken::summon().expect("aarch64 NEON"), row);
}

#[inline(never)]
fn magetypes32(row: &mut [u8]) {
    green32_neon(NeonToken::summon().expect("aarch64 NEON"), row);
}

#[inline(never)]
fn magetypes_scalar(row: &mut [u8]) {
    green16_scalar(ScalarToken, row);
}

#[inline(never)]
fn magetypes32_scalar(row: &mut [u8]) {
    green32_scalar(ScalarToken, row);
}

#[inline(never)]
fn direct_neon(row: &mut [u8]) {
    direct_neon_inner(NeonToken::summon().expect("aarch64 NEON"), row);
}

type RowKernel = fn(&mut [u8]);

const KERNELS: &[(&str, RowKernel)] = &[
    ("scalar", scalar_row),
    ("scalar_unrolled", scalar_unrolled),
    ("magetypes16", magetypes16),
    ("magetypes32", magetypes32),
    ("magetypes_scalar", magetypes_scalar),
    ("magetypes32_scalar", magetypes32_scalar),
    ("direct_neon", direct_neon),
];

fn check_kernels() {
    for offset in 0..16 {
        for len in 0..=259 {
            let input: Vec<u8> = (0..offset + len + 16)
                .map(|i| (i as u8).wrapping_mul(73))
                .collect();
            let mut expected = input.clone();
            scalar_row(&mut expected[offset..offset + len]);
            for &(name, kernel) in KERNELS {
                let mut actual = input.clone();
                kernel(&mut actual[offset..offset + len]);
                assert_eq!(actual, expected, "{name}: offset={offset}, len={len}");
            }
        }
    }
}

fn bench(suite: &mut Suite) {
    check_kernels();
    eprintln!("ARM add-green: all variants passed alignment/tail parity checks");
    // Working-set sweep, not an image-quality calibration. Every kernel is
    // branchless in pixel values, and operates on one row supplied by caller.
    for bytes in [16usize, 256, 4096, 65536] {
        suite.compare(&format!("arm_add_green/{bytes}"), |g| {
            g.throughput(Throughput::Bytes(bytes as u64));
            for &(name, kernel) in KERNELS {
                g.bench(name, move |b| {
                    b.with_input(move || {
                        (0..bytes)
                            .map(|i| (i as u8).wrapping_mul(73))
                            .collect::<Vec<_>>()
                    })
                    .run(move |mut row| {
                        black_box(kernel)(black_box(&mut row));
                        row
                    });
                });
            }
        });
    }
}

zenbench::main!(bench);

pub(super) fn run() {
    main();
}

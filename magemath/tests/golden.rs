//! Pins the mirrors' bits across commits and architectures.
//!
//! `parity.rs` shows every tier agrees with the mirrors; this shows the mirrors
//! did not move. Two checked-in files:
//!
//! - `golden/edges.txt`: readable input and output bits per op for
//!   `common::named_edges`, so a change to a special case shows in the diff.
//! - `golden/digests.txt`: an FNV-1a digest of every op's outputs over
//!   `common::corpus`, about 230,000 inputs per op.
//!
//! Regenerate after an intentional kernel change, and review the edges diff:
//! `just magemath-golden-regen`.

mod common;

use common::*;

const EDGES: &str = include_str!("golden/edges.txt");
const DIGESTS: &str = include_str!("golden/digests.txt");

/// Hash the bits, with every NaN treated as one value: payload is unspecified.
fn digest(values: impl Iterator<Item = f32>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in values {
        let bits = if v.is_nan() { 0x7fc0_0000 } else { v.to_bits() };
        for byte in bits.to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Digest key for the corpus itself, so a changed generator reads as that and
/// not as every op changing at once.
const INPUTS: &str = "(inputs)";

#[cfg(not(target_arch = "wasm32"))]
fn digests_text(corpus: &[f32]) -> String {
    let mut text =
        String::from("# op  fnv1a64 of output bits over common::corpus (NaN canonical)\n");
    let inputs = digest(corpus.iter().copied());
    text += &format!("{INPUTS} {inputs:016x}\n");
    for op in OPS {
        let d = digest(corpus.iter().map(|&x| (op.mirror)(x)));
        text += &format!("{} {d:016x}\n", op.name);
    }
    text
}

#[cfg(not(target_arch = "wasm32"))]
fn edges_text() -> String {
    let mut text =
        String::from("# op  input_bits  output_bits (hex; every NaN written 7fc00000)\n");
    for op in OPS {
        for x in named_edges() {
            let y = (op.mirror)(x);
            let bits = if y.is_nan() { 0x7fc0_0000 } else { y.to_bits() };
            text += &format!("{} {:08x} {bits:08x}\n", op.name, x.to_bits());
        }
    }
    text
}

fn golden_digest(name: &str) -> u64 {
    let line = DIGESTS
        .lines()
        .find(|l| l.split_whitespace().next() == Some(name))
        .unwrap_or_else(|| panic!("{name}: not in digests.txt; run just magemath-golden-regen"));
    u64::from_str_radix(line.split_whitespace().nth(1).unwrap(), 16).unwrap()
}

#[test]
fn corpus_is_unchanged() {
    assert_eq!(
        digest(corpus().into_iter()),
        golden_digest(INPUTS),
        "common::corpus changed, so every digest is stale: regenerate"
    );
}

#[test]
fn mirrors_match_digests() {
    let corpus = corpus();
    for op in OPS {
        let got = digest(corpus.iter().map(|&x| (op.mirror)(x)));
        assert_eq!(
            got,
            golden_digest(op.name),
            "{}: mirror bits moved",
            op.name
        );
    }
}

#[test]
fn mirrors_match_edge_table() {
    let mut rows = 0;
    for line in EDGES.lines().filter(|l| !l.starts_with('#')) {
        // Op names hold no spaces, and the last two fields are always hex.
        let mut fields = line.rsplitn(3, ' ');
        let want = u32::from_str_radix(fields.next().unwrap(), 16).unwrap();
        let x = f32::from_bits(u32::from_str_radix(fields.next().unwrap(), 16).unwrap());
        let name = fields.next().unwrap();
        let op = OPS.iter().find(|o| o.name == name).expect("op in table");
        assert_same("edge", name, x, (op.mirror)(x), f32::from_bits(want));
        rows += 1;
    }
    assert_eq!(
        rows,
        OPS.len() * named_edges().len(),
        "edges.txt is stale: regenerate"
    );
}

/// Regenerates both files from the mirrors; `include_str!` picks them up on the
/// next build. Not run by default: this overwrites the expectations.
#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "writes the golden files; run through just magemath-golden-regen"]
fn regenerate() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    std::fs::write(dir.join("digests.txt"), digests_text(&corpus())).unwrap();
    std::fs::write(dir.join("edges.txt"), edges_text()).unwrap();
}

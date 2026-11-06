#![forbid(unsafe_code)]
use math_matrix_study::*;
use std::io::Write;
fn main() {
    tier_inventory();
    let args: Vec<_> = std::env::args().collect();
    let dir = std::path::Path::new(&args[1]);
    std::fs::create_dir_all(dir).unwrap();
    let filter = args.get(2).map(String::as_str).unwrap_or("");
    let op_filter = args.get(3).map(String::as_str).unwrap_or("");
    let band_filter = args.get(4).map(String::as_str).unwrap_or("");
    let mut inventory = std::fs::File::create(dir.join("inventory.tsv")).unwrap();
    for c in cases32() {
        if (filter.starts_with('!') && c.label.contains(&filter[1..]))
            || (!filter.starts_with('!') && !c.label.contains(filter))
            || (!op_filter.is_empty() && c.op != op_filter)
        {
            continue;
        }
        for band in ["ordinary", "edge", "special"] {
            if !band_filter.is_empty() && band != band_filter {
                continue;
            }
            let (x, y) = inputs(c.op, band, 1024);
            let mut out = vec![0.; 1024];
            (c.run)(&x, &y, &mut out);
            let name = format!("{}__{}__{}.bin", c.op, band, c.label.replace('/', "_"));
            let mut f = std::fs::File::create(dir.join(&name)).unwrap();
            for ((a, b), v) in x.iter().zip(&y).zip(out) {
                for z in [*a, *b, v] {
                    f.write_all(&z.to_bits().to_le_bytes()).unwrap()
                }
            }
            writeln!(inventory, "{}\t{}\t{}\t{}\tf32", c.op, band, c.label, name).unwrap();
        }
    }
    for c in cases64() {
        if (filter.starts_with('!') && c.label.contains(&filter[1..]))
            || (!filter.starts_with('!') && !c.label.contains(filter))
            || (!op_filter.is_empty() && c.op != op_filter)
        {
            continue;
        }
        for band in ["ordinary", "edge", "special"] {
            if !band_filter.is_empty() && band != band_filter {
                continue;
            }
            let (x, y) = inputs64(band, 1024);
            let mut out = vec![0.; 1024];
            (c.run)(&x, &y, &mut out);
            let name = format!("{}__{}__{}.bin", c.op, band, c.label.replace('/', "_"));
            let mut f = std::fs::File::create(dir.join(&name)).unwrap();
            for ((a, b), v) in x.iter().zip(&y).zip(out) {
                for z in [*a, *b, v] {
                    f.write_all(&z.to_bits().to_le_bytes()).unwrap()
                }
            }
            writeln!(inventory, "{}\t{}\t{}\t{}\tf64", c.op, band, c.label, name).unwrap();
        }
    }
}

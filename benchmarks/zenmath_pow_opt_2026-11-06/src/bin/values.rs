#![forbid(unsafe_code)]
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};
use zenmath_pow_opt_study::{transfer::*, *};
fn selected(name: &str) -> bool {
    name.starts_with("current/") || name.starts_with("packed-") || name == "std/control"
}
fn main() {
    verify();
    let args: Vec<_> = std::env::args().collect();
    let root = Path::new(&args[1]);
    fs::create_dir(root).unwrap();
    let n: usize = args[2].parse().unwrap();
    assert!(n >= 1024);
    let mut inventory = BufWriter::new(File::create(root.join("inventory.tsv")).unwrap());
    let mut id = 0;
    for d in DOMAINS {
        for &distribution in distributions(d) {
            let (x, y) = distributed_inputs(d, distribution, n);
            for c in domain_cases(d, false)
                .into_iter()
                .filter(|c| selected(&c.name))
            {
                let mut out = vec![0.0; n];
                (c.run)(&x, &y, &mut out);
                let name = format!("cell-{id}.bin");
                id += 1;
                let mut file = BufWriter::new(File::create(root.join(&name)).unwrap());
                for i in 0..n {
                    for value in [x[i], y[i], out[i]] {
                        file.write_all(&value.to_bits().to_le_bytes()).unwrap();
                    }
                }
                writeln!(
                    inventory,
                    "pow\t{}\t{}\t{}\t{}\tna\t{}",
                    d.name(),
                    distribution.name(),
                    c.name,
                    c.policy,
                    name
                )
                .unwrap();
            }
        }
    }
    for d in [Domain::Decode, Domain::Encode] {
        for &distribution in distributions(d) {
            let input = transfer_inputs(d, distribution, n);
            let (mut x, mut y) = (vec![0.0; n], vec![0.0; n]);
            prepare(d, &input, &mut x, &mut y);
            for c in domain_cases(d, false)
                .into_iter()
                .filter(|c| selected(&c.name))
            {
                let mut power = vec![0.0; n];
                (c.run)(&x, &y, &mut power);
                for composition in [Composition::Separate, Composition::Fused] {
                    let mut out = vec![0.0; n];
                    compose(d, composition, &input, &power, &mut out);
                    let name = format!("cell-{id}.bin");
                    id += 1;
                    let mut file = BufWriter::new(File::create(root.join(&name)).unwrap());
                    for i in 0..n {
                        for value in [input[i], x[i], y[i], out[i]] {
                            file.write_all(&value.to_bits().to_le_bytes()).unwrap();
                        }
                    }
                    writeln!(
                        inventory,
                        "transfer\t{}\t{}\t{}\t{}\t{}\t{}",
                        d.name(),
                        distribution.name(),
                        c.name,
                        c.policy,
                        composition.name(),
                        name
                    )
                    .unwrap();
                }
            }
            for slice in [false, true] {
                let mut out = vec![0.0; n];
                if slice {
                    rational_slice(d, &input, &mut out)
                } else {
                    rational_scalar(d, &input, &mut out)
                };
                let name = format!("cell-{id}.bin");
                id += 1;
                let mut file = BufWriter::new(File::create(root.join(&name)).unwrap());
                for i in 0..n {
                    for value in [input[i], x[i], y[i], out[i]] {
                        file.write_all(&value.to_bits().to_le_bytes()).unwrap();
                    }
                }
                writeln!(
                    inventory,
                    "transfer\t{}\t{}\tlinear-srgb-0.6.12/{}-rational\t0\tnatural-platform\t{}",
                    d.name(),
                    distribution.name(),
                    if slice { "slice" } else { "scalar" },
                    name
                )
                .unwrap();
            }
        }
    }
    println!("PASS {id} value cells; {n} records per cell");
}

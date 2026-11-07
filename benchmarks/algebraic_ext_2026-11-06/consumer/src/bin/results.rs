#![forbid(unsafe_code)]
use algebraic_ext_consumer::*;
use std::io::Write;
fn main() {
    tier_inventory();
    println!("exact fixtures {}", fixtures::check_all());
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("output directory"));
    std::fs::create_dir_all(&dir).unwrap();
    let mut inventory = std::fs::File::create(dir.join("inventory.tsv")).unwrap();
    macro_rules! go {
        ($ty:ty,$cases:expr,$inputs:expr) => {
            for c in $cases {
                for band in ["timed", "finite_mixed"] {
                    let (mut x, mut y) = ($inputs)(c.group);
                    if band == "finite_mixed" {
                        for i in 0..1024 {
                            let exponent = (i % 41) as i32 - 20;
                            x[i] = (1.0 + ((i * 317) % 1024) as $ty / 1024.0)
                                * (2.0 as $ty).powi(exponent);
                            if i % 2 != 0 {
                                x[i] = -x[i];
                            }
                            y[i] = (1.0 + ((i * 719) % 1024) as $ty / 1024.0)
                                * (2.0 as $ty).powi(((i * 13) % 41) as i32 - 20);
                        }
                    }
                    let mut o = vec![0.; 1024];
                    (c.run)(&x, &y, &mut o);
                    let file = format!(
                        "{}__{}__{}.bin",
                        c.group.replace('/', "_"),
                        band,
                        c.name.replace('/', "_")
                    );
                    let mut f = std::fs::File::create(dir.join(&file)).unwrap();
                    for i in 0..1024 {
                        for z in [x[i], y[i], o[i]] {
                            f.write_all(&z.to_bits().to_le_bytes()).unwrap();
                        }
                    }
                    writeln!(
                        inventory,
                        "{}\t{}\t{}\t{}\t{}",
                        c.group,
                        band,
                        c.name,
                        stringify!($ty),
                        file
                    )
                    .unwrap();
                }
            }
        };
    }
    go!(f32, cases_f32(), inputs_f32);
    go!(f64, cases_f64(), inputs_f64);
}

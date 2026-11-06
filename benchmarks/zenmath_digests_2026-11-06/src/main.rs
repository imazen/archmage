#![forbid(unsafe_code)]
mod kernels;
use kernels::{Kernel, Op};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
use zenhash::Xxh3;
const FULL: u64 = 1u64 << 32;
const BLOCK: usize = 4096;
const VERSION: &str = "zenmath-digests-v1;XXH3-128;zenhash4c4a993ba4207f2f28529ce0406be8650d270118;seed0;LE-u32;NaN=7fc00000;signed-zero-retained;aggregate-of-chunks-not-raw-stream";

fn canonical(bits: u32) -> u32 {
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        0x7fc0_0000
    } else {
        bits
    }
}
fn same(a: f32, b: f32) -> bool {
    canonical(a.to_bits()) == canonical(b.to_bits())
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Row {
    index: u64,
    start: u64,
    count: u64,
    raw: u128,
    canonical: u128,
}
impl Row {
    fn payload(&self) -> String {
        format!(
            "{}\t{}\t{}\t{:032x}\t{:032x}",
            self.index, self.start, self.count, self.raw, self.canonical
        )
    }
    fn encode(&self) -> String {
        let s = self.payload();
        format!("{s}\t{:032x}\n", zenhash::xxh3_128(s.as_bytes()))
    }
    fn parse(s: &str) -> Result<Self, String> {
        let cells: Vec<_> = s.trim_end_matches('\n').split('\t').collect();
        if cells.len() != 6 {
            return Err("invalid chunk columns".into());
        }
        let payload = cells[..5].join("\t");
        if format!("{:032x}", zenhash::xxh3_128(payload.as_bytes())) != cells[5] {
            return Err("chunk checksum mismatch".into());
        }
        Ok(Self {
            index: cells[0].parse::<u64>().map_err(|e| e.to_string())?,
            start: cells[1].parse::<u64>().map_err(|e| e.to_string())?,
            count: cells[2].parse::<u64>().map_err(|e| e.to_string())?,
            raw: u128::from_str_radix(cells[3], 16).map_err(|e| e.to_string())?,
            canonical: u128::from_str_radix(cells[4], 16).map_err(|e| e.to_string())?,
        })
    }
}
#[derive(Debug)]
struct Config {
    op_name: String,
    op: Op,
    backend: String,
    start: u64,
    count: u64,
    chunk: u64,
    threads: usize,
    out: PathBuf,
    identity: PathBuf,
    resume: bool,
    compare: bool,
}
impl Config {
    fn parse() -> Result<Self, String> {
        let mut opts = BTreeMap::new();
        let mut resume = false;
        let mut compare = false;
        let mut args = std::env::args().skip(1);
        while let Some(k) = args.next() {
            match k.as_str() {
                "--resume" => resume = true,
                "--compare" => compare = true,
                _ => {
                    if !k.starts_with("--") {
                        return Err(format!("unexpected argument {k}"));
                    }
                    let v = args.next().ok_or(format!("missing value {k}"))?;
                    if opts.insert(k.clone(), v).is_some() {
                        return Err(format!("duplicate {k}"));
                    }
                }
            }
        }
        let mut get = |k: &str, default: Option<&str>| {
            opts.remove(k)
                .or_else(|| default.map(str::to_owned))
                .ok_or(format!("missing {k}"))
        };
        let op_name = get("--op", None)?;
        let op = Op::parse(&op_name)?;
        let backend = get("--backend", None)?;
        let start = get("--start", Some("0"))?
            .parse::<u64>()
            .map_err(|e| e.to_string())?;
        let count = get("--count", Some("4294967296"))?
            .parse::<u64>()
            .map_err(|e| e.to_string())?;
        let chunk = get("--chunk", Some("1048576"))?
            .parse::<u64>()
            .map_err(|e| e.to_string())?;
        let threads = get("--threads", Some("1"))?
            .parse::<usize>()
            .map_err(|e| e.to_string())?;
        let out = PathBuf::from(get("--out", None)?);
        let identity = PathBuf::from(get("--identity", None)?);
        if !opts.is_empty() {
            return Err(format!("unknown options {opts:?}"));
        }
        if count == 0
            || start >= FULL
            || count > FULL - start
            || chunk == 0
            || chunk > FULL
            || threads == 0
            || threads > 4
        {
            return Err("invalid domain/chunk/threads (maximum4)".into());
        }
        #[cfg(target_arch = "wasm32")]
        if threads != 1 {
            return Err("WASM requires threads1".into());
        }
        Ok(Self {
            op_name,
            op,
            backend,
            start,
            count,
            chunk,
            threads,
            out,
            identity,
            resume,
            compare,
        })
    }
    fn chunks(&self) -> u64 {
        self.count.div_ceil(self.chunk)
    }
    fn range(&self, index: u64) -> (u64, u64) {
        let offset = index * self.chunk;
        (self.start + offset, self.chunk.min(self.count - offset))
    }
    fn metadata(&self) -> Result<String, String> {
        let id = fs::read_to_string(&self.identity).map_err(|e| format!("identity: {e}"))?;
        for field in [
            "source_sha256=",
            "binary_sha256=",
            "rustc=",
            "target=",
            "rounding=nearest-even;gradual-underflow",
            "command=",
            "source_revision=",
        ] {
            if !id
                .lines()
                .any(|l| l.starts_with(field) && l.len() > field.len())
                && field != "rounding=nearest-even;gradual-underflow"
            {
                return Err(format!("identity lacks {field}"));
            }
        }
        if !id
            .lines()
            .any(|l| l == "rounding=nearest-even;gradual-underflow")
        {
            return Err("unsupported rounding identity".into());
        }
        Ok(format!(
            "{VERSION}\nop={}\nbackend={}\nstart={}\ncount={}\nchunk={}\ncompare={}\n{id}",
            self.op_name, self.backend, self.start, self.count, self.chunk, self.compare
        ))
    }
}
fn check_metadata(path: &Path, metadata: &str, resume: bool) -> Result<(), String> {
    if path.exists() {
        if !resume {
            return Err("existing run requires --resume".into());
        }
        if fs::read_to_string(path).map_err(|e| e.to_string())? != metadata {
            return Err("resume configuration/source/binary mismatch".into());
        }
    } else {
        if resume {
            return Err("resume metadata missing".into());
        }
        fs::write(path, metadata).map_err(|e| e.to_string())?;
    }
    Ok(())
}
fn chunk(
    kernel: &Kernel,
    op: Op,
    index: u64,
    start: u64,
    count: u64,
    compare: bool,
) -> Result<Row, String> {
    let mut raw = Xxh3::new();
    let mut canon = Xxh3::new();
    let mut values = vec![0.0f32; BLOCK];
    let mut raw_bytes = vec![0u8; BLOCK * 4];
    let mut canon_bytes = vec![0u8; BLOCK * 4];
    let mut offset = 0;
    while offset < count {
        let n = (count - offset).min(BLOCK as u64) as usize;
        for (i, x) in values[..n].iter_mut().enumerate() {
            *x = f32::from_bits((start + offset + i as u64) as u32);
        }
        kernel(&mut values[..n]);
        for (i, value) in values[..n].iter().enumerate() {
            let input = (start + offset + i as u64) as u32;
            if compare {
                let expected = op.reference(f32::from_bits(input));
                if !same(*value, expected) {
                    return Err(format!(
                        "MISMATCH chunk={index} input={input:08x} actual={:08x} reference={:08x}",
                        value.to_bits(),
                        expected.to_bits()
                    ));
                }
            }
            raw_bytes[4 * i..4 * i + 4].copy_from_slice(&value.to_bits().to_le_bytes());
            canon_bytes[4 * i..4 * i + 4]
                .copy_from_slice(&canonical(value.to_bits()).to_le_bytes());
        }
        raw.update(&raw_bytes[..4 * n]);
        canon.update(&canon_bytes[..4 * n]);
        offset += n as u64;
    }
    Ok(Row {
        index,
        start,
        count,
        raw: raw.digest128(),
        canonical: canon.digest128(),
    })
}
fn ordered(rows: &[Row], start: u64, count: u64, chunk_size: u64) -> Result<(u128, u128), String> {
    if rows.len() as u64 != count.div_ceil(chunk_size) {
        return Err("partial or excess chunk coverage".into());
    }
    let mut raw = Xxh3::new();
    let mut canonical = Xxh3::new();
    raw.update(b"ordered-LE-u64-index-start-count-LE-u128-chunk-digest-v1");
    canonical.update(b"ordered-LE-u64-index-start-count-LE-u128-chunk-digest-v1");
    let mut covered = 0;
    for (i, row) in rows.iter().enumerate() {
        let n = chunk_size.min(count - covered);
        if row.index != i as u64 || row.start != start + covered || row.count != n {
            return Err("chunk gap/duplicate/order/range mismatch".into());
        }
        for value in [row.index, row.start, row.count] {
            raw.update(&value.to_le_bytes());
            canonical.update(&value.to_le_bytes());
        }
        raw.update(&row.raw.to_le_bytes());
        canonical.update(&row.canonical.to_le_bytes());
        covered += n;
    }
    if covered != count {
        return Err("incorrect total coverage".into());
    }
    Ok((raw.digest128(), canonical.digest128()))
}
fn run(c: Config) -> Result<(), String> {
    drop(kernels::select(&c.backend, c.op)?);
    let metadata = c.metadata()?;
    fs::create_dir_all(&c.out).map_err(|e| e.to_string())?;
    check_metadata(&c.out.join("metadata.txt"), &metadata, c.resume)?;
    for name in ["summary.txt", "FAILURE.txt"] {
        let path = c.out.join(name);
        if path.exists() {
            let suffix = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| e.to_string())?
                .as_nanos();
            fs::rename(&path, c.out.join(format!("{name}.{suffix}.bak")))
                .map_err(|e| e.to_string())?;
        }
    }
    let result = run_admitted(&c);
    if let Err(error) = &result {
        fs::write(c.out.join("FAILURE.txt"), error).map_err(|e| e.to_string())?;
    }
    result
}
fn run_admitted(c: &Config) -> Result<(), String> {
    let mut rows = Vec::new();
    let mut pending = Vec::new();
    for i in 0..c.chunks() {
        let path = c.out.join(format!("chunk-{i:06}.tsv"));
        if path.exists() {
            if !c.resume {
                return Err("chunk present in new run".into());
            }
            let row = Row::parse(&fs::read_to_string(&path).map_err(|e| e.to_string())?)?;
            let (start, count) = c.range(i);
            if row.index != i || row.start != start || row.count != count {
                return Err("resume chunk range mismatch".into());
            }
            rows.push(row);
        } else {
            pending.push(i);
        }
    }
    let rows = Mutex::new(rows);
    let next = AtomicUsize::new(0);
    let errors = Mutex::new(Vec::<String>::new());
    let work = || {
        let kernel = kernels::select(&c.backend, c.op).expect("validated backend");
        loop {
            if !errors.lock().unwrap().is_empty() {
                break;
            }
            let slot = next.fetch_add(1, Ordering::Relaxed);
            if slot >= pending.len() {
                break;
            }
            let i = pending[slot];
            let (start, count) = c.range(i);
            match chunk(&kernel, c.op, i, start, count, c.compare) {
                Ok(row) => {
                    let path = c.out.join(format!("chunk-{i:06}.tsv"));
                    let temp = c.out.join(format!("chunk-{i:06}.pending"));
                    if let Err(e) =
                        fs::write(&temp, row.encode()).and_then(|_| fs::rename(&temp, &path))
                    {
                        errors.lock().unwrap().push(e.to_string());
                        break;
                    }
                    let mut saved = rows.lock().unwrap();
                    saved.push(row);
                    println!(
                        "chunk={i} complete_chunks={}/{} count={count}",
                        saved.len(),
                        c.chunks()
                    );
                }
                Err(e) => {
                    // Reproduce the entire mismatching chunk before failing, retaining exact bits.
                    let rerun = chunk(&kernel, c.op, i, start, count, true)
                        .err()
                        .unwrap_or_else(|| "RERUN unexpectedly passed".into());
                    errors.lock().unwrap().push(format!("{e}\nrerun: {rerun}"));
                    break;
                }
            }
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::scope(|scope| {
        for _ in 0..c.threads {
            scope.spawn(work);
        }
    });
    #[cfg(target_arch = "wasm32")]
    {
        debug_assert_eq!(c.threads, 1);
        work();
    }
    let errors = errors.into_inner().unwrap();
    if !errors.is_empty() {
        let e = errors.join("\n");
        fs::write(c.out.join("FAILURE.txt"), &e).map_err(|e| e.to_string())?;
        return Err(e);
    }
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by_key(|r| r.index);
    let (raw, canon) = ordered(&rows, c.start, c.count, c.chunk)?;
    let status = if c.start == 0 && c.count == FULL {
        "COMPLETE-F32-DOMAIN"
    } else {
        "COMPLETE-BOUNDED-DOMAIN"
    };
    let summary = format!(
        "status={status}\ncount={}\nchunks={}\nraw_aggregate_of_chunks={raw:032x}\ncanonical_aggregate_of_chunks={canon:032x}\n",
        c.count,
        rows.len()
    );
    fs::write(
        c.out.join("chunks.tsv"),
        rows.iter().map(Row::encode).collect::<String>(),
    )
    .map_err(|e| e.to_string())?;
    fs::write(c.out.join("summary.txt"), &summary).map_err(|e| e.to_string())?;
    print!("{summary}");
    Ok(())
}
fn audit(path: &Path) -> Result<(), String> {
    if path.join("FAILURE.txt").exists() {
        return Err("FAILURE marker present".into());
    }
    let metadata = fs::read_to_string(path.join("metadata.txt")).map_err(|e| e.to_string())?;
    if metadata.lines().next() != Some(VERSION) {
        return Err("unknown hash scheme".into());
    }
    let fields: BTreeMap<_, _> = metadata
        .lines()
        .skip(1)
        .filter_map(|l| l.split_once('='))
        .collect();
    let get = |name: &str| {
        fields
            .get(name)
            .ok_or(format!("missing metadata {name}"))?
            .parse::<u64>()
            .map_err(|e| e.to_string())
    };
    let start = get("start")?;
    let count = get("count")?;
    let chunk_size = get("chunk")?;
    if count == 0 || start >= FULL || count > FULL - start || chunk_size == 0 || chunk_size > FULL {
        return Err("invalid audit domain".into());
    }
    let manifest = fs::read_to_string(path.join("chunks.tsv")).map_err(|e| e.to_string())?;
    let rows: Vec<Row> = manifest.lines().map(Row::parse).collect::<Result<_, _>>()?;
    for row in &rows {
        let saved = fs::read_to_string(path.join(format!("chunk-{:06}.tsv", row.index)))
            .map_err(|e| e.to_string())?;
        if Row::parse(&saved)? != *row {
            return Err("manifest differs from chunk file".into());
        }
    }
    let (raw, canonical) = ordered(&rows, start, count, chunk_size)?;
    let summary = fs::read_to_string(path.join("summary.txt")).map_err(|e| e.to_string())?;
    let status = if start == 0 && count == FULL {
        "COMPLETE-F32-DOMAIN"
    } else {
        "COMPLETE-BOUNDED-DOMAIN"
    };
    let expected = format!(
        "status={status}\ncount={count}\nchunks={}\nraw_aggregate_of_chunks={raw:032x}\ncanonical_aggregate_of_chunks={canonical:032x}\n",
        rows.len()
    );
    if summary != expected {
        return Err("summary differs from verified chunks/domain".into());
    }
    println!("AUDITED {} {status} count={count}", path.display());
    Ok(())
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 2 && args[1] == "--build-info" {
        println!(
            "source_sha256={}\nrustc={}\ntarget={}",
            env!("DIGEST_SOURCE_SHA256"),
            env!("DIGEST_RUSTC"),
            env!("DIGEST_TARGET")
        );
        return;
    }
    let result = if args.len() == 3 && args[1] == "--audit" {
        audit(Path::new(&args[2]))
    } else {
        Config::parse().and_then(run)
    };
    if let Err(e) = result {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_contract() {
        assert_eq!(canonical(0xff80_0001), 0x7fc0_0000);
        assert_eq!(canonical(0x7fff_ffff), 0x7fc0_0000);
        assert_eq!(canonical(0x8000_0000), 0x8000_0000);
        assert!(!same(0.0, -0.0));
        assert!(same(f32::NAN, f32::from_bits(0xff80_0001)));
    }
    #[test]
    fn little_endian_stream_and_tail() {
        let k: Kernel = Box::new(|_| {});
        let r = chunk(&k, Op::Floor, 0, 0x7f7f_fffd, 11, false).unwrap();
        let raw: Vec<u8> = (0x7f7f_fffdu32..0x7f80_0008u32)
            .flat_map(u32::to_le_bytes)
            .collect();
        let canon: Vec<u8> = (0x7f7f_fffdu32..0x7f80_0008u32)
            .flat_map(|x| canonical(x).to_le_bytes())
            .collect();
        assert_eq!(r.raw, zenhash::xxh3_128(&raw));
        assert_eq!(r.canonical, zenhash::xxh3_128(&canon));
    }
    #[test]
    fn coverage_guards() {
        let rows = vec![
            Row {
                index: 0,
                start: 7,
                count: 3,
                raw: 1,
                canonical: 2,
            },
            Row {
                index: 1,
                start: 10,
                count: 2,
                raw: 3,
                canonical: 4,
            },
        ];
        assert!(ordered(&rows, 7, 5, 3).is_ok());
        assert!(ordered(&rows[..1], 7, 5, 3).is_err());
        let mut wrong = rows.clone();
        wrong[1].start = 9;
        assert!(ordered(&wrong, 7, 5, 3).is_err());
        wrong = rows.clone();
        wrong[1].index = 0;
        assert!(ordered(&wrong, 7, 5, 3).is_err());
        wrong = rows;
        wrong.reverse();
        assert!(ordered(&wrong, 7, 5, 3).is_err());
    }
    #[test]
    fn corruption_guard() {
        let row = Row {
            index: 3,
            start: 10,
            count: 5,
            raw: 0x1234,
            canonical: 0x5678,
        };
        assert_eq!(Row::parse(&row.encode()).unwrap(), row);
        assert!(Row::parse(&row.encode().replacen("1234", "1235", 1)).is_err());
    }
    #[test]
    fn mismatch_keeps_bits() {
        let k: Kernel = Box::new(|xs| xs.fill(-0.0));
        let e = chunk(&k, Op::Floor, 9, 0, 1, true).unwrap_err();
        assert!(e.contains("chunk=9 input=00000000 actual=80000000 reference=00000000"));
    }
    #[test]
    fn thread_order_invariant() {
        let k = kernels::select("reference", Op::Floor).unwrap();
        let mut rows: Vec<_> = (0..4)
            .map(|i| chunk(&k, Op::Floor, i, i * 19, 19, false).unwrap())
            .collect();
        let expected = ordered(&rows, 0, 76, 19).unwrap();
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..4)
                .rev()
                .map(|i| {
                    s.spawn(move || {
                        let k = kernels::select("reference", Op::Floor).unwrap();
                        chunk(&k, Op::Floor, i, i * 19, 19, false).unwrap()
                    })
                })
                .collect();
            rows = handles.into_iter().map(|h| h.join().unwrap()).collect();
        });
        rows.sort_by_key(|r| r.index);
        assert_eq!(ordered(&rows, 0, 76, 19).unwrap(), expected);
    }
    #[test]
    fn backend_guard() {
        assert!(kernels::select("v4x8", Op::Floor).is_err());
        assert!(kernels::select("unknown", Op::Exp).is_err());
        assert_eq!(
            Op::parse("pow:4019999a").unwrap(),
            Op::Pow(2.4f32.to_bits())
        );
    }
    #[test]
    fn directed_full_bit_corpus() {
        let mut backends = vec!["packed-scalar", "scalar-context"];
        #[cfg(target_arch = "aarch64")]
        backends.extend(["neon", "scalar-neon"]);
        #[cfg(target_arch = "x86_64")]
        {
            backends.extend(["v3", "scalar-v3"]);
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        backends.extend(["v4x", "scalar-v4x", "v4x8"]);
        #[cfg(target_arch = "wasm32")]
        backends.extend(["wasm128", "scalar-wasm128"]);
        let mut bits = vec![
            0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0xff7fffff,
            0x7f800000, 0xff800000, 0x7f800001, 0xff800001, 0x7fc01234,
        ];
        let mut state = 0x31415926u32;
        for _ in 0..4096 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            bits.push(state);
        }
        for op in [
            Op::Exp,
            Op::Ln,
            Op::Log2,
            Op::Pow(2.4f32.to_bits()),
            Op::Pow(0.5f32.to_bits()),
            Op::Pow(2.0f32.to_bits()),
            Op::Pow((-1.0f32).to_bits()),
        ] {
            for backend in &backends {
                let k = kernels::select(backend, op).unwrap();
                let mut values: Vec<f32> = bits.iter().map(|b| f32::from_bits(*b)).collect();
                k(&mut values);
                for (value, bits) in values.iter().zip(&bits) {
                    assert!(
                        same(*value, op.reference(f32::from_bits(*bits))),
                        "{backend}/{op:?} input={bits:08x} actual={:08x} reference={:08x}",
                        value.to_bits(),
                        op.reference(f32::from_bits(*bits)).to_bits()
                    );
                }
            }
        }
    }
    #[test]
    fn corrupt_completed_resume_retires_marker() {
        let dir =
            std::env::temp_dir().join(format!("zenmath-completed-resume-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let identity = dir.join("identity.txt");
        fs::write(&identity,"source_revision=5a9cc685\nsource_sha256=abc\nbinary_sha256=def\nrustc=test\ntarget=test\nrounding=nearest-even;gradual-underflow\ncommand=test\n").unwrap();
        let config = |resume| Config {
            op_name: "floor".into(),
            op: Op::Floor,
            backend: "reference".into(),
            start: 0,
            count: 11,
            chunk: 7,
            threads: 1,
            out: dir.join("run"),
            identity: identity.clone(),
            resume,
            compare: true,
        };
        run(config(false)).unwrap();
        assert!(dir.join("run/summary.txt").exists());
        fs::write(dir.join("run/chunk-000000.tsv"), "corrupt").unwrap();
        assert!(run(config(true)).is_err());
        assert!(!dir.join("run/summary.txt").exists());
        assert!(dir.join("run/FAILURE.txt").exists());
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn resume_identity_guard() {
        let dir = std::env::temp_dir().join(format!("zenmath-digest-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("metadata");
        assert!(check_metadata(&p, "source-A", false).is_ok());
        assert!(check_metadata(&p, "source-A", false).is_err());
        assert!(check_metadata(&p, "source-B", true).is_err());
        assert!(check_metadata(&p, "source-A", true).is_ok());
        fs::remove_file(p).unwrap();
        fs::remove_dir(dir).unwrap();
    }
}

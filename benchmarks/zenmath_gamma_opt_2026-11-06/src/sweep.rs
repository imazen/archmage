//! Full admitted-domain exact comparisons with ordered checksummed chunk records.
use crate::{kernel, piecewise};
use std::{
    error::Error,
    fs::{self, File},
    io::Write,
    path::Path,
};
use zenhash::Xxh3;
const CHUNK: u64 = 1 << 20;
const SCHEME: &str = "gamma-v1;XXH3-128;zenhash4c4a993ba4207f2f28529ce0406be8650d270118;seed0;LE-u32;raw-stream+ordered-chunk-aggregate";

fn meta<const K: usize, const N: usize, const PIECE: bool>(
    width: usize,
    source: &str,
    binary: &str,
) -> String {
    let (start, end) = kernel::domain::<K>();
    format!(
        "kind={K}\nfamily={}\ndegree={N}\nlayout=estrin\nwidth={width}\np_bits={:08x}\nstart={start}\nend={end}\ncount={}\nchunk={CHUNK}\nsource={source}\nbinary={binary}\nhashscheme={SCHEME}\n",
        if PIECE { "piece" } else { "global" },
        kernel::exponent::<K>().to_bits(),
        u64::from(end - start) + 1
    )
}
fn header<const K: usize>(source: &str) -> String {
    let (s, e) = kernel::domain::<K>();
    format!(
        "{SCHEME};kind={K};p={:08x};start={s};end={e};chunk={CHUNK};source={source}",
        kernel::exponent::<K>().to_bits()
    )
}
fn aggregate_update(hash: &mut Xxh3, index: u64, start: u64, count: u64, digest: u128) {
    for v in [index, start, count] {
        hash.update(&v.to_le_bytes());
    }
    hash.update(&digest.to_le_bytes());
}
fn scalar<const K: usize, const N: usize, const PIECE: bool>(x: f32) -> kernel::Evaluation {
    const {
        assert!(
            K < 2
                && if PIECE {
                    N == 5
                } else if K == 0 {
                    N == 13
                } else {
                    N == 15
                }
        );
    }
    if PIECE {
        piecewise::evaluate::<K, 5, 1>(x)
    } else if K == 0 {
        kernel::evaluate::<0, 13, 1>(x)
    } else {
        kernel::evaluate::<1, 15, 1>(x)
    }
}

fn calculate<const K: usize, const N: usize, const PIECE: bool>(
    width: usize,
    out: &Path,
    source: &str,
    binary: &str,
) -> Result<(), Box<dyn Error>> {
    fs::write(
        out.join("metadata.txt"),
        meta::<K, N, PIECE>(width, source, binary),
    )?;
    let mut file = File::create_new(out.join("chunks.tsv"))?;
    writeln!(
        file,
        "index\tstart\tcount\tcandidate\treference\tscalar_certificate_rescues\tchecksum"
    )?;
    let (minimum, maximum) = kernel::domain::<K>();
    let start = u64::from(minimum);
    let total = u64::from(maximum - minimum) + 1;
    let mut raw = Xxh3::new();
    let mut raw_ref = Xxh3::new();
    let mut agg = Xxh3::new();
    agg.update(header::<K>(source).as_bytes());
    let mut agg_ref = Xxh3::new();
    agg_ref.update(header::<K>(source).as_bytes());
    let mut input = [0.0f32; 4096];
    let mut output = [0.0f32; 4096];
    let mut bytes = [0u8; 16384];
    let mut reference_bytes = [0u8; 16384];
    let mut completed = 0u64;
    let mut index = 0u64;
    let mut total_scalar_certificate_rescues = 0u64;
    while completed < total {
        let count = (total - completed).min(CHUNK);
        let chunk_start = start + completed;
        let mut within = 0u64;
        let mut candidate = Xxh3::new();
        let mut reference = Xxh3::new();
        let mut scalar_certificate_rescues = 0u64;
        while within < count {
            let n = (count - within).min(4096) as usize;
            for (i, x) in input[..n].iter_mut().enumerate() {
                *x = f32::from_bits((chunk_start + within + i as u64) as u32);
            }
            match width {
                0 => {
                    for (y, &x) in output[..n].iter_mut().zip(&input[..n]) {
                        *y = scalar::<K, N, PIECE>(x).value;
                    }
                }
                #[cfg(target_arch = "x86_64")]
                4 => crate::vector::stream::<K, N, 1, PIECE, 4>(&input[..n], &mut output[..n]),
                #[cfg(target_arch = "x86_64")]
                8 => crate::vector::stream::<K, N, 1, PIECE, 8>(&input[..n], &mut output[..n]),
                _ => return Err("unsupported width/target".into()),
            }
            for i in 0..n {
                let x = input[i];
                let r = zenmath::scalar::reference::powf(x, kernel::exponent::<K>());
                let got = output[i];
                let certificate = scalar::<K, N, PIECE>(x);
                scalar_certificate_rescues += u64::from(certificate.rescued);
                if got.to_bits() != r.to_bits() || certificate.value.to_bits() != r.to_bits() {
                    return Err(format!("MISMATCH kind={K} N={N} piece={PIECE} width={width} x={:08x} got={:08x} scalar={:08x} CORE={:08x}",x.to_bits(),got.to_bits(),certificate.value.to_bits(),r.to_bits()).into());
                }
                bytes[i * 4..i * 4 + 4].copy_from_slice(&got.to_bits().to_le_bytes());
                reference_bytes[i * 4..i * 4 + 4].copy_from_slice(&r.to_bits().to_le_bytes());
            }
            candidate.update(&bytes[..n * 4]);
            reference.update(&reference_bytes[..n * 4]);
            raw.update(&bytes[..n * 4]);
            raw_ref.update(&reference_bytes[..n * 4]);
            within += n as u64;
        }
        let ch = candidate.digest128();
        let rh = reference.digest128();
        if ch != rh {
            return Err("chunk hash mismatch".into());
        }
        let payload = format!(
            "{index}\t{chunk_start}\t{count}\t{ch:032x}\t{rh:032x}\t{scalar_certificate_rescues}"
        );
        writeln!(
            file,
            "{payload}\t{:032x}",
            zenhash::xxh3_128(payload.as_bytes())
        )?;
        file.flush()?;
        aggregate_update(&mut agg, index, chunk_start, count, ch);
        aggregate_update(&mut agg_ref, index, chunk_start, count, rh);
        completed += count;
        total_scalar_certificate_rescues += scalar_certificate_rescues;
        index += 1;
        fs::write(
            out.join("progress.txt"),
            format!("completed={completed}\nrequired={total}\nchunks={index}\n"),
        )?;
        println!(
            "kind={K} family={} width={width} completed={completed}/{total} chunks={index}",
            if PIECE { "piece" } else { "global" }
        );
    }
    file.sync_all()?;
    let mut summary = File::create_new(out.join("summary.txt"))?;
    writeln!(
        summary,
        "status=COMPLETE-GAMMA-DOMAIN\ncount={total}\nchunks={index}\nscalar_certificate_rescues={total_scalar_certificate_rescues}\nraw_candidate={:032x}\nraw_reference={:032x}\naggregate_candidate={:032x}\naggregate_reference={:032x}",
        raw.digest128(),
        raw_ref.digest128(),
        agg.digest128(),
        agg_ref.digest128()
    )?;
    summary.sync_all()?;
    Ok(())
}
fn audit_cell<const K: usize, const N: usize, const PIECE: bool>(
    width: usize,
    out: &Path,
    source: &str,
    binary: &str,
) -> Result<(), Box<dyn Error>> {
    if out.join("FAILURE.txt").exists() {
        return Err("FAILURE marker".into());
    }
    if fs::read_to_string(out.join("metadata.txt"))? != meta::<K, N, PIECE>(width, source, binary) {
        return Err("metadata differs from expected plan".into());
    }
    let (minimum, maximum) = kernel::domain::<K>();
    let total = u64::from(maximum - minimum) + 1;
    let mut done = 0;
    let mut chunks = 0;
    let mut scalar_certificate_rescues = 0;
    let mut agg = Xxh3::new();
    agg.update(header::<K>(source).as_bytes());
    let rows = fs::read_to_string(out.join("chunks.tsv"))?;
    for (line_no, line) in rows.lines().enumerate() {
        if line_no == 0 {
            if line
                != "index\tstart\tcount\tcandidate\treference\tscalar_certificate_rescues\tchecksum"
            {
                return Err("chunk header".into());
            }
            continue;
        }
        let parts: Vec<_> = line.split('\t').collect();
        if parts.len() != 7 {
            return Err("chunk fields".into());
        }
        let payload = parts[..6].join("\t");
        if zenhash::xxh3_128(payload.as_bytes()) != u128::from_str_radix(parts[6], 16)? {
            return Err("chunk checksum".into());
        }
        let i = parts[0].parse::<u64>()?;
        let start = parts[1].parse::<u64>()?;
        let count = parts[2].parse::<u64>()?;
        let digest = u128::from_str_radix(parts[3], 16)?;
        if done >= total
            || i != chunks
            || start != u64::from(minimum) + done
            || count != (total - done).min(CHUNK)
            || parts[3] != parts[4]
        {
            return Err("chunk order/range/count/digest".into());
        }
        let rescue = parts[5].parse::<u64>()?;
        if rescue > count {
            return Err("rescue count".into());
        }
        scalar_certificate_rescues += rescue;
        aggregate_update(&mut agg, i, start, count, digest);
        done += count;
        chunks += 1;
    }
    if done != total {
        return Err("incomplete coverage".into());
    }
    let summary = fs::read_to_string(out.join("summary.txt"))?;
    let pairs: std::collections::BTreeMap<_, _> = summary
        .lines()
        .map(|s| s.split_once('=').ok_or("summary fields"))
        .collect::<Result<_, _>>()?;
    let expected = [
        "status",
        "count",
        "chunks",
        "scalar_certificate_rescues",
        "raw_candidate",
        "raw_reference",
        "aggregate_candidate",
        "aggregate_reference",
    ];
    if summary.lines().count() != 8
        || pairs.keys().any(|k| !expected.contains(k))
        || pairs.len() != 8
    {
        return Err("summary keys".into());
    }
    for key in [
        "raw_candidate",
        "raw_reference",
        "aggregate_candidate",
        "aggregate_reference",
    ] {
        let _ = u128::from_str_radix(pairs[key], 16)?;
    }
    if pairs.len() != 8
        || pairs.get("status") != Some(&"COMPLETE-GAMMA-DOMAIN")
        || pairs.get("count") != Some(&total.to_string().as_str())
        || pairs.get("chunks") != Some(&chunks.to_string().as_str())
        || pairs.get("scalar_certificate_rescues")
            != Some(&scalar_certificate_rescues.to_string().as_str())
        || pairs.get("raw_candidate") != pairs.get("raw_reference")
        || pairs.get("aggregate_candidate") != pairs.get("aggregate_reference")
        || pairs.get("aggregate_candidate") != Some(&format!("{:032x}", agg.digest128()).as_str())
    {
        return Err("summary consistency".into());
    }
    println!(
        "AUDITED kind={K} N={N} piece={PIECE} width={width} count={done} raw={} aggregate={}",
        pairs["raw_candidate"], pairs["aggregate_candidate"]
    );
    Ok(())
}

pub(crate) fn run(
    audit: bool,
    kind: usize,
    family: &str,
    width: usize,
    out: &Path,
    source: &str,
    binary: &str,
) -> Result<(), Box<dyn Error>> {
    if kind > 1
        || !matches!(family, "global" | "piece")
        || !matches!(width, 0 | 4 | 8)
        || source.len() != 64
        || binary.len() != 64
        || !source
            .bytes()
            .chain(binary.bytes())
            .all(|b| b.is_ascii_hexdigit())
    {
        return Err("invalid sweep configuration".into());
    }
    if !audit {
        fs::create_dir(out)?;
    }
    macro_rules! cell {
        ($k:literal,$n:literal,$piece:literal) => {
            if audit {
                audit_cell::<$k, $n, $piece>(width, out, source, binary)
            } else {
                calculate::<$k, $n, $piece>(width, out, source, binary)
            }
        };
    }
    let result = match (kind, family) {
        (0, "global") => cell!(0, 13, false),
        (1, "global") => cell!(1, 15, false),
        (0, "piece") => cell!(0, 5, true),
        (1, "piece") => cell!(1, 5, true),
        _ => unreachable!(),
    };
    if let Err(ref e) = result
        && !audit
    {
        fs::write(out.join("FAILURE.txt"), format!("{e}\n"))?;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn auditor_rejects_corrupt_missing_mislabeled_and_failed_fixture() {
        // Synthetic parser fixture only; never numerical sweep evidence.
        let scratch = std::env::var_os("TMPDIR").expect("explicit disk-backed TMPDIR");
        let path = std::path::PathBuf::from(scratch)
            .join(format!("gamma-auditor-test-{}", std::process::id()));
        fs::create_dir(&path).unwrap();
        let source = "1".repeat(64);
        let binary = "2".repeat(64);
        fs::write(
            path.join("metadata.txt"),
            meta::<0, 13, false>(4, &source, &binary),
        )
        .unwrap();
        let (start, end) = kernel::domain::<0>();
        let total = u64::from(end - start) + 1;
        let mut rows = String::from(
            "index\tstart\tcount\tcandidate\treference\tscalar_certificate_rescues\tchecksum\n",
        );
        let mut agg = Xxh3::new();
        agg.update(header::<0>(&source).as_bytes());
        let mut done = 0;
        let mut index = 0;
        while done < total {
            let n = (total - done).min(CHUNK);
            let s = u64::from(start) + done;
            let payload = format!("{index}\t{s}\t{n}\t{:032x}\t{:032x}\t0", 123u128, 123u128);
            rows += &format!(
                "{payload}\t{:032x}\n",
                zenhash::xxh3_128(payload.as_bytes())
            );
            aggregate_update(&mut agg, index, s, n, 123);
            done += n;
            index += 1;
        }
        fs::write(path.join("chunks.tsv"), &rows).unwrap();
        fs::write(path.join("summary.txt"),format!("status=COMPLETE-GAMMA-DOMAIN\ncount={total}\nchunks={index}\nscalar_certificate_rescues=0\nraw_candidate=00000000000000000000000000000001\nraw_reference=00000000000000000000000000000001\naggregate_candidate={:032x}\naggregate_reference={:032x}\n",agg.digest128(),agg.digest128())).unwrap();
        assert!(audit_cell::<0, 13, false>(4, &path, &source, &binary).is_ok());
        assert!(audit_cell::<0, 13, false>(8, &path, &source, &binary).is_err());
        assert!(audit_cell::<0, 5, true>(4, &path, &source, &binary).is_err());
        assert!(audit_cell::<1, 15, false>(4, &path, &source, &binary).is_err());
        assert!(audit_cell::<0, 13, false>(4, &path, &"3".repeat(64), &binary).is_err());
        let corrupt = rows.replacen(
            "0000000000000000000000000000007b",
            "0000000000000000000000000000007c",
            1,
        );
        fs::write(path.join("chunks.tsv"), corrupt).unwrap();
        assert!(audit_cell::<0, 13, false>(4, &path, &source, &binary).is_err());
        let missing = rows
            .lines()
            .take(index as usize)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        fs::write(path.join("chunks.tsv"), missing).unwrap();
        assert!(audit_cell::<0, 13, false>(4, &path, &source, &binary).is_err());
        fs::write(path.join("chunks.tsv"), rows).unwrap();
        fs::write(path.join("FAILURE.txt"), b"synthetic failure\xff").unwrap();
        assert!(audit_cell::<0, 13, false>(4, &path, &source, &binary).is_err());
        fs::remove_dir_all(path).unwrap();
    }
}

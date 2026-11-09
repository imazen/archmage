//! Separate std/Rust-libm comparison; PROVIDERS-PLAN.md freezes the protocol.
#![forbid(unsafe_code)]
use archmage::SimdToken;

struct Cell {
    name: String,
    width: usize,
    context: &'static str,
    size: usize,
    chain: bool,
    exponent: u32,
}
const LABELS: [&str; 4] = [
    "std_f32_powf_literal_exponent",
    "Rust_libm_0_2_16_powf_literal_exponent",
    "std_f32_powf_opaque_exponent",
    "Rust_libm_0_2_16_powf_opaque_exponent",
];
fn normalize(x: f32) -> f32 {
    f32::from_bits(0x3f000000 | (x.to_bits() & 0x007fffff))
}
#[archmage::magetypes(v3, neon, scalar)]
#[inline(never)]
fn provider_loop<const N: usize, const LIBM: bool, const OPAQUE: bool, const EXP: u32>(
    token: Token,
    input: &[[f32; N]],
    output: &mut [[f32; N]],
    chain: bool,
) {
    let _ = token;
    let n = f32::from_bits(EXP);
    let n = if OPAQUE { std::hint::black_box(n) } else { n };
    let mut acc = input[0];
    for (base, out) in input.iter().zip(output) {
        let x = if chain { acc } else { *base };
        let y = x.map(|x| if LIBM { libm::powf(x, n) } else { x.powf(n) });
        acc = if chain { y.map(normalize) } else { y };
        *out = acc;
    }
}
fn execute<const N: usize, const EXP: u32>(
    arm: usize,
    context: &str,
    input: &[[f32; N]],
    output: &mut [[f32; N]],
    chain: bool,
) {
    macro_rules! go {
        ($f:ident, $t:expr) => {
            match arm {
                0 => $f::<N, false, false, EXP>($t, input, output, chain),
                1 => $f::<N, true, false, EXP>($t, input, output, chain),
                2 => $f::<N, false, true, EXP>($t, input, output, chain),
                3 => $f::<N, true, true, EXP>($t, input, output, chain),
                _ => panic!("unknown provider arm"),
            }
        };
    }
    match context {
        "scalar" => go!(provider_loop_scalar, archmage::ScalarToken),
        #[cfg(target_arch = "x86_64")]
        "v3" => go!(
            provider_loop_v3,
            archmage::X64V3Token::summon().expect("required V3")
        ),
        #[cfg(target_arch = "aarch64")]
        "neon" => go!(
            provider_loop_neon,
            archmage::NeonToken::summon().expect("required NEON")
        ),
        _ => panic!("unsupported explicit provider context"),
    }
}
fn register<const N: usize, const EXP: u32>(suite: Option<&mut zenbench::Suite>, cell: &Cell) {
    let input: Vec<[f32; N]> = (0..cell.size / N)
        .map(|i| core::array::from_fn(|j| (((i * N + j) % 255) + 1) as f32 / 255.))
        .collect();
    let mut output = vec![[0.; N]; input.len()];
    for arm in 0..LABELS.len() {
        execute::<N, EXP>(arm, cell.context, &input, &mut output, cell.chain);
        assert!(output.iter().flatten().all(|x| x.is_normal()));
    }
    let Some(suite) = suite else { return };
    suite.group(&cell.name, |g| {
        g.config()
            .min_rounds(50)
            .max_rounds(50)
            .auto_rounds(false)
            .warmup_time(std::time::Duration::from_millis(10))
            .max_time(std::time::Duration::from_millis(100))
            .max_wall_time(std::time::Duration::from_secs(120));
        for (arm, label) in LABELS.iter().enumerate() {
            let input = input.clone();
            let context = cell.context;
            let chain = cell.chain;
            g.bench(*label, move |b| {
                let mut output = vec![[0.; N]; input.len()];
                execute::<N, EXP>(arm, context, &input, &mut output, chain);
                b.iter(|| {
                    execute::<N, EXP>(
                        arm,
                        context,
                        std::hint::black_box(&input),
                        std::hint::black_box(&mut output),
                        chain,
                    )
                });
                std::hint::black_box(output);
            });
        }
    });
}
fn cells() -> Vec<Cell> {
    let native = if cfg!(target_arch = "aarch64") {
        "neon"
    } else {
        "v3"
    };
    let mut cells = Vec::new();
    for width in [1, 4, 8, 16] {
        for context in ["scalar", native] {
            if width == 1 && context != "scalar" {
                continue;
            }
            for size in [64, 4096] {
                for chain in [false, true] {
                    for exponent in [0x4019999a, 0x3f000000] {
                        cells.push(Cell {
                            name: format!(
                                "providers/x{width}/{context}/{size}/{}/{exponent:08x}",
                                if chain { "chain" } else { "stream" }
                            ),
                            width,
                            context,
                            size,
                            chain,
                            exponent,
                        });
                    }
                }
            }
        }
    }
    assert_eq!(cells.len(), 56);
    cells
}
fn register_all(mut suite: Option<&mut zenbench::Suite>) {
    for cell in cells() {
        if std::env::var("POWF_FILTER").is_ok_and(|f| f != cell.name) {
            continue;
        }
        macro_rules! width {
            ($n:literal) => {
                match cell.exponent {
                    0x4019999a => register::<$n, 0x4019999a>(suite.as_deref_mut(), &cell),
                    0x3f000000 => register::<$n, 0x3f000000>(suite.as_deref_mut(), &cell),
                    _ => panic!("unknown frozen exponent"),
                }
            };
        }
        match cell.width {
            1 => width!(1),
            4 => width!(4),
            8 => width!(8),
            16 => width!(16),
            _ => panic!("unknown frozen width"),
        }
    }
}
fn main() {
    if std::env::var("POWF_LIST").is_ok() {
        for cell in cells() {
            println!("{}", cell.name);
        }
        return;
    }
    if std::env::var("POWF_VERIFY_ONLY").is_ok() {
        register_all(None);
        println!("all 56 std/Rust-libm cells verified");
        return;
    }
    assert!(!std::env::args().any(|a| a == "--no-busy-gate"));
    assert!(std::env::var("ZENBENCH_NO_BUSY_GATE").map_or(true, |v| v.is_empty() || v == "0"));
    let output = std::env::var("POWF_RESULT").expect("new POWF_RESULT required");
    assert!(!std::path::Path::new(&output).exists());
    zenbench::run_gated(zenbench::GateConfig::strict(), |s| register_all(Some(s)))
        .save(std::path::Path::new(&output))
        .unwrap();
}

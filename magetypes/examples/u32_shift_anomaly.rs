//! Issue #76 investigation harness: why is a streaming u32x4 `load -> shift ->
//! store` kernel slower with a runtime uniform count (`_mm_srl_epi32` with the
//! count in an xmm register, hoisted) than with an immediate count
//! (`_mm_srli_epi32::<3>`) at 4096 elements on Zen 5 — and not at 256?
//!
//! Kernels differ ONLY in the shift op. Raw intrinsics, no magetypes wrappers,
//! so the comparison is pure codegen. All kernels are `#[arcane]` entry points:
//! the inner `__arcane_*` function cannot inline into the featureless wrapper,
//! so each loop is a distinct symbol in the binary for objdump.
//!
//! **Findings: `benchmarks/anomaly-u32-shift-4096-INVESTIGATION.md`.** Short
//! version: the gap is a code-placement-keyed bistable dispatch state of the
//! L1-resident streaming loop, not a property of the shift instruction — the
//! same machine code runs ±10-20% depending on the binary's layout, and the
//! assignment can invert (const slow, uniform fast) with a 208-byte code shift.
//!
//! Kernels:
//!   copy128        no shift — load/store floor (LLVM turns it into memcpy)
//!   const128       `_mm_srli_epi32::<3>`             (immediate; IR `lshr`)
//!   uniform128     `_mm_srl_epi32(v, cnt)`           (xmm count hoisted before loop)
//!   uniform128_licm  same op, but `_mm_cvtsi32_si128` written IN the loop body,
//!                  mirroring the magetypes `#[inline(always)]` method shape.
//!                  Compiles to byte-identical loop code vs uniform128 (LICM
//!                  hoists it) — an identical-code control for harness noise.
//!   copy256/const256/uniform256   256-bit versions
//!   srlv128/srlv256  `_mm*_srlv_epi32(v, splat(count))` — AVX2 per-lane
//!                  variable shift, splatted count hoisted (remediation candidate;
//!                  measured: rejected)
//!   add_const128/add_uniform128   2 loads -> 2 shifts -> add -> 1 store probe
//!                  (structurally matched pair; xor can't be used — LLVM refolds
//!                  `(a>>3)^(b>>3)` into `(a^b)>>3` for the immediate form)
//!   micro_lat_srl/micro_tpt_srl/micro_lat_srlv/micro_tpt_srlv   register-only
//!                  latency chains / 8-way throughput loops (spin/--only modes;
//!                  n is the iteration count)
//!
//! Buffers live in one arena. src starts at a page boundary + `--soff` bytes;
//! dst starts at the next page boundary after src + `--doff` bytes, so the
//! load/store 4K-aliasing distance (`(dst - src) mod 4096 = doff - soff`) is
//! CONTROLLED and printed, not left to the allocator's mood.
//!
//! Sampling is interleaved round-robin (all kernels once per round, many
//! rounds) so slow frequency/interference drift hits every kernel equally.
//!
//! Usage:
//!   u32_shift_anomaly                          # all kernels, default sizes
//!   u32_shift_anomaly --sizes 256,4096         # override sizes
//!   u32_shift_anomaly --doff 0                 # exact 4K-aliased dst
//!   u32_shift_anomaly --spin uniform128 4096 3 # spin one kernel ~3s (perf stat)
//!   u32_shift_anomaly --trace uniform128 4096 60  # 50ms-slice time series
//!   u32_shift_anomaly --pad                    # calls layout_pad (layout shim)
//!       (spin/trace also honor --soff/--doff)
//!
//! Output: RESULT,<kernel>,<n>,<median_ns_per_elem>,<min_ns_per_elem>,<reps>,<rounds>

#[cfg(target_arch = "x86_64")]
mod kernels {
    use archmage::{X64V3Token, arcane};

    /// Layout shim: shifts the addresses of everything after it so a rebuild
    /// redraws the code-layout lottery (see findings doc: the degraded-state
    /// assignment is layout/window-keyed). Called only under `--pad`.
    #[inline(never)]
    pub fn layout_pad(x: u32) -> u32 {
        // ~48 bytes of code that cannot be folded away.
        let mut v = std::hint::black_box(x);
        for _ in 0..std::hint::black_box(3u32) {
            v = v.rotate_left(7) ^ 0x9e37_79b9;
        }
        v
    }

    #[arcane(import_intrinsics)]
    pub fn copy_128(_t: X64V3Token, src: &[u32], dst: &mut [u32]) {
        for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
            let s: &[u32; 4] = s.try_into().unwrap();
            let d: &mut [u32; 4] = d.try_into().unwrap();
            let v = _mm_loadu_si128(s);
            _mm_storeu_si128(d, v);
        }
    }

    #[arcane(import_intrinsics)]
    pub fn shr_const_128(_t: X64V3Token, src: &[u32], dst: &mut [u32]) {
        for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
            let s: &[u32; 4] = s.try_into().unwrap();
            let d: &mut [u32; 4] = d.try_into().unwrap();
            let v = _mm_loadu_si128(s);
            let v = _mm_srli_epi32::<3>(v);
            _mm_storeu_si128(d, v);
        }
    }

    #[arcane(import_intrinsics)]
    pub fn shr_uniform_128(_t: X64V3Token, src: &[u32], dst: &mut [u32], count: u32) {
        let cnt = _mm_cvtsi32_si128(count as i32);
        for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
            let s: &[u32; 4] = s.try_into().unwrap();
            let d: &mut [u32; 4] = d.try_into().unwrap();
            let v = _mm_loadu_si128(s);
            let v = _mm_srl_epi32(v, cnt);
            _mm_storeu_si128(d, v);
        }
    }

    /// Count conversion written inside the loop body — the shape the
    /// `#[inline(always)]` magetypes method produces before LICM.
    /// Compiles to the same loop as `shr_uniform_128`; identical-code control.
    #[arcane(import_intrinsics)]
    pub fn shr_uniform_128_licm(_t: X64V3Token, src: &[u32], dst: &mut [u32], count: u32) {
        for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
            let s: &[u32; 4] = s.try_into().unwrap();
            let d: &mut [u32; 4] = d.try_into().unwrap();
            let v = _mm_loadu_si128(s);
            let v = _mm_srl_epi32(v, _mm_cvtsi32_si128(count as i32));
            _mm_storeu_si128(d, v);
        }
    }

    #[arcane(import_intrinsics)]
    pub fn copy_256(_t: X64V3Token, src: &[u32], dst: &mut [u32]) {
        for (s, d) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
            let s: &[u32; 8] = s.try_into().unwrap();
            let d: &mut [u32; 8] = d.try_into().unwrap();
            let v = _mm256_loadu_si256(s);
            _mm256_storeu_si256(d, v);
        }
    }

    #[arcane(import_intrinsics)]
    pub fn shr_const_256(_t: X64V3Token, src: &[u32], dst: &mut [u32]) {
        for (s, d) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
            let s: &[u32; 8] = s.try_into().unwrap();
            let d: &mut [u32; 8] = d.try_into().unwrap();
            let v = _mm256_loadu_si256(s);
            let v = _mm256_srli_epi32::<3>(v);
            _mm256_storeu_si256(d, v);
        }
    }

    #[arcane(import_intrinsics)]
    pub fn shr_uniform_256(_t: X64V3Token, src: &[u32], dst: &mut [u32], count: u32) {
        let cnt = _mm_cvtsi32_si128(count as i32);
        for (s, d) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
            let s: &[u32; 8] = s.try_into().unwrap();
            let d: &mut [u32; 8] = d.try_into().unwrap();
            let v = _mm256_loadu_si256(s);
            let v = _mm256_srl_epi32(v, cnt);
            _mm256_storeu_si256(d, v);
        }
    }

    /// AVX2 per-lane variable shift with a splatted count — remediation candidate.
    #[arcane(import_intrinsics)]
    pub fn shr_srlv_128(_t: X64V3Token, src: &[u32], dst: &mut [u32], count: u32) {
        let cnt = _mm_set1_epi32(count as i32);
        for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
            let s: &[u32; 4] = s.try_into().unwrap();
            let d: &mut [u32; 4] = d.try_into().unwrap();
            let v = _mm_loadu_si128(s);
            let v = _mm_srlv_epi32(v, cnt);
            _mm_storeu_si128(d, v);
        }
    }

    /// AGU-pressure probe, imm form: per iteration load TWO vectors, shift
    /// each, xor, store ONE vector (`dst[j] = (src[2j] >> 3) ^ (src[2j+1] >> 3)`).
    /// Shift-port bound at ~1 iter/cycle with 3 AGU ops/iter (75% of the
    /// saturating rate of the plain kernels). If the plain-kernel reg/imm gap
    /// is AGU congestion at the margin, this pair should be at parity.
    /// (Two dependent imm shifts can't be used: IR `lshr` folds.)
    #[arcane(import_intrinsics)]
    pub fn shr_add_const_128(_t: X64V3Token, src: &[u32], dst: &mut [u32]) {
        for (s, d) in src.chunks_exact(8).zip(dst.chunks_exact_mut(4)) {
            let a: &[u32; 4] = s[0..4].try_into().unwrap();
            let b: &[u32; 4] = s[4..8].try_into().unwrap();
            let d: &mut [u32; 4] = d.try_into().unwrap();
            let va = _mm_srli_epi32::<3>(_mm_loadu_si128(a));
            let vb = _mm_srli_epi32::<3>(_mm_loadu_si128(b));
            _mm_storeu_si128(d, _mm_add_epi32(va, vb));
        }
    }

    /// Reg-count counterpart of [`shr_add_const_128`].
    #[arcane(import_intrinsics)]
    pub fn shr_add_uniform_128(_t: X64V3Token, src: &[u32], dst: &mut [u32], count: u32) {
        let cnt = _mm_cvtsi32_si128(count as i32);
        for (s, d) in src.chunks_exact(8).zip(dst.chunks_exact_mut(4)) {
            let a: &[u32; 4] = s[0..4].try_into().unwrap();
            let b: &[u32; 4] = s[4..8].try_into().unwrap();
            let d: &mut [u32; 4] = d.try_into().unwrap();
            let va = _mm_srl_epi32(_mm_loadu_si128(a), cnt);
            let vb = _mm_srl_epi32(_mm_loadu_si128(b), cnt);
            _mm_storeu_si128(d, _mm_add_epi32(va, vb));
        }
    }

    /// Latency chain: v = srl(v, cnt) repeated `iters` times, no memory.
    /// `psrl.d` is an opaque LLVM intrinsic, so the chain cannot be folded.
    /// cycles per iteration (from perf) = data->result latency of the reg form.
    #[arcane(import_intrinsics)]
    pub fn micro_lat_srl(_t: X64V3Token, iters: usize, seed: u32, count: u32) -> u32 {
        let cnt = _mm_cvtsi32_si128(count as i32);
        let mut v = _mm_set1_epi32(seed as i32);
        for _ in 0..iters {
            v = _mm_srl_epi32(v, cnt);
        }
        _mm_cvtsi128_si32(v) as u32
    }

    /// Throughput: 8 independent srl chains per iteration.
    /// cycles/iteration / 8 = reciprocal throughput of the reg form.
    #[arcane(import_intrinsics)]
    pub fn micro_tpt_srl(_t: X64V3Token, iters: usize, seed: u32, count: u32) -> u32 {
        let cnt = _mm_cvtsi32_si128(count as i32);
        let mut v0 = _mm_set1_epi32(seed as i32);
        let mut v1 = _mm_set1_epi32(seed.wrapping_add(1) as i32);
        let mut v2 = _mm_set1_epi32(seed.wrapping_add(2) as i32);
        let mut v3 = _mm_set1_epi32(seed.wrapping_add(3) as i32);
        let mut v4 = _mm_set1_epi32(seed.wrapping_add(4) as i32);
        let mut v5 = _mm_set1_epi32(seed.wrapping_add(5) as i32);
        let mut v6 = _mm_set1_epi32(seed.wrapping_add(6) as i32);
        let mut v7 = _mm_set1_epi32(seed.wrapping_add(7) as i32);
        for _ in 0..iters {
            v0 = _mm_srl_epi32(v0, cnt);
            v1 = _mm_srl_epi32(v1, cnt);
            v2 = _mm_srl_epi32(v2, cnt);
            v3 = _mm_srl_epi32(v3, cnt);
            v4 = _mm_srl_epi32(v4, cnt);
            v5 = _mm_srl_epi32(v5, cnt);
            v6 = _mm_srl_epi32(v6, cnt);
            v7 = _mm_srl_epi32(v7, cnt);
        }
        let a = _mm_or_si128(_mm_or_si128(v0, v1), _mm_or_si128(v2, v3));
        let b = _mm_or_si128(_mm_or_si128(v4, v5), _mm_or_si128(v6, v7));
        _mm_cvtsi128_si32(_mm_or_si128(a, b)) as u32
    }

    /// Latency chain for the AVX2 per-lane form (`vpsrlvd xmm`).
    #[arcane(import_intrinsics)]
    pub fn micro_lat_srlv(_t: X64V3Token, iters: usize, seed: u32, count: u32) -> u32 {
        let cnt = _mm_set1_epi32(count as i32);
        let mut v = _mm_set1_epi32(seed as i32);
        for _ in 0..iters {
            v = _mm_srlv_epi32(v, cnt);
        }
        _mm_cvtsi128_si32(v) as u32
    }

    /// Same throughput shape for the AVX2 per-lane form (`vpsrlvd xmm`).
    #[arcane(import_intrinsics)]
    pub fn micro_tpt_srlv(_t: X64V3Token, iters: usize, seed: u32, count: u32) -> u32 {
        let cnt = _mm_set1_epi32(count as i32);
        let mut v0 = _mm_set1_epi32(seed as i32);
        let mut v1 = _mm_set1_epi32(seed.wrapping_add(1) as i32);
        let mut v2 = _mm_set1_epi32(seed.wrapping_add(2) as i32);
        let mut v3 = _mm_set1_epi32(seed.wrapping_add(3) as i32);
        let mut v4 = _mm_set1_epi32(seed.wrapping_add(4) as i32);
        let mut v5 = _mm_set1_epi32(seed.wrapping_add(5) as i32);
        let mut v6 = _mm_set1_epi32(seed.wrapping_add(6) as i32);
        let mut v7 = _mm_set1_epi32(seed.wrapping_add(7) as i32);
        for _ in 0..iters {
            v0 = _mm_srlv_epi32(v0, cnt);
            v1 = _mm_srlv_epi32(v1, cnt);
            v2 = _mm_srlv_epi32(v2, cnt);
            v3 = _mm_srlv_epi32(v3, cnt);
            v4 = _mm_srlv_epi32(v4, cnt);
            v5 = _mm_srlv_epi32(v5, cnt);
            v6 = _mm_srlv_epi32(v6, cnt);
            v7 = _mm_srlv_epi32(v7, cnt);
        }
        let a = _mm_or_si128(_mm_or_si128(v0, v1), _mm_or_si128(v2, v3));
        let b = _mm_or_si128(_mm_or_si128(v4, v5), _mm_or_si128(v6, v7));
        _mm_cvtsi128_si32(_mm_or_si128(a, b)) as u32
    }

    #[arcane(import_intrinsics)]
    pub fn shr_srlv_256(_t: X64V3Token, src: &[u32], dst: &mut [u32], count: u32) {
        let cnt = _mm256_set1_epi32(count as i32);
        for (s, d) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
            let s: &[u32; 8] = s.try_into().unwrap();
            let d: &mut [u32; 8] = d.try_into().unwrap();
            let v = _mm256_loadu_si256(s);
            let v = _mm256_srlv_epi32(v, cnt);
            _mm256_storeu_si256(d, v);
        }
    }
}

/// Arena-backed src/dst with page-controlled placement.
/// src = page boundary + soff bytes; dst = next page boundary after src + doff bytes.
#[cfg(target_arch = "x86_64")]
struct Bufs {
    arena: Vec<u32>,
    src_start: usize, // element index
    dst_start: usize,
    n: usize,
}

#[cfg(target_arch = "x86_64")]
impl Bufs {
    fn new(n: usize, soff_bytes: usize, doff_bytes: usize) -> Self {
        assert_eq!(soff_bytes % 4, 0);
        assert_eq!(doff_bytes % 4, 0);
        let page_elems = 4096 / 4;
        let arena = vec![0u32; 2 * n + 4 * page_elems];
        let addr = arena.as_ptr() as usize;
        let base = (addr.next_multiple_of(4096) - addr) / 4;
        let src_start = base + soff_bytes / 4;
        let src_end_addr = addr + (src_start + n) * 4;
        let dst_page = src_end_addr.next_multiple_of(4096);
        let dst_start = (dst_page - addr) / 4 + doff_bytes / 4;
        assert!(dst_start + n <= arena.len(), "arena too small");
        let mut b = Bufs {
            arena,
            src_start,
            dst_start,
            n,
        };
        for i in 0..n {
            b.arena[b.src_start + i] = (i as u32).wrapping_mul(2_654_435_761);
        }
        b
    }
    fn src(&self) -> &[u32] {
        &self.arena[self.src_start..self.src_start + self.n]
    }
    // Split borrows: return raw parts for src/dst simultaneously.
    fn src_dst(&mut self) -> (&[u32], &mut [u32]) {
        let n = self.n;
        let (src_start, dst_start) = (self.src_start, self.dst_start);
        let (a, b) = self.arena.split_at_mut(dst_start);
        (&a[src_start..src_start + n], &mut b[..n])
    }
    fn report(&self) {
        let addr = self.arena.as_ptr() as usize;
        let s = addr + self.src_start * 4;
        let d = addr + self.dst_start * 4;
        println!(
            "# n={} src={s:#x} dst={d:#x} src%4096={} dst%4096={} (dst-src)%4096={}",
            self.n,
            s % 4096,
            d % 4096,
            (d.wrapping_sub(s)) % 4096
        );
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    use archmage::{SimdToken, X64V3Token};
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    let t = X64V3Token::summon().expect("needs AVX2");

    const KERNELS: &[&str] = &[
        "copy128",
        "const128",
        "uniform128",
        "uniform128_licm",
        "srlv128",
        "copy256",
        "const256",
        "uniform256",
        "srlv256",
    ];

    let run = |name: &str, src: &[u32], dst: &mut [u32], count: u32| match name {
        "copy128" => kernels::copy_128(t, src, dst),
        "const128" => kernels::shr_const_128(t, src, dst),
        "uniform128" => kernels::shr_uniform_128(t, src, dst, count),
        "uniform128_licm" => kernels::shr_uniform_128_licm(t, src, dst, count),
        "srlv128" => kernels::shr_srlv_128(t, src, dst, count),
        "add_const128" => kernels::shr_add_const_128(t, src, dst),
        "add_uniform128" => kernels::shr_add_uniform_128(t, src, dst, count),
        "copy256" => kernels::copy_256(t, src, dst),
        "const256" => kernels::shr_const_256(t, src, dst),
        "uniform256" => kernels::shr_uniform_256(t, src, dst, count),
        "srlv256" => kernels::shr_srlv_256(t, src, dst, count),
        // register-only microbenchmarks: `src.len()` is the iteration count,
        // dst[0] receives the (meaningless) result to keep it live.
        "micro_lat_srl" => {
            dst[0] = kernels::micro_lat_srl(t, src.len(), src[0], count);
        }
        "micro_lat_srlv" => {
            dst[0] = kernels::micro_lat_srlv(t, src.len(), src[0], count);
        }
        "micro_tpt_srl" => {
            dst[0] = kernels::micro_tpt_srl(t, src.len(), src[0], count);
        }
        "micro_tpt_srlv" => {
            dst[0] = kernels::micro_tpt_srlv(t, src.len(), src[0], count);
        }
        _ => panic!("unknown kernel {name}"),
    };

    // ---- correctness gate: every shift kernel must equal `x >> 3` ----
    {
        let mut b = Bufs::new(64, 0, 256);
        let expect: Vec<u32> = b.src().iter().map(|&x| x >> 3).collect();
        for name in KERNELS.iter().filter(|k| !k.starts_with("copy")) {
            let (src, dst) = b.src_dst();
            dst.fill(0);
            run(name, src, dst, 3);
            assert_eq!(dst, &expect[..], "kernel {name} produced wrong output");
        }
    }

    // ---- args ----
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name);
    if flag("--pad").is_some() {
        println!("pad={}", kernels::layout_pad(black_box(1)));
    }
    let soff: usize = flag("--soff")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(0);
    let doff: usize = flag("--doff")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(256);

    if let Some(i) = flag("--spin") {
        // perf-stat mode: spin one kernel for ~N seconds.
        let name = args[i + 1].as_str();
        let n: usize = args[i + 2].parse().unwrap();
        let secs: u64 = args[i + 3].parse().unwrap();
        let mut b = Bufs::new(n, soff, doff);
        b.report();
        let t0 = Instant::now();
        let mut calls: u64 = 0;
        while t0.elapsed() < Duration::from_secs(secs) {
            for _ in 0..64 {
                let (src, dst) = b.src_dst();
                run(
                    black_box(name),
                    black_box(src),
                    black_box(dst),
                    black_box(3),
                );
                calls += 1;
            }
        }
        let ns = t0.elapsed().as_nanos() as f64;
        println!(
            "SPIN,{name},{n},calls={calls},ns_per_elem={:.4}",
            ns / (calls as f64 * n as f64)
        );
        return;
    }

    if let Some(i) = flag("--trace") {
        // time-series mode: run one kernel continuously, print throughput per
        // ~50ms slice, to visualise fast/slow state residency and flips.
        let name = args[i + 1].as_str();
        let n: usize = args[i + 2].parse().unwrap();
        let slices: usize = args[i + 3].parse().unwrap();
        let mut b = Bufs::new(n, soff, doff);
        b.report();
        // calls per slice targeting ~50ms, calibrated on the first slice
        let mut per = 64usize;
        {
            let t0 = Instant::now();
            for _ in 0..per {
                let (src, dst) = b.src_dst();
                run(
                    black_box(name),
                    black_box(src),
                    black_box(dst),
                    black_box(3),
                );
            }
            let ns = t0.elapsed().as_nanos() as u64;
            per = (per as u64 * 50_000_000 / ns.max(1)) as usize + 1;
        }
        let mut out = Vec::with_capacity(slices);
        for _ in 0..slices {
            let t0 = Instant::now();
            for _ in 0..per {
                let (src, dst) = b.src_dst();
                run(
                    black_box(name),
                    black_box(src),
                    black_box(dst),
                    black_box(3),
                );
            }
            out.push(t0.elapsed().as_nanos() as f64 / (per as f64 * n as f64));
        }
        for (i, v) in out.iter().enumerate() {
            println!("TRACE,{name},{n},{i},{v:.4}");
        }
        return;
    }

    let only: Option<String> = flag("--only").map(|i| args[i + 1].clone());
    let sizes: Vec<usize> = flag("--sizes")
        .map(|i| args[i + 1].split(',').map(|s| s.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![16, 256, 4096, 65536, 1_048_576]);
    let rounds: usize = flag("--rounds")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(9);

    let kernels: Vec<&str> = KERNELS
        .iter()
        .copied()
        .filter(|k| only.as_deref().is_none_or(|o| o == *k))
        .collect();

    // ---- interleaved timing ----
    println!("kernel,n,median_ns_per_elem,min_ns_per_elem,reps,rounds");
    for &n in &sizes {
        let mut b = Bufs::new(n, soff, doff);
        b.report();

        // Warm every kernel, then calibrate reps on const128 so every kernel
        // at this size uses the SAME reps (comparable sample durations).
        for k in &kernels {
            let (src, dst) = b.src_dst();
            run(k, src, dst, 3);
        }
        let mut reps: usize = 1;
        loop {
            let t0 = Instant::now();
            for _ in 0..reps {
                let (src, dst) = b.src_dst();
                run(
                    black_box("const128"),
                    black_box(src),
                    black_box(dst),
                    black_box(3),
                );
            }
            if t0.elapsed() >= Duration::from_millis(20) {
                break;
            }
            reps *= 4;
        }

        // Interleaved rounds: every kernel once per round.
        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); kernels.len()];
        for r in 0..rounds {
            for ki in 0..kernels.len() {
                // rotate start kernel each round to decorrelate drift
                let ki = (ki + r) % kernels.len();
                let name = kernels[ki];
                let t0 = Instant::now();
                for _ in 0..reps {
                    let (src, dst) = b.src_dst();
                    run(
                        black_box(name),
                        black_box(src),
                        black_box(dst),
                        black_box(3u32),
                    );
                }
                samples[ki].push(t0.elapsed().as_nanos() as f64 / reps as f64);
            }
        }
        for (ki, name) in kernels.iter().enumerate() {
            let s = &mut samples[ki];
            s.sort_by(f64::total_cmp);
            let median = s[s.len() / 2] / n as f64;
            let min = s[0] / n as f64;
            println!("RESULT,{name},{n},{median:.4},{min:.4},{reps},{rounds}");
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("x86_64 only");
}

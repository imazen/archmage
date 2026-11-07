use zenbench::wasm::{WasmBench,wasmtime};
use std::time::Duration;
pub fn register(suite:&mut zenbench::Suite,path:&str) {
    let mut config=wasmtime::Config::new();
    config.wasm_simd(true).wasm_relaxed_simd(true);
    let mode=std::env::var("STUDY_WASM_MODE").unwrap_or_else(|_|"fusing".into());
    if mode=="nonfusing" {config.relaxed_simd_deterministic(true); }
    if mode=="deterministic" {config.relaxed_simd_deterministic(true);}
    let wasm=WasmBench::from_bytes_with_config(&std::fs::read(path).unwrap(),config).unwrap();
    let find=|name:&str| {let found:Vec<_>=wasm.module().exports().filter(|e|e.name().contains(name)).map(|e|e.name().to_string()).collect();assert_eq!(found.len(),1,"unique export {name}");found[0].clone()};
    let prepare=find("guest_prepare");let run=find("guest_run");
    let ops=["exp2_lowp","exp_lowp","pow_lowp","exp2_midp","exp_midp","pow_midp","sigmoid_midp","silu_midp","log2_lowp","ln_lowp","log10_lowp","log2_midp","ln_midp","log10_midp"];
    for (op,name) in ops.iter().enumerate() {for width in [4,8,16] {for n in [64,4096] {for chain in [false,true] {for (dist,distribution) in ["ordinary","bits","sparse","dense"].iter().enumerate() {
        let id=format!("wasm/{mode}/{name}/x{width}/{n}/{}/{distribution}",if chain {"chain"} else {"stream"});
        if !super::matches_filter(&id) {continue;}
        suite.group(id,|g| {
            g.config().min_rounds(50).max_rounds(50).auto_rounds(false).warmup_time(Duration::from_millis(10)).max_time(Duration::from_millis(100)).max_wall_time(Duration::from_secs(60));
            for (arm,label) in ["baseline","exceptional","fixed_fused"].iter().enumerate() {
                let wasm=wasm.clone();let prepare=prepare.clone();let run=run.clone();
                g.bench(*label,move |b| {
                    let mut instance=wasm.wasi_instance().unwrap();
                    instance.call::<_,()>(&prepare,(n,dist as i32,op as i32)).unwrap();
                    let f=instance.typed_func::<(i32,i32,i32,i32,i32),i64>(&run).unwrap();
                    b.iter(||f.call(&mut instance.store,(op as i32,arm as i32,width,chain as i32,128)).unwrap());
                });
            }
        });
    }}}}}
}

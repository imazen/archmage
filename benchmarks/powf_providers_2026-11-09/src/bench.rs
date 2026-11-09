//! Safe process boundary for native C ABI providers; report whole fixed batches.
#![forbid(unsafe_code)]
use std::io::{BufRead, Write};
use std::os::unix::net::UnixStream;
struct Worker { input:UnixStream, output:std::io::BufReader<UnixStream> }
impl Worker {
    fn new(path:&str)->Self {
        let input=UnixStream::connect(path).unwrap();
        Self {output:std::io::BufReader::new(input.try_clone().unwrap()),input}
    }
    fn request(&mut self, command:&str)->String {
        self.input.write_all(command.as_bytes()).unwrap(); self.input.flush().unwrap();
        let mut line=String::new(); assert!(self.output.read_line(&mut line).unwrap()>0,"provider worker exited"); line
    }
}
fn main() {
    let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(std::env::var("POWF_EXTERNAL_MANIFEST").unwrap()).unwrap()).unwrap();
    let output=std::env::var("POWF_RESULT").unwrap(); assert!(!std::path::Path::new(&output).exists());
    let width=manifest["width"].as_u64().unwrap() as usize;
    let size=manifest["size"].as_u64().unwrap() as usize;
    let chain=manifest["chain"].as_bool().unwrap();
    let exponent=manifest["exponent"].as_str().unwrap();
    assert!([1,4,8,16].contains(&width)&&[64,4096].contains(&size));
    let group=manifest["group"].as_str().unwrap();
    let mut workers:Vec<_>=manifest["providers"].as_array().unwrap().iter().map(|p| {
        let mut worker=Worker::new(p["socket"].as_str().unwrap());
        let command=if p["method"]=="EMPTY" {"EMPTY\n".to_string()} else {format!("B {} {width} {size} {} {exponent} {}\n",p["method"].as_str().unwrap(),u8::from(chain),1_048_576/size)};
        std::hint::black_box(worker.request(&command));
        (p["label"].as_str().unwrap().to_string(),worker,command)
    }).collect();
    assert!(std::env::var("ZENBENCH_LAUNCHER_PIDS").is_ok(),"supervising launcher must register exact workers");
    assert!(!std::env::args().any(|v|v=="--no-busy-gate"));
    assert!(std::env::var("ZENBENCH_NO_BUSY_GATE").map_or(true,|v|v=="0"||v.is_empty()));
    zenbench::run_gated(zenbench::GateConfig::strict(),|suite| {
        suite.group(group,|g| {
            g.config().min_rounds(50).max_rounds(50).auto_rounds(false)
                .warmup_time(std::time::Duration::from_millis(10))
                .max_time(std::time::Duration::from_millis(100))
                .max_wall_time(std::time::Duration::from_secs(180));
            for (label,mut worker,command) in workers.drain(..) {
                g.bench(label,move|b|b.iter(||std::hint::black_box(worker.request(&command))));
            }
        });
    }).save(std::path::Path::new(&output)).unwrap();
}

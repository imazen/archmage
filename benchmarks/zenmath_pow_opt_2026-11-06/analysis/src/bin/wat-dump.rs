#![forbid(unsafe_code)]
use std::io::Write;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3);
    let text = wasmprinter::print_file(&args[1])?;
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args[2])?;
    file.write_all(text.as_bytes())?;
    println!("PASS exact module WAT dump: {} bytes", text.len());
    Ok(())
}

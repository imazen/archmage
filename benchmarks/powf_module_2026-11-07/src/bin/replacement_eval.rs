//! Actual method outputs for identical input pairs, including rejected inputs.
#![forbid(unsafe_code)]
use std::io::{self,BufRead};
fn main(){
    let native=match std::env::args().nth(1).as_deref(){Some("scalar")=>0,Some("native")=>1,_=>panic!("explicit scalar or native tier required")};
    for line in io::stdin().lock().lines(){
        let line=line.unwrap();let row:Vec<_>=line.split_whitespace().collect();assert_eq!(row.len(),3);
        let base=u32::from_str_radix(row[0],16).unwrap();let exponent=u32::from_str_radix(row[1],16).unwrap();let domain=row[2].parse::<u32>().unwrap();
        print!("{base:08x} {exponent:08x} {domain}");
        for choice in 0..8 {let value=archmage_powf_study::replacement_evaluate(f32::from_bits(base),f32::from_bits(exponent),choice,domain,native);print!(" {value:016x}");}
        println!();
    }
}

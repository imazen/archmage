// Candidate lowerings. No attuned!/reattune! implementation is being tested.
use archmage::{SimdToken, X64V2Token, X64V3Token, X64V4Token, arcane, rite};
use std::cell::Cell;
use std::rc::Rc;

struct Payload {
    value: u32,
    drops: Rc<Cell<u32>>,
}
impl Drop for Payload {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}

#[rite(v2)]
fn direct_v2(x: Payload) -> u32 {
    x.value + 2
}
#[rite(v3)]
fn direct_v3(x: Payload) -> u32 {
    x.value + 3
}
#[arcane]
fn entry_v3(_proof: X64V3Token, x: Payload) -> u32 {
    direct_v3(x)
}
#[arcane]
fn entry_v4(_proof: X64V4Token, x: Payload) -> u32 {
    x.value + 4
}

#[arcane]
fn from_v2<P, A>(_proof: X64V2Token, probe: P, argument: A) -> u32
where
    P: FnMut() -> Option<X64V3Token>,
    A: FnMut() -> Payload,
{
    let (mut probe, mut argument) = (probe, argument);
    if let Some(proof) = probe() {
        entry_v3(proof, argument())
    } else {
        direct_v2(argument())
    }
}

#[arcane]
fn from_v3<P, A>(_proof: X64V3Token, probe: P, argument: A) -> u32
where
    P: FnMut() -> Option<X64V4Token>,
    A: FnMut() -> Payload,
{
    let (mut probe, mut argument) = (probe, argument);
    if let Some(proof) = probe() {
        entry_v4(proof, argument())
    } else {
        direct_v3(argument())
    }
}

#[arcane]
fn already_covered<P, A>(_proof: X64V3Token, _probe: P, argument: A) -> u32
where
    P: FnMut() -> Option<X64V3Token>,
    A: FnMut() -> Payload,
{
    let mut argument = argument;
    direct_v3(argument())
}

fn argument<'a>(calls: &'a Cell<u32>, drops: &'a Rc<Cell<u32>>) -> impl FnMut() -> Payload + 'a {
    || {
        calls.set(calls.get() + 1);
        Payload {
            value: 10,
            drops: drops.clone(),
        }
    }
}

#[test]
fn upgrade_success_moves_argument_once() {
    let v3 = X64V3Token::summon().expect("requires V3 host");
    let probes = Cell::new(0);
    let calls = Cell::new(0);
    let drops = Rc::new(Cell::new(0));
    let result = from_v2(
        v3.v2(),
        || {
            probes.set(probes.get() + 1);
            Some(v3)
        },
        argument(&calls, &drops),
    );
    assert_eq!(
        (result, probes.get(), calls.get(), drops.get()),
        (13, 1, 1, 1)
    );
}

#[test]
fn unavailable_upgrade_uses_covered_fallback_once() {
    let v3 = X64V3Token::summon().expect("requires V3 host");
    let probes = Cell::new(0);
    let calls = Cell::new(0);
    let drops = Rc::new(Cell::new(0));
    let result = from_v2(
        v3.v2(),
        || {
            probes.set(probes.get() + 1);
            None
        },
        argument(&calls, &drops),
    );
    assert_eq!(
        (result, probes.get(), calls.get(), drops.get()),
        (12, 1, 1, 1)
    );
}

#[test]
fn covered_selection_does_not_probe() {
    let v3 = X64V3Token::summon().expect("requires V3 host");
    let probes = Cell::new(0);
    let calls = Cell::new(0);
    let drops = Rc::new(Cell::new(0));
    let result = already_covered(
        v3,
        || {
            probes.set(probes.get() + 1);
            Some(v3)
        },
        argument(&calls, &drops),
    );
    assert_eq!(
        (result, probes.get(), calls.get(), drops.get()),
        (13, 0, 1, 1)
    );
}

#[test]
fn v3_to_v4_checks_real_availability() {
    let v3 = X64V3Token::summon().expect("requires V3 host");
    let available = X64V4Token::summon().is_some();
    println!("v4 available on this host: {available}");
    let probes = Cell::new(0);
    let calls = Cell::new(0);
    let drops = Rc::new(Cell::new(0));
    let result = from_v3(
        v3,
        || {
            probes.set(probes.get() + 1);
            X64V4Token::summon()
        },
        argument(&calls, &drops),
    );
    assert_eq!(
        (result, probes.get(), calls.get(), drops.get()),
        (if available { 14 } else { 13 }, 1, 1, 1)
    );
}

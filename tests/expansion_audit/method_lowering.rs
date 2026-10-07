use std::cell::RefCell;
macro_rules! select_method {
    ($receiver:expr, $method:ident($($arg:expr),*), $select:expr) => {
        if $select { $receiver.$method($($arg),*) }
        else { $receiver.$method($($arg),*) }
    };
}
struct Owned(String);
impl Owned {
    fn finish(self, suffix: String) -> String { self.0 + &suffix }
}
fn receiver(log: &RefCell<Vec<u8>>) -> Owned {
    log.borrow_mut().push(1);
    Owned("a".to_owned())
}
fn argument(log: &RefCell<Vec<u8>>) -> String {
    log.borrow_mut().push(2);
    "b".to_owned()
}
fn main() {
    for selected in [true, false] {
        let mut values = vec![7usize];
        select_method!(values, push(values.len()), selected);
        assert_eq!(values, [7, 1]);
        let mut boxed = Box::new(vec![9usize]);
        select_method!(boxed, push(boxed.len()), selected);
        assert_eq!(*boxed, [9, 1]);
        let log = RefCell::new(Vec::new());
        let result = select_method!(receiver(&log), finish(argument(&log)), selected);
        assert_eq!(result, "ab");
        assert_eq!(*log.borrow(), [1, 2]);
    }
}

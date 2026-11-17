# magemath slice entries vs per-chunk calls

Host r7900x (Ryzen 9 7900X, Zen 4), rustc 1.99.1, no `-C target-cpu`, thin LTO.
Source: parent commit 1cb042ba3741 plus this package. One process run, exploratory
(zenbench `inference=exploratory`); not measured on Zen 5, Intel or ARM.

    cargo bench --bench slices

BitExact MIDP presets (pow Gamma n = 2.4, exp2, log2). Mean time per whole slice:

| function | len | array8 per chunk | caller `#[attune]` kernel | slice entry |
|---|---|---|---|---|
| pow  | 64   | 170 ns  | 39 ns   | 37 ns   |
| exp2 | 64   | 179 ns  | 21 ns   | 20 ns   |
| log2 | 64   | 175 ns  | 15 ns   | 14 ns   |
| pow  | 4096 | 10.8 µs | 2.29 µs | 2.32 µs |
| exp2 | 4096 | 11.2 µs | 1.33 µs | 1.36 µs |
| log2 | 4096 | 11.1 µs | 0.92 µs | 0.94 µs |

- `array8`: `try_*_array_with::<_, 8>` per chunk, Automatic execution each call.
- `attune`: the README pattern, vector entries with Caller execution.
- `slice`: `try_*_slice_with`.

The per-chunk array calls take 4.6x to 12x as long as the slice entry. The slice
entry and a hand-written kernel are within 3% of each other.

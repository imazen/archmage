"""Named paired edges, frozen before collecting provider timings."""
from pathlib import Path
import sys
out = Path(sys.argv[1])
edges = [(1, 0, 'Rust_libm/std_literal'), (3, 2, 'Rust_libm/std_opaque'),
         (2, 0, 'std_opaque/literal'), (3, 1, 'Rust_libm_opaque/literal')]
code = (Path(__file__).parent / 'src/bin/rounding_analyze.rs').read_text()
a = code.index('            for (numerator, denominator) in [')
b = code.index('                let ratios:', a)
code = code[code.index('fn main()'):a] + '            for (numerator,denominator,edge) in [\n' + ''.join(
    f'                ({n},{d},"{label}"),\n' for n, d, label in edges) + '            ] {\n' + code[b:]
code = code.replace('"file":path,"group":group.group_name,', '"file":path,"group":group.group_name,"edge":edge,')
(out / 'providers_analyzer.rs').write_text(code)

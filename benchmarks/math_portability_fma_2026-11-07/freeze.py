"""Extract frozen main algorithms as test-only extension traits, mechanically.

No polynomial is transcribed by hand. Source hashes bind the unchanged graph.
Generated modules stay in the task's output directory, outside git.
"""
import hashlib
from pathlib import Path
import re
import subprocess
import sys

BASE = '6bfd83d744e41259ec758054631dcb0928fc4ea1'
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for width in (4, 8, 16):
    path = f'magetypes/src/simd/generic/generated/transcendentals_f32x{width}.rs'
    for label in ('baseline', 'fused'):
        source = subprocess.check_output(['git', 'show', f'{BASE}:{path}']).decode() if label == 'baseline' else Path(path).read_text()
        prefix = label + '_'
        source = re.sub(r'^//!.*\n', '', source, flags=re.M)
        source = source.replace('crate::simd', 'magetypes::simd').replace('from_repr_unchecked', 'from_repr_t')
        source = source.replace('self.1', 'token')
        methods = re.findall(r'pub fn (\w+)\(self(, n: f32)?\) -> Self', source)
        trait = f'{label.title()}{width}'
        declaration = f'pub trait {trait}<T> {{\n' + ''.join(f'fn {prefix}{name}(self, token: T{extra}) -> Self;\n' for name, extra in methods) + '}\n'
        source = re.sub(r'impl<T: (\w+)> (\w+)<T> \{', declaration + rf'impl<T: \1> {trait}<T> for \2<T> {{', source, count=1)
        for name, extra in methods:
            source = source.replace(f'pub fn {name}(self', f'fn {prefix}{name}(self, token: T')
            source = source.replace(f'.{name}()', f'.{prefix}{name}(token)')
            source = source.replace(f'.{name}(n)', f'.{prefix}{name}(token, n)')
        if label == 'fused':
            source = source.replace('.mul_add_relaxed(', '.mul_add_fused(')
        source = re.sub(r'^\s*///.*\n', '', source, flags=re.M)
        (out / f'{label}{width}.rs').write_text('#[allow(dead_code)]\nmod body {\n' + source + '\n}\npub use body::' + trait + ';\n')
        print(label, width, hashlib.sha256(source.encode()).hexdigest())

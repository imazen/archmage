#!/usr/bin/env python3
"""Check the published flat math implementation and the configured API boundary."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(__file__).with_name('published_math_0_9_31.json')


def methods(source):
    result = {}
    for match in re.finditer(r'pub fn (\w+)\s*\(', source):
        start = source.index('{', match.end())
        end, depth = start + 1, 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        body = re.sub(r'//[^\n]*|/\*.*?\*/', '', source[start:end], flags=re.S)
        tokens = re.sub(r'\s+', '', body)
        name = match[1]
        assert name not in result, f'duplicate method {name}'
        result[name] = hashlib.sha256(tokens.encode()).hexdigest()
    return result


def main():
    expected = json.loads(REFERENCE.read_text())
    for vector, reference in expected['vectors'].items():
        path = ROOT / 'magetypes/src/simd/generic/generated' / f'transcendentals_{vector}.rs'
        actual = methods(path.read_text())
        assert actual == reference['methods'], f'published flat math changed: {vector}'
    lib = (ROOT / 'magetypes/src/lib.rs').read_text()
    for family in ['powf', 'expf', 'logf', 'cbrtf']:
        assert not re.search(r'\bmod\s+' + family + r'\s*;', lib), family
        assert not (ROOT / 'magetypes/src' / family).exists(), family
        assert re.search(r'pub mod\s+' + family + r'\s*;', (ROOT / 'magemath/src/lib.rs').read_text()), family
    for path in (ROOT / 'magetypes/src/simd').rglob('*.rs'):
        assert not re.search(r'pub fn try_powf', path.read_text()), path
    print('Published 0.9.31 flat math bodies retained; configured math exists only in Magemath.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Compare frozen complete power loops with the existing exact asm normalizer."""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('before', type=Path)
p.add_argument('after', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--before-imports', type=Path, required=True)
p.add_argument('--after-imports', type=Path, required=True)
a = p.parse_args()
spec = importlib.util.spec_from_file_location(
    'codegen', Path(__file__).resolve().parents[2] / 'xtask/codegen.py')
codegen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codegen)

def load(path, imports):
    source = path.read_text()
    imported = imports.read_text()
    objects = {}
    for match in re.finditer(
            r'^([^:\s]+):[^\n]*\n(.*?)^\s*\.size\s+\1,\s*([^\n]+)$',
            imported, re.M | re.S):
        name, body, size = match.groups()
        if not re.search(r'^\s*\.type\s+' + re.escape(name) + r',[@%]object$', imported, re.M):
            continue
        alignment = re.search(r'(\.p2align[^\n]*)\n\s*$', imported[:match.start()])
        contents = (alignment[1] + '\n' if alignment else '') + body.strip() + '\nSIZE ' + size
        if name.startswith('anon.'):
            stable = '.Lanon_import_' + hashlib.sha256(contents.encode()).hexdigest()
            source = source.replace(name, stable)
            source += '\n' + (alignment[1] + '\n' if alignment else '') + stable + ':\n' + body
        else:
            stable = re.sub(r'Cs[0-9a-zA-Z]+_', 'CsHASH_', re.sub(r'\.llvm\.\d+', '.llvm.MODULE', name))
        assert stable not in objects
        objects[stable] = contents
    assert objects, 'imported data was not inspected'
    # ThinLTO's numeric module disambiguators are identities, not instructions.
    # Named object initializers and anonymous object contents are checked above.
    source = re.sub(r'\.llvm\.\d+', '.llvm.MODULE', source)
    # The benchmark includes unwinding tables between functions. Extract only
    # declared function bodies so GCC_except_table labels cannot consume the
    # next function in the consumer parser's label-to-size matcher. Retain
    # every instruction and constant; the existing normalizer does the compare.
    names = re.findall(r'^\s*\.type\s+(\S+),[@%]function', source, re.M)
    starts = {m[1]: m.start() for m in re.finditer(
        r'^([A-Za-z_][^:\s]*):[^\n]*\n', source, re.M)}
    ends = {m[1]: m.end() for m in re.finditer(
        r'^\s*\.size\s+(\S+),[^\n]*$', source, re.M)}
    aliases = dict(re.findall(r'^\s*\.set\s+(\S+),\s*(\S+)\s*$', source, re.M))
    # LLVM omits redundant .type directives for merged function aliases.
    # Resolve their type from the explicitly declared destination, retaining
    # every policy name rather than counting only distinct machine bodies.
    function_names = set(names)
    while added := {name for name, target in aliases.items()
                    if target in function_names and name not in function_names}:
        function_names.update(added)
        names.extend(sorted(added))
    chunks = []
    for name in names:
        chunks.append(f'\n.type {name},@function\n')
        if name in starts:
            assert name in ends and ends[name] > starts[name]
            chunks.append(source[starts[name]:ends[name]] + '\n')
        else:
            chunks.append(f'.set {name}, {aliases[name]}\n')
    for match in re.finditer(
            r'^(\.L(?:CPI|anon)[^:\s]+):[^\n]*\n'
            r'((?:\s*\.(?:byte|short|long|quad|word|xword|zero|ascii|asciz|int8|int16|int32|int64)\s+[^\n]*\n)+)',
            source, re.M):
        alignment = re.search(r'(\.p2align[^\n]*)\n\s*$', source[:match.start()])
        chunks.append(('\n' + alignment[1] + '\n' if alignment else '\n') + match[0])
    functions = codegen.parse(''.join(chunks))
    loops = {}
    for name in functions:
        match = re.fullmatch(
            r'_RINvCsHASH_\d+midlow_bench\d+_*(?:arcane_)?loop(\d+)'
            r'(?:_(scalar|v3|v4x))?Km([0-9a-f]+)_Kj([0-9a-f]+)_E.*', name)
        if match:
            width, context, flags, arm = match.groups()
            key = f'x{width}/{context or "scalar-api"}/flags{int(flags,16)}/arm{int(arm,16)}'
            assert key not in loops, key
            loops[key] = codegen.expand(name, functions)
    assert len(loops) == 373, ('unexpected emitted loop inventory', len(loops))
    return loops, objects

before, before_objects = load(a.before, a.before_imports)
after, after_objects = load(a.after, a.after_imports)
assert before_objects == after_objects, 'imported constant or state initializer changed'
assert before.keys() == after.keys()
rows = [{'loop': name, 'equal': before[name] == after[name],
         'before': before[name], 'after': after[name]} for name in sorted(before)]
a.output.write_text(json.dumps({'rows': rows, 'normalization': 'existing xtask/codegen.py'}, indent=2) + '\n')
print(json.dumps({'loops': len(rows), 'equal': sum(r['equal'] for r in rows),
                  'changed': [r['loop'] for r in rows if not r['equal']]}))
assert all(r['equal'] for r in rows), 'release assembly differs; inspect retained report'

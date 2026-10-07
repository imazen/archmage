#!/usr/bin/env python3
"""Harvest the macro signature shapes real crates use, as compile tests.

Usage: python3 xtask/harvest_shapes.py ROOT [ROOT...]

Walks every `.rs` file under the given roots (snapshots of consumer crates),
collects each `#[arcane]`, `#[rite]`, `#[autoversion]` and `#[magetypes]`
attribute together with the signature it annotates, normalizes the signature
to its shape (tokens, receivers, generics, bounds, patterns and lifetimes are
kept; every other type becomes a placeholder), de-duplicates, and writes one
file per distinct shape to `tests/harvest/`. `tests/harvest_shapes.rs` compiles
them all, so a macro change that stops accepting a shape some crate uses fails
here before it reaches that crate.

The body of every harvested function is a stub: the test is that the macro
accepts and expands the signature, which is where the bugs the shape suite was
built for live.
"""
import hashlib
import re
import sys
from pathlib import Path

ATTR = re.compile(
    r'#\[\s*(?:archmage::)?(arcane|rite|autoversion|magetypes|simd_fn)'
    r'(?:\((?P<args>[^\]]*)\))?\s*\]'
)
KEEP_TYPES = {
    # tokens
    'X64V1Token', 'Sse2Token', 'X64V2Token', 'X64CryptoToken', 'X64V3Token',
    'X64V3CryptoToken', 'X64V3GfniCryptoToken', 'X64V4Token', 'Avx512Token',
    'X64V4xToken', 'Avx512Fp16Token', 'Server64', 'Desktop64', 'NeonToken',
    'Arm64', 'Arm64V2Token', 'Arm64V3Token', 'NeonAesToken', 'NeonSha3Token',
    'NeonCrcToken', 'Wasm128Token', 'Wasm128RelaxedToken', 'ScalarToken',
    'SimdToken', 'Token',
    # primitives
    'f32', 'f64', 'i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64',
    'usize', 'isize', 'bool', 'Self', 'Option', 'Result', 'Vec', 'Box',
    'PhantomData',
}
TOKEN_TYPES = {t for t in KEEP_TYPES if t.endswith('Token') or t in ('Arm64', 'Server64', 'Desktop64', 'Token')}
VECTOR = re.compile(r'^[fiu](8|16|32|64)x(1|2|4|8|16|32|64)$')
TRAIT_BOUND = re.compile(r'\b(Has[A-Z][A-Za-z0-9]*|SimdToken|IntoConcreteToken|Copy|Clone|Send|Sync|Sized|Default|Fn|FnMut|FnOnce|[A-Z][A-Za-z0-9]*Backend|[A-Z][A-Za-z0-9]*Convert)\b')


def signature_after(text, pos):
    """The `fn ...` signature starting at or after `pos`, up to its body."""
    m = re.compile(r'\b(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:unsafe\s+)?fn\s+').search(text, pos)
    if not m or m.start() - pos > 400:
        return None, None
    # the preceding visibility may be before `m.start()`; include it
    start = m.start()
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c in '([<{':
            if c == '{' and depth == 0:
                return text[start:i].strip(), i
            if c != '<' or text[i - 1] != '-':
                depth += 1
        elif c in ')]>}':
            if c == '>' and text[i - 1] == '-':
                pass
            else:
                depth -= 1
        elif c == ';' and depth == 0:
            return None, None  # a declaration, not a definition
        i += 1
    return None, None


def normalize_type(ty):
    ty = ty.strip()
    # references, slices, arrays, tuples, pointers: recurse structurally
    m = re.match(r"^&(?:'[a-z_]+\s+)?(mut\s+)?(.*)$", ty, re.S)
    if m:
        lt = re.match(r"^&('[a-z_]+)", ty)
        inner = normalize_type(m.group(2))
        return '&' + (lt.group(1) + ' ' if lt else '') + ('mut ' if m.group(1) else '') + inner
    m = re.match(r'^\*(const|mut)\s+(.*)$', ty, re.S)
    if m:
        return f'*{m.group(1)} {normalize_type(m.group(2))}'
    m = re.match(r'^\[(.*)\]$', ty, re.S)
    if m:
        inner = m.group(1)
        if ';' in inner:
            elem, n = inner.rsplit(';', 1)
            n = n.strip()
            if not re.match(r'^\d+$', n):
                n = 'N' if re.match(r'^[A-Z_][A-Z0-9_]*$', n) else '4'
            return f'[{normalize_type(elem)}; {n}]'
        return f'[{normalize_type(inner)}]'
    m = re.match(r'^\((.*)\)$', ty, re.S)
    if m:
        parts = split_top(m.group(1), ',')
        return '(' + ', '.join(normalize_type(p) for p in parts if p.strip()) + ')'
    m = re.match(r'^(impl|dyn)\s+(.*)$', ty, re.S)
    if m:
        bounds = [b.strip() for b in split_top(m.group(2), '+')]
        kept = []
        for b in bounds:
            head = b.split('<')[0].split('(')[0].split('::')[-1]
            if not TRAIT_BOUND.match(head):
                continue
            b = re.sub(r'^.*::', '', b)
            fm = re.match(r'^(Fn|FnMut|FnOnce)\((.*)\)(\s*->\s*(.*))?$', b, re.S)
            if fm:
                args = ', '.join(normalize_type(a) for a in split_top(fm.group(2), ',') if a.strip())
                ret = f' -> {normalize_type(fm.group(4))}' if fm.group(4) else ''
                b = f'{fm.group(1)}({args}){ret}'
            kept.append(b)
        return f'{m.group(1)} ' + (' + '.join(kept) if kept else 'Copy')
    # path type with optional generics
    m = re.match(r'^([A-Za-z_][A-Za-z0-9_:]*)\s*(<(.*)>)?$', ty, re.S)
    if m:
        name = m.group(1).split('::')[-1]
        args = m.group(3)
        if name in KEEP_TYPES or VECTOR.match(name):
            if args is not None:
                inner = ', '.join(normalize_type(a) for a in split_top(args, ','))
                return f'{name}<{inner}>'
            return name
        if len(name) == 1 and name.isupper():
            return name  # a generic parameter
        if args is not None:
            return 'P'
        return 'P'
    return 'P'


def split_top(s, sep):
    out, depth, cur = [], 0, ''
    for c in s:
        if c in '<([':
            depth += 1
        elif c in '>)]':
            depth -= 1
        if c == sep and depth == 0:
            out.append(cur)
            cur = ''
        else:
            cur += c
    out.append(cur)
    return out


def normalize_signature(sig):
    sig = re.sub(r'//[^\n]*', ' ', sig)
    sig = re.sub(r'/\*.*?\*/', ' ', sig, flags=re.S)
    sig = re.sub(r'#\[[^\]]*\]', ' ', sig)
    sig = re.sub(r'\s+', ' ', sig)
    sig = re.sub(r'^pub(\([^)]*\))?\s+', 'pub ', sig)
    m = re.match(r'^(pub )?(const )?(unsafe )?fn ([A-Za-z_0-9]+)\s*(<[^(]*>)?\s*\((.*)\)\s*(->\s*(.*?))?\s*(where .*)?$', sig, re.S)
    if not m:
        return None
    vis, konst, unsafe, _name, generics, params, _, ret, where = m.groups()
    params = [p.strip() for p in split_top(params, ',') if p.strip()]
    out_params = []
    for i, p in enumerate(params):
        if re.match(r'^(&(\'[a-z_]+)?\s*(mut\s+)?)?(mut\s+)?self$', p) or p.startswith('self:'):
            if p.startswith('self:'):
                out_params.append('self: ' + normalize_type(p.split(':', 1)[1]))
            else:
                out_params.append(p)
            continue
        if ':' not in p:
            continue
        pat, ty = p.split(':', 1)
        pat = pat.strip()
        if pat == '_':
            pass
        elif pat.startswith('(') or pat.startswith('['):
            pat = re.sub(r'[A-Za-z_][A-Za-z0-9_]*', lambda mm: 'a', pat)
        else:
            pat = re.sub(r'^(mut\s+)?[A-Za-z_][A-Za-z0-9_]*$', lambda mm: (mm.group(1) or '') + f'p{i}', pat)
        out_params.append(f'{pat}: {normalize_type(ty)}')
    gen = ''
    if generics:
        parts = [g.strip() for g in split_top(generics[1:-1], ',') if g.strip()]
        norm = []
        for g in parts:
            if g.startswith("'"):
                norm.append(g.split(':')[0].strip())
            elif g.startswith('const '):
                mm = re.match(r'const\s+([A-Za-z_0-9]+)\s*:\s*(.*)', g)
                cty = normalize_type(mm.group(2))
                if cty not in ('usize', 'isize', 'u8', 'u16', 'u32', 'u64', 'i8', 'i16', 'i32', 'i64', 'bool', 'char'):
                    cty = 'usize'  # a type alias of an integer, by the const-generic rules
                norm.append(f'const {mm.group(1)}: {cty}')
            else:
                name, _, bounds = g.partition(':')
                b = [x.strip() for x in split_top(bounds, '+') if x.strip()]
                b = [re.sub(r'^.*::', '', x) for x in b if TRAIT_BOUND.match(x.split('<')[0].split('::')[-1])]
                norm.append(name.strip() + (': ' + ' + '.join(b) if b else ''))
        gen = '<' + ', '.join(norm) + '>'
    ret_s = ''
    if ret:
        ret_s = ' -> ' + normalize_type(ret)
    where_s = ''
    if where:
        preds = [w.strip() for w in split_top(where[len('where '):], ',') if w.strip()]
        kept = []
        for w in preds:
            name, _, bounds = w.partition(':')
            b = [x.strip() for x in split_top(bounds, '+') if x.strip()]
            b = [re.sub(r'^.*::', '', x) for x in b if TRAIT_BOUND.match(x.split('<')[0].split('::')[-1])]
            if b and re.match(r'^[A-Z]$', name.strip()):
                kept.append(name.strip() + ': ' + ' + '.join(b))
        if kept:
            where_s = ' where ' + ', '.join(kept)
    return f"{vis or ''}{konst or ''}{unsafe or ''}fn SHAPE{gen}({', '.join(out_params)}){ret_s}{where_s}"


def main(roots):
    shapes = {}
    for root in roots:
        for path in Path(root).rglob('*.rs'):
            if '/target/' in str(path):
                continue
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            # Comments and doc comments mention the macros by name; only
            # real attributes count.
            text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
            text = re.sub(r'//[^\n]*', '', text)
            # Group the macro attributes by the function they annotate, so a
            # stack such as `#[magetypes(...)] #[rite]` is one shape.
            by_fn = {}
            for m in ATTR.finditer(text):
                sig, fn_pos = signature_after(text, m.end())
                if not sig:
                    continue
                macro = m.group(1)
                args = (m.group('args') or '').strip()
                args = re.sub(r'_self\s*=\s*[A-Za-z0-9_:<>]+', '_self = P', args)
                args = re.sub(r'define\([^)]*\)', 'define(f32x8)', args)
                args = re.sub(r'\s+', ' ', args)
                entry = by_fn.setdefault(fn_pos, (sig, []))
                entry[1].append(f'#[{macro}({args})]' if args else f'#[{macro}]')
            for fn_pos, (sig, attrs) in by_fn.items():
                norm = normalize_signature(sig)
                if not norm:
                    continue
                stack = ' '.join(attrs)
                # A bare #[rite] with neither tier nor token is rejected by
                # design (it has nothing to take features from); the one
                # occurrence found is under a cfg that is off by default.
                if stack == '#[rite]' and not re.search(r': (' + '|'.join(sorted(TOKEN_TYPES)) + r')\b', norm):
                    continue
                shapes.setdefault((stack, norm), []).append(f"{path.relative_to(root)}")
    return shapes


def coarse_key(norm):
    """What the macros distinguish: token kind and position, receiver,
    wildcard/tuple patterns, generics by kind, `Self`, `impl` bounds, return
    kind, visibility and safety. Runs of ordinary parameters collapse."""
    m = re.match(r'^(pub )?(const )?(unsafe )?fn SHAPE(<[^(]*>)?\((.*)\)( -> (.*?))?( where .*)?$', norm)
    vis, k, u, gen, params, _, ret, where = m.groups()
    cats = []
    for p in split_top(params, ','):
        p = p.strip()
        if not p:
            continue
        head = p.split(':')[0].strip()
        if head.endswith('self'):
            cats.append('self:' + re.sub(r'\s+', '', p if p.startswith('self:') else head))
            continue
        pat, ty = p.split(':', 1)
        pat, ty = pat.strip(), ty.strip()
        base = re.sub(r"^(&('[a-z_]+)?\s*(mut\s+)?)", '', ty)
        if base in TOKEN_TYPES:
            cats.append(('wild:' if pat == '_' else '') + 'tok:' + base + ('&' if ty.startswith('&') else ''))
        elif base.startswith('impl '):
            cats.append('impl:' + base[5:])
        elif pat == '_':
            cats.append('wild')
        elif pat.startswith('(') or pat.startswith('['):
            cats.append('pat')
        elif 'Self' in ty:
            cats.append('Self')
        elif len(base) == 1 and base.isupper() and base != 'P':
            cats.append('gen:' + base)
        else:
            cats.append('p')
    out = []
    for c in cats:
        if c == 'p' and out and out[-1] == 'p':
            continue
        out.append(c)
    g = ''
    if gen:
        kinds = []
        for x in split_top(gen[1:-1], ','):
            x = x.strip()
            if not x:
                continue
            if x.startswith("'"):
                kinds.append('lt')
            elif x.startswith('const'):
                kinds.append('const')
            else:
                kinds.append('T:' + x.split(':', 1)[1].strip() if ':' in x else 'T')
        g = '<' + ','.join(kinds) + '>'
    r = ''
    if ret:
        r = ' -> ' + ('Self' if 'Self' in ret else ('impl' if ret.startswith('impl') else ('tok:' + ret if ret in TOKEN_TYPES else 'P')))
    w = where or ''
    return f"{vis or ''}{k or ''}{u or ''}fn{g}({', '.join(out)}){r}{w}"


HEADER = '''//! Every macro signature shape the local consumer crates use, one module per
//! shape, with placeholder types and stub bodies: the test is that the macros
//! still accept and expand each shape. Generated by `xtask/harvest_shapes.py`
//! from snapshots of {crates} crates ({date}); regenerate rather than edit.
//! Each module names the files the shape was harvested from.
#![allow(
    dead_code,
    unused_variables,
    unused_imports,
    unexpected_cfgs,
    clippy::all,
    deprecated
)]

use archmage::prelude::*;
use magetypes::simd::backends::*;
use magetypes::simd::*;

/// Placeholder for every consumer-defined type.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct P;

'''


def emit(shapes, date, crates):
    groups = {}
    for (stack, norm), where in shapes.items():
        groups.setdefault((stack, coarse_key(norm)), []).append((norm, where))
    out = [HEADER.format(crates=crates, date=date)]
    for i, ((attr, key), members) in enumerate(sorted(groups.items())):
        # The shortest signature in the group is its representative.
        norm = min((n for n, _ in members), key=len)
        files = sorted({w for _, ws in members for w in ws})
        sig = norm.replace('fn SHAPE', 'fn shape')
        if re.search(r'; N\]', sig) and not re.search(r'const N\b', sig):
            sig = sig.replace('; N]', '; 8]')
        used = set(re.findall(r'\b([A-OQ-Z])\b', sig.split('(', 1)[1]))
        gm = re.match(r'^(.*?fn shape)(<[^(]*>)?(\(.*)$', sig, re.S)
        declared = set(re.findall(r'\b([A-Z])\b', gm.group(2) or ''))
        missing = sorted(used - declared)
        if missing:
            extra = ', '.join(missing)
            gens = (gm.group(2) or '<>')[1:-1]
            gens = f'<{gens}, {extra}>' if gens else f'<{extra}>'
            sig = gm.group(1) + gens + gm.group(3)
        body = ' { todo!() }'
        uses_self = 'self' in key.split('(')[1] or 'Self' in key
        item = f'    {attr}\n    {sig}{body}'
        if uses_self:
            item = f'    impl P {{\n    {attr}\n    {sig}{body}\n    }}'
        provenance = '\n'.join(f'    // {f}' for f in files[:6]) + (f'\n    // ... and {len(files) - 6} more' if len(files) > 6 else '')
        # `import_intrinsics` with an AVX-512 token needs the `avx512` feature
        # by design; such shapes are compiled only with it.
        gates = []
        if 'import_intrinsics' in attr and re.search(r'X64V4Token|X64V4xToken|Avx512Fp16Token|Server64|Avx512Token', sig):
            gates.append('feature = "avx512"')
        # The 512-bit types exist only with the `w512` feature (no_std CI runs
        # without default features).
        if re.search(r'\b[fiu](8x64|16x32|32x16|64x8)\b', sig):
            gates.append('feature = "w512"')
        gate = ''
        if gates:
            gate = f'#[cfg({gates[0]})]\n' if len(gates) == 1 else f'#[cfg(all({", ".join(gates)}))]\n'
        out.append(f'{gate}mod h{i:03}  {{\n    use super::*;\n    // {key}\n{provenance}\n{item}\n}}\n')
    return '\n'.join(out), len(groups)


if __name__ == '__main__':
    import datetime
    roots = [a for a in sys.argv[1:] if not a.startswith('--')]
    shapes = main(roots)
    crates = len({w.split('/')[0] if not w.startswith('zen/') else '/'.join(w.split('/')[:2]) for ws in shapes.values() for w in ws})
    text, n = emit(shapes, datetime.date.today().isoformat(), crates)
    target = Path('magetypes/tests/harvest_shapes.rs')
    target.write_text(text)
    print(f"{len(shapes)} signatures, {n} shapes -> {target}", file=sys.stderr)

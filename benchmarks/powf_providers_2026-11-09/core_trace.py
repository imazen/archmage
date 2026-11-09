#!/usr/bin/env python3
"""Verify executed FMA blocks in QEMU logs, including builds without Capstone."""
import bisect
import json
from pathlib import Path
import re
import subprocess

def prove(trace, library=None):
    lines = trace.read_text().splitlines()
    block = None
    fma_blocks, executed = set(), set()
    raw = {}
    for line in lines:
        instruction = re.match(r'0x([0-9a-f]+):\s*(.*)', line)
        if line.startswith('IN:'):
            block = None
        elif instruction:
            if block is None:
                block = int(instruction[1], 16)
            if re.search(r'\bvfm(?:add|sub|nadd|nsub)', instruction[2]):
                fma_blocks.add(block)
        elif line.startswith('OBJD-T:'):
            assert block is not None
            raw.setdefault(block, bytearray()).extend(bytes.fromhex(line.split(':', 1)[1].strip()))
        match = re.search(r'Trace .*\[[0-9a-f]+/([0-9a-f]+)/', line)
        if match:
            executed.add(int(match[1], 16))
    if raw:
        image, offsets, addresses = bytearray(), [], []
        for address, code in raw.items():
            offsets.append(len(image)); addresses.append(address); image.extend(code)
        binary = trace.with_suffix('.translated.bin')
        binary.write_bytes(image)
        decoded = subprocess.check_output(['objdump', '-D', '-b', 'binary', '-m', 'i386:x86-64', str(binary)], text=True)
        trace.with_suffix('.translated.asm').write_text(decoded)
        for line in decoded.splitlines():
            match = re.match(r'\s*([0-9a-f]+):.*\bvfm(?:add|sub|nadd|nsub)', line)
            if match:
                fma_blocks.add(addresses[bisect.bisect_right(offsets, int(match[1], 16))-1])
    proof = {'translated_FMA_blocks': sorted(fma_blocks),
             'executed_FMA_blocks': sorted(fma_blocks & executed),
             'trace_bytes': trace.stat().st_size,
             'evidence': 'intersection of FMA translation block addresses with executed Trace addresses'}
    assert proof['executed_FMA_blocks'], 'FMA translation alone does not prove execution'
    if library is not None:
        # This mode uses QEMU's raw bytes to identify the retained feature DSO.
        # Every executed FMA block must match uniquely at a common load bias.
        blob = library.read_bytes()
        matches = []
        for address in proof['executed_FMA_blocks']:
            code = raw[address]
            assert len(code) >= 16
            offset = blob.find(code)
            assert offset >= 0 and blob.find(code, offset+1) < 0, 'FMA block does not uniquely match the frozen DSO'
            matches.append({'address': address, 'file_offset': offset, 'bytes': len(code)})
        biases = {m['address']-m['file_offset'] for m in matches}
        assert len(biases) == 1, 'FMA blocks do not share the DSO mapping'
        import hashlib
        proof.update(library_sha256=hashlib.sha256(blob).hexdigest(),
                     DSO_byte_matches=matches, guest_minus_file_offset=biases.pop())
    return proof

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--library', type=Path, help='also match raw blocks uniquely to the retained feature DSO')
    args = parser.parse_args()
    result = prove(args.trace, args.library)
    args.trace.with_suffix('.library-proof.json' if args.library else '.proof.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

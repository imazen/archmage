#!/usr/bin/env python3
"""Audit packed operations/calls/stack references in actual linked array contexts."""
import re
import sys
from pathlib import Path
blocks=re.split(r'(?m)^(?=[0-9a-f]+ <)',Path(sys.argv[1]).read_text());count=0
for block in blocks:
    first=block.splitlines()[0] if block else ''
    if 'vector::__arcane_stream_context::<' not in first:continue
    count+=1
    calls=[line.strip() for line in block.splitlines() if re.search(r'\bcall\b',line)]
    if any(re.search(r'\b(?:call|jmp)\b.*(?:vector::block|__arcane_entry)',line) for line in block.splitlines()):raise ValueError('block/Evaluation helper call '+first)
    if not ('vmulpd' in block and 'vaddpd' in block):raise ValueError('packed arithmetic absent '+first)
    if re.search(r'\bvf[nm]*m(?:add|sub)',block):raise ValueError('implicit FMA '+first)
    stack=[line.strip() for line in block.splitlines() if '(%rsp)' in line and re.search(r'%[xy]mm',line)]
    ymm=[line for line in stack if '%ymm' in line]
    allocations=re.findall(r'sub\s+\$0x([0-9a-f]+),%rsp',block)
    print(first,'packed_mul=',block.count('vmulpd'),'packed_add=',block.count('vaddpd'),'stack_allocation_bytes=',[int(a,16) for a in allocations],'vector_stack_refs=',len(stack),'ymm_stack_refs=',len(ymm))
    for line in calls:print(' CALL',line)
    for line in stack:print(' STACK',line)
if count!=12:raise ValueError('array context inventory '+str(count))
print('PASS12 linked array4/8 contexts: packed separate arithmetic and no block/Evaluation helper calls; stack references/cold calls listed for inspection, not erased.')

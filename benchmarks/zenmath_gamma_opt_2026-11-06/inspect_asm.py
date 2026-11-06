#!/usr/bin/env python3
"""Check linked proof-context bodies, not source attributes, before timing."""
import re
import sys
from pathlib import Path
text=Path(sys.argv[1]).read_text()
blocks=re.split(r'(?m)^(?=[0-9a-f]+ <)',text)
counts={'cert':0,'std':0}
for block in blocks:
    first=block.splitlines()[0] if block else ''
    if 'timing::__arcane_cert_context::<' in first:kind='cert'
    elif 'timing::__arcane_std_context::<' in first:kind='std'
    else:continue
    counts[kind]+=1
    calls=[line.strip() for line in block.splitlines() if re.search(r'\bcall\b',line)]
    if any(re.search(r'\b(?:call|jmp)\b.*(?:cert_stream|std_stream|kernel::evaluate)',line) for line in block.splitlines()):
        raise ValueError('stream/evaluation helper failed to inline: '+first)
    if kind=='cert':
        if any('kernel::rescue::<' not in call for call in calls):raise ValueError('unexpected certificate call: '+repr(calls))
        if not ('vmulsd' in block and 'vaddsd' in block):raise ValueError('missing V3 scalar arithmetic')
        if re.search(r'\bvf[nm]*m(?:add|sub)',block):raise ValueError('unexpected fused arithmetic')
    elif 'powf@' not in block:raise ValueError('std libm reference not visible')
    print(first, 'calls=',len(calls),'vmulsd=',block.count('vmulsd'),'vaddsd=',block.count('vaddsd'))
if counts!={'cert':16,'std':2}:raise ValueError('wrong initial context inventory '+repr(counts))
print('PASS: all16 certificate loops inline arithmetic; only conditional CORE rescue calls; both std loops inline with libm powf calls; separate f64 arithmetic.')

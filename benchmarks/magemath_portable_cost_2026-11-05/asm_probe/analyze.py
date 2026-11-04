import re,sys
fns={};cur=None
for l in open(sys.argv[1]):
    l=l.rstrip('\n')
    m=re.match(r'^([_A-Za-z][\w$.]*):\s*$',l)
    if m and not m.group(1).startswith('.'):
        cur=m.group(1);fns[cur]=[];continue
    if cur and l.startswith('\t') and not l.startswith('\t.'):
        fns[cur].append(l.strip())
arm='aarch64' in sys.argv[1] or len(sys.argv)>2
print(f"{'function':22} {'ins':>4} {'fused':>5} {'stack':>5} {'calls':>5}  memory operands (non-constant)")
for k,ins in fns.items():
    name=re.sub(r'^.*?(\d+)([a-z_0-9]+)$',r'\2',k)
    if 'probe' not in k and 'asm' not in k: continue
    ops=[i.split()[0] for i in ins]
    fused=sum(o.startswith(('vfm','vfnm','fmla','fmls')) for o in ops)
    if arm:
        mem=[i for i in ins if re.search(r'\[x[0-9]+|\[sp',i) and not 'lCPI' in i and ':lo12:' not in i]
        stack=[i for i in ins if '[sp' in i]
        calls=[i for i in ins if re.match(r'(bl|blr)\b',i)]
    else:
        mem=[i for i in ins if re.search(r'\(%r[a-z0-9]+\)|\d+\(%r',i) and 'rip' not in i]
        stack=[i for i in ins if '%rsp' in i or '%rbp' in i]
        calls=[i for i in ins if i.startswith('call')]
    print(f"{name:22} {len(ins):4} {fused:5} {len(stack):5} {len(calls):5}  {len(mem)}: {[m.split()[0] for m in mem]}")

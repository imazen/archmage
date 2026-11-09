#!/usr/bin/env python3
"""Enumerate predeclared policy edges for the zenbench paired analyzer."""
from pathlib import Path
import sys
out=Path(sys.argv[1])
flags=[x for x in range(64) if x&3!=3]+[x for x in range(64,128) if x&16 and not x&8]
index={f:i for i,f in enumerate(flags)};edges=[]
for f in flags:
 if not f&32:edges.append((index[f|32],index[f],'AssumeValid/Checked'))
 if f<64 and not f&8:edges.append((index[f|8],index[f],'LOWP/MIDP'))
 if f<64 and not f&16:edges.append((index[f|16],index[f],'BackendDependent/BitExact'))
 if not f&4:edges.append((index[f|4],index[f],'Caller/Automatic'))
 if f&3==1:edges.append((index[(f&~3)|2],index[f],'Gamma/NonNegativeNormal'))
 if f&64 and f&3==2:edges.append((index[(f&~3)|3],index[f],'Full/Gamma'))
edges += [(index[2],68,'Const/Runtime'),(index[83],66,'HIGHP_Full/std_f32'),(index[2],65,'BitExact_MIDP_Gamma/legacy_midp'),(index[10],64,'BitExact_LOWP_Gamma/legacy_lowp'),(index[83],69,'HIGHP_vector_direct/ResultMap'),(70,66,'HIGHP_array/std_f32'),(index[83],70,'HIGHP_vector/array')]
assert len(set(edges))==len(edges)
code=(Path(__file__).parent/'src/bin/rounding_analyze.rs').read_text()
a=code.index('            for (numerator, denominator) in [');b=code.index('                let ratios:',a)
code=code[code.index('fn main()'):a]+'            for (numerator,denominator,edge) in [\n'+''.join(f'                ({n},{d},"{label}"),\n'for n,d,label in edges)+'            ] {\n'+code[b:]
code=code.replace('"file":path,"group":group.group_name,','"file":path,"group":group.group_name,"edge":edge,')
(out/'policy_analyzer.rs').write_text(code)

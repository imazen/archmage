#!/usr/bin/env python3
"""Select the separately preregistered CORE-MATH comparison arms."""
import argparse
import json
from pathlib import Path
from external import discover
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('rust',type=Path);p.add_argument('native',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();assert not a.output.exists()
x=discover(a.rust,a.native)
x['providers']=[p for p in x['providers']if p['method']in ['std','system_C_powf','CORE_MATH_sys_1_4_0_baseline','CORE_MATH_sys_1_4_0_FMA_no_AVX2']]
assert any(p['method']=='CORE_MATH_sys_1_4_0_baseline'for p in x['providers'])
x['study']='CORE-PLAN.md; separate additional comparison, no replacement of prior timing evidence'
a.output.write_text(json.dumps(x,indent=2)+'\n')

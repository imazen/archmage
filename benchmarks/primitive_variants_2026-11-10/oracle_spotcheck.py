#!/usr/bin/env python3
"""Cross-check the C oracle with the repository's independent rational rounder."""
from pathlib import Path
from fractions import Fraction
import random,struct,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'xtask'))
import fma_vectors
from reciprocal_vectors import round_positive
r=random.Random(0x5eed1357)
inputs={0x3f800000,0x3fc00000,0x3fffffff,0x3ffffc00,0x3fff00ff}
inputs.update(0x3f800000+r.randrange(1<<23) for _ in range(1024))
with Path(sys.argv[1]).open('rb') as f:
 for x in sorted(inputs):
  f.seek((x-0x3f800000)*16); actual=struct.unpack('<4I',f.read(16))
  y=0x7ef127ea-x;steps=[]
  for _ in range(2):
   correction=fma_vectors.expected((x^0x80000000,y,0x40000000),23,8)
   y=fma_vectors.expected((y,correction,0),23,8);steps.append(y)
  value=Fraction((x&0x7fffff)|0x800000,1<<23)
  rounded=[round_positive(Fraction(round((1/value)*(1<<p)),1<<p)) for p in [14,8]]
  expected=(steps[1],steps[0],*rounded)
  assert actual==expected,(hex(x),actual,expected)
print('Exact Fraction oracle agrees with all four C columns for',len(inputs),'directed/random inputs')

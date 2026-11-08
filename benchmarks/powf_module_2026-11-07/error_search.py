#!/usr/bin/env python3
"""Build a pinned external math_accuracy search against actual Rust methods.

The GPL upstream source stays outside the repository, with its license notice.
This adapter restricts distance evaluations to the declared canonical domain.
Search results are witnesses, never certified global error bounds.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

HASHES = {
    'check_sample2.c':'a4c95684deaf44125e485a821741a376f5f34ea540befbff4c089727215231cd',
    'random.c':'d0afbcfdc8ef8d091ce5f28bacfb338ec3b35067da0c24fc0c35607e777641e3',
    'random.h':'ad6c510cbec0a7498ff9fac766c57fdd657c1625bc754da2a2d7ae6f35d33a6e',
}

def replace_once(source, before, after):
    assert source.count(before)==1, ('upstream patch anchor changed',before)
    return source.replace(before,after)

def build(a):
    a.directory.mkdir(parents=True,exist_ok=False)
    for name,digest in HASHES.items():
        raw=(a.source/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==digest, name
        (a.directory/name).write_bytes(raw)
    symbols=subprocess.check_output(['nm','-g','--defined-only',str(a.library)],text=True)
    symbols=[line.split()[-1]for line in symbols.splitlines()if 'replacement_evaluate'in line and ' T 'in line]
    assert len(symbols)==1
    header='''
/* Rust callback: compiler-assigned C-ABI symbol, located from this archive. */
extern uint64_t actual_pow(float,float,uint32_t,uint32_t,uint32_t) __asm__("SYMBOL");
static unsigned candidate, domain, native;
static uint64_t admitted, rejected, calls;
static void study_init(void) {
  const char *c=getenv("POWF_SEARCH_METHOD"),*d=getenv("POWF_SEARCH_DOMAIN"),*t=getenv("POWF_SEARCH_NATIVE");
  assert(c && d && t); candidate=strtoul(c,0,10);domain=strtoul(d,0,10);native=strtoul(t,0,10);
  assert(candidate<8 && domain<3 && native<2);
}
static int study_admitted(float x,float y) {
  uint64_t r=actual_pow(x,y,0,domain,native);
  if(r>>32){rejected++;return 0;}admitted++;return 1;
}
static float study_value(float x,float y) {
  uint64_t r=actual_pow(x,y,candidate,domain,native);assert(!(r>>32));
  uint32_t bits=(uint32_t)r;float value;memcpy(&value,&bits,sizeof(value));calls++;return value;
}
'''.replace('SYMBOL',symbols[0])
    source=(a.directory/'check_sample2.c').read_text()
    # Upstream STAT calls this diagnostic printer, whose definition is disabled.
    source=replace_once(source,'#if 0\nstatic void\nList_print','#ifdef STAT\nstatic void\nList_print')
    source=replace_once(source,'#include <fenv.h>','#include <fenv.h>\n'+header)
    source=replace_once(source,'  return FOO2 (x, y);','  return study_value (x, y);')
    source=replace_once(source,'distance (TYPEX x, TYPEY y)\n{','distance (TYPEX x, TYPEY y)\n{\n  if (!study_admitted(x,y)) return 0;')
    source=replace_once(source,'  Xmin = 0;','  study_init();\n  Xmin = 0;')
    source=replace_once(source,'  fflush (stdout);\n  mpfr_free_cache ();','''  fflush (stdout);
  uint32_t xb,yb;
  memcpy(&xb,&Xbest,4);memcpy(&yb,&Ybest,4);
  printf("ARCHMAGE_WITNESS %08x %08x %u\\n",xb,yb,domain);
  printf("ARCHMAGE_EFFORT admitted=%lu rejected=%lu calls=%lu\\n",admitted,rejected,calls);
  mpfr_free_cache ();''')
    patched=a.directory/'check_sample2_archmage.c'
    patched.write_text(source)
    command=['cc','-O3','-DFOO=pow','-DUSE_FLOAT','-DNO_FLOAT128','-DWORST',
             '-DNUMBER=0','-DSTAT','-fopenmp',str(patched),str(a.directory/'random.c'),
             str(a.library.resolve()),'-lmpfr','-lgmp','-lm','-ldl','-lpthread',
             '-lrt','-lutil','-o',str(a.directory/'search')]
    subprocess.run(command,check=True)
    manifest={'source_sha256':HASHES,'rust_library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),
              'rust_symbol':symbols[0],'command':command,'patched_sha256':hashlib.sha256(patched.read_bytes()).hexdigest(),
              'search_precision':72,'witness_confirmation_precision':256,
              'search_is_full_domain_bound':False}
    (a.directory/'BUILD.json').write_text(json.dumps(manifest,indent=2)+'\n')

def run(a):
    a.directory.mkdir(parents=True,exist_ok=False)
    domains=[['-xmin','0x1p-126','-xmax','0x1.fffffep127'],
             ['-xmin','0x0p0','-xmax','0x1.fffffep127'],
             ['-xmin','0x1p-16','-xmax','0x1p16','-ymin','0x0p0','-ymax','0x1p2']]
    seeds=['0x1.3e66c4p43,0x1.333334p1',
           '0x1.31c29ep-17,0x1.91c0d6p2',
           '0x1.11bb04p-16,0x1.333334p1']
    cells=[(method,domain,tier,mode)for method in range(8)for domain in range(3)
           for tier in range(2)for mode in range(3)]
    if a.pilot:cells=cells[:1]
    manifest={'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),
              'threshold':4096,'seed':271828183,'threads':1,'cells':cells,
              'budget_note':'fixed adaptive effort threshold; 300s fail-loud ceiling per search',
              'domains':domains,'initial_witnesses':seeds,'pilot':a.pilot}
    (a.directory/'MATRIX.json').write_text(json.dumps(manifest,indent=2)+'\n')
    witnesses=[]
    for index,(method,domain,tier,mode)in enumerate(cells):
        progress={'pid':os.getpid(),'stage':'running','index':index,'required':len(cells),
                  'cell':[method,domain,tier,mode],'last_check':datetime.now(timezone.utc).isoformat()}
        (a.directory/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        command=[str(a.binary.resolve()),'-threshold','4096','-seed','271828183',
                 '-mode',str(mode),'-nthreads','1','-rndn','-v','-worst',seeds[domain]]+domains[domain]
        env=dict(os.environ,POWF_SEARCH_METHOD=str(method),POWF_SEARCH_DOMAIN=str(domain),POWF_SEARCH_NATIVE=str(tier))
        with(a.directory/f'{index:03d}.log').open('w')as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        assert result.returncode==0, ('search failed',index,result.returncode)
        log=(a.directory/f'{index:03d}.log').read_text()
        found=re.findall(r'ARCHMAGE_WITNESS ([0-9a-f]{8} [0-9a-f]{8} [012])',log)
        assert len(found)==1 and 'ARCHMAGE_EFFORT'in log
        witnesses+=found
        (a.directory/'witnesses.inputs').write_text('\n'.join(sorted(set(witnesses)))+'\n')
        progress.update(stage='complete',last_check=datetime.now(timezone.utc).isoformat())
        (a.directory/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(progress),flush=True)

p=argparse.ArgumentParser(description=__doc__)
sub=p.add_subparsers(dest='operation',required=True)
b=sub.add_parser('build');b.add_argument('source',type=Path);b.add_argument('library',type=Path);b.add_argument('directory',type=Path)
r=sub.add_parser('run');r.add_argument('binary',type=Path);r.add_argument('directory',type=Path);r.add_argument('--pilot',action='store_true')
a=p.parse_args()
build(a)if a.operation=='build'else run(a)

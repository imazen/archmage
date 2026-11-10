#!/usr/bin/env python3
"""Independent per-center rational certificates for the dyadic16 experiment."""
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import subprocess
from generate import U,ETA,coeffs,polynomial_add,polynomial_mul,pow2ceil,facts
from mpfr_support import f32,bits64,certify_fraction,certify_power,lib

H=F(1,32)
def mag(poly):return sum(abs(a)*H**k for k,a in enumerate(poly))
class Node:
    def __init__(self,poly,name,error=F(0)):
        self.poly=poly;self.name=name;self.error=error;self.bound=mag(poly)
class Dag:
    def __init__(self):self.lines=[];self.operations=0;self.largest=F(0)
    def op(self,a,b,kind):
        if kind=='+':
            poly=polynomial_add(a.poly,b.poly)
            error=a.error+b.error+U*(a.bound+b.bound+a.error+b.error)+ETA
        else:
            poly=polynomial_mul(a.poly,b.poly)
            error=a.bound*b.error+b.bound*a.error+a.error*b.error+U*(a.bound+a.error)*(b.bound+b.error)+ETA
        name=f'z{self.operations}';self.operations+=1
        n=Node(poly,name,error);self.lines.append(f'let {name} = {a.name} {kind} {b.name};')
        self.largest=max(self.largest,n.bound+n.error)
        if n.bound+n.error>=8:raise ValueError('Estrin intermediate bound')
        return n

def estrin(stored,n):
    dag=Dag();nodes=[Node([v],f'COEFF[K][center][{i}]') for i,v in enumerate(stored[:n+1])];power=Node([F(0),F(1)],'v')
    while len(nodes)>1:
        nodes=[dag.op(nodes[i],dag.op(nodes[i+1],power,'*'),'+') if i+1<len(nodes) else nodes[i] for i in range(0,len(nodes),2)]
        if len(nodes)>1:power=dag.op(power,power,'*')
    if nodes[0].poly!=stored[:n+1]:raise ValueError('Estrin ideal polynomial')
    return nodes[0].error,dag.lines,dag.largest

def up(v):return math.nextafter(float(v),math.inf)
def literal(v):return f'f64::from_bits(0x{bits64(v):016x})'
def rust_array(value):
    if isinstance(value,list):return '['+','.join(rust_array(v) for v in value)+']'
    return literal(value)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rust',type=Path,required=True)
    parser.add_argument('--proof',type=Path,required=True);parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--rustfmt',required=True);args=parser.parse_args()
    info=facts();report={'version':1,'mpfr':lib.mpfr_get_version().decode(),'precision':1024,'facts':info,'cases':[],'coefficients':[]}
    all_coeff=[];all_factor=[];all_error=[];functions=[];summaries=['kind,center,degree,layout,tail_upper,coefficient_error_upper,evaluation_error_upper,max_suffix_upper,max_pad\n']
    if not ((8*H)*(1+U)+ETA+4)*(1+U)+ETA<8:raise ValueError('coarse two-round invariant')
    for kind in ['decode','encode']:
        p=F(f32(info[kind+'_p_bits']));center_coeff=[];center_tables=[];center_errors=[]
        for j in range(16):
            c=F(33+2*j,32);r=H/c
            a=[b/(F(2,3)*c)**k for k,b in enumerate(coeffs(p,6))]
            certs=[certify_fraction(v) for v in a[:6]];b=[F(v['value']) for v in certs]
            if max(abs(v) for v in b)>4:raise ValueError('new coefficient invariant')
            center_coeff.append([v['value'] for v in certs])
            report['coefficients'].append(dict(kind=kind,center=j,values=[dict(bits=f'{bits64(v["value"]):016x}',lower_bits=f'{bits64(v["lower"]):016x}',upper_bits=f'{bits64(v["upper"]):016x}',exact_numerator=str(ex.numerator),exact_denominator=str(ex.denominator)) for ex,v in zip(a,certs)]))
            tables=[certify_power(float(c*F(2)**e),float(p)) for e in range(-9,1)]
            center_tables.append([t['value'] for t in tables]);layouts=[]
            for layout in ['horner','estrin']:
                degrees=[]
                for n in range(3,6):
                    tail=abs(a[n+1])*H**(n+1)/(1-r)
                    cerror=sum(abs(b[k]-a[k])*H**k for k in range(n+1));s=sum(abs(b[k])*H**k for k in range(n+1))
                    suffix=max((1+2*(n-jj)*U/(1-2*(n-jj)*U))*sum(abs(b[k])*H**(k-jj) for k in range(jj,n+1))+8*(n-jj)*ETA for jj in range(n+1))
                    if suffix>=8:raise ValueError('suffix')
                    if layout=='horner':evaluation=2*n*U/(1-2*n*U)*s+8*n*ETA
                    else:
                        evaluation,lines,largest=estrin(b,n)
                        if kind=='decode' and j==0:
                            functions.append(f'#[inline(always)]\npub(crate) fn estrin_{n}<const K:usize>(center:usize,v:f64)->f64 {{\n'+''.join('    '+line+'\n' for line in lines[:-1])+f'    {lines[-1].split(" = ",1)[1][:-1]}\n}}\n')
                    epoly=tail+cerror+evaluation;bound=sum(abs(a[k])*H**k for k in range(n+1))+tail
                    if epoly>=F(1,64):raise ValueError('poly margin')
                    errors=[];table_details=[]
                    for e,t in zip(range(-9,1),tables):
                        stored=F(t['value']);dt=max(abs(stored-F(t['lower'])),abs(F(t['upper'])-stored))
                        if not F(1,2**32)<F(t['lower']) or not 0<stored<8 or dt>=stored/F(2**48):raise ValueError('new table invariant')
                        total=stored*epoly+dt*bound+U*stored*(s+evaluation)+ETA;delta=pow2ceil(total)
                        low=stored*(F(1,4)-epoly)-U*stored*(s+evaluation)-ETA;high=stored*(s+evaluation)*(1+U)+ETA
                        if not low>stored/8 or not delta<stored/32 or not high+delta<64:raise ValueError('endpoint invariant')
                        errors.append(float(delta));table_details.append(dict(exponent=e,table_bits=f'{bits64(t["value"]):016x}',lower_bits=f'{bits64(t["lower"]):016x}',upper_bits=f'{bits64(t["upper"]):016x}',delta_bits=f'{bits64(float(delta)):016x}'))
                    degrees.append(errors)
                    report['cases'].append(dict(kind=kind,center=j,degree=n,layout=layout,tail_upper=up(tail),coefficient_error_upper=up(cerror),evaluation_error_upper=up(evaluation),suffix_upper=up(suffix),table=table_details))
                    summaries.append(f'{kind},{j},{n},{layout},{up(tail):.18e},{up(cerror):.18e},{up(evaluation):.18e},{up(suffix):.18e},{max(errors).hex()}\n')
                layouts.append(degrees)
            center_errors.append(layouts)
        all_coeff.append(center_coeff)
        all_factor.append([[center_tables[j][e] for j in range(16)] for e in range(10)])
        all_error.append([[[[center_errors[j][l][n][e] for j in range(16)] for e in range(10)] for n in range(3)] for l in range(2)])
    source='// Generated by generate_piecewise.py; exact binary64 bit literals.\n'
    for name,typ,array in [('COEFF','[[[f64;6];16];2]',all_coeff),('FACTOR','[[[f64;16];10];2]',all_factor),('ERROR','[[[[[f64;16];10];3];2];2]',all_error)]:source+=f'pub(crate) const {name}:{typ}='+rust_array(array)+';\n'
    source+=''.join(functions)+'#[inline(always)]\npub(crate) fn estrin<const K:usize,const N:usize>(center:usize,v:f64)->f64 {match N {3=>estrin_3::<K>(center,v),4=>estrin_4::<K>(center,v),5=>estrin_5::<K>(center,v),_=>panic!("degree")}}\n'
    args.rust.write_text(source);subprocess.run([args.rustfmt,'--edition','2024',str(args.rust)],check=True)
    report['rust_sha256']=hashlib.sha256(args.rust.read_bytes()).hexdigest();report['emitted_table_bytes']=19456
    args.proof.write_text(json.dumps(report,indent=2)+'\n');args.summary.write_text(''.join(summaries))
    print(json.dumps(dict(cases=len(report['cases']),rust_sha256=report['rust_sha256'],emitted_table_bytes=report['emitted_table_bytes'])))

if __name__=='__main__':main()

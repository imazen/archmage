#!/usr/bin/env python3
"""Freeze actual Magetypes HIGHP and scalar/legacy comparison implementations."""
import argparse
import json
from pathlib import Path
from external import digest, native_identity, save
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('worker',type=Path);p.add_argument('output',type=Path)
p.add_argument('--source',required=True)
p.add_argument('--wide',action='store_true')
a=p.parse_args();assert not a.output.exists()
methods=['std','Rust_libm_0_2_16','Arm_Rust_fused_cold','Arm_Rust_fused_assume_Gamma',
         'Magetypes_HIGHP_Full_packed','Magetypes_HIGHP_Gamma_checked_packed',
         'Magetypes_HIGHP_Gamma_assumed_packed','Magetypes_legacy_midp','Magetypes_legacy_lowp']
if a.wide: methods += [m+'_V4x' for m in methods if m.startswith('Magetypes_HIGHP_')]
save(a.output,{'workers':[{'name':'Rust_library_comparison','binary':str(a.worker.resolve()),'sha256':digest(a.worker)}],
 'providers':[{'label':m,'method':m,'worker':0,'lanes':4 if 'legacy' in m else 1}for m in methods],
 'dsos':{},'native_identity':native_identity(),'source':a.source,
 'timing_unit':'ns per 1048576 actual powers, including one socket request',
 'contract':'Full canonical graph versus std/libm and the prior scalar Arm graph; Gamma declared domain checked/assumed; legacy graphs are different numerical contracts',
 'width_one':'Magetypes HIGHP x1 uses its actual scalar Automatic public API; wider shapes use Caller vector delegates in the whole-batch V3/NEON feature context. Legacy entries start at x4.',
 'matrix':'32 existing cells: widths1/4/8/16, size64/4096, stream/normalized-chain, exponent2.4/0.5; 50 interleaved rounds. First inadmissible strict Linux cell stops, with all evidence retained. macOS exploratory is labeled and excluded from promotion.'})

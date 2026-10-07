#!/usr/bin/env python3
"""Read-only lexical inventory. Run: python3 inventory.py. No third-party modules.
Every attribute (including support attrs), dispatch call and textual mention in
all archive tests/examples is indexed. Does not expand Rust macros or cfgs.
"""
from pathlib import Path
import re, subprocess, hashlib, json, argparse
from collections import Counter
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=HERE/'source', help='Pinned source archive root (read only)')
parser.add_argument('--out', type=Path, default=HERE, help='Report artifact directory')
args=parser.parse_args()
SRC=args.source.resolve()
ROOT=args.out.resolve()
if not SRC.is_dir(): parser.error('source must be an existing directory')
if ROOT == SRC or SRC in ROOT.parents: parser.error('out must be outside source')
ROOT.mkdir(parents=True, exist_ok=True)
BASE='https://github.com/imazen/archmage/blob/cf07592212e96294ef9ca5dca9a364fa8d15d8ad/'
MACROS={'arcane':'P1','simd_fn':'P8','token_target_features_boundary':'P8','rite':'P2','token_target_features':'P8','autoversion':'P3','magetypes':'P4','incant':'P5','simd_route':'P8','dispatch_variant':'P8'}
# Ordered lexical regions; retain offsets/newlines while blanking comments/literals.
LEX=re.compile(r'//[^\n]*|/\*|(?:br|cr|r)(?P<hash>\#*)"|(?:b|c)?"|\'(?:\\.|[^\'\\\n])\'',re.M)
def mask(s):
    out=list(s); regions=[]; pos=0
    while (m:=LEX.search(s,pos)):
        a=m.start(); v=m.group(); end=m.end(); kind='literal'
        if v.startswith('//'): kind='comment'
        elif v=='/*':
            kind='comment'; depth=1
            while depth and end<len(s):
                n=re.search(r'/\*|\*/',s[end:])
                if not n: end=len(s); break
                depth+=1 if n.group()=='/*' else -1; end+=n.end()
        elif m.group('hash') is not None:
            close='"'+m.group('hash'); at=s.find(close,end); end=len(s) if at<0 else at+len(close)
        elif v.endswith('"'):
            while end<len(s):
                if s[end]=='\\': end+=2
                elif s[end]=='"': end+=1; break
                else: end+=1
        for i in range(a,end):
            if s[i]!='\n': out[i]=' '
        regions.append((a,end,kind)); pos=end
    return ''.join(out),regions
PAT=re.compile(r'#\s*!?\s*\[|\b(?:[A-Za-z_]\w*\s*::\s*)*(incant|simd_route|dispatch_variant)\s*!(?:\s*[({\[])?')
def balanced(s,start):
    pairs={'[':']','(':')','{':'}'}; stack=[]
    for i in range(start,len(s)):
        c=s[i]
        if c in pairs: stack.append(pairs[c])
        elif c in '])}':
            if not stack or stack.pop()!=c:return i+1
            if not stack:return i+1
    return len(s)
def signature_end(s, start):
    stack=[]
    for i in range(start,len(s)):
        c=s[i]
        if c in '([':stack.append(')' if c=='(' else ']')
        elif c=='<':stack.append('>')
        elif c=='{' and stack:stack.append('}')
        elif c in ')]>}':
            if stack and c==stack[-1]:stack.pop()
        elif c in '{;' and not stack:return i
    return len(s)
def cls(path):
    if path.endswith('.expanded.rs'):return 'snapshot'
    if path.endswith('.stderr'):return 'diagnostic'
    if any(x in path.split('/') for x in ['compile_fail','ui','should-fail']):return 'expected-failure'
    if '/soundness/' in path:
        return 'soundness-control' if Path(path).stem=='raw_matching_context' else 'expected-failure'
    if 'v4-import-intrinsics-no-feature/' in path:return 'expected-failure'
    if '/expand/' in path:return 'expansion-input'
    return 'source'
def route(name,snippet,path):
    if name not in MACROS:return 'P0'
    if cls(path)=='expected-failure':return 'P9'
    if name in ('incant','dispatch_variant','simd_route'):
        if 'without token' in snippet:return 'P6'
        if re.search(r'\bwith\s',snippet):return 'P7'
    return MACROS[name]
def short(s):return re.sub(r'\s+',' ',s).strip()
files=sorted(subprocess.check_output(['rg','--files','--hidden',str(SRC)],text=True).splitlines())
files=[Path(p) for p in files if any(c in ('tests','examples') for c in Path(p).relative_to(SRC).parts)]
rows=[]; manifest=[]; seen=set(); rg_candidates=0
for p in files:
    path=p.relative_to(SRC).as_posix(); data=p.read_bytes(); s=data.decode('utf8',errors='replace')
    manifest.append((path,len(data),hashlib.sha256(data).hexdigest(),cls(path)))
    if p.suffix not in ('.rs','.stderr'):continue
    clean,regions=mask(s)
    for m in PAT.finditer(s):
        a=m.start(); line=s.count('\n',0,a)+1; region=next((k for x,y,k in regions if x<=a<y),None)
        literal=region or ('diagnostic' if p.suffix=='.stderr' else 'code')
        opening=m.end()-1
        if s[a]=='#':
            end=balanced(clean if not region else s,opening)
            # Mentions in prose can have unmatched delimiters; keep only this line.
            if region and ('\n' in s[a:end] or end-a>1000):end=s.find('\n',a) if '\n' in s[a:] else len(s)
            name_match=re.match(r'#\s*!?\s*\[\s*((?:\w+::)*\w+)',s[a:end])
            name=name_match.group(1).split('::')[-1] if name_match else '?'
            typ='attribute'
        else:
            end=balanced(clean if not region else s,opening) if s[opening] in '([{ ' and s[opening]!=' ' else m.end(); name=m.group(1);typ='call'
            if region and end-a>2500:end=s.find('\n',a) if '\n' in s[a:] else len(s)
        snippet=s[a:end]
        context=''
        if typ=='attribute' and literal=='code' and name in MACROS:
            tail=clean[end:]; fm=re.search(r'\bfn\s+(?:\$\w+|\w+)',tail)
            if fm and fm.start()<2000:
                st=end+fm.start(); fin=signature_end(clean,st)
                prefix=clean[end:st]
                vm=re.search(r'(pub(?:\([^)]*\))?\s+)?((?:unsafe\s+|async\s+|const\s+|extern\s+)*)$',prefix)
                sig_start=end+vm.start() if vm else st
                context=short(s[sig_start:fin])
        rule=route(name,snippet,path)
        if literal!='code':rule='P10'
        if cls(path)=='snapshot':rule='P11'
        rows.append(dict(path=path,line=line,offset=a,category=cls(path),region=literal,kind=typ,name=name,rule=rule,source=snippet,signature=context))
        seen.add((path,a))
# Secondary rg candidate reconciliation: all starts found independently by rg.
pat=r'#\s*!?\s*\[|\b(incant|simd_route|dispatch_variant)\s*!'
unaccounted=[]
for p in files:
    if p.suffix not in ('.rs','.stderr'):continue
    result=subprocess.run(['rg','--json',pat,str(p)],text=True,capture_output=True)
    for line in result.stdout.splitlines():
        obj=json.loads(line)
        if obj['type']!='match':continue
        d=obj['data']; rg_candidates+=len(d['submatches'])
        path=p.relative_to(SRC).as_posix()
        for sub in d['submatches']:
            bytepos=d['absolute_offset']+sub['start']
            pos=len(p.read_bytes()[:bytepos].decode('utf8'))
            if not any(r['path']==path and r['offset']<=pos<r['offset']+len(r['source']) for r in rows):
                unaccounted.append([path,d['line_number'],sub['match']['text']])
# Serializing full source spans makes every item searchable without source access.
counts=Counter((r['category'],r['region'],'migration' if r['name'] in MACROS else 'support') for r in rows)
summary={'unaccounted_rg_candidates':unaccounted,'files':len(files),'rust_files':sum(p.suffix=='.rs' for p in files),'records':len(rows),'rg_line_candidates':rg_candidates,'counts':{' / '.join(k):v for k,v in sorted(counts.items())},'macros':dict(Counter(r['name'] for r in rows if r['region']=='code' and r['category']!='snapshot' and r['name'] in MACROS))}
# Full rows are Markdown rather than a single oversize machine artifact.
parts=[]; chunks=[]; size=0
for i,r in enumerate(rows,1):
    link=f"{BASE}{r['path']}#L{r['line']}"
    text=f"### O{i:05d} [{r['path']}:{r['line']}]({link})\n\n{r['category']} / {r['region']} / {r['name']} → [{r['rule']}](migration.md#{r['rule'].lower()})\n\n```rust\n{r['source']}\n```\n"
    if r['signature']:text+='\nAttached signature (lexical): `'+r['signature']+'`\n'
    text+='\n'
    if size+len(text.encode())>27000 and chunks:
        parts.append(''.join(chunks));chunks=[];size=0
    chunks.append(text);size+=len(text.encode())
if chunks:parts.append(''.join(chunks))
for i,part in enumerate(parts,1):(ROOT/f'occurrences-{i:03d}.md').write_text('# Occurrence index\n\nExact source spans; destination rules in migration.md. Support attributes are retained unless the rule explains otherwise.\n\n'+part)
summary['index_parts']=len(parts)
summary['file_categories']=dict(Counter(m[3] for m in manifest))
summary['rust_file_categories']=dict(Counter(m[3] for m in manifest if m[0].endswith('.rs')))
summary['macro_by_category']={cat:dict(Counter(r['name'] for r in rows if r['category']==cat and r['region']=='code' and r['name'] in MACROS)) for cat in sorted(set(r['category'] for r in rows))}
assert not unaccounted, unaccounted
(ROOT/'inventory-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
# Manifest split as well; includes non-Rust files and zero-occurrence files.
for i in range(0,len(manifest),120):
    (ROOT/f'scope-{i//120+1:02d}.tsv').write_text('path\tbytes\tsha256\tcategory\n'+''.join('\t'.join(map(str,r))+'\n' for r in manifest[i:i+120]))
# Compact priority index: actual source macro attrs/calls only, with exact spans.
# P0 support-attribute records remain exclusively in the raw artifact.
selected=[r for r in rows if r['region']=='code' and r['category']!='snapshot' and r['name'] in MACROS]
compact=[]; chunk=''; last=None
for r in selected:
    entry=''
    if last!=r['path']:
        entry=f"\n## [{r['path']}]({BASE}{r['path']}#L{r['line']})\n\n"
    entry+=f"- L{r['line']} [{r['category']}]: `{short(r['source'])}`"
    if r['signature']:
        entry+=f" on `{r['signature']}`"
        visibility=re.match(r'pub(?:\([^)]*\))?', r['signature'])
        entry+=f" [visibility: {visibility.group() if visibility else 'implicit; method/trait context unresolved, see source'}]"
    elif r['kind']=='attribute':entry+=' [signature/visibility unresolved; see source]' 
    entry+=f" → [{r['rule']}](migration.md#{r['rule'].lower()}); apply [contract rules](migration-contracts.md#p12).\n"
    if len((chunk+entry).encode())>24500:
        compact.append(chunk); chunk='';last=None
        if not entry.startswith('\n## '):entry=f"\n## [{r['path']}]({BASE}{r['path']}#L{r['line']}) (continued)\n\n"+entry
    chunk+=entry;last=r['path']
if chunk:compact.append(chunk)
for i,part in enumerate(compact,1):
    (ROOT/f'source-index-{i:02d}.md').write_text('# Compact source migration index\n\nExact old macro/call → destination pattern. Signatures are lexical, including macro templates, not compiler-resolved. Original file links plus L numbers locate every item. All support attributes and textual mentions remain in the raw inventory.\n'+part)
summary['compact_records']=len(selected)
summary['compact_parts']=len(compact)
(ROOT/'inventory-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fail='# Failure and control file index\n\nStatus is fixture intent, not a fresh execution result. See migration-contracts.md#p9 for exceptions, harness and dynamic cases.\n\n'
for path,_,_,cat in manifest:
    if cat in ('expected-failure','soundness-control') and path.endswith('.rs'):
        status='dormant rejection (not currently run)' if path.endswith('/scalar_not_in_tier_list.rs') else cat
        fail+=f'- [{path}]({BASE}{path}#L1): {status}.\n'
(ROOT/'failure-index.md').write_text(fail)
print(json.dumps(summary,indent=2))

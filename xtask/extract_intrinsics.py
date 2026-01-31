#!/usr/bin/env python3
"""Extract all SIMD intrinsics from Rust stdarch source.

Outputs CSV: arch,name,features,unsafe,stability,file
"""

import os
import re
import sys

def extract_intrinsics(arch_dir, arch_name):
    """Extract intrinsics from a stdarch architecture directory."""
    results = []

    for dirpath, _dirnames, filenames in sorted(os.walk(arch_dir)):
        for filename in sorted(filenames):
            if not filename.endswith('.rs'):
                continue
            if 'test' in filename:
                continue

            filepath = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(filepath, arch_dir)

            with open(filepath, 'r') as f:
                content = f.read()

            tf_pattern = re.compile(r'#\[target_feature\(enable\s*=\s*"([^"]+)"\)\]')
            fn_pattern = re.compile(r'pub\s+(?:unsafe\s+)?fn\s+(\w+)')
            stable_pattern = re.compile(r'#\[stable\(')
            unstable_pattern = re.compile(r'#\[unstable\(')
            # pub use X as Y — re-exports inherit the features of X
            reexport_pattern = re.compile(r'pub\s+use\s+(\w+)\s+as\s+(\w+)\s*;')

            # Build a name→entry map for re-export resolution
            local_entries = {}

            lines = content.split('\n')
            i = 0
            while i < len(lines):
                stripped = lines[i].strip()
                tf_match = tf_pattern.search(stripped)
                if tf_match:
                    features = tf_match.group(1)

                    j = i + 1
                    fn_name = None
                    is_unsafe = False
                    stability = "unknown"
                    bracket_depth = 0

                    while j < len(lines) and j < i + 30:
                        fline = lines[j].strip()

                        if bracket_depth > 0:
                            bracket_depth += fline.count('[') - fline.count(']')
                            j += 1
                            continue

                        if stable_pattern.search(fline):
                            stability = "stable"
                        if unstable_pattern.search(fline):
                            stability = "unstable"

                        fn_match = fn_pattern.match(fline)
                        if fn_match:
                            fn_name = fn_match.group(1)
                            is_unsafe = 'unsafe' in fline.split('fn')[0]
                            break

                        if fline.startswith('#['):
                            bracket_depth = fline.count('[') - fline.count(']')
                            j += 1
                            continue

                        if (fline.startswith('//') or fline.startswith('///') or
                            fline == '' or fline.startswith('*')):
                            j += 1
                            continue

                        break

                    if fn_name:
                        entry = {
                            'arch': arch_name,
                            'name': fn_name,
                            'features': features,
                            'unsafe': is_unsafe,
                            'stability': stability,
                            'file': rel_path,
                        }
                        results.append(entry)
                        local_entries[fn_name] = entry
                        i = j + 1
                        continue

                i += 1

            # Process re-exports: pub use X as Y
            for line in lines:
                stripped = line.strip()
                re_match = reexport_pattern.match(stripped)
                if re_match:
                    source_name = re_match.group(1)
                    alias_name = re_match.group(2)
                    if source_name in local_entries:
                        src = local_entries[source_name]
                        results.append({
                            'arch': arch_name,
                            'name': alias_name,
                            'features': src['features'],
                            'unsafe': src['unsafe'],
                            'stability': src['stability'],
                            'file': rel_path,
                        })

    return results

def main():
    sysroot = os.popen('rustc --print sysroot').read().strip()
    stdarch_base = os.path.join(sysroot, 'lib/rustlib/src/rust/library/stdarch/crates/core_arch/src')

    all_intrinsics = []

    for (subdir, label, arch) in [
        ('x86', 'x86 (shared)', 'x86'),
        ('x86_64', 'x86_64 specific', 'x86_64'),
        ('arm_shared', 'arm_shared', 'aarch64'),
        ('aarch64', 'aarch64 specific', 'aarch64'),
        ('wasm32', 'wasm32', 'wasm32'),
    ]:
        d = os.path.join(stdarch_base, subdir)
        if os.path.isdir(d):
            intrinsics = extract_intrinsics(d, arch)
            print(f"{label}: {len(intrinsics)} intrinsics", file=sys.stderr)
            all_intrinsics.extend(intrinsics)

    print("arch,name,features,unsafe,stability,file")
    for entry in all_intrinsics:
        unsafe_str = "True" if entry['unsafe'] else "False"
        print(f"{entry['arch']},{entry['name']},{entry['features']},{unsafe_str},{entry['stability']},{entry['file']}")

    print(f"\nTotal: {len(all_intrinsics)} intrinsics", file=sys.stderr)

if __name__ == '__main__':
    main()

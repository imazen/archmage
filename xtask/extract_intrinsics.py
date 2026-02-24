#!/usr/bin/env python3
"""Extract all SIMD intrinsics from Rust stdarch source.

Outputs CSV: arch,name,features,unsafe,stability,file,doc,signature,instruction

Uses Python's csv.writer for proper RFC 4180 quoting.
"""

import csv
import os
import re
import sys
import io


def extract_doc_comment(lines, start_idx):
    """Extract doc comments above the target_feature attribute.

    Looks backwards from start_idx, skipping #[inline], #[cfg_attr(...)],
    and other attributes to find consecutive /// or #[doc = "..."] lines.
    Returns the full doc string (preserving Intel/ARM links).
    """
    doc_lines = []
    i = start_idx - 1
    while i >= 0:
        stripped = lines[i].strip()
        # /// style doc comments
        if stripped.startswith('///'):
            text = stripped[3:].strip()
            doc_lines.insert(0, text)
            i -= 1
            continue
        # #[doc = "..."] style (ARM generated code)
        doc_attr = re.match(r'#\[doc\s*=\s*"(.*)"\]', stripped)
        if doc_attr:
            text = doc_attr.group(1)
            doc_lines.insert(0, text)
            i -= 1
            continue
        # Skip blank lines
        if stripped == '':
            i -= 1
            continue
        # Skip non-doc attributes: #[inline], #[cfg_attr(...)], etc.
        if stripped.startswith('#[') and not stripped.startswith('#[doc'):
            # Handle multi-line attributes
            if stripped.count('[') > stripped.count(']'):
                # Walk backwards to find the opening
                depth = stripped.count(']') - stripped.count('[')
                while i > 0 and depth < 0:
                    i -= 1
                    depth += lines[i].strip().count('[') - lines[i].strip().count(']')
            i -= 1
            continue
        break
    return ' '.join(doc_lines) if doc_lines else ''


def extract_instruction(lines, start_idx, end_idx):
    """Extract the assert_instr(...) instruction name between start and end."""
    for i in range(start_idx, min(end_idx, len(lines))):
        stripped = lines[i].strip()
        m = re.search(r'assert_instr\(\s*([^,)]+)', stripped)
        if m:
            return m.group(1).strip()
    return ''


def extract_signature(lines, fn_line_idx):
    """Extract the full function signature from the fn line.

    Handles multi-line signatures by collecting until we find '{' or ';'.
    Returns 'fn name(params) -> ret' format.
    """
    sig_lines = []
    i = fn_line_idx
    while i < len(lines):
        line = lines[i].strip()
        sig_lines.append(line)
        if '{' in line or ';' in line:
            break
        i += 1

    full = ' '.join(sig_lines)
    # Remove pub, unsafe, const, extern qualifiers
    full = re.sub(r'\bpub\s+', '', full)
    full = re.sub(r'\bunsafe\s+', '', full)
    full = re.sub(r'\bconst\s+', '', full)
    full = re.sub(r'\bextern\s+"[^"]*"\s*', '', full)
    # Trim at '{' or ';'
    full = re.split(r'\s*[{;]', full)[0].strip()
    # Remove #[inline] etc. that might be on the same line
    full = re.sub(r'#\[[^\]]*\]\s*', '', full)
    return full


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
                    tf_line = i

                    # Collect doc comment above the target_feature line
                    doc = extract_doc_comment(lines, tf_line)

                    j = i + 1
                    fn_name = None
                    is_unsafe = False
                    stability = "unknown"
                    bracket_depth = 0
                    # Track multi-line cfg_attr context for ARM stability
                    in_arm_only_attr = False
                    in_not_arm_attr = False

                    while j < len(lines) and j < i + 30:
                        fline = lines[j].strip()

                        if bracket_depth > 0:
                            # Still check stability inside multi-line attrs
                            if 'not(target_arch = "arm")' in fline:
                                in_not_arm_attr = True
                                in_arm_only_attr = False
                            elif 'target_arch = "arm"' in fline and 'not(' not in fline:
                                in_arm_only_attr = True
                                in_not_arm_attr = False

                            if 'stable(' in fline and 'unstable(' not in fline:
                                if not in_arm_only_attr:
                                    stability = "stable"
                            elif 'unstable(' in fline:
                                if not in_not_arm_attr and not in_arm_only_attr:
                                    stability = "unstable"

                            bracket_depth += fline.count('[') - fline.count(']')
                            if bracket_depth <= 0:
                                bracket_depth = 0
                                in_arm_only_attr = False
                                in_not_arm_attr = False
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
                            # Enter multi-line attribute tracking
                            if 'cfg_attr' in fline:
                                if 'not(target_arch = "arm")' in fline:
                                    in_not_arm_attr = True
                                elif 'target_arch = "arm"' in fline:
                                    in_arm_only_attr = True
                            # Check stability on single-line attrs too
                            if 'stable(' in fline and 'unstable(' not in fline:
                                if not in_arm_only_attr:
                                    stability = "stable"
                            elif 'unstable(' in fline:
                                if not in_not_arm_attr and not in_arm_only_attr:
                                    stability = "unstable"
                            bracket_depth = fline.count('[') - fline.count(']')
                            if bracket_depth <= 0:
                                bracket_depth = 0
                                in_arm_only_attr = False
                                in_not_arm_attr = False
                            j += 1
                            continue

                        if (fline.startswith('//') or fline.startswith('///') or
                            fline == '' or fline.startswith('*') or
                            fline.startswith('#[doc')):
                            j += 1
                            continue

                        break

                    if fn_name:
                        instruction = extract_instruction(lines, tf_line, j)
                        signature = extract_signature(lines, j)
                        entry = {
                            'arch': arch_name,
                            'name': fn_name,
                            'features': features,
                            'unsafe': is_unsafe,
                            'stability': stability,
                            'file': rel_path,
                            'doc': doc,
                            'signature': signature,
                            'instruction': instruction,
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
                            'doc': src.get('doc', ''),
                            'signature': src.get('signature', '').replace(source_name, alias_name, 1),
                            'instruction': src.get('instruction', ''),
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

    # Merge x86 into x86_64: relabel x86 entries as x86_64, dedup by name
    # (prefer x86_64-specific if both exist)
    x86_64_names = {e['name'] for e in all_intrinsics if e['arch'] == 'x86_64'}
    merged = []
    for entry in all_intrinsics:
        if entry['arch'] == 'x86':
            entry['arch'] = 'x86_64'
            if entry['name'] in x86_64_names:
                continue  # x86_64-specific version takes priority
        merged.append(entry)
    all_intrinsics = merged

    # Write CSV with proper quoting via csv.writer
    writer = csv.writer(sys.stdout, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['arch', 'name', 'features', 'unsafe', 'stability', 'file', 'doc', 'signature', 'instruction'])
    for entry in all_intrinsics:
        writer.writerow([
            entry['arch'],
            entry['name'],
            entry['features'],
            'True' if entry['unsafe'] else 'False',
            entry['stability'],
            entry['file'],
            entry.get('doc', ''),
            entry.get('signature', ''),
            entry.get('instruction', ''),
        ])

    print(f"\nTotal: {len(all_intrinsics)} intrinsics", file=sys.stderr)


if __name__ == '__main__':
    main()

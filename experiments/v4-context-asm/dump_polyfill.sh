#!/usr/bin/env bash
# Dumps the polyfill kernels (group_p) for AVX2, NEON and WASM SIMD128:
#   $OUT/<kernel>_<tier>.s   e.g. p16_gain_neon.s
# Intel syntax on x86. Release profile, no -C target-cpu. Cross targets only
# need `rustup target add`; nothing is linked or run.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${OUT:-$HOME/tmp/handoff/archmage-v4ctx-asm/asm-polyfill}"
mkdir -p "$OUT"
strip() { sed 's/\x1b\[[0-9;]*m//g'; }

dump_target() { # tier, extra cargo-asm args...
  local tier=$1; shift
  local syms
  syms=$(cargo asm --lib "$@" 2>/dev/null | strip || true)
  for k in p4 p8 p16; do
    for body in gain sum; do
      local sym full
      sym="__arcane_${k}_${body}_impl_${tier}"
      full=$(grep -oE "\"[A-Za-z0-9_:]*${sym}\"" <<<"$syms" | tr -d '"' | head -1 || true)
      local note=""
      if [ -z "$full" ]; then
        local entry
        entry=$(grep -oE "\"[A-Za-z0-9_:]*::${k}_${body}_${tier}\"" <<<"$syms" | tr -d '"' | head -1 || true)
        [ -n "$entry" ] || { echo "no symbol for ${k}_${body} ${tier}" >&2; return 1; }
        local body_asm
        body_asm=$(cargo asm --lib "$@" --simplify "$entry" 2>/dev/null | strip)
        if [ "$(grep -cE '^\s+[a-z]' <<<"$body_asm")" -le 6 ]; then
          # The entry only forwards: LLVM merged the kernel with an identical
          # function (x86 `jmp`), or the kernel is a separate function (WASM `call`).
          full=$(grep -oE "^\s*(jmp|b|call)\s+[A-Za-z0-9_:]+" <<<"$body_asm" | awk '{print $2}' | head -1 || true)
          [ -n "$full" ] || { echo "cannot resolve ${entry}" >&2; return 1; }
          note=" (via ${full})"
        else
          # The kernel inlined into its entry (its features are the target baseline).
          full=$entry
          note=" (inlined into its entry)"
        fi
      fi
      cargo asm --lib "$@" --simplify "$full" 2>/dev/null | strip > "$OUT/${k}_${body}_${tier}.s"
      echo "${k}_${body} ${tier} $(wc -l < "$OUT/${k}_${body}_${tier}.s") lines${note}"
    done
  done
}

dump_target v3 --intel
dump_target neon --target aarch64-unknown-linux-gnu
dump_target wasm128 --target wasm32-unknown-unknown --wasm

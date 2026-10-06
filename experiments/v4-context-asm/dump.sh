#!/usr/bin/env bash
# Regenerates every assembly dump: ~/tmp/handoff/archmage-v4ctx-asm/asm/<id>_<tier>.s
# Intel syntax throughout. Release profile, no -C target-cpu.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${OUT:-$HOME/tmp/handoff/archmage-v4ctx-asm/asm}"
mkdir -p "$OUT"
SYMS=$(cargo asm --lib --features avx512 2>/dev/null || true)
SYMS=$(printf "%s" "$SYMS" | sed 's/\x1b\[[0-9;]*m//g')

dump() { # id tier symbol-suffix
  local id=$1 tier=$2 sym=$3
  local full
  full=$(grep -oE "\"v4_context_asm::([a-z_0-9]+::)+${sym}\"" <<<"$SYMS" | tr -d '"' | head -1)
  [ -n "$full" ] || { echo "no symbol for $id $tier ($sym)" >&2; return 1; }
  cargo asm --lib --features avx512 --intel --simplify "$full" 2>/dev/null \
    | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/${id}_${tier}.s"
  echo "$id $tier $full $(wc -l < "$OUT/${id}_${tier}.s") lines"
}

# Group A: magetypes kernels; inner `#[target_feature]` function of each tier
for k in a1 a2 a3 a4 a5 a6 a7 a8_round a8_u8 a9_sum a9_max a10_recip a10_rsqrt a12; do
  for t in v3 v4; do dump "$k" "$t" "__arcane_${k}_impl_${t}"; done
done
dump a8_sat v3 __arcane_a8_sat_impl_v3
dump a8_sat v4_via_v3 a8_sat_via_v3::__arcane_inner
dump a11_exp2 v3 __arcane_a11_exp2_impl_v3
dump a11_exp2 v4_via_v3 a11_exp2_via_v3::__arcane_inner
dump a11_ln v3 __arcane_a11_ln_impl_v3
dump a11_ln v4_via_v3 a11_ln_via_v3::__arcane_inner
# Groups B and C: the nested #[arcane] function inside each entry
for k in b1 b2 c1 c2 c3 c4 c5 c6; do
  for t in v3 v4; do dump "$k" "$t" "${k}_${t}::__arcane_inner"; done
done

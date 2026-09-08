#!/usr/bin/env bash
# Verify that cargo asm output matches documented claims.
# Exit code 1 on mismatch, 0 on success.
#
# Usage: ./scripts/verify-asm.sh [--update]
# --update: Update expected output files instead of comparing

# No `-e`: a failing check must be COUNTED and the run must continue, so the
# summary reports every claim rather than only the first one to break.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXPECTED_DIR="$PROJECT_DIR/tests/expected-asm"
UPDATE_MODE=false

if [[ "${1:-}" == "--update" ]]; then
    UPDATE_MODE=true
fi

mkdir -p "$EXPECTED_DIR"

PASS=0
FAIL=0
UPDATED=0

# Helper: extract just the instruction lines (skip labels, directives, metadata)
# Normalizes jump target labels (.LBB42_1 -> .LABEL) so two functions
# with identical instruction sequences but different label numbering compare equal.
extract_instructions() {
    grep -E '^\s+(v[a-z]|mov|ret|push|pop|lea|add|sub|mul|xor|and|or|cmp|j[a-z]|call|nop|test|cmov)' \
        | sed 's/^[[:space:]]*//' \
        | sed 's/\.LBB[0-9]*_[0-9]*/.LABEL/g' \
        | sed 's/\.Lanon\.[0-9a-f]*\.[0-9]*/.ANON/g' \
        | sed 's/\.LCPI[0-9]*_[0-9]*/.CONST/g' \
        | sort
}

# Get ASM output for a symbol, suppressing build noise
# The benches are split across two packages — `asm_patterns` and
# `safe_memory_overhead` belong to magetypes, `asm_inspection` to archmage — so
# try both rather than hardcoding one and reporting "no output" for the other.
get_asm() {
    local bench="$1"
    local symbol="$2"
    local out
    for pkg in magetypes archmage; do
        out=$(cargo asm -p "$pkg" --bench "$bench" \
            --features "std avx512" \
            "$symbol" 2>/dev/null \
            | grep -v '^\(warning:\|Compiling\|Finished\|Try one\)' \
            | grep -v '^$' || true)
        # A bare list of candidates means the name was ambiguous or absent, not
        # a disassembly — those lines are numbered and quoted.
        if [[ -n "$out" ]] && ! echo "$out" | grep -q '^ *[0-9]* "'; then
            echo "$out"
            return 0
        fi
    done
    echo "$out"
}

# Check a single function's ASM output contains a required instruction
check_contains() {
    local name="$1"
    local bench="$2"
    local symbol="$3"
    local required_instr="$4"
    local expected_file="$EXPECTED_DIR/${name}.asm"

    echo -n "  $name: "

    local asm_output
    asm_output=$(get_asm "$bench" "$symbol")

    if [[ -z "$asm_output" ]]; then
        echo "FAIL (no output)"
        FAIL=$((FAIL + 1))
        return 1
    fi

    # Check required instruction
    if ! echo "$asm_output" | grep -Eq "$required_instr"; then
        echo "FAIL (expected '$required_instr' not found)"
        echo "$asm_output" | head -15 | sed 's/^/    /'
        FAIL=$((FAIL + 1))
        return 1
    fi

    if $UPDATE_MODE; then
        echo "$asm_output" > "$expected_file"
        echo "UPDATED"
        UPDATED=$((UPDATED + 1))
        return 0
    fi

    if [[ -f "$expected_file" ]]; then
        local actual_instrs expected_instrs
        actual_instrs=$(echo "$asm_output" | extract_instructions)
        expected_instrs=$(cat "$expected_file" | extract_instructions)

        if [[ "$actual_instrs" != "$expected_instrs" ]]; then
            echo "FAIL (instructions changed)"
            diff <(echo "$expected_instrs") <(echo "$actual_instrs") | head -15 | sed 's/^/    /'
            echo "    Run: just verify-asm-update"
            FAIL=$((FAIL + 1))
            return 1
        fi
    else
        echo "$asm_output" > "$expected_file"
        echo "OK (baseline created)"
        PASS=$((PASS + 1))
        return 0
    fi

    echo "OK"
    PASS=$((PASS + 1))
}

# Check a function's ASM does NOT contain a forbidden instruction pattern.
# Some claims are about what must not happen — "this did not fall back to a
# lane gather" cannot be written as a required mnemonic, because instruction
# SELECTION among the good options is LLVM's to make and changes between
# versions and optimization levels.
check_absent() {
    local name="$1"
    local bench="$2"
    local symbol="$3"
    local forbidden="$4"
    local why="$5"

    echo -n "  $name: "

    local asm_output
    asm_output=$(get_asm "$bench" "$symbol")

    if [[ -z "$asm_output" ]]; then
        echo "FAIL (no output)"
        FAIL=$((FAIL + 1))
        return 1
    fi

    if echo "$asm_output" | grep -Eq "$forbidden"; then
        echo "FAIL ($why)"
        echo "$asm_output" | grep -E "$forbidden" | head -8 | sed 's/^/    /'
        FAIL=$((FAIL + 1))
        return 1
    fi

    echo "OK"
    PASS=$((PASS + 1))
}

# Check that two functions produce identical instruction sequences
check_identical() {
    local name="$1"
    local bench="$2"
    local symbol_a="$3"
    local symbol_b="$4"
    local label_a="$5"
    local label_b="$6"

    echo -n "  $name: "

    local asm_a asm_b
    asm_a=$(get_asm "$bench" "$symbol_a" | extract_instructions)
    asm_b=$(get_asm "$bench" "$symbol_b" | extract_instructions)

    if [[ -z "$asm_a" || -z "$asm_b" ]]; then
        echo "FAIL (missing output)"
        FAIL=$((FAIL + 1))
        return 1
    fi

    if [[ "$asm_a" != "$asm_b" ]]; then
        echo "FAIL ($label_a != $label_b)"
        diff <(echo "$asm_a") <(echo "$asm_b") | head -10 | sed 's/^/    /'
        FAIL=$((FAIL + 1))
        return 1
    fi

    echo "OK ($label_a == $label_b)"
    PASS=$((PASS + 1))
}

echo "=== ASM Verification ==="
echo ""

# ---- Claim 1: safe and unsafe single loads both produce vmovups ----
echo "Claim: safe_unaligned_simd::_mm256_loadu_ps compiles to vmovups"

check_contains "safe_load_single" "safe_memory_overhead" \
    "safe_memory_overhead::x86_impl::__arcane_safe_load_single" \
    "vmovups"

check_contains "unsafe_load_single" "safe_memory_overhead" \
    "safe_memory_overhead::x86_impl::__arcane_unsafe_load_single" \
    "vmovups"

check_identical "safe_vs_unsafe_load" "safe_memory_overhead" \
    "safe_memory_overhead::x86_impl::__arcane_safe_load_single" \
    "safe_memory_overhead::x86_impl::__arcane_unsafe_load_single" \
    "safe" "unsafe"

echo ""

# ---- Claim 2: #[rite] in #[arcane] matches manual inline (loop bodies) ----
echo "Claim: #[rite] in #[arcane] produces identical ASM to manual inline"

check_identical "rite_vs_manual_loop" "asm_inspection" \
    "asm_inspection::x86_impl::__arcane_loop_inner_rite" \
    "asm_inspection::x86_impl::__arcane_loop_manual_inline" \
    "rite" "manual"

echo ""

# ---- Claim 3: .first_chunk() produces vmovups (same as array ref) ----
echo "Claim: .first_chunk() produces vmovups (256-bit float load)"

check_contains "first_chunk_load" "asm_patterns" \
    "asm_patterns::x86_impl::__arcane_load_first_chunk_256" \
    "vmovups"

check_contains "array_ref_load" "asm_patterns" \
    "asm_patterns::x86_impl::__arcane_load_array_ref" \
    "vmovups"

echo ""

# ---- Claim 4: try_into produces vmovups ----
echo "Claim: .try_into() produces vmovups (256-bit float load)"

check_contains "try_into_load" "asm_patterns" \
    "asm_patterns::x86_impl::__arcane_load_try_into" \
    "vmovups"

echo ""

# ---- Claim 5: integer first_chunk → vmovups/vmovdqu (both valid for unaligned int loads) ----
echo "Claim: integer .first_chunk() produces vmovups or vmovdqu"

check_contains "first_chunk_int_load" "asm_patterns" \
    "asm_patterns::x86_impl::__arcane_load_first_chunk_i" \
    "vmov"

echo ""

# ---- Claim 6: 128-bit first_chunk → vmovups ----
echo "Claim: 128-bit .first_chunk() produces vmovups"

check_contains "first_chunk_128_load" "asm_patterns" \
    "asm_patterns::x86_impl::__arcane_load_first_chunk_128" \
    "vmovups"

echo ""

# ---- Claim 7: store via first_chunk_mut → vmovups ----
echo "Claim: store via .first_chunk_mut() produces vmovups"

check_contains "first_chunk_mut_store" "asm_patterns" \
    "asm_patterns::x86_impl::__arcane_store_first_chunk_mut" \
    "vmovups"
# ---- Claim: concat_shift reaches a cross-lane permute, not a lane gather ----
# The backend trait's default body is a lane gather, and LLVM does NOT recover a
# funnel shift from it (measured 2026-09-08: 6-7 scalar element moves against
# 1-2 native instructions). Every ISA that has the instruction overrides the
# default. If an override is deleted, or a delegation stops forwarding it, the
# result stays CORRECT and silently falls back to the gather — this is the only
# check that would notice.
#
# The claim is deliberately NOT a specific mnemonic. LLVM picks freely among the
# good encodings and the choice moves with optimization level: the same
# f32x16 shift lowers to `valignd` under fat LTO, `vpermi2ps` in this bench, and
# `vpermt2ps` with a hoisted mask register inside zensr's real loop — all of
# them one instruction per shift. What must never appear is per-lane scalar
# movement.
echo ""
echo "Claim: concat_shift lowers to a cross-lane permute, never a lane gather"

check_absent "concat_shift_f32x8_v3_no_gather" "asm_patterns" \
    "asm_patterns::x86_impl::__arcane_concat_shift_f32x8_v3" \
    "vmovs[sd]" \
    "fell back to the scalar lane gather"

check_absent "concat_shift_f32x16_v4x_no_gather" "asm_patterns" \
    "asm_patterns::x86_impl::__arcane_concat_shift_f32x16_v4x" \
    "vmovs[sd]" \
    "fell back to the scalar lane gather"

check_contains "concat_shift_f32x8_v3" "asm_patterns" \
    "asm_patterns::x86_impl::__arcane_concat_shift_f32x8_v3" \
    "valign|palignr|vperm|vshuf"

check_contains "concat_shift_f32x16_v4x" "asm_patterns" \
    "asm_patterns::x86_impl::__arcane_concat_shift_f32x16_v4x" \
    "valign|palignr|vperm|vshuf"

echo ""

# ---- Summary ----
echo "=== Results ==="
echo "  Passed:  $PASS"
if $UPDATE_MODE; then
    echo "  Updated: $UPDATED"
fi
echo "  Failed:  $FAIL"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi

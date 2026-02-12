#!/usr/bin/env bash
# Verify that cargo asm output matches documented claims.
# Exit code 1 on mismatch, 0 on success.
#
# Usage: ./scripts/verify-asm.sh [--update]
# --update: Update expected output files instead of comparing

set -euo pipefail

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
        | sort
}

# Get ASM output for a symbol, suppressing build noise
get_asm() {
    local bench="$1"
    local symbol="$2"
    cargo asm -p archmage --bench "$bench" \
        --features "std macros avx512" \
        "$symbol" 2>/dev/null \
        | grep -v '^\(warning:\|Compiling\|Finished\|Try one\)' \
        | grep -v '^$' || true
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
    if ! echo "$asm_output" | grep -q "$required_instr"; then
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
    "safe_memory_overhead::safe_load_single::__simd_inner_safe_load_single" \
    "vmovups"

check_contains "unsafe_load_single" "safe_memory_overhead" \
    "safe_memory_overhead::unsafe_load_single::__simd_inner_unsafe_load_single" \
    "vmovups"

check_identical "safe_vs_unsafe_load" "safe_memory_overhead" \
    "safe_memory_overhead::safe_load_single::__simd_inner_safe_load_single" \
    "safe_memory_overhead::unsafe_load_single::__simd_inner_unsafe_load_single" \
    "safe" "unsafe"

echo ""

# ---- Claim 2: #[rite] in #[arcane] matches manual inline (loop bodies) ----
echo "Claim: #[rite] in #[arcane] produces identical ASM to manual inline"

check_identical "rite_vs_manual_loop" "asm_inspection" \
    "asm_inspection::loop_inner_rite::__simd_inner_loop_inner_rite" \
    "asm_inspection::loop_manual_inline::__simd_inner_loop_manual_inline" \
    "rite" "manual"

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

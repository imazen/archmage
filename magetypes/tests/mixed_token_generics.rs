//! Migration patterns extracted from the consumer audit, not complete ports.
#![forbid(unsafe_code)]

use archmage::{ScalarToken, incant, magetypes, rite};
use magetypes::simd::{
    backends::I32x8Backend,
    generic::{ConstructorMode, core_types::i32x8 as Vector, local},
};

// The zenav1 clamp primitive, generalized over constructor mode. A borrowed
// buffer can keep its mode throughout the call chain; no slice cast is needed.
#[inline(always)]
fn clampv<T: I32x8Backend, M: ConstructorMode>(t: T, v: Vector<T, M>, bit: i8) -> Vector<T, M> {
    if bit <= 0 || bit >= 32 {
        return v;
    }
    let hi = ((1i64 << (bit - 1)) - 1) as i32;
    let lo = (-(1i64 << (bit - 1))) as i32;
    v.clamp(
        Vector::splat_with_token(t, lo),
        Vector::splat_with_token(t, hi),
    )
}

fn clamp_buffer<T: I32x8Backend, M: ConstructorMode>(t: T, values: &mut [Vector<T, M>], bit: i8) {
    for v in values {
        *v = clampv(t, *v, bit);
    }
}

#[magetypes(use(i32x8), v3, neon, wasm128, scalar)]
fn modes(token: Token, input: &[i32; 8], bit: i8) -> ([i32; 8], [i32; 8]) {
    let mut explicit = [magetypes::simd::generic::i32x8::load(token, input)];
    let mut context = [i32x8::load(input)];
    clamp_buffer(token, &mut explicit, bit);
    clamp_buffer(token, &mut context, bit);
    (explicit[0].to_array(), context[0].to_array())
}

// Unlike use(...) aliases, this explicit signature spelling is available
// outside the body. magetypes substitutes Token in the signature as well.
#[magetypes(rite, use(i32x8), v3, neon, wasm128, scalar)]
fn vector_signature(values: &mut [local::i32x8<Token>]) {
    for v in values {
        *v += i32x8::splat(1);
    }
}

// Models ChunkInput's independent pixel generic plus token-selected loader.
trait ChunkInput: Copy {
    fn load<T: I32x8Backend>(input: &[Self; 8], token: T) -> Vector<T>;
}

impl ChunkInput for u8 {
    fn load<T: I32x8Backend>(input: &[u8; 8], token: T) -> Vector<T> {
        Vector::from_array_with_token(token, input.map(i32::from))
    }
}

// Keeps its token for the generic trait helper while using short constructors.
#[magetypes(rite, use(i32x8), v3, neon, wasm128, scalar)]
fn legacy<const ADD: bool, R: ChunkInput>(token: Token, input: &[R; 8]) -> [i32; 8] {
    let loaded: i32x8 = R::load(input, token).into();
    let mut values = [loaded];
    incant!(vector_signature(&mut values) without token);
    if ADD {
        (values[0] + i32x8::splat(2)).to_array()
    } else {
        values[0].to_array()
    }
}

// Tokenless -> tokenful: rite supplies the covered tier's from_context().
#[rite(v3, neon, wasm128, scalar)]
fn tokenless<const ADD: bool, R: ChunkInput>(input: &[R; 8]) -> [i32; 8] {
    incant!(legacy::<ADD, R>(input), [v3, neon, wasm128, scalar])
}

#[magetypes(rite, v3, neon, wasm128, scalar)]
fn tokenless_magetypes<const ADD: bool, R: ChunkInput>(input: &[R; 8]) -> [i32; 8] {
    incant!(legacy::<ADD, R>(input), [v3, neon, wasm128, scalar])
}

// Tokenful -> tokenless: exact same-tier suffix, no token argument.
#[magetypes(v3, neon, wasm128, scalar)]
fn entry<const ADD: bool, R: ChunkInput>(_token: Token, input: &[R; 8]) -> [i32; 8] {
    let direct = incant!(tokenless::<ADD, R>(input) without token);
    let generated = incant!(tokenless_magetypes::<ADD, R>(input) without token);
    assert_eq!(direct, generated);
    generated
}

#[test]
fn mixed_calls_preserve_type_and_const_generics() {
    let input = [0u8, 1, 2, 3, 100, 127, 254, 255];
    let expected = input.map(|v| i32::from(v) + 3);
    assert_eq!(entry_scalar::<true, _>(ScalarToken, &input), expected);
    assert_eq!(
        incant!(entry::<true, u8>(&input), [v3, neon, wasm128, scalar]),
        expected
    );
    assert_eq!(
        incant!(entry::<false, u8>(&input), [v3, neon, wasm128, scalar]),
        input.map(|v| i32::from(v) + 1)
    );
}

#[test]
fn backend_generic_borrowed_buffers_accept_both_modes() {
    let input = [i32::MIN, -129, -128, -1, 0, 127, 128, i32::MAX];
    for bit in [-1, 0, 1, 8, 31, 32, 33] {
        let expected = input.map(|v| {
            if bit <= 0 || bit >= 32 {
                v
            } else {
                v.clamp(-(1i32 << (bit - 1)), (1i32 << (bit - 1)) - 1)
            }
        });
        assert_eq!(modes_scalar(ScalarToken, &input, bit), (expected, expected));
        assert_eq!(
            incant!(modes(&input, bit), [v3, neon, wasm128, scalar]),
            (expected, expected)
        );
    }
}

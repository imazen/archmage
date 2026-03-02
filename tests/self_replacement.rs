//! Comprehensive tests for `_self` parameter transformation and `Self` replacement.
//!
//! The `#[arcane(_self = Type)]` macro transforms self receivers into `_self: Type` parameters
//! in the generated inner function, and replaces all `Self` identifier tokens with the
//! concrete type. These tests cover every position where `Self` can appear.

#![allow(unused, clippy::unnecessary_wraps, clippy::needless_return)]

#[cfg(target_arch = "x86_64")]
mod self_replacement_tests {
    use archmage::{Desktop64, SimdToken, X64V3Token, arcane};

    // =========================================================================
    // Test types — plain structs, no SIMD needed for Self replacement testing
    // =========================================================================

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Point {
        x: f32,
        y: f32,
    }

    impl Point {
        fn new(x: f32, y: f32) -> Self {
            Self { x, y }
        }

        fn origin() -> Self {
            Self { x: 0.0, y: 0.0 }
        }

        fn distance(&self, other: &Self) -> f32 {
            let dx = self.x - other.x;
            let dy = self.y - other.y;
            (dx * dx + dy * dy).sqrt()
        }

        fn length(&self) -> f32 {
            (self.x * self.x + self.y * self.y).sqrt()
        }

        fn scale(&self, factor: f32) -> Self {
            Self {
                x: self.x * factor,
                y: self.y * factor,
            }
        }
    }

    // =========================================================================
    // 1. Self in return type → replaced with concrete type
    // =========================================================================

    trait ReturnSelf {
        fn identity(&self, token: X64V3Token) -> Self;
    }

    impl ReturnSelf for Point {
        #[arcane(_self = Point)]
        fn identity(&self, _token: X64V3Token) -> Self {
            // Simple: return type is -> Self, body returns a copy
            Point {
                x: _self.x,
                y: _self.y,
            }
        }
    }

    #[test]
    fn test_self_in_return_type() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(3.0, 4.0);
            let result = p.identity(token);
            assert_eq!(result, p);
        }
    }

    // =========================================================================
    // 2. Self in struct literal construction (Self { field: value })
    // =========================================================================

    trait ConstructSelf {
        fn doubled(&self, token: X64V3Token) -> Self;
    }

    impl ConstructSelf for Point {
        #[arcane(_self = Point)]
        fn doubled(&self, _token: X64V3Token) -> Self {
            Self {
                x: _self.x * 2.0,
                y: _self.y * 2.0,
            }
        }
    }

    #[test]
    fn test_self_struct_literal() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(1.5, 2.5);
            let result = p.doubled(token);
            assert_eq!(result, Point::new(3.0, 5.0));
        }
    }

    // =========================================================================
    // 3. Self in associated function calls (Self::new(), Self::origin())
    // =========================================================================

    trait AssocFnSelf {
        fn to_origin(&self, token: X64V3Token) -> Self;
        fn scaled_copy(&self, token: X64V3Token, factor: f32) -> Self;
    }

    impl AssocFnSelf for Point {
        #[arcane(_self = Point)]
        fn to_origin(&self, _token: X64V3Token) -> Self {
            // Self::origin() should become Point::origin()
            Self::origin()
        }

        #[arcane(_self = Point)]
        fn scaled_copy(&self, _token: X64V3Token, factor: f32) -> Self {
            // Self::new() should become Point::new()
            Self::new(_self.x * factor, _self.y * factor)
        }
    }

    #[test]
    fn test_self_associated_fn_calls() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(10.0, 20.0);
            let origin = p.to_origin(token);
            assert_eq!(origin, Point::origin());

            let scaled = p.scaled_copy(token, 0.5);
            assert_eq!(scaled, Point::new(5.0, 10.0));
        }
    }

    // =========================================================================
    // 4. Self in type annotation positions within the body
    // =========================================================================

    trait TypeAnnotationSelf {
        fn with_type_annotations(&self, token: X64V3Token) -> Self;
    }

    impl TypeAnnotationSelf for Point {
        #[arcane(_self = Point)]
        fn with_type_annotations(&self, _token: X64V3Token) -> Self {
            // Self in let binding type annotation
            let copy: Self = Point {
                x: _self.x,
                y: _self.y,
            };
            copy
        }
    }

    #[test]
    fn test_self_type_annotations() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(7.0, 8.0);
            let result = p.with_type_annotations(token);
            assert_eq!(result, p);
        }
    }

    // =========================================================================
    // 5. Self in generic type parameters (Vec<Self>, Option<Self>)
    // =========================================================================

    trait GenericSelf {
        fn as_option(&self, token: X64V3Token) -> Option<Self>
        where
            Self: Sized;
        fn as_vec(&self, token: X64V3Token, count: usize) -> Vec<Self>
        where
            Self: Sized;
    }

    impl GenericSelf for Point {
        #[arcane(_self = Point)]
        fn as_option(&self, _token: X64V3Token) -> Option<Self> {
            // Self inside Option<Self>
            Some(Self {
                x: _self.x,
                y: _self.y,
            })
        }

        #[arcane(_self = Point)]
        fn as_vec(&self, _token: X64V3Token, count: usize) -> Vec<Self> {
            // Self inside Vec<Self>, plus Self::new in a loop
            let mut result: Vec<Self> = Vec::with_capacity(count);
            for i in 0..count {
                result.push(Self::new(_self.x + i as f32, _self.y + i as f32));
            }
            result
        }
    }

    #[test]
    fn test_self_in_generic_types() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(1.0, 2.0);

            let opt = p.as_option(token);
            assert_eq!(opt, Some(p));

            let v = p.as_vec(token, 3);
            assert_eq!(v.len(), 3);
            assert_eq!(v[0], Point::new(1.0, 2.0));
            assert_eq!(v[1], Point::new(2.0, 3.0));
            assert_eq!(v[2], Point::new(3.0, 4.0));
        }
    }

    // =========================================================================
    // 6. Self in tuple struct construction (Self(value))
    // =========================================================================

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Wrapper(f32);

    impl Wrapper {
        fn value(&self) -> f32 {
            self.0
        }
    }

    trait TupleStructSelf {
        fn doubled(&self, token: X64V3Token) -> Self;
        fn consume_and_create(self, token: X64V3Token, val: f32) -> Self;
    }

    impl TupleStructSelf for Wrapper {
        #[arcane(_self = Wrapper)]
        fn doubled(&self, _token: X64V3Token) -> Self {
            // Self(value) tuple struct construction
            Self(_self.0 * 2.0)
        }

        #[arcane(_self = Wrapper)]
        fn consume_and_create(self, _token: X64V3Token, val: f32) -> Self {
            // Owned self + Self() tuple struct construction
            Self(val + _self.0)
        }
    }

    #[test]
    fn test_self_tuple_struct() {
        if let Some(token) = X64V3Token::summon() {
            let w = Wrapper(5.0);
            let doubled = w.doubled(token);
            assert_eq!(doubled.value(), 10.0);

            let w2 = Wrapper(3.0);
            let new_w = w2.consume_and_create(token, 42.0);
            assert_eq!(new_w.value(), 45.0);
        }
    }

    // =========================================================================
    // 7. Multiple Self references in one function
    // =========================================================================

    trait MultipleSelf {
        fn complex_transform(&self, token: X64V3Token, other: &Self) -> Self;
    }

    impl MultipleSelf for Point {
        #[arcane(_self = Point)]
        fn complex_transform(&self, _token: X64V3Token, other: &Self) -> Self {
            // Self appears in: parameter type, return type, struct literal, type annotation
            let midpoint: Self = Self::new((_self.x + other.x) / 2.0, (_self.y + other.y) / 2.0);
            midpoint
        }
    }

    #[test]
    fn test_multiple_self_references() {
        if let Some(token) = X64V3Token::summon() {
            let a = Point::new(0.0, 0.0);
            let b = Point::new(10.0, 20.0);
            let mid = a.complex_transform(token, &b);
            assert_eq!(mid, Point::new(5.0, 10.0));
        }
    }

    // =========================================================================
    // 8. _self field access patterns
    // =========================================================================

    trait FieldAccess {
        fn get_x(&self, token: X64V3Token) -> f32;
        fn get_y(&self, token: X64V3Token) -> f32;
        fn swap_xy(&self, token: X64V3Token) -> Self;
    }

    impl FieldAccess for Point {
        #[arcane(_self = Point)]
        fn get_x(&self, _token: X64V3Token) -> f32 {
            _self.x
        }

        #[arcane(_self = Point)]
        fn get_y(&self, _token: X64V3Token) -> f32 {
            _self.y
        }

        #[arcane(_self = Point)]
        fn swap_xy(&self, _token: X64V3Token) -> Self {
            Self {
                x: _self.y,
                y: _self.x,
            }
        }
    }

    #[test]
    fn test_self_field_access() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(3.0, 7.0);
            assert_eq!(p.get_x(token), 3.0);
            assert_eq!(p.get_y(token), 7.0);
            assert_eq!(p.swap_xy(token), Point::new(7.0, 3.0));
        }
    }

    // =========================================================================
    // 9. _self method calls
    // =========================================================================

    trait MethodCalls {
        fn compute_length(&self, token: X64V3Token) -> f32;
        fn distance_to_origin(&self, token: X64V3Token) -> f32;
    }

    impl MethodCalls for Point {
        #[arcane(_self = Point)]
        fn compute_length(&self, _token: X64V3Token) -> f32 {
            // Call a method on _self
            _self.length()
        }

        #[arcane(_self = Point)]
        fn distance_to_origin(&self, _token: X64V3Token) -> f32 {
            // Call a method on _self that takes &Self parameter
            let origin = Point::origin();
            _self.distance(&origin)
        }
    }

    #[test]
    fn test_self_method_calls() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(3.0, 4.0);
            assert_eq!(p.compute_length(token), 5.0);
            assert_eq!(p.distance_to_origin(token), 5.0);
        }
    }

    // =========================================================================
    // 10. &mut self receiver with mutation via _self
    // =========================================================================

    trait MutableSelf {
        fn translate(&mut self, token: X64V3Token, dx: f32, dy: f32);
        fn reset_to_origin(&mut self, token: X64V3Token);
        fn scale_in_place(&mut self, token: X64V3Token, factor: f32);
    }

    impl MutableSelf for Point {
        #[arcane(_self = Point)]
        fn translate(&mut self, _token: X64V3Token, dx: f32, dy: f32) {
            _self.x += dx;
            _self.y += dy;
        }

        #[arcane(_self = Point)]
        fn reset_to_origin(&mut self, _token: X64V3Token) {
            // Assign via struct literal with Self
            *_self = Self::origin();
        }

        #[arcane(_self = Point)]
        fn scale_in_place(&mut self, _token: X64V3Token, factor: f32) {
            _self.x *= factor;
            _self.y *= factor;
        }
    }

    #[test]
    fn test_mut_self_mutation() {
        if let Some(token) = X64V3Token::summon() {
            let mut p = Point::new(1.0, 2.0);
            p.translate(token, 3.0, 4.0);
            assert_eq!(p, Point::new(4.0, 6.0));

            p.scale_in_place(token, 2.0);
            assert_eq!(p, Point::new(8.0, 12.0));

            p.reset_to_origin(token);
            assert_eq!(p, Point::origin());
        }
    }

    // =========================================================================
    // 11. Owned self receiver (move semantics)
    // =========================================================================

    trait OwnedSelf {
        fn into_negated(self, token: X64V3Token) -> Self;
        fn into_scaled(self, token: X64V3Token, factor: f32) -> Self;
    }

    impl OwnedSelf for Point {
        #[arcane(_self = Point)]
        fn into_negated(self, _token: X64V3Token) -> Self {
            Self::new(-_self.x, -_self.y)
        }

        #[arcane(_self = Point)]
        fn into_scaled(self, _token: X64V3Token, factor: f32) -> Self {
            Self {
                x: _self.x * factor,
                y: _self.y * factor,
            }
        }
    }

    #[test]
    fn test_owned_self() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(3.0, 4.0);
            let neg = p.into_negated(token);
            assert_eq!(neg, Point::new(-3.0, -4.0));

            let p2 = Point::new(1.0, 2.0);
            let scaled = p2.into_scaled(token, 10.0);
            assert_eq!(scaled, Point::new(10.0, 20.0));
        }
    }

    // =========================================================================
    // 12. Self in closures within the function body
    // =========================================================================

    trait ClosureSelf {
        fn map_coords(&self, token: X64V3Token, f: fn(f32) -> f32) -> Self;
    }

    impl ClosureSelf for Point {
        #[arcane(_self = Point)]
        fn map_coords(&self, _token: X64V3Token, f: fn(f32) -> f32) -> Self {
            // Self used inside a closure-like context
            let transform = |p: &Point| -> Self { Self::new(f(p.x), f(p.y)) };
            transform(_self)
        }
    }

    #[test]
    fn test_self_in_closure() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(4.0, 9.0);
            let result = p.map_coords(token, |v| v.sqrt());
            assert_eq!(result, Point::new(2.0, 3.0));
        }
    }

    // =========================================================================
    // 13. Self in match expressions
    // =========================================================================

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Shape {
        Circle(f32),
        Rectangle(f32, f32),
    }

    impl Shape {
        fn area(&self) -> f32 {
            match self {
                Shape::Circle(r) => std::f32::consts::PI * r * r,
                Shape::Rectangle(w, h) => w * h,
            }
        }
    }

    trait MatchSelf {
        fn scale_shape(&self, token: X64V3Token, factor: f32) -> Self;
    }

    impl MatchSelf for Shape {
        #[arcane(_self = Shape)]
        fn scale_shape(&self, _token: X64V3Token, factor: f32) -> Self {
            // Self in match arms as enum variant constructors
            match _self {
                Self::Circle(r) => Self::Circle(r * factor),
                Self::Rectangle(w, h) => Self::Rectangle(w * factor, h * factor),
            }
        }
    }

    #[test]
    fn test_self_in_match() {
        if let Some(token) = X64V3Token::summon() {
            let circle = Shape::Circle(5.0);
            let scaled = circle.scale_shape(token, 2.0);
            assert_eq!(scaled, Shape::Circle(10.0));

            let rect = Shape::Rectangle(3.0, 4.0);
            let scaled = rect.scale_shape(token, 3.0);
            assert_eq!(scaled, Shape::Rectangle(9.0, 12.0));
        }
    }

    // =========================================================================
    // 14. Self in if/else expressions
    // =========================================================================

    trait ConditionalSelf {
        fn clamp_to_unit(&self, token: X64V3Token) -> Self;
    }

    impl ConditionalSelf for Point {
        #[arcane(_self = Point)]
        fn clamp_to_unit(&self, _token: X64V3Token) -> Self {
            let len = _self.length();
            if len > 1.0 {
                // Self::new in then-branch
                Self::new(_self.x / len, _self.y / len)
            } else {
                // Self struct literal in else-branch
                Self {
                    x: _self.x,
                    y: _self.y,
                }
            }
        }
    }

    #[test]
    fn test_self_in_conditionals() {
        if let Some(token) = X64V3Token::summon() {
            // Inside unit circle — should be unchanged
            let p = Point::new(0.3, 0.4);
            let result = p.clamp_to_unit(token);
            assert_eq!(result, p);

            // Outside unit circle — should be normalized
            let p = Point::new(3.0, 4.0);
            let result = p.clamp_to_unit(token);
            assert!((result.length() - 1.0).abs() < 1e-6);
            assert!((result.x - 0.6).abs() < 1e-6);
            assert!((result.y - 0.8).abs() < 1e-6);
        }
    }

    // =========================================================================
    // 15. Self in nested type positions (Box<Self>, Result<Self, E>)
    // =========================================================================

    trait NestedTypeSelf {
        fn try_negate(&self, token: X64V3Token) -> Result<Self, &'static str>
        where
            Self: Sized;
    }

    impl NestedTypeSelf for Point {
        #[arcane(_self = Point)]
        fn try_negate(&self, _token: X64V3Token) -> Result<Self, &'static str> {
            // Self inside Result<Self, E>
            if _self.x.is_nan() || _self.y.is_nan() {
                Err("NaN coordinates")
            } else {
                Ok(Self::new(-_self.x, -_self.y))
            }
        }
    }

    #[test]
    fn test_self_in_nested_types() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(1.0, 2.0);
            let result = p.try_negate(token);
            assert_eq!(result, Ok(Point::new(-1.0, -2.0)));

            let nan_p = Point::new(f32::NAN, 0.0);
            let result = nan_p.try_negate(token);
            assert_eq!(result, Err("NaN coordinates"));
        }
    }

    // =========================================================================
    // 16. Self with associated constants (not directly supported since const
    //     can't use Self in non-trait-const positions, but test the pattern)
    // =========================================================================

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Color {
        r: f32,
        g: f32,
        b: f32,
    }

    impl Color {
        const BLACK: Self = Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
        };
        const WHITE: Self = Self {
            r: 1.0,
            g: 1.0,
            b: 1.0,
        };

        fn new(r: f32, g: f32, b: f32) -> Self {
            Self { r, g, b }
        }
    }

    trait AssocConstSelf {
        fn invert(&self, token: X64V3Token) -> Self;
        fn is_dark(&self, token: X64V3Token) -> bool;
    }

    impl AssocConstSelf for Color {
        #[arcane(_self = Color)]
        fn invert(&self, _token: X64V3Token) -> Self {
            Self::new(1.0 - _self.r, 1.0 - _self.g, 1.0 - _self.b)
        }

        #[arcane(_self = Color)]
        fn is_dark(&self, _token: X64V3Token) -> bool {
            let luminance = _self.r * 0.299 + _self.g * 0.587 + _self.b * 0.114;
            luminance < 0.5
        }
    }

    #[test]
    fn test_self_with_assoc_constants() {
        if let Some(token) = X64V3Token::summon() {
            let black = Color::BLACK;
            let inverted = black.invert(token);
            assert_eq!(inverted, Color::WHITE);

            assert!(black.is_dark(token));
            assert!(!Color::WHITE.is_dark(token));
        }
    }

    // =========================================================================
    // 17. Self in array/slice contexts
    // =========================================================================

    trait ArraySelf {
        fn replicate(&self, token: X64V3Token) -> [Self; 4]
        where
            Self: Sized;
    }

    impl ArraySelf for Point {
        #[arcane(_self = Point)]
        fn replicate(&self, _token: X64V3Token) -> [Self; 4] {
            // Self in array type: [Self; 4]
            [
                Self::new(_self.x, _self.y),
                Self::new(_self.x + 1.0, _self.y),
                Self::new(_self.x, _self.y + 1.0),
                Self::new(_self.x + 1.0, _self.y + 1.0),
            ]
        }
    }

    #[test]
    fn test_self_in_array_type() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(0.0, 0.0);
            let arr = p.replicate(token);
            assert_eq!(arr[0], Point::new(0.0, 0.0));
            assert_eq!(arr[1], Point::new(1.0, 0.0));
            assert_eq!(arr[2], Point::new(0.0, 1.0));
            assert_eq!(arr[3], Point::new(1.0, 1.0));
        }
    }

    // =========================================================================
    // 18. Multiple trait impls on same type (verify isolation)
    // =========================================================================

    trait TraitA {
        fn method_a(&self, token: X64V3Token) -> Self;
    }

    trait TraitB {
        fn method_b(&self, token: X64V3Token) -> f32;
    }

    impl TraitA for Point {
        #[arcane(_self = Point)]
        fn method_a(&self, _token: X64V3Token) -> Self {
            Self::new(_self.x + 1.0, _self.y + 1.0)
        }
    }

    impl TraitB for Point {
        #[arcane(_self = Point)]
        fn method_b(&self, _token: X64V3Token) -> f32 {
            _self.x * _self.y
        }
    }

    #[test]
    fn test_multiple_trait_impls() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(3.0, 4.0);
            let a_result = p.method_a(token);
            assert_eq!(a_result, Point::new(4.0, 5.0));

            let b_result = p.method_b(token);
            assert_eq!(b_result, 12.0);
        }
    }

    // =========================================================================
    // 19. Self in deeply nested expressions
    // =========================================================================

    trait DeepNesting {
        fn deep_transform(&self, token: X64V3Token) -> Self;
    }

    impl DeepNesting for Point {
        #[arcane(_self = Point)]
        fn deep_transform(&self, _token: X64V3Token) -> Self {
            // Self nested inside: if → block → method call → struct literal
            let result = {
                let scaled = if _self.length() > 0.0 {
                    let normalized = Self::new(_self.x / _self.length(), _self.y / _self.length());
                    normalized
                } else {
                    Self::origin()
                };
                scaled
            };
            result
        }
    }

    #[test]
    fn test_deeply_nested_self() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(3.0, 4.0);
            let result = p.deep_transform(token);
            assert!((result.x - 0.6).abs() < 1e-6);
            assert!((result.y - 0.8).abs() < 1e-6);

            let zero = Point::origin();
            let result = zero.deep_transform(token);
            assert_eq!(result, Point::origin());
        }
    }

    // =========================================================================
    // 20. &_self borrow patterns
    // =========================================================================

    trait BorrowSelf {
        fn measure_distance(&self, token: X64V3Token, target: &Point) -> f32;
    }

    impl BorrowSelf for Point {
        #[arcane(_self = Point)]
        fn measure_distance(&self, _token: X64V3Token, target: &Point) -> f32 {
            // Borrow _self to pass to a function expecting &Point
            _self.distance(target)
        }
    }

    #[test]
    fn test_borrow_self() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(0.0, 0.0);
            let target = Point::new(3.0, 4.0);
            let dist = p.measure_distance(token, &target);
            assert_eq!(dist, 5.0);
        }
    }

    // =========================================================================
    // 21. Self with enum variants in more complex patterns
    // =========================================================================

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Expr {
        Const(f32),
        Add(f32, f32),
        Mul(f32, f32),
    }

    impl Expr {
        fn eval(&self) -> f32 {
            match self {
                Expr::Const(v) => *v,
                Expr::Add(a, b) => a + b,
                Expr::Mul(a, b) => a * b,
            }
        }
    }

    trait ExprOps {
        fn simplify(&self, token: X64V3Token) -> Self;
    }

    impl ExprOps for Expr {
        #[arcane(_self = Expr)]
        fn simplify(&self, _token: X64V3Token) -> Self {
            match _self {
                Self::Add(a, b) if *a == 0.0 => Self::Const(*b),
                Self::Add(a, b) if *b == 0.0 => Self::Const(*a),
                Self::Mul(_, b) if *b == 0.0 => Self::Const(0.0),
                Self::Mul(a, _) if *a == 0.0 => Self::Const(0.0),
                Self::Mul(a, b) if *a == 1.0 => Self::Const(*b),
                Self::Mul(a, b) if *b == 1.0 => Self::Const(*a),
                other => Self::Const(other.eval()),
            }
        }
    }

    #[test]
    fn test_self_enum_variants_complex() {
        if let Some(token) = X64V3Token::summon() {
            let e = Expr::Add(0.0, 5.0);
            assert_eq!(e.simplify(token), Expr::Const(5.0));

            let e = Expr::Mul(1.0, 7.0);
            assert_eq!(e.simplify(token), Expr::Const(7.0));

            let e = Expr::Mul(3.0, 0.0);
            assert_eq!(e.simplify(token), Expr::Const(0.0));

            let e = Expr::Add(2.0, 3.0);
            assert_eq!(e.simplify(token), Expr::Const(5.0));
        }
    }

    // =========================================================================
    // 22. Self in where clauses and trait bounds (function body)
    // =========================================================================

    trait CollectSelf {
        fn collect_into(&self, token: X64V3Token, dest: &mut Vec<Self>)
        where
            Self: Sized + Clone;
    }

    impl CollectSelf for Point {
        #[arcane(_self = Point)]
        fn collect_into(&self, _token: X64V3Token, dest: &mut Vec<Self>)
        where
            Self: Sized + Clone,
        {
            // Vec<Self> in parameter, Self in where clause
            dest.push(_self.clone());
            dest.push(Self::new(_self.x * 2.0, _self.y * 2.0));
        }
    }

    #[test]
    fn test_self_in_where_clause_context() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(1.0, 2.0);
            let mut dest = Vec::new();
            p.collect_into(token, &mut dest);
            assert_eq!(dest.len(), 2);
            assert_eq!(dest[0], Point::new(1.0, 2.0));
            assert_eq!(dest[1], Point::new(2.0, 4.0));
        }
    }

    // =========================================================================
    // 23. Self with impl blocks that have generic constraints
    // =========================================================================

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Pair<T: Copy> {
        first: T,
        second: T,
    }

    impl<T: Copy + Default> Pair<T> {
        fn new(first: T, second: T) -> Self {
            Self { first, second }
        }
    }

    trait PairOps {
        fn swap(&self, token: X64V3Token) -> Self;
    }

    impl PairOps for Pair<f32> {
        #[arcane(_self = Pair::<f32>)]
        fn swap(&self, _token: X64V3Token) -> Self {
            // Self with generic type: Pair<f32> (use turbofish in attribute)
            Self::new(_self.second, _self.first)
        }
    }

    #[test]
    fn test_self_with_generic_type() {
        if let Some(token) = X64V3Token::summon() {
            let p = Pair::new(1.0f32, 2.0f32);
            let swapped = p.swap(token);
            assert_eq!(swapped.first, 2.0);
            assert_eq!(swapped.second, 1.0);
        }
    }

    // =========================================================================
    // 24. Desktop64 alias works with _self
    // =========================================================================

    trait Desktop64Self {
        fn negate(&self, token: Desktop64) -> Self;
    }

    impl Desktop64Self for Point {
        #[arcane(_self = Point)]
        fn negate(&self, _token: Desktop64) -> Self {
            Self::new(-_self.x, -_self.y)
        }
    }

    #[test]
    fn test_self_with_desktop64_alias() {
        if let Some(token) = Desktop64::summon() {
            let p = Point::new(3.0, 4.0);
            let neg = p.negate(token);
            assert_eq!(neg, Point::new(-3.0, -4.0));
        }
    }

    // =========================================================================
    // 26. Self replacement does NOT affect the string "Self" in string literals
    // =========================================================================

    trait StringSelf {
        fn describe(&self, token: X64V3Token) -> String;
    }

    impl StringSelf for Point {
        #[arcane(_self = Point)]
        fn describe(&self, _token: X64V3Token) -> String {
            // "Self" in a string literal should NOT be replaced
            format!("Self({}, {})", _self.x, _self.y)
        }
    }

    #[test]
    fn test_self_not_replaced_in_strings() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(1.0, 2.0);
            let desc = p.describe(token);
            assert_eq!(desc, "Self(1, 2)");
        }
    }

    // =========================================================================
    // 27. Self replacement handles path segments correctly
    // =========================================================================

    #[derive(Debug, PartialEq)]
    struct Container {
        items: Vec<f32>,
    }

    impl Container {
        fn new() -> Self {
            Self { items: Vec::new() }
        }

        fn with_items(items: Vec<f32>) -> Self {
            Self { items }
        }

        fn len(&self) -> usize {
            self.items.len()
        }
    }

    trait ContainerOps {
        fn filtered(&self, token: X64V3Token, min: f32) -> Self;
    }

    impl ContainerOps for Container {
        #[arcane(_self = Container)]
        fn filtered(&self, _token: X64V3Token, min: f32) -> Self {
            let filtered: Vec<f32> = _self.items.iter().copied().filter(|&x| x >= min).collect();
            Self::with_items(filtered)
        }
    }

    #[test]
    fn test_self_path_segments() {
        if let Some(token) = X64V3Token::summon() {
            let c = Container::with_items(vec![1.0, 5.0, 2.0, 8.0, 3.0]);
            let filtered = c.filtered(token, 3.0);
            assert_eq!(filtered.items, vec![5.0, 8.0, 3.0]);
        }
    }

    // =========================================================================
    // 28. Chained _self method calls
    // =========================================================================

    trait ChainedSelf {
        fn scale_and_offset(&self, token: X64V3Token, scale: f32, offset: f32) -> Self;
    }

    impl ChainedSelf for Point {
        #[arcane(_self = Point)]
        fn scale_and_offset(&self, _token: X64V3Token, scale: f32, offset: f32) -> Self {
            // Chain: _self.scale() returns Point, then we construct Self from it
            let scaled = _self.scale(scale);
            Self::new(scaled.x + offset, scaled.y + offset)
        }
    }

    #[test]
    fn test_chained_self_methods() {
        if let Some(token) = X64V3Token::summon() {
            let p = Point::new(1.0, 2.0);
            let result = p.scale_and_offset(token, 3.0, 10.0);
            assert_eq!(result, Point::new(13.0, 16.0));
        }
    }
}

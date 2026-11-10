//! Reciprocal and reciprocal-sqrt policies for f32.
/// Input range; RSQRT additionally requires positive inputs in the Normal domain.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RecipDomain {
    /// All inputs, including IEEE specials. Requires Highp precision.
    #[default]
    Full,
    /// 2^-125 <= abs(x) <= 2^125 (positive for rsqrt). This keeps the
    /// fixed Newton graphs and native seeds away from underflow/overflow edges.
    Normal,
}
/// Precision of the selected algorithm, with bounds on the Normal domain.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RecipPrecision {
    /// Division for reciprocal; division after rounded sqrt for rsqrt.
    /// RSQRT is exact versus that two-operation graph, not correctly rounded mathematical rsqrt.
    #[default]
    Highp,
    /// Relative error <= 1e-4; two canonical Newton steps or native refinement.
    Midp,
    /// Relative error <= 6e-3; one canonical Newton step or native estimates.
    Lowp,
}
/// Cross-ISA agreement for approximate algorithms.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RecipPortability {
    /// Fixed integer seed and separate IEEE mul/sub steps; scalar/vector bits match.
    #[default]
    BitExact,
    /// Native estimates/refinement; result bits may differ between ISAs and widths.
    BackendDependent,
}
/// Enforcement of restricted input ranges.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RecipValidation {
    /// Whole-vector rejection with an exact invalid-lane mask.
    #[default]
    Checked,
    /// Omit input checks. Domain violations have unspecified numerical output, never UB.
    AssumeValid,
}
/// Selection for scalar/array entries; explicit vector/token entries keep their caller.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RecipExecution {
    /// Select once for the whole operation.
    #[default]
    Automatic,
    /// Scalar/array entries stay scalar; token entries retain that token.
    Caller,
}
/// Const builder. Profiles include range and validation, not only precision.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RecipOptions {
    /// Accepted input/output operating range.
    pub domain: RecipDomain,
    /// Algorithm precision.
    pub precision: RecipPrecision,
    /// Cross-backend agreement.
    pub portability: RecipPortability,
    /// Range validation.
    pub validation: RecipValidation,
    /// Scalar/array backend selection.
    pub execution: RecipExecution,
}
impl RecipOptions {
    pub(crate) const BACKEND: u32 = 8;
    pub(crate) const ASSUME: u32 = 16;
    pub(crate) const CALLER: u32 = 32;
    /// Full-domain division / sqrt-then-division.
    pub const HIGHP: Self = Self::new();
    /// Checked normal-domain fixed-seed two-step approximation.
    pub const PORTABLE_MIDP: Self = Self::new()
        .with_domain(RecipDomain::Normal)
        .with_precision(RecipPrecision::Midp);
    /// Checked normal-domain fixed-seed one-step approximation.
    pub const PORTABLE_LOWP: Self = Self::PORTABLE_MIDP.with_precision(RecipPrecision::Lowp);
    /// Checked normal-domain native/refined approximation.
    pub const BACKEND_MIDP: Self =
        Self::PORTABLE_MIDP.with_portability(RecipPortability::BackendDependent);
    /// Checked normal-domain native estimate.
    pub const BACKEND_LOWP: Self =
        Self::PORTABLE_LOWP.with_portability(RecipPortability::BackendDependent);
    /// Full-domain exact-graph defaults.
    pub const fn new() -> Self {
        Self {
            domain: RecipDomain::Full,
            precision: RecipPrecision::Highp,
            portability: RecipPortability::BitExact,
            validation: RecipValidation::Checked,
            execution: RecipExecution::Automatic,
        }
    }
    /// Set range.
    pub const fn with_domain(self, domain: RecipDomain) -> Self {
        Self { domain, ..self }
    }
    /// Set algorithm accuracy.
    pub const fn with_precision(self, precision: RecipPrecision) -> Self {
        Self { precision, ..self }
    }
    /// Set portability.
    pub const fn with_portability(self, portability: RecipPortability) -> Self {
        Self {
            portability,
            ..self
        }
    }
    /// Set validation.
    pub const fn with_validation(self, validation: RecipValidation) -> Self {
        Self { validation, ..self }
    }
    /// Set scalar/array selection.
    pub const fn with_execution(self, execution: RecipExecution) -> Self {
        Self { execution, ..self }
    }
    /// Encode flags; Highp has one graph, so its portability choice is canonicalized.
    /// Full-domain validation is also canonicalized away.
    pub const fn bits(self) -> u32 {
        self.domain as u32
            | ((self.precision as u32) << 1)
            | if matches!(self.portability, RecipPortability::BackendDependent)
                && !matches!(self.precision, RecipPrecision::Highp)
            {
                Self::BACKEND
            } else {
                0
            }
            | if matches!(self.validation, RecipValidation::AssumeValid)
                && !matches!(self.domain, RecipDomain::Full)
            {
                Self::ASSUME
            } else {
                0
            }
            | if matches!(self.execution, RecipExecution::Caller) {
                Self::CALLER
            } else {
                0
            }
    }
    pub(crate) const fn valid(flags: u32) -> bool {
        flags & !63 == 0
            && (flags >> 1) & 3 < 3
            && (flags & 1 != 0 || flags & 6 == 0)
            && (flags & 6 != 0 || flags & Self::BACKEND == 0)
    }
}

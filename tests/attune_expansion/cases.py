"""Declarative expansion corpus. Expected rejection is part of each case.

Products are intentionally split by concern: definition policies, placement,
modifiers, and invocation selection. The manifest records the exact finite axes;
this does not claim an exhaustive enumeration of arbitrary Rust programs.
"""
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
import tomllib


TARGETS = {
    "x86_64": "x86_64-unknown-linux-gnu",
    "aarch64": "aarch64-unknown-linux-gnu",
    "wasm32": "wasm32-unknown-unknown",
    "x86": "i686-unknown-linux-gnu",
}
NATIVE = {"x86_64": "v3", "aarch64": "neon", "wasm32": "wasm128", "x86": "scalar"}
TOKEN = {"v2": "X64V2Token", "v3": "X64V3Token", "v4x": "X64V4xToken",
         "neon": "NeonToken", "wasm128": "Wasm128Token", "scalar": "ScalarToken"}
ARCH = {"v2": "x86_64", "v3": "x86_64", "v4x": "x86_64", "neon": "aarch64",
        "wasm128": "wasm32", "scalar": None}
RANK = {"scalar": 0, "v2": 20, "v3": 30, "v4x": 50, "neon": 30, "wasm128": 30}
FORMS = {
    "direct": "_*", "proof": "_*_t", "both": "_*, _*_t",
    "dispatch": "dispatch", "all": "all", "sparse": "_v3, _v3_t, _scalar, _scalar_t",
}


@dataclass
class Case:
    name: str
    group: str
    source: str
    error: str | None = None
    tags: dict = field(default_factory=dict)
    absent: tuple[str, ...] = ()
    present: tuple[str, ...] = ()


def attribute(text):
    return f"#[archmage::attune({text})]"


def definition(selectors, syntax, visibility, inline):
    """One spelling of the same requested outputs, with explicit policy scope."""
    selected = []
    for selector in selectors.split(", "):
        if syntax == "positional":
            selected.append(f"{visibility} {selector}".strip())
        else:
            selected.append(f"{selector}({visibility})" if visibility else selector)
    outputs = ", ".join(selected)
    if syntax != "flat":
        outputs = f"make({outputs})"
    if inline != "omitted":
        outputs += f", inline({inline})"
    return outputs


def definitions(arch, enabled):
    for form, syntax, vis, policy in product(
        FORMS, ("flat", "grouped", "positional"),
        ("", "pub", "pub(crate)", "pub(self)"),
        ("omitted", "default", "hint", "none", "never", "always"),
    ):
        args = definition(FORMS[form], syntax, vis, policy)
        name = f"def_{form}_{syntax}_{vis.replace('(', '_').replace(')', '') or 'inherit'}_{policy}"
        yield Case(name, "definition-policy", f"{attribute(args)}\npub fn kernel(x: u32) -> u32 {{ x }}",
                   "inline(always) on target-feature bodies" if policy == "always" else None,
                   dict(form=form, syntax=syntax, visibility=vis, policy=policy))

    for form, syntax, policy, body_override in product(
        FORMS, ("flat", "grouped"), ("default", "hint", "none", "never", "always"), (False, True),
    ):
        outputs = ", ".join(f"{s}(pub, inline({policy}))" for s in FORMS[form].split(", "))
        args = f"make({outputs})" if syntax == "grouped" else outputs
        if body_override:
            args += ", inline(never)"
        error = "inline(always) on target-feature bodies" if policy == "always" and form not in ("proof", "dispatch") else None
        yield Case(f"selector_policy_{form}_{syntax}_{policy}_{'override' if body_override else 'inherit'}", "selector-policy",
                   f"{attribute(args)} pub fn kernel(x: u32) -> u32 {{ x }}", error,
                   dict(form=form, syntax=syntax, selector_policy=policy, body_override=body_override))

    modifiers = {
        "none": "", "remove_neon": "-_neon", "remove_absent": "-_v4",
        "remove_proof": "-_neon_t", "remove_twice": "-_neon, -_neon",
        "add_wide": "+v4x(cfg(avx512))", "remove_then_add": "-_neon, +neon",
        "add_then_remove": "+v4x(cfg(avx512)), -_v4x",
    }
    for form, syntax, (modifier, text) in product(FORMS, ("flat", "grouped"), modifiers.items()):
        # +tier needs wildcard output forms or a dispatcher. Sparse explicit
        # families exercise that rejection rather than silently avoiding it.
        outputs = FORMS[form] + (", " + text if text else "")
        args = f"make({outputs})" if syntax == "grouped" else outputs
        error = "+tier requires a wildcard" if form == "sparse" and "+" in text else None
        absent = ()
        if arch == "aarch64" and modifier in ("remove_neon", "remove_twice"):
            absent = ("kernel_neon", "NeonToken")
        if modifier == "add_then_remove":
            absent += ("kernel_v4x", "X64V4xToken")
        yield Case(f"modifier_{form}_{syntax}_{modifier}", "definition-modifier",
                   f"{attribute(args)}\npub fn kernel(x: u32) -> u32 {{ x }}", error,
                   dict(form=form, syntax=syntax, modifier=text), absent=absent)

    for form, placement in product(FORMS, ("free", "generic", "receiver", "associated", "trait")):
        args = FORMS[form]
        if placement == "free":
            source = f"{attribute(args)} pub fn kernel(x: u32) -> u32 {{ x }}"
        elif placement == "generic":
            source = f"{attribute(args)} pub fn kernel<'a, T: Copy, const N: usize>(x: &'a [T; N]) -> &'a [T; N] {{ x }}"
        elif placement == "receiver":
            source = f"pub struct Kernel; impl Kernel {{ {attribute(args)} pub fn kernel(&self, x: u32) -> u32 {{ x }} }}"
        elif placement == "associated":
            source = f"pub struct Kernel; impl Kernel {{ {attribute(args + ', in_impl')} pub fn kernel<T: Copy>(x: T) -> T {{ x }} }}"
        else:
            source = f"pub trait Kernel {{ {attribute(args + ', in_trait')} fn kernel(&self, x: u32) -> u32 {{ x }} }}"
        yield Case(f"placement_{form}_{placement}", "definition-placement", source,
                   "family generation adds sibling functions" if placement == "trait" else None,
                   dict(form=form, placement=placement))


@dataclass(frozen=True)
class Context:
    name: str
    tier: str | None
    style: str = "direct"
    parent: str | None = None


def contexts(arch, enabled):
    native = NATIVE[arch]
    result = [Context("ordinary", None, "ordinary"), Context("scalar", "scalar"),
              Context("generic_parent", "scalar", "generic", "generic"),
              Context("where_parent", "scalar", "where", "generic"),
              Context("impl_parent", "scalar", "impl", "generic"),
              Context("borrowed_parent", "scalar", "borrow", native),
              Context("concrete_parent", "scalar", "concrete", native),
              Context("ambiguous_parent", "scalar", "ambiguous", "ambiguous"),
              Context("family", "scalar", "family"),
              Context("autoversion", "scalar", "autoversion"),
              Context("magetypes", "scalar", "magetypes"),
              Context("magetypes_rite", "scalar", "magetypes_rite")]
    if native != "scalar":
        result += [Context("native", native), Context("inferred", native, "inferred"),
                   Context("inferred_proof", native, "inferred_proof", native),
                   Context("wrap", native, "wrap", native), Context("rite", native, "rite", native),
                   Context("rite_tier", native, "rite_tier"), Context("arcane", native, "arcane", native),
                   Context("receiver", native, "receiver"), Context("associated", native, "associated"),
                   Context("trait_default", native, "trait", native),
                   Context("trait_default_associated", native, "trait_associated", native),
                   Context("trait_impl", native, "trait_impl", native)]
        result.append(Context("manual_target_feature", None, "manual_target"))
    if arch == "x86_64":
        result.append(Context("lower_v2", "v2"))
        result.append(Context("feature_bound", "v2", "feature_bound"))
        if enabled:
            result.append(Context("higher_v4x", "v4x"))
    if arch == "aarch64":
        result.append(Context("feature_bound", "neon", "feature_bound"))
    return result


def caller(context, invocation, native):
    tier, style = context.tier, context.style
    tok = f"archmage::{TOKEN[native]}"
    params = "x: u32"
    generic = ""
    tail = ""
    if style in ("generic", "where", "impl", "borrow", "concrete", "ambiguous"):
        proof = {"generic": "P", "where": "P", "impl": "impl archmage::IntoConcreteToken",
                 "borrow": f"&{tok}", "concrete": tok, "ambiguous": tok}[style]
        params = f"parent: {proof}, x: u32"
        if style in ("generic", "where"):
            generic = "<P: archmage::IntoConcreteToken>" if style == "generic" else "<P>"
        if style == "where":
            tail = "where P: archmage::IntoConcreteToken"
        if style == "ambiguous":
            params += ", other: archmage::ScalarToken"
    attrs = "" if tier is None else attribute(tier)
    name = "caller"
    if style == "inferred":
        attrs, name = "#[archmage::attune]", f"caller_{tier}"
    elif style == "inferred_proof":
        attrs, name = "#[archmage::attune]", f"caller_{tier}_t"
    elif style in ("wrap", "arcane", "rite", "trait", "trait_associated", "trait_impl"):
        params = f"parent: {tok}, x: u32"
        attrs = attribute("wrap") if style in ("wrap", "trait", "trait_associated", "trait_impl") else f"#[archmage::{style}]"
    elif style == "rite_tier":
        attrs = f"#[archmage::rite({tier})]"
    elif style == "feature_bound":
        attrs = attribute("wrap")
        bound = "HasX64V2" if tier == "v2" else "HasNeon"
        generic = f"<P: archmage::{bound}>"
        params = "parent: P, x: u32"
    elif style == "family":
        attrs = attribute("all")
    elif style == "autoversion":
        attrs = "#[archmage::autoversion]"
    elif style in ("magetypes", "magetypes_rite"):
        modes = f"{native}, scalar" if native != "scalar" else "scalar"
        if style == "magetypes_rite":
            modes = "rite, " + modes
        attrs = f"#[archmage::magetypes({modes})]"
        params = "parent: Token, x: u32"
    elif style == "manual_target":
        feature = {"v3": "avx2", "neon": "neon", "wasm128": "simd128"}[native]
        attrs = f'#[target_feature(enable = "{feature}")]'
    if style == "receiver":
        return f"pub struct Kernel; impl Kernel {{ {attrs} pub fn caller(&self, {params}) -> u32 {{ {invocation} }} }}"
    if style == "associated":
        return f"pub struct Kernel; impl Kernel {{ {attribute(tier + ', in_impl')} pub fn caller({params}) -> u32 {{ {invocation} }} }}"
    if style == "trait":
        return f"pub trait Kernel {{ {attribute('wrap, in_trait')} fn caller(&self, {params}) -> u32 {{ {invocation} }} }}"
    if style == "trait_associated":
        return f"pub trait Kernel {{ {attribute('wrap, in_trait')} fn caller({params}) -> u32 {{ {invocation} }} }}"
    if style == "trait_impl":
        return f"pub struct Kernel; pub trait Apply {{ fn caller(&self, {params}) -> u32; }} impl Apply for Kernel {{ {attribute('wrap, in_trait, _self = Kernel')} fn caller(&self, {params}) -> u32 {{ {invocation} }} }}"
    return f"{attrs}\npub fn {name}{generic}({params}) -> u32 {tail} {{ {invocation} }}"


LISTS = {
    "implicit": None,
    "portable": [(t, False, False) for t in ("v3", "neon", "wasm128", "scalar")],
    "lower": [("v2", False, False), ("scalar", False, False)],
    "sparse_v3": [("v3", False, False)],
    "foreign_neon": [("neon", False, False)],
    "forced_proof": [("v3", True, False), ("scalar", True, False)],
    "wide": [("v4x", False, True), ("v3", False, False), ("neon", False, False), ("wasm128", False, False), ("scalar", False, False)],
    "gated_only": [("v4x", False, True), ("scalar", False, False)],
    "foreign_proof_scalar": [("neon", True, False), ("scalar", False, False)],
    "scalar_only": [("scalar", False, False)],
    "empty": [],
}


def covered(proof, tier):
    if tier == "scalar":
        return True
    return proof == tier or (proof == "v4x" and tier in ("v3", "v2")) or (proof == "v3" and tier == "v2")


def selection_contract(context, macro, entries, explicit, arch, enabled):
    """Semantic oracle: a call needs a guaranteed fallback, not merely a probe.

    Multi-body callers must work in their scalar body too. Foreign architectures
    and disabled Cargo gates cannot provide a guarantee on this build.
    """
    active = [t for t, _, gated in (LISTS["portable"] if entries is None else entries)
              if (not gated or enabled) and ARCH[t] in (None, arch)]
    probes = False
    for tier in sorted(active, key=RANK.get, reverse=True):
        if tier == "scalar":
            return None, probes
        if explicit:
            continue  # exact-type extraction is conditional, never a guarantee
        if covered(context.tier, tier):
            return None, probes
        if context.tier is not None and macro == "attuned":
            if context.parent == "ambiguous":
                return "multiple parent proof parameters", False
            if context.parent and context.parent != "generic" and covered(context.parent, tier):
                return None, False
        else:
            probes = True
    return "no guaranteed fallback", probes


def list_spelling(entries):
    if entries is None:
        return ""
    return ", [" + ", ".join("_" + t + ("_t" if proof else "") + ("(cfg(avx512))" if gated else "")
                            for t, proof, gated in entries) + "]"


SUPPORT = """
pub mod support {
    #[archmage::attune(all, +v2, +v4x(cfg(avx512)))]
    pub fn leaf<T: Copy>(x: T) -> T { x }
}
"""


def calls(arch, enabled):
    for context, macro, (label, entries), explicit in product(
        contexts(arch, enabled), ("attuned", "reattune"), LISTS.items(), (False, True),
    ):
        expression = f"archmage::{macro}!(support::leaf(x){list_spelling(entries)}" + (", using(archmage::ScalarToken)" if explicit else "") + ")"
        error, probes = selection_contract(context, macro, entries, explicit, arch, enabled)
        if context.style == "trait":
            # Receiverful nested defaults already required a concrete _self in
            # the pre-rewrite implementation. Keep every call form as rejection.
            error = "requires `_self = Type`"
        # Body scans below exclude support, whose dispatcher legitimately probes.
        yield Case(f"call_{context.name}_{macro}_{label}_{'using' if explicit else 'implicit'}", "invocation",
                   SUPPORT + caller(context, expression, NATIVE[arch]), error,
                   dict(context=context.name, macro=macro, tiers=label, proof="explicit-scalar" if explicit else "inferred-or-context",
                        runtime_probe=probes, inspect_caller=context.style not in ("family", "autoversion", "magetypes", "magetypes_rite")))


def extra_syntax(arch, enabled):
    native = NATIVE[arch]
    tok = f"archmage::{TOKEN[native]}"
    for mode, args, signature in [
        ("explicit", native, "pub fn kernel(x: u32) -> u32"),
        ("inferred", "", f"pub fn kernel_{native}(x: u32) -> u32"),
        ("inferred_proof", "", f"pub fn kernel_{native}_t(x: u32) -> u32"),
        ("token_position", "", f"pub fn kernel_{native}_t(x: u32, proof: Token) -> u32"),
        ("wrap", "wrap", f"pub fn kernel(proof: {tok}, x: u32) -> u32"),
        ("cfg", f"{native}, cfg(optional)", "pub fn kernel(x: u32) -> u32"),
        ("intrinsics", f"{native}, import_intrinsics", "pub fn kernel(x: u32) -> u32"),
        ("magetypes_import", f"{native}, import_magetypes", "pub fn kernel(x: u32) -> u32"),
        ("define", "all, define(f32x8)", "pub fn kernel(x: u32) -> u32"),
        ("names", "_scalar, _scalar_t, names(_scalar = direct, _scalar_t = proof)", "pub fn kernel(x: u32) -> u32"),
    ]:
        body = "let _: Option<f32x8> = None; x" if mode == "define" else "x"
        yield Case(f"option_{mode}", "options", f"{attribute(args)} {signature} {{ {body} }}", tags=dict(option=mode))

    for shape, body in {
        "qualified": "::archmage::attuned!(support::leaf(x), [_scalar])",
        "turbofish": "archmage::attuned!(support::leaf::<u32>(x), [_scalar])",
        "closure": "let f = || archmage::attuned!(support::leaf(x), [_scalar]); f()",
        "nested_fn": "fn nested(x: u32) -> u32 { archmage::attuned!(support::leaf(x), [_scalar]) } nested(x)",
        "nested_argument": "archmage::attuned!(support::leaf(archmage::attuned!(support::leaf(x), [_scalar])), [_scalar])",
        "using_expression": "archmage::attuned!(support::leaf(x), [_scalar], using({ let proof = archmage::ScalarToken; proof }))",
        "renamed_path": "archmage::attuned!(missing(x), [_scalar], names(_scalar = support::leaf_scalar, _scalar_t = support::leaf_scalar_t))",
        "local_shadow": "let parent = x; archmage::attuned!(support::leaf(parent), [_scalar])",
        "brace_delimiter": "archmage::attuned!{support::leaf(x), [_scalar]}",
        "bracket_delimiter": "archmage::attuned![support::leaf(x), [_scalar]]",
    }.items():
        for context in (Context("ordinary", None, "ordinary"), Context("scalar", "scalar"), Context("native", native)):
            yield Case(f"shape_{context.name}_{shape}", "call-shape", SUPPORT + caller(context, body, native), tags=dict(context=context.name, shape=shape))

    for modifier in ("-_neon", "+v4x(cfg(avx512))", "_*, -_neon"):
        # These are definition modifiers, not currently a call-list grammar.
        for macro in ("attuned", "reattune"):
            yield Case(f"unsupported_list_{macro}_{len(modifier)}", "unsupported-invocation",
                       SUPPORT + f"pub fn caller(x: u32) -> u32 {{ archmage::{macro}!(support::leaf(x), [{modifier}]) }}",
                       "expected identifier", dict(macro=macro, modifier=modifier))

    rejections = {
        "mixed_modes": ("v3, _v3", "cannot be combined"),
        "mixed_group": ("make(_v3), _scalar", "keep all outputs"),
        "duplicate_inline": ("v3, inline(hint), inline(never)", "duplicate body inline policy"),
        "duplicate_visibility": ("_scalar(pub, pub(crate))", "duplicate visibility option"),
        "duplicate_gate": ("_scalar(cfg(a), cfg(b))", "duplicate cfg option"),
        "unknown_option": ("_scalar(unknown)", "expected pub visibility"),
        "unknown_policy": ("_scalar(inline(maybe))", "expected default"),
        "gate_disagreement": ("_v3(cfg(a)), _v3_t(cfg(b))", "same feature gate"),
        "scalar_removal": ("dispatch, -_scalar", "requires its scalar fallback"),
        "scalar_gate": ("dispatch, _scalar(cfg(optional))", "ungated scalar fallback"),
        "wildcard_gate": ("_*(cfg(optional))", "cfg belongs on a named tier"),
        "remove_options": ("all, -_neon(pub)", "removals do not accept options"),
        "empty_outputs": ("make()", "selects no outputs"),
        "unsupported_auto": ("auto", "unknown attune tier"),
        "unknown_tier": ("_future_tier", "unknown attune tier"),
        "conflicting_placement": ("wrap, in_impl, in_trait", "different placements"),
        "trait_visibility": ("wrap, in_trait, inline(default)", "cannot infer trait visibility"),
    }
    for name, (args, error) in rejections.items():
        yield Case(f"reject_{name}", "rejection", f"{attribute(args)} pub fn kernel(x: u32) -> u32 {{ x }}", error)
    yield Case("reject_method_invocation", "unsupported-invocation",
               "pub struct Kernel; pub fn caller(k: Kernel) { archmage::attuned!(k.run()); }", "expected parentheses")

    for position, params, arguments in [
        ("first", "proof: Token, x: u32", "Token, x"),
        ("middle", "x: u32, proof: Token, y: u32", "x, Token, 0"),
        ("last", "x: u32, proof: Token", "x, Token"),
    ]:
        support = f"{attribute('all')} pub fn positioned({params}) -> u32 {{ let _ = proof; x }}"
        for style in ("ordinary", "scalar", "native"):
            context = Context(style, None if style == "ordinary" else native if style == "native" else "scalar", "ordinary" if style == "ordinary" else "direct")
            call = f"archmage::attuned!(positioned({arguments}), [_v3, _neon, _wasm128, _scalar])"
            yield Case(f"marker_{position}_{style}", "token-marker", support + caller(context, call, native), tags=dict(position=position, context=style))

    for macro in ("attuned", "reattune"):
        source = f"""
pub struct Kernel;
impl Kernel {{
    #[archmage::attune(_scalar, _scalar_t, in_impl)]
    pub fn leaf<T: Copy>(x: T) -> T {{ x }}
    #[archmage::attune(scalar, in_impl)]
    pub fn caller(x: u32) -> u32 {{ archmage::{macro}!(Self::leaf::<u32>(x), [_scalar]) }}
}}
pub fn ordinary(x: u32) -> u32 {{ archmage::{macro}!(Kernel::leaf::<u32>(x), [_scalar]) }}
"""
        yield Case(f"associated_paths_{macro}", "call-shape", source)

    for attr_name, extra in [
        ("inline", "#[inline]"), ("never", "#[inline(never)]"),
        ("track_caller", "#[track_caller]"), ("must_use", "#[must_use]"),
        ("doc", '#[doc = "Expanded operation"]'),
        ("cfg_attr", "#[cfg_attr(feature = \"optional\", inline(never))]"),
        ("expect", '#[expect(unused_variables, reason = "operation ignores x")]'),
    ]:
        for form in ("direct", "proof", "dispatch", "all"):
            body = "0" if attr_name == "expect" else "x"
            yield Case(f"attribute_{form}_{attr_name}", "rust-attribute",
                       f"{attribute(FORMS[form])} {extra} pub fn kernel(x: u32) -> u32 {{ {body} }}",
                       tags=dict(form=form, attribute=extra))

    for macro in ("attuned", "reattune"):
        expression = f"archmage::{macro}!(support::leaf(x), [_v3, _neon, _wasm128, _scalar], using({tok}::from_context()))"
        yield Case(f"from_context_{macro}", "proof-expression",
                   SUPPORT + caller(Context("native", native), expression, native))

    for shape, signature, body in [
        ("pattern", "pub fn kernel((x, _): (u32, u32), _: u32) -> u32", "x"),
        ("move", "pub fn kernel(x: String) -> usize", "x.len()"),
        ("borrow", "pub fn kernel<'a>(x: &'a u32) -> &'a u32", "x"),
        ("const_generic", "pub fn kernel<const N: usize>(x: [u32; N]) -> [u32; N]", "x"),
        ("where", "pub fn kernel<T>(x: T) -> T where T: Copy", "x"),
    ]:
        yield Case(f"signature_{shape}", "signature", f"{attribute('all')} {signature} {{ {body} }}")


def corpus(arch, enabled):
    cases = list(definitions(arch, enabled)) + list(calls(arch, enabled)) + list(extra_syntax(arch, enabled)) + list(registered_tiers()) + list(nesting_and_inference(arch))
    names = [case.name for case in cases]
    assert len(names) == len(set(names)), "duplicate case IDs"
    return cases


def registered_tiers():
    registry = tomllib.loads((Path(__file__).resolve().parents[2] / "token-registry.toml").read_text())
    for token in registry["token"]:
        if "dispatch_priority" not in token:
            continue
        tier = token["short_name"]
        gate = token.get("legacy_dispatch_gate")
        options = f"(cfg({gate}))" if gate else ""
        guard = f'#[cfg(feature = "{gate}")]' if gate else ""
        source = f"""
#[archmage::attune(_{tier}{options}, _{tier}_t{options}, dispatch)]
pub fn leaf(x: u32) -> u32 {{ x }}
{guard}
#[archmage::attune({tier})]
pub fn caller(x: u32) -> u32 {{ archmage::attuned!(leaf(x), [_{tier}]) }}
{guard}
#[archmage::attune(wrap)]
pub fn proof_caller(proof: archmage::{token['name']}, x: u32) -> u32 {{
    archmage::attuned!(leaf(x), [_{tier}_t])
}}
"""
        yield Case(f"registered_{tier}", "registered-tier", source,
                   tags=dict(tier=tier, architecture=token["arch"], gate=gate))


def nesting_and_inference(arch):
    native = NATIVE[arch]
    token = f"archmage::{TOKEN[native]}"
    for frontend, options in (("arcane", ""), ("attune", "wrap, ")):
        for spelling in ("nested", "in_trait", "implied"):
            placement = "" if spelling == "implied" else spelling + ", "
            args = options + placement + "_self = Kernel"
            source = f"""
pub struct Kernel {{ bias: u32 }}
pub trait Apply {{ fn caller(&self, proof: {token}, x: u32) -> u32; }}
impl Apply for Kernel {{
    #[archmage::{frontend}({args})]
    fn caller(&self, proof: {token}, x: u32) -> u32 {{ self.bias + x }}
}}
"""
            yield Case(f"nesting_{frontend}_{spelling}", "nesting", source)
        for spelling in ("nested", "in_trait"):
            source = f"""
pub trait Apply {{
    #[archmage::{frontend}({options}{spelling})]
    fn caller(&self, proof: {token}, x: u32) -> u32 {{ x }}
}}
"""
            yield Case(f"nesting_{frontend}_{spelling}_missing_type", "nesting", source,
                       "requires `_self = Type`")
    yield Case("inference_token_alone", "inference",
               f"#[archmage::attune] pub fn caller(proof: {token}, x: u32) -> u32 {{ x }}",
               "attune needs a tier")
    yield Case("inference_direct_with_token", "inference",
               f"#[archmage::attune] pub fn caller_{native}(proof: {token}, x: u32) -> u32 {{ x }}",
               absent=("__arcane_",))

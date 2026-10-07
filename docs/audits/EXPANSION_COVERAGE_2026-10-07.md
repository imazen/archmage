# Expansion coverage inventory — 2026-10-07

Pinned main: `52bf060d`. All 161 checked-in expanded outputs were read at source level; none remain unreviewed. This is not full consumer-body coverage.

Each row names the output relative to its section root, omitting `.expanded.rs`. The corresponding input is the same stem plus `.rs`; all 161 inputs exist. Inputs were inventoried and selected inputs inspected; do not interpret output review as an independent line-by-line review of every input.

`S` = expanded output source-reviewed. `C` = compared by archmage macrotest on main default, main avx512, and draft default. `P` = ordinary input/output compile harness passed. `K` = known-failure category, not an ordinary pass test. `M` = parent independently passed main magetypes default/avx512 harnesses (not draft).

Counts: archmage 152 (145 ordinary + 7 known-failure); magetypes 9. All checked-in output is host-filtered x86_64, not portable expansion coverage.

## `tests/expand/`

| Stem | Coverage | Source-review note |
|---|---|---|
| arcane/associated_type_bound | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| arcane/box_dyn_return | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| arcane/cfg_feature | S C P | Disabled body absent in host snapshot; enabled branch not reviewed here. |
| arcane/closure_param | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| arcane/destructured_array | S C P | Reviewed renamed wrapper parameters and inner pattern/body forwarding. |
| arcane/destructured_nested | S C P | Reviewed renamed wrapper parameters and inner pattern/body forwarding. |
| arcane/destructured_tuple | S C P | Reviewed renamed wrapper parameters and inner pattern/body forwarding. |
| arcane/dyn_trait_param | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| arcane/fn_ptr_param | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| arcane/generic_const | S C P | Reviewed type/const forwarding and lifetime inference or receiver lifetime. |
| arcane/generic_lifetime | S C P | Reviewed type/const forwarding and lifetime inference or receiver lifetime. |
| arcane/generic_lifetime_and_type | S C P | Reviewed type/const forwarding and lifetime inference or receiver lifetime. |
| arcane/generic_multi_params | S C P | Reviewed type/const forwarding and lifetime inference or receiver lifetime. |
| arcane/generic_where_clause | S C P | Reviewed type/const forwarding and lifetime inference or receiver lifetime. |
| arcane/higher_ranked_trait_bound | S C P | Reviewed bound feature list and absolute trait proof; forwarding retained. |
| arcane/import_intrinsics | S C P | Body imports architecture combined intrinsics; safe reference memory calls. |
| arcane/multi_bounds | S C P | Reviewed bound feature list and absolute trait proof; forwarding retained. |
| arcane/multiple_wildcards | S C P | Reviewed renamed wrapper parameters and inner pattern/body forwarding. |
| arcane/nested | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| arcane/nested_self | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| arcane/return_impl_trait | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| arcane/sibling | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| arcane/token_first | S C P | Reviewed call argument positions and fallback/selected-tier forwarding. |
| arcane/token_generic | S C P | Reviewed type/const forwarding and lifetime inference or receiver lifetime. |
| arcane/token_last | S C P | Reviewed call argument positions and fallback/selected-tier forwarding. |
| arcane/token_middle | S C P | Reviewed call argument positions and fallback/selected-tier forwarding. |
| arcane/token_neon | S C P | Disabled body absent in host snapshot; enabled branch not reviewed here. |
| arcane/token_trait_bound | S C P | Reviewed bound feature list and absolute trait proof; forwarding retained. |
| arcane/token_v2 | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| arcane/token_v3 | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| arcane/token_v4 | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| arcane/token_wasm | S C P | Disabled body absent in host snapshot; enabled branch not reviewed here. |
| arcane/unsafe_fn | S C P | Unsafe wrapper retained; safe native sibling is finding A1 (rite preserves unsafe). |
| arcane/wildcard_token | S C P | Reviewed renamed wrapper parameters and inner pattern/body forwarding. |
| autoversion/basic | S C P | Dispatcher selection and private variants reviewed; fallback retained. |
| autoversion/cfg_feature | S C P | Disabled body absent in host snapshot; enabled branch not reviewed here. |
| autoversion/default_tier | S C P | Dispatcher selection and private variants reviewed; fallback retained. |
| autoversion/explicit_gated | S C P | Reviewed host-filtered gate/fallback; enabled-gate combinations limited. |
| autoversion/explicit_v3_neon_scalar | S C P | Dispatcher selection and private variants reviewed; fallback retained. |
| autoversion/explicit_v3_scalar | S C P | Dispatcher selection and private variants reviewed; fallback retained. |
| autoversion/plain_self | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| autoversion/scalar_token | S C P | Dispatcher selection and private variants reviewed; fallback retained. |
| autoversion/self_type | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| autoversion/tier_modifiers | S C P | Dispatcher selection and private variants reviewed; fallback retained. |
| autoversion/unsafe_fn | S C P | Unsafe wrapper retained; safe native sibling is finding A1 (rite preserves unsafe). |
| combinations/arcane_calls_rite | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| combinations/autoversion_chain | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| combinations/plain_token_fn | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| combinations/token_downgrade | S C P | Reviewed direct proof downgrade or runtime upgrade; host gates apply. |
| combinations/token_upgrade | S C P | Reviewed direct proof downgrade or runtime upgrade; host gates apply. |
| deprecated/autoversion_simdtoken | S C P | Alias lowering reviewed against corresponding thematic macro shape. |
| deprecated/dispatch_variant | S C P | Alias lowering reviewed against corresponding thematic macro shape. |
| deprecated/simd_fn | S C P | Alias lowering reviewed against corresponding thematic macro shape. |
| deprecated/simd_route | S C P | Alias lowering reviewed against corresponding thematic macro shape. |
| deprecated/token_target_features | S C P | Alias lowering reviewed against corresponding thematic macro shape. |
| deprecated/token_target_features_boundary | S C P | Alias lowering reviewed against corresponding thematic macro shape. |
| incant/default_tiers | S C P | Reviewed call argument positions and fallback/selected-tier forwarding. |
| incant/feature_gated | S C P | Reviewed host-filtered gate/fallback; enabled-gate combinations limited. |
| incant/tier_modifiers | S C P | Reviewed call argument positions and fallback/selected-tier forwarding. |
| incant/token_explicit_first | S C P | Reviewed call argument positions and fallback/selected-tier forwarding. |
| incant/token_explicit_last | S C P | Reviewed call argument positions and fallback/selected-tier forwarding. |
| incant/token_prepend | S C P | Reviewed call argument positions and fallback/selected-tier forwarding. |
| rewrite/arcane_downgrade | S C P | Reviewed direct proof downgrade or runtime upgrade; host gates apply. |
| rewrite/arcane_exact | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| rewrite/arcane_named_token | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| rewrite/arcane_token_last | S C P | Reviewed call argument positions and fallback/selected-tier forwarding. |
| rewrite/arcane_upgrade | S C P | Reviewed direct proof downgrade or runtime upgrade; host gates apply. |
| rewrite/arcane_upgrade_gated | S C P | Reviewed host-filtered gate/fallback; enabled-gate combinations limited. |
| rewrite/autoversion_exact | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| rewrite/autoversion_upgrade | S C P | Reviewed direct proof downgrade or runtime upgrade; host gates apply. |
| rewrite/autoversion_upgrade_gated | S C P | Reviewed host-filtered gate/fallback; enabled-gate combinations limited. |
| rewrite/cross_branch_no_downgrade | S C P | Reviewed direct proof downgrade or runtime upgrade; host gates apply. |
| rewrite/magetypes_rite_flag | S C P | Direct feature body plus intrinsic import; scalar has no boundary. |
| rewrite/rite_exact | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| rewrite/rite_multi | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| rewrite/rite_tokenless_passthrough | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| rewrite/scalar_fallback | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| rewrite/without_token_from_arcane | S C P | Matching-tier suffixed call has no token; scalar call remains tokenless. |
| rewrite/without_token_rite_multi | S C P | Matching-tier suffixed call has no token; scalar call remains tokenless. |
| rite/import_intrinsics | S C P | Body imports architecture combined intrinsics; safe reference memory calls. |
| rite/modifier_minus_scalar | S C P | Direct feature/inline policy reviewed; scalar/default omit feature attr. |
| rite/modifier_mixed | S C P | Direct feature/inline policy reviewed; scalar/default omit feature attr. |
| rite/modifier_plus_multi | S C P | Direct feature/inline policy reviewed; scalar/default omit feature attr. |
| rite/multi_v3_neon | S C P | Direct feature/inline policy reviewed; scalar/default omit feature attr. |
| rite/multi_v3_v4_neon | S C P | Direct feature/inline policy reviewed; scalar/default omit feature attr. |
| rite/multi_with_cfg | S C P | Disabled body absent in host snapshot; enabled branch not reviewed here. |
| rite/multi_with_scalar_and_default | S C P | Direct feature/inline policy reviewed; scalar/default omit feature attr. |
| rite/single_default | S C P | Direct feature/inline policy reviewed; scalar/default omit feature attr. |
| rite/single_scalar | S C P | Direct feature/inline policy reviewed; scalar/default omit feature attr. |
| rite/single_tier | S C P | Direct feature/inline policy reviewed; scalar/default omit feature attr. |
| rite/single_token | S C P | Direct feature/inline policy reviewed; scalar/default omit feature attr. |
| rite/trait_bound | S C P | Reviewed bound feature list and absolute trait proof; forwarding retained. |
| rite/unsafe_fn | S C P | Unsafe wrapper retained; safe native sibling is finding A1 (rite preserves unsafe). |
| shapes/arcane_assoc_no_receiver_in_impl | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/arcane_assoc_no_receiver_nested | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/arcane_bounds_copy_then_where | S C P | Reviewed bound feature list and absolute trait proof; forwarding retained. |
| shapes/arcane_bounds_inline | S C P | Reviewed bound feature list and absolute trait proof; forwarding retained. |
| shapes/arcane_bounds_split | S C P | Reviewed bound feature list and absolute trait proof; forwarding retained. |
| shapes/arcane_bounds_where_only | S C P | Reviewed bound feature list and absolute trait proof; forwarding retained. |
| shapes/arcane_bounds_where_two | S C P | Reviewed bound feature list and absolute trait proof; forwarding retained. |
| shapes/arcane_gated_direct_tier_nested_incant | S C P | Reviewed host-filtered gate/fallback; enabled-gate combinations limited. |
| shapes/arcane_generics_lifetime_const_type | S C P | Reviewed type/const forwarding and lifetime inference or receiver lifetime. |
| shapes/arcane_method_mut_self | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/arcane_method_owned_self | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/arcane_method_ref_self | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/arcane_named_token_scalar_nested_incant | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| shapes/arcane_return_impl_trait | S C P | Reviewed body, feature boundary and forwarding; no extra issue found. |
| shapes/arcane_trait_impl_in_trait_box | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/arcane_trait_impl_in_trait_lifetime | S C P | Reviewed type/const forwarding and lifetime inference or receiver lifetime. |
| shapes/arcane_trait_impl_in_trait_nested_impl | S C P | Nested impl keeps its own self; outer receiver is rewritten. |
| shapes/arcane_trait_impl_in_trait_self | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/arcane_trait_impl_in_trait_self_param | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/arcane_trait_impl_nested_self | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/arcane_tuple_pattern | S C P | Reviewed renamed wrapper parameters and inner pattern/body forwarding. |
| shapes/arcane_where_on_other_param | S C P | Reviewed bound feature list and absolute trait proof; forwarding retained. |
| shapes/arcane_wildcard_generic | S C P | Reviewed type/const forwarding and lifetime inference or receiver lifetime. |
| shapes/arcane_wildcard_impl_trait | S C P | Reviewed renamed wrapper parameters and inner pattern/body forwarding. |
| shapes/arcane_wildcard_token_nested_incant | S C P | Reviewed renamed wrapper parameters and inner pattern/body forwarding. |
| shapes/autoversion_assoc_no_receiver_in_impl | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/autoversion_generics_lifetime_const_type | S C P | Reviewed type/const forwarding and lifetime inference or receiver lifetime. |
| shapes/autoversion_method_ref_self | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/autoversion_token_last_param | S C P | Reviewed call argument positions and fallback/selected-tier forwarding. |
| shapes/autoversion_trait_impl_in_trait | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/autoversion_trait_impl_in_trait_box | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/autoversion_trait_impl_in_trait_lifetime | S C P | Reviewed type/const forwarding and lifetime inference or receiver lifetime. |
| shapes/autoversion_trait_impl_in_trait_nested_impl | S C P | Nested impl keeps its own self; outer receiver is rewritten. |
| shapes/autoversion_trait_impl_in_trait_self_param | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/autoversion_tuple_pattern | S C P | Reviewed renamed wrapper parameters and inner pattern/body forwarding. |
| shapes/autoversion_wildcard_param | S C P | Reviewed renamed wrapper parameters and inner pattern/body forwarding. |
| shapes/magetypes_assoc_no_receiver_in_impl | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/magetypes_method_ref_self | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/rite_assoc_no_receiver | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/rite_bounds_copy_then_where | S C P | Reviewed bound feature list and absolute trait proof; forwarding retained. |
| shapes/rite_bounds_inline | S C P | Reviewed bound feature list and absolute trait proof; forwarding retained. |
| shapes/rite_bounds_split | S C P | Reviewed bound feature list and absolute trait proof; forwarding retained. |
| shapes/rite_bounds_where_only | S C P | Reviewed bound feature list and absolute trait proof; forwarding retained. |
| shapes/rite_bounds_where_two | S C P | Reviewed bound feature list and absolute trait proof; forwarding retained. |
| shapes/rite_generics_lifetime_const_type | S C P | Reviewed type/const forwarding and lifetime inference or receiver lifetime. |
| shapes/rite_method_mut_self | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/rite_method_owned_self | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/rite_method_ref_self | S C P | Reviewed receiver/associated call placement and signature forwarding. |
| shapes/rite_tuple_pattern | S C P | Reviewed renamed wrapper parameters and inner pattern/body forwarding. |
| shapes/rite_where_on_other_param | S C P | Reviewed bound feature list and absolute trait proof; forwarding retained. |
| shapes/rite_wildcard_generic | S C P | Reviewed type/const forwarding and lifetime inference or receiver lifetime. |
| shapes/rite_wildcard_impl_trait | S C P | Reviewed renamed wrapper parameters and inner pattern/body forwarding. |
| should-fail/arcane_assoc_no_receiver | S C K | Known context restriction; not an ordinary pass fixture. |
| should-fail/arcane_trait_impl_sibling | S C K | Known context restriction; not an ordinary pass fixture. |
| should-fail/autoversion_assoc_no_receiver | S C K | Known context restriction; not an ordinary pass fixture. |
| should-fail/autoversion_trait_impl | S C K | Known context restriction; not an ordinary pass fixture. |
| should-fail/incant_passthrough | S C K | Standalone output uses panic internals; not proof that macro input fails. |
| should-fail/magetypes_assoc_no_receiver | S C K | Known context restriction; not an ordinary pass fixture. |
| should-fail/rite_trait_impl | S C K | Known context restriction; not an ordinary pass fixture. |

## `magetypes/tests/expand/`

| Stem | Coverage | Source-review note |
|---|---|---|
| define/empty_list | S M | Reviewed local generic vector aliases and scalar/V3 specialization. |
| define/multiple_types | S M | Reviewed local generic vector aliases and scalar/V3 specialization. |
| define/position_middle | S M | Reviewed local generic vector aliases and scalar/V3 specialization. |
| define/single_type | S M | Reviewed local generic vector aliases and scalar/V3 specialization. |
| define/strict_lints | S M | Reviewed local generic vector aliases and scalar/V3 specialization. |
| define/with_rite_flag | S M | Reviewed local generic vector aliases and scalar/V3 specialization. |
| rite_flag/basic | S M | Direct feature body plus intrinsic import; scalar has no boundary. |
| rite_flag/with_magetypes_body | S M | Direct feature body plus intrinsic import; scalar has no boundary. |
| without_token/magetypes_body | S M | Matching-tier suffixed call has no token; scalar call remains tokenless. |

## Generated harvested-signature artifacts

All four cargo-expand commands succeeded; only selected output regions were manually read (h002 x86/default/AVX-512 and its non-x86 omission, h000/h001/h007/h008 NEON, h090–h092 WASM dispatcher/const generic examples). The remaining generated harvested bodies were not individually reviewed.

| Artifact | Modules retained | Shape function declarations | target_feature attributes |
|---|---:|---:|---:|
| harvest-default.rs | 230 | 419 | 183 |
| harvest-avx512.rs | 233 | 489 | 219 |
| harvest-aarch64.rs | 230 | 307 | 125 |
| harvest-wasm.rs | 230 | 168 | 81 |

These are lexical counts, including wrappers and empty cfg-filtered modules, not independent behavior tests. Source has 233 normalized shape modules. Bodies are `todo!()`; no consumer algorithm was executed or fully expanded.

Repository lexical macro-use inventory: `EXPANSION_MACRO_USES_2026-10-07.tsv`. Every row is inventory-only unless covered above; no claim of full expansion review for those 324 source files. Counts can include macro syntax in comments and templates.

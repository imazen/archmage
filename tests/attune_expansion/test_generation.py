"""Generator contracts; rustc and raw replay are the implementation oracle."""
from collections import Counter
from itertools import product
import unittest

from cases import FORMS, TARGETS, Context, corpus, covered, selection_contract
from run import brace_end, source_cases


class CorpusTests(unittest.TestCase):
    def test_axes_have_no_holes_or_duplicate_names(self):
        for arch, enabled in product(TARGETS, (False, True)):
            cases = corpus(arch, enabled)
            names = [c.name for c in cases]
            self.assertEqual(len(names), len(set(names)))
            for name in names:
                self.assertRegex(name, r"^[a-z][a-z0-9_]*$")
            counts = Counter(c.group for c in cases)
            self.assertEqual(counts["definition-policy"], len(FORMS) * 3 * 4 * 6)
            self.assertEqual(counts["selector-policy"], len(FORMS) * 2 * 5 * 2)
            self.assertEqual(counts["definition-placement"], len(FORMS) * 5)
            self.assertEqual(counts["definition-modifier"], len(FORMS) * 2 * 8)
            self.assertEqual(counts["token-marker"], 3 * 3)
            self.assertEqual(counts["rust-attribute"], 7 * 4)
            self.assertTrue(any(c.error is None for c in cases))
            self.assertTrue(any(c.error is not None for c in cases))

    def test_selection_guarantees_are_not_cpu_probes(self):
        v3 = [("v3", False, False)]
        self.assertIsNone(selection_contract(Context("v3", "v3"), "attuned", v3, False, "x86_64", False)[0])
        self.assertEqual(selection_contract(Context("outside", None), "reattune", v3, False, "x86_64", False)[0], "no guaranteed fallback")
        self.assertEqual(selection_contract(Context("v3", "v3"), "attuned", v3, True, "x86_64", False)[0], "no guaranteed fallback")
        self.assertFalse(covered("v3", "neon"))
        self.assertTrue(covered("v4x", "v2"))

    def test_gate_and_foreign_architecture_cannot_supply_fallback(self):
        gated = [("v4x", False, True)]
        context = Context("v4x", "v4x")
        self.assertEqual(selection_contract(context, "attuned", gated, False, "x86_64", False)[0], "no guaranteed fallback")
        self.assertIsNone(selection_contract(context, "attuned", gated, False, "x86_64", True)[0])
        self.assertEqual(selection_contract(context, "attuned", gated, False, "aarch64", True)[0], "no guaranteed fallback")

    def test_generated_module_extraction_ignores_literal_braces(self):
        block = 'pub mod x { fn f() { let _ = "}"; let _ = \'{\'; /* } */ } // }\n }'
        text = block + "\npub mod later {}"
        self.assertEqual(brace_end(text, text.index("{")), len(block))

    def test_diagnostic_owner_uses_macro_expansion_source(self):
        span = {"file_name": "/repo/macros/src/lib.rs", "expansion": {"span": {"file_name": "src/cases/example.rs"}}}
        self.assertEqual(source_cases(span), {"example"})


if __name__ == "__main__":
    unittest.main()

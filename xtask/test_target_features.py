"""Regression cases for concrete-impl target-feature comparison."""
import copy
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from check_target_features import compare, safe_methods
from check_semver import ROOT, check_artifacts


def document():
    result = {'paths': {'100': {'path': ['magetypes', 'f32x16']}}, 'index': {}}
    for number, (token, features) in enumerate([
            ('V4', ['avx512f']), ('V4x', ['avx512f', 'avx512vbmi'])]):
        type_id, method_id, impl_id = str(number), str(number + 10), str(number + 20)
        result['paths'][type_id] = {'path': ['archmage', token]}
        result['index'][method_id] = {
            'name': 'from_raw', 'visibility': 'public',
            'attrs': [{'target_feature': {'enable': features}}],
            'inner': {'function': {'header': {'is_unsafe': False}}}}
        result['index'][impl_id] = {'inner': {'impl': {
            'trait': None, 'for': {'resolved_path': {
                'id': '100', 'path': 'f32x16', 'args': {'angle_bracketed': {
                    'args': [{'type': {'resolved_path': {
                        'id': type_id, 'path': token, 'args': None}}}],
                    'constraints': []}}}},
            'generics': {}, 'items': [method_id]}}}
    return result


class TargetFeatureTests(unittest.TestCase):
    def test_only_the_replaced_lint_is_overridden(self):
        manifest = tomllib.loads((ROOT / 'magetypes/Cargo.toml').read_text())
        self.assertEqual(manifest['package']['metadata']['cargo-semver-checks'], {
            'lints': {'safe_inherent_method_requires_more_target_features': 'allow'}})
        workspace = tomllib.loads((ROOT / 'Cargo.toml').read_text())
        for section in ('workspace', 'package'):
            self.assertNotIn('cargo-semver-checks', workspace[section].get('metadata', {}))

    def test_distinct_specializations_do_not_cross_match(self):
        self.assertEqual(compare(document(), document()), ([], 2, 2))

    def test_strengthening_either_specialization_fails(self):
        for method_id in ['10', '11']:
            with self.subTest(method=method_id):
                current = document()
                current['index'][method_id]['attrs'][0]['target_feature']['enable'].append('gfni')
                self.assertEqual(len(compare(document(), current)[0]), 1)

    def test_previously_unannotated_method_is_checked(self):
        baseline = document()
        baseline['index']['10']['attrs'] = []
        self.assertEqual(len(compare(baseline, document())[0]), 1)

    def test_removing_requirements_is_allowed(self):
        current = document()
        current['index']['11']['attrs'] = []
        self.assertEqual(compare(document(), current)[0], [])

    def test_missing_method_fails(self):
        current = document()
        current['index']['20']['inner']['impl']['items'] = []
        self.assertEqual(len(compare(document(), current)[0]), 1)

    def test_rustdoc_ids_and_local_path_spelling_are_not_identity(self):
        current = document()
        current['paths']['99'] = current['paths'].pop('0')
        vector = current['index']['20']['inner']['impl']['for']['resolved_path']
        vector['args']['angle_bracketed']['args'][0]['type']['resolved_path'] = {
            'id': '99', 'path': 'different::spelling', 'args': None}
        self.assertEqual(compare(document(), current)[0], [])

    def test_unknown_type_and_duplicate_identity_fail_closed(self):
        current = document()
        del current['paths']['0']
        with self.assertRaises(KeyError):
            safe_methods(current)
        current = document()
        current['index']['22'] = copy.deepcopy(current['index']['20'])
        with self.assertRaises(ValueError):
            safe_methods(current)

    def test_empty_input_is_not_a_pass(self):
        with self.assertRaises(ValueError):
            safe_methods({'index': {}, 'paths': {}})


class SemverArtifactTests(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / 'tmp'
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch, prefix='semver-test-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        cache = self.root / 'semver-checks'
        self.baseline = cache / 'cache/magetypes-baseline.json'
        self.current = self.root / 'doc/magetypes.json'
        manifest = cache / 'local-magetypes-test/Cargo.toml'
        manifest.parent.mkdir(parents=True)
        manifest.write_text('[dependencies.magetypes]\nfeatures = ["avx512"]\n')
        self.metadata = {'packages': [{'name': 'magetypes', 'id': 'magetypes'}],
                         'resolve': {'nodes': [{'id': 'magetypes', 'features': ['avx512']}]}}
        mock = patch('check_semver.subprocess.check_output',
                     return_value=json.dumps(self.metadata))
        self.current_metadata = mock.start()
        self.addCleanup(mock.stop)
        self.baseline.parent.mkdir(parents=True)
        self.baseline_metadata = self.baseline.with_suffix('.metadata.json')
        self.baseline_metadata.write_text(json.dumps(self.metadata))
        for path, version in [(self.baseline, '0.9.30'), (self.current, '0.9.31-beta')]:
            doc = document()
            doc.update(root='200', format_version=60, crate_version=version,
                       target={'triple': 'x86_64-unknown-linux-gnu'})
            doc['index']['200'] = {'name': 'magetypes', 'inner': {'module': {}}}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(doc))

    def check(self):
        return check_artifacts(self.root, '0.9.31-beta', '0.9.30')

    def test_complete_artifacts_pass_and_strengthening_fails(self):
        self.assertFalse(self.check())
        current = json.loads(self.current.read_text())
        current['index']['10']['attrs'][0]['target_feature']['enable'].append('gfni')
        self.current.write_text(json.dumps(current))
        self.assertTrue(self.check())

    def test_missing_or_ambiguous_baseline_fails(self):
        duplicate = self.baseline.with_name('magetypes-duplicate.json')
        duplicate.write_text(self.baseline.read_text())
        with self.assertRaises(ValueError):
            self.check()
        duplicate.unlink()
        self.baseline.unlink()
        with self.assertRaises(ValueError):
            self.check()

    def test_missing_current_fails(self):
        self.current.unlink()
        with self.assertRaises(FileNotFoundError):
            self.check()

    def test_mismatched_target_version_and_format_fail(self):
        current = json.loads(self.current.read_text())
        for key, value in [('target', {'triple': 'aarch64-unknown-linux-gnu'}),
                           ('crate_version', '0.9.29'), ('format_version', 59)]:
            with self.subTest(key=key):
                self.current.write_text(json.dumps(dict(current, **{key: value})))
                with self.assertRaises(ValueError):
                    self.check()
        self.current.write_text(json.dumps(current))
        baseline = json.loads(self.baseline.read_text())
        baseline['crate_version'] = '0.9.29'
        self.baseline.write_text(json.dumps(baseline))
        with self.assertRaises(ValueError):
            self.check()

    def test_mismatched_or_non_avx512_features_fail(self):
        self.metadata['resolve']['nodes'][0]['features'] = []
        self.baseline_metadata.write_text(json.dumps(self.metadata))
        with self.assertRaises(ValueError):
            self.check()
        self.current_metadata.return_value = json.dumps(self.metadata)
        with self.assertRaises(ValueError):
            self.check()

    def test_absent_feature_methods_fail(self):
        for path in (self.baseline, self.current):
            doc = json.loads(path.read_text())
            for method in ('10', '11'):
                doc['index'][method]['attrs'] = []
            path.write_text(json.dumps(doc))
        with self.assertRaises(ValueError):
            self.check()


if __name__ == '__main__':
    unittest.main()

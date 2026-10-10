"""Regression cases for concrete-impl target-feature comparison."""
import copy
import unittest

from check_target_features import compare, safe_methods


def document():
    result = {'paths': {}, 'index': {}}
    for number, (token, features) in enumerate([
            ('V4', ['avx512f']), ('V4x', ['avx512f', 'avx512vbmi'])]):
        type_id, method_id, impl_id = str(number), str(number + 10), str(number + 20)
        result['paths'][type_id] = {'path': ['archmage', token]}
        result['index'][method_id] = {
            'name': 'from_raw', 'visibility': 'public',
            'attrs': [{'target_feature': {'enable': features}}],
            'inner': {'function': {'header': {'is_unsafe': False}}}}
        result['index'][impl_id] = {'inner': {'impl': {
            'trait': None, 'for': {'vector': {'id': type_id, 'path': token}},
            'generics': {}, 'items': [method_id]}}}
    return result


class TargetFeatureTests(unittest.TestCase):
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
        current['index']['20']['inner']['impl']['for']['vector'] = {
            'id': '99', 'path': 'different::spelling'}
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


if __name__ == '__main__':
    unittest.main()

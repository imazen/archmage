#!/usr/bin/env python3
"""Fail closed on incomplete CI, including missing/skipped matrix job groups."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflows():
    return tuple(yaml.load((ROOT / '.github/workflows' / name).read_text(),
                           Loader=yaml.BaseLoader)
                 for name in ('ci.yml', 'publish.yml'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_contract(ci, publish):
    jobs = ci['jobs']
    required = set(jobs) - {'ci-gate'}
    require(set(jobs['ci-gate']['needs']) == required,
            'ci-gate must depend on every other CI job')
    require(jobs['ci-gate']['if'] == '${{ always() }}',
            'ci-gate must run even after upstream failures')
    require(all(j.get('continue-on-error', 'false') == 'false' for j in jobs.values()),
            'CI job failures must not be masked by continue-on-error')
    require(jobs['storage-codegen']['if'] == "github.event_name == 'pull_request'",
            'Only the PR-base comparison may be skipped outside PRs')
    require('workflow_call' in ci['on'], 'CI must remain reusable')
    caller = publish['jobs']['full-ci']
    require(caller['uses'] == './.github/workflows/ci.yml',
            'Release must call CI from the same commit')
    require('if' not in caller and caller.get('needs') == 'check-tags',
            'Full release CI must run after tag verification')
    require(caller.get('continue-on-error', 'false') == 'false',
            'Release must not ignore CI failure')
    publisher = publish['jobs']['publish']
    require(set(publisher['needs']) == {'check-tags', 'pre-publish-check', 'full-ci'},
            'Publishing requires tags, package checks, and the full CI matrix')
    require('if' not in publisher, 'Publishing must retain the default success gate')
    for name in ('check-tags', 'pre-publish-check', 'publish'):
        checkout = next(step for step in publish['jobs'][name]['steps']
                        if step.get('uses', '').startswith('actions/checkout@'))
        require(checkout.get('with', {}).get('ref') == '${{ github.sha }}',
                f'{name} must check out the release commit')
    return required


def check_results(results, required, event):
    require(set(results) == required, 'Missing or unexpected CI job results')
    failures = []
    for name, job in results.items():
        result = job.get('result')
        pr_only_skip = (name == 'storage-codegen' and event != 'pull_request'
                        and result == 'skipped')
        if result != 'success' and not pr_only_skip:
            failures.append(f'{name}: {result}')
    require(not failures, 'CI did not fully pass: ' + ', '.join(failures))


class GateTests(unittest.TestCase):
    def setUp(self):
        self.ci, self.publish = workflows()
        self.required = check_contract(self.ci, self.publish)
        self.results = {name: {'result': 'success'} for name in self.required}

    def test_complete_matrix(self):
        check_results(self.results, self.required, 'pull_request')
        self.results['storage-codegen']['result'] = 'skipped'
        check_results(self.results, self.required, 'release')
        with self.assertRaises(ValueError):
            check_results(self.results, self.required, 'pull_request')

    def test_every_required_job_fails_closed(self):
        for name in self.required:
            for status in ('failure', 'cancelled', 'skipped', None):
                if name == 'storage-codegen' and status == 'skipped':
                    continue
                with self.subTest(job=name, status=status):
                    results = copy.deepcopy(self.results)
                    results[name]['result'] = status
                    with self.assertRaises(ValueError):
                        check_results(results, self.required, 'release')
            results = copy.deepcopy(self.results)
            del results[name]
            with self.assertRaises(ValueError):
                check_results(results, self.required, 'release')

    def test_new_jobs_cannot_escape_gate(self):
        self.ci['jobs']['new-test'] = {'runs-on': 'ubuntu-latest'}
        with self.assertRaises(ValueError):
            check_contract(self.ci, self.publish)

    def test_publish_cannot_bypass_ci(self):
        self.publish['jobs']['publish']['if'] = '${{ always() }}'
        with self.assertRaises(ValueError):
            check_contract(self.ci, self.publish)

    def test_publish_requires_full_matrix(self):
        self.publish['jobs']['publish']['needs'].remove('full-ci')
        with self.assertRaises(ValueError):
            check_contract(self.ci, self.publish)

    def test_test_failures_cannot_be_ignored(self):
        self.ci['jobs']['test-x64']['continue-on-error'] = 'true'
        with self.assertRaises(ValueError):
            check_contract(self.ci, self.publish)

    def test_checkout_cannot_follow_a_moving_ref(self):
        self.publish['jobs']['publish']['steps'][0]['with']['ref'] = 'main'
        with self.assertRaises(ValueError):
            check_contract(self.ci, self.publish)

    def test_release_cannot_use_another_revision(self):
        self.publish['jobs']['full-ci']['uses'] = 'imazen/archmage/.github/workflows/ci.yml@main'
        with self.assertRaises(ValueError):
            check_contract(self.ci, self.publish)


if __name__ == '__main__':
    if sys.argv[1:] == ['--test']:
        unittest.main(argv=[sys.argv[0]])
    else:
        required = check_contract(*workflows())
        check_results(json.loads(os.environ['CI_RESULTS']), required, os.environ['CI_EVENT'])
        print('Every applicable CI job passed.')

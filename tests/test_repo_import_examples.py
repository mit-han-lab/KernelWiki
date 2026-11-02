import unittest
from pathlib import Path
from repo_import_support import ROOT, read


class ExampleTests(unittest.TestCase):
    def test_examples_are_review_materials_without_publication_outputs(self):
        self.assertEqual({}, read(ROOT / 'data/repo-import-presets.yaml')['presets'])
        examples = list((ROOT / 'examples/repo-import').glob('*.yaml'))
        self.assertGreaterEqual(len(examples), 2)
        for path in examples:
            example = read(path)
            self.assertTrue(example['example_only'])
            self.assertTrue(example['snapshot']['repo'].startswith('https://github.com/'))
            self.assertEqual(40, len(example['snapshot']['commit']))
            self.assertTrue(example['decisions'])
            self.assertNotIn('implementations', example)
            self.assertNotIn('publication-plan', example)
        self.assertEqual([], list((ROOT / 'examples/repo-import').rglob('registry.yaml')))
        self.assertEqual([], list((ROOT / 'examples/repo-import').rglob('wiki')))

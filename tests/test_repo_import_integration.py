import shutil
import tempfile
import unittest
from pathlib import Path
from repo_import_support import ROOT, make_repo, import_repo, load_config, read, write, document, commit


class ImportIntegrationTests(unittest.TestCase):
    def test_real_validator_publication_recovery_and_version_update(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            wiki = base / 'wiki'
            shutil.copytree(ROOT, wiki, ignore=shutil.ignore_patterns('.git', '.cache', '__pycache__'))
            repo = make_repo(base)
            config = load_config(wiki)
            run = import_repo.prepare(wiki, str(repo), 'HEAD', config)
            request = read(run / 'review-request.yaml')
            write(run / 'decisions.yaml', document(request))
            import_repo.build(wiki, run)
            import_repo.apply(wiki, run)
            import_repo.apply(wiki, run)
            self.assertEqual('applied', read(run / 'result.yaml')['status'])
            registry = read(wiki / 'data/repo-import/registry.yaml')
            implementation = next(iter(registry['implementations'].values()))
            page = wiki / implementation['page_path']
            self.assertIn('reproducibility: snippet', page.read_text())
            # A commit-only update preserves content and reuses the old admission.
            (repo / 'README.md').write_text('new documentation')
            commit(repo)
            second = import_repo.prepare(wiki, str(repo), 'HEAD', config)
            decisions = read(second / 'decisions.yaml')
            self.assertEqual('record_duplicate', decisions['decisions'][0]['action'])
            import_repo.build(wiki, second)
            # A body update must return to external review, not inherit old metrics.
            (repo / 'kernel.py').write_text((repo / 'kernel.py').read_text().replace('x + n', 'x * n'))
            commit(repo)
            third = import_repo.prepare(wiki, str(repo), 'HEAD', config)
            self.assertEqual('defer', read(third / 'decisions.yaml')['decisions'][0]['decision'])

"""Strategy-owned recall, bounded output, and auditable page reads."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import get_page
import query
from _retrieval_log import RunLog, digest


def page(path, title='', body='', **metadata):
    return {'path': path, 'fm': {'id': path, 'title': title, **metadata}, 'body': body}


class StrategyIndexTests(unittest.TestCase):
    def test_alias_or_recall_without_implicit_score_or_cutoff(self):
        pages = [page(f'a-{i:02d}', body='UMMA') for i in range(40)]
        pages += [page('z', 'tcgen05', tags=['tcgen05']), page('t', tags=['tmem']), page('absent')]
        index = query.StrategyIndex(pages)
        with patch.object(query, 'score_keyword_match', side_effect=AssertionError('implicit scoring')):
            ids = index.search('UMMA tmem')
        self.assertEqual(42, len(ids))
        self.assertEqual('a-00', ids[0])
        self.assertEqual('z', ids[-1])
        self.assertEqual(ids, index.searches[0]['matched_ids'])
        self.assertEqual({'title': ['tcgen05'], 'metadata': ['tcgen05']}, index.matches['z'])

    def test_filters_and_metadata_copy_reuse_base_behavior(self):
        index = query.StrategyIndex([
            page('b', type='hardware', tags=['tcgen05'], architectures=['sm100']),
            page('a', type='hardware', tags=['tcgen05'], architectures=['sm90']),
        ])
        self.assertEqual(['b'], index.search(tag='UMMA', architecture='B200'))
        metadata = index.metadata('b')
        metadata['tags'].clear()
        self.assertEqual(['tcgen05'], index.metadata('b')['tags'])
        for filters in ({'invalid': 'x'}, {'tag': ['tma']}, {'has_code': 'true'}):
            with self.assertRaises(ValueError):
                index.search(**filters)

    def test_full_recall_can_select_late_page_and_preserves_strategy_order(self):
        index = query.StrategyIndex([page(str(i), body='TMA') for i in range(50)])
        source = b'def retrieve(index, context):\n    return list(reversed(index.search("TMA")))\n'
        result = query.execute_strategy(source, 'strategy.py', index, {})
        self.assertEqual(list(reversed(sorted(str(i) for i in range(50)))), [p['path'] for p in result])

    def test_duplicates_and_path_id_references_are_canonicalized(self):
        p = page('wiki/a.md', body='TMA')
        p['fm']['id'] = 'hw-a'
        index = query.StrategyIndex([p])
        self.assertEqual(['wiki/a.md'], [p['path'] for p in index.candidates(['hw-a', 'wiki/a.md'])])
        with self.assertRaises(ValueError):
            index.candidates(['missing'])
        with self.assertRaises(ValueError):
            index.candidates('hw-a')

    def test_body_reads_are_separate_from_search_and_scoring_is_optional(self):
        index = query.StrategyIndex([page('a', 'UMMA tcgen05', body='tcgen05 ' * 100, tags=['tcgen05'])])
        index.search('UMMA')
        self.assertEqual([], index.reads)
        self.assertEqual(18, index.keyword_score('a', 'UMMA'))
        body = index.read('a')
        self.assertEqual(digest(body), index.reads[0]['body_sha256'])

    def test_template_supports_variable_route_counts_without_quotas(self):
        index = query.StrategyIndex([page('a', 'TMA'), page('b', 'TMA'), page('c', 'wait')])
        source = (ROOT / 'scripts/strategy_template.py').read_bytes()
        for context, expected in (({'query': 'TMA'}, ['a', 'b']),
                                  ({'queries': [{'query': 'TMA'}, {'query': 'wait'}]}, ['a', 'c', 'b'])):
            with self.subTest(context=context):
                results = query.execute_strategy(source, 'strategy.py', index, context)
                self.assertEqual(expected, [p['path'] for p in results])

    def test_excerpt_is_bounded_and_filter_only_has_labeled_preview(self):
        body = 'opening ' * 100 + 'mbarrier wait\nconsumer release ' + 'trailing ' * 100
        excerpt = query.match_excerpt(body, ['MBARRIER', 'release'])
        self.assertIn('mbarrier wait consumer release', excerpt)
        self.assertLessEqual(len(excerpt), 242)
        self.assertNotIn('\n', excerpt)
        for compact in (True, False):
            output = query.format_result(page('a', body='No keyword needed', architectures=['sm100']), compact=compact)
            self.assertIn('body preview', output)
            self.assertIn('No keyword needed', output)
            self.assertIn('sm100', output)


class StrategyCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.run = self.directory / 'run'
        self.strategy = self.directory / 'strategy.py'
        self.strategy.write_text('def retrieve(index, context):\n    return index.search(context["query"])\n')
        self.pages = [page(f'wiki/hardware/{i:02d}.md', 'TMA', body='consumer wait', type='hardware') for i in range(40)]

    def cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(query, 'load_all_pages', return_value=self.pages), \
                patch.object(sys, 'argv', ['query.py', *map(str, args)]), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            query.main()
        return stdout.getvalue(), stderr.getvalue()

    def strategy_cli(self, *args):
        return self.cli('--strategy', self.strategy, '--run-dir', self.run, *args)

    def receipts(self):
        return sorted((self.run / 'queries').glob('*/result.json'))

    def test_strategy_has_no_default_count_cap_or_hidden_keyword_score(self):
        with patch.object(query, 'score_keyword_match', side_effect=AssertionError('implicit scoring')):
            stdout, stderr = self.strategy_cli('TMA', '--paths-only')
        self.assertEqual(40, len(stdout.splitlines()))
        self.assertIn('40 proposed', stderr)
        result = json.loads(self.receipts()[0].read_text())
        self.assertIsNone(result['budget']['limit'])
        self.assertEqual(40, len(result['emitted_ids']))

    def test_limit_is_exact_candidate_count_and_plain_query_keeps_master_order(self):
        self.pages[-1]['fm']['tags'] = ['tma']
        stdout, _ = self.cli('TMA', '--limit', 3, '--paths-only')
        self.assertEqual(3, len(stdout.splitlines()))
        self.assertEqual(self.pages[-1]['path'], stdout.splitlines()[0])
        stdout, _ = self.cli('TMA', '--paths-only')
        self.assertEqual(10, len(stdout.splitlines()))
        stdout, _ = self.strategy_cli('TMA', '--limit', 3, '--paths-only')
        self.assertEqual([p['path'] for p in self.pages[:3]], stdout.splitlines())

    def test_budget_covers_summary_and_preserves_order_in_all_modes(self):
        for flags in ([], ['--compact'], ['--paths-only']):
            with self.subTest(flags=flags):
                stdout, stderr = self.strategy_cli('TMA', '--max-chars', 700, *flags)
                self.assertLessEqual(len(stdout) + len(stderr), 700)
        for path in self.receipts():
            result = json.loads(path.read_text())
            self.assertEqual(result['proposed_ids'], result['emitted_ids'] + result['omitted_by_output_limits'])
            self.assertTrue(result['omitted_by_output_limits'])
            self.assertLessEqual(result['output_chars'], 700)

    def test_oversized_first_card_is_disclosed_without_silently_skipping_it(self):
        self.pages[0]['fm']['title'] = 'TMA ' * 1000
        stdout, _ = self.strategy_cli('TMA', '--max-chars', 512)
        self.assertIn('0 candidate(s)', stdout)
        result = json.loads(self.receipts()[0].read_text())
        self.assertEqual(40, len(result['omitted_by_output_limits']))

    def test_global_filters_cannot_be_weakened_by_strategy(self):
        self.pages[0]['fm']['architectures'] = ['sm100']
        self.pages[1]['fm']['architectures'] = ['sm90']
        stdout, _ = self.strategy_cli('TMA', '--architecture', 'B200', '--paths-only')
        self.assertEqual(self.pages[0]['path'] + '\n', stdout)
        self.strategy.write_text('def retrieve(index, context):\n    return ["wiki/hardware/01.md"]\n')
        with self.assertRaises(SystemExit):
            self.strategy_cli('TMA', '--architecture', 'B200')
        self.assertTrue(any(json.loads(p.read_text())['status'] == 'error' for p in self.receipts()))

    def test_strategy_context_and_source_are_snapshotted_per_query(self):
        context = self.directory / 'context.json'
        context.write_text(json.dumps({'query': 'ignored', 'phase': 'profiled', 'target': 'B200'}))
        self.strategy.write_text('def retrieve(index, context):\n    assert context["phase"] == "profiled"\n    return index.search(context["query"])\n')
        source = self.strategy.read_bytes()
        self.strategy_cli('TMA', '--context', context, '--limit', 2)
        folder = self.receipts()[0].parent
        request = json.loads((folder / 'request.json').read_text())
        self.assertEqual(digest(source), request['strategy_sha256'])
        self.assertEqual(source, (folder / 'strategy.py').read_bytes())
        self.assertEqual('TMA', json.loads((folder / 'context.json').read_text())['query'])
        self.assertEqual(digest((ROOT / 'scripts/query.py').read_bytes()), request['engine_sha256']['query.py'])
        self.strategy.write_text('def retrieve(index, context):\n    return []\n')
        self.strategy_cli('TMA')
        self.assertEqual(2, len(self.receipts()))
        self.assertEqual(source, (folder / 'strategy.py').read_bytes())

    def test_searches_selection_omissions_and_strategy_reads_are_distinct(self):
        self.strategy.write_text('def retrieve(index, context):\n    ids = index.search("TMA")\n    index.read(ids[-1])\n    return [ids[-1], ids[0], ids[-1]]\n')
        self.strategy_cli('--limit', 1)
        result = json.loads(self.receipts()[0].read_text())
        self.assertEqual(40, result['searches'][0]['match_count'])
        self.assertEqual([self.pages[-1]['path'], self.pages[0]['path']], result['proposed_ids'])
        self.assertEqual([self.pages[0]['path']], result['omitted_by_output_limits'])
        self.assertEqual(38, len(result['matched_but_not_proposed']))
        self.assertEqual(self.pages[-1]['path'], result['strategy_reads'][0]['path'])
        self.assertFalse((self.run / 'reads').exists())

    def test_diagnostics_do_not_escape_the_output_budget(self):
        self.strategy.write_text('print("x" * 10000)\ndef retrieve(index, context):\n    return index.search("TMA")\n')
        stdout, stderr = self.strategy_cli('--max-chars', 700, '--compact')
        self.assertLessEqual(len(stdout) + len(stderr), 700)
        folder = self.receipts()[0].parent
        self.assertEqual(4096, len((folder / 'strategy-output.txt').read_text()))
        self.assertEqual(10001, json.loads((folder / 'result.json').read_text())['strategy_diagnostic_chars'])

    def test_invalid_and_failing_strategies_have_receipts_not_fallback_results(self):
        for source in ('x = 1', 'def retrieve(index, context):\n    return ["missing"]',
                       'def retrieve(index, context):\n    raise RuntimeError("broken")',
                       'def retrieve(index, context):\n    return "TMA"', 'syntax error!'):
            self.strategy.write_text(source)
            with self.subTest(source=source), self.assertRaises(SystemExit) as raised:
                self.strategy_cli('TMA')
            self.assertEqual(2, raised.exception.code)
        for path in self.receipts():
            self.assertEqual('error', json.loads(path.read_text())['status'])
            self.assertEqual('', (path.parent / 'output.txt').read_text())

    def test_invalid_arguments_and_context_are_rejected(self):
        for args in (('--limit', 0), ('--limit', -1), ('--max-chars', 511),
                     ('--offset', 0), ('--match', 'all'), ('--strategy', self.strategy)):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.cli(*args)
        context = self.directory / 'context.json'
        context.write_text('[]')
        with self.assertRaises(SystemExit):
            self.strategy_cli('--context', context)

    def test_filter_only_and_empty_results_remain_usable(self):
        stdout, _ = self.cli('--type', 'hardware', '--limit', 2, '--paths-only')
        self.assertEqual([p['path'] for p in self.pages[:2]], stdout.splitlines())
        stdout, _ = self.strategy_cli('nonexistentidentifier')
        self.assertIn('0 candidate(s); 0 proposed', stdout)


class ReadReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'wiki'
        self.root.mkdir()
        (self.root / 'wiki').mkdir()
        (self.root / 'sources').mkdir()
        self.page = self.root / 'wiki/a.md'
        self.page.write_text('---\nid: hw-a\nsources: [doc-a]\nartifact_dir: artifacts/a\n---\nPage body\n')
        (self.root / 'sources/a.md').write_text('---\nid: doc-a\n---\n' + 'source ' * 200)
        bundle = self.root / 'artifacts/a'
        bundle.mkdir(parents=True)
        (bundle / 'kernel.cu').write_text('x' * (210 * 1024))
        self.run = Path(self.tmp.name) / 'run'

    def cli(self, *args):
        stdout = io.StringIO()
        with patch.object(get_page, 'WIKI_ROOT', self.root), \
                patch.object(sys, 'argv', ['get_page.py', 'hw-a', *map(str, args)]), \
                contextlib.redirect_stdout(stdout):
            get_page.main()
        return stdout.getvalue()

    def test_logging_preserves_output_and_distinguishes_source_and_code_previews(self):
        original = self.cli('--follow-sources', '--include-code')
        logged = self.cli('--follow-sources', '--include-code', '--run-dir', self.run)
        self.assertEqual(original, logged)
        receipt_path = next((self.run / 'reads').glob('*/receipt.json'))
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(['full', 'source-preview', 'artifact-prefix'], [r['view'] for r in receipt['files']])
        self.assertEqual(500, receipt['files'][1]['shown_chars'])
        self.assertEqual(200 * 1024, receipt['files'][2]['shown_chars'])
        self.assertEqual(digest(original), receipt['output_sha256'])
        self.assertEqual(original, (receipt_path.parent / 'output.txt').read_text())

    def test_partial_modes_are_not_recorded_as_full_page_reads(self):
        for flag, expected in (('--body-only', 'body'), ('--frontmatter-only', 'frontmatter')):
            self.cli(flag, '--run-dir', self.run)
        views = {json.loads(p.read_text())['files'][0]['view'] for p in (self.run / 'reads').glob('*/receipt.json')}
        self.assertEqual({'body', 'frontmatter'}, views)

    def test_read_can_be_linked_to_exact_query_and_unknown_ids_fail(self):
        log = RunLog(self.run)
        query_id, _ = log.start_query(ROOT, {}, {})
        self.cli('--body-only', '--run-dir', self.run, '--query-id', query_id)
        receipt = json.loads(next((self.run / 'reads').glob('*/receipt.json')).read_text())
        self.assertEqual(query_id, receipt['query_id'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.cli('--run-dir', self.run, '--query-id', '0' * 32)
        self.assertEqual(1, len(list((self.run / 'reads').glob('*/receipt.json'))))

    def test_query_receipt_separates_engine_and_overridden_corpus(self):
        (self.root / 'data').mkdir()
        aliases = b'tcgen05: [UMMA]\n'
        (self.root / 'data/aliases.yaml').write_bytes(aliases)
        _, directory = RunLog(self.run).start_query(self.root, {}, {})
        request = json.loads((directory / 'request.json').read_text())
        self.assertEqual(str(ROOT), request['repository']['root'])
        self.assertEqual(str(self.root), request['corpus_repository']['root'])
        self.assertEqual(digest(aliases), request['aliases_sha256'])
        self.assertEqual(aliases, (directory / 'aliases.yaml').read_bytes())
        self.assertEqual((ROOT / 'scripts/query.py').read_bytes(), (directory / 'engine/query.py').read_bytes())


if __name__ == '__main__':
    unittest.main()

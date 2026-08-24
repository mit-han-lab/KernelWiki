import importlib.util
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location("validate_contracts", SCRIPTS / "validate.py")
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)


@contextmanager
def validation_root(root):
    names = ("REPO_ROOT", "SOURCES_DIR", "WIKI_DIR", "DATA_DIR", "ARTIFACTS_DIR", "CANDIDATES_DIR")
    previous = {name: getattr(validate, name) for name in names}
    validate.REPO_ROOT = root
    validate.SOURCES_DIR = root / "sources"
    validate.WIKI_DIR = root / "wiki"
    validate.DATA_DIR = root / "data"
    validate.ARTIFACTS_DIR = root / "artifacts"
    validate.CANDIDATES_DIR = root / "candidates"
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(validate, name, value)


class ReproducibilityContractTests(unittest.TestCase):
    def validate_page(self, reproducibility):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / "wiki" / "techniques" / "example.md"
            page.parent.mkdir(parents=True)
            page.write_text(
                "---\n"
                "id: technique-example\n"
                "type: technique\n"
                f"reproducibility: {reproducibility}\n"
                "---\n\n"
                "# Evidence-bounded concept\n\nNo implementation is claimed.\n",
                encoding="utf-8",
            )
            schemas = {"wiki-technique": {"required": [], "constraints": {"reproducibility_minimum": "concept"}}}
            tags = {"reproducibility": ["concept", "pseudocode", "snippet", "runnable", "benchmarked"]}
            with validation_root(root):
                return validate.validate_file(page, schemas, tags, set(), {"python", "cuda"})

    def test_concept_page_is_not_forced_to_invent_code(self):
        errors = self.validate_page("concept")
        self.assertFalse(any("fenced code block" in error for error in errors), errors)

    def test_snippet_claim_still_requires_real_code(self):
        errors = self.validate_page("snippet")
        self.assertTrue(any("fenced code block" in error for error in errors), errors)

    def test_typed_cpp_builder_fragment_counts_as_code(self):
        body = """```cpp
using Layout = decltype(Config::deduce_layout());
using Mainloop = typename Builder<Layout>::CollectiveOp;
auto args = Mainloop::to_underlying_arguments(problem);
```"""
        self.assertTrue(validate.has_compilable_code(body, {"cpp"}))

    def test_shell_build_comparison_counts_as_code(self):
        body = """```bash
nvcc -arch=sm_100a kernel.cu -o default
nvcc -arch=sm_100a --maxrregcount=80 kernel.cu -o r80
nvcc -arch=sm_100a --maxrregcount=64 kernel.cu -o r64
```"""
        self.assertTrue(validate.has_compilable_code(body, {"bash"}))

    def test_exact_two_line_api_usage_counts_as_code(self):
        body = """```python
from flash_attn.cute import flash_attn_func
out = flash_attn_func(q, k, v, causal=True)
```"""
        self.assertTrue(validate.has_compilable_code(body, {"python"}))

    def test_python_control_flow_excerpt_counts_as_code(self):
        body = """```python
if warp_idx < softmax_warp:
    softmax_loop(stage=0)
if warp_idx >= softmax_warp:
    softmax_loop(stage=1)
```"""
        self.assertTrue(validate.has_compilable_code(body, {"python"}))

    def test_one_line_stub_does_not_count_as_code(self):
        body = """```python
out = implementation()
```"""
        self.assertFalse(validate.has_compilable_code(body, {"python"}))


class CaptureCutoffContractTests(unittest.TestCase):
    def run_cutoff(self, captured_at):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            prs = root / "sources" / "prs" / "example"
            data.mkdir(parents=True)
            prs.mkdir(parents=True)
            data.joinpath("refresh-cutoff.yaml").write_text(
                "cutoff_date: 2026-08-01\nprevious_pages_manifest: []\n",
                encoding="utf-8",
            )
            prs.joinpath("PR-1.md").write_text(
                f"---\nid: pr-example-1\ncaptured_at: {captured_at}\n---\n",
                encoding="utf-8",
            )
            with validation_root(root):
                return validate.validate_captured_at_cutoff()

    def test_post_cutoff_capture_is_valid_fresh_evidence(self):
        self.assertEqual([], self.run_cutoff("2026-08-18"))

    def test_pre_cutoff_capture_is_rejected_for_new_page(self):
        errors = self.run_cutoff("2026-07-31")
        self.assertEqual(1, len(errors))
        self.assertIn("precedes cutoff_date", errors[0])


class RepositoryCountContractTests(unittest.TestCase):
    def test_count_gate_covers_repositories_outside_old_hardcoded_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prs = root / "sources" / "prs" / "cccl"
            refs = root / "references"
            prs.mkdir(parents=True)
            refs.mkdir(parents=True)
            for number in (1, 2):
                prs.joinpath(f"PR-{number}.md").write_text(
                    f"---\nid: pr-cccl-{number}\nrepo: NVIDIA/cccl\n---\n",
                    encoding="utf-8",
                )
            refs.joinpath("primer.md").write_text(
                "## Source Repositories (PR coverage)\n\n"
                "| Repo | PR pages | Ledger |\n|---|---:|---|\n"
                "| NVIDIA/cccl | 3 | ledger |\n",
                encoding="utf-8",
            )
            with validation_root(root):
                errors = validate.validate_discoverability()
            self.assertTrue(any("NVIDIA/cccl" in error and "2 exist" in error for error in errors))


class ChangedFileInventoryContractTests(unittest.TestCase):
    def test_capped_listing_requires_honest_total_and_completeness(self):
        valid = {
            "changed_files_count": 3750,
            "changed_files_enumerated_count": 3000,
            "changed_files_listing_complete": False,
            "changed_paths": ["kernel.cu"],
            "changed_paths_complete": False,
        }
        self.assertEqual([], validate.changed_file_inventory_errors(valid))
        invalid = dict(valid, changed_files_count=3000)
        errors = validate.changed_file_inventory_errors(invalid)
        self.assertTrue(any("listing_complete" in error for error in errors), errors)

    def test_reconstructed_evidence_requires_full_count_and_receipt(self):
        valid = {
            "changed_files_count": 3298,
            "changed_files_enumerated_count": 3000,
            "changed_files_listing_complete": False,
            "changed_files_evidence_count": 3298,
            "changed_files_evidence_complete": True,
            "changed_files_evidence_method": "github-pull-diff",
            "changed_files_evidence_receipt": "audit/pr-file-cap-reconstruction.json",
            "changed_paths": ["src/tail_kernel.cu"],
            "changed_paths_complete": False,
        }
        self.assertEqual([], validate.changed_file_inventory_errors(valid))
        invalid = {**valid, "changed_files_evidence_count": 3000}
        errors = validate.changed_file_inventory_errors(invalid)
        self.assertTrue(any("evidence_complete" in error for error in errors), errors)

    def test_reconstructed_page_must_match_policy_receipt(self):
        fm = {
            "repo": "example/project",
            "pr": 9,
            "scope_disposition": "retained",
            "scope_evidence": {
                "rule": "device-code-signal",
                "paths": ["src/tail.h"],
                "path_source": "github-pull-diff",
            },
            "changed_files_count": 3001,
            "changed_files_enumerated_count": 3000,
            "changed_files_listing_complete": False,
            "changed_files_evidence_count": 3001,
            "changed_files_evidence_complete": True,
            "changed_files_evidence_method": "github-pull-diff",
            "changed_files_evidence_receipt": "audit/pr-file-cap-reconstruction.json",
            "upstream_files_sha256": "a" * 64,
        }
        row = {
            "authoritative_changed_files": 3001,
            "files_api_enumerated": 3000,
            "enumeration_complete": False,
            "full_diff_paths": 3001,
            "full_diff_complete": True,
            "policy_files_sha256": "a" * 64,
            "full_policy": {
                "disposition": "retained",
                "rule": "device-code-signal",
                "evidence_paths": ["src/tail.h"],
            },
        }
        self.assertEqual([], validate.cap_reconstruction_page_errors(fm, row))
        errors = validate.cap_reconstruction_page_errors(
            {**fm, "upstream_files_sha256": "b" * 64}, row
        )
        self.assertTrue(any("upstream_files_sha256" in error for error in errors), errors)

    def test_complete_cuda_receipt_binds_content_and_policy_pattern(self):
        row = {
            "complete_file_evidence": [{
                "path": "src/kernel.cu",
                "sha256": "a" * 64,
                "device_signal": True,
                "device_pattern_sha256": validate.device_code_pattern_sha256(),
            }],
        }
        self.assertEqual([], validate.cap_complete_file_evidence_errors(row))
        stale = {
            "complete_file_evidence": [
                {**row["complete_file_evidence"][0], "device_pattern_sha256": "b" * 64}
            ],
        }
        errors = validate.cap_complete_file_evidence_errors(stale)
        self.assertTrue(any("stale" in error for error in errors), errors)


class SourceAttributionContractTests(unittest.TestCase):
    def test_tcgen05_tutorial_uses_source_supported_author_identity(self):
        source = (ROOT / "sources/blogs/tcgen05-tutorial.md").read_text(encoding="utf-8")
        pipeline = (ROOT / "wiki/techniques/pipeline-stages.md").read_text(encoding="utf-8")
        persistent = (ROOT / "wiki/techniques/persistent-kernels.md").read_text(encoding="utf-8")

        self.assertIn("author: Thien Tran (gau-nernst)", source)
        for text in (source, pipeline, persistent):
            self.assertNotIn("Gau Nernst", text)
        self.assertIn("Thien Tran (`gau-nernst`)", pipeline)
        self.assertIn("Thien Tran (`gau-nernst`)", persistent)

    def test_flash_attention_blog_credits_all_six_authors(self):
        source = (ROOT / "sources/blogs/flash-attention-4.md").read_text(encoding="utf-8")
        kernel = (ROOT / "wiki/kernels/flash-attention-4.md").read_text(encoding="utf-8")
        examples = (ROOT / "references/examples.md").read_text(encoding="utf-8")
        authors = (
            "Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, "
            "and Tri Dao"
        )

        self.assertIn(f"author: {authors}", source)
        self.assertIn("The FlashAttention-4 authors' article", source)
        for text in (source, kernel, examples):
            self.assertNotIn("Tri Dao's", text)
        self.assertIn("FlashAttention-4 author blog", kernel)
        self.assertIn("(author blog)", examples)

    def test_qwen3_next_uses_the_article_byline_not_model_organizations(self):
        source = (ROOT / "sources/blogs/qwen3-next-architecture.md").read_text(encoding="utf-8")

        self.assertIn("author: Anu Srivastava (NVIDIA)", source)
        self.assertNotIn("author: NVIDIA / Alibaba", source)

    def test_modular_blackwell_post_credits_its_four_person_byline(self):
        source = (ROOT / "sources/blogs/modular-blackwell-matmul.md").read_text(encoding="utf-8")

        self.assertIn(
            "author: Ali Taha, Jiexiang Liu, Hengjie Wang, and Abdul Dakkak (Modular)",
            source,
        )
        self.assertNotIn("author: Modular\n", source)

    def test_nsa_paper_credits_people_and_all_three_institutions(self):
        source = (ROOT / "sources/docs/nsa.md").read_text(encoding="utf-8")
        kernel = (ROOT / "wiki/kernels/nsa.md").read_text(encoding="utf-8")

        self.assertIn(
            "author: Jingyang Yuan et al. (DeepSeek-AI, Peking University, University of Washington)",
            source,
        )
        for text in (source, kernel):
            self.assertIn("DeepSeek-led", text)
            self.assertNotIn("DeepSeek's", text)

    def test_hugging_face_community_post_uses_rendered_author_identity(self):
        source = (ROOT / "sources/blogs/tflops-gap-fp4-moe.md").read_text(encoding="utf-8")

        self.assertIn("author: Konstantin (apsys)", source)
        self.assertIn("This Hugging Face community benchmark", source)
        self.assertNotIn("author: apsys (HuggingFace)", source)

    def test_cutlass_changelog_rows_preserve_official_label_dates(self):
        source = (ROOT / "sources/docs/cutlass-changelog-sm100.md").read_text(encoding="utf-8")

        self.assertIn("| 4.6.2 | 2026-08-08 | 2026-08-03 (rendered changelog) |", source)
        self.assertIn("| 4.6.1 | 2026-07-15 | 2026-07-13 |", source)
        self.assertIn("no\n4.6.2 section", source)
        self.assertNotIn("| 4.6.2 | 2026-08-08 | patch release |", source)
        self.assertNotIn("| 4.6.1 | 2026-07-15 | patch release |", source)

    def test_jax_tutorial_uses_its_rendered_author_credit(self):
        source = (ROOT / "sources/blogs/jax-pallas-blackwell-matmul.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("author: The JAX authors", source)
        self.assertIn("credited by the page to The JAX authors", source)
        self.assertNotIn("author: JAX Team (Google)", source)

    def test_reward_hack_post_credits_participant_and_scopes_platform_as_host(self):
        source = (ROOT / "sources/blogs/gpu-mode-reward-hack.md").read_text(
            encoding="utf-8"
        )
        contest = (
            ROOT / "sources/contests/gpu-mode-nvfp4/problem-4-grouped-gemm.md"
        ).read_text(encoding="utf-8")

        self.assertIn("author: Natalia Kokoromyti\n", source)
        self.assertIn(
            "GPU Mode-hosted post-mortem by contest participant Natalia Kokoromyti",
            " ".join(source.split()),
        )
        self.assertIn("describes the author's agent-produced submission", source)
        self.assertIn(
            "participant post-mortem by Natalia Kokoromyti, published on GPU Mode's news page",
            contest,
        )
        research = (ROOT / "research-contests.md").read_text(encoding="utf-8")
        self.assertIn(
            "participant post-mortem published on GPU Mode's news page", research
        )

        shipped = list((ROOT / "sources").rglob("*.md"))
        shipped.extend(ROOT.glob("research-*.md"))
        consumers = []
        for path in shipped:
            text = path.read_text(encoding="utf-8")
            if "post-mortem" in text.lower():
                consumers.append((path, text))
        self.assertGreaterEqual(len(consumers), 3)

        gendered_pronoun = re.compile(
            r"\b(?:she|her|hers|herself|he|him|his|himself)\b", re.IGNORECASE
        )

        def named_sentences_with_gendered_pronouns(text):
            sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
            return [
                sentence
                for sentence in sentences
                if "Natalia Kokoromyti" in sentence
                and gendered_pronoun.search(sentence)
            ]

        for path, text in consumers:
            self.assertNotIn("author: Natalia Kokoromyti / GPU Mode", text, path)
            self.assertNotIn("organizer's reward-hacking post-mortem", text, path)
            self.assertNotIn("GPU Mode's post-mortem", text, path)
            self.assertNotIn("GPU Mode's separate reward-hacking post-mortem", text, path)
            self.assertEqual([], named_sentences_with_gendered_pronouns(text), path)

        adversarial = source.replace(
            "the author's agent-produced submission", "her agent-produced submission"
        )
        self.assertTrue(named_sentences_with_gendered_pronouns(adversarial))

    def test_size_budget_matches_core_manifest_and_artifact_scope(self):
        budget = validate.yaml.safe_load(
            (ROOT / "data/phase3-size-budget.yaml").read_text(encoding="utf-8")
        )
        core = validate.yaml.safe_load(
            (ROOT / "data/core-prs.yaml").read_text(encoding="utf-8")
        )
        checker = (ROOT / "scripts/repo_size_check.py").read_text(encoding="utf-8")

        self.assertEqual(core["total_captured"], budget["core_prs_total"])
        self.assertEqual(core["checksum_sha256"], budget["core_prs_checksum_sha256"])
        self.assertEqual(6000, budget["artifacts_file_budget"])
        self.assertIn("FILE_COUNT_BUDGET = 6000", checker)
        self.assertNotIn("pilot_entries", budget)

    def test_skill_load_memo_scopes_file_budget_to_artifacts(self):
        memo = (ROOT / "data/phase3-skill-load.md").read_text(encoding="utf-8")

        self.assertIn("6,000-file budget applies only to files under `artifacts/`", memo)
        self.assertIn("not enforce a whole-working-tree file-count budget", memo)
        self.assertNotIn("5002 files of headroom", memo)


class ShippedDocumentationContractTests(unittest.TestCase):
    @staticmethod
    def frontmatter(path):
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            text = text.split("---", 2)[1]
        return validate.yaml.safe_load(text)

    def test_inclusion_policy_rejects_obsolete_triton_claim_in_comments(self):
        self.assertEqual([], validate.validate_inclusion_policy_scalars())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir(parents=True)
            data.joinpath("inclusion-policy.yaml").write_text(
                "## Triton on SM100 has no direct tcgen05/TMEM access\n"
                "triton:\n  description: Triton 3.6 supports SM100 lowering.\n",
                encoding="utf-8",
            )
            with validation_root(root):
                errors = validate.validate_inclusion_policy_scalars()
        self.assertTrue(any("including comments" in error for error in errors), errors)

    def test_examples_triton_navigation_matches_current_page_structure(self):
        examples = (ROOT / "references/examples.md").read_text(encoding="utf-8")
        triton = (ROOT / "wiki/languages/triton-blackwell.md").read_text(encoding="utf-8")
        headings = [line for line in triton.splitlines() if line.startswith("#")]

        self.assertIn("its opening paragraph records", examples)
        self.assertIn("earlier blanket statement", triton)
        self.assertNotIn("pre-3.6 historical context preserved in a clearly-marked subsection", examples)
        self.assertFalse(any("Pre-3.6 historical context" in heading for heading in headings))

    @staticmethod
    def yaml_block_after(path, marker):
        text = path.read_text(encoding="utf-8")
        fenced = text.split(marker, 1)[1].split("```yaml", 1)[1].split("```", 1)[0]
        return validate.yaml.safe_load(fenced)

    def test_extended_and_condensed_performance_examples_match(self):
        extended = self.yaml_block_after(ROOT / "CLAUDE.md", "## Performance Claim Format")
        condensed = self.yaml_block_after(ROOT / "references/schema.md", "### wiki-kernel")
        extended_claim = extended["performance_claims"][0]
        condensed_claim = condensed["performance_claims"][0]

        self.assertEqual(condensed_claim, extended_claim)
        self.assertEqual(1613, extended_claim["value"])
        self.assertEqual("doc-flash-attention-4", extended_claim["source_id"])
        self.assertIn("exact maximizing shape not stated", extended_claim["shape"])

    def test_source_pr_schema_example_matches_its_named_page(self):
        example = self.yaml_block_after(ROOT / "references/schema.md", "### source-pr")
        actual = self.frontmatter(ROOT / "sources/prs/cutlass/PR-2472.md")

        self.assertEqual(actual, example)

    def test_numeric_receipt_totals_are_recomputed_and_synchronized(self):
        ledger = (ROOT / "audit/numeric-claims-ledger.md").read_text(encoding="utf-8")
        statuses = []
        for line in ledger.splitlines():
            if line.startswith("| comparison-base |") or line.startswith("| current |"):
                statuses.append(line.split("|")[4].strip())
        counts = {status: statuses.count(status) for status in set(statuses)}

        self.assertEqual(3812, len(statuses))
        self.assertEqual(
            {"supported": 1121, "corrected": 2, "removed": 1946, "non-material": 743},
            counts,
        )
        for path in (
            ROOT / "audit/factual-errors-fixed.md",
            ROOT / "audit/evidence-ledger.md",
            ROOT / "audit/validation-results.md",
        ):
            text = path.read_text(encoding="utf-8")
            self.assertIn("3,812", text)
            self.assertNotIn("3,806 current terminal", text)

    def test_concrete_documented_query_commands_return_results(self):
        docs = [ROOT / "README.md", ROOT / "SKILL.md", ROOT / "CLAUDE.md", ROOT / "index.md"]
        docs.extend((ROOT / "references").glob("*.md"))
        commands = []
        for path in docs:
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip().strip("`")
                if not re.match(r"python3 scripts/(?:grep_wiki|query|get_page)\.py\b", stripped):
                    continue
                if "<" in stripped or "[--" in stripped:
                    continue
                commands.append((path, stripped))

        self.assertGreaterEqual(len(commands), 25)
        for path, command in commands:
            result = subprocess.run(
                shlex.split(command, comments=True),
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=20,
            )
            output = result.stdout + result.stderr
            self.assertEqual(0, result.returncode, (path, command, output))
            self.assertTrue(output.strip(), (path, command))
            self.assertNotIn("No matches.", output, (path, command, output))

    def test_current_audit_receipts_match_discovered_tests_and_bundles(self):
        discovered_tests = sum(
            len(re.findall(r"^\s+def test_", path.read_text(encoding="utf-8"), re.MULTILINE))
            for path in (ROOT / "tests").glob("test_*.py")
        )
        bundles = len(list((ROOT / "artifacts").rglob("PROVENANCE.yaml")))

        self.assertEqual(158, discovered_tests)
        for path in (ROOT / "audit/regression-tests.md", ROOT / "audit/validation-results.md"):
            self.assertIn(f"{discovered_tests} tests", path.read_text(encoding="utf-8"))
        for path in (
            ROOT / "data/phase3-verify-verbatim-audit.md",
            ROOT / "audit/evidence-ledger.md",
            ROOT / "audit/validation-results.md",
        ):
            self.assertIn(f"{bundles} bundle", path.read_text(encoding="utf-8"))

        receipt = (ROOT / "data/phase3-verify-verbatim-audit.md").read_text(encoding="utf-8")
        self.assertIn("Historical stdout", receipt)
        self.assertIn("count follows the current corpus", receipt)
        self.assertNotIn("Verified 76 bundle(s)", receipt)

    def test_skill_technique_count_matches_generated_index_and_pages(self):
        index_count = sum(
            line.startswith("| [")
            for line in (ROOT / "queries/by-technique.md").read_text(encoding="utf-8").splitlines()
        )
        page_count = len(list((ROOT / "wiki/techniques").glob("*.md")))
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

        self.assertEqual(page_count, index_count)
        self.assertIn(f"— {index_count} techniques with architectures", skill)

    def test_nvfp4_primer_row_matches_retained_workload_scale_contract(self):
        primer = (ROOT / "references/primer.md").read_text(encoding="utf-8")
        row = next(line for line in primer.splitlines() if line.startswith("| NVFP4 GEMM |"))
        kernel = self.frontmatter(ROOT / "wiki/kernels/nvfp4-gemm.md")

        self.assertEqual([], kernel["performance_claims"])
        self.assertIn("| — |", row)
        self.assertIn("per-16 FP8 scales", row)
        self.assertIn("task prose: E4M3FNUZ", row)
        self.assertIn("reference code: `torch.float8_e4m3fn`", row)
        self.assertIn("PTX `UE4M3` is distinct", row)
        self.assertIn("MXFP4's block-32 `UE8M0`", row)

    def test_nvfp4_scale_encoding_discrepancy_is_disclosed(self):
        paths = (
            ROOT / "wiki/hardware/nvfp4.md",
            ROOT / "wiki/kernels/nvfp4-gemm.md",
            ROOT / "wiki/kernels/nvfp4-gemv.md",
            ROOT / "wiki/techniques/fine-grained-quantization.md",
            ROOT / "sources/contests/gpu-mode-nvfp4/problem-1-gemv.md",
            ROOT / "sources/contests/gpu-mode-nvfp4/problem-2-gemm.md",
            ROOT / "references/primer.md",
        )
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn("E4M3FNUZ", text.upper(), path)
            self.assertIn("float8_e4m3fn", text, path)
        self.assertNotIn("E4M3FNUZ scales (PTX `UE4M3`)", combined)
        self.assertNotIn("E4M3 (PTX `UE4M3`)", combined)

    def test_nvfp4_contest_input_tuple_has_no_global_scale_operand(self):
        hardware = (ROOT / "wiki/hardware/nvfp4.md").read_text(encoding="utf-8")
        gemm = (ROOT / "wiki/kernels/nvfp4-gemm.md").read_text(encoding="utf-8")
        gemv = (ROOT / "wiki/kernels/nvfp4-gemv.md").read_text(encoding="utf-8")
        contests = (
            ROOT / "sources/contests/gpu-mode-nvfp4/problem-1-gemv.md",
            ROOT / "sources/contests/gpu-mode-nvfp4/problem-2-gemm.md",
        )

        hardware_flat = " ".join(hardware.split())
        gemm_flat = " ".join(gemm.split())
        gemv_flat = " ".join(gemv.split())
        self.assertIn("do not expose a separate tensor-level scale input", hardware_flat)
        self.assertIn("per-16 FP8 scales, and FP16 output", gemv_flat)
        self.assertIn("no separate tensor-level scale operand", gemv_flat)
        self.assertIn("per-16 FP8 scale tensors", gemm_flat)
        self.assertNotIn("tensor-level scaling", gemv)
        self.assertNotIn("tensor scales", gemm)
        for path in contests:
            text = path.read_text(encoding="utf-8")
            self.assertIn("one FP8 scale per 16 values and FP16 output", text, path)
            self.assertNotIn("tensor-level scale", text, path)
            self.assertNotIn("global scale", text, path)

    def test_contest_navigation_does_not_promise_removed_submissions(self):
        contest_pages = list((ROOT / "sources/contests").rglob("*.md"))
        declared = [path for path in contest_pages if "submissions" in self.frontmatter(path)]
        primer = (ROOT / "references/primer.md").read_text(encoding="utf-8")
        examples = (ROOT / "references/examples.md").read_text(encoding="utf-8")
        claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

        self.assertEqual([], declared)
        self.assertNotIn("has a `submissions:` block", primer)
        self.assertNotIn("has `submissions:`", examples)
        self.assertNotIn("Competition problems and solutions", claude)
        self.assertNotIn("FlashInfer MLSys 2026 submissions", skill)

    def test_skill_pr_status_distribution_matches_pages(self):
        statuses = [
            self.frontmatter(path)["status"]
            for path in (ROOT / "sources/prs").rglob("*.md")
        ]
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        schema = (ROOT / "references/schema.md").read_text(encoding="utf-8")
        merged = statuses.count("merged")
        closed = statuses.count("closed")

        self.assertIn(
            f"current distribution: {merged} merged, {closed} closed without merge",
            skill,
        )
        self.assertIn(
            f"({merged} of {len(statuses)} are `merged`; two are `closed` without merge)",
            schema,
        )

    def test_shipped_artifact_mode_description_matches_provenance(self):
        modes = {
            self.frontmatter(path)["asset_mode"]
            for path in (ROOT / "artifacts").rglob("PROVENANCE.yaml")
        }
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

        self.assertEqual({"verbatim"}, modes)
        for text in (readme, skill):
            self.assertIn("Verbatim upstream asset bundles", text)
            self.assertNotIn("Verbatim/extracted/derived asset bundles", text)

    def test_gemv_progression_points_to_technique_page(self):
        examples = (ROOT / "references/examples.md").read_text(encoding="utf-8")
        technique = (ROOT / "wiki/techniques/vectorized-loads.md").read_text(encoding="utf-8")

        self.assertIn("2000 → 443 → 39 → 27 → 22.392 μs", examples)
        self.assertIn("`wiki/techniques/vectorized-loads.md`", examples)
        for value in ("2000", "443", "39", "27", "22.392"):
            self.assertIn(value, technique)

    def test_flash_attention_locator_and_source_ordering_are_evidence_bounded(self):
        kernel = (ROOT / "wiki/kernels/flash-attention-4.md").read_text(encoding="utf-8")
        blog = (ROOT / "sources/blogs/flash-attention-4.md").read_text(encoding="utf-8")

        self.assertIn("#S5 (§5 Empirical Evaluation)", kernel)
        self.assertNotIn("Performance Evaluation", kernel)
        self.assertNotIn("earlier result set", kernel)
        self.assertNotIn("later paper", blog.lower())

    def test_pipeline_percentage_receipt_names_reproducing_rows(self):
        pipeline = (ROOT / "wiki/techniques/pipeline-stages.md").read_text(encoding="utf-8")
        ledger = (ROOT / "audit/numeric-claims-ledger.md").read_text(encoding="utf-8")
        receipt = next(
            line for line in ledger.splitlines()
            if "`wiki/techniques/pipeline-stages.md:21` | `35%`" in line
        )

        self.assertIn("v2b and v3 rows (939.61 / 695.43 = 1.35)", receipt)
        self.assertIn("695.43 TFLOPS", pipeline)
        self.assertIn("939.61 TFLOPS", pipeline)
        self.assertEqual(35, round((939.61 / 695.43 - 1) * 100))

    def test_shipped_id_link_labels_resolve_to_frontmatter(self):
        ids = {
            self.frontmatter(path)["id"]
            for root in (ROOT / "sources", ROOT / "wiki")
            for path in root.rglob("*.md")
        }
        docs = [ROOT / "README.md", ROOT / "index.md", ROOT / "CLAUDE.md", ROOT / "SKILL.md"]
        docs.extend((ROOT / "references").glob("*.md"))
        prefixes = ("pr-", "doc-", "blog-", "contest-", "hw-", "technique-", "kernel-", "pattern-", "lang-", "migration-")
        labels = {
            label
            for path in docs
            for label in re.findall(r"\[([a-z][a-z0-9-]+)\]\(", path.read_text(encoding="utf-8"))
            if label.startswith(prefixes)
        }

        self.assertEqual(set(), labels - ids)

    def test_source_directory_categories_match_shipped_description(self):
        doc_categories = {
            self.frontmatter(path)["source_category"]
            for path in (ROOT / "sources/docs").glob("*.md")
        }
        blog_categories = {
            self.frontmatter(path)["source_category"]
            for path in (ROOT / "sources/blogs").glob("*.md")
        }
        claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")

        self.assertEqual({"official-doc", "paper"}, doc_categories)
        self.assertEqual({"benchmark-blog", "community-note"}, blog_categories)
        self.assertIn("`source_category: official-doc` or `paper`", claude)

    def test_arxiv_sources_require_paper_contract(self):
        bad = validate.arxiv_source_classification_errors(
            ROOT / "sources/blogs/example.md",
            {"url": "https://arxiv.org/abs/1234.5678", "source_category": "benchmark-blog"},
            "source-blog",
        )
        good = validate.arxiv_source_classification_errors(
            ROOT / "sources/docs/example.md",
            {"url": "https://arxiv.org/abs/1234.5678", "source_category": "paper"},
            "source-doc",
        )

        self.assertTrue(bad)
        self.assertEqual([], good)

    def test_ptx_source_uses_current_exact_instruction_locators(self):
        source = (ROOT / "sources/docs/nvidia-ptx-isa-sm100.md").read_text(encoding="utf-8")

        for locator in ("§9.7.17.7", "§9.7.17.10.7", "§9.7.9.26.5.2"):
            self.assertIn(locator, source)
        for stale in ("§9.7.17.8", "§9.7.17.10.1", "§9.7.9.25"):
            self.assertNotIn(stale, source)

        paths = list((ROOT / "wiki").rglob("*.md"))
        paths.extend((ROOT / "sources/docs").glob("*.md"))
        for path in paths:
            body = path.read_text(encoding="utf-8")
            self.assertEqual([], validate.ptx_isa_curated_anchor_errors(body, path))
        self.assertTrue(validate.ptx_isa_curated_anchor_errors(
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-alloc"
        ))
        self.assertTrue(validate.ptx_isa_curated_anchor_errors(
            "https://docs.nvidia.com/cuda/parallel-thread-execution/"
            "#cluster-launch-control-instructions"
        ))

    def test_blackwell_relevance_promise_matches_wiki_only_validator_scope(self):
        hopper = validate.HOPPER_ARCHITECTURES
        blackwell = validate.BLACKWELL_ARCHITECTURES
        wiki_hopper_only = []
        source_hopper_only_without_field = []
        for root, destination in (
            (ROOT / "wiki", wiki_hopper_only),
            (ROOT / "sources", source_hopper_only_without_field),
        ):
            for path in root.rglob("*.md"):
                fm = self.frontmatter(path)
                archs = set(fm.get("architectures") or [])
                if archs & hopper and not archs & blackwell:
                    if root == ROOT / "wiki" or "blackwell_relevance" not in fm:
                        destination.append((path, fm))

        self.assertTrue(wiki_hopper_only)
        self.assertTrue(source_hopper_only_without_field)
        self.assertTrue(all("blackwell_relevance" in fm for _, fm in wiki_hopper_only))
        for arch in blackwell:
            fm = {"architectures": ["sm90", arch]}
            self.assertEqual([], validate.blackwell_relevance_errors(fm, "wiki-hardware"))
        for arch in hopper:
            fm = {"architectures": [arch]}
            self.assertTrue(validate.blackwell_relevance_errors(fm, "wiki-pattern"))
        source_fm = {"architectures": sorted(hopper)}
        self.assertEqual([], validate.blackwell_relevance_errors(source_fm, "source-doc"))
        for path in (ROOT / "README.md", ROOT / "SKILL.md", ROOT / "CLAUDE.md", ROOT / "references/schema.md"):
            text = path.read_text(encoding="utf-8").lower()
            self.assertIn("wiki pages", text)
            self.assertIn("source pages", text)
            self.assertIn("exempt", text)

    def test_merge_sha_exists_only_for_merged_prs_and_is_full_length(self):
        statuses = []
        for path in (ROOT / "sources/prs").rglob("*.md"):
            fm = self.frontmatter(path)
            statuses.append(fm["status"])
            self.assertEqual([], validate.merge_sha_contract_errors(fm, path))

        self.assertEqual(942, statuses.count("merged"))
        self.assertEqual(2, statuses.count("closed"))
        self.assertTrue(validate.merge_sha_contract_errors({"status": "merged"}))
        self.assertTrue(validate.merge_sha_contract_errors({"status": "merged", "merge_sha": "unknown"}))
        self.assertTrue(validate.merge_sha_contract_errors({"status": "closed", "merge_sha": "a" * 40}))

        spec = importlib.util.spec_from_file_location(
            "generate_pr_pages_merge_contract", SCRIPTS / "generate-pr-pages.py"
        )
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        payload = {
            "number": 1,
            "title": "Closed CUDA implementation",
            "user": {"login": "author"},
            "created_at": "2026-01-02T03:04:05Z",
            "html_url": "https://github.com/example/project/pull/1",
            "state": "closed",
            "merged": False,
            "merge_commit_sha": "ephemeral-test-merge",
            "body": "Upstream body.",
        }
        files = [{
            "filename": "csrc/kernel.cu",
            "status": "added",
            "patch": "+__global__ void kernel() {}",
        }]
        rendered = generator.generate_page(
            "example/project", payload, files, "ignored", "2026-08-18"
        )
        generated_fm = validate.yaml.safe_load(rendered.split("---", 2)[1])
        self.assertEqual("closed", generated_fm["status"])
        self.assertNotIn("merge_sha", generated_fm)
        payload.update({"merged": True, "merge_commit_sha": None})
        with self.assertRaisesRegex(ValueError, "has no merge_commit_sha"):
            generator.generate_page(
                "example/project", payload, files, "ignored", "2026-08-18"
            )


if __name__ == "__main__":
    unittest.main()

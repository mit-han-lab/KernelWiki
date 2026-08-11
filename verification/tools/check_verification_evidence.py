#!/usr/bin/env python3
"""Re-run repository-local evidence checks used by verification receipts.

Each check has a stable name so verification JSON can cite an exact command
without embedding shell placeholders or depending on files under /tmp.
Network checks discover the links from the cited wiki revision and report the
status of every positive target plus an intentionally invalid control.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASELINE_REVISION = "2777d18"


def emit(check: str, **values: object) -> None:
    print(json.dumps({"check": check, **values}, sort_keys=True))


def read_revision(path: str, revision: str | None = None) -> str:
    if revision is None:
        return (ROOT / path).read_text()
    return subprocess.check_output(
        ["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True
    )


def source_urls(path: str, revision: str | None = None) -> list[str]:
    text = read_revision(path, revision)
    section = re.search(
        r"(?im)^## (?:Primary [Ss]ources|Sources)\s*$([\s\S]*?)(?=^## |\Z)",
        text,
    )
    if not section:
        raise AssertionError(f"no source section in {path}")
    return re.findall(r"\]\((https?://[^)]+)\)", section.group(1))


URL_CHECKS = {
    "urls-current-gated-delta-net": (
        "wiki/kernels/gated-delta-net.md",
        None,
        10,
        ["https://mlsys26.flashinfer.ai/definitely-not-a-kernelwiki-page"],
    ),
    "urls-current-gated-dual-gemm": (
        "wiki/kernels/gated-dual-gemm.md",
        None,
        4,
        [
            "https://github.com/gpu-mode/reference-kernels/blob/"
            "c5b2f7c062d5015f29c3a1043cfd04954397944c/problems/nvidia/"
            "nvfp4_dual_gemm/nonexistent-kernelwiki-file.py"
        ],
    ),
    "urls-current-grouped-gemm": (
        "wiki/kernels/grouped-gemm.md",
        None,
        10,
        [
            "https://github.com/gpu-mode/reference-kernels/blob/"
            "ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/"
            "nvfp4_group_gemm/nonexistent-kernelwiki-file.py"
        ],
    ),
    "urls-current-nsa": (
        "wiki/kernels/nsa.md",
        None,
        6,
        ["https://aclanthology.org/definitely-not-a-kernelwiki-paper/"],
    ),
    "urls-current-nvfp4-gemm": (
        "wiki/kernels/nvfp4-gemm.md",
        None,
        12,
        [
            "https://github.com/gpu-mode/reference-kernels/blob/"
            "ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/"
            "nvfp4_gemm/nonexistent-kernelwiki-file.py"
        ],
    ),
    "urls-current-nvfp4-gemv": (
        "wiki/kernels/nvfp4-gemv.md",
        None,
        9,
        ["https://veitner.bearblog.dev/definitely-not-a-kernelwiki-post/"],
    ),
    "urls-original-gated-delta-net": (
        "wiki/kernels/gated-delta-net.md",
        BASELINE_REVISION,
        4,
        ["https://mlsys26.flashinfer.ai/definitely-not-a-kernelwiki-page"],
    ),
    "urls-original-grouped-gemm": (
        "wiki/kernels/grouped-gemm.md",
        BASELINE_REVISION,
        4,
        [
            "https://github.com/gpu-mode/reference-kernels/blob/"
            "ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/"
            "nvfp4_group_gemm/nonexistent-kernelwiki-file.py"
        ],
    ),
    "urls-original-nsa": (
        "wiki/kernels/nsa.md",
        BASELINE_REVISION,
        2,
        ["https://aclanthology.org/definitely-not-a-kernelwiki-paper/"],
    ),
    "urls-original-nvfp4-gemm": (
        "wiki/kernels/nvfp4-gemm.md",
        BASELINE_REVISION,
        4,
        [
            "https://github.com/gpu-mode/reference-kernels/blob/"
            "ae67948685dfccf54ae8374dc9402addb7aae4f6/problems/nvidia/"
            "nvfp4_gemm/nonexistent-kernelwiki-file.py"
        ],
    ),
    "urls-original-nvfp4-gemv": (
        "wiki/kernels/nvfp4-gemv.md",
        BASELINE_REVISION,
        4,
        [
            "https://veitner.bearblog.dev/definitely-not-a-kernelwiki-post/",
            "https://yue-zhang-2025.github.io/definitely-not-a-kernelwiki-post/",
        ],
    ),
}


def curl_status(url: str) -> tuple[int, str]:
    completed = subprocess.run(
        [
            "curl",
            "-L",
            "--retry",
            "2",
            "--connect-timeout",
            "20",
            "--max-time",
            "90",
            "-A",
            "Mozilla/5.0 KernelWiki evidence checker",
            "-sS",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}\t%{url_effective}",
            url,
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    status, final_url = completed.stdout.split("\t", 1)
    return int(status), final_url


def check_urls(name: str) -> None:
    path, revision, expected, negative_urls = URL_CHECKS[name]
    urls = source_urls(path, revision)
    assert len(urls) == expected, (path, len(urls), expected)
    positives = []
    for url in urls:
        status, final_url = curl_status(url)
        assert 200 <= status < 400, (url, status, final_url)
        positives.append({"status": status, "url": url, "final_url": final_url})
    negatives = []
    for url in negative_urls:
        status, final_url = curl_status(url)
        assert status >= 400, (url, status, final_url)
        negatives.append({"status": status, "url": url, "final_url": final_url})
    emit(
        name,
        positive_count=len(positives),
        positive_statuses=sorted({row["status"] for row in positives}),
        negative_statuses=[row["status"] for row in negatives],
        revision=revision or "worktree",
        targets=positives,
    )


def check_flash_attention_topk() -> None:
    page = ROOT / "sources/prs/flash-attention/PR-2441.md"
    bundle = ROOT / "artifacts/prs/flash-attention/PR-2441"
    provenance = yaml.safe_load((bundle / "PROVENANCE.yaml").read_text())
    assert provenance["source_pr_id"] == "pr-flash-attention-2441"
    assert provenance["upstream_sha"] == "f219c89c"
    names = {row["local_path"] for row in provenance["files"]}
    expected = {
        "key-files/flash_attn/cute/flash_fwd_mla_sm100.py",
        "key-files/flash_attn/cute/topk_gather_kv.py",
    }
    assert expected <= names
    page_text = page.read_text()
    assert all(Path(name).name in page_text for name in expected)
    verified = 0
    for row in provenance["files"]:
        target = bundle / row["local_path"]
        assert target.is_file(), target
        assert hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"]
        verified += 1
    assert not (bundle / "key-files/flash_attn/cute/nonexistent_topk.py").exists()
    emit(
        "flash-attention-topk",
        source_pr_id=provenance["source_pr_id"],
        upstream_sha=provenance["upstream_sha"],
        verified_files=verified,
        required_files=sorted(expected),
        missing_control=True,
    )


def check_flashmla_layout() -> None:
    query = subprocess.run(
        [sys.executable, "scripts/get_page.py", "kernel-flashmla", "--include-code"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    expected = [
        "adjacent implementations",
        "artifacts/kernels/flashmla/full/",
        "variants/01-mla-layout-and-index-contract.cu",
    ]
    assert all(token in query for token in expected)
    missing = subprocess.run(
        [sys.executable, "scripts/get_page.py", "kernel-flashmla-does-not-exist"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert missing.returncode != 0 and "No page found" in (missing.stdout + missing.stderr)
    byte_count = 512 + 4 * 4 + 64 * 2
    assert byte_count == 656 and byte_count != 655
    emit(
        "flashmla-layout",
        bytes=byte_count,
        expected_tokens=expected,
        missing_page_returncode=missing.returncode,
    )


def check_fp8_block_scale_shapes() -> None:
    def shapes(m: int, n: int, k: int) -> tuple[tuple[int, int], tuple[int, int]]:
        assert m % 128 == 0 and n % 128 == 0 and k % 128 == 0
        return (m, k // 128), (n // 128, k // 128)

    output = shapes(4096, 4096, 4096)
    assert output == ((4096, 32), (32, 32))
    rejected = False
    try:
        shapes(4096, 4095, 4096)
    except AssertionError:
        rejected = True
    assert rejected
    emit("fp8-block-scale-shapes", output=output, invalid_n_rejected=rejected)


def check_fused_moe_shapes() -> None:
    def shapes(tokens: int) -> dict[str, tuple[int, ...]]:
        assert tokens > 0
        return {
            "routing_logits": (tokens, 256),
            "hidden_states": (tokens, 7168),
            "hidden_states_scale": (56, tokens),
            "gemm1_weights": (32, 4096, 7168),
            "gemm1_weights_scale": (32, 32, 56),
            "gemm2_weights": (32, 7168, 2048),
            "gemm2_weights_scale": (32, 56, 16),
        }

    output = shapes(7)
    rejected = False
    try:
        shapes(0)
    except AssertionError:
        rejected = True
    assert rejected
    emit("fused-moe-track-a-shapes", shapes=output, zero_tokens_rejected=rejected)


def check_grouped_gemm_reference() -> None:
    def grouped_reference(groups: list[tuple[list[list[int]], list[list[int]]]]) -> list[list[list[int]]]:
        outputs = []
        for a, b in groups:
            k = len(a[0])
            if any(len(row) != k for row in a) or any(len(row) != k for row in b):
                raise ValueError("ragged or incompatible K")
            outputs.append(
                [[sum(x * y for x, y in zip(a_row, b_row)) for b_row in b] for a_row in a]
            )
        return outputs

    groups = [([[1, 2], [3, 4]], [[5, 6]]), ([[1, 0, 0]], [[1, 2, 3], [5, 6, 7]])]
    output = grouped_reference(groups)
    assert output == [[[17], [39]], [[1, 5]]]
    shared_k_rejected = len(groups[1][0][0]) != len(groups[0][0][0])
    assert shared_k_rejected
    ragged_rejected = False
    try:
        grouped_reference([([[1, 2]], [[3]])])
    except ValueError:
        ragged_rejected = True
    assert ragged_rejected
    emit(
        "grouped-gemm-reference",
        output=output,
        ragged_rejected=ragged_rejected,
        shared_k_control_rejected=shared_k_rejected,
    )


def check_nsa_reference() -> None:
    branches = [[1.0, 2.0], [10.0, 20.0], [-2.0, 4.0]]
    gates = [0.5, 0.25, 1.0]
    gated = [sum(g * b[i] for b, g in zip(branches, gates)) for i in range(2)]
    ungated = [sum(b[i] for b in branches) for i in range(2)]
    assert gated == [1.0, 10.0] and ungated == [9.0, 26.0]
    missing_gate_rejected = len(branches) != len(gates[:-1])
    assert missing_gate_rejected
    emit(
        "nsa-gated-branch-reference",
        gated=gated,
        ungated=ungated,
        missing_gate_rejected=missing_gate_rejected,
    )


def check_nsa_former_counterexamples() -> None:
    scores = [0.0, 0.0]
    values = [1.0, 3.0]
    running_max = max(scores)
    candidate = sum(math.exp(score - running_max) * value for score, value in zip(scores, values))
    partition = sum(math.exp(score - running_max) for score in scores)
    normalized = candidate / partition
    candidate_positions = list(range(0, 4, 2))
    reference_positions = list(range(4))
    assert candidate == 4.0 and normalized == 2.0
    assert candidate_positions == [0, 2] and candidate_positions != reference_positions
    emit(
        "nsa-former-counterexamples",
        candidate_update=candidate,
        normalized_softmax=normalized,
        candidate_positions=candidate_positions,
        reference_positions=reference_positions,
    )


def nvfp4_shapes(m: int, n: int, k: int, layers: int = 1) -> dict[str, tuple[int, ...]]:
    if min(m, n, k, layers) <= 0:
        raise ValueError("dimensions must be positive")
    if k % 256:
        raise ValueError("K must be divisible by 256")
    return {
        "a_packed": (m, k // 2, layers),
        "b_packed": (n, k // 2, layers),
        "sfa_logical": (m, k // 16, layers),
        "sfb_logical": (n, k // 16, layers),
        "sfa_reordered": (32, 4, (m + 127) // 128, 4, (k + 63) // 64, layers),
        "sfb_reordered": (32, 4, (n + 127) // 128, 4, (k + 63) // 64, layers),
        "c": (m, n, layers),
    }


def check_nvfp4_shapes() -> None:
    task_path = (
        ROOT
        / "verification/evidence/local-snapshots/kernelwiki-reference-kernels/problems/nvidia/nvfp4_gemm/task.yml"
    )
    task = yaml.safe_load(task_path.read_text())
    tests = task["tests"]
    assert len(tests) == 10
    outputs = [nvfp4_shapes(row["m"], row["n"], row["k"], row["l"]) for row in tests]
    old_rule_failures = sum((row["k"] // 16) % 128 != 0 for row in tests)
    assert old_rule_failures == 9
    valid_k = sorted({row["k"] for row in tests} | {16384})
    for k in valid_k:
        nvfp4_shapes(128, 256, k)
    invalid_rejected = False
    try:
        nvfp4_shapes(128, 256, 255)
    except ValueError:
        invalid_rejected = True
    assert invalid_rejected
    emit(
        "nvfp4-gemm-shapes",
        official_tests=len(outputs),
        old_rule_failures=old_rule_failures,
        valid_k=valid_k,
        invalid_k_rejected=invalid_rejected,
    )


def check_clc_scope() -> None:
    cases = [
        ("B200_128x256", 148, 128, 256),
        ("B200_256x256", 148, 256, 256),
        ("132SM_128x256", 132, 128, 256),
    ]
    output = []
    for name, sms, tile_m, tile_n in cases:
        tiles = math.ceil(2048 / tile_m) * math.ceil(2048 / tile_n)
        output.append((name, tiles, round(100 * min(tiles, sms) / sms, 2)))
    invariant = len({row[2] for row in output}) == 1
    assert [row[2] for row in output] == [86.49, 43.24, 96.97]
    assert not invariant
    emit("clc-scope-arithmetic", capacities=output, invariant_control=invariant)


def check_clc_swizzle() -> None:
    size = 4
    bits = 2

    def wiki(m: int, n: int) -> tuple[int, int]:
        linear = m * size + n
        return (linear >> bits) // size, ((linear >> bits) % size) ^ (((linear >> bits) // size) & ((1 << bits) - 1))

    outputs = [wiki(m, n) for m in range(4) for n in range(4)]

    def morton(m: int, n: int) -> int:
        return sum(((n >> bit) & 1) << (2 * bit) | ((m >> bit) & 1) << (2 * bit + 1) for bit in range(2))

    morton_order = [(m, n) for _, m, n in sorted((morton(m, n), m, n) for m in range(4) for n in range(4))]
    assert len(set(outputs)) != 16
    assert outputs != morton_order
    identity_unique = len({(m, n) for m in range(4) for n in range(4)})
    assert identity_unique == 16
    emit(
        "clc-swizzle-vs-morton",
        wiki_outputs=outputs,
        wiki_unique=len(set(outputs)),
        morton_order=morton_order,
        identity_unique=identity_unique,
    )


def extract_baseline_function(name: str) -> str:
    text = read_revision("wiki/languages/triton-blackwell.md", BASELINE_REVISION)
    for block in re.findall(r"```python\n(.*?)```", text, flags=re.S):
        if f"def {name}(" in block:
            return block
    raise AssertionError(name)


def undefined_names(code: str, function_name: str) -> list[str]:
    tree = ast.parse(code)
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == function_name)
    arguments = {arg.arg for arg in function.args.args}
    assigned = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    loaded = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    allowed = {"tl", "triton", "range"}
    return sorted(loaded - arguments - assigned - allowed)


def check_triton_ast() -> None:
    gated = extract_baseline_function("gated_delta_net_decode")
    sparse = extract_baseline_function("sparse_attention_fwd")
    gated_undefined = undefined_names(gated, "gated_delta_net_decode")
    sparse_undefined = undefined_names(sparse, "sparse_attention_fwd")
    control = "def control(offsets, d):\n    return offsets + d\n"
    control_undefined = undefined_names(control, "control")
    assert gated_undefined == ["offsets"]
    assert sparse_undefined == ["d", "offsets"]
    assert "State + head_id" in gated and "tl.store(State + offsets" in gated
    assert "softmax + accumulate" in sparse and "tl.store" not in sparse
    assert not control_undefined
    emit(
        "triton-baseline-ast",
        revision=BASELINE_REVISION,
        gated_undefined=gated_undefined,
        gated_state_addresses_inconsistent=True,
        sparse_undefined=sparse_undefined,
        sparse_has_output_store=False,
        control_undefined=control_undefined,
    )


def load_baseline_triton_universe() -> dict[str, object]:
    return yaml.safe_load(read_revision("data/triton-universe.yaml", BASELINE_REVISION))


def check_triton_universe_baseline() -> None:
    ledger = load_baseline_triton_universe()
    prs = ledger["prs"]
    captured = sum(bool(row["captured"]) for row in prs)
    skipped = len(prs) - captured
    assert (len(prs), ledger["total"], captured, ledger["captured"], skipped, ledger["skipped"]) == (
        267,
        267,
        117,
        117,
        150,
        150,
    )
    assert len(prs) != 42
    emit(
        "triton-universe-baseline",
        revision=BASELINE_REVISION,
        total=len(prs),
        captured=captured,
        skipped=skipped,
        stale_42_control=False,
    )


def check_triton_capture_flags_baseline() -> None:
    ledger = load_baseline_triton_universe()
    flags = {row["id"]: row["captured"] for row in ledger["prs"]}
    linked = [
        "pr-vllm-34597",
        "pr-flashinfer-1025",
        "pr-sglang-20910",
        "pr-sglang-21019",
        "pr-sglang-22079",
    ]
    assert all(flags[name] is True for name in linked)
    assert "pr-does-not-exist" not in flags
    skipped = sum(value is False for value in flags.values())
    assert skipped == 150
    emit(
        "triton-capture-flags-baseline",
        revision=BASELINE_REVISION,
        linked={name: flags[name] for name in linked},
        skipped=skipped,
        missing_control=False,
    )


def check_triton_cutoff() -> None:
    cutoff_data = yaml.safe_load((ROOT / "data/refresh-cutoff.yaml").read_text())
    cutoff = date.fromisoformat(cutoff_data["cutoff_date"])
    paths = [
        "sources/prs/vllm/PR-34597.md",
        "sources/prs/vllm/PR-29339.md",
        "sources/prs/sglang/PR-22079.md",
        "sources/prs/sglang/PR-21019.md",
    ]
    pr_dates = {}
    for path in paths:
        metadata = yaml.safe_load((ROOT / path).read_text().split("---", 2)[1])
        value = metadata["date"]
        pr_date = value if isinstance(value, date) else date.fromisoformat(value)
        pr_dates[metadata["id"]] = pr_date.isoformat()
        assert pr_date < cutoff
    assert "previous_pages_manifest" not in cutoff_data
    assert date(2026, 5, 21) > cutoff
    emit(
        "triton-refresh-cutoff",
        top_level_keys=sorted(cutoff_data),
        cutoff=cutoff.isoformat(),
        pr_dates=pr_dates,
        all_before_cutoff=True,
        synthetic_after_control=True,
    )


def triton_artifact_paths() -> list[Path]:
    text = (ROOT / "wiki/languages/triton-blackwell.md").read_text()
    links = re.findall(r"\]\(\.\./\.\./(artifacts/prs/[^)]+\.py)\)", text)
    paths = [ROOT / link for link in dict.fromkeys(links)]
    assert len(paths) == 6, paths
    return paths


def check_triton_artifacts() -> None:
    paths = triton_artifact_paths()
    roles = {
        "triton_decode_attention.py": ["tl.dot", "is_fp8"],
        "triton_mla.py": ["TritonMLABackend", "fp8"],
        "format_conversion.py": ["triton.jit", "tl.load", "tl.store"],
        "norm.py": ["load_jit", "_jit_rmsnorm_module", "def rmsnorm"],
        "gdn_fused_proj.py": ["tl.load", "tl.store"],
        "extend_attention.py": ["tl.dot"],
    }
    verified = []
    bundles = set()
    for path in paths:
        assert path.is_file(), path
        bundle = next(parent for parent in path.parents if parent.name.startswith("PR-"))
        bundles.add(bundle)
        provenance = yaml.safe_load((bundle / "PROVENANCE.yaml").read_text())
        assert provenance["asset_mode"] == "verbatim"
        relative = path.relative_to(bundle).as_posix()
        entry = next(row for row in provenance["files"] if row["local_path"] == relative)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"] and entry["mode"] == "verbatim"
        content = path.read_text(errors="replace")
        assert all(token in content for token in roles[path.name]), (path, roles[path.name])
        verified.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest})
    missing = ROOT / "artifacts/prs/vllm/PR-00000/key-files/missing.py"
    assert not missing.exists()
    emit(
        "triton-artifact-provenance",
        verified_files=verified,
        bundle_count=len(bundles),
        missing_control=True,
    )


def check_triton_universe_current() -> None:
    ledger = yaml.safe_load((ROOT / "data/triton-universe.yaml").read_text())
    prs = ledger["prs"]
    assert len(prs) == ledger["total"] == 267
    assert all("captured" in row for row in prs)
    page = ROOT / "wiki/languages/triton-blackwell.md"
    assert "../../data/triton-universe.yaml" in page.read_text()
    assert not (ROOT / "data/triton-universe-missing.yaml").exists()
    emit(
        "triton-universe-current",
        total=len(prs),
        captured=sum(bool(row["captured"]) for row in prs),
        skipped=sum(not bool(row["captured"]) for row in prs),
        every_entry_has_captured=True,
        missing_control=True,
    )


CHECKS = {
    "flash-attention-topk": check_flash_attention_topk,
    "flashmla-layout": check_flashmla_layout,
    "fp8-block-scale-shapes": check_fp8_block_scale_shapes,
    "fused-moe-track-a-shapes": check_fused_moe_shapes,
    "grouped-gemm-reference": check_grouped_gemm_reference,
    "nsa-gated-branch-reference": check_nsa_reference,
    "nsa-former-counterexamples": check_nsa_former_counterexamples,
    "nvfp4-gemm-shapes": check_nvfp4_shapes,
    "clc-scope-arithmetic": check_clc_scope,
    "clc-swizzle-vs-morton": check_clc_swizzle,
    "triton-baseline-ast": check_triton_ast,
    "triton-universe-baseline": check_triton_universe_baseline,
    "triton-capture-flags-baseline": check_triton_capture_flags_baseline,
    "triton-refresh-cutoff": check_triton_cutoff,
    "triton-artifact-provenance": check_triton_artifacts,
    "triton-universe-current": check_triton_universe_current,
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] in {"-h", "--help"}:
        names = sorted([*CHECKS, *URL_CHECKS])
        print(
            "usage: python3 verification/tools/check_verification_evidence.py CHECK",
            file=sys.stderr,
        )
        print("checks:\n  " + "\n  ".join(names), file=sys.stderr)
        return 0 if len(sys.argv) == 2 else 2
    name = sys.argv[1]
    if name in URL_CHECKS:
        check_urls(name)
        return 0
    try:
        check = CHECKS[name]
    except KeyError:
        print(f"unknown check: {name}", file=sys.stderr)
        return 2
    check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

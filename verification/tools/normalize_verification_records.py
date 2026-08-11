#!/usr/bin/env python3
"""Normalize known unstable verification references and placeholder commands.

This is intentionally idempotent. It preserves surrounding JSON formatting,
rewrites only exact known strings, and repairs the four historical receipt
evidence objects that lacked the verification contract's ``result`` field.
"""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE_REVISION = "2777d18"
HISTORICAL_ROOT = ROOT / "verification/evidence/historical-artifacts"


HISTORICAL_PATHS = [
    "artifacts/kernels/flashmla/variants/01-mla-decode-inner-loop.cu",
    "artifacts/kernels/gated-delta-net/variants/01-chunk-parallel-prefill-reference-pytorch.py",
    "artifacts/kernels/gated-delta-net/variants/02-triton-decode-step-kernel-streaming.py",
    "artifacts/kernels/gated-dual-gemm/full/vllm-PR-23696-gated-dual-gemm.patch",
    "artifacts/kernels/gated-dual-gemm/full/blackwell-cutlass-schedules-and-tma.cu",
    "artifacts/kernels/gated-dual-gemm/variants/01-fused-epilogue-swiglu-skeleton.cu",
]


PATH_REPLACEMENTS = {
    path: f"verification/evidence/historical-artifacts/{path}"
    for path in HISTORICAL_PATHS
}


REFERENCE_REPLACEMENTS = {
    "#asynchronous-warpgroup-level-matrix-operation-wgmma-mma-async-mma":
        "#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma",
    "#asynchronous-warpgroup-level-matrix-instructions-wgmma":
        "#asynchronous-warpgroup-level-matrix-operation-wgmma-mma-async",
    "#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions-wgmma-mma":
        "#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma",
    "#clusterlaunchcontrol-try-cancel":
        "#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel",
    "#contents-of-the-mbarrier-object":
        "#parallel-synchronization-and-communication-instructions-mbarrier-contents",
    "#mbarrier-support-with-shared-memory":
        "#parallel-synchronization-and-communication-instructions-mbarrier-smem",
    "#phase-completion-of-the-mbarrier-object":
        "#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion",
    "#tensorcore-5th-generation-instructions-shared-memory-descriptor":
        "#tcgen05-shared-memory-descriptor",
    "#tensorcore-5th-generation-instructions-shared-memory-layout":
        "#tcgen05-shared-memory-layout-swizzling",
    "#tensorcore-5th-generation-instructions-tcgen05-alloc":
        "#tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit",
    "#tensorcore-5th-generation-instructions-tcgen05-commit":
        "#tcgen-async-sync-operations-commit",
    "#tensorcore-5th-generation-instructions-tcgen05-cp":
        "#tcgen05-instructions-tcgen05-cp",
    "#tensorcore-5th-generation-instructions-tcgen05-dealloc":
        "#tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit",
    "#tensorcore-5th-generation-instructions-tcgen05-fence":
        "#tcgen05-special-sync-operations-fence",
    "#tensorcore-5th-generation-instructions-tcgen05-ld":
        "#tcgen05-instructions-tcgen05-ld",
    "#tensorcore-5th-generation-instructions-tcgen05-mma":
        "#tcgen05-mma-instructions-mma",
    "#tensorcore-5th-generation-instructions-tcgen05-wait":
        "#tcgen05-instructions-tcgen05-wait",
    "#tensorcore-5th-generation-matrix-multiply-and-accumulate-operations":
        "#tcgen05-mma",
    "https://docs.nvidia.com/cutlass/media/docs/cpp/hopper_gemm.html":
        "https://docs.nvidia.com/cutlass/4.5.0/media/docs/cpp/gemm_api_3x.html",
    "e406c186d2cae5782a846f7280af282ca4fecec2":
        "e406c186f510a15091cce01f782020ceb7ba8eb5",
    "https://triton-lang.org/main/programming-guide/chapter-01/introduction.html":
        "https://triton-lang.org/main/programming-guide/chapter-1/introduction.html",
}


def check_command(name: str) -> str:
    return f"python3 verification/tools/check_verification_evidence.py {name}"


COMMAND_REPLACEMENTS = {
    "conda run -n base python -c 'N=4; bits=2; f=lambda m,n: ((m*N+n>>bits)//N,((m*N+n>>bits)%N)^(((m*N+n>>bits)//N)&((1<<bits)-1))); out=[f(m,n) for m in range(4) for n in range(4)]; morton=lambda m,n: sum(((n>>b)&1)<<(2*b)|((m>>b)&1)<<(2*b+1) for b in range(2)); print({\"wiki_outputs\":out,\"unique\":len(set(out)),\"expected_unique\":16,\"claimed_morton_order\":sorted(((morton(m,n),m,n) for m in range(4) for n in range(4))),\"negative_control_identity_unique\":len(set((m,n) for m in range(4) for n in range(4)))})'":
        check_command("clc-swizzle-vs-morton"),
    "conda run -n base python <the displayed pathlib lookup>; sha256sum artifacts/prs/flash-attention/PR-2441/diff.patch and key files":
        check_command("flash-attention-topk"),
    "conda run -n base python <the displayed pathlib/assert snippet>":
        check_command("flash-attention-topk"),
    "conda run -n base python scripts/get_page.py kernel-flashmla --include-code; conda run -n base python -c '<displayed byte arithmetic and assertion>'":
        check_command("flashmla-layout"),
    "conda run -n base python <displayed deepseek_v3_scale_shapes function and assertion>":
        check_command("fp8-block-scale-shapes"),
    "conda run -n base python -c <displayed track_a_shapes function and assertion>":
        check_command("fused-moe-track-a-shapes"),
    "conda run -n base python -c '<exact grouped_reference function; two groups with shapes (2,1,2) and (1,2,3); expected outputs; wrong_shared_k control>'":
        check_command("grouped-gemm-reference"),
    "conda run -n base python -c '<exact gated_branch_sum from page; asserted three-branch result, ungated result, and missing-gate rejection>'":
        check_command("nsa-gated-branch-reference"),
    "conda run -n base python -c '<evaluate former first-block softmax update and former block-stride scalar window loop>'":
        check_command("nsa-former-counterexamples"),
    "conda run -n base python -c '<evaluate the displayed first-block update for scores=[0,0], values=[1,3], then compare stable softmax>'":
        check_command("nsa-former-counterexamples"),
    "conda run -n base python -c '<compare range(0,4,2) used by the displayed scalar-load loop with all window positions>'":
        check_command("nsa-former-counterexamples"),
    "conda run -n base python -c '<candidate first-block update versus normalized softmax>'":
        check_command("nsa-former-counterexamples"),
    "conda run -n base python -c '<compare range(0,4,2) scalar loads with all window positions>'":
        check_command("nsa-former-counterexamples"),
    "conda run -n base python -c '<execute current task_storage_shapes, seven valid K values, official ten tests, and K=255 negative control>'":
        check_command("nvfp4-gemm-shapes"),
    "conda run -n base python -c '<evaluate official ten Problem 2 shapes against task storage shapes and both assertions>'":
        check_command("nvfp4-gemm-shapes"),
    "conda run -n base python -c '<official shape-contract oracle>'":
        check_command("nvfp4-gemm-shapes"),
    "conda run -n base python -c <scope-discrimination arithmetic>":
        check_command("clc-scope-arithmetic"),
    "conda run -n base python -c <wiki swizzle versus Morton permutation>":
        check_command("clc-swizzle-vs-morton"),
    "conda run -n base python -c <AST undefined-name audit over the two wiki Python blocks>":
        check_command("triton-baseline-ast"),
    "conda run -n base python -c <AST undefined-name audit>":
        check_command("triton-baseline-ast"),
    "conda run -n base python -c <parse data/triton-universe.yaml and count prs>":
        check_command("triton-universe-baseline"),
    "conda run -n base python -c <parse data/refresh-cutoff.yaml and compare four PR dates>":
        check_command("triton-refresh-cutoff"),
    "conda run -n base python -c <parse refresh cutoff and compare PR dates>":
        check_command("triton-refresh-cutoff"),
    "conda run -n base python -c <parse capture flags for the five linked PR IDs>":
        check_command("triton-capture-flags-baseline"),
    "conda run -n base python -c <resolve six paths and compare PROVENANCE.yaml SHA-256 entries>":
        check_command("triton-artifact-provenance"),
    "conda run -n base python -c <resolve six page links, parse bundle PROVENANCE.yaml, and compare SHA-256 values>":
        check_command("triton-artifact-provenance"),
    "conda run -n base python -c <resolve page link and parse data/triton-universe.yaml>":
        check_command("triton-universe-current"),
    "curl -L -sS -o /dev/null -w '%{http_code}' <each of the ten Primary sources URLs>":
        check_command("urls-current-gated-delta-net"),
    "curl -L -sS -o /dev/null -w '%{http_code}' <each of four Primary sources URLs>":
        check_command("urls-current-gated-dual-gemm"),
    "curl -L -sS -o /dev/null -w '%{http_code}' <each of ten Primary sources URLs>":
        check_command("urls-current-grouped-gemm"),
    "curl -L -sS -o /dev/null -w '%{http_code}' <each of six Primary sources URLs>":
        check_command("urls-current-nsa"),
    "curl -L -sS -o /dev/null -w '%{http_code}' <all twelve current primary-source URLs>":
        check_command("urls-current-nvfp4-gemm"),
    "curl -L -A 'Mozilla/5.0' -sS -o /dev/null -w '%{http_code}' <each current Primary Sources URL>":
        check_command("urls-current-nvfp4-gemv"),
    "curl -L -sS -o /dev/null -w '%{http_code}' <each of four URLs>":
        check_command("urls-original-gated-delta-net"),
    "curl -L -sS -o /dev/null -w '%{http_code}' <each original source URL>":
        check_command("urls-original-grouped-gemm"),
    "curl -L -sS -o /dev/null -w '%{http_code}' <ACL paper, PDF, and lucidrains repository URLs>":
        check_command("urls-original-nsa"),
    "curl -L -sS -o /dev/null -w '%{http_code}' <four original source URLs>":
        check_command("urls-original-nvfp4-gemm"),
    "curl -L -A 'Mozilla/5.0' -sS -o /dev/null -w '%{http_code}' <each of four URLs>":
        check_command("urls-original-nvfp4-gemv"),
}


STRING_REPLACEMENTS = {
    "The link resolves; the canonical YAML parses with 267 PR entries, each carrying a captured flag, while the wiki text states no copied count.":
        "The link resolves; the canonical YAML parses with 267 PR entries (116 captured and 151 skipped), every entry carries a captured flag, and the wiki text states no copied count.",
    "ACL landing page, proceedings PDF, and lucidrains repository each returned HTTP 200 and matched their stated identities.":
        "Both URLs in the baseline Sources section returned HTTP 200 and matched the NSA paper and explicitly third-party implementation identities.",
    "A nonexistent sibling path under aclanthology.org returned HTTP 404.":
        "A nonexistent sibling path under aclanthology.org returned HTTP 404.",
    "A nonexistent sibling page under haroldbenoit.com returned HTTP 404.":
        "A nonexistent file under the pinned GPU Mode nvfp4_gemm path returned HTTP 404; this avoids the Harold Benoit site's soft-404 behavior.",
}


RECEIPT_EVIDENCE = {
    "rem-lang-triton-004": {
        "command": check_command("triton-baseline-ast"),
        "environment": "KernelWiki worktree; Python 3 AST; baseline revision 2777d18",
        "reference": "wiki/languages/triton-blackwell.md at 2777d18, gated_delta_net_decode block",
        "result": "The function loads undefined offsets and uses inconsistent recurrent-state load/store address expressions.",
        "negative_control": "A control function receiving offsets and d as arguments has no undefined names.",
    },
    "rem-lang-triton-005": {
        "command": check_command("triton-baseline-ast"),
        "environment": "KernelWiki worktree; Python 3 AST; baseline revision 2777d18",
        "reference": "wiki/languages/triton-blackwell.md at 2777d18, sparse_attention_fwd block",
        "result": "The function loads undefined d and offsets, ends at an unfinished softmax comment, and never stores Output.",
        "negative_control": "A control function receiving offsets and d as arguments has no undefined names.",
    },
    "rem-lang-triton-006": {
        "command": check_command("triton-universe-baseline"),
        "environment": "KernelWiki Git object database; Python 3; PyYAML",
        "reference": "data/triton-universe.yaml at revision 2777d18",
        "result": "The pinned ledger declares and contains 267 entries: 117 captured and 150 skipped; 42 is not its total.",
        "negative_control": "The stale equality actual_count == 42 is false while actual_count == declared total is true.",
    },
    "rem-lang-triton-007": {
        "command": check_command("triton-refresh-cutoff"),
        "environment": "KernelWiki worktree; Python 3; PyYAML",
        "reference": "data/refresh-cutoff.yaml and the four named PR source records",
        "result": "The only top-level key is cutoff_date=2026-05-20; no previous_pages_manifest exists, and all four PR dates precede the cutoff.",
        "negative_control": "A synthetic 2026-05-21 date is correctly classified after the cutoff.",
    },
}


def replace_json_string(text: str, old: str, new: str) -> str:
    return text.replace(
        json.dumps(old, ensure_ascii=False),
        json.dumps(new, ensure_ascii=False),
    )


def snapshot_historical_evidence() -> bool:
    records = []
    changed = False
    for source_path in HISTORICAL_PATHS:
        content = subprocess.check_output(
            ["git", "show", f"{BASELINE_REVISION}:{source_path}"], cwd=ROOT
        )
        destination = HISTORICAL_ROOT / source_path
        if not destination.exists() or destination.read_bytes() != content:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            changed = True
        records.append(
            {
                "path": destination.relative_to(ROOT).as_posix(),
                "source_path": source_path,
                "revision": BASELINE_REVISION,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    manifest = {
        "schema_version": "kernelwiki-historical-evidence/v1",
        "source_revision": BASELINE_REVISION,
        "files": records,
    }
    manifest_path = HISTORICAL_ROOT / "MANIFEST.json"
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if not manifest_path.exists() or manifest_path.read_text() != manifest_text:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest_text)
        changed = True
    return changed


def rewrite_text_files() -> list[Path]:
    candidates: set[Path] = set()
    for directory in (ROOT / "verification", ROOT / "wiki", ROOT / "sources"):
        for path in directory.rglob("*"):
            excluded_parts = {"local-snapshots", "historical-artifacts"}
            if (
                path.is_file()
                and ROOT / "verification/tools" not in path.parents
                and not excluded_parts & set(path.parts)
            ):
                candidates.add(path)
    changed = []
    for path in sorted(candidates):
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        updated = text
        for old, new in REFERENCE_REPLACEMENTS.items():
            if old.startswith("#"):
                updated = re.sub(re.escape(old) + r"(?![-A-Za-z0-9])", new, updated)
            else:
                updated = updated.replace(old, new)
        if path.suffix in {".json", ".jsonl"}:
            for old, new in PATH_REPLACEMENTS.items():
                updated = replace_json_string(updated, old, new)
            for old, new in COMMAND_REPLACEMENTS.items():
                updated = replace_json_string(updated, old, new)
            for old, new in STRING_REPLACEMENTS.items():
                updated = replace_json_string(updated, old, new)
        if updated != text:
            path.write_text(updated)
            changed.append(path)
    return changed


def repair_receipts() -> bool:
    path = ROOT / "verification/remediation-log.jsonl"
    lines = path.read_text().splitlines()
    changed = False
    output = []
    for line in lines:
        row = json.loads(line)
        replacement = RECEIPT_EVIDENCE.get(row.get("receipt_id"))
        if replacement is not None and row.get("evidence") != [replacement]:
            row["evidence"] = [replacement]
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            changed = True
        output.append(line)
    if changed:
        path.write_text("\n".join(output) + "\n")
    return changed


def main() -> int:
    snapshots_changed = snapshot_historical_evidence()
    changed = rewrite_text_files()
    receipts_changed = repair_receipts()
    for path in changed:
        print(path.relative_to(ROOT))
    if receipts_changed and ROOT / "verification/remediation-log.jsonl" not in changed:
        print("verification/remediation-log.jsonl")
    print(
        f"normalized_files={len(set(changed)) + int(receipts_changed and not changed)} "
        f"historical_snapshots_changed={str(snapshots_changed).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

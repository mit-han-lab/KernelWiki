#!/usr/bin/env python3
"""Evidence-based architecture, scope, metadata, and body policy for PR pages.

The functions in this module are deliberately pure.  Intake, refresh, generation,
validation tests, and audit tooling can therefore exercise exactly the same rules.
Repository membership and a generic CUDA/NVFP4 token are never architecture proof,
and directory words such as ``moe`` or ``fused`` are never kernel-scope proof.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shlex
import tokenize
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable


BODY_CONTRACT = "upstream-pr-v1"
UPSTREAM_EXCERPT_LIMIT = 1200

ARCHITECTURE_FAMILY_PREFIXES = {
    "turing": ("sm75",),
    "ampere": ("sm80", "sm86", "sm87", "sm88"),
    "ada": ("sm89",),
    "hopper": ("sm90",),
    "blackwell": ("sm100", "sm103", "sm110", "sm120", "sm121"),
}
BLACKWELL_EXACT_PREFIXES = ARCHITECTURE_FAMILY_PREFIXES["blackwell"]

# Canonical exact targets accepted by both extraction and schema validation.
# SM88 is intentional: CUDA 13.3 NVCC lists ``sm_88`` as a supported GPU code.
SUPPORTED_EXACT_ARCHITECTURES = (
    "sm75",
    "sm80", "sm86", "sm87", "sm88", "sm89",
    "sm90", "sm90a",
    "sm100", "sm100a", "sm100f",
    "sm103", "sm103a", "sm103f",
    "sm110", "sm110a", "sm110f",
    "sm120", "sm120a", "sm120f",
    "sm121", "sm121a", "sm121f",
)
_SUPPORTED_EXACT_ARCHITECTURE_SET = set(SUPPORTED_EXACT_ARCHITECTURES)

_EXACT_SM_RE = re.compile(
    r"(?i:sm)[_ -]?(75|80|86|87|88|89|90|100|103|110|120|121)([aAfF]?)(?![0-9])"
)
_FAMILY_PATTERNS = {
    family: re.compile(rf"(?<![a-z0-9]){family}(?![a-z0-9])", re.IGNORECASE)
    for family in ARCHITECTURE_FAMILY_PREFIXES
}
_PORTABLE_ARCH_ATOM_SUFFIX_RE = re.compile(
    r"^_(?:cp_async|bulk_copy|tma|mma|wgmma|ldmatrix|stmatrix|copy|load|store)(?:_|$)",
    re.IGNORECASE,
)

_ARCHITECTURE_GUARD_LINE_RE = re.compile(
    r"__CUDA_ARCH__|CUTE_ARCH_|CUTLASS_ARCH_|CUDA_ARCHITECTURES|"
    r"(?:^|\W)(?:gencode|compute_capability|device_capability|"
    r"supports?_sm[_-]?\d+|is_sm[_-]?\d+|get_sm_version|sm_version|"
    r"target\s*[:=]\s*['\"]?sm[_-]?\d+|"
    r"architecture\s*[:=]\s*['\"]?sm[_-]?\d+)",
    re.IGNORECASE,
)
_NUMERIC_CUDA_ARCH_RE = re.compile(
    r"__CUDA_ARCH__\s*(?:==|>=)\s*(750|800|860|870|880|890|900|1000|1030|1100|1200|1210)\b"
)
_NUMERIC_RUNTIME_ARCH_RE = re.compile(
    r"(?:get_sm_version\s*\([^)]*\)|sm_version|compute_capability)\s*"
    r"(?:==|>=)\s*(75|80|86|87|88|89|90|100|103|110|120|121)\b",
    re.IGNORECASE,
)
_TARGET_ASSERTION_RE = re.compile(
    r"\b(?:requires?|supports?|supported|target(?:s|ed|ing)?|tuned|tested|"
    r"optimized|available|works?|compile[ds]?|architecture|compute capability|"
    r"dispatch|kernel)\b",
    re.IGNORECASE,
)
_COMMENTED_OUT_CODE_RE = re.compile(
    r"^\s*(?://\s*|#\s+)(?:bool|if|elif|else|def|class|from|import|return|"
    r"auto|const|constexpr|std::|[A-Za-z_]\w*\s*=)",
    re.IGNORECASE,
)
_NON_TARGET_ARCHITECTURE_BOUNDARY_RE = re.compile(
    r"\b(?:less\s+than|below|older\s+than|prior\s+to|before)\s*$",
    re.IGNORECASE,
)
_NON_TARGET_PRODUCT_BOUNDARY_RE = re.compile(
    r"\b(?:unless|except)\s+(?:when\s+)?(?:running\s+)?(?:in|on|with)\s*$",
    re.IGNORECASE,
)

# These mappings are intentionally narrow. Each product is named together with
# its compute capability by the cited NVIDIA page; no other product name is
# canonicalized by inference.
PRODUCT_ARCHITECTURE_MAPPINGS = {
    "b200": (
        "sm100",
        "https://developer.nvidia.com/cuda/gpus",
    ),
    "gb200": (
        "sm100",
        "https://developer.nvidia.com/cuda/gpus",
    ),
    "b300": (
        "sm103",
        "https://developer.nvidia.com/cuda/gpus",
    ),
    "gb300": (
        "sm103",
        "https://developer.nvidia.com/cuda/gpus",
    ),
    "h100": (
        "sm90",
        "https://developer.nvidia.com/cuda/gpus",
    ),
    "h200": (
        "sm90",
        "https://developer.nvidia.com/cuda/gpus",
    ),
    "gh200": (
        "sm90",
        "https://developer.nvidia.com/cuda/gpus",
    ),
    "h800": (
        "sm90",
        "https://docs.omniverse.nvidia.com/dang/latest/common/technical-requirements.html",
    ),
    "h20": (
        "sm90",
        "https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html",
    ),
    "a100": (
        "sm80",
        "https://developer.nvidia.com/cuda/gpus",
    ),
}

_PRODUCT_RE = re.compile(
    r"(" + "|".join(map(re.escape, sorted(PRODUCT_ARCHITECTURE_MAPPINGS, key=len, reverse=True))) + r")(?![a-z0-9])",
    re.IGNORECASE,
)

_HARD_SCOPE_RE = re.compile(
    r"(?<![a-z0-9])(eplb|deep[ _-]?ep|dual[ _-]?pipe)(?![a-z0-9])",
    re.IGNORECASE,
)
_DISTRIBUTED_SCOPE_RE = re.compile(
    r"(?<![a-z0-9])(?:all[ _-]?reduce|all[ _-]?to[ _-]?all|alltoall|"
    r"nvshmem|mnnvl|symmetric[ _-]?mem(?:ory)?|multi[ _-]?node|"
    r"multi[ _-]?gpu|communication[ _-]?kernels?|eplb|deep[ _-]?ep|dual[ _-]?pipe)"
    r"(?![a-z0-9])",
    re.IGNORECASE,
)

_NON_IMPLEMENTATION_SEGMENTS = {
    "test", "tests", "testing", "benchmark", "benchmarks", "bench", "docs", "doc",
    ".github", "ci", "examples_tests",
}
_NON_GPU_IMPLEMENTATION_SEGMENTS = {"cpu"}
_STRONG_DEVICE_EXTENSIONS = {".cu", ".cuh", ".ptx"}
_DEVICE_SIGNAL_EXTENSIONS = {".cc", ".cpp", ".cxx", ".h", ".hpp", ".inl", ".jinja"}
_EXPLICIT_PYTHON_KERNEL_PATH_RE = re.compile(
    r"(?:^|/)(?:triton_kernels?|cute_dsl(?:_kernels?)?|cutedsl(?:_kernels?)?|"
    r"CuTeDSL|tilelang_kernels?)(?:/|$)",
    re.IGNORECASE,
)

_PYTHON_DSL_PATTERNS = {
    "triton": re.compile(
        r"@(?:triton\.)?(?:jit|autotune|heuristics)\b|@triton\.jit\b|"
        r"\btriton\.language\b|\btl\.[a-z_]\w*\s*\(",
        re.IGNORECASE,
    ),
    "cute-dsl": re.compile(
        r"@(?:cutlass\.)?cute\.(?:jit|kernel)\b|@cute\.(?:jit|kernel)\b|"
        r"\bcute\.[a-z_]\w*\s*\(",
        re.IGNORECASE,
    ),
    "tilelang": re.compile(
        r"@T\.prim_func\b|@tilelang\.(?:jit|autotune)\b|"
        r"\bT\.(?:Kernel|alloc_shared|gemm|copy|clear|Parallel|Pipelined)\s*\(",
        re.IGNORECASE,
    ),
    "jax-pallas": re.compile(r"@(?:pl|pallas)\.kernel\b|\bpallas_call\s*\(", re.IGNORECASE),
    "cutile": re.compile(r"@cutile\.kernel\b|\bcutile\.compile\s*\(", re.IGNORECASE),
}
_WEAK_KERNEL_TUNING_RE = re.compile(
    r"\b(?:block_[mnk]|num_warps)\b",
    re.IGNORECASE,
)
_DEVICE_CODE_RE = re.compile(
    r"\b__global__\b|\b__device__\b|\bCUTLASS_DEVICE\b|\bCUTE_HOST_DEVICE\b|"
    r"\b__CUDA_ARCH__\b|\btcgen05\b|\bwgmma\b|\bcute::[A-Za-z_]\w*|"
    r"\bcutlass::arch::[A-Za-z_]\w*|\bmake_tiled_copy\s*\(|"
    r"\b(?:GemmType|TiledMma|TileShape|CollectiveBuilder)\b",
    re.IGNORECASE,
)

# CUDA translation units also contain host-only bindings, dispatchers, launchers,
# and shape plumbing.  The broad signal above is useful for headers and DSL
# snippets, but tile/warp variables, host template-type names, and host-side
# kernel-launch configuration cannot prove that a .cu file implements device code.
# Complete-file receipts and the .cu intake lane therefore use this stricter
# construct set.
_CUDA_TRANSLATION_UNIT_DEVICE_RE = re.compile(
    r"\b__global__\b|\b__device__\b|\b__shared__\b|\b__launch_bounds__\b|"
    r"(?<!\.)\b(?:threadIdx|blockIdx|blockDim|gridDim)\s*\.\s*[xyz]\b|"
    r"\basm\s+volatile\b|"
    r"\bCUTLASS_DEVICE\b|\bCUTE_HOST_DEVICE\b|\b__CUDA_ARCH__\b|"
    r"\btcgen05\b|\bwgmma\b|\bcute::[A-Za-z_]\w*|"
    r"\bcutlass::arch::[A-Za-z_]\w*|\bmake_tiled_copy\s*\(",
    re.IGNORECASE,
)

_C_CPP_NON_CODE_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
    re.DOTALL,
)


def _c_cpp_code_only(text: str) -> str:
    """Remove comments and literals before scanning C/C++ source constructs."""
    return _C_CPP_NON_CODE_RE.sub(" ", text or "")


def _python_code_only(text: str) -> str:
    """Remove Python comments and strings without disturbing code positions."""
    source = text or ""
    offsets = []
    cursor = 0
    for line in source.splitlines(keepends=True):
        offsets.append(cursor)
        cursor += len(line)
    offsets.append(cursor)
    chars = list(source)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type not in {tokenize.COMMENT, tokenize.STRING}:
                continue
            start_row, start_col = token.start
            end_row, end_col = token.end
            start = offsets[start_row - 1] + start_col
            end = offsets[end_row - 1] + end_col
            for index in range(start, min(end, len(chars))):
                if chars[index] not in "\r\n":
                    chars[index] = " "
    except (IndentationError, SyntaxError, tokenize.TokenError):
        # A partial GitHub hunk may not tokenize as a module. Strip full-line
        # comments conservatively; a malformed hunk never gains evidence from
        # a comment-only token.
        return "\n".join(
            "" if line.lstrip().startswith("#") else line
            for line in source.splitlines()
        )
    return "".join(chars)


def cuda_translation_unit_device_signal(text: str) -> bool:
    """Whether complete/added .cu text contains a strong device construct."""
    return bool(_CUDA_TRANSLATION_UNIT_DEVICE_RE.search(_c_cpp_code_only(text)))


def device_code_pattern_sha256() -> str:
    """Digest the strict .cu signal pattern used for complete-file receipts."""
    raw = (
        "cuda-source-code-only-v2\0"
        f"{_CUDA_TRANSLATION_UNIT_DEVICE_RE.pattern}\0"
        f"{_CUDA_TRANSLATION_UNIT_DEVICE_RE.flags}"
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def python_dsl_languages(text: str) -> tuple[str, ...]:
    """DSLs with executable constructs in Python source, excluding comments/strings."""
    code = _python_code_only(text)
    return tuple(sorted(
        language for language, pattern in _PYTHON_DSL_PATTERNS.items()
        if pattern.search(code)
    ))


def python_dsl_pattern_sha256() -> str:
    """Digest the Python DSL patterns and their code-only preprocessing contract."""
    raw = ["python-token-code-only-v1"]
    for language, pattern in sorted(_PYTHON_DSL_PATTERNS.items()):
        raw.extend((language, pattern.pattern, str(pattern.flags)))
    return hashlib.sha256("\0".join(raw).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScopeDecision:
    retain: bool
    disposition: str
    rule: str
    evidence_paths: tuple[str, ...]
    reason: str


def normalize_files(files: Iterable[Any]) -> list[dict[str, Any]]:
    """Normalize either GitHub file mappings or the legacy list of path strings."""
    normalized = []
    for item in files or []:
        if isinstance(item, str):
            normalized.append({"filename": item, "patch": ""})
        elif isinstance(item, dict):
            path = item.get("filename") or item.get("path")
            if path:
                normalized.append({**item, "filename": str(path), "patch": item.get("patch") or ""})
    return normalized


def parse_git_diff_files(diff_text: str) -> list[dict[str, Any]]:
    """Parse a complete GitHub pull ``.diff`` into policy-compatible files.

    GitHub's JSON files endpoint stops at 3,000 rows.  The pull ``.diff`` is a
    separate evidence channel which can expose the remaining paths and hunks.
    This intentionally parses only fields consumed by this repository's pure
    classifiers and hashes; it is not a general-purpose Git patch parser.
    """
    starts = list(re.finditer(r"(?m)^diff --git (.+)$", diff_text or ""))
    files: list[dict[str, Any]] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(diff_text)
        section = diff_text[match.start():end].rstrip("\n")
        try:
            header = shlex.split(match.group(0))
        except ValueError as exc:
            raise ValueError(f"invalid diff header: {match.group(0)!r}") from exc
        if len(header) != 4 or header[:2] != ["diff", "--git"]:
            raise ValueError(f"invalid diff header: {match.group(0)!r}")
        old_path, new_path = header[2], header[3]
        path = new_path[2:] if new_path.startswith("b/") else new_path
        status = "modified"
        if re.search(r"(?m)^new file mode ", section):
            status = "added"
        elif re.search(r"(?m)^deleted file mode ", section):
            status = "removed"
            path = old_path[2:] if old_path.startswith("a/") else old_path
        elif re.search(r"(?m)^rename (?:from|to) ", section):
            status = "renamed"
        additions = sum(
            line.startswith("+") and not line.startswith("+++")
            for line in section.splitlines()
        )
        deletions = sum(
            line.startswith("-") and not line.startswith("---")
            for line in section.splitlines()
        )
        files.append({
            "filename": path,
            "status": status,
            "additions": additions,
            "deletions": deletions,
            "patch": section,
        })
    return files


def added_patch_text(patch: str) -> str:
    """Return only added hunk lines, excluding the ``+++`` file marker."""
    return "\n".join(
        line[1:]
        for line in (patch or "").splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def changed_implementation_text(patch: str) -> str:
    """Changed hunk text with blank/comment/license-only lines removed."""
    result = []
    in_block_comment = False
    for line in (patch or "").splitlines():
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        value = line[1:].strip()
        if not value:
            continue
        if in_block_comment:
            if "*/" in value:
                in_block_comment = False
            continue
        if value.startswith("/*"):
            if "*/" not in value[2:]:
                in_block_comment = True
            continue
        if value.startswith(("//", "*", "# SPDX-", "SPDX-")):
            continue
        result.append(value)
    return "\n".join(result)


def cuda_requires_complete_evidence(item: dict[str, Any]) -> bool:
    """Whether a ``.cu`` patch is inconclusive by itself.

    A changed host registration hunk does not prove device implementation code,
    and GitHub may omit or truncate patches even for added files.  Any .cu path
    without a strong device construct in its available added text therefore
    needs an immutable PR-head complete-file receipt before it can be positive.
    """
    path = str(item.get("filename") or item.get("path") or "")
    if PurePosixPath(path).suffix.lower() != ".cu":
        return False
    status = str(item.get("status") or item.get("changeType") or "").lower()
    if status in {"removed", "deleted"}:
        return False
    patch = item.get("patch") or ""
    if patch and not changed_implementation_text(patch):
        return False
    return not cuda_translation_unit_device_signal(added_patch_text(patch))


def python_requires_complete_evidence(item: dict[str, Any]) -> bool:
    """Whether weak Python tuning text needs immutable full-file DSL proof."""
    path = str(item.get("filename") or item.get("path") or "")
    if PurePosixPath(path).suffix.lower() != ".py":
        return False
    status = str(item.get("status") or item.get("changeType") or "").lower()
    if status in {"removed", "deleted"}:
        return False
    if _path_is_non_implementation(path) or _path_is_distributed_implementation(path):
        return False
    if (
        _EXPLICIT_PYTHON_KERNEL_PATH_RE.search(path)
        and PurePosixPath(path).name.lower() != "__init__.py"
    ):
        return False
    added = added_patch_text(item.get("patch") or "")
    if python_dsl_languages(added):
        return False
    return bool(_WEAK_KERNEL_TUNING_RE.search(_python_code_only(added)))


def complete_file_device_signal(item: dict[str, Any]) -> bool:
    """Validate and return a complete PR-head file's device-code verdict."""
    return bool(
        item.get("complete_file_evidence_complete") is True
        and re.fullmatch(r"[0-9a-f]{64}", str(item.get("complete_file_sha256", "")))
        and item.get("complete_file_device_pattern_sha256")
        == device_code_pattern_sha256()
        and item.get("complete_file_device_signal") is True
    )


def complete_file_python_dsl_signal(item: dict[str, Any]) -> bool:
    """Validate a complete PR-head Python file's DSL-kernel verdict."""
    languages = item.get("complete_file_python_dsl_languages")
    return bool(
        item.get("complete_file_python_dsl_evidence_complete") is True
        and re.fullmatch(
            r"[0-9a-f]{64}", str(item.get("complete_file_python_dsl_sha256", ""))
        )
        and item.get("complete_file_python_dsl_pattern_sha256")
        == python_dsl_pattern_sha256()
        and item.get("complete_file_python_dsl_signal") is True
        and isinstance(languages, list)
        and languages
        and languages == sorted(set(languages))
        and set(languages) <= set(_PYTHON_DSL_PATTERNS)
    )


def _path_is_non_implementation(path: str) -> bool:
    parts = {part.lower() for part in PurePosixPath(path).parts}
    name = PurePosixPath(path).name.lower()
    return (
        bool(parts & (_NON_IMPLEMENTATION_SEGMENTS | _NON_GPU_IMPLEMENTATION_SEGMENTS))
        or name.startswith("test_")
        or "_test." in name
    )


def _path_is_distributed_implementation(path: str) -> bool:
    """Recognize communication-kernel paths, including joined/CamelCase names.

    A bare ``collective`` is deliberately absent: CUTLASS uses that term for
    single-device GEMM collectives.  The tokens below are specific to
    cross-device communication or the named hard exclusions.
    """
    compact = re.sub(r"[^a-z0-9]+", "", path.lower())
    return any(token in compact for token in (
        "allreduce",
        "alltoall",
        "nvshmem",
        "mnnvl",
        "symmetricmemory",
        "multinode",
        "multigpu",
        "communicationkernel",
        "commops",
        "eplb",
        "deepep",
        "dualpipe",
    ))


def classify_scope(title: str, body: str, files: Iterable[Any]) -> ScopeDecision:
    """Apply the single-device kernel-only intake rule to authoritative PR evidence."""
    normalized = normalize_files(files)
    # Named exclusions are hard when they are the PR subject. Incidental body
    # mentions (for example a benchmark command selecting a DeepEP backend) do
    # not erase an independently qualifying single-device kernel change.
    hard_match = _HARD_SCOPE_RE.search(title or "")
    distributed_title_match = _DISTRIBUTED_SCOPE_RE.search(title or "")
    if hard_match or distributed_title_match:
        match = hard_match or distributed_title_match
        return ScopeDecision(
            False,
            "removed",
            "hard-distributed-system-exclusion" if hard_match else "distributed-system-implementation-exclusion",
            (),
            f"remove: distributed-system topic matched {match.group(0)!r} in PR title",
        )

    strong_paths: list[str] = []
    python_paths: list[str] = []
    signaled_paths: list[str] = []
    excluded_paths = []
    for item in normalized:
        path = item["filename"]
        if _path_is_non_implementation(path):
            continue
        if _path_is_distributed_implementation(path):
            excluded_paths.append(path)
            continue
        suffix = PurePosixPath(path).suffix.lower()
        added = added_patch_text(item.get("patch") or "")
        if suffix in _STRONG_DEVICE_EXTENSIONS:
            status = str(item.get("status") or item.get("changeType") or "").lower()
            patch = item.get("patch") or ""
            if status in {"removed", "deleted"}:
                continue
            if patch and not changed_implementation_text(patch):
                continue
            # A .cu extension alone is never device-kernel evidence: bindings,
            # dispatchers, and UVA helpers are also CUDA translation units. Any
            # hunk without an independent device construct must be backed by an
            # immutable PR-head complete-file receipt because GitHub may omit or
            # truncate patches. Missing content remains unknown evidence rather
            # than becoming a positive.
            if suffix == ".cu":
                if cuda_requires_complete_evidence(item):
                    if not complete_file_device_signal(item):
                        continue
            strong_paths.append(path)
            continue
        if suffix == ".py":
            explicit_kernel_path = (
                _EXPLICIT_PYTHON_KERNEL_PATH_RE.search(path)
                and PurePosixPath(path).name.lower() != "__init__.py"
            )
            if (
                explicit_kernel_path
                or python_dsl_languages(added)
                or (
                    python_requires_complete_evidence(item)
                    and complete_file_python_dsl_signal(item)
                )
            ):
                python_paths.append(path)
                continue
        if suffix in _DEVICE_SIGNAL_EXTENSIONS and _DEVICE_CODE_RE.search(added):
            signaled_paths.append(path)

    if strong_paths:
        paths = tuple(sorted(set(strong_paths)))
        return ScopeDecision(
            True,
            "retained",
            "cuda-cute-ptx-device-source",
            paths,
            "retain: CUDA/CuTe/PTX device implementation path(s): " + ", ".join(paths),
        )
    if python_paths:
        paths = tuple(sorted(set(python_paths)))
        return ScopeDecision(
            True,
            "retained",
            "python-dsl-device-kernel",
            paths,
            "retain: Python DSL device-kernel implementation added in: " + ", ".join(paths),
        )
    if signaled_paths:
        paths = tuple(sorted(set(signaled_paths)))
        return ScopeDecision(
            True,
            "retained",
            "device-code-signal",
            paths,
            "retain: added device-code construct in: " + ", ".join(paths),
        )

    inspected = len(normalized)
    if excluded_paths:
        return ScopeDecision(
            False,
            "removed",
            "distributed-system-implementation-exclusion",
            tuple(sorted(set(excluded_paths))),
            "remove: only distributed-system implementation path(s) qualified: "
            + ", ".join(sorted(set(excluded_paths))),
        )
    return ScopeDecision(
        False,
        "removed",
        "no-positive-device-kernel-evidence",
        (),
        f"remove: no positive device-kernel implementation in {inspected} authoritative changed file(s)",
    )


def _strip_architecture_boilerplate(body: str) -> str:
    """Remove hidden PR templates and generated/bot help from architecture evidence."""
    text = body or ""
    text = re.sub(
        r"<!-- This is an auto-generated comment:.*?<!-- end of auto-generated comment:.*?-->",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.split(r"^## GitHub Bot Help\s*$", text, maxsplit=1, flags=re.MULTILINE)[0]
    return text


def _match_is_non_target_boundary(text: str, match: re.Match[str]) -> bool:
    """Reject an SM token used only as an open-ended range boundary.

    Unsupported/failing architectures are deliberately *not* rejected: a PR may
    exist specifically to repair support for one.  The narrow range check keeps
    phrases such as ``kernels less than sm80`` from mislabeling an SM75 repair as
    an SM80 target.
    """
    return bool(_NON_TARGET_ARCHITECTURE_BOUNDARY_RE.search(
        text[max(0, match.start() - 48):match.start()]
    ))


def _evidence_fragments(
    title: str,
    body: str,
    files: Iterable[Any],
    implementation_paths: Iterable[str] = (),
):
    yield "PR title", title or ""
    yield "PR description", _strip_architecture_boilerplate(body)
    implementation_paths = set(implementation_paths)
    for item in normalize_files(files):
        path = item["filename"]
        yield f"changed-path:{path}", path
        added = added_patch_text(item.get("patch") or "")
        # Raw implementation text is not target evidence: portable atom names,
        # compatibility headers, and referenced kernels routinely name older
        # architectures.  Retain only explicit build/dispatch/guard lines.
        guard_lines = [
            line for line in added.splitlines()
            if _ARCHITECTURE_GUARD_LINE_RE.search(line)
            and not _COMMENTED_OUT_CODE_RE.search(line)
        ]
        if guard_lines:
            yield f"architecture-guard:{path}", "\n".join(guard_lines)

        # Architecture tokens in the changed implementation itself are valid
        # target evidence when they are not compatibility includes/atoms or a
        # negative comparison.  Tests and other non-implementation files are
        # accepted only when the line explicitly states a target/support rule.
        signal_lines = []
        for line in added.splitlines():
            if line.lstrip().startswith("#include") or _COMMENTED_OUT_CODE_RE.search(line):
                continue
            has_signal = (
                _EXACT_SM_RE.search(line)
                or _PRODUCT_RE.search(line)
                or any(pattern.search(line) for pattern in _FAMILY_PATTERNS.values())
            )
            if not has_signal:
                continue
            if path in implementation_paths or _TARGET_ASSERTION_RE.search(line):
                signal_lines.append(line)
        if signal_lines:
            yield f"added-patch:{path}", "\n".join(signal_lines)


def _iter_exact_sm(text: str):
    for match in _EXACT_SM_RE.finditer(text):
        start, end = match.span()
        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""
        spelling = text[start:start + 2]
        # Ordinary word substrings are not targets; CamelCase `Sm100Kernel`
        # and filename/guard separators are accepted.
        camel_case_boundary = spelling.startswith("S") and (
            before.islower() or spelling == "SM"
        )
        if before.isalnum() and not camel_case_boundary:
            continue
        # Preserve CamelCase symbols such as ``FmhaSm100aKernel`` while
        # rejecting architecture-like invalid spellings such as ``SM100X``.
        if after.isalnum():
            if not after.isupper():
                continue
            if spelling == "SM" and not re.match(r"^[A-Z][a-z]", text[end:]):
                continue
        arch = "sm" + match.group(1) + match.group(2).lower()
        if arch not in _SUPPORTED_EXACT_ARCHITECTURE_SET:
            continue
        if _PORTABLE_ARCH_ATOM_SUFFIX_RE.match(text[end:]):
            continue
        yield match


def _iter_products(text: str):
    for match in _PRODUCT_RE.finditer(text):
        start = match.start()
        before = text[start - 1] if start else ""
        if before.isalpha():
            multiplier = before.lower() == "x" and start >= 2 and text[start - 2].isdigit()
            if not multiplier:
                continue
        elif before.isdigit():
            continue
        if _NON_TARGET_PRODUCT_BOUNDARY_RE.search(text[max(0, start - 64):start]):
            continue
        yield match


def _family_for_exact(architecture: str) -> str | None:
    for family, prefixes in ARCHITECTURE_FAMILY_PREFIXES.items():
        if architecture.startswith(prefixes):
            return family
    return None


def derive_architectures(title: str, body: str, files: Iterable[Any]) -> tuple[list[str], str, list[dict[str, str]]]:
    """Return canonical architectures, disposition, and exact evidence records."""
    exact_evidence: dict[str, list[dict[str, str]]] = {}
    family_evidence: dict[str, dict[str, str]] = {}

    normalized_files = normalize_files(files)
    scope = classify_scope(title, body, normalized_files)
    fragments = list(_evidence_fragments(
        title, body, normalized_files, scope.evidence_paths
    ))

    def record_fragment(locator: str, text: str, include_numeric_guards: bool = False):
        for match in _iter_exact_sm(text):
            if locator != "PR title" and _match_is_non_target_boundary(text, match):
                continue
            arch = "sm" + match.group(1) + match.group(2).lower()
            exact_evidence.setdefault(arch, []).append({
                "architecture": arch,
                "basis": "exact-sm-token",
                "locator": locator,
                "evidence": match.group(0),
            })
        for match in _iter_products(text):
            product = match.group(1).lower()
            arch, mapping_source = PRODUCT_ARCHITECTURE_MAPPINGS[product]
            exact_evidence.setdefault(arch, []).append({
                "architecture": arch,
                "basis": "documented-product-mapping",
                "locator": locator,
                "evidence": match.group(0),
                "mapping_source": mapping_source,
            })
        for family, pattern in _FAMILY_PATTERNS.items():
            match = pattern.search(text)
            if match and family not in family_evidence:
                family_evidence[family] = {
                    "architecture": family,
                    "basis": "explicit-family-name",
                    "locator": locator,
                    "evidence": match.group(0),
                }

        if include_numeric_guards:
            for pattern, scale, basis in (
                (_NUMERIC_CUDA_ARCH_RE, 10, "cuda-arch-guard"),
                (_NUMERIC_RUNTIME_ARCH_RE, 1, "runtime-architecture-guard"),
            ):
                for match in pattern.finditer(text):
                    value = int(match.group(1)) // scale
                    arch = f"sm{value}"
                    exact_evidence.setdefault(arch, []).append({
                        "architecture": arch,
                        "basis": basis,
                        "locator": locator,
                        "evidence": match.group(0),
                    })

    # Preserve the high-precision sources used by the original policy: title,
    # description, paths, and explicit textual build/dispatch/guard lines.
    # Broader implementation signals and numeric guard decoding are a fallback
    # only when those sources establish no family or exact target.
    baseline_fragments = [
        fragment for fragment in fragments
        if not fragment[0].startswith("added-patch:")
    ]
    fallback_fragments = [
        fragment for fragment in fragments
        if fragment[0].startswith(("architecture-guard:", "added-patch:"))
    ]
    for locator, text in baseline_fragments:
        record_fragment(locator, text)
    if not exact_evidence and not family_evidence:
        for locator, text in fallback_fragments:
            record_fragment(locator, text, include_numeric_guards=True)

    # SM75/Turing support PRs exposed the concrete contrastive failure that
    # motivated adding this architecture: adjacent SM80 compatibility blocks
    # were being reported as the target.  When the title names only SM75 as the
    # support target, that direct statement outranks unrelated hunk context.
    title_exact = {
        architecture
        for architecture, rows in exact_evidence.items()
        if any(row["locator"] == "PR title" for row in rows)
    }
    if title_exact == {"sm75"} and re.search(r"\b(?:support|turing)\b", title, re.I):
        exact_evidence = {"sm75": exact_evidence["sm75"]}
        family_evidence = {
            family: row for family, row in family_evidence.items()
            if family == "turing"
        }

    if exact_evidence:
        architectures = sorted(exact_evidence, key=lambda value: (int(re.match(r"sm(\d+)", value).group(1)), value))
        evidence = []
        for arch in architectures:
            seen = set()
            for row in exact_evidence[arch]:
                key = (row["basis"], row["locator"], row["evidence"])
                if key not in seen:
                    evidence.append(row)
                    seen.add(key)
                if len(seen) >= 3:
                    break
        exact_families = {_family_for_exact(value) for value in architectures}
        additional_families = sorted(set(family_evidence) - exact_families)
        for family in additional_families:
            evidence.append(family_evidence[family])
        return architectures + additional_families, "mixed" if additional_families else "exact", evidence

    if family_evidence:
        families = sorted(family_evidence)
        return families, "family", [family_evidence[family] for family in families]

    return [], "unknown", [{
        "architecture": "unknown",
        "basis": "no-discriminating-architecture-evidence",
        "locator": f"PR title, description, paths, and target-signaling additions in {len(normalized_files)} changed files",
        "evidence": "No qualifying target-architecture evidence survived the deterministic context filters.",
    }]


_HARDWARE_PATTERNS = {
    "tcgen05": re.compile(r"(?<![a-z0-9])tcgen05(?![a-z0-9])", re.I),
    "tmem": re.compile(r"\b(?:tmem|tensor memory)\b", re.I),
    "tma": re.compile(r"\b(?:tma|tensor memory accelerator)\b", re.I),
    "clc": re.compile(r"\b(?:clc|cluster launch control)\b", re.I),
    "nvfp4": re.compile(r"\bnvfp4\b", re.I),
    "fp8": re.compile(r"\bfp8\b", re.I),
    "fp6": re.compile(r"\bfp6\b", re.I),
    "fp4": re.compile(r"\b(?:fp4|mxfp4|nvfp4)\b", re.I),
    "block-scale": re.compile(r"\b(?:block[ _-]?scale|microscal)\w*", re.I),
    "wgmma": re.compile(r"\bwgmma\b", re.I),
    "mbarrier": re.compile(r"\bmbarrier\b", re.I),
}
_KERNEL_TYPE_PATTERNS = {
    "grouped-gemm": re.compile(r"\bgrouped[ _-]?gemm\b", re.I),
    "gemm": re.compile(r"\bgemm\b", re.I),
    "gemv": re.compile(r"\bgemv\b", re.I),
    "flash-attention": re.compile(r"\b(?:flash[ _-]?attention|fmha)\b", re.I),
    "sparse-attention": re.compile(r"\bsparse[ _-]?attention\b", re.I),
    "attention": re.compile(r"\battention\b|\bfmha\b", re.I),
    "mla": re.compile(r"\bmla\b|multi-head latent attention", re.I),
    "moe": re.compile(r"\bmoe\b|mixture of experts", re.I),
    "quantization": re.compile(r"\bquanti[sz]\w*", re.I),
    "topk": re.compile(r"\btop[ -]?k\b", re.I),
    "scan": re.compile(
        r"\b(?:BlockScan|WarpScan|exclusive[ _-]?scan|inclusive[ _-]?scan|"
        r"prefix[ _-]?scan|parallel[ _-]?scan|scan[ _-]?(?:algorithm|kernel))\b",
        re.I,
    ),
    "reduction": re.compile(r"\breduct(?:ion|e)\b", re.I),
    "sort": re.compile(r"\b(?:sort|radix)\b", re.I),
}
_TECHNIQUE_PATTERNS = {
    "warp-specialization": re.compile(r"\bwarp[ _-]?speciali[sz]\w*", re.I),
    "persistent-kernel": re.compile(r"\bpersistent[ _-]?(?:kernel|schedul)\w*", re.I),
    "swizzling": re.compile(r"\bswizzl\w*", re.I),
    "pipeline-stages": re.compile(r"\bpipeline(?:d|s| stages?)?\b", re.I),
    "double-buffering": re.compile(r"\bdouble[ _-]?buffer\w*", re.I),
    "epilogue-fusion": re.compile(r"\bepilogue[ _-]?fusion\b", re.I),
    "tile-scheduling": re.compile(r"\btile[ _-]?schedul\w*", re.I),
    "kernel-fusion": re.compile(r"\b(?:kernel[ _-]?fusion|fused[ _-]?kernel|fuse kernels?)\b", re.I),
    "stream-k": re.compile(r"\bstream[ -]?k\b", re.I),
}


def derive_metadata(title: str, body: str, files: Iterable[Any], scope: ScopeDecision | None = None) -> dict[str, list[str]]:
    """Derive only metadata supported by positive kernel evidence."""
    normalized = normalize_files(files)
    scope = scope or classify_scope(title, body, normalized)
    if not scope.retain:
        return {"tags": [], "hardware_features": [], "kernel_types": [], "techniques": [], "languages": []}

    evidence_paths = set(scope.evidence_paths)
    semantic_parts = [title or "", body or ""]
    languages = set()
    for item in normalized:
        path = item["filename"]
        if path not in evidence_paths:
            continue
        added = added_patch_text(item.get("patch") or "")
        semantic_parts.append(added)
        semantic_parts.append(path)
        suffix = PurePosixPath(path).suffix.lower()
        if suffix in {".cu", ".cuh", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".inl", ".jinja"}:
            languages.add("cuda-cpp")
        if suffix == ".ptx":
            languages.add("ptx")
        if suffix == ".py":
            languages.update(python_dsl_languages(added))
            if complete_file_python_dsl_signal(item):
                languages.update(item["complete_file_python_dsl_languages"])
            lowered_path = path.lower()
            if "triton_kernel" in lowered_path:
                languages.add("triton")
            if re.search(r"(?:^|/)(?:cute_dsl|cutedsl)(?:/|_kernel)", lowered_path):
                languages.add("cute-dsl")
            if "tilelang_kernel" in lowered_path:
                languages.add("tilelang")

    semantic_text = "\n".join(semantic_parts)
    # Negative compatibility inventories describe what a change does *not*
    # implement. They must not become positive kernel taxonomy.
    semantic_text = re.sub(
        r"\b(?:not compatible with|does not support|do not support|unsupported for)\b[^.\n]*",
        "",
        semantic_text,
        flags=re.IGNORECASE,
    )
    hardware = {tag for tag, pattern in _HARDWARE_PATTERNS.items() if pattern.search(semantic_text)}
    kernel_types = {tag for tag, pattern in _KERNEL_TYPE_PATTERNS.items() if pattern.search(semantic_text)}
    techniques = {tag for tag, pattern in _TECHNIQUE_PATTERNS.items() if pattern.search(semantic_text)}
    tags = hardware | kernel_types
    return {
        "tags": sorted(tags),
        "hardware_features": sorted(hardware),
        "kernel_types": sorted(kernel_types),
        "techniques": sorted(techniques),
        "languages": sorted(languages),
    }


def upstream_files_sha256(files: Iterable[Any]) -> str:
    """Hash authoritative file metadata and patches in a canonical representation."""
    canonical = []
    for item in normalize_files(files):
        record = {
            "filename": item["filename"],
            "status": item.get("status") or item.get("changeType") or "",
            "additions": item.get("additions"),
            "deletions": item.get("deletions"),
            "patch_sha256": hashlib.sha256((item.get("patch") or "").encode("utf-8")).hexdigest(),
        }
        if (
            cuda_requires_complete_evidence(item)
            and item.get("complete_file_evidence_complete") is True
        ):
            record.update({
                "complete_file_evidence_complete": True,
                "complete_file_sha256": item.get("complete_file_sha256"),
                "complete_file_device_signal": item.get("complete_file_device_signal"),
                "complete_file_device_pattern_sha256": item.get(
                    "complete_file_device_pattern_sha256"
                ),
            })
        if (
            python_requires_complete_evidence(item)
            and item.get("complete_file_python_dsl_evidence_complete") is True
        ):
            record.update({
                "complete_file_python_dsl_evidence_complete": True,
                "complete_file_python_dsl_sha256": item.get(
                    "complete_file_python_dsl_sha256"
                ),
                "complete_file_python_dsl_signal": item.get(
                    "complete_file_python_dsl_signal"
                ),
                "complete_file_python_dsl_languages": item.get(
                    "complete_file_python_dsl_languages"
                ),
                "complete_file_python_dsl_pattern_sha256": item.get(
                    "complete_file_python_dsl_pattern_sha256"
                ),
            })
        canonical.append(record)
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def upstream_excerpt(body: str) -> str:
    # Python's text readers normalize line endings, so the generated digest must
    # use the same canonical representation. GitHub PR bodies commonly contain
    # CRLF even on Linux; hashing those raw newlines made a freshly generated
    # page fail its own offline validator.
    canonical_body = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    canonical_lines = []
    for line in canonical_body.split("\n"):
        line = line.rstrip(" \t")
        # Git treats these exact line prefixes as unresolved conflict markers,
        # even inside a Markdown code fence. Preserve the upstream text while
        # making the generated document unambiguous to `git diff --check`.
        if re.match(r"^(?:<{7}|={7}|>{7})(?: |$)", line):
            line = "\\" + line
        canonical_lines.append(line)
    canonical_body = "\n".join(canonical_lines)
    # Trim again after truncation so a cut that lands on whitespace remains
    # idempotent when the validator re-renders the already-rendered excerpt.
    excerpt = canonical_body.strip()[:UPSTREAM_EXCERPT_LIMIT].rstrip()
    # Prevent an upstream description from forging our local contract sentinels.
    return excerpt.replace("<!-- upstream-excerpt-start -->", "&lt;!-- upstream-excerpt-start --&gt;").replace(
        "<!-- upstream-excerpt-end -->", "&lt;!-- upstream-excerpt-end --&gt;"
    )


def render_generated_body(body: str, changed_paths: Iterable[str], total_files: int | None = None) -> tuple[str, str]:
    """Render the only allowed generated source-PR body template."""
    excerpt = upstream_excerpt(body)
    excerpt_hash = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
    paths = list(changed_paths)
    total = len(paths) if total_files is None else total_files
    lines = [
        "## Upstream PR description (verbatim excerpt)",
        "",
        "<!-- upstream-excerpt-start -->",
        excerpt if excerpt else "_The upstream PR description is empty._",
        "<!-- upstream-excerpt-end -->",
        "",
        "## Changed files (upstream)",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    if total > len(paths):
        lines.append(f"- _{total - len(paths)} additional changed file(s) omitted from this display._")
    lines.append("")
    return "\n".join(lines), excerpt_hash


def body_contract_errors(frontmatter: dict[str, Any], body: str) -> list[str]:
    """Validate the offline generated-body shape and its excerpt digest."""
    errors = []
    if frontmatter.get("body_contract") != BODY_CONTRACT:
        return [f"body_contract must be {BODY_CONTRACT!r}"]
    start = "<!-- upstream-excerpt-start -->\n"
    end = "\n<!-- upstream-excerpt-end -->"
    if body.count(start) != 1 or body.count(end) != 1:
        errors.append("generated body must contain exactly one upstream excerpt sentinel pair")
        return errors
    prefix, rest = body.split(start, 1)
    excerpt_rendered, suffix = rest.split(end, 1)
    if prefix != "## Upstream PR description (verbatim excerpt)\n\n":
        errors.append("generated body has text outside the upstream-pr-v1 prefix template")
    if not suffix.startswith("\n\n## Changed files (upstream)\n\n"):
        errors.append("generated body has text outside the upstream-pr-v1 suffix template")
    empty_template = "_The upstream PR description is empty._"
    excerpt = "" if excerpt_rendered == empty_template else excerpt_rendered
    actual = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
    if frontmatter.get("upstream_excerpt_sha256") != actual:
        errors.append("upstream_excerpt_sha256 does not match rendered upstream excerpt")
    if not re.fullmatch(r"[0-9a-f]{64}", str(frontmatter.get("upstream_body_sha256", ""))):
        errors.append("upstream_body_sha256 must be a full SHA-256 digest")
    if not re.fullmatch(r"[0-9a-f]{64}", str(frontmatter.get("upstream_files_sha256", ""))):
        errors.append("upstream_files_sha256 must be a full SHA-256 digest")
    expected_body, _ = render_generated_body(
        excerpt,
        frontmatter.get("changed_paths") or [],
        frontmatter.get("changed_files_count"),
    )
    if body != expected_body:
        errors.append("generated body differs from the deterministic upstream-pr-v1 template")
    return errors


def architecture_matches_filter(architectures: Iterable[str], disposition: str, requested: str) -> bool:
    """Canonical user-facing architecture hierarchy semantics."""
    archs = {str(value).lower() for value in architectures or []}
    requested = requested.lower().replace("_", "")
    if requested in PRODUCT_ARCHITECTURE_MAPPINGS:
        requested = PRODUCT_ARCHITECTURE_MAPPINGS[requested][0]
    if requested == "unknown":
        return disposition == "unknown" and not archs
    if requested in ARCHITECTURE_FAMILY_PREFIXES:
        return requested in archs or any(
            value.startswith(ARCHITECTURE_FAMILY_PREFIXES[requested]) for value in archs
        )
    return requested in archs

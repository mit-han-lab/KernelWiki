#!/usr/bin/env bash
# Verify that the skill's read paths work with a Python interpreter that has
# no site-packages. `-S` disables the system site directory, including PyYAML.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

python3 -S - <<'PY'
import re
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from _yaml_compat import USING_BUNDLED, yaml

assert USING_BUNDLED, "fallback test unexpectedly imported host PyYAML"

frontmatter = Path("wiki/kernels/flash-attention-4.md").read_text(encoding="utf-8")
match = re.match(r"^---\s*\n(.*?)\n---", frontmatter, re.S)
data = yaml.safe_load(match.group(1))
assert data["id"] == "kernel-flash-attention-4"
assert "tcgen05" in data["tags"]

policy = yaml.safe_load(Path("data/inclusion-policy.yaml").read_text(encoding="utf-8"))
assert isinstance(policy["cute-dsl"]["capture_criteria"], list)
assert isinstance(policy["cute-dsl"]["description"], str)

round_trip = yaml.safe_load(yaml.safe_dump({"date": data["id"], "tags": data["tags"]}))
assert round_trip["tags"] == data["tags"]
PY

query_output="$(python3 -S scripts/query.py --tag nvfp4 --type kernel --compact)"
grep -q "kernel-nvfp4-gemm" <<<"$query_output"

page_output="$(python3 -S scripts/get_page.py kernel-flash-attention-4 --frontmatter-only)"
grep -q "id: kernel-flash-attention-4" <<<"$page_output"

echo "OK: KernelWiki works without host PyYAML"

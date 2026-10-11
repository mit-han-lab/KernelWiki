"""Local, replayable receipts for query strategies and page output."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid

from _yaml_compat import USING_BUNDLED, yaml


def digest(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def repository_state(root):
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
                text=True, timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"root": str(root.resolve()), "commit": git("rev-parse", "HEAD"),
            "status": git("status", "--porcelain")}


class RunLog:
    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def event(self, kind, **fields):
        # A separate immutable directory holds each operation's larger payloads.
        # Keep the index entry small and append it in one write.
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, **fields}, ensure_ascii=False) + "\n")

    def start_query(self, root, context, arguments, strategy_source=None,
                    strategy_path=None):
        query_id = uuid.uuid4().hex
        directory = self.directory / "queries" / query_id
        directory.mkdir(parents=True)
        engine = directory / "engine"
        engine.mkdir()
        # Scripts and corpus can come from different checkouts via the root override.
        script_directory = Path(__file__).resolve().parent
        hashes = {}
        for name in ("query.py", "get_page.py", "_retrieval_log.py", "_wiki_root.py",
                     "_yaml_compat.py", "pr_policy.py"):
            data = (script_directory / name).read_bytes()
            (engine / name).write_bytes(data)
            hashes[name] = digest(data)
        if strategy_source is not None:
            (directory / "strategy.py").write_bytes(strategy_source)
        alias_path = root / "data" / "aliases.yaml"
        alias_data = alias_path.read_bytes() if alias_path.is_file() else None
        if alias_data is not None:
            (directory / "aliases.yaml").write_bytes(alias_data)
        write_json(directory / "context.json", context)
        write_json(directory / "request.json", {
            "query_id": query_id, "repository": repository_state(script_directory.parent),
            "corpus_repository": repository_state(root),
            "aliases_sha256": digest(alias_data) if alias_data is not None else None,
            "runtime": {"python": sys.version, "yaml_version": getattr(yaml, "__version__", None),
                        "bundled_yaml": USING_BUNDLED},
            "arguments": arguments, "engine_sha256": hashes,
            "strategy_path": str(strategy_path) if strategy_path else None,
            "strategy_sha256": digest(strategy_source) if strategy_source is not None else None,
        })
        return query_id, directory

    def finish_query(self, query_id, result, output, diagnostics=""):
        directory = self.query_directory(query_id)
        write_json(directory / "result.json", result)
        (directory / "output.txt").write_text(output, encoding="utf-8")
        if diagnostics:
            (directory / "strategy-output.txt").write_text(diagnostics, encoding="utf-8")
        self.event("query", query_id=query_id, status=result.get("status", "ok"))

    def query_directory(self, query_id):
        if not re.fullmatch(r"[0-9a-f]{32}", query_id):
            raise ValueError("invalid query ID")
        directory = self.directory / "queries" / query_id
        if not (directory / "request.json").is_file():
            raise ValueError(f"query ID not found in this run: {query_id}")
        return directory

    def record_read(self, root, arguments, entries, output, query_id=None):
        if query_id:
            self.query_directory(query_id)
        read_id = uuid.uuid4().hex
        directory = self.directory / "reads" / read_id
        directory.mkdir(parents=True)
        (directory / "output.txt").write_text(output, encoding="utf-8")
        write_json(directory / "receipt.json", {
            "read_id": read_id, "query_id": query_id,
            "repository": repository_state(Path(__file__).resolve().parent.parent),
            "corpus_repository": repository_state(root), "arguments": arguments,
            "files": entries, "output_chars": len(output),
            "output_sha256": digest(output),
            "meaning": "Content returned by get_page.py; not proof of model attention.",
        })
        self.event("page_read", read_id=read_id, query_id=query_id)


def read_entry(root, path, content, view, shown):
    """Distinguish full reads, metadata, excerpts, and truncated code output."""
    path = Path(path).resolve()
    try:
        name = str(path.relative_to(root.resolve()))
    except ValueError:
        name = str(path)
    return {"path": name, "file_sha256": digest(content), "view": view,
            "shown_chars": len(shown), "shown_sha256": digest(shown)}
